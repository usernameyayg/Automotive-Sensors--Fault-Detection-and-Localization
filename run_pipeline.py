#!/usr/bin/env python3
"""
Standalone pipeline script: Runs Parts 1-3 of the V13 notebook
with paths adjusted for the repository root directory.
Outputs all results to JSON for updating Chapter 5 tables.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import zipfile
import os
import time
import sys

from scipy import interpolate, stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             accuracy_score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONFIGURATION
# ============================================================================
REPO_ROOT = Path("/home/user/Automotive-Sensors--Fault-Detection-and-Localization")
os.chdir(REPO_ROOT)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 200
STRIDE = 100
BATCH_SIZE = 128
EPOCHS = 50
INPUT_CHANNELS = 2
EMBEDDING_DIM = 256
PROJECTION_DIM = 128
LEARNING_RATE = 1e-3
TEMPERATURE = 0.5
JITTER_SIGMA = 0.1
SCALE_RANGE = (0.8, 1.2)
MASK_RATIO = 0.1
HIL_HEALTHY_DURATION_SECONDS = 90
HIL_SAMPLING_RATE = 100
THRESHOLD_PERCENTILES = [15, 20, 25, 30, 35, 40]

# Results collector
results = {}

print(f"Device: {DEVICE}")
print(f"Working directory: {os.getcwd()}")
print()

# ============================================================================
# PART 1: A2D2 DATA LOADING
# ============================================================================
part1_start = time.time()
print("=" * 80)
print("PART 1: A2D2 DATA LOADING")
print("=" * 80)

# Extract ZIP files if needed
for zf in REPO_ROOT.glob("*bus_signals.zip"):
    extract_dir = REPO_ROOT / zf.stem
    if not extract_dir.exists():
        print(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(REPO_ROOT)

# Find JSON files
all_json_files = []
for f in sorted(REPO_ROOT.rglob("*bus_signals.json")):
    if f.stat().st_size > 1024 * 1024:  # > 1MB
        all_json_files.append(f)
all_json_files = sorted(list(set(all_json_files)))[:3]

print(f"Found {len(all_json_files)} A2D2 JSON files:")
for i, f in enumerate(all_json_files):
    print(f"  {i+1}. {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

def load_json_signals(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = {'filename': json_path.name, 'sensors_found': []}

    acc_ts = acc_vals = acc_rate = acc_unit = None
    spd_ts = spd_vals = spd_rate = spd_unit = None

    if 'accelerator_pedal' in data and isinstance(data['accelerator_pedal'], dict):
        if 'values' in data['accelerator_pedal']:
            metadata['sensors_found'].append('accelerator_pedal')
            vlist = data['accelerator_pedal']['values']
            acc_ts = np.array([v[0] for v in vlist])
            acc_vals = np.array([v[1] for v in vlist])
            if len(acc_ts) > 1:
                acc_rate = 1e6 / np.mean(np.diff(acc_ts))
            acc_unit = data['accelerator_pedal'].get('unit', '%')

    if 'vehicle_speed' in data and isinstance(data['vehicle_speed'], dict):
        if 'values' in data['vehicle_speed']:
            metadata['sensors_found'].append('vehicle_speed')
            vlist = data['vehicle_speed']['values']
            spd_ts = np.array([v[0] for v in vlist])
            spd_vals = np.array([v[1] for v in vlist])
            if len(spd_ts) > 1:
                spd_rate = 1e6 / np.mean(np.diff(spd_ts))
            spd_unit = data['vehicle_speed'].get('unit', 'km/h')

    if acc_vals is None or spd_vals is None:
        return None, None, "Missing sensors"

    metadata['accelerator'] = {
        'samples_original': len(acc_vals), 'sampling_rate_hz': acc_rate,
        'unit': acc_unit, 'min': float(acc_vals.min()), 'max': float(acc_vals.max()),
        'mean': float(acc_vals.mean())
    }
    metadata['speed'] = {
        'samples_original': len(spd_vals), 'sampling_rate_hz': spd_rate,
        'unit': spd_unit, 'min': float(spd_vals.min()), 'max': float(spd_vals.max()),
        'mean': float(spd_vals.mean())
    }

    if spd_ts is not None and acc_ts is not None:
        interp_func = interpolate.interp1d(spd_ts, spd_vals, kind='linear', fill_value='extrapolate')
        spd_up = interp_func(acc_ts)
        metadata['speed']['samples_upsampled'] = len(spd_up)
        metadata['speed']['upsampling'] = f"{spd_rate:.1f}Hz -> {acc_rate:.1f}Hz"
    else:
        spd_up = spd_vals

    df = pd.DataFrame({'accelerator': acc_vals, 'speed': spd_up})
    return df, metadata, None

all_dfs = []
all_meta = []
for jf in all_json_files:
    print(f"\nLoading {jf.name}...")
    df, meta, err = load_json_signals(jf)
    if df is not None:
        all_dfs.append(df)
        all_meta.append(meta)
        print(f"  Acc: {meta['accelerator']['samples_original']:,} samples @ {meta['accelerator']['sampling_rate_hz']:.1f}Hz")
        print(f"  Spd: {meta['speed']['samples_original']:,} samples @ {meta['speed']['sampling_rate_hz']:.1f}Hz -> upsampled")
    else:
        print(f"  SKIPPED: {err}")

if not all_dfs:
    print("ERROR: No A2D2 data loaded!")
    sys.exit(1)

combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nCombined: {len(combined_df):,} samples ({len(combined_df)/100:.1f}s @ 100Hz)")

train_df = combined_df.copy()
train_df.to_csv('a2d2_train.csv', index=False)

# Save A2D2 stats
results['a2d2'] = {
    'total_samples': len(combined_df),
    'duration_sec': len(combined_df) / 100.0,
    'n_datasets': len(all_dfs),
    'acc_mean': float(combined_df['accelerator'].mean()),
    'acc_std': float(combined_df['accelerator'].std()),
    'acc_min': float(combined_df['accelerator'].min()),
    'acc_max': float(combined_df['accelerator'].max()),
    'spd_mean': float(combined_df['speed'].mean()),
    'spd_std': float(combined_df['speed'].std()),
    'spd_min': float(combined_df['speed'].min()),
    'spd_max': float(combined_df['speed'].max()),
    'correlation': float(np.corrcoef(combined_df['accelerator'], combined_df['speed'])[0, 1]),
}

# Create Part 1 visualization
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(5, 3, hspace=0.4, wspace=0.3)

ax_info = fig.add_subplot(gs[0, :])
ax_info.axis('off')
info_data = []
for i, meta in enumerate(all_meta):
    info_data.append([
        f"Dataset {i+1}", meta['filename'], f"{len(all_dfs[i]):,}",
        f"{meta['accelerator']['sampling_rate_hz']:.1f}", meta['accelerator']['unit'],
        f"{meta['speed']['sampling_rate_hz']:.1f}", meta['speed'].get('upsampling', ''),
        meta['speed']['unit']
    ])
table = ax_info.table(
    cellText=info_data,
    colLabels=['Dataset', 'Filename', 'Samples', 'Acc Rate', 'Acc Unit', 'Speed Rate', 'Upsampling', 'Speed Unit'],
    cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)
for i in range(len(all_meta) + 1):
    for j in range(8):
        if i == 0:
            table[(i, j)].set_facecolor('#4CAF50')
            table[(i, j)].set_text_props(weight='bold', color='white')
        else:
            table[(i, j)].set_facecolor('#E8E8E8' if i % 2 == 0 else '#FFFFFF')
ax_info.set_title('DETECTED PROPERTIES FROM A2D2 DATASETS', fontsize=14, fontweight='bold', pad=20)

acc_unit = all_meta[0]['accelerator']['unit']
speed_unit = all_meta[0]['speed']['unit']

for i, sensor in enumerate(['accelerator', 'speed']):
    ax = fig.add_subplot(gs[1, i])
    n = min(len(train_df), 3000)
    t = np.arange(n) / 100.0
    ax.plot(t, train_df[sensor].values[:n], 'b-', alpha=0.7, linewidth=0.8)
    unit = acc_unit if sensor == 'accelerator' else speed_unit
    ax.set_ylabel(f'{sensor.capitalize()} ({unit})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{sensor.capitalize()} - First 30s', fontsize=12, fontweight='bold')

ax3 = fig.add_subplot(gs[1, 2])
ax3.hist(combined_df['accelerator'].values, bins=50, alpha=0.6, color='red', label=f'Acc ({acc_unit})', density=True)
ax3.hist(combined_df['speed'].values, bins=50, alpha=0.6, color='blue', label=f'Speed ({speed_unit})', density=True)
ax3.set_xlabel('Value', fontsize=10)
ax3.set_ylabel('Density', fontsize=10)
ax3.set_title('Signal Distributions', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

for i, sensor in enumerate(['accelerator', 'speed']):
    ax = fig.add_subplot(gs[2, i])
    ax.boxplot([combined_df[sensor].values], tick_labels=[sensor.capitalize()], vert=True)
    unit = acc_unit if sensor == 'accelerator' else speed_unit
    ax.set_ylabel(f'Value ({unit})', fontsize=10)
    ax.set_title(f'{sensor.capitalize()} Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

ax_corr = fig.add_subplot(gs[3, 0])
ax_corr.scatter(combined_df['accelerator'].values[::100], combined_df['speed'].values[::100], alpha=0.3, s=10)
ax_corr.set_xlabel(f'Accelerator ({acc_unit})', fontsize=10)
ax_corr.set_ylabel(f'Speed ({speed_unit})', fontsize=10)
corr = results['a2d2']['correlation']
ax_corr.set_title(f'Correlation = {corr:.4f}', fontsize=12, fontweight='bold')
ax_corr.grid(True, alpha=0.3)

acc_vel = np.diff(combined_df['accelerator'].values) * 100
ax_vel1 = fig.add_subplot(gs[3, 1])
ax_vel1.hist(acc_vel, bins=50, alpha=0.7, color='red', density=True)
ax_vel1.set_xlabel(f'Acc Velocity ({acc_unit}/s)', fontsize=10)
ax_vel1.set_ylabel('Density', fontsize=10)
ax_vel1.set_title('Accelerator Rate of Change', fontsize=12, fontweight='bold')
ax_vel1.grid(True, alpha=0.3)
ax_vel1.set_xlim(-50, 50)

speed_vel = np.diff(combined_df['speed'].values) * 100
ax_vel2 = fig.add_subplot(gs[3, 2])
ax_vel2.hist(speed_vel, bins=50, alpha=0.7, color='blue', density=True)
ax_vel2.set_xlabel(f'Speed Velocity ({speed_unit}/s)', fontsize=10)
ax_vel2.set_ylabel('Density', fontsize=10)
ax_vel2.set_title('Speed Rate of Change', fontsize=12, fontweight='bold')
ax_vel2.grid(True, alpha=0.3)
ax_vel2.set_xlim(-30, 30)

ax_summary = fig.add_subplot(gs[4, :])
ax_summary.axis('off')
summary_text = f"PART 1 SUMMARY\n{'='*60}\nDATASETS: {len(all_meta)} | TOTAL: {len(combined_df):,} samples ({len(combined_df)/100:.1f}s @ 100Hz)"
ax_summary.text(0.05, 0.5, summary_text, fontsize=10, family='monospace', verticalalignment='center', transform=ax_summary.transAxes)

plt.suptitle('PART 1: A2D2 DATA ANALYSIS WITH DETECTED PROPERTIES', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('part1_a2d2_comprehensive.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: part1_a2d2_comprehensive.png")

part1_end = time.time()
part1_duration = part1_end - part1_start
results['timing'] = {'part1': part1_duration}
print(f"PART 1 TIME: {part1_duration:.2f}s ({part1_duration/60:.2f}min)")
print()

# ============================================================================
# PART 2: SIMCLR TRAINING
# ============================================================================
part2_start = time.time()
print("=" * 80)
print("PART 2: SIMCLR TRAINING")
print("=" * 80)

# Normalize
scaler = StandardScaler()
train_normalized = scaler.fit_transform(train_df[['accelerator', 'speed']].values)
print(f"Scaler: acc mu={scaler.mean_[0]:.4f} sigma={scaler.scale_[0]:.4f}")
print(f"        spd mu={scaler.mean_[1]:.4f} sigma={scaler.scale_[1]:.4f}")

results['scaler'] = {
    'acc_mean': float(scaler.mean_[0]), 'acc_std': float(scaler.scale_[0]),
    'spd_mean': float(scaler.mean_[1]), 'spd_std': float(scaler.scale_[1])
}

# Create windows
def create_windows(data, window_size, stride):
    windows = []
    for start in range(0, len(data) - window_size + 1, stride):
        windows.append(data[start:start + window_size])
    return np.array(windows)

train_windows = create_windows(train_normalized, WINDOW_SIZE, STRIDE)
print(f"Windows: {len(train_windows)} (shape: {train_windows.shape})")

results['training'] = {
    'n_windows': int(len(train_windows)),
    'window_shape': list(train_windows.shape),
}

# Augmentation
def augment_window(window):
    aug = window.copy()
    aug += np.random.normal(0, JITTER_SIGMA, window.shape)
    aug *= np.random.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
    if np.random.random() < 0.5:
        ml = int(WINDOW_SIZE * MASK_RATIO)
        ms = np.random.randint(0, WINDOW_SIZE - ml)
        aug[ms:ms + ml] = 0
    return aug

# Dataset
class SimCLRDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        w = self.windows[idx]
        v1 = torch.FloatTensor(augment_window(w)).transpose(0, 1)
        v2 = torch.FloatTensor(augment_window(w)).transpose(0, 1)
        return v1, v2

train_dataset = SimCLRDataset(train_windows)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

results['training']['batches_per_epoch'] = len(train_loader)
results['training']['total_steps'] = EPOCHS * len(train_loader)

# Model
class Encoder(nn.Module):
    def __init__(self, input_channels=2, embedding_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(128, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        return self.global_pool(x).squeeze(-1)

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=256, projection_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim, projection_dim)
    def forward(self, x):
        return self.fc2(self.relu(self.bn1(self.fc1(x))))

class SimCLRModel(nn.Module):
    def __init__(self, input_channels=2, embedding_dim=256, projection_dim=128):
        super().__init__()
        self.encoder = Encoder(input_channels, embedding_dim)
        self.projection_head = ProjectionHead(embedding_dim, projection_dim)
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

model = SimCLRModel(INPUT_CHANNELS, EMBEDDING_DIM, PROJECTION_DIM).to(DEVICE)
enc_params = sum(p.numel() for p in model.encoder.parameters())
proj_params = sum(p.numel() for p in model.projection_head.parameters())
total_params = enc_params + proj_params
print(f"Params: encoder={enc_params:,}, proj={proj_params:,}, total={total_params:,}")

results['training']['encoder_params'] = enc_params
results['training']['projection_params'] = proj_params
results['training']['total_params'] = total_params

# NT-Xent Loss
def nt_xent_loss(z1, z2, temperature=0.5):
    bs = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * bs, dtype=torch.bool, device=DEVICE)
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.cat([torch.arange(bs) + bs, torch.arange(bs)]).to(DEVICE)
    pos_sim = sim[torch.arange(2 * bs), pos]
    loss = -pos_sim + torch.logsumexp(sim, dim=1)
    return loss.mean()

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.train()
loss_history = []
epoch_losses = []

training_start = time.time()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for view1, view2 in train_loader:
        view1, view2 = view1.to(DEVICE), view2.to(DEVICE)
        _, z1 = model(view1)
        _, z2 = model(view2)
        loss = nt_xent_loss(z1, z2, TEMPERATURE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loss_history.append(loss.item())
    avg_loss = epoch_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch [{epoch+1:2d}/{EPOCHS}] Loss: {avg_loss:.4f}")

training_end = time.time()
training_duration = training_end - training_start
print(f"\nTraining complete! Time: {training_duration:.2f}s ({training_duration/60:.2f}min)")
print(f"Initial loss: {epoch_losses[0]:.4f}, Final loss: {epoch_losses[-1]:.4f}")

results['training']['initial_loss'] = float(epoch_losses[0])
results['training']['final_loss'] = float(epoch_losses[-1])
results['training']['loss_reduction_pct'] = float((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100)
results['training']['training_time_sec'] = training_duration
results['training']['device'] = str(DEVICE)

# Save model
checkpoint = {
    'encoder_state_dict': model.encoder.state_dict(),
    'scaler': scaler,
    'window_size': WINDOW_SIZE,
    'stride': STRIDE,
    'sensor_names': ['accelerator', 'speed'],
    'embedding_dim': EMBEDDING_DIM,
    'final_loss': epoch_losses[-1],
    'training_time': training_duration
}
torch.save(checkpoint, 'simclr_encoder_final.pth')

# Augmentation visualization
sample_window = train_windows[np.random.randint(0, len(train_windows))]
aug1_jitter = sample_window + np.random.normal(0, JITTER_SIGMA, sample_window.shape)
aug2_scale = sample_window * 1.15
aug3_mask = sample_window.copy()
ml = int(WINDOW_SIZE * MASK_RATIO)
ms = 50
aug3_mask[ms:ms + ml] = 0

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for si, sn in enumerate(['Accelerator', 'Speed']):
    axes[si, 0].plot(sample_window[:, si], 'b-', linewidth=1.5)
    axes[si, 0].set_title(f'{sn}: Original', fontweight='bold')
    axes[si, 0].grid(True, alpha=0.3)
    axes[si, 1].plot(aug1_jitter[:, si], 'g-', linewidth=1.5)
    axes[si, 1].set_title(f'{sn}: Jittered (sigma={JITTER_SIGMA})', fontweight='bold')
    axes[si, 1].grid(True, alpha=0.3)
    axes[si, 2].plot(aug2_scale[:, si], 'r-', linewidth=1.5)
    axes[si, 2].set_title(f'{sn}: Scaled (x1.15)', fontweight='bold')
    axes[si, 2].grid(True, alpha=0.3)
    axes[si, 3].plot(aug3_mask[:, si], 'm-', linewidth=1.5)
    axes[si, 3].set_title(f'{sn}: Masked ({int(MASK_RATIO*100)}%)', fontweight='bold')
    axes[si, 3].grid(True, alpha=0.3)
    axes[si, 3].axvspan(ms, ms+ml, alpha=0.3, color='red')
plt.suptitle('SIMCLR DATA AUGMENTATION EXAMPLES', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('part2_augmentation_examples.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: part2_augmentation_examples.png")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(loss_history, linewidth=0.5, alpha=0.7)
axes[0].set_xlabel('Training Step')
axes[0].set_ylabel('NT-Xent Loss')
axes[0].set_title('Training Loss (All Steps)', fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[1].plot(range(1, EPOCHS+1), epoch_losses, 'o-', linewidth=2, markersize=5)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Average Loss')
axes[1].set_title('Loss per Epoch', fontweight='bold')
axes[1].grid(True, alpha=0.3)
stats_text = f"Final Loss: {epoch_losses[-1]:.4f}\nInitial Loss: {epoch_losses[0]:.4f}\nImprovement: {results['training']['loss_reduction_pct']:.1f}%\nTraining Time: {training_duration/60:.1f} min"
axes[1].text(0.98, 0.95, stats_text, transform=axes[1].transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.suptitle('PART 2: SIMCLR TRAINING CURVES', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('part2_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: part2_training_curves.png")

part2_end = time.time()
part2_duration = part2_end - part2_start
results['timing']['part2'] = part2_duration
results['timing']['training_only'] = training_duration
results['timing']['part2_other'] = part2_duration - training_duration
print(f"PART 2 TIME: {part2_duration:.2f}s ({part2_duration/60:.2f}min)")
print()

# ============================================================================
# PART 3: HIL FAULT DETECTION
# ============================================================================
part3_start = time.time()
print("=" * 80)
print("PART 3: ANOMALY DETECTION ON HIL DATA")
print("=" * 80)

# Load encoder
encoder = Encoder(input_channels=2, embedding_dim=256).to(DEVICE)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder.eval()

# HIL CSV parser
def parse_hil_csv(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    cols = None
    for line in lines[:50]:
        if line.startswith('path,'):
            cols = line.strip().split(',')[1:]
            break
    if cols is None:
        return None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('trace_values,'):
            data_start = i
            break
    if data_start is None:
        return None
    rows = []
    for line in lines[data_start:]:
        parts = line.strip().split(',')[1:]
        if len(parts) == len(cols):
            rows.append(parts)
    df = pd.DataFrame(rows, columns=cols)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def extract_sensors(df, filename):
    speed_col = accel_col = None
    for col in df.columns:
        if 'v_Vehicle' in col or 'vehicle_speed' in col or 'speed' in col.lower():
            speed_col = col
            break
    for col in df.columns:
        if 'AccPedal' in col or 'accelerator' in col.lower():
            accel_col = col
            break
    if speed_col is None or accel_col is None:
        print(f"  Missing columns in {filename}: speed={speed_col}, accel={accel_col}")
        print(f"  Available: {list(df.columns)}")
        return None
    acc = df[accel_col].values
    spd = df[speed_col].values
    if np.any(np.isnan(acc)) or np.any(np.isnan(spd)):
        acc = pd.Series(acc).interpolate(method='linear').ffill().bfill().values
        spd = pd.Series(spd).interpolate(method='linear').ffill().bfill().values
    return pd.DataFrame({'accelerator': acc, 'speed': spd})

def extract_embeddings(windows):
    embs = []
    with torch.no_grad():
        for w in windows:
            t = torch.FloatTensor(w).transpose(0, 1).unsqueeze(0).to(DEVICE)
            embs.append(encoder(t).cpu().numpy().flatten())
    return np.array(embs)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# Load healthy data
HIL_PATH = REPO_ROOT  # Files are in repo root
healthy_path = HIL_PATH / "healthy.csv"
print(f"Loading healthy: {healthy_path}")

healthy_raw = parse_hil_csv(healthy_path)
healthy_sensors = extract_sensors(healthy_raw, "healthy.csv")
cutoff = HIL_HEALTHY_DURATION_SECONDS * HIL_SAMPLING_RATE
healthy_df = healthy_sensors.iloc[:cutoff].copy()
print(f"  Full: {len(healthy_sensors)} samples, Using first {HIL_HEALTHY_DURATION_SECONDS}s: {len(healthy_df)} samples")

# Load fault data
fault_files = [
    "acc fault gain.csv", "acc fault noise.csv", "acc fault stuck.csv",
    "rpm fault gain.csv", "rpm fault noise.csv", "rpm fault stuck at.csv"
]

fault_raw_data = {}
for ff in fault_files:
    fp = HIL_PATH / ff
    if fp.exists():
        raw = parse_hil_csv(fp)
        if raw is not None:
            sensors = extract_sensors(raw, ff)
            if sensors is not None:
                fault_raw_data[ff] = sensors
                print(f"  Loaded {ff}: {len(sensors)} samples")

print(f"Loaded {len(fault_raw_data)} fault files")

# Extract embeddings
embeddings_start = time.time()

healthy_data = healthy_df[['accelerator', 'speed']].values
healthy_norm = scaler.transform(healthy_data)
healthy_windows = create_windows(healthy_norm, WINDOW_SIZE, STRIDE)
healthy_embs = extract_embeddings(healthy_windows)
healthy_mean = np.mean(healthy_embs, axis=0)
healthy_sims = [cosine_sim(e, healthy_mean) for e in healthy_embs]

print(f"  Healthy: {len(healthy_windows)} windows")

results['healthy'] = {
    'total_samples': len(healthy_sensors),
    'used_samples': len(healthy_df),
    'n_windows': len(healthy_windows),
}

fault_cache = {}
for ff, fdf in fault_raw_data.items():
    fdata = fdf[['accelerator', 'speed']].values
    fnorm = scaler.transform(pd.DataFrame(fdata, columns=['accelerator', 'speed']))
    fwin = create_windows(fnorm, WINDOW_SIZE, STRIDE)
    fembs = extract_embeddings(fwin)
    fsims = [cosine_sim(e, healthy_mean) for e in fembs]
    fault_cache[ff] = {
        'windows': fwin, 'embeddings': fembs, 'similarities': fsims,
        'n_windows': len(fwin)
    }
    print(f"  {ff}: {len(fwin)} windows")

embeddings_end = time.time()
embeddings_duration = embeddings_end - embeddings_start
results['timing']['embeddings'] = embeddings_duration
print(f"Embeddings time: {embeddings_duration:.2f}s")

# Evaluate all thresholds
eval_start = time.time()
all_threshold_results = {}

for pctl in THRESHOLD_PERCENTILES:
    threshold = np.percentile(healthy_sims, pctl)
    thresh_results = []
    all_sims_roc = list(healthy_sims)
    all_labels_roc = [0] * len(healthy_sims)

    for ff, cache in fault_cache.items():
        sims = cache['similarities']
        preds = (np.array(sims) < threshold).astype(int)
        y_true = np.ones(len(preds))
        detected = int(preds.sum())
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        acc = float(accuracy_score(y_true, preds))

        all_sims_roc.extend(sims)
        all_labels_roc.extend([1] * len(sims))

        thresh_results.append({
            'fault': ff, 'windows': len(sims), 'detected': detected,
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc
        })

    try:
        roc_auc = float(roc_auc_score(all_labels_roc, [-s for s in all_sims_roc]))
    except:
        roc_auc = 0.0

    all_threshold_results[pctl] = {
        'threshold_value': float(threshold),
        'results': thresh_results,
        'avg_precision': float(np.mean([r['precision'] for r in thresh_results])),
        'avg_recall': float(np.mean([r['recall'] for r in thresh_results])),
        'avg_f1': float(np.mean([r['f1'] for r in thresh_results])),
        'avg_accuracy': float(np.mean([r['accuracy'] for r in thresh_results])),
        'roc_auc': roc_auc,
        'total_windows': sum(r['windows'] for r in thresh_results),
        'total_detected': sum(r['detected'] for r in thresh_results),
        'all_sims': all_sims_roc,
        'all_labels': all_labels_roc
    }

    print(f"  {pctl}th pctl: threshold={threshold:.4f} Recall={all_threshold_results[pctl]['avg_recall']:.2%} F1={all_threshold_results[pctl]['avg_f1']:.2%}")

eval_end = time.time()
eval_duration = eval_end - eval_start
results['timing']['evaluation'] = eval_duration

# Binary classification
results['binary'] = {}
for pctl in THRESHOLD_PERCENTILES:
    data = all_threshold_results[pctl]
    threshold = data['threshold_value']
    all_labels = np.array(data['all_labels'])
    all_sims = np.array(data['all_sims'])
    all_preds = (all_sims < threshold).astype(int)

    bp = float(precision_score(all_labels, all_preds, zero_division=0))
    br = float(recall_score(all_labels, all_preds, zero_division=0))
    bf1 = float(f1_score(all_labels, all_preds, zero_division=0))
    ba = float(accuracy_score(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    results['binary'][str(pctl)] = {
        'precision': bp, 'recall': br, 'f1': bf1, 'accuracy': ba,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
    print(f"  Binary {pctl}th: P={bp:.4f} R={br:.4f} F1={bf1:.4f} Acc={ba:.4f} TN={tn} FP={fp} FN={fn} TP={tp}")

# Store per-threshold results
results['thresholds'] = {}
for pctl in THRESHOLD_PERCENTILES:
    data = all_threshold_results[pctl]
    results['thresholds'][str(pctl)] = {
        'threshold_value': data['threshold_value'],
        'avg_precision': data['avg_precision'],
        'avg_recall': data['avg_recall'],
        'avg_f1': data['avg_f1'],
        'avg_accuracy': data['avg_accuracy'],
        'roc_auc': data['roc_auc'],
        'per_fault': data['results']
    }

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\nCreating visualizations...")

# 1. Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()
for idx, pctl in enumerate(THRESHOLD_PERCENTILES):
    data = all_threshold_results[pctl]
    threshold = data['threshold_value']
    all_labels = np.array(data['all_labels'])
    all_sims = np.array(data['all_sims'])
    all_preds = (all_sims < threshold).astype(int)
    cm = confusion_matrix(all_labels, all_preds)
    ax = axes_flat[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Healthy', 'Faulty'], yticklabels=['Healthy', 'Faulty'])
    ax.set_title(f'{pctl}th Percentile\nThreshold={threshold:.3f}', fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
plt.suptitle('CONFUSION MATRICES: Binary Classification (Healthy vs Faulty)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('part3_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: part3_confusion_matrices.png")

# 2. ROC curves
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes_flat = axes.flatten()
for idx, pctl in enumerate(THRESHOLD_PERCENTILES):
    data = all_threshold_results[pctl]
    fpr, tpr, _ = roc_curve(data['all_labels'], [-s for s in data['all_sims']])
    ax = axes_flat[idx]
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {data["roc_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{pctl}th Percentile\nROC-AUC = {data["roc_auc"]:.4f}', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
plt.suptitle('ROC CURVES FOR ALL THRESHOLDS', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('part3_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: part3_roc_curves.png")

# 3. Metrics vs threshold
fig, ax = plt.subplots(figsize=(10, 6))
metrics_data = {
    'Recall': [all_threshold_results[p]['avg_recall'] for p in THRESHOLD_PERCENTILES],
    'Precision': [all_threshold_results[p]['avg_precision'] for p in THRESHOLD_PERCENTILES],
    'F1': [all_threshold_results[p]['avg_f1'] for p in THRESHOLD_PERCENTILES],
    'Accuracy': [all_threshold_results[p]['avg_accuracy'] for p in THRESHOLD_PERCENTILES],
    'ROC-AUC': [all_threshold_results[p]['roc_auc'] for p in THRESHOLD_PERCENTILES]
}
for mn, mv in metrics_data.items():
    ax.plot(THRESHOLD_PERCENTILES, mv, 'o-', linewidth=2, markersize=8, label=mn)
ax.set_xlabel('Threshold Percentile', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
ax.set_title('PERFORMANCE METRICS VS THRESHOLD PERCENTILE', fontsize=14, fontweight='bold')
ax.set_xticks(THRESHOLD_PERCENTILES)
ax.set_xticklabels([f'{p}th' for p in THRESHOLD_PERCENTILES])
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('part3_metrics_vs_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: part3_metrics_vs_threshold.png")

# 4. Recall heatmap
fig, ax = plt.subplots(figsize=(12, 6))
fault_names = [r['fault'].replace('.csv', '') for r in all_threshold_results[THRESHOLD_PERCENTILES[0]]['results']]
n_faults = len(fault_names)
n_thresholds = len(THRESHOLD_PERCENTILES)
recall_matrix = np.zeros((n_faults, n_thresholds))
for j, pctl in enumerate(THRESHOLD_PERCENTILES):
    for i, r in enumerate(all_threshold_results[pctl]['results']):
        recall_matrix[i, j] = r['recall']
im = ax.imshow(recall_matrix, cmap='RdYlGn', aspect='auto', vmin=0.75, vmax=1.0)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Recall', fontsize=12)
ax.set_yticks(range(n_faults))
ax.set_yticklabels(fault_names, fontsize=10)
ax.set_xticks(range(n_thresholds))
ax.set_xticklabels([f'{p}th' for p in THRESHOLD_PERCENTILES])
for i in range(n_faults):
    for j in range(n_thresholds):
        v = recall_matrix[i, j]
        color = 'white' if v < 0.85 else 'black'
        ax.text(j, i, f'{v:.0%}', ha='center', va='center', color=color, fontsize=9)
ax.set_xlabel('Threshold Percentile', fontsize=12, fontweight='bold')
ax.set_ylabel('Fault Type', fontsize=12, fontweight='bold')
ax.set_title('RECALL HEAT MAP: Per-Fault Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('part3_recall_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: part3_recall_heatmap.png")

# 5. Similarity distributions
fig, ax = plt.subplots(figsize=(14, 7))
ax.hist(healthy_sims, bins=30, alpha=0.5, color='green', label='Healthy', density=True)
all_fault_sims = []
for ff, cache in fault_cache.items():
    all_fault_sims.extend(cache['similarities'])
ax.hist(all_fault_sims, bins=30, alpha=0.5, color='red', label='Faulty (All)', density=True)
threshold_colors = ['purple', 'blue', 'cyan', 'orange', 'brown', 'magenta']
for i, pctl in enumerate(THRESHOLD_PERCENTILES):
    thr = np.percentile(healthy_sims, pctl)
    ax.axvline(thr, linestyle='--', linewidth=2, color=threshold_colors[i],
               label=f'{pctl}th %ile ({thr:.3f})')
ax.set_xlabel('Cosine Similarity to Healthy Centroid', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12, fontweight='bold')
ax.set_title('SIMILARITY DISTRIBUTIONS: Healthy vs Faulty', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part3_similarity_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: part3_similarity_distributions.png")

part3_end = time.time()
part3_duration = part3_end - part3_start
results['timing']['part3'] = part3_duration
results['timing']['part3_embeddings'] = embeddings_duration
results['timing']['part3_evaluation'] = eval_duration
results['timing']['part3_other'] = part3_duration - embeddings_duration - eval_duration
results['timing']['total'] = part1_duration + part2_duration + part3_duration

print(f"\nPART 3 TIME: {part3_duration:.2f}s ({part3_duration/60:.2f}min)")
print(f"TOTAL PIPELINE: {results['timing']['total']:.2f}s ({results['timing']['total']/60:.2f}min)")

# Remove non-serializable data from results before saving
for pctl_key in results['thresholds']:
    pass  # Already clean

# Save results JSON (excluding large arrays)
results_clean = {}
for k, v in results.items():
    results_clean[k] = v

# Remove all_sims and all_labels from threshold results (too large)
with open('pipeline_results.json', 'w') as f:
    json.dump(results_clean, f, indent=2, default=str)

print("\nSaved: pipeline_results.json")
print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
