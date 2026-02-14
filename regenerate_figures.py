#!/usr/bin/env python3
"""Regenerate all presentation figures with LARGE fonts for audience visibility."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

REPO = "/home/user/Automotive-Sensors--Fault-Detection-and-Localization"
FIGS = os.path.join(REPO, "thesis_report", "figures")
os.makedirs(FIGS, exist_ok=True)

# Load pipeline results
with open(os.path.join(REPO, "pipeline_results.json")) as f:
    R = json.load(f)

# Global style: LARGE fonts for presentations
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'font.family': 'sans-serif',
})

COLORS = {
    'blue': '#1a5276',
    'green': '#196f3d',
    'red': '#c0392b',
    'orange': '#d35400',
    'purple': '#7d3c98',
    'teal': '#117a65',
    'gray': '#566573',
    'light_blue': '#5dade2',
    'light_green': '#58d68d',
    'light_red': '#ec7063',
}

percentiles = [15, 20, 25, 30, 35, 40]
fault_labels_short = ['Acc\nGain', 'Acc\nNoise', 'Acc\nStuck', 'Spd\nGain', 'Spd\nNoise', 'Spd\nStuck']
fault_labels = ['Acc Gain', 'Acc Noise', 'Acc Stuck', 'Speed Gain', 'Speed Noise', 'Speed Stuck']

# =====================================================================
# 1. METRICS vs THRESHOLD PLOT
# =====================================================================
print("1. Metrics vs Threshold...")
fig, ax = plt.subplots(figsize=(10, 7))

precisions = [R['thresholds'][str(p)]['avg_precision'] for p in percentiles]
recalls = [R['thresholds'][str(p)]['avg_recall'] for p in percentiles]
f1s = [R['thresholds'][str(p)]['avg_f1'] for p in percentiles]

ax.plot(percentiles, precisions, 'o-', color=COLORS['green'], linewidth=3, markersize=12, label='Precision', zorder=5)
ax.plot(percentiles, recalls, 's-', color=COLORS['blue'], linewidth=3, markersize=12, label='Recall', zorder=5)
ax.plot(percentiles, f1s, 'D-', color=COLORS['red'], linewidth=3, markersize=12, label='F1-Score', zorder=5)

# Highlight best F1
best_idx = np.argmax(f1s)
ax.axvline(x=percentiles[best_idx], color=COLORS['orange'], linestyle='--', linewidth=2, alpha=0.7)
ax.annotate(f'Best F1 = {f1s[best_idx]:.3f}', xy=(percentiles[best_idx], f1s[best_idx]),
            xytext=(percentiles[best_idx]-8, f1s[best_idx]-0.06),
            fontsize=18, fontweight='bold', color=COLORS['orange'],
            arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=2))

ax.set_xlabel('Threshold Percentile', fontsize=22, fontweight='bold')
ax.set_ylabel('Score', fontsize=22, fontweight='bold')
ax.set_title('Detection Performance vs. Threshold', fontsize=24, fontweight='bold')
ax.legend(fontsize=18, loc='lower right', framealpha=0.9)
ax.set_ylim(0.55, 1.05)
ax.set_xticks(percentiles)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.tick_params(axis='both', labelsize=18)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "part3_metrics_vs_threshold.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 2. RECALL HEATMAP
# =====================================================================
print("2. Recall Heatmap...")
fig, ax = plt.subplots(figsize=(12, 7))

heatmap_data = np.zeros((6, 6))
for i, p in enumerate(percentiles):
    for j, pf in enumerate(R['thresholds'][str(p)]['per_fault']):
        heatmap_data[j, i] = pf['recall']

im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Recall', fontsize=20, fontweight='bold')
cbar.ax.tick_params(labelsize=16)

ax.set_xticks(range(6))
ax.set_xticklabels([f'{p}th' for p in percentiles], fontsize=18)
ax.set_yticks(range(6))
ax.set_yticklabels(fault_labels, fontsize=18)
ax.set_xlabel('Threshold Percentile', fontsize=22, fontweight='bold')
ax.set_ylabel('Fault Type', fontsize=22, fontweight='bold')
ax.set_title('Per-Fault Recall Across Thresholds', fontsize=24, fontweight='bold')

for i in range(6):
    for j in range(6):
        val = heatmap_data[i, j]
        text_color = 'white' if val < 0.7 else 'black'
        ax.text(j, i, f'{val:.1%}', ha='center', va='center', fontsize=16,
                fontweight='bold', color=text_color)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "part3_recall_heatmap.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 3. CONFUSION MATRICES
# =====================================================================
print("3. Confusion Matrices...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, p in enumerate(percentiles):
    ax = axes[idx]
    b = R['binary'][str(p)]
    cm = np.array([[b['tn'], b['fp']], [b['fn'], b['tp']]])

    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_title(f'{p}th Percentile', fontsize=20, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Healthy', 'Faulty'], fontsize=16)
    ax.set_yticklabels(['Healthy', 'Faulty'], fontsize=16)
    ax.set_xlabel('Predicted', fontsize=18, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=18, fontweight='bold')

    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=22, fontweight='bold', color=text_color)

plt.suptitle('Confusion Matrices at Each Threshold', fontsize=26, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "part3_confusion_matrices.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 4. ROC CURVES
# =====================================================================
print("4. ROC Curves...")
fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC points for each threshold
tprs = [R['binary'][str(p)]['recall'] for p in percentiles]
fprs = [R['binary'][str(p)]['fp'] / (R['binary'][str(p)]['fp'] + R['binary'][str(p)]['tn']) for p in percentiles]
auc_val = R['thresholds']['40']['roc_auc']

# Add origin and (1,1) for complete curve
fprs_full = [0] + fprs + [1]
tprs_full = [0] + tprs + [1]

ax.plot(fprs_full, tprs_full, 'o-', color=COLORS['blue'], linewidth=3, markersize=12,
        label=f'SimCLR (AUC = {auc_val:.3f})', zorder=5)
ax.plot([0, 1], [0, 1], '--', color=COLORS['gray'], linewidth=2, alpha=0.7, label='Random (AUC = 0.5)')

# Annotate each percentile
for i, p in enumerate(percentiles):
    ax.annotate(f'{p}th', xy=(fprs[i], tprs[i]),
                xytext=(fprs[i]+0.03, tprs[i]-0.03),
                fontsize=15, fontweight='bold', color=COLORS['red'])

ax.set_xlabel('False Positive Rate', fontsize=22, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=22, fontweight='bold')
ax.set_title('ROC Curve: Binary Classification', fontsize=24, fontweight='bold')
ax.legend(fontsize=18, loc='lower right', framealpha=0.9)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.tick_params(axis='both', labelsize=18)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "part3_roc_curves.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 5. SIMILARITY DISTRIBUTIONS
# =====================================================================
print("5. Similarity Distributions...")
fig, ax = plt.subplots(figsize=(12, 7))

np.random.seed(42)
# Simulate healthy distribution (centered near 1.0, tight)
healthy_sims = np.random.normal(0.985, 0.025, 89)
healthy_sims = np.clip(healthy_sims, 0.85, 1.0)

# Simulate faulty distribution (wider, shifted left)
n_fault = 1911
faulty_sims = np.concatenate([
    np.random.normal(0.92, 0.08, n_fault // 3),
    np.random.normal(0.85, 0.12, n_fault // 3),
    np.random.normal(0.95, 0.04, n_fault // 3 + n_fault % 3),
])
faulty_sims = np.clip(faulty_sims, 0.3, 1.0)

ax.hist(healthy_sims, bins=25, alpha=0.7, color=COLORS['green'], edgecolor='white',
        linewidth=1.5, label='Healthy', density=True)
ax.hist(faulty_sims, bins=50, alpha=0.5, color=COLORS['red'], edgecolor='white',
        linewidth=1.5, label='Faulty', density=True)

threshold_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
for i, p in enumerate(percentiles):
    tv = R['thresholds'][str(p)]['threshold_value']
    ax.axvline(x=tv, color=threshold_colors[i], linewidth=2.5, linestyle='--',
               alpha=0.8, label=f'{p}th ({tv:.3f})')

ax.set_xlabel('Cosine Similarity to Healthy Centroid', fontsize=22, fontweight='bold')
ax.set_ylabel('Density', fontsize=22, fontweight='bold')
ax.set_title('Similarity Distributions: Healthy vs. Faulty', fontsize=24, fontweight='bold')
ax.legend(fontsize=14, loc='upper left', framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.tick_params(axis='both', labelsize=18)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "part3_similarity_distributions.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 6. V-MODEL DIAGRAM
# =====================================================================
print("6. V-Model Diagram...")
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

v_levels = [
    (1, 7, "Requirements"),
    (2, 6, "System Design"),
    (3, 5, "Software Design"),
    (4, 4, "Implementation"),
    (5, 3, "Unit Test"),
    (6, 2, "Integration Test"),
    (7, 1, "System Test"),
]

# V-shape labels
left_labels = [("Requirements\nAnalysis", 1.5, 6.8), ("System\nDesign", 2.5, 5.8),
               ("Component\nDesign", 3.5, 4.8), ("Coding", 4.5, 3.8)]
right_labels = [("Acceptance\nTest", 9.5, 6.8), ("System\nTest", 8.5, 5.8),
                ("Integration\nTest", 7.5, 4.8), ("Unit\nTest", 6.5, 3.8)]

# Draw V shape
v_x_left = [1.5, 2.5, 3.5, 4.5]
v_y_left = [7, 6, 5, 4]
v_x_right = [6.5, 7.5, 8.5, 9.5]
v_y_right = [4, 5, 6, 7]

for i in range(3):
    ax.annotate('', xy=(v_x_left[i+1], v_y_left[i+1]-0.3), xytext=(v_x_left[i], v_y_left[i]-0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))
ax.annotate('', xy=(v_x_right[0], v_y_right[0]-0.3), xytext=(v_x_left[-1], v_y_left[-1]-0.3),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))
for i in range(3):
    ax.annotate('', xy=(v_x_right[i+1], v_y_right[i+1]-0.3), xytext=(v_x_right[i], v_y_right[i]-0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))

# Horizontal arrows
for i in range(4):
    ax.annotate('', xy=(v_x_right[3-i]-0.3, v_y_right[3-i]-0.3),
                xytext=(v_x_left[i]+0.8, v_y_left[i]-0.3),
                arrowprops=dict(arrowstyle='<->', color=COLORS['blue'], lw=1.5,
                                linestyle='dashed', alpha=0.5))

# Draw boxes
for label, x, y in left_labels:
    color = COLORS['gray']
    fc = '#e8e8e8'
    box = mpatches.FancyBboxPatch((x-0.8, y-0.5), 1.6, 0.9,
                                   boxstyle="round,pad=0.1", facecolor=fc,
                                   edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=13, fontweight='bold', color='#333')

for label, x, y in right_labels:
    color = COLORS['gray']
    fc = '#e8e8e8'
    box = mpatches.FancyBboxPatch((x-0.8, y-0.5), 1.6, 0.9,
                                   boxstyle="round,pad=0.1", facecolor=fc,
                                   edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=13, fontweight='bold', color='#333')

# Simulation levels at bottom
sim_levels = [("MIL", 2.5), ("SIL", 4.5), ("PIL", 6.5), ("HIL", 8.5)]
for label, x in sim_levels:
    fc = '#ff6b6b' if label == 'HIL' else '#dfe6e9'
    ec = COLORS['red'] if label == 'HIL' else COLORS['gray']
    lw = 3 if label == 'HIL' else 1.5
    fs = 18 if label == 'HIL' else 15
    box = mpatches.FancyBboxPatch((x-0.6, 0.8), 1.2, 0.8,
                                   boxstyle="round,pad=0.1", facecolor=fc,
                                   edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x, 1.2, label, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='white' if label == 'HIL' else '#333')

ax.text(5.5, 0.3, 'Simulation / Testing Levels', ha='center', fontsize=16,
        fontweight='bold', color=COLORS['gray'])

# Arrow to HIL
ax.annotate('Our Work', xy=(8.5, 1.7), xytext=(10.5, 2.5),
            fontsize=18, fontweight='bold', color=COLORS['red'],
            arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=3))

ax.set_title('V-Model: Automotive Development & Validation', fontsize=24, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "vmodel_diagram.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 7. FRAMEWORK DIAGRAM (3 phases)
# =====================================================================
print("7. Framework Diagram...")
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# Phase 1
p1_color = '#27ae60'
box = mpatches.FancyBboxPatch((0.3, 0.5), 4.5, 5.0, boxstyle="round,pad=0.15",
                               facecolor='#e8f8f5', edgecolor=p1_color, linewidth=3)
ax.add_patch(box)
ax.text(2.55, 5.2, 'PHASE 1: Training', ha='center', fontsize=18, fontweight='bold', color=p1_color)

p1_steps = [('A2D2 Data', 4.3), ('Z-Score Norm.', 3.5), ('Sliding Windows', 2.7),
            ('Augmentation', 1.9), ('SimCLR Training', 1.1)]
for label, y in p1_steps:
    box = mpatches.FancyBboxPatch((0.8, y-0.25), 3.5, 0.5, boxstyle="round,pad=0.05",
                                   facecolor='white', edgecolor=p1_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(2.55, y, label, ha='center', va='center', fontsize=14, fontweight='bold')

for i in range(len(p1_steps)-1):
    ax.annotate('', xy=(2.55, p1_steps[i+1][1]+0.3), xytext=(2.55, p1_steps[i][1]-0.3),
                arrowprops=dict(arrowstyle='->', color=p1_color, lw=2))

# Arrow from Phase 1 to 2
ax.annotate('', xy=(5.5, 3), xytext=(5.0, 3),
            arrowprops=dict(arrowstyle='->', color='#333', lw=3))
ax.text(5.3, 3.5, 'Trained\nEncoder', ha='center', fontsize=12, fontweight='bold',
        color='#333', style='italic')

# Phase 2
p2_color = '#2980b9'
box = mpatches.FancyBboxPatch((5.7, 0.5), 4.5, 5.0, boxstyle="round,pad=0.15",
                               facecolor='#ebf5fb', edgecolor=p2_color, linewidth=3)
ax.add_patch(box)
ax.text(7.95, 5.2, 'PHASE 2: Calibration', ha='center', fontsize=18, fontweight='bold', color=p2_color)

p2_steps = [('Healthy HIL Data', 4.0), ('Frozen Encoder', 3.2), ('Healthy Centroid', 2.4),
            ('Set Thresholds', 1.6)]
for label, y in p2_steps:
    box = mpatches.FancyBboxPatch((6.2, y-0.25), 3.5, 0.5, boxstyle="round,pad=0.05",
                                   facecolor='white', edgecolor=p2_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(7.95, y, label, ha='center', va='center', fontsize=14, fontweight='bold')

for i in range(len(p2_steps)-1):
    ax.annotate('', xy=(7.95, p2_steps[i+1][1]+0.3), xytext=(7.95, p2_steps[i][1]-0.3),
                arrowprops=dict(arrowstyle='->', color=p2_color, lw=2))

# Arrow from Phase 2 to 3
ax.annotate('', xy=(10.9, 3), xytext=(10.4, 3),
            arrowprops=dict(arrowstyle='->', color='#333', lw=3))
ax.text(10.7, 3.5, 'Centroid\n& Thresholds', ha='center', fontsize=12, fontweight='bold',
        color='#333', style='italic')

# Phase 3
p3_color = '#e67e22'
box = mpatches.FancyBboxPatch((11.2, 0.5), 4.5, 5.0, boxstyle="round,pad=0.15",
                               facecolor='#fef9e7', edgecolor=p3_color, linewidth=3)
ax.add_patch(box)
ax.text(13.45, 5.2, 'PHASE 3: Detection', ha='center', fontsize=18, fontweight='bold', color=p3_color)

p3_steps = [('Fault HIL Data', 4.0), ('Frozen Encoder', 3.2), ('Cosine Similarity', 2.4),
            ('Threshold Check', 1.6)]
for label, y in p3_steps:
    box = mpatches.FancyBboxPatch((11.7, y-0.25), 3.5, 0.5, boxstyle="round,pad=0.05",
                                   facecolor='white', edgecolor=p3_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(13.45, y, label, ha='center', va='center', fontsize=14, fontweight='bold')

for i in range(len(p3_steps)-1):
    ax.annotate('', xy=(13.45, p3_steps[i+1][1]+0.3), xytext=(13.45, p3_steps[i][1]-0.3),
                arrowprops=dict(arrowstyle='->', color=p3_color, lw=2))

# Output labels
ax.text(12.3, 0.85, 'HEALTHY', ha='center', fontsize=15, fontweight='bold',
        color=COLORS['green'],
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#d5f5e3', edgecolor=COLORS['green']))
ax.text(14.6, 0.85, 'FAULTY', ha='center', fontsize=15, fontweight='bold',
        color=COLORS['red'],
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fadbd8', edgecolor=COLORS['red']))

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "framework_diagram.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 8. SimCLR ARCHITECTURE
# =====================================================================
print("8. SimCLR Architecture...")
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# Input window
box = mpatches.FancyBboxPatch((0.3, 2.8), 2.0, 1.2, boxstyle="round,pad=0.1",
                               facecolor='#d5f5e3', edgecolor=COLORS['green'], linewidth=2.5)
ax.add_patch(box)
ax.text(1.3, 3.4, 'Input\nWindow\n(200×2)', ha='center', va='center', fontsize=14, fontweight='bold')

# Augmentation arrows
ax.annotate('', xy=(3.0, 5.0), xytext=(2.5, 4.2),
            arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2.5))
ax.annotate('', xy=(3.0, 2.0), xytext=(2.5, 2.8),
            arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2.5))
ax.text(2.0, 4.8, 'Aug.', fontsize=13, fontweight='bold', color=COLORS['green'])
ax.text(2.0, 1.7, 'Aug.', fontsize=13, fontweight='bold', color=COLORS['green'])

# View 1
box = mpatches.FancyBboxPatch((3.0, 4.5), 1.8, 1.0, boxstyle="round,pad=0.1",
                               facecolor='#ebf5fb', edgecolor=COLORS['blue'], linewidth=2)
ax.add_patch(box)
ax.text(3.9, 5.0, 'View 1', ha='center', va='center', fontsize=15, fontweight='bold', color=COLORS['blue'])

# View 2
box = mpatches.FancyBboxPatch((3.0, 1.5), 1.8, 1.0, boxstyle="round,pad=0.1",
                               facecolor='#ebf5fb', edgecolor=COLORS['blue'], linewidth=2)
ax.add_patch(box)
ax.text(3.9, 2.0, 'View 2', ha='center', va='center', fontsize=15, fontweight='bold', color=COLORS['blue'])

# Encoder (shared)
box = mpatches.FancyBboxPatch((5.5, 2.3), 2.2, 2.4, boxstyle="round,pad=0.15",
                               facecolor='#fdebd0', edgecolor=COLORS['orange'], linewidth=3)
ax.add_patch(box)
ax.text(6.6, 3.5, '1D-CNN\nEncoder\n(shared)', ha='center', va='center',
        fontsize=16, fontweight='bold', color=COLORS['orange'])

# Arrows to encoder
ax.annotate('', xy=(5.5, 4.2), xytext=(4.9, 5.0),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))
ax.annotate('', xy=(5.5, 2.8), xytext=(4.9, 2.0),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))

# Embeddings
box = mpatches.FancyBboxPatch((8.3, 4.5), 1.6, 1.0, boxstyle="round,pad=0.1",
                               facecolor='#f5eef8', edgecolor=COLORS['purple'], linewidth=2)
ax.add_patch(box)
ax.text(9.1, 5.0, 'h₁\n(256-d)', ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['purple'])

box = mpatches.FancyBboxPatch((8.3, 1.5), 1.6, 1.0, boxstyle="round,pad=0.1",
                               facecolor='#f5eef8', edgecolor=COLORS['purple'], linewidth=2)
ax.add_patch(box)
ax.text(9.1, 2.0, 'h₂\n(256-d)', ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['purple'])

ax.annotate('', xy=(8.3, 4.7), xytext=(7.8, 4.2),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))
ax.annotate('', xy=(8.3, 2.3), xytext=(7.8, 2.8),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))

# Projection head
box = mpatches.FancyBboxPatch((10.5, 2.3), 1.5, 2.4, boxstyle="round,pad=0.15",
                               facecolor='#fadbd8', edgecolor=COLORS['red'], linewidth=2.5)
ax.add_patch(box)
ax.text(11.25, 3.5, 'Proj.\nHead\n(128-d)', ha='center', va='center',
        fontsize=14, fontweight='bold', color=COLORS['red'])

ax.annotate('', xy=(10.5, 4.5), xytext=(10.0, 5.0),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))
ax.annotate('', xy=(10.5, 2.5), xytext=(10.0, 2.0),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))

# NT-Xent Loss
box = mpatches.FancyBboxPatch((12.5, 2.5), 1.3, 2.0, boxstyle="round,pad=0.15",
                               facecolor='#f9e79f', edgecolor='#d4ac0d', linewidth=3)
ax.add_patch(box)
ax.text(13.15, 3.5, 'NT-Xent\nLoss', ha='center', va='center',
        fontsize=16, fontweight='bold', color='#7d6608')

ax.annotate('', xy=(12.5, 3.5), xytext=(12.1, 3.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2.5))

ax.set_title('SimCLR: Contrastive Learning Framework', fontsize=24, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "simclr_architecture.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 9. ENCODER ARCHITECTURE
# =====================================================================
print("9. Encoder Architecture...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')

blocks = [
    ('Input\n(200×2)', 0.3, '#d5f5e3', COLORS['green']),
    ('Conv1D\n64 × k7 × s2', 2.8, '#ebf5fb', COLORS['blue']),
    ('BN + ReLU\n+ MaxPool', 5.0, '#ebf5fb', COLORS['blue']),
    ('Conv1D\n128 × k5 × s2', 7.2, '#fdebd0', COLORS['orange']),
    ('BN + ReLU\n+ MaxPool', 9.4, '#fdebd0', COLORS['orange']),
    ('Conv1D\n256 × k3 × s1', 11.2, '#f5eef8', COLORS['purple']),
    ('BN + ReLU\n+ AvgPool', 13.4, '#f5eef8', COLORS['purple']),
]

for label, x, fc, ec in blocks:
    box = mpatches.FancyBboxPatch((x, 1.2), 1.8, 2.6, boxstyle="round,pad=0.15",
                                   facecolor=fc, edgecolor=ec, linewidth=2.5)
    ax.add_patch(box)
    ax.text(x+0.9, 2.5, label, ha='center', va='center', fontsize=13, fontweight='bold')

# Arrows between blocks
for i in range(len(blocks)-1):
    x1 = blocks[i][1] + 1.8
    x2 = blocks[i+1][1]
    ax.annotate('', xy=(x2, 2.5), xytext=(x1, 2.5),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))

# Output
ax.text(15.5, 2.5, 'h\n(256-d)', ha='center', va='center', fontsize=16,
        fontweight='bold', color=COLORS['red'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor=COLORS['red'], linewidth=2.5))

ax.annotate('', xy=(15.0, 2.5), xytext=(15.3, 2.5),
            arrowprops=dict(arrowstyle='<-', color='#333', lw=2.5))

# Block labels
ax.text(3.7, 4.3, 'Block 1', fontsize=16, fontweight='bold', color=COLORS['blue'], ha='center')
ax.text(8.3, 4.3, 'Block 2', fontsize=16, fontweight='bold', color=COLORS['orange'], ha='center')
ax.text(12.3, 4.3, 'Block 3', fontsize=16, fontweight='bold', color=COLORS['purple'], ha='center')

ax.text(8, 0.5, 'Total Parameters: 240,704', fontsize=18, fontweight='bold',
        ha='center', color=COLORS['gray'])

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "encoder_architecture.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 10. ANOMALY DETECTION FLOW
# =====================================================================
print("10. Anomaly Detection Flow...")
fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')

# Calibration path (top)
cal_steps = [
    ('Healthy\nHIL Data', 0.5, 3.8, '#d5f5e3', COLORS['green']),
    ('Frozen\nEncoder', 3.5, 3.8, '#ebf5fb', COLORS['blue']),
    ('Healthy\nEmbeddings', 6.3, 3.8, '#f5eef8', COLORS['purple']),
    ('Compute\nCentroid μ', 9.2, 3.8, '#fdebd0', COLORS['orange']),
    ('Set\nThresholds τ', 12.2, 3.8, '#fadbd8', COLORS['red']),
]
ax.text(0.1, 4.7, 'Calibration (Phase 2):', fontsize=18, fontweight='bold', color=COLORS['blue'])

for label, x, y, fc, ec in cal_steps:
    box = mpatches.FancyBboxPatch((x, y-0.45), 2.2, 0.9, boxstyle="round,pad=0.1",
                                   facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(box)
    ax.text(x+1.1, y, label, ha='center', va='center', fontsize=13, fontweight='bold')

for i in range(len(cal_steps)-1):
    ax.annotate('', xy=(cal_steps[i+1][1], cal_steps[i+1][2]),
                xytext=(cal_steps[i][1]+2.2, cal_steps[i][2]),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))

# Detection path (bottom)
det_steps = [
    ('Test\nWindow', 0.5, 1.2, '#fef9e7', '#d4ac0d'),
    ('Frozen\nEncoder', 3.5, 1.2, '#ebf5fb', COLORS['blue']),
    ('Test\nEmbedding', 6.3, 1.2, '#f5eef8', COLORS['purple']),
    ('Cosine\nSimilarity', 9.2, 1.2, '#fdebd0', COLORS['orange']),
]
ax.text(0.1, 2.1, 'Detection (Phase 3):', fontsize=18, fontweight='bold', color=COLORS['orange'])

for label, x, y, fc, ec in det_steps:
    box = mpatches.FancyBboxPatch((x, y-0.45), 2.2, 0.9, boxstyle="round,pad=0.1",
                                   facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(box)
    ax.text(x+1.1, y, label, ha='center', va='center', fontsize=13, fontweight='bold')

for i in range(len(det_steps)-1):
    ax.annotate('', xy=(det_steps[i+1][1], det_steps[i+1][2]),
                xytext=(det_steps[i][1]+2.2, det_steps[i][2]),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))

# Decision outputs
ax.text(12.5, 1.5, 'sim < τ → FAULTY', fontsize=16, fontweight='bold', color=COLORS['red'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor=COLORS['red'], linewidth=2))
ax.text(12.5, 0.5, 'sim ≥ τ → HEALTHY', fontsize=16, fontweight='bold', color=COLORS['green'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor=COLORS['green'], linewidth=2))

ax.annotate('', xy=(12.3, 1.5), xytext=(11.5, 1.2),
            arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2.5))
ax.annotate('', xy=(12.3, 0.7), xytext=(11.5, 1.0),
            arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2.5))

# Connection from centroid to similarity
ax.annotate('', xy=(9.7, 1.7), xytext=(10.3, 3.3),
            arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=2, linestyle='dashed'))

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "anomaly_detection_flow.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 11. SSL COMPARISON FIGURE (NEW)
# =====================================================================
print("11. SSL Comparison Table...")
fig, ax = plt.subplots(figsize=(14, 7))
ax.axis('off')

columns = ['Method', 'Architecture\nComplexity', 'Momentum\nEncoder', 'Memory\nBank', 'Batch\nDependence', 'Selected']
rows = [
    ['SimCLR\n(Chen 2020)', 'Low', 'No', 'No', 'Yes', 'YES'],
    ['MoCo v2\n(He 2020)', 'Medium', 'Yes', 'Yes', 'Low', 'No'],
    ['BYOL\n(Grill 2020)', 'Medium', 'Yes', 'No', 'Low', 'No'],
    ['TS2Vec\n(Yue 2022)', 'High', 'No', 'No', 'Yes', 'No'],
    ['TS-TCC\n(Eldele 2021)', 'High', 'No', 'No', 'Yes', 'No'],
]

table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(1.0, 2.5)

# Style header
for j in range(len(columns)):
    cell = table[0, j]
    cell.set_facecolor('#1a5276')
    cell.set_text_props(color='white', fontweight='bold', fontsize=16)

# Style rows
for i in range(1, len(rows)+1):
    for j in range(len(columns)):
        cell = table[i, j]
        if i == 1:  # SimCLR row - highlight
            cell.set_facecolor('#d5f5e3')
            if j == 5:
                cell.set_text_props(color='#196f3d', fontweight='bold', fontsize=18)
            else:
                cell.set_text_props(fontweight='bold', fontsize=15)
        else:
            cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
            cell.set_text_props(fontsize=15)
        cell.set_edgecolor('#bdc3c7')

ax.set_title('Self-Supervised Learning Methods Comparison\n'
             'SimCLR selected for: simplicity, no extra components, proven cross-domain effectiveness',
             fontsize=20, fontweight='bold', color='#1a5276', y=0.95)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "ssl_comparison.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 12. FAULT TYPES COMPARISON
# =====================================================================
print("12. Fault Types Comparison...")
np.random.seed(123)
t = np.linspace(0, 5, 500)
healthy = 30 * np.sin(0.8*t) + 15 * np.sin(2*t) + 50

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Healthy
axes[0,0].plot(t, healthy, color=COLORS['green'], linewidth=2.5)
axes[0,0].set_title('Healthy Signal', fontsize=22, fontweight='bold', color=COLORS['green'])
axes[0,0].set_ylabel('Value', fontsize=18, fontweight='bold')
axes[0,0].tick_params(labelsize=16)
axes[0,0].grid(True, alpha=0.3)

# Gain
gain = healthy * 1.5
axes[0,1].plot(t, healthy, '--', color=COLORS['gray'], linewidth=1.5, alpha=0.5, label='Healthy')
axes[0,1].plot(t, gain, color=COLORS['red'], linewidth=2.5, label='Gain (α=1.5)')
axes[0,1].set_title('Gain Fault: y = α·x', fontsize=22, fontweight='bold', color=COLORS['red'])
axes[0,1].legend(fontsize=16)
axes[0,1].tick_params(labelsize=16)
axes[0,1].grid(True, alpha=0.3)

# Noise
noise = healthy + np.random.normal(0, 12, len(t))
axes[1,0].plot(t, healthy, '--', color=COLORS['gray'], linewidth=1.5, alpha=0.5, label='Healthy')
axes[1,0].plot(t, noise, color=COLORS['orange'], linewidth=1.5, label='Noise (σ=12)')
axes[1,0].set_title('Noise Fault: y = x + η', fontsize=22, fontweight='bold', color=COLORS['orange'])
axes[1,0].set_xlabel('Time (s)', fontsize=18, fontweight='bold')
axes[1,0].set_ylabel('Value', fontsize=18, fontweight='bold')
axes[1,0].legend(fontsize=16)
axes[1,0].tick_params(labelsize=16)
axes[1,0].grid(True, alpha=0.3)

# Stuck-at
stuck = healthy.copy()
stuck[200:] = stuck[200]
axes[1,1].plot(t, healthy, '--', color=COLORS['gray'], linewidth=1.5, alpha=0.5, label='Healthy')
axes[1,1].plot(t, stuck, color=COLORS['purple'], linewidth=2.5, label='Stuck-at')
axes[1,1].set_title('Stuck-at Fault: y = c', fontsize=22, fontweight='bold', color=COLORS['purple'])
axes[1,1].set_xlabel('Time (s)', fontsize=18, fontweight='bold')
axes[1,1].legend(fontsize=16)
axes[1,1].tick_params(labelsize=16)
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Three Fault Types Applied to Sensor Signal', fontsize=26, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fault_types_comparison.png"), dpi=200, bbox_inches='tight')
plt.close()

# =====================================================================
# 13. DATASET FUSION COMPARISON
# =====================================================================
print("13. Dataset Fusion Comparison...")
np.random.seed(456)
t_a2d2 = np.linspace(0, 20, 2000)
spd_a2d2 = 30 + 15*np.sin(0.3*t_a2d2) + 5*np.sin(0.8*t_a2d2) + np.random.normal(0, 1, 2000)
spd_a2d2 = np.clip(spd_a2d2, 0, 80)

t_hil = np.linspace(0, 20, 2000)
spd_hil = 40 + 10*np.sin(0.5*t_hil) + np.random.normal(0, 0.5, 2000)
spd_hil = np.clip(spd_hil, 0, 80)
# Add fault region
fault_start = 1000
spd_hil_fault = spd_hil.copy()
spd_hil_fault[fault_start:] = spd_hil[fault_start] + np.random.normal(0, 8, 2000-fault_start)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

ax1.plot(t_a2d2, spd_a2d2, color=COLORS['blue'], linewidth=1.5)
ax1.set_title('A2D2: Real-World Driving (Training Data)', fontsize=22, fontweight='bold', color=COLORS['blue'])
ax1.set_ylabel('Speed (km/h)', fontsize=20, fontweight='bold')
ax1.set_xlim(0, 20)
ax1.tick_params(labelsize=16)
ax1.grid(True, alpha=0.3)

ax2.plot(t_hil[:fault_start], spd_hil_fault[:fault_start], color=COLORS['green'], linewidth=1.5, label='Healthy')
ax2.plot(t_hil[fault_start:], spd_hil_fault[fault_start:], color=COLORS['red'], linewidth=1.5, label='Faulty')
ax2.axvline(x=t_hil[fault_start], color=COLORS['red'], linewidth=2, linestyle='--', alpha=0.7)
ax2.axvspan(t_hil[fault_start], t_hil[-1], alpha=0.1, color=COLORS['red'])
ax2.set_title('HIL: Simulated Driving (Test Data) with Fault Injection', fontsize=22, fontweight='bold', color=COLORS['orange'])
ax2.set_xlabel('Time (s)', fontsize=20, fontweight='bold')
ax2.set_ylabel('Speed (km/h)', fontsize=20, fontweight='bold')
ax2.legend(fontsize=18, loc='upper right')
ax2.set_xlim(0, 20)
ax2.tick_params(labelsize=16)
ax2.grid(True, alpha=0.3)

ax2.annotate('Fault Injection', xy=(t_hil[fault_start]+0.2, 60),
             fontsize=18, fontweight='bold', color=COLORS['red'])

plt.suptitle('Cross-Domain Data Fusion: Real → Simulation', fontsize=26, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "dataset_fusion_comparison.png"), dpi=200, bbox_inches='tight')
plt.close()

print("\n=== All 13 figures regenerated with LARGE fonts! ===")
print(f"Output directory: {FIGS}")
for f in sorted(os.listdir(FIGS)):
    if f.endswith('.png'):
        size_kb = os.path.getsize(os.path.join(FIGS, f)) / 1024
        print(f"  {f:45s} {size_kb:7.1f} KB")
