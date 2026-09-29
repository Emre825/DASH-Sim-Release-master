"""
VGG BS=16: two-panel figure.

Linear chain of 37 tasks. Task layout:
  Feature extractor (indices 0..30):
    13 conv_k (CB), 13 relu_k (MB), 5 maxpool (MB)
    - 5 stages: (conv, relu) x 2or3 then maxpool
  Classifier head (indices 31..36):
    adaptive (MB), linear_1 (CB), relu_14 (MB), linear_2 (CB), relu_15 (MB), linear_3 (CB)

Edge counts by type:
  CB -> MB : every conv->relu  +  each linear->relu      = 13 + 2       = 15
  MB -> CB : relu->conv where a conv follows  +  adaptive->linear_1  + relu_14->linear_2  + relu_15->linear_3
  MB -> MB : relu->maxpool  +  maxpool_5->adaptive       = 5 + 1        = 6
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Edges (in file order)
# ---------------------------------------------------------------------------
edges = [
    25122442800, 124570773600,  25346575800, 12784255739, 28615393661,
    12874243200,  63056502600,  12818715000,  6423363547, 14327116003,
     6525284220,  31665723180,   6435046800, 31593700320,  6448494780,
     3212437246,   7086228513,   3267580680, 15935556000,  3286947300,
    15843811620,   3270954960,   1634893260,  3685103276,   887330262,
     4381019370,    894129600,   4225526214,   883828764,   470299448,
      129633941,    291662280,    121945496,   108739886,   105966315,
      101755727,
]
N_TASKS = 37
assert len(edges) == N_TASKS - 1

# ---------------------------------------------------------------------------
# Task-type classification (CB = conv/linear, MB = relu/maxpool/adaptive)
# ---------------------------------------------------------------------------
task_types = [
    'CB',  # 0  conv_1
    'MB',  # 1  relu_1
    'CB',  # 2  conv_2
    'MB',  # 3  relu_2
    'MB',  # 4  maxpool_1
    'CB',  # 5  conv_3
    'MB',  # 6  relu_3
    'CB',  # 7  conv_4
    'MB',  # 8  relu_4
    'MB',  # 9  maxpool_2
    'CB',  # 10 conv_5
    'MB',  # 11 relu_5
    'CB',  # 12 conv_6
    'MB',  # 13 relu_6
    'CB',  # 14 conv_7
    'MB',  # 15 relu_7
    'MB',  # 16 maxpool_3
    'CB',  # 17 conv_8
    'MB',  # 18 relu_8
    'CB',  # 19 conv_9
    'MB',  # 20 relu_9
    'CB',  # 21 conv_10
    'MB',  # 22 relu_10
    'MB',  # 23 maxpool_4
    'CB',  # 24 conv_11
    'MB',  # 25 relu_11
    'CB',  # 26 conv_12
    'MB',  # 27 relu_12
    'CB',  # 28 conv_13
    'MB',  # 29 relu_13
    'MB',  # 30 maxpool_5
    'MB',  # 31 adaptive
    'CB',  # 32 linear_1
    'MB',  # 33 relu_14
    'CB',  # 34 linear_2
    'MB',  # 35 relu_15
    'CB',  # 36 linear_3
]
assert len(task_types) == N_TASKS
edge_types = [f'{task_types[i]}->{task_types[i+1]}' for i in range(len(edges))]

# Preferred display order for edge-type colors when rendering split fusion blocks
EDGE_TYPE_ORDER = ['CB->MB', 'MB->MB', 'MB->CB']

# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------
nz = [v for v in edges if v > 0]
mean_v    = sum(nz) / len(nz)
threshold = min(nz, key=lambda v: abs(v - mean_v))

# ---------------------------------------------------------------------------
# Fusion groups
# ---------------------------------------------------------------------------
# DTF
#   conv_1 - relu_1 - conv_2 - relu_2       -> tasks 0..3     (4)
#   maxpool_1 - conv_3 - relu_3 - conv_4    -> tasks 4..7     (4)
#   maxpool_2 - conv_5                      -> tasks 9..10    (2)
#   relu_5 - conv_6                         -> tasks 11..12   (2)
#   relu_6 - conv_7                         -> tasks 13..14   (2)
#   relu_8 - conv_9                         -> tasks 18..19   (2)
#   relu_9 - conv_10                        -> tasks 20..21   (2)
dtf_groups = [(0, 3), (4, 7),
              (8, 8),                       # relu_4 singleton between groups
              (9, 10), (11, 12), (13, 14),
              (15, 15), (16, 16), (17, 17), # relu_7, maxpool_3, conv_8
              (18, 19), (20, 21)] \
             + [(i, i) for i in range(22, N_TASKS)]

# TVM: (conv_k, relu_k) for k=1..13, plus (linear_1, relu_14), (linear_2, relu_15)
tvm_groups = []
tvm_groups += [(0, 1)]       # conv_1 + relu_1
tvm_groups += [(2, 3)]       # conv_2 + relu_2
tvm_groups += [(4, 4)]       # maxpool_1
tvm_groups += [(5, 6)]       # conv_3 + relu_3
tvm_groups += [(7, 8)]       # conv_4 + relu_4
tvm_groups += [(9, 9)]       # maxpool_2
tvm_groups += [(10, 11)]     # conv_5 + relu_5
tvm_groups += [(12, 13)]     # conv_6 + relu_6
tvm_groups += [(14, 15)]     # conv_7 + relu_7
tvm_groups += [(16, 16)]     # maxpool_3
tvm_groups += [(17, 18)]     # conv_8 + relu_8
tvm_groups += [(19, 20)]     # conv_9 + relu_9
tvm_groups += [(21, 22)]     # conv_10 + relu_10
tvm_groups += [(23, 23)]     # maxpool_4
tvm_groups += [(24, 25)]     # conv_11 + relu_11
tvm_groups += [(26, 27)]     # conv_12 + relu_12
tvm_groups += [(28, 29)]     # conv_13 + relu_13
tvm_groups += [(30, 30)]     # maxpool_5
tvm_groups += [(31, 31)]     # adaptive
tvm_groups += [(32, 33)]     # linear_1 + relu_14
tvm_groups += [(34, 35)]     # linear_2 + relu_15
tvm_groups += [(36, 36)]     # linear_3

# PyTorch: only maxpool_5 + adaptive fused; everything else singleton
pyt_groups = (
    [(i, i) for i in range(0, 30)]
    + [(30, 31)]                    # maxpool_5 + adaptive
    + [(i, i) for i in range(32, N_TASKS)]
)

def audit(groups, name, expected_fusions):
    tot = sum(e - s + 1 for s, e in groups)
    fus = sum(e - s for s, e in groups)
    assert tot == N_TASKS, f'{name}: covers {tot}'
    print(f'{name:8s}  groups={len(groups):3d}  tasks={tot}  '
          f'fusions={fus}   (expected ~{expected_fusions})')

audit(dtf_groups, 'DTF',     11)
audit(tvm_groups, 'TVM',     15)
audit(pyt_groups, 'PyTorch',  1)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
edge_palette = {
    'CB->MB': '#D4537E',
    'MB->CB': '#1D9E75',
    'MB->MB': '#BA7517',
}

size_palette = {
    1: '#B4B2A9',
}
UNFUSED_COLOR = '#B4B2A9'

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15.5, 9.2))
outer = GridSpec(2, 1, figure=fig, height_ratios=[3.5, 1.5], hspace=0.45)
ax_top = fig.add_subplot(outer[0])
inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.12)
ax_dtf = fig.add_subplot(inner[0])
ax_tvm = fig.add_subplot(inner[1])
ax_pyt = fig.add_subplot(inner[2])

XLIM = (-0.5, N_TASKS - 0.5)
for ax in (ax_top, ax_dtf, ax_tvm, ax_pyt):
    ax.set_xlim(XLIM)

# ---- Top panel -----------------------------------------------------------
SCALE = 1e9   # Gb (values up to 124.6 Gb)
for i, vol in enumerate(edges):
    ax_top.bar(i + 0.5, vol / SCALE,
               width=0.9,
               color=edge_palette[edge_types[i]],
               edgecolor='none', zorder=2)

ax_top.axhline(threshold / SCALE, color='#C0392B', linestyle='--',
               linewidth=1.4, zorder=3)

ax_top.set_ylabel('Comm. volume (Gb, log scale)', fontsize=14, fontweight='bold')
ax_top.set_yscale('log')
# Log limits: min non-zero edge / 2  to  max * 1.5
ax_top.set_ylim(min(nz) / SCALE / 2, max(edges) / SCALE * 1.5)
ax_top.set_axisbelow(True)
for side in ('top', 'right'):
    ax_top.spines[side].set_visible(False)

# Label every edge — 36 of them. Rotated 45° to fit.
edge_tick_positions = [i + 0.5 for i in range(len(edges))]
edge_tick_labels    = [f'{i} to {i+1}' for i in range(len(edges))]
ax_top.set_xticks(edge_tick_positions)
ax_top.set_xticklabels(edge_tick_labels,
                       rotation=45, ha='right', rotation_mode='anchor')
ax_top.tick_params(axis='x', labelbottom=True, bottom=True,
                   length=3, labelsize=11)
ax_top.tick_params(axis='y', which='major', labelsize=12.5,
                   length=6, width=1.2, direction='out')
ax_top.tick_params(axis='y', which='minor',
                   length=3.5, width=1.0, direction='out')
ax_top.set_xlabel('Edge', fontsize=14, fontweight='bold')

top_legend = [
    Patch(facecolor=edge_palette['CB->MB'], label='CB → MB edge'),
    Patch(facecolor=edge_palette['MB->CB'], label='MB → CB edge'),
    Patch(facecolor=edge_palette['MB->MB'], label='MB → MB edge'),
    Line2D([0], [0], color='#C0392B', linestyle='--', linewidth=1.4,
           label=f'Threshold = {threshold/SCALE:,.2f} Gb'),
]
ax_top.legend(handles=top_legend, loc='upper right', fontsize=14,
              ncol=4, frameon=False, prop={'size': 14, 'weight': 'bold'})

# ---- Fusion sub-panels ---------------------------------------------------
def draw_fusion(ax, groups, label):
    for start, end in groups:
        size = end - start + 1
        if size == 1:
            ax.add_patch(Rectangle((start - 0.5, 0), size, 1,
                                   facecolor=UNFUSED_COLOR,
                                   edgecolor='white', linewidth=0.6))
        else:
            fused_edge_types = sorted({edge_types[i] for i in range(start, end)},
                                      key=lambda t: EDGE_TYPE_ORDER.index(t))
            seg_w = size / len(fused_edge_types)
            for j, et in enumerate(fused_edge_types):
                ax.add_patch(Rectangle((start - 0.5 + j * seg_w, 0), seg_w, 1,
                                       facecolor=edge_palette[et],
                                       edgecolor='none', linewidth=0))
            ax.add_patch(Rectangle((start - 0.5, 0), size, 1,
                                   facecolor='none', edgecolor='black',
                                   linewidth=2.8, zorder=6))
        if size >= 3:
            ax.text(start + size / 2 - 0.5, 0.5, str(size),
                    ha='center', va='center', fontsize=13,
                    color='white', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([label], fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

draw_fusion(ax_dtf, dtf_groups, 'DTF')
draw_fusion(ax_tvm, tvm_groups, 'TVM')
draw_fusion(ax_pyt, pyt_groups, 'PyTorch')

ax_dtf.tick_params(axis='x', labelbottom=False, bottom=False)
ax_tvm.tick_params(axis='x', labelbottom=False, bottom=False)

ax_pyt.set_xticks(list(range(N_TASKS)))
ax_pyt.set_xticklabels([str(i) for i in range(N_TASKS)])
ax_pyt.tick_params(axis='x', labelsize=11, length=3)
ax_pyt.set_xlabel('Task index', fontsize=14, fontweight='bold')

bottom_legend = [
    Patch(facecolor=UNFUSED_COLOR, label='1 task (unfused)'),
    Patch(facecolor=edge_palette['CB->MB'], label='Fused task includes CB -> MB'),
    Patch(facecolor=edge_palette['MB->MB'], label='Fused task includes MB -> MB'),
    Patch(facecolor=edge_palette['MB->CB'], label='Fused task includes MB -> CB'),
]
fig.legend(handles=bottom_legend,
           loc='lower center',
           ncol=len(bottom_legend),
           bbox_to_anchor=(0.5, -0.02),
           fontsize=12,
           frameon=False,
           handlelength=1.4,
           handletextpad=0.5,
           columnspacing=1.0,
           prop={'size': 12, 'weight': 'bold'})

plt.tight_layout(rect=[0, 0.07, 1, 1])
out_pdf = 'vgg_fusion_patterns.pdf'
plt.savefig(out_pdf, bbox_inches='tight')

print(f'\nThreshold = {threshold/SCALE:.3f} Gb  '
      f'(mean was {mean_v/SCALE:.3f} Gb, raw={threshold} bits)')
print(f'Saved {out_pdf}')