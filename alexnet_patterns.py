"""
AlexNet BS=16: two-panel figure (same layout as MobileNet version).

Task layout (19 tasks, 18 edges):
  0 conv_1 (CB)       9 relu_4    (MB)
  1 relu_1 (MB)      10 conv_5    (CB)
  2 maxpool_1 (MB)   11 relu_5    (MB)
  3 conv_2 (CB)      12 maxpool_3 (MB)
  4 relu_2 (MB)      13 adaptive  (MB)
  5 maxpool_2 (MB)   14 linear_1  (CB)
  6 conv_3 (CB)      15 relu_6    (MB)
  7 relu_3 (MB)      16 linear_2  (CB)
  8 conv_4 (CB)      17 relu_7    (MB)
                     18 linear_3  (CB)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Edges  (i -> i+1)
# ---------------------------------------------------------------------------
edges = [
    1695930600,  902205923, 4569845389, 1241642220,  648273608,
    1219495859,  610118418, 2677244388,  445630458, 1851841992,
     432255096,  242486845,   60370838,  164014578,  138818480,
     132680621,  126097335,  154820866,
]
N_TASKS = 19
assert len(edges) == N_TASKS - 1

# ---------------------------------------------------------------------------
# Task-type classification (CB = conv/linear, MB = relu/maxpool/adaptive)
# ---------------------------------------------------------------------------
task_types = [
    'CB',  # 0  conv_1
    'MB',  # 1  relu_1
    'MB',  # 2  maxpool_1
    'CB',  # 3  conv_2
    'MB',  # 4  relu_2
    'MB',  # 5  maxpool_2
    'CB',  # 6  conv_3
    'MB',  # 7  relu_3
    'CB',  # 8  conv_4
    'MB',  # 9  relu_4
    'CB',  # 10 conv_5
    'MB',  # 11 relu_5
    'MB',  # 12 maxpool_3
    'MB',  # 13 adaptive
    'CB',  # 14 linear_1
    'MB',  # 15 relu_6
    'CB',  # 16 linear_2
    'MB',  # 17 relu_7
    'CB',  # 18 linear_3
]
assert len(task_types) == N_TASKS
edge_types = [f'{task_types[i]}->{task_types[i+1]}' for i in range(len(edges))]

# Preferred display order for edge-type colors when rendering split fusion blocks
EDGE_TYPE_ORDER = ['CB->MB', 'MB->MB', 'MB->CB']

# ---------------------------------------------------------------------------
# Threshold (actual non-zero edge closest to mean of non-zeros)
# ---------------------------------------------------------------------------
nz = [v for v in edges if v > 0]
mean_v = sum(nz) / len(nz)
threshold = min(nz, key=lambda v: abs(v - mean_v))

# ---------------------------------------------------------------------------
# Fusion groups per framework  (inclusive (start, end))
# ---------------------------------------------------------------------------
# DTF:
#   conv_1 - relu_1 - maxpool_1 - conv_2 - relu_2   -> tasks 0..4    (5)
#   maxpool_2 - conv_3                              -> tasks 5..6    (2)
#   relu_3 - conv_4                                 -> tasks 7..8    (2)
#   relu_4 - conv_5                                 -> tasks 9..10   (2)
#   rest: singletons (tasks 11..18)
dtf_groups = [(0, 4), (5, 6), (7, 8), (9, 10)] + [(i, i) for i in range(11, N_TASKS)]

# TVM: 7 (conv, relu) pairs + singletons for the three maxpools, adaptive, linear_3
tvm_groups = [
    (0, 0)  ,           # placeholder; built below explicitly for clarity
]
tvm_groups = []
tvm_groups += [(0, 1)]              # conv_1 + relu_1
tvm_groups += [(2, 2)]              # maxpool_1
tvm_groups += [(3, 4)]              # conv_2 + relu_2
tvm_groups += [(5, 5)]              # maxpool_2
tvm_groups += [(6, 7)]              # conv_3 + relu_3
tvm_groups += [(8, 9)]              # conv_4 + relu_4
tvm_groups += [(10, 11)]            # conv_5 + relu_5
tvm_groups += [(12, 12), (13, 13)]  # maxpool_3, adaptive
tvm_groups += [(14, 15)]            # linear_1 + relu_6
tvm_groups += [(16, 17)]            # linear_2 + relu_7
tvm_groups += [(18, 18)]            # linear_3

# PyTorch:
#   relu_1 + maxpool_1                  -> tasks 1..2  (2)
#   relu_2 + maxpool_2                  -> tasks 4..5  (2)
#   relu_5 + maxpool_3 + adaptive       -> tasks 11..13 (3)   [interpretation]
pyt_groups = []
pyt_groups += [(0, 0)]         # conv_1
pyt_groups += [(1, 2)]         # relu_1 + maxpool_1
pyt_groups += [(3, 3)]         # conv_2
pyt_groups += [(4, 5)]         # relu_2 + maxpool_2
pyt_groups += [(6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]  # conv_3, relu_3, conv_4, relu_4, conv_5
pyt_groups += [(11, 13)]       # relu_5 + maxpool_3 + adaptive
pyt_groups += [(i, i) for i in range(14, N_TASKS)]        # linear_1, relu_6, linear_2, relu_7, linear_3

# Sanity
def audit(groups, name, expected_fusions):
    tot = sum(e - s + 1 for s, e in groups)
    fus = sum(e - s for s, e in groups)
    assert tot == N_TASKS, f'{name}: sum of sizes = {tot}'
    print(f'{name:8s}  groups={len(groups):3d}  tasks={tot}  fusions={fus}'
          f'   (expected ~{expected_fusions})')

audit(dtf_groups, 'DTF',      7)
audit(tvm_groups, 'TVM',      7)
audit(pyt_groups, 'PyTorch',  4)

# ---------------------------------------------------------------------------
# Colors  (same palette as the MobileNet figure for consistency)
# ---------------------------------------------------------------------------
edge_palette = {
    'CB->MB': '#D4537E',
    'MB->CB': '#1D9E75',
    'MB->MB': '#BA7517',
}

size_palette = {
    1:  '#B4B2A9',   # gray
}
UNFUSED_COLOR = '#B4B2A9'

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 8.8))
outer = GridSpec(2, 1, figure=fig, height_ratios=[3.5, 1.5], hspace=0.18)
ax_top = fig.add_subplot(outer[0])
inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.12)
ax_dtf = fig.add_subplot(inner[0])
ax_tvm = fig.add_subplot(inner[1])
ax_pyt = fig.add_subplot(inner[2])

# All four share the same x-range, but have their own tick systems
XLIM = (-0.5, N_TASKS - 0.5)
for ax in (ax_top, ax_dtf, ax_tvm, ax_pyt):
    ax.set_xlim(XLIM)

# ---- Top panel -----------------------------------------------------------
SCALE = 1e9   # Gb (AlexNet edges go up to 4.57 Gb)
for i, vol in enumerate(edges):
    ax_top.bar(i + 0.5, vol / SCALE,
               width=0.9,
               color=edge_palette[edge_types[i]],
               edgecolor='none', zorder=2)

ax_top.axhline(threshold / SCALE, color='#C0392B', linestyle='--',
               linewidth=1.4, zorder=3)

ax_top.set_ylabel('Comm. volume (Gb)', fontsize=19, fontweight='bold')
ax_top.set_ylim(0, max(edges) / SCALE * 1.14)
ax_top.set_axisbelow(True)
for side in ('top', 'right'):
    ax_top.spines[side].set_visible(False)

# Edge tick labels removed; keep no x ticks on the top panel.
ax_top.set_xticks([])
ax_top.tick_params(axis='x', labelbottom=False, bottom=False)
ax_top.tick_params(axis='y', labelsize=14)
ax_top.set_xlabel('Edge', fontsize=19, fontweight='bold')

top_legend = [
    Patch(facecolor=edge_palette['CB->MB'], label='CB → MB edge'),
    Patch(facecolor=edge_palette['MB->CB'], label='MB → CB edge'),
    Patch(facecolor=edge_palette['MB->MB'], label='MB → MB edge'),
    Line2D([0], [0], color='#C0392B', linestyle='--', linewidth=1.4,
           label=f'Threshold = {threshold/SCALE:,.2f} Gb'),
]
ax_top.legend(handles=top_legend, loc='upper right',
              ncol=1, frameon=False,
              prop={'size': 20, 'weight': 'bold'})

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
                    ha='center', va='center', fontsize=15,
                    color='white', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([label], fontsize=15, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

draw_fusion(ax_dtf, dtf_groups, 'DTF')
draw_fusion(ax_tvm, tvm_groups, 'TVM')
draw_fusion(ax_pyt, pyt_groups, 'PyTorch')

ax_dtf.tick_params(axis='x', labelbottom=False, bottom=False)
ax_tvm.tick_params(axis='x', labelbottom=False, bottom=False)

# Every integer task index gets a tick — 19 is small enough to show them all.
ax_pyt.set_xticks(list(range(N_TASKS)))
ax_pyt.set_xticklabels([str(i) for i in range(N_TASKS)])
ax_pyt.tick_params(axis='x', labelsize=13, length=3)
ax_pyt.set_xlabel('Task index', fontsize=19, fontweight='bold')

used_fused_edge_types = sorted(
    {
        edge_types[i]
        for groups in (dtf_groups, tvm_groups, pyt_groups)
        for start, end in groups
        if end > start
        for i in range(start, end)
    },
    key=lambda t: EDGE_TYPE_ORDER.index(t)
)

bottom_legend = [Patch(facecolor=UNFUSED_COLOR, label='unfused')]
bottom_legend.extend(
    Patch(facecolor=edge_palette[t], label=f'includes {t.replace("->", " -> ")}')
    for t in used_fused_edge_types
)
fig.legend(handles=bottom_legend,
           loc='lower center',
           ncol=len(bottom_legend),
           bbox_to_anchor=(0.5, -0.02),
           frameon=False,
           handlelength=1.6,
           handletextpad=0.6,
           columnspacing=1.6,
           prop={'size': 15, 'weight': 'bold'})

plt.tight_layout(rect=[0, 0.07, 1, 1])
out_pdf = 'alexnet_fusion_patterns.pdf'
plt.savefig(out_pdf, bbox_inches='tight')

print(f'\nThreshold = {threshold/SCALE:.3f} Gb '
      f'(mean was {mean_v/SCALE:.3f} Gb, raw={threshold} bits)')
print(f'Saved {out_pdf}')