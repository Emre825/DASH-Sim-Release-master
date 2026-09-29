"""
AutoEncoder BS=16: two-panel figure.

The model is a pure linear chain of alternating linear/ReLU layers, which
means every edge is either CB→MB or MB→CB — there are no MB→MB edges at
all.  This is why PyTorch's fusion pass emits no fusions: PyTorch only
fuses MB→MB edges.

Task layout (19 tasks, 18 edges):
  0  linear_1  (CB)      9  relu_5   (MB)
  1  relu_1    (MB)     10  linear_6 (CB)
  2  linear_2  (CB)     11  relu_6   (MB)
  3  relu_2    (MB)     12  linear_7 (CB)
  4  linear_3  (CB)     13  relu_7   (MB)
  5  relu_3    (MB)     14  linear_8 (CB)
  6  linear_4  (CB)     15  relu_8   (MB)
  7  relu_4    (MB)     16  linear_9 (CB)
  8  linear_5  (CB)     17  relu_9   (MB)
                        18  linear_10(CB)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Edges (all sequential, 18 of them)
# ---------------------------------------------------------------------------
edges = [
    22595336, 24443110, 19993373, 24352146, 20094875, 24041636,
    20275710, 18272414, 19737081, 17301987, 20455344, 23643438,
    20025915, 23576935, 20060313, 23651683, 20225587, 23167162,
]
N_TASKS = 19
assert len(edges) == N_TASKS - 1

# ---------------------------------------------------------------------------
# Task-type classification — perfectly alternating CB/MB
# ---------------------------------------------------------------------------
# Even indices = linear (CB), odd indices = relu (MB)
task_types = ['CB' if i % 2 == 0 else 'MB' for i in range(N_TASKS)]

edge_types = [f'{task_types[i]}->{task_types[i+1]}' for i in range(len(edges))]
# Confirm no MB→MB edges (sanity check of the user's observation)
assert 'MB->MB' not in edge_types, 'Model should have no MB->MB edges'

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
# DTF:
#   linear_1 + relu_1 + linear_2   -> tasks 0..2   (3)
#   relu_2  + linear_3             -> tasks 3..4   (2)
#   relu_3  + linear_4             -> tasks 5..6   (2)
#   [tasks 7, 8, 9 unfused]
#   linear_6 + relu_6 + linear_7   -> tasks 10..12 (3)
#   relu_7  + linear_8             -> tasks 13..14 (2)
#   relu_8  + linear_9             -> tasks 15..16 (2)
#   relu_9  + linear_10            -> tasks 17..18 (2)
dtf_groups = [
    (0, 2), (3, 4), (5, 6),
    (7, 7), (8, 8), (9, 9),
    (10, 12), (13, 14), (15, 16), (17, 18),
]

# TVM: 9 (linear_k, relu_k) pairs; task 18 (linear_10) left alone
tvm_groups = []
for k in range(9):
    tvm_groups.append((2 * k, 2 * k + 1))     # linear_{k+1} + relu_{k+1}
tvm_groups.append((18, 18))                   # linear_10 unfused

# PyTorch: nothing fused
pyt_groups = [(i, i) for i in range(N_TASKS)]

def audit(groups, name, expected_fusions):
    tot = sum(e - s + 1 for s, e in groups)
    fus = sum(e - s for s, e in groups)
    assert tot == N_TASKS
    print(f'{name:8s}  groups={len(groups):3d}  tasks={tot}  '
          f'fusions={fus}   (expected ~{expected_fusions})')

audit(dtf_groups, 'DTF',     9)
audit(tvm_groups, 'TVM',     9)
audit(pyt_groups, 'PyTorch', 0)

# ---------------------------------------------------------------------------
# Colors (dropped the MB→MB entry from the top palette since none exist)
# ---------------------------------------------------------------------------
edge_palette = {
    'CB->MB': '#D4537E',
    'MB->CB': '#1D9E75',
}

UNFUSED_COLOR = '#B4B2A9'

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 8.8))
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
SCALE = 1e6
for i, vol in enumerate(edges):
    ax_top.bar(i + 0.5, vol / SCALE,
               width=0.9,
               color=edge_palette[edge_types[i]],
               edgecolor='none', zorder=2)

ax_top.axhline(threshold / SCALE, color='#C0392B', linestyle='--',
               linewidth=1.4, zorder=3)

ax_top.set_ylabel('Comm. volume (Mb)', fontsize=14, fontweight='bold')
ax_top.set_ylim(0, max(edges) / SCALE * 1.14)
ax_top.set_axisbelow(True)
for side in ('top', 'right'):
    ax_top.spines[side].set_visible(False)

# All 18 edges get a label — the figure is wide enough to fit them.
edge_tick_positions = [i + 0.5 for i in range(len(edges))]
edge_tick_labels    = [f'{i} to {i+1}' for i in range(len(edges))]
ax_top.set_xticks(edge_tick_positions)
ax_top.set_xticklabels(edge_tick_labels,
                       rotation=45, ha='right', rotation_mode='anchor')
ax_top.tick_params(axis='x', labelbottom=True, bottom=True, length=3, labelsize=11)
ax_top.tick_params(axis='y', labelsize=11)
ax_top.set_xlabel('Edge', fontsize=14, fontweight='bold')

top_legend = [
    Patch(facecolor=edge_palette['CB->MB'], label='CB → MB edge'),
    Patch(facecolor=edge_palette['MB->CB'], label='MB → CB edge'),
    Line2D([0], [0], color='#C0392B', linestyle='--', linewidth=1.4,
           label=f'Threshold = {threshold/SCALE:,.2f} Mb'),
]
ax_top.legend(handles=top_legend, loc='upper right', fontsize=14,
              ncol=3, frameon=False, prop={'size': 14, 'weight': 'bold'})

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

bottom_legend = [Patch(facecolor=UNFUSED_COLOR, label='1 task (unfused)')]
bottom_legend.extend(
    Patch(facecolor=edge_palette[t], label=f'Fused task includes {t.replace("->", " -> ")}')
    for t in used_fused_edge_types
)
fig.legend(handles=bottom_legend,
           loc='lower center',
           ncol=len(bottom_legend),
           bbox_to_anchor=(0.5, -0.04),
           fontsize=14,
           frameon=False,
           prop={'size': 15, 'weight': 'bold'})

plt.tight_layout(rect=[0, 0.02, 1, 1])
out_pdf = 'autoencoder_fusion_patterns.pdf'
plt.savefig(out_pdf, bbox_inches='tight')

print(f'\nThreshold = {threshold/SCALE:.2f} Mb  '
      f'(mean was {mean_v/SCALE:.2f} Mb, raw={threshold} bits)')
print(f'Saved {out_pdf}')