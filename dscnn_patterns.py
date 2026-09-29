"""
DSCNN BS=32: two-panel figure (comm vol + DTF/TVM/PyTorch fusion patterns).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

edges = [
    87664013, 80752745, 65969303, 85189868, 79225528, 79426238, 80190802,
    75939973, 61959916, 76551821, 74326598, 80795442, 81444472, 76377265,
    61369526, 79052774, 74380051, 79236612, 81376714, 76352476, 62855575,
    77153240, 74772298, 80308465, 82284985, 73982563, 58705898, 18312764,
]
N_TASKS = 29
assert len(edges) == N_TASKS - 1

task_types = []
for _ in range(9):
    task_types.extend(['CB', 'MB', 'MB'])     # conv, bn, relu per block
task_types.extend(['MB', 'CB'])               # pool, linear
assert len(task_types) == N_TASKS
edge_types = [f'{task_types[i]}->{task_types[i+1]}' for i in range(len(edges))]

# Preferred display order for edge-type colors when rendering split fusion blocks
EDGE_TYPE_ORDER = ['CB->MB', 'MB->MB', 'MB->CB']

nz = [v for v in edges if v > 0]
mean_v = sum(nz) / len(nz)
threshold = min(nz, key=lambda v: abs(v - mean_v))

# DTF: conv_1..relu_1 (3), then four 6-task chains (conv_{k}..relu_{k+1}),
# starting at tasks 3, 9, 15, 21
dtf_groups = [(0, 2), (3, 8), (9, 14), (15, 20), (21, 26), (27, 27), (28, 28)]

# TVM: (bn_k, relu_k) pairs, convs and last two singletons
tvm_groups = []
for k in range(9):
    c, b, r = 3*k, 3*k+1, 3*k+2
    tvm_groups.append((c, c))
    tvm_groups.append((b, r))
tvm_groups.append((27, 27))
tvm_groups.append((28, 28))

# PyTorch: same as TVM except last group is bn_9 + relu_9 + pool
pyt_groups = []
for k in range(8):
    c, b, r = 3*k, 3*k+1, 3*k+2
    pyt_groups.append((c, c))
    pyt_groups.append((b, r))
pyt_groups.append((24, 24))    # conv_9
pyt_groups.append((25, 27))    # bn_9 + relu_9 + pool
pyt_groups.append((28, 28))    # linear

def audit(groups, name, expected):
    tot = sum(e - s + 1 for s, e in groups)
    fus = sum(e - s for s, e in groups)
    assert tot == N_TASKS, f'{name}: covers {tot}'
    print(f'{name:8s}  groups={len(groups):3d}  tasks={tot}  fusions={fus}'
          f'   (expected ~{expected})')

audit(dtf_groups, 'DTF',     22)
audit(tvm_groups, 'TVM',      9)
audit(pyt_groups, 'PyTorch', 10)

edge_palette = {
    'CB->MB': '#D4537E',
    'MB->CB': '#1D9E75',
    'MB->MB': '#BA7517',
}
size_palette = {
    1: '#B4B2A9',
}
UNFUSED_COLOR = '#B4B2A9'

fig = plt.figure(figsize=(13.5, 8.8))
outer = GridSpec(2, 1, figure=fig, height_ratios=[3.5, 1.5], hspace=0.45)
ax_top = fig.add_subplot(outer[0])
inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.12)
ax_dtf = fig.add_subplot(inner[0])
ax_tvm = fig.add_subplot(inner[1])
ax_pyt = fig.add_subplot(inner[2])

XLIM = (-0.5, N_TASKS - 0.5)
for ax in (ax_top, ax_dtf, ax_tvm, ax_pyt):
    ax.set_xlim(XLIM)

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
    Patch(facecolor=edge_palette['MB->MB'], label='MB → MB edge'),
    Line2D([0], [0], color='#C0392B', linestyle='--', linewidth=1.4,
           label=f'Threshold = {threshold/SCALE:,.2f} Mb'),
]
ax_top.legend(handles=top_legend, loc='upper right', fontsize=14,
              ncol=4, frameon=False, prop={'size': 13, 'weight': 'bold'})

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
out_pdf = 'dscnn_fusion_patterns.pdf'
plt.savefig(out_pdf, bbox_inches='tight')

print(f'\nThreshold = {threshold/SCALE:.2f} Mb '
      f'(mean was {mean_v/SCALE:.2f} Mb, raw={threshold} bits)')
print(f'Saved {out_pdf}')