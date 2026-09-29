"""
MobileNet BS=16: two-panel figure.

Top panel
---------
Bar chart of per-edge comm volumes. Bars are colored by edge type:
  - CB -> MB  (conv/linear -> bn/relu/pool)
  - MB -> CB  (bn/relu/pool -> conv/linear)
  - MB -> MB
Threshold = actual non-zero comm_vol closest to the mean (dashed red line).

Bottom panel
------------
Three stacked sub-panels (DTF, TVM, PyTorch), sharing x-axis = task index.
Each sub-panel shows fusion groups as blocks.  Block color encodes the
number of tasks fused into a single group (gray=1, progressively stronger
colors for larger groups).

Task layout of MobileNet-27:
  3 * k     -> conv_{k+1}      (CB)   for k = 0..26
  3 * k + 1 -> bn_{k+1}        (MB)   for k = 0..26
  3 * k + 2 -> relu_{k+1}      (MB)   for k = 0..26
  81        -> adaptive_avg_pool (MB)
  82        -> linear          (CB)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Comm-volume edges (between task i and task i+1)
# ---------------------------------------------------------------------------
edges = [
    226687188, 208500474, 224891394, 226736874, 219844716, 210353052,
    379284360, 379639260, 233309349, 118469387, 126182784, 126268451,
    216647886, 221658528, 152268644, 212470986, 220801308, 208169598,
    225833244, 223397538, 137575183,  75811172,  76530854,  77423674,
    119287459, 126384040,  97642927, 116237339, 132075107, 126924907,
    133509831, 126415489,  97900148,  34234964,  35804496,  37589425,
     74602819,  73505086,  58183180,  73659495,  73784802,  82162571,
     75306613,  76018215,  59371385,  72708691,  74189388,  81351161,
     78517858,  76548818,  58790332,  73372463,  73282810,  82136909,
     80134181,  74155864,  57722028,  77447588,  77047480,  79799210,
     75913055,  73422586,  57745124,  72833288,  73450868,  82067021,
     75862223,  75986765,  56322215,  27485695,  26087607,  30503109,
     55696696,  32086618,  22931858,  32271985,  32888965,  36072582,
     60010205,  32988829,  31783752,  18750241,
]
N_TASKS = 83
assert len(edges) == N_TASKS - 1

# ---------------------------------------------------------------------------
# Task-type classification (CB = Compute Bound, MB = Memory Bound)
# ---------------------------------------------------------------------------
task_types = []
for k in range(27):
    task_types.extend(['CB', 'MB', 'MB'])   # conv, bn, relu
task_types.extend(['MB', 'CB'])             # adaptive_avg_pool, linear
assert len(task_types) == N_TASKS

edge_types = [f'{task_types[i]}->{task_types[i+1]}' for i in range(len(edges))]

# Preferred display order for edge-type colors when rendering split fusion blocks
EDGE_TYPE_ORDER = ['CB->MB', 'MB->MB', 'MB->CB']

# ---------------------------------------------------------------------------
# Threshold: actual non-zero comm_vol closest to the mean of non-zeros
# ---------------------------------------------------------------------------
nz = [v for v in edges if v > 0]
mean_v = sum(nz) / len(nz)
threshold = min(nz, key=lambda v: abs(v - mean_v))

# ---------------------------------------------------------------------------
# Fusion groups per framework  (inclusive range tuples: (start, end))
# ---------------------------------------------------------------------------
# DTF: 3 fused groups, plus singletons for the rest
#   [0..21]  -> conv_1 .. conv_8        (22 tasks)
#   [24..26] -> conv_9 .. relu_9        (3 tasks)
#   [27..33] -> conv_10 .. conv_12      (7 tasks)
dtf_groups = (
    [(0, 21)]
    + [(22, 22), (23, 23)]        # bn_8, relu_8 unfused
    + [(24, 26)]
    + [(27, 33)]
    + [(i, i) for i in range(34, N_TASKS)]
)

# TVM: each (bn_k, relu_k) fused; everything else single
tvm_groups = []
for k in range(27):
    c, b, r = 3*k, 3*k+1, 3*k+2
    tvm_groups.append((c, c))     # conv_{k+1}
    tvm_groups.append((b, r))     # (bn_{k+1}, relu_{k+1})
tvm_groups.append((81, 81))       # adaptive
tvm_groups.append((82, 82))       # linear

# PyTorch: (bn_k, relu_k) for k=1..26 fused, (bn_27, relu_27, adaptive) triple
pyt_groups = []
for k in range(26):
    c, b, r = 3*k, 3*k+1, 3*k+2
    pyt_groups.append((c, c))
    pyt_groups.append((b, r))
pyt_groups.append((78, 78))       # conv_27 alone
pyt_groups.append((79, 81))       # bn_27, relu_27, adaptive fused
pyt_groups.append((82, 82))       # linear

# Sanity checks
def audit(groups, name, expected_fusions):
    tot = sum(e - s + 1 for s, e in groups)
    fus = sum(e - s for s, e in groups)
    assert tot == N_TASKS, f'{name}: sum of sizes = {tot}, expected {N_TASKS}'
    print(f'{name:8s}  groups={len(groups):3d}  tasks={tot}  fusions={fus}'
          f'   (expected fusions ~{expected_fusions})')

audit(dtf_groups, 'DTF', 29)
audit(tvm_groups, 'TVM', 27)
audit(pyt_groups, 'PyTorch', 28)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
edge_palette = {
    'CB->MB': '#D4537E',   # pink
    'MB->CB': '#1D9E75',   # teal
    'MB->MB': '#BA7517',   # amber
}

UNFUSED_COLOR = '#B4B2A9'

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(17, 9.5))
outer = GridSpec(2, 1, figure=fig, height_ratios=[3.5, 1.5], hspace=0.15)

ax_top = fig.add_subplot(outer[0])

inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1], hspace=0.12)
ax_dtf = fig.add_subplot(inner[0])
ax_tvm = fig.add_subplot(inner[1])
ax_pyt = fig.add_subplot(inner[2])

# All panels share the same x-range; ticks are independent.
XLIM = (-0.5, N_TASKS - 0.5)
for ax in (ax_top, ax_dtf, ax_tvm, ax_pyt):
    ax.set_xlim(XLIM)

# ---- Top panel -----------------------------------------------------------
SCALE = 1e6   # display in Mb
for i, vol in enumerate(edges):
    ax_top.bar(i + 0.5,                  # center bar between task i and task i+1
               vol / SCALE,
               width=0.9,
               color=edge_palette[edge_types[i]],
               edgecolor='none',
               zorder=2)

ax_top.axhline(threshold / SCALE, color='#C0392B', linestyle='--',
               linewidth=1.4, zorder=3)

ax_top.set_ylabel('Comm. volume (Mb)', fontsize=19, fontweight='bold')
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
           label=f'Threshold = {threshold/SCALE:,.2f} Mb'),
]
ax_top.legend(handles=top_legend, loc='upper right',
              ncol=1, frameon=False,
              prop={'size': 17, 'weight': 'bold'})

# ---- Bottom sub-panels ---------------------------------------------------
def draw_fusion(ax, groups, label):
    for start, end in groups:
        size = end - start + 1
        if size == 1:
            # Singleton task: no internal fused edge, keep a neutral color.
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
            # Draw one enclosing border so the whole multi-task span reads as one fused task.
            ax.add_patch(Rectangle((start - 0.5, 0), size, 1,
                                   facecolor='none',
                                   edgecolor='black', linewidth=2.8,
                                   zorder=6))
        # Annotate substantial fusions with their size.
        if size >= 3:
            ax.text(start + size / 2 - 0.5, 0.5, str(size),
                    ha='center', va='center', fontsize=12,
                    color='white', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([label], fontsize=16, fontweight='bold')
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)

draw_fusion(ax_dtf, dtf_groups, 'DTF')
draw_fusion(ax_tvm, tvm_groups, 'TVM')
draw_fusion(ax_pyt, pyt_groups, 'PyTorch')

ax_dtf.tick_params(axis='x', labelbottom=False, bottom=False)
ax_tvm.tick_params(axis='x', labelbottom=False, bottom=False)

# Task-index ticks on the PyTorch (bottom) panel
step = 10
ticks = list(range(0, N_TASKS, step))
if ticks[-1] != N_TASKS - 1:
    if (N_TASKS - 1) - ticks[-1] < step:
        ticks[-1] = N_TASKS - 1
    else:
        ticks.append(N_TASKS - 1)
ax_pyt.set_xticks(ticks)
ax_pyt.set_xticklabels([str(t) for t in ticks])
ax_pyt.tick_params(axis='x', labelsize=13, length=3)
ax_pyt.set_xlabel('Task index', fontsize=19, fontweight='bold')

# Bottom legend: fusion color semantics by fused edge type.
bottom_legend = [
    Patch(facecolor=UNFUSED_COLOR, label='unfused'),
    Patch(facecolor=edge_palette['CB->MB'], label='includes CB -> MB'),
    Patch(facecolor=edge_palette['MB->MB'], label='includes MB -> MB'),
    Patch(facecolor=edge_palette['MB->CB'], label='includes MB -> CB'),
]
fig.legend(handles=bottom_legend,
           loc='lower center',
           ncol=len(bottom_legend),
           bbox_to_anchor=(0.5, -0.02),
           frameon=False,
           handlelength=1.6,
           handletextpad=0.6,
           columnspacing=1.6,
           prop={'size': 18, 'weight': 'bold'})

plt.tight_layout(rect=[0, 0.08, 1, 1])
out_pdf = 'mobilenet_fusion_patterns.pdf'
plt.savefig(out_pdf, bbox_inches='tight')

print(f'\nThreshold = {threshold/SCALE:.2f} Mb (mean was {mean_v/SCALE:.2f} Mb, raw={threshold} bits)')
print(f'Saved {out_pdf}')