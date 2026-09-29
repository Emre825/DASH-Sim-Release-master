"""
SqueezeNet BS=16 two-panel figure.

Bar chart of per-edge comm volumes with a threshold line.
  Edges are colored by their role in the DAG:
    - fork edges: source has out-degree > 1 (branch-out of a squeeze ReLU)
    - join edges: destination has in-degree > 1 (branch-in to a concat)
    - other:      purely sequential edges
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Edge list (in file order) with volumes.
# ---------------------------------------------------------------------------
edge_rows = [
    (0,1,6373032120),(1,2,3204759503),(2,3,927495114),(3,4,476480004),
    (4,5,490791756),(5,6,1621445280),(4,7,1968221346),(7,8,1600522560),
    (6,9,1738027200),(8,9,1674134280),(9,10,3115770840),(10,11,468225030),
    (11,12,450081450),(12,13,1588772640),(11,14,1989907920),(14,15,1595919780),
    (13,16,1683874920),(15,16,1692054000),(16,17,15275114),(17,18,406569618),
    (18,19,264971616),(19,20,280804524),(20,21,818269452),(19,22,978232710),
    (22,23,849120090),(21,24,922746552),(23,24,919254336),(24,25,1466741640),
    (25,26,276164070),(26,27,244446384),(27,28,832723164),(26,29,975843414),
    (29,30,844196262),(28,31,930721428),(30,31,928460988),(31,32,13326986),
    (32,33,183315623),(33,34,128806205),(34,35,127537082),(35,36,330618288),
    (34,37,378229488),(37,38,330573516),(36,39,394887948),(38,39,395594472),
    (39,40,508092312),(40,41,127757339),(41,42,130679258),(42,43,325081848),
    (41,44,366711618),(44,45,328983564),(43,46,390516672),(45,46,401327472),
    (46,47,521236716),(47,48,156861978),(48,49,166323066),(49,50,430665690),
    (48,51,506863266),(51,52,450641646),(50,53,492880752),(52,53,494372424),
    (53,54,680822688),(54,55,166681788),(55,56,171511158),(56,57,441205674),
    (55,58,502088496),(58,59,457908906),(57,60,493094784),(59,60,497228004),
    (60,61,700311612),(61,62,1504672260),(62,63,1467855480),
]
N_TASKS  = 64
edges    = [(a, b) for (a, b, _) in edge_rows]
N_EDGES  = len(edges)

# Degree stats computed from the original adjacency (order-independent)
out_d, in_d = {}, {}
for s, d in edges:
    out_d[s] = out_d.get(s, 0) + 1
    in_d[d]  = in_d.get(d, 0) + 1

def edge_class(src, dst):
    if out_d[src] > 1: return 'fork'
    if in_d[dst]  > 1: return 'join'
    return 'seq'

# ---------------------------------------------------------------------------
# Reorder edges so the two fork edges of each fire module sit adjacent to
# each other, and the two join edges sit adjacent. Inside each fire-module
# block we use: [ seq, seq, seq, seq, fork, fork, join, join ].  Non-fire
# regions (stem, maxpools, tail) keep file order.
# ---------------------------------------------------------------------------
def reorder_fire_edges(rows):
    """Reorder the seven edges of each fire module into
    seq-seq-seq-seq-fork-fork-join-join layout.
    A fire module occupies a contiguous span where the set of local srcs is
    {f, f+1, f+2, f+3, f+4, f+5} and ends at a concat node d = f+6 (an index
    with in_deg=2).  In SqueezeNet the fire modules are rooted at tasks
    3, 10, 18, 25, 33, 40, 47, 54.
    """
    fire_roots = [3, 10, 18, 25, 33, 40, 47, 54]
    new = list(rows)

    # Build a map: (src, dst) -> list index in `rows`
    index_of = {(s, d): i for i, (s, d, _) in enumerate(rows)}

    for r in fire_roots:
        # Tasks within a fire module:
        #   r   conv (squeeze), r+1 relu (squeeze),
        #   r+2 conv_1x1,       r+3 relu_1x1,
        #   r+4 conv_3x3,       r+5 relu_3x3,
        #   r+6 concat
        #
        # Dataflow order: squeeze-seq -> fork (x2) -> expand-seq (x2) -> join (x2)
        squeeze_seq = [(r,     r+1)]    # squeeze conv -> squeeze relu
        fork_edges  = [(r+1,   r+2),    # relu -> 1x1 conv
                       (r+1,   r+4)]    # relu -> 3x3 conv
        expand_seq  = [(r+2,   r+3),    # 1x1 conv -> 1x1 relu
                       (r+4,   r+5)]    # 3x3 conv -> 3x3 relu
        join_edges  = [(r+3,   r+6),    # 1x1 relu -> concat
                       (r+5,   r+6)]    # 3x3 relu -> concat

        ordered = squeeze_seq + fork_edges + expand_seq + join_edges
        # pick their rows in `new` order; we'll replace the range with them.
        positions = sorted(index_of[e] for e in ordered)
        assert positions == list(range(positions[0], positions[-1] + 1)), \
            f'fire module at {r}: positions not contiguous: {positions}'
        for i, e in enumerate(ordered):
            new[positions[i]] = rows[index_of[e]]
    return new

edge_rows = reorder_fire_edges(edge_rows)
edges    = [(a, b) for (a, b, _) in edge_rows]
edge_vol = [v for (_, _, v) in edge_rows]
edge_types = [edge_class(s, d) for (s, d) in edges]

# Threshold
nz = [v for v in edge_vol if v > 0]
mean_v    = sum(nz) / len(nz)
threshold = min(nz, key=lambda v: abs(v - mean_v))

# ---------------------------------------------------------------------------
# Fusion groups — each group is a list of task IDs.  Tasks may be
# non-contiguous (for PyTorch groups that span parallel branches via concat).
# ---------------------------------------------------------------------------
# DTF (8 groups, 11 fusions)
dtf_fusions = [
    [0, 1, 2, 3],      # conv_1 + relu_1 + maxpool_1 + conv_2
    [5, 6],            # conv_3 + relu_3
    [7, 8],            # conv_4 + relu_4
    [12, 13],          # conv_6 + relu_6
    [14, 15],          # conv_7 + relu_7
    [22, 23],          # conv_10 + relu_10
    [29, 30],          # conv_13 + relu_13
    [61, 62, 63],      # conv_26 + relu_26 + adaptive
]

# TVM: 26 conv-relu pairs
# task layout: each fire module is 7 tasks — conv (squeeze), relu, conv (1x1 expand),
#              relu, conv (3x3 expand), relu, sum.
# The 26 conv-relu pairs are:
#   conv_1+relu_1  = [0, 1]
#   conv_2+relu_2  = [3, 4]
#   conv_3+relu_3  = [5, 6]
#   conv_4+relu_4  = [7, 8]
#   conv_5+relu_5  = [10, 11]
#   ...
# Layout in file: tasks 0-2 stem, 3-9 fire1, 10-16 fire2, 17 maxpool, 18-24 fire3,
# 25-31 fire4, 32 maxpool, 33-39 fire5, 40-46 fire6, 47-53 fire7, 54-60 fire8,
# 61-63 tail (conv_26, relu_26, adaptive).
# conv-relu pair IDs:
tvm_fusions = [
    [0, 1],                        # conv_1 relu_1
    [3, 4],                        # conv_2 relu_2
    [5, 6], [7, 8],                # fire1 expand
    [10, 11],                      # fire2 squeeze conv_5 relu_5
    [12, 13], [14, 15],            # fire2 expand
    [18, 19],                      # fire3 squeeze
    [20, 21], [22, 23],            # fire3 expand
    [25, 26],                      # fire4 squeeze
    [27, 28], [29, 30],            # fire4 expand
    [33, 34],                      # fire5 squeeze
    [35, 36], [37, 38],            # fire5 expand
    [40, 41],                      # fire6 squeeze
    [42, 43], [44, 45],            # fire6 expand
    [47, 48],                      # fire7 squeeze
    [49, 50], [51, 52],            # fire7 expand
    [54, 55],                      # fire8 squeeze
    [56, 57], [58, 59],            # fire8 expand
    [61, 62],                      # conv_26 relu_26
]

# PyTorch: 9 groups, each is (relu_x - relu_y - concat) triple or the tail.
# The pair (relu_x, relu_y) is the two expand ReLUs whose outputs feed the
# concat.  Those are non-contiguous in task ID.
pyt_fusions = [
    [6, 8, 9],      # relu_3 (idx 6) + relu_4 (idx 8) + sum_1 (idx 9)
    [13, 15, 16],   # relu_6 + relu_7 + sum_2
    [21, 23, 24],   # relu_9 + relu_10 + sum_3
    [28, 30, 31],   # relu_12 + relu_13 + sum_4
    [36, 38, 39],   # relu_15 + relu_16 + sum_5
    [43, 45, 46],   # relu_18 + relu_19 + sum_6
    [50, 52, 53],   # relu_21 + relu_22 + sum_7
    [57, 59, 60],   # relu_24 + relu_25 + sum_8
    [62, 63],       # relu_26 + adaptive
]

def pad_with_singletons(groups, n):
    covered = {t for g in groups for t in g}
    return [list(g) for g in groups] + [[t] for t in range(n) if t not in covered]

dtf_groups = pad_with_singletons(dtf_fusions, N_TASKS)
tvm_groups = pad_with_singletons(tvm_fusions, N_TASKS)
pyt_groups = pad_with_singletons(pyt_fusions, N_TASKS)

def audit(groups, name, expected):
    tot = sum(len(g) for g in groups)
    fus = sum(len(g) - 1 for g in groups)
    assert tot == N_TASKS
    print(f'{name:8s}  groups={len(groups):3d}  tasks={tot}  '
          f'fusions={fus}   (expected ~{expected})')

audit(dtf_groups, 'DTF',     11)
audit(tvm_groups, 'TVM',     26)
audit(pyt_groups, 'PyTorch', 17)

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
edge_palette = {
    'seq':  '#378ADD',    # strong blue  (cool)
    'fork': '#E24B4A',    # bold red     (warm)
    'join': '#27500A',    # dark green   (deep, distinct from both)
}
size_palette = {
    1: '#B4B2A9',
    2: '#85B7EB',
    3: '#185FA5',
    4: '#7F77DD',
}
sizes_in_use = sorted({len(g) for grp in (dtf_groups, tvm_groups, pyt_groups)
                       for g in grp})

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax_top = plt.subplots(figsize=(17, 5.2))

SCALE = 1e9     # Gb (values up to ~6.4 Gb)
for i, vol in enumerate(edge_vol):
    ax_top.bar(i + 0.5, vol / SCALE, width=0.9,
               color=edge_palette[edge_types[i]],
               edgecolor='none', zorder=2)

ax_top.axhline(threshold / SCALE, color='#C0392B', linestyle='--',
               linewidth=1.4, zorder=3)

ax_top.set_xlim(-0.5, N_EDGES + 0.5)
ax_top.set_ylim(0, max(edge_vol) / SCALE * 1.14)
ax_top.set_ylabel('Comm. volume (Gb)', fontsize=16, fontweight='bold')
ax_top.set_axisbelow(True)
for side in ('top', 'right'):
    ax_top.spines[side].set_visible(False)

# Remove edge tick labels entirely
ax_top.set_xticks([])
ax_top.tick_params(axis='y', labelsize=11)
ax_top.set_xlabel('Edge', fontsize=16, fontweight='bold')

top_legend = [
    Patch(facecolor=edge_palette['seq'],  label='Sequential edge'),
    Patch(facecolor=edge_palette['fork'], label='Fork edge'),
    Patch(facecolor=edge_palette['join'], label='Join edge'),
    Line2D([0], [0], color='#C0392B', linestyle='--', linewidth=1.4,
           label=f'Threshold = {threshold/SCALE:,.2f} Gb'),
]
ax_top.legend(handles=top_legend, loc='upper right',
              ncol=4, frameon=False,
              prop={'size': 16, 'weight': 'bold'})

plt.tight_layout()
out_pdf = 'squeezenet_barchart.pdf'
out_png = 'squeezenet_barchart.png'
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, bbox_inches='tight', dpi=300)

print(f'\nThreshold = {threshold/SCALE:.3f} Gb '
      f'(mean was {mean_v/SCALE:.3f} Gb, raw={threshold} bits)')
print(f'Saved {out_pdf} and {out_png}')