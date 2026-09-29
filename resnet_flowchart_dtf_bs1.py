"""
ResNet BS=1 — horizontal DAG flowchart.

Node placement:
- Main chain is laid out left-to-right by task index, but residual blocks
  claim two parallel y-lanes so that branch tasks (conv_6, bn_6, conv_9,
  bn_9) sit on a secondary track while the main path continues on y=0.
- Summation nodes sit at the rejoin point on the main track.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# 30 tasks — (id, short_label)
# ---------------------------------------------------------------------------
labels = {
     0: 'conv₁',  1: 'bn₁',    2: 'relu₁',
     3: 'conv₂',  4: 'bn₂',    5: 'relu₂',
     6: 'conv₃',  7: 'bn₃',    8: 'Σ₁',    9: 'relu₃',
    10: 'conv₄', 11: 'bn₄',   12: 'relu₄',
    13: 'conv₅', 14: 'bn₅',
    15: 'conv₆', 16: 'bn₆',
    17: 'Σ₂',   18: 'relu₅',
    19: 'conv₇', 20: 'bn₇',   21: 'relu₆',
    22: 'conv₈', 23: 'bn₈',
    24: 'conv₉', 25: 'bn₉',
    26: 'Σ₃',   27: 'relu₇',
    28: 'pool', 29: 'linear',
}

edges = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(2,8),
    (8,9),(9,10),(10,11),(11,12),(12,13),(13,14),
    (9,15),(15,16),(16,17),(14,17),
    (17,18),(18,19),(19,20),(20,21),(21,22),(22,23),
    (18,24),(24,25),(25,26),(23,26),
    (26,27),(27,28),(28,29),
]

# ---------------------------------------------------------------------------
# Layout: assign (x, y) to each node.
# Main path gets y=0; branch tasks get y = +1.2 (above).
# x = horizontal column (shared by tasks that should stack vertically).
#
# Strategy: walk task IDs in order, increment x each step, but give branch
# tasks (15, 16 and 24, 25) the same x as their main-path peers (13, 14 and
# 22, 23 respectively) and put them on the upper lane.
# ---------------------------------------------------------------------------
branch = {15, 16, 24, 25}

# Manually place x so branch columns align with main-path columns
pos = {}
col = 0

def place(task, c, lane=0):
    pos[task] = (c, lane)

# Pre-block 1 stem: 0,1,2
for i, t in enumerate([0, 1, 2]):
    place(t, col); col += 1

# Residual block 1 (identity shortcut): 3,4,5,6,7,8 with skip 2->8
for t in [3, 4, 5, 6, 7, 8]:
    place(t, col); col += 1

# Between blocks: 9
place(9, col); col += 1

# Block 2: main path 10..14; branch 15,16; rejoin at 17; 18 after
# columns: 10  11  12  13  14  17  18
# branch       .   .  15  16
# We need col(15)==col(13) and col(16)==col(14).
c_main_start = col
for t in [10, 11, 12, 13, 14]:
    place(t, col); col += 1
# branch 15,16 share columns with 13,14
place(15, c_main_start + 3, lane=1)   # same col as task 13
place(16, c_main_start + 4, lane=1)   # same col as task 14
# summation 17 sits to the right of col of 14
place(17, col); col += 1
place(18, col); col += 1

# Block 3: main 19..23; branch 24,25; rejoin at 26; 27 after
c_main_start = col
for t in [19, 20, 21, 22, 23]:
    place(t, col); col += 1
place(24, c_main_start + 3, lane=1)   # aligned with conv_8 column
place(25, c_main_start + 4, lane=1)   # aligned with bn_8 column
place(26, col); col += 1
place(27, col); col += 1

# Tail: 28, 29
place(28, col); col += 1
place(29, col); col += 1

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
cols = col
fig_w = max(12, cols * 0.5)
fig_h = 3.7
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# Node dimensions (data coords) — smaller now that we dropped the task-type label
NODE_W, NODE_H = 0.46, 0.46

# Single soft color for all nodes
NODE_FACE = '#E6F1FB'   # blue 50
NODE_EDGE = '#6B8EB0'
ID_COLOR  = '#123'

# Edge styling: three categories.
#   fusible       — eligible by volume AND allowed by topology
#   blocked       — eligible by volume but src.out>1 or dst.in>1
#   sub-threshold — below the volume threshold
col_edge_sub     = '#CCCCCC'   # soft gray
col_edge_fusible = '#1D9E75'   # teal green
col_edge_blocked = '#C0392B'   # solid red
edge_lw_sub      = 0.9
edge_lw_strong   = 2.4
arrow_size       = 12

# ------- threshold computation ---------
edge_vol = {
    (0,1):24969781,(1,2):24776279,(2,3):53322032,(3,4):49258700,(4,5):24765413,
    (5,6):53235055,(6,7):48981660,(7,8):28459158,(2,8):28188888,(8,9):20921410,
    (9,10):27198553,(10,11):20613302,(11,12):20611664,(12,13):38952950,
    (13,14):19553570,(9,15):18872048,(15,16):19912729,(16,17):26289572,
    (14,17):26372018,(17,18):14924692,(18,19):24391513,(19,20):17684066,
    (20,21):18568150,(21,22):23939042,(22,23):16640770,(18,24):17869488,
    (24,25):18468286,(25,26):25519603,(23,26):25515781,(26,27):11822647,
    (27,28):17284214,(28,29):6455080,
}
_vals = list(edge_vol.values())
_mean = sum(_vals) / len(_vals)
THRESHOLD = min(_vals, key=lambda v: abs(v - _mean))

# Topology-based blocking
_out_deg, _in_deg = {}, {}
for s, d in edges:
    _out_deg[s] = _out_deg.get(s, 0) + 1
    _in_deg[d]  = _in_deg.get(d, 0) + 1

def edge_category(src, dst):
    v = edge_vol[(src, dst)]
    if v < THRESHOLD:
        return 'sub'
    if _out_deg[src] > 1 or _in_deg[dst] > 1:
        return 'blocked'
    return 'fusible'

# Horizontal lane spacing (y = 0 for main, y = 1.2 for branch)
LANE = 1.2

# Draw edges first so nodes sit on top
def anchor(task, side):
    x, lane = pos[task]
    y = lane * LANE
    if side == 'l': return (x - NODE_W/2, y)
    if side == 'r': return (x + NODE_W/2, y)
    if side == 't': return (x, y + NODE_H/2)
    if side == 'b': return (x, y - NODE_H/2)

# Helper: draw a polyline ending in an arrowhead at the final segment.
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
import matplotlib.patches as mpatches

def draw_path(points, color, lw, linestyle='solid', z_line=1, z_arrow=2):
    """Draw polyline through `points` with an arrowhead on the final segment."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, linewidth=lw,
                linestyle=linestyle,
                solid_capstyle='butt', solid_joinstyle='miter', zorder=z_line)
    arrow = FancyArrowPatch(points[-2], points[-1],
                            arrowstyle='-|>', mutation_scale=arrow_size,
                            color=color, linewidth=lw, zorder=z_arrow,
                            linestyle=linestyle,
                            shrinkA=0, shrinkB=0)
    ax.add_patch(arrow)

for src, dst in edges:
    sx = pos[src][0]
    sy = pos[src][1] * LANE
    dx = pos[dst][0]
    dy = pos[dst][1] * LANE

    cat = edge_category(src, dst)
    if cat == 'fusible':
        color, lw, ls = col_edge_fusible, edge_lw_strong, 'solid'
        z_line, z_arrow = 3, 4                 # top priority
    elif cat == 'blocked':
        color, lw, ls = col_edge_blocked, edge_lw_strong, 'solid'
        z_line, z_arrow = 2, 3
    else:
        color, lw, ls = col_edge_sub, edge_lw_sub, 'solid'
        z_line, z_arrow = 1, 2

    if src == 2 and dst == 8:
        # Identity shortcut: arch over the main row at the branch-lane height.
        x0, y0 = anchor(src, 't')
        x1, y1 = anchor(dst, 't')
        arch_y = LANE
        draw_path([(x0, y0), (x0, arch_y), (x1, arch_y), (x1, y1)],
                  color, lw, ls, z_line, z_arrow)
        continue

    if sy == dy:
        x0, y0 = anchor(src, 'r')
        x1, y1 = anchor(dst, 'l')
        draw_path([(x0, y0), (x1, y1)], color, lw, ls, z_line, z_arrow)
    else:
        if sy < dy:
            x0, y0 = anchor(src, 't')
            x1, y1 = anchor(dst, 'l')
            x_v = x0
            draw_path([(x0, y0), (x_v, y1), (x1, y1)],
                      color, lw, ls, z_line, z_arrow)
        else:
            x0, y0 = anchor(src, 'r')
            x1, y1 = anchor(dst, 't')
            x_v = x1
            draw_path([(x0, y0), (x_v, y0), (x1, y1)],
                      color, lw, ls, z_line, z_arrow)

# Now draw nodes
for t, (x, lane) in pos.items():
    y = lane * LANE
    box = FancyBboxPatch(
        (x - NODE_W/2, y - NODE_H/2), NODE_W, NODE_H,
        boxstyle='round,pad=0.01,rounding_size=0.06',
        facecolor=NODE_FACE, edgecolor=NODE_EDGE, linewidth=0.7,
        zorder=4)
    ax.add_patch(box)
    ax.text(x, y, str(t),
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            color=ID_COLOR, zorder=5)

# Axes cosmetics — extra top headroom for the legend with padding
ax.set_xlim(-0.7, cols - 0.3)
ax.set_ylim(-0.7, LANE + 1.9)   # extended top to leave space for legend + gap
ax.set_aspect('equal')
ax.set_axis_off()

# Legend above the figure, three entries
legend_items = [
    Line2D([0], [0], color=col_edge_fusible, linewidth=edge_lw_strong,
           linestyle='solid', label='Fusible'),
    Line2D([0], [0], color=col_edge_blocked, linewidth=edge_lw_strong,
           linestyle='solid', label='Blocked'),
    Line2D([0], [0], color=col_edge_sub, linewidth=edge_lw_sub,
           linestyle='solid', label='Sub-threshold'),
]
ax.legend(handles=legend_items, loc='upper center',
          bbox_to_anchor=(0.5, 1.02), ncol=3,
          frameon=False, handlelength=2.6,
          prop={'size': 11, 'weight': 'bold'})

plt.tight_layout()
out_pdf = 'resnet_flowchart_dtf_bs1.pdf'
plt.savefig(out_pdf, bbox_inches='tight')
print(f'Saved {out_pdf}')