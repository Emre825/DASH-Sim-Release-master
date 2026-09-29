"""
ResNet BS=32 — horizontal DAG flowchart combining two fusion views:
  top panel    : DTF fusions (amber)
  bottom panel : TVM & PyTorch fusions (teal for both, coral for PyTorch-only)
Single shared legend at the top of the figure.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Task graph
# ---------------------------------------------------------------------------
labels = {
     0: 'conv_1',  1: 'bn_1',    2: 'relu_1',
     3: 'conv_2',  4: 'bn_2',    5: 'relu_2',
     6: 'conv_3',  7: 'bn_3',    8: 'sum_1',  9: 'relu_3',
    10: 'conv_4', 11: 'bn_4',   12: 'relu_4',
    13: 'conv_5', 14: 'bn_5',
    15: 'conv_6', 16: 'bn_6',
    17: 'sum_2',  18: 'relu_5',
    19: 'conv_7', 20: 'bn_7',   21: 'relu_6',
    22: 'conv_8', 23: 'bn_8',
    24: 'conv_9', 25: 'bn_9',
    26: 'sum_3',  27: 'relu_7',
    28: 'pool',   29: 'linear',
}

edges = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(2,8),
    (8,9),(9,10),(10,11),(11,12),(12,13),(13,14),
    (9,15),(15,16),(16,17),(14,17),
    (17,18),(18,19),(19,20),(20,21),(21,22),(22,23),
    (18,24),(24,25),(25,26),(23,26),
    (26,27),(27,28),(28,29),
]

# Fusion sets
fused_dtf = {
    (0,1), (1,2),
    (3,4), (4,5), (5,6), (6,7),
    (12,13),
}
fused_both = {
    (1,2), (4,5), (7,8), (8,9), (11,12),
    (14,17), (16,17), (17,18),
    (20,21),
    (23,26), (25,26), (26,27),
}
fused_pyt_only = {(27, 28)}

# ---------------------------------------------------------------------------
# Layout (shared between panels)
# ---------------------------------------------------------------------------
pos = {}
col = 0
def place(t, c, lane=0): pos[t] = (c, lane)

for t in [0, 1, 2]:                place(t, col); col += 1
for t in [3, 4, 5, 6, 7, 8]:       place(t, col); col += 1
place(9, col); col += 1

c_main_start = col
for t in [10, 11, 12, 13, 14]:     place(t, col); col += 1
place(15, c_main_start + 3, lane=1)
place(16, c_main_start + 4, lane=1)
place(17, col); col += 1
place(18, col); col += 1

c_main_start = col
for t in [19, 20, 21, 22, 23]:     place(t, col); col += 1
place(24, c_main_start + 3, lane=1)
place(25, c_main_start + 4, lane=1)
place(26, col); col += 1
place(27, col); col += 1

place(28, col); col += 1
place(29, col); col += 1

cols = col

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
NODE_W, NODE_H = 0.46, 0.46
LANE = 1.2

NODE_FACE_CB   = '#F4C0D1'
NODE_EDGE_CB   = '#993556'
NODE_FACE_MB   = '#B5D4F4'
NODE_EDGE_MB   = '#0C447C'
ID_COLOR_CB    = '#4B1528'
ID_COLOR_MB    = '#042C53'

col_edge      = '#CCCCCC'
col_edge_dtf  = '#EF9F27'   # amber — DTF
col_edge_both = '#1D9E75'   # teal  — TVM & PyTorch
col_edge_pyt  = '#D85A30'   # coral — PyTorch only

edge_lw    = 0.9
edge_lw_hi = 2.6
arrow_size = 12


def task_is_cb(t):
    name = labels[t]
    return name.startswith('conv') or name.startswith('linear')


def anchor(task, side):
    x, lane = pos[task]
    y = lane * LANE
    if side == 'l': return (x - NODE_W/2, y)
    if side == 'r': return (x + NODE_W/2, y)
    if side == 't': return (x, y + NODE_H/2)
    if side == 'b': return (x, y - NODE_H/2)


def draw_path(ax, points, color, lw, z_line, z_arrow):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, linewidth=lw,
                linestyle='solid',
                solid_capstyle='butt', solid_joinstyle='miter', zorder=z_line)
    arrow = FancyArrowPatch(points[-2], points[-1],
                            arrowstyle='-|>', mutation_scale=arrow_size,
                            color=color, linewidth=lw, zorder=z_arrow,
                            shrinkA=0, shrinkB=0)
    ax.add_patch(arrow)


def draw_edge(ax, src, dst, color, lw, z_line, z_arrow):
    sy = pos[src][1] * LANE
    dy = pos[dst][1] * LANE

    # Special arch: 2 -> 8 routes above the main row
    if src == 2 and dst == 8:
        x0, y0 = anchor(src, 't')
        x1, y1 = anchor(dst, 't')
        arch_y = LANE
        draw_path(ax, [(x0, y0), (x0, arch_y), (x1, arch_y), (x1, y1)],
                  color, lw, z_line, z_arrow)
        return

    if sy == dy:
        x0, y0 = anchor(src, 'r')
        x1, y1 = anchor(dst, 'l')
        draw_path(ax, [(x0, y0), (x1, y1)], color, lw, z_line, z_arrow)
    elif sy < dy:
        x0, y0 = anchor(src, 't')
        x1, y1 = anchor(dst, 'l')
        draw_path(ax, [(x0, y0), (x0, y1), (x1, y1)],
                  color, lw, z_line, z_arrow)
    else:
        x0, y0 = anchor(src, 'r')
        x1, y1 = anchor(dst, 't')
        draw_path(ax, [(x0, y0), (x1, y0), (x1, y1)],
                  color, lw, z_line, z_arrow)


def render_flowchart(ax, edge_classifier, title):
    """edge_classifier: (src, dst) -> (color, lw, z_line, z_arrow)"""
    for src, dst in edges:
        color, lw, z_line, z_arrow = edge_classifier(src, dst)
        draw_edge(ax, src, dst, color, lw, z_line, z_arrow)

    for t, (x, lane) in pos.items():
        y = lane * LANE
        is_cb = task_is_cb(t)
        face   = NODE_FACE_CB if is_cb else NODE_FACE_MB
        edge_c = NODE_EDGE_CB if is_cb else NODE_EDGE_MB
        txt_c  = ID_COLOR_CB  if is_cb else ID_COLOR_MB
        box = FancyBboxPatch(
            (x - NODE_W/2, y - NODE_H/2), NODE_W, NODE_H,
            boxstyle='round,pad=0.01,rounding_size=0.06',
            facecolor=face, edgecolor=edge_c, linewidth=0.8,
            zorder=4)
        ax.add_patch(box)
        ax.text(x, y, str(t),
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color=txt_c, zorder=5)

    ax.set_xlim(-0.7, cols - 0.3)
    ax.set_ylim(-0.5, LANE + 0.7)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.text(-0.6, LANE + 0.55, title,
            fontsize=14, fontweight='bold',
            ha='left', va='top')


# ---------------------------------------------------------------------------
# Edge classifiers
# ---------------------------------------------------------------------------
def classify_dtf(src, dst):
    if (src, dst) in fused_dtf:
        return col_edge_dtf, edge_lw_hi, 3, 4
    return col_edge, edge_lw, 1, 2


def classify_tvm_pyt(src, dst):
    if (src, dst) in fused_both:
        return col_edge_both, edge_lw_hi, 3, 4
    if (src, dst) in fused_pyt_only:
        return col_edge_pyt, edge_lw_hi, 3, 4
    return col_edge, edge_lw, 1, 2


# ---------------------------------------------------------------------------
# Figure: two rows, one shared legend
# ---------------------------------------------------------------------------
fig_w = max(12, cols * 0.5)
fig_h = 2.6
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(fig_w, fig_h),
    gridspec_kw={'hspace': 0.0},
)

render_flowchart(ax_top, classify_dtf,     '(a) DTF')
render_flowchart(ax_bot, classify_tvm_pyt, '(b) TVM & PyTorch')

legend_items = [
    Patch(facecolor=NODE_FACE_CB, edgecolor=NODE_EDGE_CB,
          linewidth=0.8, label='CB'),
    Patch(facecolor=NODE_FACE_MB, edgecolor=NODE_EDGE_MB,
          linewidth=0.8, label='MB'),
    Line2D([0], [0], color=col_edge_dtf, linewidth=edge_lw_hi,
           label='Fused by DTF'),
    Line2D([0], [0], color=col_edge_both, linewidth=edge_lw_hi,
           label='Fused by TVM & PyTorch'),
    Line2D([0], [0], color=col_edge_pyt, linewidth=edge_lw_hi,
           label='Fused by PyTorch only'),
]
fig.legend(handles=legend_items,
           loc='lower center',
           bbox_to_anchor=(0.5, 1.0),
           ncol=len(legend_items),
           frameon=False,
           handlelength=1.6,
           handleheight=1.1,
           handletextpad=0.7,
           columnspacing=2.2,
           prop={'weight': 'bold', 'size': 11})

fig.subplots_adjust(top=1.0, bottom=0.0, left=0.02, right=0.98, hspace=0.0)

out_pdf = 'resnet_flowchart_bs32_combined.pdf'
plt.savefig(out_pdf, bbox_inches='tight')
print(f'Saved {out_pdf}')
