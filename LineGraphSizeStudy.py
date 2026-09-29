import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# (batch_size, speedup, fusion_count) for non-Oracle; (batch_size, speedup) for Oracle.
# Oracle uses "Oracle (Constrd)" value if available, otherwise the normal Oracle value.
data = {
    'AlexNet': {
        'DTF':     [(1, 1.69, 8),  (16, 2.26, 7),  (32, 2.41, 7)],
        'TVM':     [(1, 1.26, 7),  (16, 1.23, 7),  (32, 1.24, 7)],
        'PyTorch': [(1, 1.07, 4),  (16, 1.07, 4),  (32, 1.07, 4)],
        'Oracle':  [(1, 2.43),     (16, 3.15),     (32, 3.34)],   # bs32 from Constrd
    },
    'DSCNN': {
        'DTF':     [(1, 1.96, 16), (16, 1.34, 6),  (32, 4.29, 22)],
        'TVM':     [(1, 1.39, 9),  (16, 1.36, 9),  (32, 1.44, 9)],
        'PyTorch': [(1, 1.45, 10), (16, 1.41, 10), (32, 1.49, 10)],
        'Oracle':  [(1, 6.04),     (16, 7.10),     (32, 9.23)],   # bs32 from Constrd
    },
    'AutoEncoder': {
        'DTF':     [(1, 1.70, 8),  (16, 1.81, 9),  (32, 1.69, 8)],
        'TVM':     [(1, 1.64, 9),  (16, 1.63, 9),  (32, 1.59, 9)],
        'PyTorch': [(1, 1.00, 0),  (16, 1.00, 0),  (32, 1.00, 0)],
    },
    'ResNet': {
        'DTF':     [(1, 1.34, 4),  (16, 1.70, 7),  (32, 1.70, 7)],
        'TVM':     [(1, 1.39, 12), (16, 1.33, 12), (32, 1.32, 12)],
        'PyTorch': [(1, 1.43, 13), (16, 1.35, 13), (32, 1.34, 13)],
        'Oracle':  [(1, 3.65),     (16, 4.68),     (32, 4.82)],
    },
    'MobileNet': {
        'DTF':     [(1, 1.67, 33), (16, 2.27, 29), (32, 2.17, 27)],
        'TVM':     [(1, 1.40, 27), (16, 1.45, 27), (32, 1.47, 27)],
        'PyTorch': [(1, 1.42, 28), (16, 1.46, 28), (32, 1.48, 28)],
        'Oracle':  [(1, 6.84),     (16, 8.87),     (32, 8.00)],   # bs16, bs32 from Constrd
    },
    'VGG': {
        'DTF':     [(1, 2.41, 11), (16, 2.72, 11), (32, 2.82, 11)],
        'TVM':     [(1, 1.22, 15), (16, 1.21, 15), (32, 1.21, 15)],
        'PyTorch': [(1, 1.00, 1),  (16, 1.00, 1),  (32, 1.00, 1)],
    },
    'SqueezeNet': {
        'DTF':     [(1, 1.44, 12), (16, 1.56, 11), (32, 1.54, 11)],
        'TVM':     [(1, 1.54, 26), (16, 1.49, 26), (32, 1.48, 26)],
        'PyTorch': [(1, 1.26, 17), (16, 1.19, 17), (32, 1.18, 17)],
    },
}

sns.set_theme(style="whitegrid")

mode_colors = {
    'DTF':     '#1f77b4',
    'TVM':     '#2ca02c',
    'PyTorch': '#ff7f0e',
    'Oracle':  '#9467bd',
}
mode_order = ['DTF', 'TVM', 'PyTorch', 'Oracle']

batch_sizes = [1, 16, 32]
x_positions = [0, 1, 2]  # evenly spaced for readability
bs_to_x = dict(zip(batch_sizes, x_positions))

title_fontsize = 18
axis_label_fontsize = 15
tick_fontsize = 13
annotation_size = 10
legend_fontsize = 15


def plot_model(ax, model, add_legend=False):
    model_data = data[model]
    all_speedups = []

    for mode in mode_order:
        if mode not in model_data:
            continue
        points = model_data[mode]
        xs = [bs_to_x[p[0]] for p in points]
        ys = [p[1] for p in points]
        all_speedups.extend(ys)

        ax.plot(
            xs, ys,
            marker='o', markersize=8, linewidth=2.5,
            color=mode_colors[mode], label=mode
        )

        if mode != 'Oracle':
            for (_, su, fc), x in zip(points, xs):
                ax.annotate(
                    f"{su:.2f} ({fc})",
                    xy=(x, su),
                    xytext=(7, 4),
                    textcoords='offset points',
                    fontsize=annotation_size,
                    color=mode_colors[mode],
                    fontweight='semibold',
                    va='bottom',
                    ha='left',
                )

    ax.set_title(model, fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel('Batch Size', fontsize=axis_label_fontsize, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=axis_label_fontsize, fontweight='bold')
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlim(-0.35, 2.6)

    mn, mx = min(all_speedups), max(all_speedups)
    pad = max((mx - mn) * 0.25, 0.4)
    ax.set_ylim(mn - pad * 0.3, mx + pad)

    if add_legend:
        ax.legend(loc='best', frameon=True, fontsize=legend_fontsize)


# ==========================================
# Combined Figure: 7 models in a 3+3+1 layout
# ==========================================
model_order = ['AlexNet', 'DSCNN', 'AutoEncoder',
               'ResNet', 'MobileNet', 'VGG',
               'SqueezeNet']

fig = plt.figure(figsize=(22, 17))
gs = fig.add_gridspec(3, 6, hspace=0.45, wspace=0.55)

axes = [
    fig.add_subplot(gs[0, 0:2]),  # AlexNet
    fig.add_subplot(gs[0, 2:4]),  # DSCNN
    fig.add_subplot(gs[0, 4:6]),  # AutoEncoder
    fig.add_subplot(gs[1, 0:2]),  # ResNet
    fig.add_subplot(gs[1, 2:4]),  # MobileNet
    fig.add_subplot(gs[1, 4:6]),  # VGG
    fig.add_subplot(gs[2, 2:4]),  # SqueezeNet (centered)
]

legend_handles = [
    Line2D([0], [0], color=mode_colors[m], lw=3, marker='o', markersize=10, label=m)
    for m in mode_order
]

fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.965),
    ncol=4,
    frameon=False,
    fontsize=legend_fontsize
)

for ax, model in zip(axes, model_order):
    plot_model(ax, model)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('speedup_fusion_7models_layout.pdf', format='pdf', bbox_inches='tight')

# ==========================================
# Individual per-model figures (disabled for now; re-enable if needed)
# ==========================================
# for model in model_order:
#     fig_single, ax_single = plt.subplots(figsize=(9, 6))
#     plot_model(ax_single, model, add_legend=True)
#     safe_model_name = model.lower().replace(' ', '_')
#     fig_single.tight_layout()
#     fig_single.savefig(f'speedup_fusion_{safe_model_name}.pdf', format='pdf', bbox_inches='tight')
#     plt.close(fig_single)

plt.show()
