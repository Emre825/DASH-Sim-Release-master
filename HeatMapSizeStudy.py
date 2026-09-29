"""
Fusion speedup heatmap: 5 models x 3 batch sizes x 5 methods.

- Cell color encodes speedup magnitude (blue ramp).
- Cell text shows 'speedup\n(fusion_count)'.
- Oracle columns are shown with a diagonal hatch + italic text to flag
  that Oracle is an upper bound, not a real system.
- CV columns show coefficient of variation values for communication volume.
- Missing Oracle values (VGG, SqueezeNet) render as '—'.
- Where Oracle (Constrd) is reported, we use it instead of the
  unconstrained Oracle, per the paper's tables.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# DATA (speedup, fusion_count). None means "not reported".
# Oracle uses the Constrd value when available; otherwise the plain Oracle.
# ---------------------------------------------------------------------------
data = {
    'AlexNet': {
        1:  {'DTF': (1.69, 8),  'TVM': (1.26, 7),  'PyTorch': (1.07, 4),  'Oracle': (2.43, 1)},
        16: {'DTF': (2.26, 7),  'TVM': (1.23, 7),  'PyTorch': (1.07, 4),  'Oracle': (3.15, 1)},
        32: {'DTF': (2.41, 7),  'TVM': (1.24, 7),  'PyTorch': (1.07, 4),  'Oracle': (3.34, 2)},   # Constrd
    },
    'ResNet': {
        1:  {'DTF': (1.34, 4),  'TVM': (1.39, 12), 'PyTorch': (1.43, 13), 'Oracle': (7.27, 1)},
        16: {'DTF': (1.70, 7),  'TVM': (1.33, 12), 'PyTorch': (1.35, 13), 'Oracle': (9.88, 1)},
        32: {'DTF': (1.70, 7),  'TVM': (1.32, 12), 'PyTorch': (1.34, 13), 'Oracle': (6.92, 2)},
    },
    'MobileNet': {
        1:  {'DTF': (1.67, 33), 'TVM': (1.40, 27), 'PyTorch': (1.42, 28), 'Oracle': (6.84, 1)},
        16: {'DTF': (2.27, 29), 'TVM': (1.45, 27), 'PyTorch': (1.46, 28), 'Oracle': (8.87, 3)},   # Constrd
        32: {'DTF': (2.17, 27), 'TVM': (1.47, 27), 'PyTorch': (1.48, 28), 'Oracle': (8.00, 6)},   # Constrd
    },
    'VGG': {
        1:  {'DTF': (2.41, 11), 'TVM': (1.22, 15), 'PyTorch': (1.00, 1),  'Oracle': (3.99, 1)},
        16: {'DTF': (2.72, 11), 'TVM': (1.21, 15), 'PyTorch': (1.00, 1),  'Oracle': (4.62, 2)},
        32: {'DTF': (2.82, 11), 'TVM': (1.21, 15), 'PyTorch': (1.00, 1),  'Oracle': (4.62, 3)},
    },
    'SqueezeNet': {
        1:  {'DTF': (1.44, 12), 'TVM': (1.54, 26), 'PyTorch': (1.26, 17), 'Oracle': (7.03, 1)},
        16: {'DTF': (1.56, 11), 'TVM': (1.49, 26), 'PyTorch': (1.19, 17), 'Oracle': (7.38, 2)},
        32: {'DTF': (1.54, 11), 'TVM': (1.48, 26), 'PyTorch': (1.18, 17), 'Oracle': (5.07, 4)},
    },
}

cv_data = {
    'AlexNet':    {1: 0.61, 16: 1.17, 32: 1.24},
    'ResNet':     {1: 0.44, 16: 0.96, 32: 1.05},
    'MobileNet':  {1: 0.20, 16: 0.70, 32: 0.71},
    'VGG':        {1: 1.61, 16: 1.72, 32: 1.73},
    'SqueezeNet': {1: 0.84, 16: 1.11, 32: 1.13},
}

models      = ['AlexNet', 'ResNet', 'MobileNet', 'VGG', 'SqueezeNet']
batch_sizes = [1, 16, 32]
methods     = ['DTF', 'TVM', 'PyTorch', 'Oracle', 'CV']

n_rows = len(models)
n_cols = len(batch_sizes) * len(methods)

# Pack into matrices
speedup = np.full((n_rows, n_cols), np.nan)
fusion  = np.full((n_rows, n_cols), np.nan)
for i, model in enumerate(models):
    for j, bs in enumerate(batch_sizes):
        for k, m in enumerate(methods):
            col = j * len(methods) + k
            if m == 'CV':
                speedup[i, col] = cv_data[model][bs]
                continue

            sp, fc = data[model][bs][m]
            if sp is not None:
                speedup[i, col] = sp
            if fc is not None:
                fusion[i, col] = fc

# ---------------------------------------------------------------------------
# Blue colormap matching the palette from the mock-up
# ---------------------------------------------------------------------------
ramp = ['#FFFFFF', '#E6F1FB', '#B5D4F4', '#85B7EB', '#378ADD', '#185FA5', '#0C447C']
cmap = LinearSegmentedColormap.from_list('blues', ramp, N=256)
vmin, vmax = 1.0, 3.0                          # speedup ceiling for the ramp
                                                # Oracle values often exceed vmax — they
                                                # still render (clipped to top of ramp) but
                                                # don't compress the DTF/TVM/PyTorch range.

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 5.6))

for i in range(n_rows):
    for j in range(n_cols):
        sp = speedup[i, j]
        method_idx = j % len(methods)
        is_oracle = (method_idx == 3)
        is_cv = (method_idx == 4)
        y = n_rows - 1 - i                      # flip so AlexNet is on top

        if np.isnan(sp):
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor='#f2f2f2',
                                   edgecolor='white', linewidth=1.2))
            ax.text(j + 0.5, y + 0.5, '—', ha='center', va='center',
                    fontsize=11, color='#aaa')
            continue

        # Colored fill — Oracle/CV cells use flat backgrounds so only
        # DTF/TVM/PyTorch use the speedup ramp.
        t = (sp - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        if is_oracle:
            facecolor = 'white'
        elif is_cv:
            facecolor = '#f7f7f7'
        else:
            facecolor = cmap(t)
        ax.add_patch(Rectangle((j, y), 1, 1, facecolor=facecolor,
                               edgecolor='white', linewidth=1.2))

        # Subtle diagonal hatch on Oracle cells
        if is_oracle:
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor='none',
                                   edgecolor='#333', linewidth=0,
                                   hatch='////', alpha=0.25))

        # Annotation: speedup (bold-ish) + fusion count (smaller)
        # Oracle cells have a white background → use dark text for contrast.
        txt_color = 'white' if ((not is_oracle and not is_cv) and t > 0.55) else '#111'
        fc = fusion[i, j]
        sp_str = f'{sp:.2f}'
        # Oracle cells: show speedup only (no fusion count in parentheses).
        show_fc = (not is_oracle and not is_cv) and (not np.isnan(fc))
        ax.text(j + 0.5, y + (0.58 if show_fc else 0.5), sp_str,
                ha='center', va='center',
                fontsize=12, color=txt_color,
                fontweight='semibold',
                style='italic' if is_oracle else 'normal')
        if show_fc:
            ax.text(j + 0.5, y + 0.25, f'({int(fc)})',
                    ha='center', va='center',
                    fontsize=10, color=txt_color,
                    alpha=0.85)

# Axes setup
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)
# Let cells stretch to fill the figure width — avoids large white gaps that
# appear with 'equal' when the data aspect doesn't match the figure aspect,
# and widens the columns so method labels no longer interleave.
ax.set_aspect('auto')

# Method labels on top of columns (bold, above the heatmap)
ax.set_xticks([])
for j, m in enumerate(methods * 3):
    ax.text(j + 0.5, n_rows + 0.15, m, ha='center', va='bottom',
            fontsize=12, fontweight='bold')

# Model ticks on left
ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
ax.set_yticklabels(models, fontsize=13)
ax.tick_params(axis='y', length=0, pad=4)

# Batch-size group labels on top (above the method labels)
for mid, label in zip([2.5, 7.5, 12.5], ['Batch size 1', 'Batch size 16', 'Batch size 32']):
    ax.text(mid, n_rows + 0.55, label, ha='center', va='bottom',
            fontsize=12.5, fontweight='semibold')

# Thick dividers between batch-size groups
for x in [5, 10]:
    ax.plot([x, x], [0, n_rows], color='#222', linewidth=1.4, clip_on=False)
# Top + bottom frame
for y in [0, n_rows]:
    ax.plot([0, n_cols], [y, y], color='#222', linewidth=0.8, clip_on=False)
for x in [0, n_cols]:
    ax.plot([x, x], [0, n_rows], color='#222', linewidth=0.8, clip_on=False)

# Strip default spines
for s in ax.spines.values():
    s.set_visible(False)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.015, aspect=28)
cbar.set_label('Speedup (×)', fontsize=13, fontweight='bold', labelpad=10)
cbar.ax.tick_params(labelsize=10.5)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
out_pdf = 'fusion_speedup_heatmap.pdf'
plt.savefig(out_pdf, bbox_inches='tight')
print(f'Saved {out_pdf}')