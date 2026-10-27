"""Plot input-domain internal cross-section diversity from production geometry slices."""

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = pd.read_csv(ROOT / 'Source_Data' / 'Extended_Geometries' / 'domain_cross_section_topology.csv')
OUT = ROOT / 'Figures' / 'Supplementary_Figures'
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
})

styles = [
    ('large_hole', 'Large aperture', '#44749A', 'o'),
    ('gaussian_sign_change', 'Gaussian sign change', '#8468A5', 's'),
    ('local_negative_curvature', 'Local negative curvature', '#C36593', '^'),
    ('smooth_variable_curvature', 'Smooth varying curvature', '#6A9BAC', 'D'),
    ('knitted_elephant', 'Multibranch free boundary', '#A85B84', 'P'),
]
fig, axs = plt.subplots(1, 3, figsize=(17.5 / 2.54, 2.7), constrained_layout=True)
for slug, label, color, marker in styles:
    df = DATA[DATA.domain == slug].sort_values('height_mm')
    x = df.height_mm.to_numpy()
    axs[0].plot(x, df.cross_section_component_count, color=color, lw=1.5,
                marker=marker, ms=2.3, label=label)
    axs[1].plot(x, df.cross_section_hole_count, color=color, lw=1.5,
                marker=marker, ms=2.3)
    area = df.cross_section_area_mm2.to_numpy()
    axs[2].plot(x, area / max(area.max(), 1e-12), color=color, lw=1.5,
                marker=marker, ms=2.3)
for ax, letter in zip(axs, 'abc'):
    ax.text(-.12, 1.07, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top')
    ax.set_xlabel('Height (mm)')
    ax.set_xlim(0, 25)
    ax.set_xticks([0, 8, 16, 24])
    ax.grid(axis='y', color='#E8E8EC', lw=.6)
axs[0].set_ylabel('Connected cross-section domains')
axs[0].set_ylim(.5, 20.5)
axs[0].set_yscale('log')
axs[0].set_yticks([1, 2, 5, 10, 20])
axs[0].set_yticklabels(['1', '2', '5', '10', '20'])
axs[1].set_ylabel('Cross-section holes')
axs[1].set_ylim(-1, 45)
axs[1].yaxis.set_major_locator(MaxNLocator(5, integer=True))
axs[2].set_ylabel('Area / domain maximum')
axs[2].set_ylim(0, 1.08)
axs[2].yaxis.set_major_locator(MaxNLocator(5))
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.17),
           ncol=3, columnspacing=1.5, handlelength=1.5)
fig.savefig(OUT / 'FigureS6.png', dpi=300,
            bbox_inches='tight', facecolor='white')
fig.savefig(OUT / 'FigureS6.svg',
            bbox_inches='tight', facecolor='white')
plt.close(fig)
