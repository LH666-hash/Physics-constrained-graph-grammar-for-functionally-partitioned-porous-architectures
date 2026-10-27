"""Plot the final surface, internal, and assembled nodes of five domains."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Source_Data/Extended_Geometries'
OUT = ROOT / 'Figures/Supplementary_Figures'
CASES = [
    ('Local_High_Negative_Curvature_seed42', 'Local negative curvature'),
    ('Gaussian_Curvature_Sign_Change_seed42', 'Gaussian sign change'),
    ('Large_Hole_seed42', 'Large-hole boundary'),
    ('Smooth_Variable_Curvature_seed42', 'Smooth varying curvature'),
    ('Multibranch_Free_Boundary_seed42', 'Multibranch free boundary'),
]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.5,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, axs = plt.subplots(5, 3, figsize=(6.89, 8.9), constrained_layout=True)
for row, (folder, label) in enumerate(CASES):
    base = SOURCE / folder
    if folder.startswith('Multibranch_'):
        base /= 'Final_Merged_Dual_Domain'
    nodes = pd.read_csv(base / 'assembly_nodes_final.csv',
                        usecols=['x', 'z', 'node_role'])
    surface = nodes[nodes.node_role.notna()]
    internal = nodes[nodes.node_role.isna()]
    xmax = float(np.ceil(nodes.x.max()))
    for col, ax in enumerate(axs[row]):
        if col != 1:
            ax.scatter(surface.x, surface.z, s=.12, c='#8abedb',
                       alpha=.65, linewidths=0, rasterized=True)
        if col != 0:
            ax.scatter(internal.x, internal.z, s=.55, c='#ce79a1',
                       alpha=.65, linewidths=0, rasterized=True)
        ax.set(xlim=(0, xmax), ylim=(0, 25))
        ax.set_xticks([0, xmax / 2, xmax])
        ax.set_yticks([0, 12.5, 25])
        ax.tick_params(direction='out', length=2, pad=1, labelsize=7.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(.02, .96, chr(97 + row * 3 + col), transform=ax.transAxes,
                ha='left', va='top', fontweight='bold', fontsize=9)
        if col == 0:
            ax.set_ylabel('Height (mm)', fontsize=8.5)
            ax.set_title(label, loc='left', fontsize=8.5, pad=2)
        if row == 0:
            ax.text(.5, 1.16, ('Surface', 'Internal', 'Assembled')[col],
                    transform=ax.transAxes, ha='center', fontsize=9)
        if row == 4:
            ax.set_xlabel('x (mm)', fontsize=8.5)
for extension in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / ('FigureS8.' + extension), dpi=400,
                bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Figure S8 uses the five final assembled node tables')
