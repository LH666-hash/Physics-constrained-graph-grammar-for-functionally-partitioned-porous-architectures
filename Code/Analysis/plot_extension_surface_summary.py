"""Plot terminal, contour-audited surface pores for accepted design domains."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Source_Data'
OUTPUT = ROOT / 'Figures' / 'Supplementary_Figures'
CASES = [
    ('Gaussian_Curvature_Sign_Change_seed42', 'Gaussian sign change'),
    ('Large_Hole_seed42', 'Large aperture'),
    ('Local_High_Negative_Curvature_seed42', 'Local negative curvature'),
    ('Smooth_Variable_Curvature_seed42', 'Smooth varying curvature'),
    ('Multibranch_Free_Boundary_seed42', 'Multibranch free boundary'),
]
COLORS = ['#765A9B', '#44749A', '#C36593', '#6A9BAC', '#A85B84', '#27679A']
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .8, 'svg.fonttype': 'none', 'savefig.dpi': 300,
})


def result_path(name: str) -> Path:
    if name == 'Rectangle control':
        return SOURCE / 'Representative_Structure_Seed42'
    if name == 'Multibranch_Free_Boundary_seed42':
        return SOURCE / 'Extended_Geometries' / name / 'Final_Merged_Dual_Domain'
    return SOURCE / 'Extended_Geometries' / name


def accepted(name: str) -> bool:
    path = result_path(name) / 'final_assembly_status.json'
    if not path.exists():
        return False
    return bool(json.loads(path.read_text(encoding='utf-8')).get('graph_screen_pass'))


def summary(name: str) -> dict:
    path = result_path(name) / 'global_surface_pore_audit' / 'global_surface_pore_summary.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('method') != 'patch_independent_global_3d_contour_open_component_audit_v3':
        raise ValueError(f'{name}: contour audit v3 is required')
    return data


def save(fig, stem: str) -> None:
    fig.savefig(OUTPUT / (stem + '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT / (stem + '.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT / (stem + '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_s7(names: list[tuple[str, str]]) -> None:
    labels = [label for _, label in names] + ['Rectangle control']
    data = [summary(name) for name, _ in names] + [summary('Rectangle control')]
    ar = [d['global_area_weighted_mean_complete_pore_aspect_ratio'] for d in data]
    sf = [d['global_area_weighted_mean_complete_pore_shape_factor'] for d in data]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(17.5 / 2.54, 3.15), layout='constrained')
    for ax, values, xlabel, letter in zip(
        axes, (ar, sf), ('Area-weighted aspect ratio', 'Area-weighted shape factor'), 'ab'
    ):
        ax.barh(y, values, color=COLORS[:len(names)] + [COLORS[-1]], height=.7)
        ax.set_yticks(y, labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.text(-.1, 1.04, letter, transform=ax.transAxes, fontweight='bold', fontsize=11)
    save(fig, 'FigureS7')


def plot_s12(names: list[tuple[str, str]]) -> None:
    n = len(names)
    columns = 2 if n == 4 else 3
    fig, axes = plt.subplots(2, columns, figsize=(17.5 / 2.54, 4.1), layout='constrained')
    axes = list(axes.flat)
    plotted = []
    for idx, (name, label) in enumerate(names):
        path = result_path(name) / 'global_surface_pore_audit' / 'global_complete_pores.csv'
        df = pd.read_csv(path)
        df = df[df.complete_pore.eq(1) & df.below_area_resolution.eq(0)]
        if df.empty:
            raise ValueError(f'{name}: no complete pores')
        plotted.append((df, label))
    all_area = np.concatenate([df.area_mm2.to_numpy(float) for df, _ in plotted])
    all_ar = np.concatenate([df.aspect_ratio_3d_pca.to_numpy(float) for df, _ in plotted])
    a5, a95 = np.nanpercentile(all_area, [5, 95])
    r5, r95 = np.nanpercentile(all_ar, [5, 95])
    for idx, (df, label) in enumerate(plotted):
        ax = axes[idx]
        areas = np.clip(df.area_mm2.to_numpy(float), a5, a95)
        sizes = 5 + 45 * (areas-a5) / max(a95-a5, 1e-12)
        sc = ax.scatter(df.equivalent_radius_mm, df.shape_factor_3d,
                        c=np.clip(df.aspect_ratio_3d_pca, r5, r95),
                        s=sizes, cmap='coolwarm', vmin=r5, vmax=r95,
                        alpha=.7, edgecolors='white', linewidths=.15, rasterized=True)
        ax.axvline(df.radius_acceptance_mm.iloc[0], color='#777777', lw=.9, ls='--')
        ax.set_xlim(0, .6)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, .2, .4, .6])
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_title(f'{chr(97+idx)}  {label}', fontsize=9, loc='left')
        if idx >= columns:
            ax.set_xlabel('Equivalent radius (mm)')
        if idx % columns == 0:
            ax.set_ylabel('Shape factor')
    for ax in axes[n:]:
        ax.axis('off')
    colorbar = fig.colorbar(sc, ax=axes[:n], location='right', shrink=.74,
                           label='Aspect ratio (5th–95th percentile scale)')
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(5))
    save(fig, 'FigureS12')


def main() -> None:
    names = [(name, label) for name, label in CASES if accepted(name)]
    if len(names) < 4:
        raise RuntimeError('Fewer than four extension domains have final graph-screen passes')
    plot_s7(names)
    plot_s12(names)
    print('Included:', ', '.join(name for name, _ in names))


if __name__ == '__main__':
    main()
