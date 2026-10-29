"""Corrected internal screen followed by natural-surface and slit references."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / "Source_Data/Benchmark_Analysis"
SURFACE = PACKAGE / "Source_Data/Ablation_Analysis/表面"
OUT = PACKAGE / "ReviewChecks/figures"
BLUE, PURPLE, PINK = "#145D93", "#72538F", "#B85E78"
FAMILIES = {
    "Graded BCC": ("#1B669D", "o"),
    "Graded Gyroid": ("#72538F", "s"),
    "Simple cubic": ("#A65578", "^"),
    "Diamond": ("#397F9D", "D"),
    "Schwarz P": ("#9A779F", "P"),
}


def main(out_dir=None):
    output = Path(out_dir) if out_dir is not None else OUT
    if not output.is_absolute():
        output = PACKAGE / output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    preset = pd.read_csv(ROOT / "内部/01_预设内部结构_全部45220.csv")
    greedy = pd.read_csv(ROOT / "内部/02_局部随机贪心_全部6000次评价_5988个不同结构.csv").drop_duplicates("geometry_sha1")
    assert len(preset) == 45220 and len(greedy) == 5988
    pc = pd.read_csv(ROOT / "内部/07_PC-AEGG完整组5次终态内部指标.csv")
    slit = pd.read_csv(ROOT / "表面/surface_only_reference_comparison.csv")
    natural = pd.read_csv(ROOT / "Natural_Surface_Comparison/all_241_internal_eligible_natural_surfaces.csv")
    pc_surface = pd.read_csv(SURFACE / "surface_ablation_group_seed_final_graph_metrics.csv")
    pc_surface = pc_surface[pc_surface.group.eq("surface_full")]
    fig, (left, right) = plt.subplots(1, 2, figsize=(17.5/2.54, 7.4/2.54), layout="constrained")

    assert set(preset.family) == set(FAMILIES) and set(greedy.family) == set(FAMILIES)
    for family, (color, marker) in FAMILIES.items():
        one = preset[preset.family.eq(family)]
        left.scatter(one.epsilon, one.predicted_rise_height_mm,
                     s=5.5, alpha=0.17, color=color, marker=marker,
                     linewidths=0, rasterized=True, zorder=1)
        feasible = one[one.feasible.eq(True)]
        left.scatter(feasible.epsilon, feasible.predicted_rise_height_mm,
                     s=6.5, alpha=0.42, color=color, marker=marker,
                     linewidths=0, rasterized=True, zorder=2)
        local = greedy[greedy.family.eq(family)]
        left.scatter(local.epsilon, local.predicted_rise_height_mm,
                     s=10, alpha=0.34, facecolors="none", edgecolors=color,
                     marker=marker, linewidths=0.55, rasterized=True, zorder=3)
    left.scatter(pc.epsilon, pc.predicted_rise_height_mm, s=90,
                 color="#173F65", marker="*", edgecolor="white", lw=0.8,
                 label="PC-AEGG assembled runs", zorder=5)
    eligible_preset = preset[preset.feasible.astype(bool) & preset.epsilon.ge(pc.epsilon.mean()) & preset.predicted_rise_height_mm.ge(25)]
    eligible_greedy = greedy[greedy.feasible.astype(bool) & greedy.epsilon.ge(pc.epsilon.mean()) & greedy.predicted_rise_height_mm.ge(25)]
    assert (len(eligible_preset), len(eligible_greedy)) == (1, 240)
    left.scatter(eligible_greedy.epsilon, eligible_greedy.predicted_rise_height_mm,
                 s=15, facecolors="none", edgecolors="#C1303C", linewidths=.6, rasterized=True, zorder=4)
    left.scatter(eligible_preset.epsilon, eligible_preset.predicted_rise_height_mm,
                 s=85, marker="X", color="#C1303C", edgecolor="white", linewidth=.6, zorder=6)
    left.axvline(pc.epsilon.mean(), ls="--", color=BLUE, lw=0.9)
    left.axhline(pc.predicted_rise_height_mm.mean(), ls="--", color=BLUE, lw=0.9)
    left.set(xlabel="Mean cross-sectional porosity", ylabel="Predicted rise height (mm)",
             title="d  Structure-performance screen", xlim=(0, 1), ylim=(0, 26.5))
    left.text(0.98, 0.83, "Eligible internal designs\n1 preset; 240 greedy",
              transform=left.transAxes, ha="right", va="top", fontsize=7.2,
              color="#344054", bbox=dict(facecolor="white", edgecolor="none", alpha=.92, pad=2))
    family_handles = [Line2D([0], [0], marker=m, linestyle="none", markersize=5.5,
                             markerfacecolor=c, markeredgecolor=c, label=name)
                      for name, (c, m) in FAMILIES.items()]
    family_handles += [Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                              markerfacecolor="none", markeredgecolor="#4B5563",
                              label="Local greedy: open markers"),
                       Line2D([0], [0], marker="*", linestyle="none", markersize=8,
                              markerfacecolor="#173F65", markeredgecolor="#173F65",
                              label="PC-AEGG")]
    left.legend(handles=family_handles, frameon=True, facecolor="white", edgecolor="none", framealpha=.92, fontsize=6.6,
                loc="lower left", ncol=2, columnspacing=0.8, handletextpad=0.35)

    both = ((slit.surface_A_eff_star >= pc_surface.liquid_open_area_fraction.mean()) &
            (slit.surface_r_eq_ctrl_mm <= pc_surface.maximum_complete_pore_radius_mm.mean()))
    slit["reaches_pc_surface_mean_area"] = slit.surface_A_eff_star >= pc_surface.liquid_open_area_fraction.mean()
    slit["reaches_pc_surface_mean_radius"] = slit.surface_r_eq_ctrl_mm <= pc_surface.maximum_complete_pore_radius_mm.mean()
    slit["reaches_both_pc_surface_means"] = both
    slit.to_csv(ROOT / "表面/surface_only_reference_comparison.csv", index=False)
    right.scatter(slit.surface_A_eff_star, slit.surface_r_eq_ctrl_mm,
                  s=5, alpha=0.20, color=PURPLE, linewidths=0, rasterized=True,
                  label="0.30 mm slit surfaces (n = 8,212)")
    right.scatter(slit.loc[both, "surface_A_eff_star"],
                  slit.loc[both, "surface_r_eq_ctrl_mm"],
                  s=13, color=PINK, alpha=0.85, linewidths=0,
                  label=f"Reach both surface means (n = {int(both.sum())})")
    greedy_natural = natural[natural.search_group.eq("local_greedy")]
    preset_natural = natural[natural.search_group.eq("preset")]
    assert len(greedy_natural) == 240 and len(preset_natural) == 1
    right.scatter(greedy_natural.surface_open_area_fraction,
                  greedy_natural.maximum_complete_pore_equivalent_radius_mm,
                  s=28, color="#16856A", alpha=.35, linewidths=0,
                  label="Eligible greedy natural surfaces (n = 240)", zorder=6)
    right.scatter(preset_natural.surface_open_area_fraction,
                  preset_natural.maximum_complete_pore_equivalent_radius_mm,
                  s=75, marker="D", color="#0E6655", edgecolor="white", linewidths=.7,
                  label="Eligible preset natural surface (n = 1)", zorder=7)
    right.scatter(pc_surface.liquid_open_area_fraction,
                  pc_surface.maximum_complete_pore_radius_mm,
                  s=45, color=BLUE, edgecolor="white", lw=0.7,
                  label="PC-AEGG surface runs", zorder=5)
    right.axvline(pc_surface.liquid_open_area_fraction.mean(),
                  ls="--", color=BLUE, lw=0.9)
    right.axhline(pc_surface.maximum_complete_pore_radius_mm.mean(),
                  ls="--", color=BLUE, lw=0.9)
    right.set(xlabel="Liquid open-area fraction", ylabel="Controlling equivalent radius (mm)",
              title="e  Natural surfaces and slit-only reference")
    right.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.96, fontsize=6.1, loc="lower right")
    from matplotlib.ticker import MaxNLocator
    for ax in (left,right):
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=6.6)
        ax.xaxis.label.set_size(7.5)
        ax.yaxis.label.set_size(7.5)
        ax.title.set_size(8.5)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"Figure2_de.{ext}",
                    dpi=350, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(OUT))
    main(parser.parse_args().out)
