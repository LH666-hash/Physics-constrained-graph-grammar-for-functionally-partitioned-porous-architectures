"""Six-panel manuscript figure matching the established ablation visual grammar."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
SURF = PACKAGE / "Source_Data/Ablation_Analysis/表面"
INTR = PACKAGE / "Source_Data/Ablation_Analysis/内部"
OUT = PACKAGE / "Figures/Main_Figures"
S_ORDER = ("full", "random_pore", "none")
S_COLOR = {"full": "#145D93", "random_pore": "#B85E78", "none": "#72538F"}
S_LABEL = {"full": "Full surface rules", "random_pore": "Random pore, same rule",
           "none": "No surface update"}
I_ORDER = ("Full PC-AEGG", "Fully random", "Initial gradient skeleton",
           "Matched random position")
I_COLOR = {"Full PC-AEGG": "#145D93", "Fully random": "#B85E78",
           "Initial gradient skeleton": "#72538F", "Matched random position": "#B97516"}


def mean_sd_errorbar(ax, x, y, color, marker="o", size=8, zorder=5):
    """Show five independent runs and their arithmetic mean ± sample SD."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.errorbar(x.mean(), y.mean(), xerr=x.std(ddof=1), yerr=y.std(ddof=1),
                fmt=marker, markersize=size, color=color, ecolor=color,
                markeredgecolor="white", markeredgewidth=0.8, elinewidth=1.55,
                capsize=3.5, capthick=1.3, zorder=zorder)


def box_cloud(ax, position, values, color, jitter):
    values = np.asarray(values, dtype=float)
    bp = ax.boxplot(values, positions=[position], widths=0.42, patch_artist=True,
                    showfliers=False, medianprops={"color": "#25303A", "linewidth": 1.5},
                    boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color,
                              "linewidth": 1.3},
                    whiskerprops={"color": color, "linewidth": 1.3},
                    capprops={"color": color, "linewidth": 1.3})
    ax.scatter(position + jitter, values, s=29, color=color, edgecolor="white",
               linewidth=0.6, zorder=4)
    ax.errorbar(position, values.mean(), yerr=values.std(ddof=1), fmt="D",
                color=color, markeredgecolor="white", markeredgewidth=0.6,
                markersize=5.2, capsize=3.5, elinewidth=1.35, zorder=5)
    return bp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 7.8,
                         "axes.labelsize": 7.8, "axes.titlesize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic"})
    surface = pd.read_csv(SURF / "surface_ablation_group_seed_final_graph_metrics.csv")
    surface["method"] = surface.group.str.replace("surface_", "", regex=False)
    internal = pd.read_csv(INTR / "internal_metrics_fixed_budget_four_groups.csv")
    profiles = pd.read_csv(INTR / "internal_profiles_fixed_budget_four_groups.csv")
    assert set(surface.method) == set(S_ORDER)
    assert surface.groupby("method").size().eq(5).all()
    assert surface.final_surface_graph_nodes.gt(0).all()
    assert internal.groupby("method").size().eq(5).all()
    assert internal.evaluation_scope.str.startswith("assembled_").all()
    jitter = np.array([-0.13, -0.065, 0, 0.065, 0.13])
    fig, axs = plt.subplots(2, 3, figsize=(17.5 / 2.54, 12.0 / 2.54), layout="constrained")
    a, b, c, d, e, f = axs.flat

    trajectory = pd.read_csv(SURF / "surface_ablation_group_round_mean_sd_n.csv")
    assert trajectory["count"].eq(5).all()
    longest = int(trajectory.iteration.max())
    for method in S_ORDER[:2]:
        summary = trajectory[trajectory.group.eq(f"surface_{method}")]
        x = summary.iteration.to_numpy(dtype=float)
        mean = summary["mean"].to_numpy(dtype=float)
        sd = summary["std"].to_numpy(dtype=float)
        a.fill_between(x, mean - sd, mean + sd, color=S_COLOR[method], alpha=0.20,
                       linewidth=0, zorder=1)
        a.plot(x, mean, color=S_COLOR[method], lw=2.6,
               label=f"{S_LABEL[method]} (n = 5)", zorder=3)
    no_ratios = []
    observed = pd.read_csv(SURF / "surface_ablation_group_seed_observed_refinement_trajectory.csv")
    for seed in surface[surface.method.eq("none")].seed:
        no_ratios.append(float(observed.loc[
            observed.group.eq("surface_none") & observed.seed.eq(seed),
            "maximum_local_radius_ratio"].iloc[0]))
    a.errorbar(1, np.mean(no_ratios), yerr=np.std(no_ratios, ddof=1), fmt="D",
               color=S_COLOR["none"], markersize=5, capsize=3,
               label="No update (iteration 1 only)", zorder=4)
    a.axhline(1, color="#65717C", ls=":", lw=1.0)
    a.set(xlabel="Surface refinement iteration", ylabel="Maximum radius / critical radius",
          xlim=(1, longest), title="a  Surface refinement trajectories")
    a.legend(frameon=False, fontsize=5.6, loc="upper right")

    for method in S_ORDER[:2]:
        rows = surface[surface.method.eq(method)].sort_values("seed")
        b.scatter(rows.liquid_open_area_fraction,
                  rows.complete_pore_satisfaction_fraction * 100,
                  s=38, color=S_COLOR[method], edgecolor="white", linewidth=0.7,
                  label=S_LABEL[method], zorder=3)
        mean_sd_errorbar(b, rows.liquid_open_area_fraction,
                         rows.complete_pore_satisfaction_fraction * 100,
                         S_COLOR[method], size=7.2)
    main_rows = surface[surface.method.isin(S_ORDER[:2])]
    xmin, xmax = main_rows.liquid_open_area_fraction.min(), main_rows.liquid_open_area_fraction.max()
    b.set(xlim=(xmin - 0.025, xmax + 0.025), ylim=(86, 102),
          xlabel="Liquid open-area fraction", ylabel="Complete pores within radius limit (%)",
          title="b  Surface acceptance")
    b.axhline(100, color="#65717C", ls="--", lw=0.9)
    inset = b.inset_axes([0.61, 0.62, 0.35, 0.31])
    no_rows = surface[surface.method.eq("none")]
    inset.scatter(no_rows.liquid_open_area_fraction,
                  no_rows.complete_pore_satisfaction_fraction * 100,
                  s=19, color=S_COLOR["none"], edgecolor="white", lw=0.4)
    mean_sd_errorbar(inset, no_rows.liquid_open_area_fraction,
                     no_rows.complete_pore_satisfaction_fraction * 100,
                     S_COLOR["none"], size=5.3)
    inset.set(xlim=(no_rows.liquid_open_area_fraction.min() - 0.004,
                    no_rows.liquid_open_area_fraction.max() + 0.004),
              ylim=(0, max(20, no_rows.complete_pore_satisfaction_fraction.max() * 115)),
              title="No update")
    inset.set_xticks([0.92, 0.93])
    inset.set_yticks([0, 10, 20])
    inset.tick_params(labelsize=5.5)
    inset.title.set_fontsize(6)

    for i, method in enumerate(S_ORDER):
        rows = surface[surface.method.eq(method)].sort_values("seed")
        box_cloud(c, i, rows.final_surface_graph_nodes, S_COLOR[method], jitter)
    c.set(ylabel="Final surface graph nodes", title="c  Final surface graph size")
    c.set_xticks(range(3), ["Full", "Random\npore", "No update"])

    for i, method in enumerate(I_ORDER):
        rows = internal[internal.method.eq(method)].sort_values("seed")
        sub = profiles[profiles.method.eq(method)]
        center = (sub.groupby("z_mid_mm").Transport_Margin_Pa
                  .agg(["mean", "std", "count"]).reset_index())
        assert center["count"].eq(5).all(), f"Incomplete profile: {method}"
        x = center.z_mid_mm.to_numpy(dtype=float)
        mean = center["mean"].to_numpy(dtype=float)
        sd = center["std"].to_numpy(dtype=float)
        d.fill_between(x, mean - sd, mean + sd, color=I_COLOR[method],
                       alpha=0.18, linewidth=0, zorder=1)
        d.plot(x, mean, color=I_COLOR[method], lw=2.35, label=method, zorder=3)
        e.scatter(rows.weighted_porosity_fraction * 100,
                  rows.minimum_transport_margin_pa,
                  s=24, color=I_COLOR[method], edgecolor="white", linewidth=0.5,
                  alpha=0.65)
        mean_sd_errorbar(e, rows.weighted_porosity_fraction * 100,
                         rows.minimum_transport_margin_pa, I_COLOR[method],
                         size=7.3)
        box_cloud(f, i, rows.final_internal_nodes, I_COLOR[method], jitter)
    d.axhline(0, color="#737373", ls="--", lw=0.9)
    e.axhline(0, color="#737373", ls="--", lw=0.9)
    d.set(xlabel="Height (mm)", ylabel=r"Transport margin $R_{\mathrm{tr}}$ (Pa)",
          title="d  Assembled transport")
    d.legend(frameon=False, fontsize=5.2, loc="upper right")
    e.set(xlabel="Assembled porosity (%)", ylabel=r"Minimum transport margin $R_{\mathrm{tr}}$ (Pa)",
          title="e  Assembled porosity–transport")
    f.set(ylabel="Final internal graph nodes", title="f  Internal structural budget")
    f.set_xticks(range(4), ["Full", "Fully\nrandom", "Initial\ngradient", "Random\nposition"])

    from matplotlib.ticker import MaxNLocator
    for ax in (a, b, c, d, e, f):
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=6.6)
    c.set_xticks(range(3), ["Full", "Random\npore", "No update"])
    f.set_xticks(range(4), ["Full", "Fully\nrandom", "Initial\ngradient", "Random\nposition"])
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Figure5.{extension}",
                    dpi=350, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

