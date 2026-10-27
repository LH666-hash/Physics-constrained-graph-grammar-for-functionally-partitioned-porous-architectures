"""Structure-level pore morphology for final same-seed surface ablation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PACKAGE = Path(__file__).resolve().parents[2]
DATA = PACKAGE / "Source_Data/Ablation_Analysis/表面"
REF = PACKAGE / "Source_Data/Benchmark_Analysis/表面/surface_only_reference_comparison.csv"
OUT = PACKAGE / "Figures/Supplementary_Figures"
COLORS = {"surface_full":"#2171B5", "surface_random_pore":"#DDA7B0", "surface_none":"#8B6FA8"}
LABELS = {"surface_full":"Full PC-AEGG", "surface_random_pore":"Random pore", "surface_none":"No update"}

def main():
    OUT.mkdir(exist_ok=True)
    pores = pd.read_csv(DATA / "surface_ablation_group_seed_complete_pore_contours.csv")
    runs = pd.read_csv(DATA / "surface_ablation_group_seed_final_graph_metrics.csv")
    ref = pd.read_csv(REF)
    weighted = (pores.groupby(["group","seed"]).apply(
        lambda x: pd.Series(dict(
            aspect=np.average(x.aspect_ratio_3d_pca, weights=x.area_mm2),
            shape=np.average(x.shape_factor, weights=x.area_mm2))),
        include_groups=False).reset_index())
    weighted = weighted.merge(runs[["group","seed","liquid_open_area_fraction"]], on=["group","seed"])
    weighted.to_csv(DATA / "surface_ablation_group_seed_area_weighted_shape_metrics.csv", index=False)
    plt.rcParams.update({"font.family":"Arial","font.size":8,"axes.labelsize":8.5,
                         "axes.titlesize":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"svg.fonttype":"none"})
    fig, axs = plt.subplots(1,2,figsize=(17.5/2.54,7.0/2.54),layout="constrained")
    a,b=axs
    a.scatter(ref.pore_aspect_ratio,ref.pore_shape_factor,s=2.4,
              alpha=.14,color="#AAB5D1",linewidths=0,rasterized=True,
              label="0.30 mm slit surfaces (n = 8,212)")
    b.scatter(ref.surface_A_eff_star,ref.pore_aspect_ratio,s=2.4,
              alpha=.14,color="#AAB5D1",linewidths=0,rasterized=True)
    for group in COLORS:
        row=weighted[weighted.group.eq(group)]
        a.scatter(row.aspect,row["shape"],s=24,color=COLORS[group],edgecolor="white",
                  linewidth=.5,label=f"{LABELS[group]} (n = 5)",zorder=4)
        b.scatter(row.liquid_open_area_fraction,row.aspect,s=24,color=COLORS[group],
                  edgecolor="white",linewidth=.5,zorder=4)
    a.set(xlabel="Area-weighted mean pore aspect ratio",ylabel="Area-weighted mean pore shape factor",
          title="a  Pore morphology by structure",xlim=(1,12),ylim=(0.2,.85))
    b.set(xlabel="Liquid open-area fraction",ylabel="Area-weighted mean pore aspect ratio",
          title="b  Openness and pore morphology",xlim=(.13,.95),ylim=(1,12.2))
    a.legend(frameon=False,fontsize=5.8,loc="upper right")
    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=7)
    for ext in ("png","pdf","svg"):
        fig.savefig(OUT/f"FigureS20.{ext}",dpi=350,facecolor="white")
    plt.close(fig)

if __name__=="__main__":
    main()
