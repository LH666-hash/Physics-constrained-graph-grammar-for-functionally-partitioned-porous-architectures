"""Plot only centrifugal observations actually tabulated in Supplementary Information."""
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from PIL import Image

PKG = Path(__file__).resolve().parents[2]
SI = PKG / "Supplementary_Information/Supplementary_Information.docx"
RAW = PKG / "Source_Data/Raw_Data"
FIG = PKG / "Figures/Main_Figures"
RAW.mkdir(exist_ok=True)


def main() -> None:
    table = RAW / "centrifugal_si_tables_S8_S9.csv"
    if SI.exists():
        doc = Document(SI)
        pressure = []
        for row in doc.tables[7].rows[1:]:
            c = [x.text.strip() for x in row.cells]
            pressure.append(dict(record_type="theory_grid", speed_rpm=float(c[0]),
                                 centrifugal_pressure_pa=float(c[1]), total_pressure_pa=float(c[2]),
                                 critical_radius_mm=float(c[3]), specimen_radius_mm=np.nan,
                                 first_leak_speed_rpm=np.nan, first_leak_mass_loss_pct=np.nan,
                                 theory_threshold_rpm=np.nan, source_table="S8"))
        first = []
        for row in doc.tables[8].rows[1:]:
            c = [x.text.strip().replace("%", "") for x in row.cells]
            first.append(dict(record_type="first_detected_loss", speed_rpm=float(c[2]),
                              centrifugal_pressure_pa=np.nan, total_pressure_pa=np.nan,
                              critical_radius_mm=np.nan, specimen_radius_mm=float(c[0]),
                              first_leak_speed_rpm=float(c[2]), first_leak_mass_loss_pct=float(c[3]),
                              theory_threshold_rpm=float(c[1]), source_table="S9"))
        data = pd.DataFrame(pressure + first)
        data.to_csv(table, index=False, float_format="%.10g")
    else:
        data = pd.read_csv(table)
        first = data.loc[data.record_type.eq("first_detected_loss")].to_dict("records")
    rho, gamma, theta, gravity, height, radius = 998., .072, math.radians(15), 9.80665, .025, .01
    rpm = np.linspace(0, 1000, 500)
    p = rho * gravity * height + .5 * rho * (2 * np.pi * rpm / 60 * radius) ** 2
    rcrit = 2 * gamma * math.cos(theta) / p * 1000
    observed = pd.DataFrame(first)
    colors = {0.5: "#2171B5", 0.4: "#8B83AF", 0.3: "#C56C83"}
    markers = {0.5: "*", 0.4: "D", 0.3: "P"}
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
    ax = axes[0]
    for row in observed.itertuples(index=False):
        ax.scatter(row.first_leak_speed_rpm, row.first_leak_mass_loss_pct, s=65,
                   color=colors[row.specimen_radius_mm], marker=markers[row.specimen_radius_mm],
                   edgecolor="black", linewidth=.5, label=f"{row.specimen_radius_mm:.1f} mm")
    ax.set(xlim=(0, 830), ylim=(0, 20), xlabel="Rotation speed (rpm)",
           ylabel="Mass loss at first detection (%)", title="h  First detected loss")
    ax.legend(frameon=False, title="Equivalent radius")
    ax.text(.02, .97, "One reported point per radius", transform=ax.transAxes, va="top", fontsize=7)
    ax = axes[1]
    ax.plot(rpm, rcrit, color="#183B56", lw=1.6, label="Model")
    for row in observed.itertuples(index=False):
        ax.axhline(row.specimen_radius_mm, color=colors[row.specimen_radius_mm], lw=.7, ls="--", alpha=.7)
        ax.scatter(row.theory_threshold_rpm, row.specimen_radius_mm, marker=markers[row.specimen_radius_mm],
                   s=40, color=colors[row.specimen_radius_mm], edgecolor="black", linewidth=.4)
    ax.set(xlim=(0, 1000), ylim=(0, .62), xlabel="Rotation speed (rpm)",
           ylabel="Critical equivalent radius (mm)", title="i  Calculated stability boundary")
    ax.legend(frameon=False)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"Figure3_hi_revised.{ext}", dpi=350, facecolor="white")
    plt.close(fig)
    # Preserve original panels a–g and replace the lower-right h–i region.
    original = Image.open(RAW / "Figure3_base_image.png").convert("RGBA")
    white = Image.new("RGBA", original.size, "white")
    white.alpha_composite(original)
    revised = Image.open(FIG / "Figure3_hi_revised.png").convert("RGBA")
    revised.thumbnail((1460, 630), Image.Resampling.LANCZOS)
    white.paste(Image.new("RGBA", (1530, 700), "white"), (930, 1730))
    white.paste(revised, (990 + (1460 - revised.width) // 2, 1760 + (630 - revised.height) // 2))
    white.convert("RGB").save(FIG / "Figure3_revised.png", dpi=(350, 350))
    print("Saved SI S8/S9 data and Figure3 h–i; no unreported loss trajectory was interpolated.")


if __name__ == "__main__":
    main()
