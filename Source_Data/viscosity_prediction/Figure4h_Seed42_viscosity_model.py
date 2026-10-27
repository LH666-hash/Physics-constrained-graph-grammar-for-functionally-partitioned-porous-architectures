"""Rebuild Fig. 4h with the current slice-wise reduced-order filling model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.0,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.75,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def load_model(model_dir: Path):
    path = model_dir / "capillary_viscous_filling.py"
    spec = importlib.util.spec_from_file_location("capillary_viscous_filling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_experiment(path: Path) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    frame = pd.read_excel(path)
    time_s = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    columns = {1.0: 2, 5.0: 3, 15.0: 4, 25.0: 5, 50.0: 1}
    values = {
        viscosity: pd.to_numeric(frame.iloc[:, column], errors="coerce").to_numpy(float)
        for viscosity, column in columns.items()
    }
    return time_s, values


def raw_profile(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    z_col = "z_mid_mm" if "z_mid_mm" in frame.columns else "z_mm"
    required = [
        z_col, "A_liq_mm2", "axial_radius4_sum_mm4", "P_cap_Pa",
        "P_hydro_Pa", "safety_factor_z",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing profile columns: {missing}")
    out = frame[required].copy().rename(columns={z_col: "z_mid_mm"})
    out["z_mm"] = out["z_mid_mm"]
    return out.sort_values("z_mid_mm").drop_duplicates("z_mid_mm")


def relative_time_profile(model, profile: pd.DataFrame, viscosity_cp: float,
                          resistance_factor: float, initial_height_mm: float) -> tuple[np.ndarray, np.ndarray]:
    evaluated = model.apply_state_updated_viscous_profile(
        profile,
        viscosity_pa_s=viscosity_cp * 1.0e-3,
        reference_flow_m3_s=2.0e-7,
        default_dz_mm=float(np.median(np.diff(profile["z_mid_mm"]))),
        target_fill_time_s=0.0,
    )
    z = evaluated["z_mid_mm"].to_numpy(float)
    dz_m = evaluated["slice_dz_mm"].to_numpy(float) * 1.0e-3
    volume_m3 = evaluated["A_liq_mm2"].to_numpy(float) * 1.0e-6 * dz_m
    flow_m3_s = evaluated["Q_available_proxy_m3_s"].to_numpy(float)
    increments = resistance_factor * volume_m3 / np.maximum(flow_m3_s, 1.0e-300)
    cumulative = np.cumsum(increments)
    if not np.all(np.isfinite(cumulative)):
        finite = np.isfinite(cumulative)
        z, cumulative = z[finite], cumulative[finite]
    base = float(np.interp(initial_height_mm, z, cumulative))
    keep = z >= initial_height_mm
    return z[keep], np.maximum(cumulative[keep] - base, 0.0)


def fit_resistance_factor(model, profile: pd.DataFrame, time_s: np.ndarray,
                          height_mm: np.ndarray, initial_height_mm: float,
                          sample_height_mm: float) -> tuple[float, int]:
    z, unit_time = relative_time_profile(model, profile, 1.0, 1.0, initial_height_mm)
    assert np.all(np.diff(z) > 0.0), "Slice heights must be strictly increasing."
    valid = (
        np.isfinite(time_s) & np.isfinite(height_mm) & (time_s > 0.0)
        & (height_mm > initial_height_mm) & (height_mm < sample_height_mm)
    )
    if valid.sum() < 2:
        raise ValueError("At least two uncensored 1 cP calibration points are required.")
    geometric_time = np.interp(height_mm[valid], z, unit_time)
    observed_time = time_s[valid]
    factor = float(np.dot(geometric_time, observed_time) / np.dot(geometric_time, geometric_time))
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("Invalid fitted resistance factor.")
    return factor, int(valid.sum())


def height_at_time(model, profile: pd.DataFrame, viscosity_cp: float,
                   resistance_factor: float, time_s: float,
                   initial_height_mm: float, sample_height_mm: float) -> float:
    z, relative_time = relative_time_profile(
        model, profile, viscosity_cp, resistance_factor, initial_height_mm
    )
    assert np.all(np.diff(z) > 0.0), "Slice heights must be strictly increasing."
    assert np.all(np.diff(relative_time) >= 0.0), "Cumulative filling time must be monotone."
    if time_s >= relative_time[-1]:
        return float(sample_height_mm)
    return float(np.interp(time_s, relative_time, z, left=initial_height_mm))


def draw_panel(data: pd.DataFrame, output_dir: Path) -> None:
    colors = ["#2B6EA6", "#4E91BF", "#7B8FB2", "#A66E82"]
    x = np.arange(len(data), dtype=float)
    fig, ax = plt.subplots(figsize=(3.5039, 2.6378))  # 89 × 67 mm

    for i, row in data.iterrows():
        color = colors[i]
        ax.plot([x[i], x[i]], [row.observed_height_mm, row.predicted_height_mm],
                color=color, lw=1.4, zorder=1)
        ax.scatter(x[i], row.predicted_height_mm, s=30, facecolor="white",
                   edgecolor=color, linewidth=1.15, zorder=3,
                   label="Slice-model prediction" if i == 0 else None)
        ax.scatter(x[i], row.observed_height_mm, s=26, marker="s", color=color,
                   zorder=3, label="Experiment" if i == 0 else None)

    ax.axhline(25.0, color="#A7ADB3", lw=0.8, ls=(0, (3, 2)))
    ax.text(4.42, 25.16, "Sample top", ha="right", va="bottom", color="#6F777D", fontsize=6.0)
    ax.set_xticks(x, [f"{value:g}" for value in data["viscosity_cP"]])
    ax.set_xlabel("Viscosity (cP)")
    ax.set_ylabel("Rise height at 55 s (mm)")
    ax.set_ylim(10, 26.5)
    ax.set_yticks([10, 15, 20, 25])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.7)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
              handletextpad=0.35, columnspacing=1.0, borderaxespad=0.0)
    ax.text(-0.15, 1.08, "h", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.84)

    stem = output_dir / "Figure4h_Seed42_viscosity_prediction"
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--initial-height-mm", type=float, default=3.0)
    parser.add_argument("--sample-height-mm", type=float, default=25.0)
    parser.add_argument("--observation-time-s", type=float, default=55.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_dir)
    profile = raw_profile(args.profile)
    time_obs, experiment = read_experiment(args.experiment)
    factor, n_calibration = fit_resistance_factor(
        model, profile, time_obs, experiment[1.0],
        args.initial_height_mm, args.sample_height_mm,
    )

    rows = []
    for viscosity in (5.0, 15.0, 25.0, 50.0):
        prediction = height_at_time(
            model, profile, viscosity, factor, args.observation_time_s,
            args.initial_height_mm, args.sample_height_mm,
        )
        observed = experiment[viscosity]
        valid = np.isfinite(time_obs) & np.isfinite(observed) & (time_obs <= args.observation_time_s)
        endpoint = float(observed[np.flatnonzero(valid)[-1]])
        rows.append({
            "viscosity_cP": viscosity,
            "role": "prediction",
            "observation_time_s": args.observation_time_s,
            "predicted_height_mm": prediction,
            "observed_height_mm": endpoint,
            "residual_model_minus_experiment_mm": prediction - endpoint,
            "absolute_error_mm": abs(prediction - endpoint),
            "top_censored_experiment": endpoint >= args.sample_height_mm,
        })

    endpoints = pd.DataFrame(rows)
    endpoints.to_csv(args.out_dir / "Figure4h_Seed42_viscosity_source_data.csv", index=False)
    prediction_rows = endpoints
    summary = {
        "model": "seed 42 full post-assembly slice-wise reduced-order filling model",
        "structure_input": "Representative_Structure_Seed42/post_assembly_pressure_profile.csv",
        "structure_input_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "experimental_input": "Figure4h_Seed42_experimental_rise.xlsx",
        "experimental_input_sha256": hashlib.sha256(args.experiment.read_bytes()).hexdigest(),
        "physical_model_sha256": hashlib.sha256(
            (args.model_dir / "capillary_viscous_filling.py").read_bytes()
        ).hexdigest(),
        "resistance_factor_fitted_from_1cP": factor,
        "uncensored_1cP_calibration_points": n_calibration,
        "observation_time_s": args.observation_time_s,
        "out_of_sample_mae_mm_5_to_50cP": float(prediction_rows["absolute_error_mm"].mean()),
        "out_of_sample_max_abs_error_mm_5_to_50cP": float(prediction_rows["absolute_error_mm"].max()),
        "top_censoring_note": "A measured value at 25 mm means the liquid reached the sample top.",
        "interpretation": "Calibrated reduced-order prediction; not a pore-network front simulation.",
    }
    (args.out_dir / "Figure4h_Seed42_viscosity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draw_panel(endpoints, args.out_dir)
    print(endpoints.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
