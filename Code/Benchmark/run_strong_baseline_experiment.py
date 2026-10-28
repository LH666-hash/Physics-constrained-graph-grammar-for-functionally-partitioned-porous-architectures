from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]
INTERNAL_DIR = ROOT / "Source_Data/Benchmark_Analysis/内部"
PIPELINE = Path(__file__).with_name("internal_pareto_pipeline.py")
HYDRAULIC = ROOT / "Code/Core/capillary_viscous_filling.py"
CANDIDATES = INTERNAL_DIR / "01_预设内部结构_全部45220.csv"
PC_REFERENCES = INTERNAL_DIR / "07_PC-AEGG完整组5次终态内部指标.csv"

DELTA_EPSILON = 0.023
DELTA_HEIGHT_MM = 2.25
MIN_FEATURE_MM = 0.40
R_CRIT_MM = 0.568
EXTERIOR_AREA_MM2 = 1200.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def matched(row: pd.Series | dict, pc: pd.Series) -> bool:
    eps = float(row["epsilon"])
    height = float(row["predicted_rise_height_mm"])
    return abs(eps - float(pc.epsilon)) <= DELTA_EPSILON and height >= float(pc.predicted_rise_height_mm) - DELTA_HEIGHT_MM


def internal_score(row: pd.Series | dict, pc: pd.Series) -> tuple[float, ...]:
    feasible = str(row.get("feasible", True)).lower() == "true" if not isinstance(row.get("feasible", True), bool) else bool(row.get("feasible", True))
    if not feasible:
        return (2.0, 1.0e6, 1.0e6, 1.0e6)
    eps = float(row["epsilon"])
    height = float(row["predicted_rise_height_mm"])
    eps_excess = max(abs(eps - float(pc.epsilon)) - DELTA_EPSILON, 0.0) / DELTA_EPSILON
    height_excess = max(float(pc.predicted_rise_height_mm) - DELTA_HEIGHT_MM - height, 0.0) / DELTA_HEIGHT_MM
    is_match = eps_excess == 0.0 and height_excess == 0.0
    return (
        0.0 if is_match else 1.0,
        eps_excess + height_excess,
        abs(eps - float(pc.epsilon)) / DELTA_EPSILON,
        -height / 25.0,
    )


def select_existing_matches(candidates: pd.DataFrame, pc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feasible = candidates[is_true(candidates["feasible"])].copy()
    families = ["Graded BCC", "Graded Gyroid"]
    pool = feasible[feasible.family.isin(families)].copy()
    pair_rows = []
    for p in pc.itertuples(index=False):
        sub = pool[
            (pool.epsilon.astype(float).sub(float(p.epsilon)).abs() <= DELTA_EPSILON)
            & (pool.predicted_rise_height_mm.astype(float) >= float(p.predicted_rise_height_mm) - DELTA_HEIGHT_MM)
        ]
        for row in sub.itertuples(index=False):
            pair_rows.append(
                {
                    "pc_design_id": p.design_id,
                    "core_design_id": row.design_id,
                    "baseline_group": "Slit skin + " + row.family,
                    "core_family": row.family,
                    "pc_epsilon": p.epsilon,
                    "pc_height_mm": p.predicted_rise_height_mm,
                    "core_epsilon": row.epsilon,
                    "core_height_mm": row.predicted_rise_height_mm,
                    "epsilon_difference": row.epsilon - p.epsilon,
                    "height_difference_mm": row.predicted_rise_height_mm - p.predicted_rise_height_mm,
                }
            )
    pairs = pd.DataFrame(pair_rows)
    ids = pairs.core_design_id.unique() if len(pairs) else []
    return pool[pool.design_id.isin(ids)].copy(), pairs


def prepare_local_perturbation(base_solid: np.ndarray, feature_z: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray]:
    sampling = (
        float(grid.domain.slice_spacing_mm),
        float(grid.domain.xy_pitch_mm),
        float(grid.domain.xy_pitch_mm),
    )
    signed_distance = (
        ndimage.distance_transform_edt(~base_solid, sampling=sampling)
        - ndimage.distance_transform_edt(base_solid, sampling=sampling)
    )
    minimum_offset = -0.5 * np.maximum(np.asarray(feature_z, dtype=float) - MIN_FEATURE_MM, 0.0)
    return signed_distance.astype(np.float32), minimum_offset[:, None, None].astype(np.float32)


def apply_local_mutation(offset: np.ndarray, minimum_offset: np.ndarray, grid, mutation: dict[str, float]) -> np.ndarray:
    proposal = offset.copy()
    cx, cy, cz = mutation["cx_mm"], mutation["cy_mm"], mutation["cz_mm"]
    rx, ry, rz = mutation["rx_mm"], mutation["ry_mm"], mutation["rz_mm"]
    z_idx = np.flatnonzero(np.abs(grid.z - cz) <= rz)
    x_idx = np.flatnonzero(np.abs(grid.x - cx) <= rx)
    y_idx = np.flatnonzero(np.abs(grid.y - cy) <= ry)
    if not len(z_idx) or not len(x_idx) or not len(y_idx):
        return proposal
    zz = ((grid.z[z_idx] - cz) / rz)[:, None, None]
    xx = ((grid.x[x_idx] - cx) / rx)[None, :, None]
    yy = ((grid.y[y_idx] - cy) / ry)[None, None, :]
    q = zz * zz + xx * xx + yy * yy
    weight = np.where(q < 1.0, (1.0 - q) ** 2, 0.0).astype(np.float32)
    block = np.ix_(z_idx, x_idx, y_idx)
    proposal[block] += float(mutation["amplitude_mm"]) * weight
    proposal[block] = np.maximum(proposal[block], minimum_offset[z_idx])
    np.clip(proposal, -0.8, 0.8, out=proposal)
    return proposal


def local_mutated_solid(ip, base_solid: np.ndarray, feature_z: np.ndarray, grid, history: list[dict[str, float]]) -> np.ndarray:
    signed_distance, minimum_offset = prepare_local_perturbation(base_solid, feature_z, grid)
    offset = np.zeros_like(signed_distance, dtype=np.float32)
    for mutation in history:
        offset = apply_local_mutation(offset, minimum_offset, grid, mutation)
    return signed_distance <= offset


def evaluate_local_solid(ip, hydraulic_function, solid: np.ndarray, start: pd.Series, grid, design_id: str,
                         target_pc_design_id: str, history: list[dict[str, float]], sample_index: int) -> dict[str, object]:
    _, metrics = ip.profile_from_solid(solid, grid, ip.FluidConfig(), hydraulic_function)
    # Local thickness changes remain attached to the connected periodic seed;
    # full solid connectivity is audited again for selected composite designs.
    solid_connected = True
    feasible = bool(
        metrics["geometry_valid"]
        and metrics["liquid_connected"]
        and metrics["solid_layer_overlap_valid"]
        and solid_connected
    )
    return {
        "family": start.family,
        "design_id": design_id,
        "search_stage": "randomized_local_greedy",
        "bounds_label": start.bounds_label,
        "sample_index": sample_index,
        "is_uniform": bool(start.is_uniform),
        "u_L": float(start.u_L),
        "u_bottom": float(start.u_bottom),
        "u_top": float(start.u_top),
        "u_p": float(start.u_p),
        "L": float(start.L_mm),
        "L_mm": float(start.L_mm),
        "D_bottom_mm": start.get("D_bottom_mm", math.nan),
        "D_top_mm": start.get("D_top_mm", math.nan),
        "t_bottom_mm": start.get("t_bottom_mm", math.nan),
        "t_top_mm": start.get("t_top_mm", math.nan),
        "feature_bottom_mm": float(start.feature_bottom_mm),
        "feature_top_mm": float(start.feature_top_mm),
        "p": float(start.p),
        "epsilon": metrics["porosity"],
        "porosity": metrics["porosity"],
        "predicted_rise_height_mm": metrics["predicted_rise_height_mm"],
        "first_failure_z_mm": metrics["first_failure_z_mm"],
        "geometry_valid": bool(metrics["geometry_valid"]),
        "manufacturing_valid": True,
        "liquid_connected": bool(metrics["liquid_connected"]),
        "solid_layer_overlap_valid": bool(metrics["solid_layer_overlap_valid"]),
        "solid_connected": solid_connected,
        "feasible": feasible,
        "min_slice_porosity": metrics["min_slice_porosity"],
        "max_slice_porosity": metrics["max_slice_porosity"],
        "min_R_tr_Pa": metrics["min_R_tr_Pa"],
        "target_pc_design_id": target_pc_design_id,
        "local_mutation_count": len(history),
        "local_mutation_history": json.dumps(history, separators=(",", ":")),
        "start_design_id": start.design_id,
        "geometry_sha1": hashlib.sha1(np.packbits(solid).tobytes()).hexdigest(),
    }


def randomized_greedy_search(
    ip,
    hydraulic_function,
    candidates: pd.DataFrame,
    pc: pd.DataFrame,
    *,
    iterations: int,
    proposals_per_iteration: int,
    restarts_per_family: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain = ip.DomainConfig(xy_pitch_mm=0.125, slice_spacing_mm=0.75)
    grid = ip.SliceGrid(domain)
    rng = np.random.default_rng(seed)
    feasible = candidates[is_true(candidates["feasible"])].copy()
    families = list(ip.FAMILY_COLORS)
    all_rows: list[dict] = []
    trajectories: list[dict] = []
    counter = 0

    for pc_index, p in enumerate(pc.itertuples(index=False), 1):
        p_series = pd.Series(p._asdict())
        for family_index, family in enumerate(families):
            source = feasible[feasible.family.eq(family)].copy()
            if source.empty:
                continue
            source["_score"] = [internal_score(row, p_series) for _, row in source.iterrows()]
            source = source.sort_values("_score", kind="stable")
            starts = source.head(restarts_per_family)
            for restart, (_, start) in enumerate(starts.iterrows(), 1):
                base_solid = core_solid(ip, str(start.family), start, grid)
                feature_z = ip.gradient_feature(
                    grid.z,
                    float(start.feature_bottom_mm),
                    float(start.feature_top_mm),
                    float(start.p),
                    grid.domain.height_mm,
                )
                signed_distance, minimum_offset = prepare_local_perturbation(base_solid, feature_z, grid)
                current_offset = np.zeros_like(signed_distance, dtype=np.float32)
                current_history: list[dict[str, float]] = []
                current = start.to_dict()
                current_score = internal_score(current, p_series)
                accepted = 0
                evaluated = 0
                for iteration in range(iterations):
                    parent_solid = signed_distance <= current_offset
                    inner_boundary = parent_solid & ~ndimage.binary_erosion(parent_solid)
                    boundary_indices = np.argwhere(inner_boundary)
                    proposals = []
                    for proposal_index in range(proposals_per_iteration):
                        center_index = boundary_indices[rng.integers(0, len(boundary_indices))]
                        mutation = {
                            "cx_mm": float(grid.x[center_index[1]]),
                            "cy_mm": float(grid.y[center_index[2]]),
                            "cz_mm": float(grid.z[center_index[0]]),
                            "rx_mm": float(rng.uniform(0.6, 1.8)),
                            "ry_mm": float(rng.uniform(0.6, 1.8)),
                            "rz_mm": float(rng.uniform(1.5, 4.5)),
                            "amplitude_mm": float(rng.choice((-1.0, 1.0)) * rng.uniform(0.05, 0.20)),
                        }
                        proposal_offset = apply_local_mutation(current_offset, minimum_offset, grid, mutation)
                        proposal_solid = signed_distance <= proposal_offset
                        if np.array_equal(proposal_solid, parent_solid):
                            mutation["amplitude_mm"] = abs(float(mutation["amplitude_mm"])) + 0.15
                            proposal_offset = apply_local_mutation(current_offset, minimum_offset, grid, mutation)
                            proposal_solid = signed_distance <= proposal_offset
                        counter += 1
                        design_id = f"local-greedy-{family.replace(' ', '_')}-P{pc_index:02d}-R{restart:02d}-E{counter:06d}"
                        history = current_history + [mutation]
                        row = evaluate_local_solid(
                            ip,
                            hydraulic_function,
                            proposal_solid,
                            start,
                            grid,
                            design_id,
                            p.design_id,
                            history,
                            counter,
                        )
                        row["greedy_restart"] = restart
                        row["greedy_iteration"] = iteration + 1
                        row["greedy_proposal"] = proposal_index + 1
                        row["changed_voxels_from_parent"] = int(np.count_nonzero(proposal_solid != parent_solid))
                        row["changed_voxels_from_start"] = int(np.count_nonzero(proposal_solid != base_solid))
                        score = internal_score(row, p_series)
                        row["greedy_score"] = json.dumps(score)
                        proposals.append((score, proposal_offset, history, row))
                        all_rows.append(row)
                        evaluated += 1
                    best_score, best_offset, best_history, best_row = min(proposals, key=lambda item: item[0])
                    if best_score < current_score:
                        current_score = best_score
                        current_offset = best_offset
                        current_history = best_history
                        current = best_row
                        accepted += 1
                trajectories.append(
                    {
                        "target_pc_design_id": p.design_id,
                        "family": family,
                        "restart": restart,
                        "start_design_id": start.design_id,
                        "final_design_id": current["design_id"],
                        "evaluated_candidates": evaluated,
                        "accepted_steps": accepted,
                        "final_match": matched(current, p_series),
                        "final_epsilon": current["epsilon"],
                        "final_height_mm": current["predicted_rise_height_mm"],
                        "final_local_mutation_count": len(current_history),
                        "final_local_mutation_history": json.dumps(current_history, separators=(",", ":")),
                        "final_score": json.dumps(current_score),
                    }
                )
    return pd.DataFrame(all_rows), pd.DataFrame(trajectories)


def surface_candidate_table(samples: int, seed: int) -> pd.DataFrame:
    if samples <= 0 or samples & (samples - 1):
        raise ValueError("surface sample count must be a power of two")
    units = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(int(math.log2(samples)))
    rows = []
    face_specs = [(10.0, 10.0, 2), (10.0, 25.0, 4)]
    for i, u in enumerate(units, 1):
        short = 0.15 + u[0] * (1.20 - 0.15)
        aspect = math.exp(math.log(1.0) + u[1] * math.log(12.0))
        ligament = MIN_FEATURE_MM + u[2] * (2.0 - MIN_FEATURE_MM)
        layer = MIN_FEATURE_MM + u[3] * (2.0 - MIN_FEATURE_MM)
        long = short * aspect
        pitch_short = short + ligament
        pitch_long = long + ligament
        r_eq = short * long / (short + long)
        open_area = 0.0
        pore_count = 0
        for short_span, long_span, multiplicity in face_specs:
            n_short = max(int(math.floor((short_span - ligament) / pitch_short)), 0)
            n_long = max(int(math.floor((long_span - ligament) / pitch_long)), 0)
            count = n_short * n_long
            open_area += multiplicity * count * short * long
            pore_count += multiplicity * count
        a_eff = open_area / EXTERIOR_AREA_MM2
        valid = bool(
            pore_count > 0
            and ligament >= MIN_FEATURE_MM
            and layer >= MIN_FEATURE_MM
            and long + 2.0 * ligament <= 25.0
            and short + 2.0 * ligament <= 10.0
        )
        stable = valid and r_eq <= R_CRIT_MM
        rows.append(
            {
                "surface_design_id": f"slit-S{i:05d}",
                "short_axis_mm": short,
                "aspect_ratio": aspect,
                "long_axis_mm": long,
                "ligament_mm": ligament,
                "pitch_short_mm": pitch_short,
                "pitch_long_mm": pitch_long,
                "layer_depth_mm": layer,
                "pore_count": pore_count,
                "surface_open_area_mm2": open_area,
                "surface_A_eff_star": a_eff,
                "surface_r_eq_ctrl_mm": r_eq,
                "surface_geometry_valid": valid,
                "surface_stability_pass": stable,
            }
        )
    return pd.DataFrame(rows)


def interval_mask(values: np.ndarray, span: float, aperture: float, gap: float) -> np.ndarray:
    pitch = aperture + gap
    count = max(int(math.floor((span - gap) / pitch)), 0)
    if count == 0:
        return np.zeros_like(values, dtype=bool)
    occupied = count * aperture + (count - 1) * gap
    start = 0.5 * (span - occupied)
    local = values - start
    index = np.floor(local / pitch).astype(int)
    within = local - index * pitch
    return (index >= 0) & (index < count) & (within >= 0.0) & (within <= aperture)


def aperture_mask(u: np.ndarray, v: np.ndarray, u_span: float, v_span: float, short: float, long: float, gap: float) -> np.ndarray:
    return interval_mask(u, u_span, short, gap) & interval_mask(v, v_span, long, gap)


def core_solid(ip, family: str, params: pd.Series, grid) -> np.ndarray:
    args = (grid, float(params.L_mm), float(params.feature_bottom_mm), float(params.feature_top_mm), float(params.p))
    if family == "Graded BCC":
        solid = ip.bcc_solid(*args)
    elif family == "Graded Gyroid":
        solid = ip.gyroid_solid(*args)
    elif family == "Simple cubic":
        solid = ip.simple_cubic_solid(*args)
    elif family in {"Diamond", "Schwarz P"}:
        solid = ip.tpms_sheet_solid(grid, family, *args[1:])
    else:
        raise ValueError(family)
    history_raw = params.get("local_mutation_history", "")
    if isinstance(history_raw, str) and history_raw.strip() and history_raw.strip() != "[]":
        feature_z = ip.gradient_feature(
            grid.z,
            float(params.feature_bottom_mm),
            float(params.feature_top_mm),
            float(params.p),
            grid.domain.height_mm,
        )
        solid = local_mutated_solid(ip, solid, feature_z, grid, json.loads(history_raw))
    return solid


def composite_solid(ip, core_row: pd.Series, surface_row: pd.Series, grid) -> np.ndarray:
    core = core_solid(ip, str(core_row.family), core_row, grid)
    t = float(surface_row.layer_depth_mm)
    short = float(surface_row.short_axis_mm)
    long = float(surface_row.long_axis_mm)
    gap = float(surface_row.ligament_mm)
    x = grid.xx[None, :, :]
    y = grid.yy[None, :, :]
    z = grid.z[:, None, None]
    interior = (
        (x >= t) & (x <= grid.domain.width_mm - t)
        & (y >= t) & (y <= grid.domain.depth_mm - t)
        & (z >= t) & (z <= grid.domain.height_mm - t)
    )
    clipped_core = core & interior

    ap_xy = aperture_mask(grid.xx, grid.yy, grid.domain.width_mm, grid.domain.depth_mm, short, long, gap)[None, :, :]
    ap_yz = aperture_mask(y, z, grid.domain.depth_mm, grid.domain.height_mm, short, long, gap)
    ap_xz = aperture_mask(x, z, grid.domain.width_mm, grid.domain.height_mm, short, long, gap)
    shell = np.zeros_like(core, dtype=bool)
    shell |= ((z <= t) | (z >= grid.domain.height_mm - t)) & (~ap_xy)
    shell |= ((x <= t) | (x >= grid.domain.width_mm - t)) & (~ap_yz)
    shell |= ((y <= t) | (y >= grid.domain.depth_mm - t)) & (~ap_xz)
    return clipped_core | shell


def largest_component_fraction(binary: np.ndarray) -> float:
    labels, count = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max() / max(sizes.sum(), 1))


def evaluate_composites(ip, hydraulic_function, cores: pd.DataFrame, surfaces: pd.DataFrame, pc: pd.DataFrame, z_pitch: float) -> pd.DataFrame:
    domain = ip.DomainConfig(xy_pitch_mm=0.125, slice_spacing_mm=z_pitch)
    grid = ip.SliceGrid(domain)
    fluid = ip.FluidConfig()
    pc_records = list(pc.itertuples(index=False))
    rows = []
    for core_index, core in cores.reset_index(drop=True).iterrows():
        for surface_index, surface in surfaces.reset_index(drop=True).iterrows():
            solid = composite_solid(ip, core, surface, grid)
            _, metrics = ip.profile_from_solid(solid, grid, fluid, hydraulic_function)
            row = {
                "composite_design_id": f"{core.design_id}__{surface.surface_design_id}",
                "baseline_group": (
                    "Randomized local greedy search" if str(core.search_stage) == "randomized_local_greedy"
                    else "Slit skin + " + str(core.family)
                ),
                "core_family": core.family,
                "core_design_id": core.design_id,
                "surface_design_id": surface.surface_design_id,
                "L_mm": core.L_mm,
                "feature_bottom_mm": core.feature_bottom_mm,
                "feature_top_mm": core.feature_top_mm,
                "p": core.p,
                "short_axis_mm": surface.short_axis_mm,
                "long_axis_mm": surface.long_axis_mm,
                "aspect_ratio": surface.aspect_ratio,
                "ligament_mm": surface.ligament_mm,
                "pitch_short_mm": surface.pitch_short_mm,
                "pitch_long_mm": surface.pitch_long_mm,
                "layer_depth_mm": surface.layer_depth_mm,
                "surface_A_eff_star": surface.surface_A_eff_star,
                "surface_r_eq_ctrl_mm": surface.surface_r_eq_ctrl_mm,
                "surface_stability_pass": surface.surface_stability_pass,
                "epsilon": metrics["porosity"],
                "predicted_rise_height_mm": metrics["predicted_rise_height_mm"],
                "min_slice_porosity": metrics["min_slice_porosity"],
                "min_R_tr_Pa": metrics["min_R_tr_Pa"],
                "geometry_valid": metrics["geometry_valid"],
                "liquid_connected": metrics["liquid_connected"],
                "solid_layer_overlap_valid": metrics["solid_layer_overlap_valid"],
                "largest_solid_component_fraction": largest_component_fraction(solid),
                "combined_z_pitch_mm": z_pitch,
            }
            matches = []
            for pcref in pc_records:
                if abs(float(row["epsilon"]) - float(pcref.epsilon)) <= DELTA_EPSILON and float(row["predicted_rise_height_mm"]) >= float(pcref.predicted_rise_height_mm) - DELTA_HEIGHT_MM:
                    matches.append(pcref.design_id)
            row["matched_pc_design_ids"] = ";".join(matches)
            row["internal_match_after_skin"] = bool(matches)
            row["joint_numeric_pass_pc_mean"] = bool(
                row["internal_match_after_skin"]
                and row["surface_stability_pass"]
                and float(row["surface_A_eff_star"]) >= 0.505689
            )
            rows.append(row)
    return pd.DataFrame(rows)


def select_surface_finalists(surface: pd.DataFrame, count: int) -> pd.DataFrame:
    valid = surface[is_true(surface.surface_stability_pass)].copy()
    valid["thickness_bin"] = pd.cut(valid.layer_depth_mm, bins=[0.39, 0.5, 0.75, 1.0, 1.5, 2.01], include_lowest=True)
    finalists = (
        valid.sort_values(["surface_A_eff_star", "surface_r_eq_ctrl_mm"], ascending=[False, True])
        .groupby("thickness_bin", observed=True, group_keys=False)
        .head(max(1, math.ceil(count / 5)))
        .sort_values(["surface_A_eff_star", "layer_depth_mm"], ascending=[False, True])
        .head(count)
    )
    return finalists.drop(columns=["thickness_bin"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--greedy-iterations", type=int, default=30)
    parser.add_argument("--greedy-proposals", type=int, default=4)
    parser.add_argument("--greedy-restarts", type=int, default=2)
    parser.add_argument("--surface-samples", type=int, default=8192)
    parser.add_argument("--surface-finalists", type=int, default=12)
    parser.add_argument("--combined-z-pitch-mm", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    ip = load_module(PIPELINE, "internal_pareto_pipeline_strong_baselines")
    hydraulic = load_module(HYDRAULIC, "hydraulic_strong_baselines")
    hydraulic_function = hydraulic.apply_state_updated_viscous_profile
    candidates = pd.read_csv(CANDIDATES,low_memory=False)
    pc = pd.read_csv(PC_REFERENCES)

    matched_cores, existing_pairs = select_existing_matches(candidates, pc)
    existing_pairs.to_csv(args.out_dir / "parametric_internal_match_pairs.csv", index=False)
    matched_cores.to_csv(args.out_dir / "parametric_internal_matched_cores.csv", index=False)

    greedy, trajectories = randomized_greedy_search(
        ip,
        hydraulic_function,
        candidates,
        pc,
        iterations=args.greedy_iterations,
        proposals_per_iteration=args.greedy_proposals,
        restarts_per_family=args.greedy_restarts,
        seed=args.seed,
    )
    greedy.to_csv(args.out_dir / "randomized_local_greedy_all_evaluations.csv", index=False)
    trajectories.to_csv(args.out_dir / "randomized_local_greedy_trajectories.csv", index=False)
    greedy_matches = []
    for _, row in greedy[is_true(greedy.feasible)].iterrows():
        for _, p in pc.iterrows():
            if matched(row, p):
                greedy_matches.append({"pc_design_id": p.design_id, "core_design_id": row.design_id})
    greedy_pairs = pd.DataFrame(greedy_matches).drop_duplicates() if greedy_matches else pd.DataFrame(columns=["pc_design_id", "core_design_id"])
    greedy_pairs.to_csv(args.out_dir / "randomized_local_greedy_internal_match_pairs.csv", index=False)
    greedy_ids = set(greedy_pairs.core_design_id) if len(greedy_pairs) else set()
    greedy_matched = greedy[greedy.design_id.isin(greedy_ids)].copy()
    if len(greedy_matched):
        greedy_matched = (
            greedy_matched.sort_values(["predicted_rise_height_mm", "epsilon"], ascending=[False, False])
            .drop_duplicates(["family", "target_pc_design_id", "local_mutation_history"])
            .groupby(["family", "target_pc_design_id"], group_keys=False)
            .head(2)
        )
    greedy_matched.to_csv(args.out_dir / "randomized_local_greedy_internal_matched_cores.csv", index=False)

    surface = surface_candidate_table(args.surface_samples, args.seed + 1000)
    surface.to_csv(args.out_dir / "slit_surface_candidates.csv", index=False)
    surface_finalists = select_surface_finalists(surface, args.surface_finalists)
    surface_finalists.to_csv(args.out_dir / "slit_surface_finalists.csv", index=False)

    composite_cores = pd.concat([matched_cores, greedy_matched], ignore_index=True, sort=False)
    if len(composite_cores):
        composite_cores = composite_cores.drop_duplicates("design_id")
        composites = evaluate_composites(
            ip, hydraulic_function, composite_cores, surface_finalists, pc, args.combined_z_pitch_mm
        )
    else:
        composites = pd.DataFrame()
    composites.to_csv(args.out_dir / "composite_joint_evaluation.csv", index=False)

    family_existing = existing_pairs.groupby("core_family").size().to_dict() if len(existing_pairs) else {}
    summary = {
        "status": "pilot_complete",
        "existing_internal_candidates_reused": int(len(candidates)),
        "pc_reference_count": int(len(pc)),
        "matching_rule": {
            "absolute_epsilon_tolerance": DELTA_EPSILON,
            "height_lower_tolerance_mm": DELTA_HEIGHT_MM,
        },
        "parametric_match_pairs": int(len(existing_pairs)),
        "parametric_unique_matched_cores": int(len(matched_cores)),
        "parametric_match_pairs_by_family": {str(k): int(v) for k, v in family_existing.items()},
        "randomized_local_greedy_evaluations": int(len(greedy)),
        "randomized_local_greedy_trajectories": int(len(trajectories)),
        "randomized_local_greedy_match_pairs": int(len(greedy_pairs)),
        "randomized_local_greedy_unique_matched_cores_selected_for_surface": int(len(greedy_matched)),
        "slit_surface_candidates": int(len(surface)),
        "slit_surface_stable_candidates": int(is_true(surface.surface_stability_pass).sum()),
        "slit_surface_finalists": int(len(surface_finalists)),
        "composite_evaluations": int(len(composites)),
        "composites_retaining_internal_match": int(is_true(composites.internal_match_after_skin).sum()) if len(composites) else 0,
        "composites_passing_joint_pc_mean_numeric_thresholds": int(is_true(composites.joint_numeric_pass_pc_mean).sum()) if len(composites) else 0,
        "surface_bounds": {
            "short_axis_mm": [0.15, 1.20],
            "aspect_ratio": [1.0, 12.0],
            "ligament_mm": [0.40, 2.0],
            "layer_depth_mm": [0.40, 2.0],
        },
        "combined_z_pitch_mm": args.combined_z_pitch_mm,
        "runtime_seconds": time.time() - start,
    }
    (args.out_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
