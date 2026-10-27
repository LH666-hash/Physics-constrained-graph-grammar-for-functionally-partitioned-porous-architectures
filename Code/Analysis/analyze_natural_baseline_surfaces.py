"""Compare natural outer surfaces of all internally eligible Diamond baselines."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "Code/Benchmark"))
import internal_pareto_pipeline as ip  # noqa: E402
import run_strong_baseline_experiment as baseline  # noqa: E402
sys.path.insert(0, str(PKG / "Code/Core"))
from capillary_viscous_filling import apply_state_updated_viscous_profile  # noqa: E402

ROOT = PKG / "Source_Data/Benchmark_Analysis"
OUT = ROOT / "Natural_Surface_Comparison"
OUT.mkdir(exist_ok=True)
PITCH = .125


def surface_metrics(solid: np.ndarray, pitch: float = PITCH) -> dict:
    liquid = ~solid
    shell = np.zeros_like(liquid, dtype=bool)
    shell[0] = liquid[0]
    shell[-1] = liquid[-1]
    shell[:, 0, :] = liquid[:, 0, :]
    shell[:, -1, :] = liquid[:, -1, :]
    shell[:, :, 0] = liquid[:, :, 0]
    shell[:, :, -1] = liquid[:, :, -1]
    labels, count = ndimage.label(shell, structure=ndimage.generate_binary_structure(3, 1))
    faces = [labels[0], labels[-1], labels[:, 0, :], labels[:, -1, :], labels[:, :, 0], labels[:, :, -1]]
    area_pixels = np.zeros(count + 1, dtype=np.int64)
    perimeter_edges = np.zeros(count + 1, dtype=np.int64)
    for face in faces:
        area_pixels += np.bincount(face.ravel(), minlength=count + 1)
        for a, b in ((face[1:], face[:-1]), (face[:, 1:], face[:, :-1])):
            perimeter_edges += np.bincount(a[(a > 0) & (b == 0)], minlength=count + 1)
            perimeter_edges += np.bincount(b[(b > 0) & (a == 0)], minlength=count + 1)
        for edge in (face[0], face[-1], face[:, 0], face[:, -1]):
            perimeter_edges += np.bincount(edge[edge > 0], minlength=count + 1)
    # Every open cube-edge pixel was counted as an exterior edge on two adjacent faces.
    edge_views = [labels[:, i, j] for i in (0, -1) for j in (0, -1)]
    edge_views += [labels[i, :, j] for i in (0, -1) for j in (0, -1)]
    edge_views += [labels[i, j, :] for i in (0, -1) for j in (0, -1)]
    for edge in edge_views:
        perimeter_edges -= 2 * np.bincount(edge[edge > 0], minlength=count + 1)
    area = area_pixels[1:] * pitch**2
    perimeter = np.maximum(perimeter_edges[1:], 0) * pitch
    radii = np.divide(2 * area, perimeter, out=np.full_like(area, np.nan), where=perimeter > 0)
    resolved = area >= 1e-6
    return dict(surface_open_area_fraction=float(sum(int((~solid_face).sum()) for solid_face in
                        (solid[0], solid[-1], solid[:, 0, :], solid[:, -1, :], solid[:, :, 0], solid[:, :, -1]))
                        / sum(face.size for face in faces)),
                global_open_component_count=int(count),
                resolved_complete_pore_count=int(resolved.sum()),
                maximum_complete_pore_equivalent_radius_mm=float(np.nanmax(radii[resolved])) if resolved.any() else np.nan,
                minimum_complete_pore_equivalent_radius_mm=float(np.nanmin(radii[resolved])) if resolved.any() else np.nan,
                maximum_complete_pore_area_mm2=float(np.max(area[resolved])) if resolved.any() else np.nan)


def main() -> None:
    data = pd.read_csv(ROOT / "内部/internally_eligible_baseline_structures.csv", low_memory=False)
    assert len(data) == 241 and set(data.family) == {"Diamond"}
    grid = ip.SliceGrid(ip.DomainConfig(xy_pitch_mm=PITCH, slice_spacing_mm=PITCH))
    grid.z = np.arange(PITCH / 2, 25, PITCH)
    preset = data.loc[data.search_group.eq("preset")].iloc[0]
    base = baseline.core_solid(ip, "Diamond", preset, grid)
    signed_distance, minimum_offset = baseline.prepare_local_perturbation(base, np.full_like(grid.z, float(preset.feature_bottom_mm)), grid)
    original_grid = ip.SliceGrid(ip.DomainConfig(slice_spacing_mm=.75))
    original_base = baseline.core_solid(ip, "Diamond", preset, original_grid)
    coarse_signed, coarse_min_offset = baseline.prepare_local_perturbation(
        original_base, np.full_like(original_grid.z, float(preset.feature_bottom_mm)), original_grid)
    current_fluid = ip.FluidConfig(density_difference_kg_m3=998.0, reference_flow_m3_s=2e-7)
    pc_surface = pd.read_csv(PKG / "Source_Data/Ablation_Analysis/表面/surface_ablation_group_seed_final_graph_metrics.csv")
    pc_surface = pc_surface.loc[pc_surface.group.eq("surface_full")]
    area_goal = float(pc_surface.liquid_open_area_fraction.mean())
    radius_goal = float(pc_surface.maximum_complete_pore_radius_mm.mean())
    rows = []
    for index, row in enumerate(data.itertuples(index=False), start=1):
        if row.search_group == "preset":
            solid = base
            coarse_solid = original_base
        else:
            history = json.loads(row.local_mutation_history)
            offset = np.zeros_like(signed_distance, dtype=np.float32)
            for mutation in history:
                offset = baseline.apply_local_mutation(offset, minimum_offset, grid, mutation)
            solid = signed_distance <= offset
            coarse_offset = np.zeros_like(coarse_signed, dtype=np.float32)
            for mutation in history:
                coarse_offset = baseline.apply_local_mutation(coarse_offset, coarse_min_offset, original_grid, mutation)
            coarse_solid = coarse_signed <= coarse_offset
        metrics = surface_metrics(solid)
        _, original_metrics = ip.profile_from_solid(coarse_solid, original_grid, ip.FluidConfig(),
                                                     apply_state_updated_viscous_profile)
        _, current_metrics = ip.profile_from_solid(coarse_solid, original_grid, current_fluid,
                                                    apply_state_updated_viscous_profile)
        source_hash_ok = (row.search_group == "preset" or
                          hashlib.sha1(np.packbits(coarse_solid).tobytes()).hexdigest() == row.geometry_sha1)
        original_match = (np.isclose(original_metrics["porosity"], row.epsilon, atol=1e-12) and
                          np.isclose(original_metrics["min_R_tr_Pa"], row.min_R_tr_Pa, atol=1e-8) and
                          np.isclose(original_metrics["predicted_rise_height_mm"], row.predicted_rise_height_mm, atol=1e-8)
                          and source_hash_ok)
        metrics.update(search_group=row.search_group, family=row.family, design_id=row.design_id,
                       original_epsilon=row.epsilon, original_predicted_height_mm=row.predicted_rise_height_mm,
                       original_reconstruction_match=bool(original_match),
                       current_min_R_tr_Pa=float(current_metrics["min_R_tr_Pa"]),
                       current_predicted_height_mm=float(current_metrics["predicted_rise_height_mm"]),
                       current_internal_screen_pass=bool(current_metrics["geometry_valid"] and current_metrics["liquid_connected"]
                          and current_metrics["solid_layer_overlap_valid"] and current_metrics["porosity"] >= .6074781974940404
                          and current_metrics["predicted_rise_height_mm"] >= 25.0),
                       reaches_pc_mean_open_area=metrics["surface_open_area_fraction"] >= area_goal,
                       reaches_pc_mean_radius=metrics["maximum_complete_pore_equivalent_radius_mm"] <= radius_goal)
        metrics["reaches_both_pc_surface_means"] = metrics["reaches_pc_mean_open_area"] and metrics["reaches_pc_mean_radius"]
        rows.append(metrics)
        if index % 40 == 0 or index == len(data):
            print(f"Natural exterior evaluated: {index}/{len(data)}", flush=True)
    frame = pd.DataFrame(rows)
    assert frame.original_reconstruction_match.all(), "Original search-table metrics did not reproduce"
    frame.to_csv(OUT / "all_241_internal_eligible_natural_surfaces.csv", index=False)
    fine_pitch = .0625
    fine_grid = ip.SliceGrid(ip.DomainConfig(xy_pitch_mm=fine_pitch, slice_spacing_mm=fine_pitch))
    fine_grid.z = np.arange(fine_pitch / 2, 25, fine_pitch)
    fine_base = baseline.core_solid(ip, "Diamond", preset, fine_grid)
    fine_signed, fine_min_offset = baseline.prepare_local_perturbation(
        fine_base, np.full_like(fine_grid.z, float(preset.feature_bottom_mm)), fine_grid)
    sample_ids = [str(preset.design_id), str(frame.loc[frame.search_group.eq("local_greedy")]
                    .sort_values("surface_open_area_fraction").iloc[0].design_id)]
    resolution = []
    data_by_id = data.set_index("design_id")
    for design_id in sample_ids:
        row = data_by_id.loc[design_id]
        if row.search_group == "preset":
            fine_solid = fine_base
        else:
            fine_offset = np.zeros_like(fine_signed, dtype=np.float32)
            for mutation in json.loads(row.local_mutation_history):
                fine_offset = baseline.apply_local_mutation(fine_offset, fine_min_offset, fine_grid, mutation)
            fine_solid = fine_signed <= fine_offset
        fine_metrics = surface_metrics(fine_solid, fine_pitch)
        coarse = frame.set_index("design_id").loc[design_id]
        for pitch, area, radius in ((PITCH, coarse.surface_open_area_fraction,
                                     coarse.maximum_complete_pore_equivalent_radius_mm),
                                    (fine_pitch, fine_metrics["surface_open_area_fraction"],
                                     fine_metrics["maximum_complete_pore_equivalent_radius_mm"])):
            resolution.append(dict(design_id=design_id, surface_pitch_mm=pitch,
                                   surface_open_area_fraction=area,
                                   maximum_complete_pore_equivalent_radius_mm=radius,
                                   reaches_both_pc_surface_means=area >= area_goal and radius <= radius_goal))
    pd.DataFrame(resolution).to_csv(OUT / "natural_surface_resolution_audit.csv", index=False)
    summary = dict(internal_eligible_preset=1, internal_eligible_greedy=240,
                   natural_surface_preset_reaching_both=int(frame.query("search_group == 'preset'").reaches_both_pc_surface_means.sum()),
                   natural_surface_greedy_reaching_both=int(frame.query("search_group == 'local_greedy'").reaches_both_pc_surface_means.sum()),
                   current_physics_internal_screen_pass_count=int(frame.current_internal_screen_pass.sum()),
                   pc_surface_mean_open_area=area_goal, pc_surface_mean_max_complete_radius_mm=radius_goal,
                   surface_sampling_pitch_mm=PITCH,
                   natural_surface_definition="Outer boundary of original Diamond voxel solid; no replacement skin",
                   analysis_scope="Global cube-boundary 6-connected open components; 2A/P from rasterized surface area and wet perimeter")
    (OUT / "natural_surface_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
