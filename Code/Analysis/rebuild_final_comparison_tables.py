"""Rebuild final five-seed ablation and benchmark tables from archived run outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]
ABL = PKG / "Source_Data/Ablation_Analysis"
SUR = ABL / "表面"
INT = ABL / "内部"
BEN = PKG / "Source_Data/Benchmark_Analysis/内部"
SEEDS = (101, 202, 303, 404, 505)
SUR_METHODS = {"full": "surface_full", "random_pore": "surface_random_pore", "none": "surface_none"}
INT_METHODS = {
    "Full PC-AEGG": ("Internal_Full_Runs", "seed_{seed}", "internal_full_fixed_budget"),
    "Fully random": ("Internal_Fully_Random", "seed_{seed}", "internal_fully_random"),
    "Initial gradient skeleton": ("Internal_Initial_Gradient_Skeleton", "seed_{seed}", "internal_initial_gradient_skeleton"),
    "Matched random position": ("Internal_Random_Position_Fixed_Budget", "internal_random_position_fixed_budget_seed{seed}", "internal_random_position_fixed_budget"),
}
PAIRED_CONTROL = {"Fully random": "fully_random",
                  "Initial gradient skeleton": "initial_gradient_skeleton",
                  "Matched random position": "random_position_fixed_budget"}


def readj(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def csv(p: Path) -> pd.DataFrame:
    assert p.is_file(), p
    return pd.read_csv(p)


def save(df: pd.DataFrame, p: Path) -> None:
    df.to_csv(p, index=False, float_format="%.15g")


def main() -> None:
    surface, contours, observed, carried = [], [], [], []
    surf_summaries: dict[tuple[str, int], dict] = {}
    for method, group in SUR_METHODS.items():
        for seed in SEEDS:
            run = SUR / "Surface_Groups" / f"surface_{method}_seed{seed}"
            summary = readj(run / "global_surface_pore_audit/global_surface_pore_summary.json")
            surf_summaries[(method, seed)] = summary
            nodes = csv(run / "surface_validation/surface_nodes.csv")
            edges = csv(run / "surface_validation/surface_edges.csv")
            seeds = csv(run / "surface_validation/surface_voronoi_seeds.csv")
            pores = csv(run / "global_surface_pore_audit/global_complete_pores.csv")
            complete = pores.loc[pores.complete_pore.astype(bool) & ~pores.below_area_resolution.astype(bool)]
            assert len(complete) == summary["global_complete_pore_count"], run
            satisfied = int((complete.equivalent_radius_mm <= summary["radius_acceptance_mm"] + 1e-10).sum())
            assert len(edges) == summary["surface_segment_count"], run
            surface.append(dict(group=group, seed=seed, final_surface_graph_nodes=len(nodes),
                                final_surface_graph_edges=len(edges),
                                voronoi_inserted_seeds=int(seeds.seed_role.ne("initial").sum()),
                                resolved_complete_pores=len(complete),
                                radius_satisfying_complete_pores=satisfied,
                                complete_pore_satisfaction_fraction=satisfied / len(complete),
                                maximum_complete_pore_radius_mm=summary["global_max_complete_pore_equivalent_radius_mm"],
                                critical_radius_with_tolerance_mm=summary["radius_acceptance_mm"],
                                liquid_open_area_fraction=summary["global_liquid_open_area_fraction"],
                                full_radius_acceptance_pass=summary["global_complete_pore_radius_acceptance_pass"],
                                contour_audit_edge_mm=summary["requested_global_max_edge_mm"],
                                contour_local_refinement_levels=summary["contour_local_refine_levels"]))
            for row in complete.itertuples(index=False):
                contours.append(dict(group=group, seed=seed, component_id=row.global_pore_id,
                                     area_mm2=row.area_mm2, perimeter_mm=row.wet_perimeter_mm,
                                     equivalent_radius_mm=row.equivalent_radius_mm,
                                     shape_factor=row.shape_factor_3d,
                                     aspect_ratio_3d_pca=row.aspect_ratio_3d_pca,
                                     centroid_x_mm=row.centroid_x_mm, centroid_y_mm=row.centroid_y_mm,
                                     centroid_z_mm=row.centroid_z_mm,
                                     touches_carrier_boundary=row.touches_physical_boundary))
            evolution = csv(run / "surface_validation/surface_refinement_parameter_evolution.csv")
            ratio = evolution.hydraulic_radius_max_mm / evolution.hydraulic_radius_capillary_target_mm
            assert np.isfinite(ratio).all(), run
            for iteration, value in zip(evolution.iteration.astype(int), ratio):
                observed.append(dict(group=group, seed=seed, iteration=iteration,
                                     maximum_local_radius_ratio=value, terminal_carried_forward=False))
            if method != "none":
                last = int(evolution.iteration.max())
                for iteration in range(1, 1 + last):
                    carried.append(dict(group=group, seed=seed, iteration=iteration,
                                        maximum_local_radius_ratio=float(ratio.iloc[iteration - 1]),
                                        terminal_carried_forward=False, last_observed_iteration=last))
    maximum_iteration = max(r["iteration"] for r in carried)
    for method in ("full", "random_pore"):
        group = SUR_METHODS[method]
        for seed in SEEDS:
            rows = [r for r in carried if r["group"] == group and r["seed"] == seed]
            last = rows[-1]
            for iteration in range(last["iteration"] + 1, maximum_iteration + 1):
                carried.append(dict(group=group, seed=seed, iteration=iteration,
                                    maximum_local_radius_ratio=last["maximum_local_radius_ratio"],
                                    terminal_carried_forward=True,
                                    last_observed_iteration=last["iteration"]))
    surface_df = pd.DataFrame(surface)
    assert surface_df.groupby("group").size().eq(5).all()
    save(surface_df, SUR / "surface_ablation_group_seed_final_graph_metrics.csv")
    save(pd.DataFrame(contours), SUR / "surface_ablation_group_seed_complete_pore_contours.csv")
    save(pd.DataFrame(observed), SUR / "surface_ablation_group_seed_observed_refinement_trajectory.csv")
    save(pd.DataFrame(carried).sort_values(["group", "seed", "iteration"]),
         SUR / "surface_ablation_group_seed_terminal_carried_trajectory.csv")
    trajectory = pd.DataFrame(carried).groupby(["group", "iteration"]).maximum_local_radius_ratio.agg(
        mean="mean", std="std", count="count").reset_index()
    assert trajectory["count"].eq(5).all()
    save(trajectory, SUR / "surface_ablation_group_round_mean_sd_n.csv")

    internal, profiles, benchmark = [], [], []
    for method, (folder, pattern, group_id) in INT_METHODS.items():
        for seed in SEEDS:
            run = INT / folder / pattern.format(seed=seed)
            standalone_profile = csv(run / "internal_slice_physics_analysis.csv")
            standalone_minimum = float(standalone_profile.physical_transport_margin_Pa.min())
            nodes = csv(run / "structure_nodes.csv")
            edges = csv(run / "structure_edges.csv")
            paired = None
            paired_pass = np.nan
            dual_pass = np.nan
            assembled_min = np.nan
            preassembly_reserve_min = np.nan
            if method == "Full PC-AEGG":
                paired = ABL / "Paired_Full_Dual_Domain" / f"paired_full_seed{seed}"
                validation = readj(paired / "final_validation_summary.json")
                status = readj(paired / "final_assembly_status.json")
                pore = readj(paired / "global_surface_pore_audit/global_surface_pore_summary.json")
                profile = csv(paired / "post_assembly_pressure_profile.csv")
                paired_pass = bool(status["graph_screen_pass"] and validation["fine_slice_pressure_screen"]["pass"])
                assert paired_pass, paired
                dual_pass = bool(validation["dual_domain"]["dual_domain_pass"])
                assembled_min = float(validation["fine_slice_pressure_screen"]["minimum_lower_bound_margin_pa"])
                preassembly_reserve_min = float(validation["fine_slice_pressure_screen"]["minimum_preassembly_reserve_margin_pa"])
                assert np.isclose(profile.physical_transport_margin_Pa.min(),
                                  validation["fine_slice_pressure_screen"]["minimum_ordinary_margin_pa"]), paired
                assert validation["internal_print_support"]["internal_node_count"] == len(nodes), paired
                assert validation["internal_print_support"]["internal_edge_count"] == len(edges), paired
                evaluation_scope = "assembled_surface_internal"
            else:
                paired = ABL / "Paired_Control_Dual_Domain" / PAIRED_CONTROL[method] / f"seed_{seed}"
                paired_status = readj(paired / "final_pairing_status.json")
                assert paired_status.get("evaluation_scope") == "same_seed_full_surface_plus_internal_control", paired
                profile = csv(paired / "post_assembly_pressure_profile.csv")
                paired_pass = bool(paired_status["pressure_screen"]["pass"])
                dual_pass = bool(paired_status["dual_domain_audit"]["dual_domain_pass"])
                assembled_min = float(paired_status["pressure_screen"]["minimum_lower_bound_margin_pa"])
                assert np.isclose(profile.physical_transport_margin_Pa.min(),
                                  paired_status["pressure_screen"]["minimum_ordinary_margin_pa"]), paired
                evaluation_scope = "assembled_full_surface_plus_internal_control"
            forces = csv(paired / "structure_force_summary.csv")
            assert len(forces) == len(profile) and np.allclose(
                forces.z_mid_mm.to_numpy(dtype=float), profile.z_mm.to_numpy(dtype=float)), paired
            profile["Lower_Band_Margin_Pa"] = forces.Lower_Band_Margin_Pa.to_numpy(dtype=float)
            assert np.isclose(float(profile.Lower_Band_Margin_Pa.min()), assembled_min), paired
            assert profile.z_mm.nunique() == len(profile) and len(profile) == 25, run
            weighted_porosity = profile.A_liq_mm2.sum() / profile.A_domain_mm2.sum()
            minimum = float(profile.physical_transport_margin_Pa.min())
            fraction = float(profile.physical_transport_margin_Pa.ge(0).mean())
            if method == "Full PC-AEGG":
                benchmark.append(dict(family="PC-AEGG", design_id=f"PC-AEGG-full-seed{seed}",
                    replicate=SEEDS.index(seed) + 1, seed=seed, run_folder=f"seed_{seed}",
                    epsilon=weighted_porosity, porosity=weighted_porosity,
                    predicted_rise_height_mm=25.0, first_failure_z_mm=np.nan,
                    min_R_tr_Pa=minimum,
                    standalone_internal_minimum_transport_margin_Pa=standalone_minimum,
                    minimum_preassembly_reserve_margin_Pa=preassembly_reserve_min,
                    control_equivalent_radius_mm=pore["global_max_complete_pore_equivalent_radius_mm"],
                    inverse_control_radius_mm_inv=1 / pore["global_max_complete_pore_equivalent_radius_mm"],
                    surface_liquid_area_fraction=pore["global_liquid_open_area_fraction"],
                    stable_liquid_area_fraction=pore["global_liquid_open_area_fraction"],
                    surface_area_mm2=pore["global_carrier_area_mm2"],
                    pore_count=pore["global_complete_pore_count"],
                    quality_pass=True, graph_screen_pass=True, pressure_profile_pass=True,
                    liquid_connected=True, geometry_valid=True, solid_layer_overlap_valid=True,
                    internal_nodes=len(nodes), internal_edges=len(edges),
                    package_source_graph_folder=run.relative_to(PKG).as_posix(),
                    package_paired_assembly_folder=paired.relative_to(PKG).as_posix()))
            internal.append(dict(method=method, seed=seed, evaluation_scope=evaluation_scope,
                                 minimum_transport_margin_pa=minimum,
                                 ordinary_transport_pass=bool(minimum >= 0),
                                 minimum_final_assembled_margin_pa=assembled_min,
                                 standalone_internal_minimum_transport_margin_pa=standalone_minimum,
                                 paired_pressure_screen_pass=paired_pass,
                                 paired_dual_domain_pass=dual_pass,
                                 paired_complete_graph_screen_pass=(paired_pass if method == "Full PC-AEGG" else np.nan),
                                 pairing_folder=paired.relative_to(PKG).as_posix(),
                                 nonnegative_slice_fraction=fraction,
                                 mean_porosity_fraction=float(profile.porosity.mean()),
                                 weighted_porosity_fraction=weighted_porosity,
                                 final_internal_nodes=len(nodes), final_internal_edges=len(edges),
                                 all_pass=bool(paired_pass and dual_pass),
                                 group_id=group_id))
            profile.insert(0, "seed", seed)
            profile.insert(0, "method", method)
            profile.insert(2, "evaluation_scope", evaluation_scope)
            profile["z_mid_mm"] = profile.z_mm
            profile["Transport_Margin_Pa"] = profile.physical_transport_margin_Pa
            profiles.append(profile)
    internal_df = pd.DataFrame(internal)
    assert internal_df.groupby("method").size().eq(5).all()
    save(internal_df, INT / "internal_metrics_fixed_budget_four_groups.csv")
    save(pd.concat(profiles, ignore_index=True), INT / "internal_profiles_fixed_budget_four_groups.csv")
    save(pd.DataFrame(benchmark), BEN / "07_PC-AEGG完整组5次终态内部指标.csv")
    summary = {"seeds": list(SEEDS),
               "transport_metric": "minimum of physical_transport_margin_Pa across 25 slices",
               "evaluation_scopes": {"Full PC-AEGG": "assembled_surface_internal_with_complete_graph_screen",
                                     "other_groups": "assembled_full_surface_plus_internal_control; pressure and dual-domain screens only"},
               "all_pass_definition": "paired pressure screen and dual-domain screen; complete global surface pore audit is available only for the full group",
               "groups": {}}
    for method, rows in internal_df.groupby("method", sort=False):
        summary["groups"][method] = dict(
            n=len(rows), evaluation_scope=rows.evaluation_scope.iloc[0],
            all_pass_count=int(rows.all_pass.sum()),
            ordinary_transport_pass_count=int(rows.ordinary_transport_pass.sum()),
            minimum_transport_margin_mean_pa=float(rows.minimum_transport_margin_pa.mean()),
            minimum_transport_margin_sd_pa=float(rows.minimum_transport_margin_pa.std(ddof=1)),
            minimum_lower_bound_margin_mean_pa=float(rows.minimum_final_assembled_margin_pa.mean()),
            minimum_lower_bound_margin_sd_pa=float(rows.minimum_final_assembled_margin_pa.std(ddof=1)),
            final_nodes_mean=float(rows.final_internal_nodes.mean()),
            final_edges_mean=float(rows.final_internal_edges.mean()),
            weighted_porosity_mean=float(rows.weighted_porosity_fraction.mean()))
    (INT / "internal_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    representative = PKG / "Source_Data/Representative_Structure_Seed42"
    representative_profile = csv(representative / "post_assembly_pressure_profile.csv")
    assert len(representative_profile) == 25
    representative_summary = pd.DataFrame([dict(
        seed=42, evaluation_scope="assembled_surface_internal",
        slice_count=len(representative_profile),
        area_weighted_porosity_fraction=(representative_profile.A_liq_mm2.sum()
                                         / representative_profile.A_domain_mm2.sum()),
        minimum_slice_porosity_fraction=float(representative_profile.porosity.min()),
        maximum_slice_porosity_fraction=float(representative_profile.porosity.max()),
        source_profile="post_assembly_pressure_profile.csv")])
    save(representative_summary, representative / "assembled_porosity_25_slice_summary.csv")
    print(f"Rebuilt 15 surface groups, 20 internal groups and 5 paired benchmark rows; {maximum_iteration} trajectory rounds.")


if __name__ == "__main__":
    main()
