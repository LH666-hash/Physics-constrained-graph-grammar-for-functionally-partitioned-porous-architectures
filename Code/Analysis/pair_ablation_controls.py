"""Pair each final full surface with its same-seed internal ablation controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "Code/Core"))
import PC_AEGG_主程序 as pc  # noqa: E402
from saved_configuration import apply_saved_configuration, pairing_fingerprint
from surface_patch_builder import _mesh_boundary_vertex_mask, _vertex_gaussian_curvature  # noqa: E402

ABL = PACKAGE / "Source_Data/Ablation_Analysis"
STAGE = PACKAGE.parent / "_paired_control_stage_20261011"
DEST = ABL / "Paired_Control_Dual_Domain"
BACKUP = PACKAGE.parent / "_paired_control_previous_20261011"
SEEDS = (101, 202, 303, 404, 505)
CONTROL = {
    "random_position_fixed_budget": ("Internal_Random_Position_Fixed_Budget",
                                     "internal_random_position_fixed_budget_seed{seed}"),
    "initial_gradient_skeleton": ("Internal_Initial_Gradient_Skeleton", "seed_{seed}"),
    "fully_random": ("Internal_Fully_Random", "seed_{seed}"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def charts_for_saved_surface(mesh, paired: Path):
    config = pc.SurfaceBuildConfig(**json.loads(
        (paired / "surface_validation/surface_build_config.json").read_text(encoding="utf-8")))
    builder = pc.SurfacePatchBuilder(mesh, config)
    builder.base_mesh = builder._prepare_base_mesh()
    builder.base_vertex_gaussian_curvature = _vertex_gaussian_curvature(builder.base_mesh)
    builder.base_boundary_vertex_mask = _mesh_boundary_vertex_mask(builder.base_mesh)
    initial = builder._merge_small_patches(builder._segment_initial_patches())
    patches = builder._merge_small_patches(builder._split_distorted_patches(initial), min_faces=2)
    patches = [faces for faces in patches if builder._patch_is_valid(faces)]
    builder.charts = [builder._make_chart(i, faces) for i, faces in enumerate(patches)]
    builder.atoms = pd.read_csv(paired / "surface_validation/surface_nodes.csv")
    builder.edges = pd.read_csv(paired / "surface_validation/surface_edges.csv")
    assert set(builder.atoms.patch_id.astype(int)) == set(range(len(builder.charts)))
    return builder


def stitch(builder, mesh, internal_nodes: pd.DataFrame, internal_edges: pd.DataFrame,
           audit_dir: Path):
    return pc.stitch_surface_to_core_optimized(
        builder.atoms, internal_nodes, pc.STITCH_DISTANCE_MM,
        anchor_spacing=pc.STITCH_ANCHOR_SPACING_MM,
        existing_edges=internal_edges, surface_edges=builder.edges,
        mesh=mesh, z_bounds=(0.0, 25.0),
        max_internal_degree=pc.INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW,
        max_surface_degree=8, relaxed_distance_mm=pc.STITCH_RELAXED_DISTANCE_MM,
        surface_anchor_area_guard=builder.make_stitch_anchor_area_guard(builder.edges),
        audit_dir=audit_dir)


def remap_control(nodes: pd.DataFrame, edges: pd.DataFrame, first_id: int):
    nodes, edges = nodes.copy(), edges.copy()
    old_ids = nodes.atom_id.astype(int)
    assert old_ids.is_unique
    mapping = {old: first_id + i for i, old in enumerate(sorted(old_ids))}
    nodes["atom_id"] = old_ids.map(mapping).astype(int)
    for key in ("source", "target"):
        assert edges[key].astype(int).isin(mapping).all()
        edges[key] = edges[key].astype(int).map(mapping).astype(int)
    id_map = pd.DataFrame({"source_internal_node_id": list(mapping),
                           "paired_internal_node_id": list(mapping.values())})
    return nodes, edges, id_map


def verify_full_baseline(builder, mesh, paired: Path, probe: Path) -> None:
    atoms = pd.read_csv(paired / "assembly_nodes_final.csv")
    edges = pd.read_csv(paired / "assembly_edges_final.csv")
    internal = atoms.loc[atoms.face_name.astype(str).ne("surface")].copy()
    internal_edges = edges.loc[edges.bond_type.eq("internal")].copy()
    actual = edges.loc[edges.bond_type.eq("stitch"), ["source", "target"]]
    recreated = stitch(builder, mesh, internal, internal_edges, probe)
    expected_pairs = set(map(tuple, actual.to_numpy(dtype=int)))
    observed_pairs = set(map(tuple, recreated[["source", "target"]].to_numpy(dtype=int)))
    if observed_pairs != expected_pairs:
        raise RuntimeError(f"Stitch algorithm does not reproduce paired full baseline: {paired.name}")


def write_control_pair(seed: int, group: str, builder, mesh, paired: Path) -> None:
    control_folder, control_name = CONTROL[group]
    source = ABL / "内部" / control_folder / control_name.format(seed=seed)
    final = DEST / group / f"seed_{seed}"
    replace_existing = False
    if final.exists():
        status = final / "final_pairing_status.json"
        saved = json.loads(status.read_text(encoding="utf-8")) if status.is_file() else {}
        sources_match = saved.get("input_fingerprint") == pairing_fingerprint(paired, source)
        required = ("final_stitched.stl", "assembly_nodes_final.csv", "assembly_edges_final.csv",
                    "post_assembly_pressure_profile.csv")
        outputs_match = bool(saved.get("output_fingerprint")) and all(
            (final/name).is_file() and saved["output_fingerprint"].get(name) == sha(final/name)
            for name in required)
        if sources_match and outputs_match:
            print(f"SKIP {group} seed {seed}: inputs, configuration, source and outputs match", flush=True)
            return
        replace_existing = True
    stage = STAGE / group / f"seed_{seed}"
    if stage.exists():
        raise RuntimeError(f"Partial stage output requires review: {stage}")
    stage.mkdir(parents=True)
    pc.OUT_DIR = stage
    internal, internal_edges, id_map = remap_control(
        pd.read_csv(source / "structure_nodes.csv"),
        pd.read_csv(source / "structure_edges.csv"),
        int(builder.atoms.atom_id.max()) + 1)
    id_map.to_csv(stage / "internal_node_id_map.csv", index=False)
    stitch_edges = stitch(builder, mesh, internal, internal_edges, stage / "dual_validation")
    if stitch_edges.empty:
        raise RuntimeError(f"No legal cross-domain stitches: {group} seed {seed}")
    atoms = pd.concat((builder.atoms, internal), ignore_index=True)
    edges = pd.concat((builder.edges, internal_edges, stitch_edges), ignore_index=True)
    assert atoms.atom_id.is_unique
    edges = pc._dedup_edges_local(edges)
    atoms.to_csv(stage / "assembly_nodes_final.csv", index=False)
    edges.to_csv(stage / "assembly_edges_final.csv", index=False)
    profile = pc._pressure_profile_stl(
        mesh, atoms, pc._edge_records(edges), (0.0, 25.0),
        pc.INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache={})
    assert len(profile) == 25 and np.isfinite(profile.physical_transport_margin_Pa).all()
    profile.to_csv(stage / "post_assembly_pressure_profile.csv", index=False)
    profile.to_csv(stage / "internal_slice_physics_analysis.csv", index=False)
    shutil.copy2(paired / "surface_only.stl", stage / "surface_only.stl")
    shutil.copy2(source / "internal_only.stl", stage / "internal_only.stl")
    pc.export_stl(atoms, edges, stage / "final_stitched.stl")
    pc.export_dynamic_analysis_outputs(
        mesh, atoms, edges, pc._make_nodes_np_for_analysis(atoms, (0.0, 25.0)), profile, stage)
    pressure = pc.summarize_final_pressure_screen(profile, (0.0, 25.0))
    dual = pc.audit_final_dual_domain(
        atoms, edges, z_bounds=(0.0, 25.0),
        strict_length_mm=pc.STITCH_DISTANCE_MM,
        relaxed_length_mm=pc.STITCH_RELAXED_DISTANCE_MM,
        output_dir=stage / "dual_validation")
    support = pc.export_internal_support_audit(
        atoms, edges, stage / "internal_validation", z_bounds=(0.0, 25.0))
    abnormal = pc.audit_abnormal_layer_connectivity(
        atoms, edges, z_bounds=(0.0, 25.0), output_dir=stage / "internal_validation")
    stl = pc.audit_final_stl_file(stage / "final_stitched.stl", mesh, atoms, edges)
    summary = {
        "input_fingerprint": pairing_fingerprint(paired, source),
        "output_fingerprint": {n: sha(stage/n) for n in ("final_stitched.stl", "assembly_nodes_final.csv", "assembly_edges_final.csv", "post_assembly_pressure_profile.csv")},
        "configuration_sha256": sha(paired / "run_config.json"),
        "group": group, "seed": seed, "evaluation_scope": "same_seed_full_surface_plus_internal_control",
        "surface_source": (paired / "surface_validation").relative_to(PACKAGE).as_posix(),
        "internal_source": source.relative_to(PACKAGE).as_posix(),
        "surface_nodes_sha256": sha(paired / "surface_validation/surface_nodes.csv"),
        "internal_nodes_sha256": sha(source / "structure_nodes.csv"),
        "core_main_code_sha256": sha(PACKAGE / "Code/Core/PC_AEGG_主程序.py"),
        "surface_builder_code_sha256": sha(PACKAGE / "Code/Core/surface_patch_builder.py"),
        "stitch_count": int(len(stitch_edges)),
        "pressure_screen": pressure,
        "dual_domain_audit": dual,
        "internal_print_support": support,
        "abnormal_layer_connectivity": abnormal,
        "exported_stl": stl,
        "surface_global_pore_reaudit_performed": False,
    }
    (stage / "final_pairing_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert (stage / "final_stitched.stl").stat().st_size > 84
    assert (stage / "post_assembly_pressure_profile.csv").stat().st_size > 100
    assert len(list(stage.glob("*.svg"))) >= 5
    final.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if replace_existing:
        if not final.resolve().is_relative_to(PACKAGE.resolve()):
            raise RuntimeError(f"Unexpected paired output path: {final}")
        backup_dir = BACKUP / group
        backup_dir.mkdir(parents=True, exist_ok=True)
        if not backup_dir.resolve().is_relative_to(PACKAGE.parent.resolve()):
            raise RuntimeError(f"Unexpected backup path: {backup_dir}")
        previous = backup_dir / f"seed_{seed}_{datetime.now():%Y%m%d_%H%M%S}"
        if previous.exists():
            raise RuntimeError(f"Backup collision: {previous}")
        final.rename(previous)
    try:
        stage.rename(final)
    except Exception:
        if previous is not None and not final.exists():
            previous.rename(final)
        raise
    print(f"DONE {group} seed {seed}: stitches={len(stitch_edges)}, "
          f"ordinary_min={pressure['minimum_ordinary_margin_pa']:.3f} Pa, "
          f"lower_min={pressure['minimum_lower_bound_margin_pa']:.3f} Pa, "
          f"pressure_pass={pressure['pass']}, "
          f"dual_pass={dual.get('dual_domain_pass')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if any(seed not in SEEDS for seed in args.seeds):
        parser.error("Seeds must be selected from 101, 202, 303, 404, 505")
    STAGE.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        paired = ABL / "Paired_Full_Dual_Domain" / f"paired_full_seed{seed}"
        standalone_surface = ABL / "表面/Surface_Groups" / f"surface_full_seed{seed}"
        assert sha(standalone_surface / "surface_validation/surface_nodes.csv") == sha(
            paired / "surface_validation/surface_nodes.csv")
        assert sha(standalone_surface / "surface_validation/surface_edges.csv") == sha(
            paired / "surface_validation/surface_edges.csv")
        apply_saved_configuration(pc, paired / "run_config.json")
        mesh, _ = pc.load_and_scale_mesh(PACKAGE / "Input_Geometries/长方体.stl", 25.0)
        builder = charts_for_saved_surface(mesh, paired)
        verify_full_baseline(builder, mesh, paired, STAGE / f"baseline_seed{seed}")
        print(f"BASELINE seed {seed}: original full-group stitches reproduced", flush=True)
        for group in CONTROL:
            write_control_pair(seed, group, builder, mesh, paired)


if __name__ == "__main__":
    main()
