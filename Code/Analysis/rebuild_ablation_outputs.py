"""Export complete ablation structures using the packaged PC-AEGG code.

The fully random and initial-skeleton graph inputs are retained controls; their
models and physics figures are recalculated here without relabelling the graphs
as newly generated full PC-AEGG structures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "Code" / "Core"
sys.path.insert(0, str(CORE))


from saved_configuration import apply_saved_configuration

pc = importlib.import_module("PC_AEGG_主程序")
audit = importlib.import_module("global_surface_pore_audit")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_and_charts(mesh, nodes: pd.DataFrame, edges: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    pc.OUT_DIR = out
    bounds = (float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2]))
    points = pc._make_nodes_np_for_analysis(nodes, bounds)
    frame = pc.run_full_analysis(mesh, nodes, edges, points,
                                 pc._pressure_feedback_heights(bounds))
    if frame.empty:
        raise RuntimeError("Physics profile is empty")
    frame.to_csv(out / "internal_slice_physics_analysis.csv", index=False)
    pc.export_dynamic_analysis_outputs(mesh, nodes, edges, points, frame, out)
    pc.export_stl(nodes, edges, out / "internal_only.stl")


def split_full(source: Path, stl: Path, internal_out: Path, surface_out: Path):
    apply_saved_configuration(pc, source / "run_config.json")
    mesh, _ = pc.load_and_scale_mesh(stl, 25.0)
    nodes = pd.read_csv(source / "assembly_nodes_final.csv")
    edges = pd.read_csv(source / "assembly_edges_final.csv")
    internal = edges.loc[edges.bond_type.eq("internal")].copy()
    internal_ids = set(internal.source.astype(int)) | set(internal.target.astype(int))
    core_nodes = nodes.loc[nodes.atom_id.astype(int).isin(internal_ids)].copy()
    if core_nodes.empty or internal.empty:
        raise RuntimeError("Paired full structure lacks the internal graph")
    profile_and_charts(mesh, core_nodes, internal, internal_out)
    for name in ("evolution_internal.csv", "evolution_process_forces.csv",
                 "build_stage_timing.csv", "build_stage_timing.svg",
                 "run_config.json", "final_validation_summary.json"):
        src = source / name
        if src.is_file():
            shutil.copy2(src, internal_out / name)
    shutil.copytree(source / "surface_validation", surface_out / "surface_validation")
    shutil.copytree(source / "global_surface_pore_audit",
                    surface_out / "global_surface_pore_audit")
    shutil.copy2(source / "surface_only.stl", surface_out / "surface_only.stl")
    shutil.copy2(source / "run_config.json", surface_out / "paired_run_config.json")
    (internal_out / "source.json").write_text(json.dumps({
        "kind": "internal graph from current-code paired full generation",
        "paired_assembly_nodes_sha256": digest(source / "assembly_nodes_final.csv"),
        "paired_assembly_edges_sha256": digest(source / "assembly_edges_final.csv"),
    }, indent=2), encoding="utf-8")
    (surface_out / "source.json").write_text(json.dumps({
        "kind": "surface graph from current-code paired full generation",
        "paired_surface_nodes_sha256": digest(source / "surface_validation" / "surface_nodes.csv"),
        "paired_surface_edges_sha256": digest(source / "surface_validation" / "surface_edges.csv"),
    }, indent=2), encoding="utf-8")


def surface_control(full: Path, stl: Path, out: Path, policy: str, seed: int):
    apply_saved_configuration(pc, full / "run_config.json")
    cfg_data = json.loads((full / "surface_validation" / "surface_build_config.json").read_text(encoding="utf-8"))
    cfg = pc.SurfaceBuildConfig(**cfg_data)
    history = pd.read_csv(full / "surface_validation" / "surface_subdomain_refinement_history.csv")
    quota = []
    for patch_id, group in history.groupby("patch_id", sort=True):
        if int(patch_id) != len(quota):
            raise RuntimeError("Non-contiguous surface patch IDs")
        ordered = group.sort_values("round")
        quota.append(max(0, int(ordered.seed_count.iloc[-1] - ordered.seed_count.iloc[0])))
    cfg = replace(cfg, random_seed=seed, refinement_policy=policy,
                  max_insertions_by_patch=tuple(quota) if policy == "random" else (),
                  max_total_insertions=sum(quota) if policy == "random" else 0,
                  max_insertions_per_patch=0,
                  anisotropy_rewrite_enable=policy != "none",
                  global_ar_cap_rewrite_enable=policy != "none")
    mesh, _ = pc.load_and_scale_mesh(stl, 25.0)
    val = out / "surface_validation"
    val.mkdir(parents=True, exist_ok=True)
    builder = pc.SurfacePatchBuilder(mesh, cfg).build()
    builder.export_diagnostics(val, model_name="cuboid", export_stl=True)
    pc.export_pore_and_pressure_analysis(
        val, cfg, layers=20, gamma_npm=pc.GAMMA_NPM,
        liquid_density_kgm3=pc.RHO_LIQ, gas_density_kgm3=pc.RHO_GAS,
        gravity_ms2=pc.G_GRAV, contact_angle_deg=pc.CONTACT_ANGLE_DEG,
        additional_pressure_pa=pc.SURFACE_EXTRA_PRESSURE_PA,
        safety_factor=pc.SURFACE_CAPILLARY_SAFETY_FACTOR)
    builder.audit_final_retained_graph(builder.atoms, builder.edges, val,
                                       node_spheres_exported=False,
                                       include_stitch_anchors=False,
                                       apply_quality_gate=True)
    shutil.copy2(val / "surface_graph.stl", out / "surface_only.stl")
    args = argparse.Namespace(
        stl=stl, run_dir=out, target_height_mm=25.0,
        global_max_edge_mm=0.08, solid_vote_fraction=0.5,
        aspect_ratio_max=5.0, aspect_ratio_min_area_mm2=0.5,
        minimum_pore_area_mm2=1.0e-6, radius_acceptance_mm=None,
        search_padding_mm=0.02, out_dir_name="global_surface_pore_audit",
        export_face_classification=False)
    result = audit.global_audit(args)
    (out / "control_result.json").write_text(json.dumps({
        "policy": policy, "seed": seed,
        "paired_full_surface_nodes_sha256": digest(full / "surface_validation" / "surface_nodes.csv"),
        "paired_full_insertions_by_patch": quota,
        "surface_nodes": len(builder.atoms), "surface_edges": len(builder.edges),
        "global_complete_pore_acceptance_pass": result.get("global_complete_pore_acceptance_pass"),
    }, indent=2), encoding="utf-8")


def retained_control(source: Path, stl: Path, out: Path, kind: str):
    config = source / "run_config.json"
    if not config.is_file():
        config = ROOT / "Code/Configurations/water25_reference.json"
    apply_saved_configuration(pc, config)
    shutil.copytree(source, out, dirs_exist_ok=True)
    if kind == "initial_gradient":
        nodes = pd.read_csv(out / "atoms.csv")
        edges = pd.read_csv(out / "edges.csv")
    else:
        nodes = pd.read_csv(out / "nodes.csv")
        edges = pd.read_csv(out / "edges.csv")
    mesh, _ = pc.load_and_scale_mesh(stl, 25.0)
    profile_and_charts(mesh, nodes, edges, out)
    (out / "source.json").write_text(json.dumps({
        "kind": kind,
        "graph_generation": "retained original control graph",
        "physics_and_stl_export": "current packaged code",
        "source_nodes_sha256": digest(source / ("atoms.csv" if kind == "initial_gradient" else "nodes.csv")),
        "source_edges_sha256": digest(source / "edges.csv"),
    }, indent=2), encoding="utf-8")


def cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split-full")
    for flag in ("source", "stl", "internal-out", "surface-out"):
        s.add_argument("--" + flag, type=Path, required=True)
    s = sub.add_parser("surface-control")
    for flag in ("full", "stl", "out"):
        s.add_argument("--" + flag, type=Path, required=True)
    s.add_argument("--policy", choices=("random", "none"), required=True)
    s.add_argument("--seed", type=int, required=True)
    s = sub.add_parser("retained-control")
    for flag in ("source", "stl", "out"):
        s.add_argument("--" + flag, type=Path, required=True)
    s.add_argument("--kind", choices=("initial_gradient", "fully_random"), required=True)
    a = p.parse_args()
    if a.command == "split-full":
        split_full(a.source, a.stl, a.internal_out, a.surface_out)
    elif a.command == "surface-control":
        surface_control(a.full, a.stl, a.out, a.policy, a.seed)
    else:
        retained_control(a.source, a.stl, a.out, a.kind)


if __name__ == "__main__":
    cli()
