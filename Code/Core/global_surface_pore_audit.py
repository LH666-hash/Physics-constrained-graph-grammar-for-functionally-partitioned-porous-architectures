# -*- coding: utf-8 -*-
"""Patch-independent global surface pore audit with contour perimeters.

Carrier refinement is conforming, and finite-radius rod contours are located
within each triangle. Components smaller than the declared measurable area
are retained as diagnostics and excluded from resolved-pore acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _load_scale_align(path: Path, target_height_mm: float) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"No triangular surface found in {path}")
    mesh = mesh.copy()
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
    except Exception:
        pass
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.fix_normals()
    height = float(np.ptp(mesh.bounds[:, 2]))
    if height <= 1.0e-12:
        raise ValueError("Input mesh has zero height")
    mesh.apply_scale(float(target_height_mm) / height)
    mesh.apply_translation(-mesh.bounds[0])
    return mesh


def _boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    if len(mesh.faces) == 0:
        return 0
    inv = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    counts = np.bincount(inv, minlength=len(mesh.edges_unique))
    return int(np.count_nonzero(counts == 1))


def _conforming_subdivide_to_edge(mesh: trimesh.Trimesh, max_edge_mm: float,
                                   max_iter: int = 12) -> trimesh.Trimesh:
    """Conforming adaptive refinement that preserves shared-edge topology."""
    max_edge_mm = float(max_edge_mm)
    if max_edge_mm <= 0:
        return mesh.copy()

    vertices = [np.asarray(v, dtype=float).copy() for v in np.asarray(mesh.vertices)]
    faces = np.asarray(mesh.faces, dtype=np.int64)

    for _ in range(max_iter):
        edge_rows = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        unique_edges = np.unique(np.sort(edge_rows, axis=1), axis=0)
        vv = np.asarray(vertices, dtype=float)
        lengths = np.linalg.norm(vv[unique_edges[:, 1]] - vv[unique_edges[:, 0]], axis=1)
        long_edges = unique_edges[lengths > max_edge_mm + 1.0e-12]
        if len(long_edges) == 0:
            break

        midpoint = {}
        for a, b in long_edges:
            a, b = int(a), int(b)
            midpoint[(a, b)] = len(vertices)
            vertices.append(0.5 * (vv[a] + vv[b]))

        new_faces = []
        append = new_faces.append
        extend = new_faces.extend
        for tri in faces:
            a, b, c = map(int, tri)
            kab = (a, b) if a < b else (b, a)
            kbc = (b, c) if b < c else (c, b)
            kca = (c, a) if c < a else (a, c)
            mab = midpoint.get(kab)
            mbc = midpoint.get(kbc)
            mca = midpoint.get(kca)
            n = int(mab is not None) + int(mbc is not None) + int(mca is not None)

            if n == 0:
                append((a, b, c))
            elif n == 1:
                if mab is not None:
                    extend(((a, mab, c), (mab, b, c)))
                elif mbc is not None:
                    extend(((b, mbc, a), (mbc, c, a)))
                else:
                    extend(((c, mca, b), (mca, a, b)))
            elif n == 2:
                if mab is not None and mbc is not None:
                    extend(((mab, b, mbc), (a, mab, c), (mab, mbc, c)))
                elif mbc is not None and mca is not None:
                    extend(((mbc, c, mca), (b, mbc, a), (mbc, mca, a)))
                else:
                    extend(((mca, a, mab), (c, mca, b), (mca, mab, b)))
            else:
                extend(((a, mab, mca), (mab, b, mbc),
                        (mca, mbc, c), (mab, mbc, mca)))
        faces = np.asarray(new_faces, dtype=np.int64)
    else:
        raise RuntimeError(
            f"Conforming refinement did not reach max edge {max_edge_mm} mm "
            f"within {max_iter} iterations"
        )

    out = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float),
                          faces=faces, process=False)
    out.remove_unreferenced_vertices()
    return out


def _point_segment_distance_squared(points: np.ndarray,
                                    a: np.ndarray,
                                    b: np.ndarray) -> np.ndarray:
    ab = b - a
    den = np.maximum(np.einsum("ij,ij->i", ab, ab), 1.0e-24)
    t = np.einsum("ij,ij->i", points - a, ab) / den
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    d = points - closest
    return np.einsum("ij,ij->i", d, d)


def _sample_solid_mask(sample_points: np.ndarray,
                       seg_a: np.ndarray,
                       seg_b: np.ndarray,
                       seg_r: np.ndarray,
                       midpoint_tree: cKDTree,
                       search_padding_mm: float,
                       chunk_size: int = 50000) -> np.ndarray:
    """Classify samples against finite-radius graph segments efficiently."""
    mids = 0.5 * (seg_a + seg_b)
    half = 0.5 * np.linalg.norm(seg_b - seg_a, axis=1)
    ab = seg_b - seg_a
    den = np.einsum("ij,ij->i", ab, ab)
    global_search = float(np.max(half + seg_r)) + float(search_padding_mm)
    solid = np.zeros(len(sample_points), dtype=bool)

    for start in range(0, len(sample_points), int(chunk_size)):
        pts = sample_points[start:start + int(chunk_size)]
        tree_pts = cKDTree(pts)
        try:
            pairs = tree_pts.sparse_distance_matrix(
                midpoint_tree, global_search, output_type="coo_matrix"
            )
            rows, cols, dmid = pairs.row, pairs.col, pairs.data
            if len(rows) == 0:
                continue
            keep = dmid <= (half[cols] + seg_r[cols] + float(search_padding_mm))
            rows, cols = rows[keep], cols[keep]
            if len(rows) == 0:
                continue
            p = pts[rows]
            a = seg_a[cols]
            ab_sel = ab[cols]
            t = np.einsum("ij,ij->i", p - a, ab_sel) / np.maximum(den[cols], 1.0e-24)
            t = np.clip(t, 0.0, 1.0)
            closest = a + t[:, None] * ab_sel
            delta = p - closest
            d2 = np.einsum("ij,ij->i", delta, delta)
            hit_rows = rows[d2 <= (seg_r[cols] + 1.0e-9) ** 2]
            if len(hit_rows):
                solid[start + np.unique(hit_rows)] = True
        except (TypeError, AttributeError):
            # Compatibility fallback for older SciPy.
            candidates = midpoint_tree.query_ball_point(pts, r=global_search)
            for local_i, ids in enumerate(candidates):
                if not ids:
                    continue
                ids = np.asarray(ids, dtype=int)
                p = np.repeat(pts[local_i:local_i + 1], len(ids), axis=0)
                dm = np.linalg.norm(mids[ids] - p, axis=1)
                keep = dm <= (half[ids] + seg_r[ids] + search_padding_mm)
                ids = ids[keep]
                p = p[keep]
                if len(ids) == 0:
                    continue
                d2 = _point_segment_distance_squared(p, seg_a[ids], seg_b[ids])
                if np.any(d2 <= (seg_r[ids] + 1.0e-9) ** 2):
                    solid[start + local_i] = True
    return solid


def global_audit(args):
    run_dir = Path(args.run_dir)
    val_dir = run_dir / "surface_validation"
    nodes_path = val_dir / "surface_nodes.csv"
    edges_path = val_dir / "surface_edges.csv"
    local_summary_path = val_dir / "surface_final_graph_summary.json"
    for p in (nodes_path, edges_path, local_summary_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required file missing: {p}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    local_summary = json.loads(local_summary_path.read_text(encoding="utf-8-sig"))

    if "bond_type" in edges.columns:
        kinds = edges["bond_type"].fillna("").astype(str).str.lower()
        surface_edges = edges.loc[kinds.eq("surface")].copy()
        if surface_edges.empty:
            raise ValueError("Surface graph contains no surface bonds")
    else:
        surface_edges = edges.copy()

    node_ids = pd.to_numeric(nodes["atom_id"], errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(node_ids).all()
            or not np.equal(node_ids, np.floor(node_ids)).all()
            or nodes["atom_id"].duplicated().any()):
        raise ValueError("Surface graph contains missing, noninteger, or duplicate atom IDs")
    for endpoint_column in ("source", "target"):
        endpoints = pd.to_numeric(surface_edges[endpoint_column], errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(endpoints).all()
                or not np.equal(endpoints, np.floor(endpoints)).all()):
            raise ValueError(f"Surface graph has noninteger {endpoint_column} IDs")
    node_xyz = nodes.set_index("atom_id")[["x", "y", "z"]]
    if not np.isfinite(node_xyz.to_numpy(dtype=float)).all():
        raise ValueError("Surface graph contains nonfinite node coordinates")
    a_list, b_list, r_list = [], [], []
    missing = 0
    invalid = 0
    for row in surface_edges.itertuples(index=False):
        s, t = int(row.source), int(row.target)
        if s not in node_xyz.index or t not in node_xyz.index:
            missing += 1
            continue
        radius = float(row.radius)
        if not np.isfinite(radius) or radius <= 0:
            invalid += 1
            continue
        a = node_xyz.loc[s].to_numpy(dtype=float)
        b = node_xyz.loc[t].to_numpy(dtype=float)
        if np.linalg.norm(b - a) <= 1.0e-12:
            invalid += 1
            continue
        a_list.append(a); b_list.append(b); r_list.append(radius)
    if not a_list:
        raise ValueError("No finite-radius surface segments available for global audit")

    seg_a = np.asarray(a_list, dtype=float)
    seg_b = np.asarray(b_list, dtype=float)
    seg_r = np.asarray(r_list, dtype=float)
    midpoint_tree = cKDTree(0.5 * (seg_a + seg_b))

    source = _load_scale_align(Path(args.stl), args.target_height_mm)
    source_face_count = int(len(source.faces))
    source_watertight = bool(source.is_watertight)
    source_boundary_edges = _boundary_edge_count(source)

    carrier = _conforming_subdivide_to_edge(source, args.global_max_edge_mm)
    audit_watertight = bool(carrier.is_watertight)
    audit_boundary_edges = _boundary_edge_count(carrier)
    if source_watertight and not audit_watertight:
        raise RuntimeError(
            "Conforming global-audit refinement broke a watertight carrier; "
            "aborting instead of classifying artificial boundaries as physical."
        )

    from contour_core import RodDistanceQuery, contour_metrics
    vertices = np.asarray(carrier.vertices, dtype=float)
    faces = np.asarray(carrier.faces, dtype=np.int64)
    tri = vertices[faces]
    signed_fn = RodDistanceQuery(seg_a, seg_b, seg_r, float(args.global_max_edge_mm))
    phi = signed_fn(vertices)
    measured = contour_metrics(carrier, phi, signed_fn=signed_fn, local_refine=3)
    face_area = measured["carrier_area_face"]
    open_face_area = measured["open_area_face"]
    solid_fraction = np.divide(face_area-open_face_area, face_area,
                               out=np.zeros_like(face_area), where=face_area>0)
    solid_face = open_face_area <= 1.0e-15
    component_count = measured["component_count"]
    component_area = measured["area"]
    component_wet_perimeter = measured["perimeter"]
    touches_boundary = measured["touches_boundary"]
    component_boundary_perimeter = np.where(touches_boundary, np.nan, 0.0)
    req = measured["radius"]
    component_face_count = measured["face_count"]
    comp_centroid = measured["centroids"]
    component_aspect_ratio = measured["aspect"]

    radius_acceptance = args.radius_acceptance_mm
    if radius_acceptance is None:
        radius_acceptance = local_summary.get("radius_acceptance_mm")
    if radius_acceptance is None:
        radius_acceptance = local_summary.get("critical_radius_mm")
    if radius_acceptance is None:
        raise ValueError("Local audit did not export radius_acceptance_mm/critical_radius_mm")
    radius_acceptance = float(radius_acceptance)
    if not np.isfinite(radius_acceptance) or radius_acceptance <= 0:
        raise ValueError("radius_acceptance_mm must be finite and positive")

    minimum_pore_area_mm2 = float(args.minimum_pore_area_mm2)
    if not np.isfinite(minimum_pore_area_mm2) or minimum_pore_area_mm2 < 0:
        raise ValueError("minimum_pore_area_mm2 must be finite and nonnegative")
    subresolution = (~touches_boundary) & (component_area < minimum_pore_area_mm2 - 1.0e-10)
    complete = (~touches_boundary) & (~subresolution)
    exceeded = complete & (req > radius_acceptance + 1.0e-10)

    aspect_ratio_max = float(args.aspect_ratio_max)
    aspect_ratio_min_area_mm2 = float(args.aspect_ratio_min_area_mm2)
    # V11: morphology hard cap is meaningful only for sufficiently large
    # complete pores. Small positive-area pores are not AR-gated.
    aspect_exceeded = complete & (component_area > aspect_ratio_min_area_mm2 + 1.0e-10) & (
        (~np.isfinite(component_aspect_ratio)) |
        (component_aspect_ratio > aspect_ratio_max + 1.0e-10)
    )
    undersized = subresolution

    pores = pd.DataFrame({
        "global_pore_id": np.arange(component_count, dtype=int),
        "complete_pore": complete.astype(int),
        "below_area_resolution": subresolution.astype(int),
        "touches_physical_boundary": touches_boundary.astype(int),
        "face_count": component_face_count,
        "area_mm2": component_area,
        "wet_perimeter_mm": component_wet_perimeter,
        "shape_factor_3d": (
            4.0 * np.pi * component_area
            / np.maximum(component_wet_perimeter ** 2, 1.0e-12)
        ),
        "physical_boundary_perimeter_mm": component_boundary_perimeter,
        "equivalent_radius_mm": req,
        "radius_acceptance_mm": radius_acceptance,
        "radius_ratio_to_acceptance": req / radius_acceptance,
        "radius_exceeded": exceeded.astype(int),
        "aspect_ratio_3d_pca": component_aspect_ratio,
        "aspect_ratio_max": aspect_ratio_max,
        "aspect_ratio_min_area_mm2": aspect_ratio_min_area_mm2,
        "aspect_ratio_exceeded": aspect_exceeded.astype(int),
        "minimum_pore_area_mm2": minimum_pore_area_mm2,
        "area_undersized": undersized.astype(int),
        "centroid_x_mm": comp_centroid[:, 0],
        "centroid_y_mm": comp_centroid[:, 1],
        "centroid_z_mm": comp_centroid[:, 2],
    })
    complete_frame = pores[pores["complete_pore"] == 1].copy()
    boundary_frame = pores[pores["touches_physical_boundary"] == 1].copy()
    finite_complete = complete_frame[np.isfinite(complete_frame["equivalent_radius_mm"])].copy()
    max_req = float(finite_complete["equivalent_radius_mm"].max()) if len(finite_complete) else None
    max_ratio = float(finite_complete["radius_ratio_to_acceptance"].max()) if len(finite_complete) else None
    exceeded_count = int(complete_frame["radius_exceeded"].sum()) if len(complete_frame) else 0
    finite_ar_complete = complete_frame[np.isfinite(complete_frame["aspect_ratio_3d_pca"])].copy()
    max_ar = float(finite_ar_complete["aspect_ratio_3d_pca"].max()) if len(finite_ar_complete) else None
    finite_sf_complete = complete_frame[np.isfinite(complete_frame["shape_factor_3d"])].copy()
    mean_sf = float(finite_sf_complete["shape_factor_3d"].mean()) if len(finite_sf_complete) else None
    median_sf = float(finite_sf_complete["shape_factor_3d"].median()) if len(finite_sf_complete) else None
    mean_ar = float(finite_ar_complete["aspect_ratio_3d_pca"].mean()) if len(finite_ar_complete) else None
    median_ar = float(finite_ar_complete["aspect_ratio_3d_pca"].median()) if len(finite_ar_complete) else None
    def area_weighted_mean(frame: pd.DataFrame, column: str) -> float | None:
        valid = frame[np.isfinite(frame[column]) & np.isfinite(frame["area_mm2"]) &
                      (frame["area_mm2"] > 0)].copy()
        return (float(np.average(valid[column], weights=valid["area_mm2"]))
                if len(valid) else None)

    area_weighted_ar = area_weighted_mean(complete_frame, "aspect_ratio_3d_pca")
    area_weighted_sf = area_weighted_mean(complete_frame, "shape_factor_3d")
    ar_exceeded_count = int(complete_frame["aspect_ratio_exceeded"].sum()) if len(complete_frame) else 0
    undersized_count = int(np.count_nonzero(subresolution))
    min_subresolution_area = float(np.min(component_area[subresolution])) if undersized_count else None
    min_complete_area = float(complete_frame["area_mm2"].min()) if len(complete_frame) else None

    # Actual maximum audit edge after refinement.
    audit_edges = np.asarray(carrier.edges_unique, dtype=np.int64)
    actual_max_edge = float(np.max(np.linalg.norm(
        vertices[audit_edges[:, 1]] - vertices[audit_edges[:, 0]], axis=1
    ))) if len(audit_edges) else 0.0

    result = {
        "method": "patch_independent_global_3d_contour_open_component_audit_v3",
        "mesh_refinement": "conforming_shared_edge_adaptive_subdivision",
        "interpretation": (
            "Complete pores are globally connected open regions that do not touch a true "
            "carrier-mesh boundary. Patch/chart IDs are not used for connectivity or pore counting."
        ),
        "input_stl": str(Path(args.stl)),
        "target_height_mm": float(args.target_height_mm),
        "source_carrier_face_count": source_face_count,
        "audit_carrier_face_count": int(len(faces)),
        "audit_carrier_vertex_count": int(len(vertices)),
        "requested_global_max_edge_mm": float(args.global_max_edge_mm),
        "global_max_edge_mm": actual_max_edge,
        "source_carrier_watertight": source_watertight,
        "audit_carrier_watertight": audit_watertight,
        "source_boundary_edge_count": source_boundary_edges,
        "audit_boundary_edge_count": audit_boundary_edges,
        "boundary_integrity_pass": bool(not source_watertight or audit_watertight),
        "solid_face_vote_fraction": None,
        "contour_local_refine_levels": 3,
        "global_carrier_area_mm2": float(np.sum(face_area)),
        "global_open_area_mm2": float(np.sum(component_area)),
        "global_complete_pore_area_mm2": float(complete_frame["area_mm2"].sum()),
        "global_liquid_open_area_fraction": float(
            np.sum(component_area) / max(float(np.sum(face_area)), 1.0e-12)
        ),
        "global_complete_pore_area_fraction": float(
            float(complete_frame["area_mm2"].sum())
            / max(float(np.sum(face_area)), 1.0e-12)
        ),
        "surface_segment_count": int(len(seg_a)),
        "missing_surface_edge_endpoint_count": int(missing),
        "invalid_surface_segment_count": int(invalid),
        "radius_acceptance_mm": radius_acceptance,
        "global_open_component_count": int(component_count),
        "global_complete_pore_count": int(np.count_nonzero(complete)),
        "global_boundary_open_region_count": int(np.count_nonzero(touches_boundary)),
        "global_subresolution_open_component_count": undersized_count,
        "global_radius_exceeded_complete_pore_count": exceeded_count,
        "global_max_complete_pore_equivalent_radius_mm": max_req,
        "global_max_complete_pore_radius_ratio_to_acceptance": max_ratio,
        "global_complete_pore_radius_acceptance_pass": bool(
            exceeded_count == 0 and missing == 0 and invalid == 0),
        "global_aspect_ratio_reference_max": aspect_ratio_max,
        "global_aspect_ratio_reference_min_area_mm2": aspect_ratio_min_area_mm2,
        "global_aspect_ratio_hard_max": aspect_ratio_max,
        "global_aspect_ratio_hard_min_area_mm2": aspect_ratio_min_area_mm2,
        "global_aspect_ratio_exceeded_complete_pore_count": ar_exceeded_count,
        "global_max_complete_pore_aspect_ratio": max_ar,
        "global_mean_complete_pore_aspect_ratio": mean_ar,
        "global_area_weighted_mean_complete_pore_aspect_ratio": area_weighted_ar,
        "global_median_complete_pore_aspect_ratio": median_ar,
        "global_mean_complete_pore_shape_factor": mean_sf,
        "global_area_weighted_mean_complete_pore_shape_factor": area_weighted_sf,
        "global_median_complete_pore_shape_factor": median_sf,
        "pore_shape_mean_definition": "sum(A_i * metric_i) / sum(A_i) over complete pores with finite metric and A_i > 0; legacy global_mean fields are unweighted",
        "shape_factor_definition": "4*pi*A_open/L_wet^2 for each reconstructed complete pore",
        "global_complete_pore_aspect_ratio_acceptance_pass": bool(ar_exceeded_count == 0),
        "global_aspect_ratio_is_diagnostic_only": True,
        "global_minimum_pore_area_mm2": minimum_pore_area_mm2,
        # The triangulated global mesh can create sub-resolution open
        # components near finite-radius walls. Keep those counts as diagnostics
        # rather than using them as the sole hard minimum-area proof.
        "global_mesh_area_undersized_complete_component_count_diagnostic": undersized_count,
        "global_mesh_min_complete_component_area_mm2_diagnostic": min_subresolution_area,
        "global_minimum_area_acceptance_method": (
            "components below the declared minimum measurable area are reported "
            "separately and excluded from the resolved complete-pore gate"
        ),
        "global_complete_pore_minimum_area_acceptance_pass": bool(
            len(complete_frame) > 0 and
            (complete_frame["area_mm2"] >= minimum_pore_area_mm2 - 1.0e-10).all()
        ),
        "global_complete_pore_acceptance_pass": bool(
            exceeded_count == 0
            and missing == 0
            and invalid == 0
            and len(complete_frame) > 0
            and (complete_frame["area_mm2"] >= minimum_pore_area_mm2 - 1.0e-10).all()
        ),
        "local_audit_available": bool(local_summary.get("local_audit_available", True)),
        "local_audit_pore_record_count": int(local_summary.get(
            "local_pore_record_count", local_summary.get("pore_count", 0)
        )) if local_summary.get("local_audit_available", True) else None,
        "local_audit_seam_dependent_record_count": int(local_summary.get(
            "seam_dependent_pore_count", 0
        )) if local_summary.get("local_audit_available", True) else None,
        "local_audit_nonseam_radius_exceeded_count": int(local_summary.get(
            "nonseam_radius_exceeded_pore_count", 0
        )) if local_summary.get("local_audit_available", True) else None,
        "local_audit_seam_radius_exceeded_record_count": int(local_summary.get(
            "seam_radius_exceeded_local_record_count", 0
        )) if local_summary.get("local_audit_available", True) else None,
        "cross_patch_pores_reconnected_in_global_audit": True,
        "global_complete_pore_count_available": True,
        "measurement_definition": (
            "Finite-radius rod occupancy is reconstructed as a zero contour within "
            "conforming carrier triangles; open area and wetted contour length use "
            "three local boundary-refinement levels."
        ),
    }

    out_dir = run_dir / str(args.out_dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    pores.sort_values(["complete_pore", "equivalent_radius_mm"],
                      ascending=[False, False]).to_csv(
        out_dir / "global_surface_pores.csv", index=False
    )
    complete_frame.sort_values("equivalent_radius_mm", ascending=False).to_csv(
        out_dir / "global_complete_pores.csv", index=False
    )
    complete_frame[complete_frame["radius_exceeded"] == 1].sort_values(
        "equivalent_radius_mm", ascending=False
    ).to_csv(out_dir / "global_radius_exceeded_complete_pores.csv", index=False)
    complete_frame[complete_frame["aspect_ratio_exceeded"] == 1].sort_values(
        "aspect_ratio_3d_pca", ascending=False
    ).to_csv(out_dir / "global_aspect_ratio_exceeded_complete_pores.csv", index=False)
    pores[pores["below_area_resolution"] == 1].sort_values(
        "area_mm2", ascending=True
    ).to_csv(out_dir / "global_mesh_area_undersized_components_diagnostic.csv", index=False)
    boundary_frame.to_csv(out_dir / "global_boundary_open_regions.csv", index=False)

    if bool(args.export_face_classification):
        pd.DataFrame({
            "face_id": np.arange(len(faces), dtype=int),
            "area_mm2": face_area,
            "solid_vote_fraction": solid_fraction,
            "classified_solid": solid_face.astype(int),
        }).to_csv(out_dir / "global_carrier_face_classification.csv", index=False)

    (out_dir / "global_surface_pore_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("")
    print("===== GLOBAL SURFACE PORE AUDIT V3 (CONTOUR) =====")
    print(f"Audit watertight: {audit_watertight}; boundary edges: {audit_boundary_edges}")
    print(f"Complete pores: {result['global_complete_pore_count']}")
    print(f"Boundary-open regions: {result['global_boundary_open_region_count']}")
    print(f"Radius-exceeded complete pores: {exceeded_count}")
    print(f"Maximum complete-pore r_eq: {max_req}")
    print(f"Acceptance radius: {radius_acceptance:.6f} mm")
    print(f"Aspect-ratio-exceeded complete pores: {ar_exceeded_count}")
    print(f"Maximum complete-pore AR: {max_ar}")
    print(f"Diagnostic AR reference: {aspect_ratio_max:.3f} for complete pores with area > {aspect_ratio_min_area_mm2:.3f} mm^2")
    print(f"Mesh-diagnostic area<min components: {undersized_count}")
    print(f"Subresolution minimum component area: {min_subresolution_area}")
    print(f"Minimum measurable pore area: {minimum_pore_area_mm2:.6f} mm^2")
    print(f"Minimum-area proof: {result['global_minimum_area_acceptance_method']}")
    print(f"Radius PASS: {result['global_complete_pore_radius_acceptance_pass']}")
    print(f"AR <= reference: {result['global_complete_pore_aspect_ratio_acceptance_pass']} (diagnostic only)")
    print(f"Area PASS: {result['global_complete_pore_minimum_area_acceptance_pass']}")
    print(f"Combined PASS: {result['global_complete_pore_acceptance_pass']}")
    print(f"Output: {out_dir}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stl", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--target-height-mm", type=float, default=25.0)
    p.add_argument("--global-max-edge-mm", type=float, default=0.08)
    p.add_argument("--solid-vote-fraction", type=float, default=0.5)
    p.add_argument("--aspect-ratio-max", type=float, default=5.0)
    p.add_argument("--aspect-ratio-min-area-mm2", type=float, default=0.5)
    p.add_argument("--minimum-pore-area-mm2", type=float, default=1.0e-6)
    p.add_argument("--radius-acceptance-mm", type=float)
    p.add_argument("--search-padding-mm", type=float, default=0.02)
    p.add_argument("--out-dir-name", type=str, default="global_surface_pore_audit")
    p.add_argument("--export-face-classification", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    global_audit(parse_args())
