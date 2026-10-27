# -*- coding: utf-8 -*-
"""Optional legacy global aspect-ratio rewrite and shared wall helpers.

This module is deliberately patch-independent.  It reconstructs connected open
regions on a conformingly refined carrier mesh using the *current retained
surface graph*, then adds transverse walls only to complete pores satisfying
both

    A_pore > config.anisotropy_cap_min_area_mm2
    AR     > config.anisotropy_max_aspect_ratio

The transverse wall is constructed across the minor PCA direction of the
complete pore and may cross artificial patch seams.  It is therefore a true
post-patch graph rewrite rather than a chart-local surrogate.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from global_surface_pore_audit import (
    _conforming_subdivide_to_edge,
    _sample_solid_mask,
)


@dataclass
class _State:
    carrier: trimesh.Trimesh
    vertices: np.ndarray
    faces: np.ndarray
    tri: np.ndarray
    face_area: np.ndarray
    open_face: np.ndarray
    open_ids: np.ndarray
    open_index: np.ndarray
    component_label: np.ndarray
    component_count: int
    area: np.ndarray
    wet_perimeter: np.ndarray
    equivalent_radius: np.ndarray
    touches_boundary: np.ndarray
    complete: np.ndarray
    aspect_ratio: np.ndarray
    centroid: np.ndarray
    eigvecs: List[Optional[np.ndarray]]
    interface_edges: np.ndarray
    interface_component: np.ndarray
    eligible: np.ndarray
    radius_exceeded: np.ndarray
    zero_area: np.ndarray


class GlobalARCapRewriteEngine:
    def __init__(self, builder):
        self.builder = builder
        cfg = builder.config
        self.audit_edge = float(cfg.global_ar_cap_audit_max_edge_mm)
        self.min_area = float(cfg.anisotropy_cap_min_area_mm2)
        self.max_ar = float(cfg.anisotropy_max_aspect_ratio)
        self.zero_tol = float(cfg.zero_pore_min_area_mm2)
        self.radius_accept = float(cfg.target_hydraulic_radius_mm) * (
            1.0 + float(cfg.final_graph_radius_tolerance_fraction)
        )
        self.surface_radius = 0.5 * float(cfg.surface_diameter_mm)
        self.max_passes = max(1, int(cfg.global_ar_cap_max_passes))
        self.max_walls_per_pass = max(1, int(cfg.global_ar_cap_max_walls_per_pass))
        self.anchor_padding = max(0.0, float(cfg.global_ar_cap_anchor_search_padding_mm))
        self.min_improvement = max(0.0, float(cfg.global_ar_cap_min_improvement_fraction))
        self.logs: List[Dict] = []
        self.added_wall_count = 0

        self.carrier = _conforming_subdivide_to_edge(
            builder.original_mesh, self.audit_edge
        )
        self.vertices = np.asarray(self.carrier.vertices, dtype=float)
        self.faces = np.asarray(self.carrier.faces, dtype=np.int64)
        self.tri = self.vertices[self.faces]
        self.face_area = 0.5 * np.linalg.norm(
            np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0]),
            axis=1,
        )
        self.face_adj = np.asarray(self.carrier.face_adjacency, dtype=np.int64)
        self.face_adj_edges = np.asarray(self.carrier.face_adjacency_edges, dtype=np.int64)

        # True carrier boundaries; patch IDs are intentionally irrelevant.
        edge_inv = np.asarray(self.carrier.edges_unique_inverse, dtype=np.int64)
        unique_counts = np.bincount(edge_inv, minlength=len(self.carrier.edges_unique))
        boundary_uid_mask = unique_counts == 1
        boundary_occ = np.flatnonzero(boundary_uid_mask[edge_inv])
        self.boundary_perimeter_face = np.zeros(len(self.faces), dtype=float)
        if len(boundary_occ):
            boundary_face = boundary_occ // 3
            boundary_uid = edge_inv[boundary_occ]
            boundary_edges = np.asarray(self.carrier.edges_unique, dtype=np.int64)[boundary_uid]
            boundary_length = np.linalg.norm(
                self.vertices[boundary_edges[:, 1]] - self.vertices[boundary_edges[:, 0]],
                axis=1,
            )
            np.add.at(self.boundary_perimeter_face, boundary_face, boundary_length)

    # ------------------------------------------------------------------
    # retained-graph -> complete-pore state
    # ------------------------------------------------------------------
    def _segments(self):
        nodes = self.builder.atoms.drop_duplicates("atom_id").set_index("atom_id")
        edges = self.builder.edges.copy()
        if "bond_type" in edges.columns:
            kinds = edges["bond_type"].fillna("").astype(str).str.lower()
            surface = edges.loc[kinds.eq("surface")].copy()
            if surface.empty:
                surface = edges.copy()
        else:
            surface = edges.copy()
        aa, bb, rr, row_ids = [], [], [], []
        for idx, row in surface.iterrows():
            s, t = int(row["source"]), int(row["target"])
            if s not in nodes.index or t not in nodes.index:
                continue
            rad = float(row.get("radius", self.surface_radius))
            if not np.isfinite(rad) or rad <= 0:
                continue
            a = nodes.loc[s][["x", "y", "z"]].to_numpy(float)
            b = nodes.loc[t][["x", "y", "z"]].to_numpy(float)
            if np.linalg.norm(b - a) <= 1.0e-12:
                continue
            aa.append(a); bb.append(b); rr.append(rad); row_ids.append(idx)
        if not aa:
            raise ValueError("No finite-radius surface segments for global AR-cap rewrite")
        return (
            np.asarray(aa, dtype=float), np.asarray(bb, dtype=float),
            np.asarray(rr, dtype=float), np.asarray(row_ids, dtype=object),
        )

    def state(self) -> _State:
        seg_a, seg_b, seg_r, _ = self._segments()
        midpoint_tree = cKDTree(0.5 * (seg_a + seg_b))

        solid_votes = np.zeros(len(self.faces), dtype=np.int8)
        sample_sets = (
            self.tri.mean(axis=1),
            0.5 * (self.tri[:, 0] + self.tri[:, 1]),
            0.5 * (self.tri[:, 1] + self.tri[:, 2]),
            0.5 * (self.tri[:, 2] + self.tri[:, 0]),
        )
        for pts in sample_sets:
            solid_votes += _sample_solid_mask(
                pts, seg_a, seg_b, seg_r, midpoint_tree,
                search_padding_mm=0.0,
            ).astype(np.int8)
        solid_face = solid_votes.astype(float) / 4.0 >= 0.5
        open_face = ~solid_face
        open_ids = np.flatnonzero(open_face)
        open_index = np.full(len(self.faces), -1, dtype=np.int64)
        open_index[open_ids] = np.arange(len(open_ids), dtype=np.int64)

        if len(open_ids) == 0:
            empty = np.empty(0, dtype=float)
            return _State(
                self.carrier, self.vertices, self.faces, self.tri, self.face_area,
                open_face, open_ids, open_index, np.empty(0, dtype=np.int64), 0,
                empty, empty, empty, np.empty(0, dtype=bool), np.empty(0, dtype=bool),
                empty, np.empty((0, 3)), [], np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=bool),
                np.empty(0, dtype=bool), np.empty(0, dtype=bool),
            )

        both_open = open_face[self.face_adj[:, 0]] & open_face[self.face_adj[:, 1]]
        rows = np.concatenate((
            open_index[self.face_adj[both_open, 0]],
            open_index[self.face_adj[both_open, 1]],
        ))
        cols = np.concatenate((
            open_index[self.face_adj[both_open, 1]],
            open_index[self.face_adj[both_open, 0]],
        ))
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(len(open_ids), len(open_ids)),
        ).tocsr()
        component_count, labels = connected_components(graph, directed=False, return_labels=True)

        area = np.bincount(labels, weights=self.face_area[open_ids], minlength=component_count)

        interface = open_face[self.face_adj[:, 0]] ^ open_face[self.face_adj[:, 1]]
        interface_edges = self.face_adj_edges[interface]
        interface_open_face = np.where(
            open_face[self.face_adj[interface, 0]],
            self.face_adj[interface, 0], self.face_adj[interface, 1],
        )
        interface_component = labels[open_index[interface_open_face]]
        interface_length = np.linalg.norm(
            self.vertices[interface_edges[:, 1]] - self.vertices[interface_edges[:, 0]], axis=1
        )
        wet = np.bincount(interface_component, weights=interface_length, minlength=component_count)
        req = np.divide(
            2.0 * area, wet,
            out=np.full(component_count, np.inf, dtype=float),
            where=wet > 1.0e-12,
        )

        boundary_open_faces = np.flatnonzero(open_face & (self.boundary_perimeter_face > 0))
        boundary_perim = np.zeros(component_count, dtype=float)
        if len(boundary_open_faces):
            np.add.at(
                boundary_perim,
                labels[open_index[boundary_open_faces]],
                self.boundary_perimeter_face[boundary_open_faces],
            )
        touches = boundary_perim > 0
        complete = ~touches

        centers = self.tri.mean(axis=1)
        centroid = np.zeros((component_count, 3), dtype=float)
        for axis in range(3):
            centroid[:, axis] = np.bincount(
                labels,
                weights=self.face_area[open_ids] * centers[open_ids, axis],
                minlength=component_count,
            ) / np.maximum(area, 1.0e-18)

        ar = np.full(component_count, np.nan, dtype=float)
        eigvecs: List[Optional[np.ndarray]] = [None] * component_count
        for cid in range(component_count):
            fids = open_ids[labels == cid]
            if not len(fids):
                continue
            pts = self.tri[fids].reshape(-1, 3)
            w = np.repeat(self.face_area[fids] / 3.0, 3)
            wsum = float(w.sum())
            if wsum <= 1.0e-18 or len(pts) < 3:
                ar[cid] = np.inf
                continue
            c = np.sum(pts * w[:, None], axis=0) / wsum
            d = pts - c
            cov = (d * w[:, None]).T @ d / wsum
            cov = 0.5 * (cov + cov.T)
            vals, vec = np.linalg.eigh(cov)
            order = np.argsort(vals)
            vals = np.maximum(vals[order], 0.0)
            vec = vec[:, order]
            eigvecs[cid] = vec
            ar[cid] = math.sqrt(float(vals[2]) / float(vals[1])) if vals[1] > 1.0e-18 else np.inf

        eligible = complete & (area > self.min_area + 1.0e-10) & (
            (~np.isfinite(ar)) | (ar > self.max_ar + 1.0e-10)
        )
        radius_exceeded = complete & (req > self.radius_accept + 1.0e-10)
        zero_area = complete & (area <= self.zero_tol + 1.0e-12)

        return _State(
            self.carrier, self.vertices, self.faces, self.tri, self.face_area,
            open_face, open_ids, open_index, labels, component_count,
            area, wet, req, touches, complete, ar, centroid, eigvecs,
            interface_edges, interface_component, eligible, radius_exceeded, zero_area,
        )

    # ------------------------------------------------------------------
    # geometry of one transverse wall
    # ------------------------------------------------------------------
    def _component_path(self, st: _State, cid: int) -> Optional[Tuple[np.ndarray, float]]:
        face_ids = st.open_ids[st.component_label == int(cid)]
        if len(face_ids) == 0:
            return None
        verts = np.unique(st.faces[face_ids].reshape(-1))
        if len(verts) < 3:
            return None
        center = st.centroid[int(cid)]
        eig = st.eigvecs[int(cid)]
        if eig is None:
            return None
        major = eig[:, 2]
        minor = eig[:, 1]

        iface = st.interface_edges[st.interface_component == int(cid)]
        if len(iface) < 2:
            return None
        mids = 0.5 * (st.vertices[iface[:, 0]] + st.vertices[iface[:, 1]])
        md = mids - center
        mp = md @ major
        npj = md @ minor
        comp_pts = st.vertices[verts] - center
        major_span = max(float(np.ptp(comp_pts @ major)), 1.0e-9)
        minor_span = max(float(np.ptp(comp_pts @ minor)), 1.0e-9)

        # Build the within-pore carrier-vertex graph once. A mild major-axis
        # penalty is added per candidate shift so the path behaves like a
        # transverse cut rather than a generic shortest geodesic.
        edge_set = set()
        for tri in st.faces[face_ids]:
            a, b, c = map(int, tri)
            edge_set.add(tuple(sorted((a, b))))
            edge_set.add(tuple(sorted((b, c))))
            edge_set.add(tuple(sorted((c, a))))
        edges = np.asarray(sorted(edge_set), dtype=np.int64)
        if len(edges) == 0:
            return None
        local = {int(v): i for i, v in enumerate(verts)}

        best = None
        shifts = getattr(self, "shift_choices", (0.0, 0.10, -0.10, 0.20, -0.20))
        for sr in shifts:
            target_m = sr * major_span
            # Score opposite boundary sides, while keeping the cut close to the
            # requested major-axis station.
            major_pen = np.abs(mp - target_m) / major_span
            plus_score = npj / minor_span - 1.5 * major_pen
            minus_score = -npj / minor_span - 1.5 * major_pen
            ip = int(np.argmax(plus_score))
            im = int(np.argmax(minus_score))
            if ip == im or npj[ip] <= 0.0 or npj[im] >= 0.0:
                continue

            def choose_edge_vertex(edge, sign):
                cand = np.asarray(edge, dtype=int)
                pts = st.vertices[cand] - center
                pmajor = pts @ major
                pminor = pts @ minor
                # Prefer the intended side and the selected longitudinal station.
                score = sign * pminor / minor_span - 1.5 * np.abs(pmajor - target_m) / major_span
                return int(cand[int(np.argmax(score))])

            sv = choose_edge_vertex(iface[im], -1.0)
            tv = choose_edge_vertex(iface[ip], +1.0)
            if sv == tv or sv not in local or tv not in local:
                continue

            ea = st.vertices[edges[:, 0]]
            eb = st.vertices[edges[:, 1]]
            mid = 0.5 * (ea + eb)
            base_len = np.linalg.norm(eb - ea, axis=1)
            mid_major = (mid - center) @ major
            penalty = 1.0 + 2.0 * ((mid_major - target_m) / major_span) ** 2
            weights = base_len * penalty
            rr = np.concatenate(([local[int(a)] for a in edges[:, 0]], [local[int(b)] for b in edges[:, 1]]))
            cc = np.concatenate(([local[int(b)] for b in edges[:, 1]], [local[int(a)] for a in edges[:, 0]]))
            ww = np.concatenate((weights, weights))
            g = coo_matrix((ww, (rr, cc)), shape=(len(verts), len(verts))).tocsr()
            dist, pred = dijkstra(g, directed=False, indices=local[sv], return_predecessors=True)
            end = local[tv]
            if not np.isfinite(dist[end]):
                continue
            chain = [end]
            cur = end
            while cur != local[sv]:
                cur = int(pred[cur])
                if cur < 0:
                    chain = []
                    break
                chain.append(cur)
            if not chain:
                continue
            chain.reverse()
            path_ids = verts[np.asarray(chain, dtype=int)]
            path = st.vertices[path_ids]
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
            if length <= 1.0e-9:
                continue
            # Keep the shortest valid transverse option among the tested stations.
            if best is None or length < best[1]:
                best = (path, length)
        return best

    # ------------------------------------------------------------------
    # graph mutation helpers
    # ------------------------------------------------------------------
    def _nearest_graph_segment_projection(self, point: np.ndarray):
        nodes = self.builder.atoms.drop_duplicates("atom_id").set_index("atom_id")
        edges = self.builder.edges
        if "bond_type" in edges.columns:
            mask = edges["bond_type"].fillna("").astype(str).str.lower().eq("surface")
            work = edges.loc[mask]
        else:
            work = edges
        if work.empty:
            return None
        pairs = work[["source", "target"]].to_numpy(dtype=np.int64)
        valid = np.isin(pairs[:, 0], nodes.index) & np.isin(pairs[:, 1], nodes.index)
        work, pairs = work.iloc[np.flatnonzero(valid)], pairs[valid]
        if not len(pairs):
            return None
        xyz = nodes[["x", "y", "z"]]
        a = xyz.loc[pairs[:, 0]].to_numpy(dtype=float)
        b = xyz.loc[pairs[:, 1]].to_numpy(dtype=float)
        ab = b - a
        den = np.einsum("ij,ij->i", ab, ab)
        valid = den > 1.0e-18
        work, a, ab, den = work.iloc[np.flatnonzero(valid)], a[valid], ab[valid], den[valid]
        if not len(den):
            return None
        u = np.clip(np.einsum("ij,ij->i", point - a, ab) / den, 0.0, 1.0)
        q = a + u[:, None] * ab
        d = np.linalg.norm(point - q, axis=1)
        nearest = int(np.argmin(d))
        max_d = self.surface_radius + self.anchor_padding + self.audit_edge
        return (float(d[nearest]), work.index[nearest], work.iloc[nearest].copy(),
                q[nearest], float(u[nearest])) if d[nearest] <= max_d else None

    def _new_node_row(self, atom_id: int, xyz: np.ndarray, role: str,
                      component_id: int = 0, memberships: str = "") -> Dict:
        row = {c: np.nan for c in self.builder.atoms.columns}
        row.update({
            "atom_id": int(atom_id), "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
            "face_name": "surface", "patch_id": -1, "component_id": int(component_id),
            "source_vertex_id": -1, "uv_x": np.nan, "uv_y": np.nan,
            "node_role": role, "patch_memberships": memberships,
        })
        return row

    def _split_edge_at(self, nearest) -> Optional[int]:
        if nearest is None:
            return None
        dist, idx, row, q, u = nearest
        s, t = int(row["source"]), int(row["target"])
        nodes = self.builder.atoms.drop_duplicates("atom_id").set_index("atom_id")
        a = nodes.loc[s][["x", "y", "z"]].to_numpy(float)
        b = nodes.loc[t][["x", "y", "z"]].to_numpy(float)
        tol = 1.0e-5
        if float(np.linalg.norm(q - a)) <= tol:
            return s
        if float(np.linalg.norm(q - b)) <= tol:
            return t

        new_id = int(self.builder.atoms["atom_id"].max()) + 1 if len(self.builder.atoms) else 0
        memberships = str(row.get("patch_memberships", "")) if pd.notna(row.get("patch_memberships", "")) else ""
        comp = int(row.get("component_id", 0)) if pd.notna(row.get("component_id", 0)) else 0
        self.builder.atoms = pd.concat([
            self.builder.atoms,
            pd.DataFrame([self._new_node_row(
                new_id, q, getattr(self, 'wall_anchor_role', 'global_ar_cap_anchor'),
                comp, memberships)])
        ], ignore_index=True)

        # Replace exactly one exported segment by two segments while retaining
        # its logical-edge identity and metadata.
        self.builder.edges = self.builder.edges.drop(index=idx)
        r1 = row.copy(); r2 = row.copy()
        r1["source"], r1["target"] = s, new_id
        r2["source"], r2["target"] = new_id, t
        self.builder.edges = pd.concat([
            self.builder.edges,
            pd.DataFrame([r1, r2], columns=self.builder.edges.columns),
        ], ignore_index=True)

        if not self.builder.edge_metrics.empty and "logical_edge_id" in row:
            lid = int(row["logical_edge_id"])
            mask = self.builder.edge_metrics["logical_edge_id"].astype(int).eq(lid)
            if mask.any() and "polyline_segments" in self.builder.edge_metrics.columns:
                self.builder.edge_metrics.loc[mask, "polyline_segments"] = (
                    pd.to_numeric(self.builder.edge_metrics.loc[mask, "polyline_segments"], errors="coerce").fillna(0) + 1
                )
        return new_id

    def _append_wall(self, path: np.ndarray, cid: int) -> Tuple[bool, float]:
        def reject(reason: str) -> Tuple[bool, float]:
            self.last_rejection_reason = reason
            return False, 0.0
        self.last_rejection_reason = None
        self.last_surface_error_mm = None
        if path is None or len(path) < 2:
            return reject("empty_path")
        # Attach the path to existing retained graph walls on both sides.
        near_a = self._nearest_graph_segment_projection(path[0])
        near_b = self._nearest_graph_segment_projection(path[-1])
        if near_a is None or near_b is None:
            return reject("no_graph_anchor")
        if near_a[1] == near_b[1] and np.linalg.norm(near_a[3] - near_b[3]) < float(self.builder.config.anisotropy_min_wall_length_mm):
            return reject("anchors_on_same_short_segment")
        trial_points = np.vstack((near_a[3], path, near_b[3]))
        probes = np.concatenate([
            trial_points[:-1] * (1.0 - t) + trial_points[1:] * t
            for t in (0.25, 0.5, 0.75)
        ])
        _, deviation, _ = trimesh.proximity.closest_point(self.builder.original_mesh, probes)
        maximum_surface_error = float(np.max(deviation))
        tolerance = float(getattr(self, 'surface_fit_tolerance_mm',
                                  self.builder.config.surface_tolerance_mm))
        if (maximum_surface_error > tolerance
                and getattr(self, 'project_anchor_connectors', False)):
            # The carrier path already follows the input surface. A straight
            # final hop to a retained rod may cut through a curved fold, so
            # sample and project only those two short connector pieces.
            ts = np.arange(1, 8, dtype=float) / 8.0
            start_raw = ((1.0 - ts[:, None]) * near_a[3] +
                         ts[:, None] * path[0])
            end_raw = ((1.0 - ts[:, None]) * path[-1] +
                       ts[:, None] * near_b[3])
            start_projected, _, _ = trimesh.proximity.closest_point(
                self.builder.original_mesh, start_raw)
            end_projected, _, _ = trimesh.proximity.closest_point(
                self.builder.original_mesh, end_raw)
            path = np.vstack((start_projected, path, end_projected))
            trial_points = np.vstack((near_a[3], path, near_b[3]))
            probes = np.concatenate([
                trial_points[:-1] * (1.0 - t) + trial_points[1:] * t
                for t in (0.25, 0.5, 0.75)
            ])
            _, deviation, _ = trimesh.proximity.closest_point(
                self.builder.original_mesh, probes)
            maximum_surface_error = float(np.max(deviation))
        self.last_surface_error_mm = maximum_surface_error
        if maximum_surface_error > tolerance:
            return reject("surface_fit")

        anchor_a = self._split_edge_at(near_a)
        # Recompute second anchor after the first split because dataframe row
        # indices have changed.
        near_b = self._nearest_graph_segment_projection(path[-1])
        anchor_b = self._split_edge_at(near_b)
        if anchor_a is None or anchor_b is None or anchor_a == anchor_b:
            return reject("invalid_or_identical_anchor")

        node_table = self.builder.atoms.drop_duplicates("atom_id").set_index("atom_id")
        comp = int(node_table.loc[anchor_a, "component_id"])
        if int(node_table.loc[anchor_b, "component_id"]) != comp:
            return reject("different_graph_components")
        a_xyz = node_table.loc[anchor_a][["x", "y", "z"]].to_numpy(float)
        b_xyz = node_table.loc[anchor_b][["x", "y", "z"]].to_numpy(float)
        pts = np.vstack([a_xyz, path, b_xyz])
        # Remove consecutive duplicates.
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1.0e-6:
                keep.append(i)
        pts = pts[keep]
        if len(pts) < 2:
            return reject("collapsed_wall_path")
        wall_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if wall_len < float(self.builder.config.anisotropy_min_wall_length_mm):
            return reject("wall_below_minimum_length")

        ids = [anchor_a]
        next_id = int(self.builder.atoms["atom_id"].max()) + 1 if len(self.builder.atoms) else 0
        new_nodes = []
        for p in pts[1:-1]:
            nid = next_id; next_id += 1
            new_nodes.append(self._new_node_row(
                nid, p, getattr(self, 'wall_sample_role', 'global_ar_cap_sample'), comp, ""))
            ids.append(nid)
        ids.append(anchor_b)
        if new_nodes:
            self.builder.atoms = pd.concat([self.builder.atoms, pd.DataFrame(new_nodes)], ignore_index=True)

        next_lid = int(pd.to_numeric(self.builder.edges.get("logical_edge_id", pd.Series([-1])), errors="coerce").max()) + 1
        edge_rows = []
        metric_rows = []
        for u, v, pa, pb in zip(ids[:-1], ids[1:], pts[:-1], pts[1:]):
            seg_len = float(np.linalg.norm(pb - pa))
            if seg_len <= 1.0e-8:
                continue
            row = {c: np.nan for c in self.builder.edges.columns}
            row.update({
                "source": int(u), "target": int(v),
                "radius": self.surface_radius, "bond_type": "surface",
                "role": getattr(self, 'wall_role', 'surface_global_ar_cap_wall'), "patch_id": -1,
                "patch_memberships": "", "component_id": int(comp),
                "logical_edge_id": int(next_lid),
            })
            edge_rows.append(row)
            m = {c: np.nan for c in self.builder.edge_metrics.columns}
            m.update({
                "logical_edge_id": int(next_lid), "patch_id": -1,
                "component_id": int(comp), "polyline_segments": 1,
                "max_chord_surface_error_mm": maximum_surface_error, "domain_violation": 0,
                "mapping_fallback": 0, "cross_component": 0,
                "anisotropy_wall": 1, "accepted": 1,
            })
            metric_rows.append(m)
            next_lid += 1
        if not edge_rows:
            return reject("no_positive_length_segments")
        self.builder.edges = pd.concat([
            self.builder.edges,
            pd.DataFrame(edge_rows, columns=self.builder.edges.columns),
        ], ignore_index=True)
        if metric_rows:
            self.builder.edge_metrics = pd.concat([
                self.builder.edge_metrics,
                pd.DataFrame(metric_rows, columns=self.builder.edge_metrics.columns),
            ], ignore_index=True)
        endpoint_degree = pd.concat((
            self.builder.edges["source"], self.builder.edges["target"]
        )).value_counts()
        if any(int(endpoint_degree.get(node_id, 0)) > int(self.builder.config.max_node_degree)
               for node_id in (anchor_a, anchor_b)):
            # Callers restore the saved graph when this candidate is rejected.
            return reject("anchor_degree_limit")
        self.added_wall_count += 1
        return True, wall_len

    # ------------------------------------------------------------------
    # closed-loop rewrite
    # ------------------------------------------------------------------
    def _close_tiny_complete_pores(self, current: _State) -> _State:
        """Close unresolved sub-threshold carrier faces with surface-fitted rods.

        This changes the retained graph. It does not reclassify a positive-area
        pore as zero or relax the minimum-area acceptance criterion.
        """
        for repair_id in range(1, 65):
            tiny = np.flatnonzero(current.zero_area & (current.area > 0.0))
            if not len(tiny):
                break
            cid = int(tiny[np.argmin(current.area[tiny])])
            face_ids = current.open_ids[current.component_label == cid]
            if len(face_ids) != 1:
                self.logs.append(dict(rewrite_pass=self.max_passes + repair_id,
                                      global_pore_id=cid, accepted=0,
                                      reason="tiny_pore_not_single_carrier_face"))
                break
            center = current.centroid[cid]
            nearest = self._nearest_graph_segment_projection(center)
            if nearest is None:
                self.logs.append(dict(rewrite_pass=self.max_passes + repair_id,
                                      global_pore_id=cid, accepted=0,
                                      reason="tiny_pore_no_near_surface_wall"))
                break
            s, t = int(nearest[2]["source"]), int(nearest[2]["target"])
            nodes = self.builder.atoms.drop_duplicates("atom_id").set_index("atom_id")
            endpoint = min((s, t), key=lambda i: float(np.linalg.norm(
                center - nodes.loc[i][["x", "y", "z"]].to_numpy(float))))
            degree = (int((self.builder.edges["source"] == endpoint).sum())
                      + int((self.builder.edges["target"] == endpoint).sum()))
            if degree >= int(self.builder.config.max_node_degree):
                self.logs.append(dict(rewrite_pass=self.max_passes + repair_id,
                                      global_pore_id=cid, accepted=0,
                                      reason="tiny_pore_anchor_degree_limit"))
                break
            q = np.asarray(nearest[3], dtype=float)
            endpoint_xyz = nodes.loc[endpoint][["x", "y", "z"]].to_numpy(float)
            probes = np.asarray((q, 0.5 * (q + center), center,
                                 0.5 * (center + endpoint_xyz)), dtype=float)
            distance = self.builder._surface_distance_batch(probes)
            if np.any(distance > float(self.builder.config.surface_tolerance_mm)):
                self.logs.append(dict(rewrite_pass=self.max_passes + repair_id,
                                      global_pore_id=cid, accepted=0,
                                      reason="tiny_pore_bridge_surface_fit_failed",
                                      max_surface_error_mm=float(np.max(distance))))
                break

            atoms_before = self.builder.atoms.copy(deep=True)
            edges_before = self.builder.edges.copy(deep=True)
            metrics_before = self.builder.edge_metrics.copy(deep=True)
            anchor = self._split_edge_at(nearest)
            if anchor is None or anchor == endpoint:
                self.builder.atoms, self.builder.edges, self.builder.edge_metrics = (
                    atoms_before, edges_before, metrics_before)
                break
            nodes = self.builder.atoms.drop_duplicates("atom_id").set_index("atom_id")
            component = int(nodes.loc[endpoint, "component_id"])
            if int(nodes.loc[anchor, "component_id"]) != component:
                self.builder.atoms, self.builder.edges, self.builder.edge_metrics = (
                    atoms_before, edges_before, metrics_before)
                break
            nid = int(self.builder.atoms["atom_id"].max()) + 1
            self.builder.atoms = pd.concat((self.builder.atoms, pd.DataFrame([
                self._new_node_row(nid, center, "global_tiny_pore_bridge", component, "")
            ])), ignore_index=True)
            next_lid = int(pd.to_numeric(self.builder.edges["logical_edge_id"],
                                          errors="coerce").max()) + 1
            edge_rows, metric_rows = [], []
            for u, v in ((anchor, nid), (nid, endpoint)):
                row = {c: np.nan for c in self.builder.edges.columns}
                row.update(source=int(u), target=int(v), radius=self.surface_radius,
                           bond_type="surface", role="surface_global_tiny_pore_bridge",
                           patch_id=-1, patch_memberships="", component_id=component,
                           logical_edge_id=next_lid)
                edge_rows.append(row)
                metric = {c: np.nan for c in self.builder.edge_metrics.columns}
                metric.update(logical_edge_id=next_lid, patch_id=-1,
                              component_id=component, polyline_segments=1,
                              max_chord_surface_error_mm=float(np.max(distance)),
                              domain_violation=0, mapping_fallback=0,
                              cross_component=0, anisotropy_wall=0, accepted=1)
                metric_rows.append(metric)
                next_lid += 1
            self.builder.edges = pd.concat((self.builder.edges,
                                            pd.DataFrame(edge_rows, columns=self.builder.edges.columns)),
                                           ignore_index=True)
            self.builder.edge_metrics = pd.concat((self.builder.edge_metrics,
                                                   pd.DataFrame(metric_rows, columns=self.builder.edge_metrics.columns)),
                                                  ignore_index=True)
            after = self.state()
            before_max = float(np.max(current.equivalent_radius[current.complete])) if np.any(current.complete) else 0.0
            after_max = float(np.max(after.equivalent_radius[after.complete])) if np.any(after.complete) else 0.0
            safe = (int(after.zero_area.sum()) < int(current.zero_area.sum())
                    and int(after.radius_exceeded.sum()) <= int(current.radius_exceeded.sum())
                    and after_max <= before_max + 1.0e-6)
            self.logs.append(dict(rewrite_pass=self.max_passes + repair_id,
                                  global_pore_id=cid, accepted=int(safe),
                                  reason="accepted_surface_fitted_tiny_pore_bridge" if safe else
                                         "tiny_pore_bridge_global_guard_failed",
                                  area_before_mm2=float(current.area[cid]),
                                  max_surface_error_mm=float(np.max(distance)),
                                  max_radius_before_mm=before_max,
                                  max_radius_after_mm=after_max))
            if not safe:
                self.builder.atoms, self.builder.edges, self.builder.edge_metrics = (
                    atoms_before, edges_before, metrics_before)
                break
            current = after
        return current

    def run(self):
        initial = self.state()
        initial_eligible = int(initial.eligible.sum())
        current = initial
        shifts = (0.0, 0.10, -0.10, 0.20, -0.20, 0.30, -0.30, 0.40, -0.40)

        def radius_excess(st):
            ratio = np.divide(st.equivalent_radius, self.radius_accept,
                              out=np.zeros_like(st.equivalent_radius),
                              where=np.isfinite(st.equivalent_radius))
            return float(np.sum(st.area * np.maximum(ratio - 1.0, 0.0) * st.complete))

        for pass_id in range(1, self.max_passes + 1):
            if not np.any(current.radius_exceeded):
                break
            accepted_this_pass = 0
            attempted = set()
            for _ in range(self.max_walls_per_pass):
                bad_ids = np.flatnonzero(current.radius_exceeded)
                bad_ids = np.asarray([i for i in bad_ids if int(i) not in attempted], dtype=int)
                if len(bad_ids) == 0:
                    break
                score = current.area[bad_ids] * np.nan_to_num(
                    np.maximum(current.equivalent_radius[bad_ids] / self.radius_accept - 1.0, 0.0),
                    nan=0.0, posinf=1.0e6)
                # A near-threshold pore may be closed with one short wall and
                # change the feasible cut topology of a larger neighbouring
                # pore. Try the least deficient complete pore first.
                cid = int(bad_ids[np.argmin(score)])
                attempted.add(cid)
                for shift in shifts:
                    self.shift_choices = (shift,)
                    path_info = self._component_path(current, cid)
                    if path_info is None:
                        self.logs.append(dict(rewrite_pass=pass_id, global_pore_id=cid,
                                              shift_fraction=shift, accepted=0,
                                              reason="no_transverse_surface_path"))
                        continue
                    atoms_before = self.builder.atoms.copy(deep=True)
                    edges_before = self.builder.edges.copy(deep=True)
                    metrics_before = self.builder.edge_metrics.copy(deep=True)
                    ok, wall_len = self._append_wall(path_info[0], cid)
                    if not ok:
                        self.builder.atoms, self.builder.edges, self.builder.edge_metrics = (
                            atoms_before, edges_before, metrics_before)
                        self.logs.append(dict(rewrite_pass=pass_id, global_pore_id=cid,
                                              shift_fraction=shift, accepted=0,
                                              reason="graph_attachment_failed"))
                        continue
                    after = self.state()
                    before_max = float(np.max(current.equivalent_radius[current.complete])) if np.any(current.complete) else 0.0
                    after_max = float(np.max(after.equivalent_radius[after.complete])) if np.any(after.complete) else 0.0
                    before_excess, after_excess = radius_excess(current), radius_excess(after)
                    before_bad = int(current.radius_exceeded.sum())
                    after_bad = int(after.radius_exceeded.sum())
                    safe = int(after.zero_area.sum()) <= int(current.zero_area.sum()) and after_max <= before_max + 1.0e-6
                    improved = (after_excess < before_excess - 1.0e-6 or after_bad < before_bad)
                    accepted = bool(safe and improved)
                    self.logs.append(dict(
                        rewrite_pass=pass_id, global_pore_id=cid,
                        shift_fraction=shift, accepted=int(accepted),
                        reason="accepted_global_transverse_wall" if accepted else
                               ("candidate_rejected_global_safety_guard" if not safe else
                                "candidate_rejected_no_global_improvement"),
                        area_before_mm2=float(current.area[cid]),
                        ar_before=float(current.aspect_ratio[cid]),
                        r_eq_before_mm=float(current.equivalent_radius[cid]),
                        wall_length_mm=float(wall_len),
                        eligible_before_pass=before_bad, eligible_after_pass=after_bad,
                        radius_excess_before_pass=before_excess,
                        radius_excess_after_pass=after_excess,
                        max_radius_before_mm=before_max, max_radius_after_mm=after_max))
                    if accepted:
                        current = after
                        accepted_this_pass += 1
                        attempted.clear()  # component IDs changed after the wall
                        break
                    self.builder.atoms, self.builder.edges, self.builder.edge_metrics = (
                        atoms_before, edges_before, metrics_before)
                    self.added_wall_count -= 1
            if accepted_this_pass == 0:
                break

        current = self._close_tiny_complete_pores(current)
        final = current
        self.builder.global_ar_cap_rewrite_history = pd.DataFrame(self.logs)
        self.builder.global_ar_cap_summary = {
            "enabled": True,
            "completed": True,
            "definition": (
                "global complete-pore transverse rewrite; physical equivalent-radius "
                f"limit {self.radius_accept:.6g} mm; AR diagnostic target applies when "
                f"A_pore > {self.min_area:.6g} mm^2 and AR > {self.max_ar:.6g}"
            ),
            "audit_max_edge_mm": self.audit_edge,
            "ar_cap_min_area_mm2": self.min_area,
            "ar_hard_max": self.max_ar,
            "initial_eligible_count": initial_eligible,
            "final_eligible_count": int(final.eligible.sum()),
            "initial_radius_exceeded_count": int(initial.radius_exceeded.sum()),
            "final_radius_exceeded_count": int(final.radius_exceeded.sum()),
            "initial_zero_area_complete_pore_count": int(initial.zero_area.sum()),
            "final_zero_area_complete_pore_count": int(final.zero_area.sum()),
            "added_wall_count": int(self.added_wall_count),
            "pass_count": int(max([r.get("rewrite_pass", 0) for r in self.logs], default=0)),
            "final_pass": bool(
                int(final.radius_exceeded.sum()) == 0
                and int(final.zero_area.sum()) == 0
            ),
        }
        self.builder.curr_vertices = self.builder.atoms[["x", "y", "z"]].to_numpy(float).tolist()
        self.builder.curr_edges = list(
            self.builder.edges[["source", "target"]].itertuples(index=False, name=None)
        ) if not self.builder.edges.empty else []
        return self.builder.global_ar_cap_summary


def apply_global_ar_cap_rewrite(builder):
    """Run the V11 patch-independent complete-pore AR-cap closed loop."""
    engine = GlobalARCapRewriteEngine(builder)
    return engine.run()
