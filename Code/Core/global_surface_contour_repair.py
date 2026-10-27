"""Repair complete surface pores measured with physical contour perimeters."""
from __future__ import annotations

from itertools import combinations
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from contour_core import RodDistanceQuery, contour_metrics
from global_surface_pore_audit import _conforming_subdivide_to_edge
from global_surface_pore_rewrite_V11 import GlobalARCapRewriteEngine


def _surface_segments(nodes: pd.DataFrame, edges: pd.DataFrame):
    if nodes.atom_id.isna().any() or nodes.atom_id.duplicated().any():
        raise ValueError('Contour repair requires unique, present atom IDs')
    node = nodes.set_index('atom_id')
    work = edges.loc[edges.bond_type.fillna('').astype(str).str.lower().eq('surface')]
    if work.empty:
        raise ValueError('Contour repair found no surface rods')
    if (~work.source.isin(node.index)).any() or (~work.target.isin(node.index)).any():
        raise ValueError('Contour repair found a missing surface-rod endpoint')
    pairs = work[['source', 'target']].to_numpy(dtype=np.int64)
    starts = node.loc[pairs[:, 0], ['x', 'y', 'z']].to_numpy(dtype=float)
    ends = node.loc[pairs[:, 1], ['x', 'y', 'z']].to_numpy(dtype=float)
    radius = work.radius.to_numpy(dtype=float)
    valid = np.isfinite(radius) & (radius > 0) & (np.linalg.norm(ends-starts,axis=1)>1e-12)
    if not valid.all() or not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError('Contour repair found an invalid surface rod')
    return starts, ends, radius


class ContourRadiusRepair:
    def __init__(self, builder, *, max_repairs=128, candidates_per_pore=24,
                 audit_edge_mm=None, repair_target_mm=None, local_refine=3):
        self.builder = builder
        self.config = builder.config
        self.max_repairs = int(max_repairs)
        self.candidates_per_pore = int(candidates_per_pore)
        self.edge_mm = float(audit_edge_mm if audit_edge_mm is not None
                             else self.config.global_ar_cap_audit_max_edge_mm)
        physical_limit = (float(self.config.target_hydraulic_radius_mm)
                          * float(self.config.hydraulic_radius_mapping_safety)
                          * (1.0+float(self.config.final_graph_radius_tolerance_fraction)))
        self.acceptance_mm = float(repair_target_mm if repair_target_mm is not None
                                   else physical_limit)
        self.local_refine = int(local_refine)
        if not np.isfinite(self.edge_mm) or self.edge_mm <= 0:
            raise ValueError('Contour repair carrier edge must be finite and positive')
        if not np.isfinite(self.acceptance_mm) or self.acceptance_mm <= 0:
            raise ValueError('Contour repair target radius must be finite and positive')
        if self.local_refine < 1 or self.local_refine > 3:
            raise ValueError('Contour repair refinement level must be 1, 2, or 3')
        self.minimum_area_mm2 = float(self.config.zero_pore_min_area_mm2)
        self.carrier = _conforming_subdivide_to_edge(builder.original_mesh,self.edge_mm)
        self.logs = []
        self.used_pairs=set()
        self._mesh_graph = None
        self._mesh_tree = None
        self._face_center_tree = None
        self._face_diameter_mm = None
        self._physical_boundary_keys = None

    def _geodesic_chain(self, pa, pb):
        """Follow the actual input surface when a short chord cuts through a fold."""
        mesh = getattr(self.builder, 'base_mesh', self.builder.original_mesh)
        if self._mesh_graph is None:
            vertices = np.asarray(mesh.vertices, dtype=float)
            pairs = np.asarray(mesh.edges_unique, dtype=np.int64)
            weights = np.linalg.norm(vertices[pairs[:, 0]]-vertices[pairs[:, 1]],axis=1)
            self._mesh_graph = sparse.csr_matrix((np.r_[weights,weights],
                (np.r_[pairs[:, 0],pairs[:, 1]],np.r_[pairs[:, 1],pairs[:, 0]])),
                shape=(len(vertices),len(vertices)))
            self._mesh_tree = cKDTree(vertices)
        vertices = np.asarray(mesh.vertices, dtype=float)
        points=np.vstack((pa,pb))
        closest,endpoint_distance,face_ids=trimesh.proximity.closest_point(mesh,points)
        if np.max(endpoint_distance)>float(self.config.contour_repair_surface_tolerance_mm):
            return None
        endpoint_faces=np.asarray(mesh.faces,dtype=np.int64)[face_ids]
        start,end=[int(face[np.argmin(np.linalg.norm(vertices[face]-point,axis=1))])
                   for face,point in zip(endpoint_faces,points)]
        distance, predecessor = dijkstra(self._mesh_graph,directed=False,
            indices=int(start),return_predecessors=True,
            limit=3*float(self.config.max_edge_length_mm))
        if not np.isfinite(distance[end]):
            return None
        path=[int(end)]
        while path[-1] != start:
            parent=int(predecessor[path[-1]])
            if parent<0: return None
            path.append(parent)
        chain=np.vstack((pa,vertices[path[::-1]],pb))
        keep=np.r_[True,np.linalg.norm(np.diff(chain,axis=0),axis=1)>1e-7]
        keep[-1]=True
        return chain[keep]

    def measure(self):
        a,b,r = _surface_segments(self.builder.atoms,self.builder.edges)
        signed = RodDistanceQuery(a,b,r,self.edge_mm)
        phi = signed(np.asarray(self.carrier.vertices,dtype=float))
        result = contour_metrics(self.carrier,phi,signed_fn=signed,
                                 local_refine=self.local_refine)
        result['signed_distance_at_vertices']=phi
        complete = (~result['touches_boundary']) & (result['area'] >= self.minimum_area_mm2-1e-10)
        bad = complete & (result['radius'] > self.acceptance_mm+1e-10)
        return result,complete,bad

    def measure_local(self, previous, old_edge_count):
        """Recompute components near newly inserted rods; preserve untouched pores."""
        old, _, _ = previous
        new_edges = self.builder.edges.iloc[old_edge_count:]
        if new_edges.empty:
            return previous
        a, b, r = _surface_segments(self.builder.atoms, new_edges)
        faces = np.asarray(self.carrier.faces, dtype=np.int64)
        vertices = np.asarray(self.carrier.vertices, dtype=float)
        if self._face_center_tree is None:
            tri = vertices[faces]
            centers = tri.mean(axis=1)
            self._face_center_tree = cKDTree(centers)
            self._face_diameter_mm = float(np.max(np.linalg.norm(
                tri[:, [1, 2, 0]] - tri[:, [0, 1, 2]], axis=2)))
            inv = np.asarray(self.carrier.edges_unique_inverse, dtype=np.int64)
            counts = np.bincount(inv, minlength=len(self.carrier.edges_unique))
            boundary = np.asarray(self.carrier.edges_unique, dtype=np.int64)[
                np.flatnonzero(counts == 1)]
            self._physical_boundary_keys = boundary[:, 0] * len(vertices) + boundary[:, 1]
        nearby = set()
        for start, end, radius in zip(a, b, r):
            mid = .5 * (start + end)
            reach = (.5 * float(np.linalg.norm(end - start)) + float(radius)
                     + self._face_diameter_mm + 1e-8)
            nearby.update(self._face_center_tree.query_ball_point(mid, reach))
        if not nearby:
            return previous
        affected = np.unique(old['face_component_id'][list(nearby)])
        affected = affected[affected >= 0]
        if not len(affected):
            return previous
        selected = np.flatnonzero(np.isin(old['face_component_id'], affected))
        if len(selected) > .7 * len(faces):
            return self.measure()
        query = RodDistanceQuery(*_surface_segments(self.builder.atoms,
                                                     self.builder.edges), self.edge_mm)
        phi = old['signed_distance_at_vertices'].copy()
        vertex_ids = np.unique(faces[selected])
        phi[vertex_ids] = query(vertices[vertex_ids])
        local_mesh = trimesh.Trimesh(vertices=vertices, faces=faces[selected],
                                     process=False)
        local = contour_metrics(local_mesh, phi, signed_fn=query,
                                local_refine=self.local_refine,
                                physical_boundary_edges=self._physical_boundary_keys)
        retained = np.flatnonzero(~np.isin(np.arange(old['component_count']), affected))
        component_keys = ('area', 'perimeter', 'radius', 'shape', 'aspect',
                          'centroids', 'touches_boundary', 'face_count')
        result = {key: np.concatenate((old[key][retained], local[key]))
                  for key in component_keys}
        result['component_count'] = len(result['area'])
        labels = np.full(len(faces), -1, dtype=np.int32)
        for new_id, old_id in enumerate(retained):
            labels[old['face_component_id'] == old_id] = new_id
        local_labels = local['face_component_id']
        active_local = local_labels >= 0
        labels[selected[active_local]] = local_labels[active_local] + len(retained)
        result['face_component_id'] = labels
        result['open_area_face'] = old['open_area_face'].copy()
        result['open_area_face'][selected] = local['open_area_face']
        result['carrier_area_face'] = old['carrier_area_face']
        result['open_area_total'] = float(result['open_area_face'].sum())
        result['carrier_area_total'] = old['carrier_area_total']
        result['signed_distance_at_vertices'] = phi
        complete = (~result['touches_boundary']) & (
            result['area'] >= self.minimum_area_mm2 - 1e-10)
        bad = complete & (result['radius'] > self.acceptance_mm + 1e-10)
        return result, complete, bad

    def candidate_pairs(self, center):
        nodes = self.builder.atoms.drop_duplicates('atom_id').copy()
        xyz = nodes[['x','y','z']].to_numpy(float)
        radius = max(1.5,3*self.acceptance_mm)
        near = cKDTree(xyz).query_ball_point(center,radius)
        near = sorted(near,key=lambda i:np.linalg.norm(xyz[i]-center))[:64]
        degrees = pd.concat([self.builder.edges.source,self.builder.edges.target]).value_counts()
        existing = {tuple(sorted((int(a),int(b)))) for a,b in
                    self.builder.edges[['source','target']].to_numpy()}
        scored=[]
        for i,j in combinations(near,2):
            na,nb=nodes.iloc[i],nodes.iloc[j]
            a_id,b_id=int(na.atom_id),int(nb.atom_id)
            if int(na.component_id)!=int(nb.component_id): continue
            if int(na.patch_id)<0 or int(nb.patch_id)<0: continue
            if tuple(sorted((a_id,b_id))) in existing: continue
            if tuple(sorted((a_id,b_id))) in self.used_pairs: continue
            if int(degrees.get(a_id,0))>=int(self.config.max_node_degree): continue
            if int(degrees.get(b_id,0))>=int(self.config.max_node_degree): continue
            pa,pb=xyz[i],xyz[j]
            va,vb=pa-center,pb-center
            length=float(np.linalg.norm(pa-pb))
            if not (float(self.config.min_logical_edge_length_mm)<=length<=float(self.config.max_edge_length_mm)): continue
            if float(va@vb)>-0.1: continue
            mid=float(np.linalg.norm(.5*(pa+pb)-center))
            scored.append((mid+.05*length,a_id,b_id))
        scored.sort()
        return [(a,b) for _,a,b in scored[:self.candidates_per_pore]]

    def _transverse_paths(self, measured, cid, angle_deg=0.0):
        """Find cuts inside one actual open component on the original carrier."""
        face_ids = np.flatnonzero(measured['face_component_id'] == cid)
        if not len(face_ids):
            return []
        face = np.asarray(self.carrier.faces, dtype=np.int64)[face_ids]
        edges = np.vstack((face[:, [0, 1]], face[:, [1, 2]], face[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        phi = measured['signed_distance_at_vertices']
        crossing = ((phi[edges[:, 0]] > 0) & (phi[edges[:, 1]] <= 0)) | (
            (phi[edges[:, 1]] > 0) & (phi[edges[:, 0]] <= 0))
        boundary = np.unique(np.where(phi[edges[crossing]] > 0,
                                      edges[crossing], -1))
        boundary = boundary[boundary >= 0]
        positive = edges[(phi[edges[:, 0]] > 0) & (phi[edges[:, 1]] > 0)]
        vertices = np.unique(positive)
        if len(boundary) < 2 or len(vertices) < 3:
            return []
        xyz = np.asarray(self.carrier.vertices, dtype=float)
        center = measured['centroids'][cid]
        points = xyz[vertices] - center
        covariance = points.T @ points / len(points)
        eigval, eigvec = np.linalg.eigh(covariance)
        angle = np.deg2rad(angle_deg)
        major = np.cos(angle)*eigvec[:, 2] + np.sin(angle)*eigvec[:, 1]
        minor = -np.sin(angle)*eigvec[:, 2] + np.cos(angle)*eigvec[:, 1]
        bpoint = xyz[boundary] - center
        bmajor, bminor = bpoint @ major, bpoint @ minor
        major_span = max(np.ptp(points @ major), 1e-9)
        minor_span = max(np.ptp(points @ minor), 1e-9)
        lookup = {int(v): k for k, v in enumerate(vertices)}
        weights = np.linalg.norm(xyz[positive[:, 0]] - xyz[positive[:, 1]], axis=1)
        paths = []
        for offset in (0.0, .1, -.1, .2, -.2, .3, -.3, .4, -.4):
            station_paths = []
            station = offset * major_span
            minus_score = -bminor / minor_span - 1.5 * np.abs(bmajor-station) / major_span
            plus_score = bminor / minor_span - 1.5 * np.abs(bmajor-station) / major_span
            left = boundary[np.argsort(minus_score)[-4:][::-1]]
            right = boundary[np.argsort(plus_score)[-4:][::-1]]
            left = left[((xyz[left]-center) @ minor) < 0]
            right = right[((xyz[right]-center) @ minor) > 0]
            mid = .5 * (xyz[positive[:, 0]] + xyz[positive[:, 1]])
            penalty = 1.0 + 2.0 * (((mid-center) @ major-station) / major_span) ** 2
            ii = np.fromiter((lookup[int(v)] for v in positive[:, 0]), dtype=np.int64)
            jj = np.fromiter((lookup[int(v)] for v in positive[:, 1]), dtype=np.int64)
            graph = sparse.csr_matrix((np.r_[weights*penalty, weights*penalty],
                                       (np.r_[ii, jj], np.r_[jj, ii])),
                                      shape=(len(vertices), len(vertices)))
            for start in left:
                if int(start) not in lookup:
                    continue
                distances, predecessors = dijkstra(graph, directed=False,
                    indices=lookup[int(start)], return_predecessors=True,
                    limit=4 * float(self.config.max_edge_length_mm))
                for stop in right:
                    if int(stop) not in lookup or int(stop) == int(start):
                        continue
                    end = lookup[int(stop)]
                    if not np.isfinite(distances[end]):
                        continue
                    chain = [end]
                    while chain[-1] != lookup[int(start)]:
                        parent = int(predecessors[chain[-1]])
                        if parent < 0:
                            chain = []
                            break
                        chain.append(parent)
                    if chain:
                        path = xyz[vertices[np.asarray(chain[::-1], dtype=int)]]
                        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                        if length >= float(self.config.anisotropy_min_wall_length_mm):
                            station_paths.append((length, path))
            station_paths.sort(key=lambda item: item[0])
            paths.extend(station_paths[:max(1, self.candidates_per_pore // 9)])
        # Preserve the centre-first station order. Sorting all stations by
        # length favours tiny cuts near the rim and repeatedly misses the
        # large open core that sets the hydraulic-radius limit.
        return [path for _, path in paths[:self.candidates_per_pore]]

    def _append_transverse_wall(self, path, cid):
        helper = object.__new__(GlobalARCapRewriteEngine)
        helper.builder = self.builder
        helper.audit_edge = self.edge_mm
        helper.surface_radius = .5 * float(self.config.surface_diameter_mm)
        helper.anchor_padding = float(self.config.global_ar_cap_anchor_search_padding_mm)
        helper.added_wall_count = 0
        helper.wall_role = 'surface_global_contour_radius_wall'
        helper.wall_sample_role = 'global_contour_radius_sample'
        helper.wall_anchor_role = 'global_contour_radius_anchor'
        helper.project_anchor_connectors = True
        helper.surface_fit_tolerance_mm = float(self.config.contour_repair_surface_tolerance_mm)
        result = helper._append_wall(path, cid)
        self.last_rejection_reason = getattr(helper, 'last_rejection_reason', None)
        self.last_surface_error_mm = getattr(helper, 'last_surface_error_mm', None)
        return result

    def _allowed_open_area_loss(self, measured, wall_length):
        radius = .5 * float(self.config.surface_diameter_mm)
        physical_wall_area = 1.3 * (2 * radius) * wall_length + np.pi * radius**2
        return max(.01 * measured['carrier_area_total'], physical_wall_area)

    def _bridge(self, a_id, b_id):
        nodes=self.builder.atoms.drop_duplicates('atom_id').set_index('atom_id')
        a,b=nodes.loc[[a_id,b_id]].itertuples(index=False)
        pa=nodes.loc[a_id,['x','y','z']].to_numpy(float)
        pb=nodes.loc[b_id,['x','y','z']].to_numpy(float)
        ts=np.array([.25,.5,.75])
        raw=(1-ts[:,None])*pa+ts[:,None]*pb
        mapping_mesh=getattr(self.builder, 'base_mesh', self.builder.original_mesh)
        projected,_,_=trimesh.proximity.closest_point(mapping_mesh,raw)
        chain=np.vstack((pa,projected,pb))
        probes=np.concatenate([chain[:-1]*(1-t)+chain[1:]*t for t in (.25,.5,.75)])
        _,deviation,_=trimesh.proximity.closest_point(mapping_mesh,probes)
        maximum=float(np.max(deviation))
        if maximum>float(self.config.contour_repair_surface_tolerance_mm):
            chain=self._geodesic_chain(pa,pb)
            if chain is None: return False,maximum
            probes=np.concatenate([chain[:-1]*(1-t)+chain[1:]*t for t in (.25,.5,.75)])
            _,deviation,_=trimesh.proximity.closest_point(mapping_mesh,probes)
            maximum=float(np.max(deviation))
            if maximum>float(self.config.contour_repair_surface_tolerance_mm): return False,maximum
            progress=np.r_[0,np.cumsum(np.linalg.norm(np.diff(chain,axis=0),axis=1))]
            ts=progress[1:-1]/progress[-1]
            projected=chain[1:-1]
        first=int(self.builder.atoms.atom_id.max())+1
        ids=[a_id,*range(first,first+len(projected)),b_id]
        patch=int(nodes.loc[a_id,'patch_id'])
        other_patch=int(nodes.loc[b_id,'patch_id'])
        memberships=';'.join(str(k) for k in sorted({patch,other_patch}))
        component=int(nodes.loc[a_id,'component_id'])
        add_nodes=[]
        for t,p,node_id in zip(ts,projected,ids[1:-1]):
            row=nodes.loc[a_id].copy()
            row['atom_id']=node_id
            row[['x','y','z']]=p
            row['uv_x']=(1-t)*float(nodes.loc[a_id,'uv_x'])+t*float(nodes.loc[b_id,'uv_x'])
            row['uv_y']=(1-t)*float(nodes.loc[a_id,'uv_y'])+t*float(nodes.loc[b_id,'uv_y'])
            row['source_vertex_id']=-1
            row['node_role']='global_contour_radius_sample'
            add_nodes.append(row)
        self.builder.atoms=pd.concat([self.builder.atoms,pd.DataFrame(add_nodes)],ignore_index=True)
        logical_id=int(pd.to_numeric(self.builder.edges.logical_edge_id,errors='coerce').max())+1
        add_edges=[]
        for u,v in zip(ids[:-1],ids[1:]):
            row={col:np.nan for col in self.builder.edges.columns}
            row.update(source=u,target=v,radius=.5*float(self.config.surface_diameter_mm),
                       bond_type='surface',role='surface_global_contour_radius_wall',
                       patch_id=patch,patch_memberships=memberships,component_id=component,
                       logical_edge_id=logical_id)
            add_edges.append(row)
        self.builder.edges=pd.concat([self.builder.edges,pd.DataFrame(add_edges)],ignore_index=True)
        if hasattr(self.builder,'edge_metrics') and not self.builder.edge_metrics.empty:
            metric={col:np.nan for col in self.builder.edge_metrics.columns}
            metric.update(logical_edge_id=logical_id,patch_id=patch,component_id=component,
                          polyline_segments=len(ids)-1,max_chord_surface_error_mm=maximum,
                          domain_violation=0,mapping_fallback=0,cross_component=0,
                          anisotropy_wall=0,accepted=1)
            self.builder.edge_metrics=pd.concat([self.builder.edge_metrics,pd.DataFrame([metric])],ignore_index=True)
        return True,maximum

    def run(self, initial_measurement=None):
        measured,complete,bad=(initial_measurement if initial_measurement is not None
                               else self.measure())
        def pore_state(result, pore_id):
            # Component IDs can change after another pore is split. Geometry
            # identifies an unchanged search target across those renumberings.
            values=[result['radius'][pore_id],result['area'][pore_id],
                    *np.asarray(result['centroids'][pore_id],dtype=float)]
            return tuple(round(float(value),9) for value in values)
        initial_bad=int(np.count_nonzero(bad))
        initial_max=float(np.max(measured['radius'][complete])) if np.any(complete) else None
        accepted=0
        stalled=set()
        attempted_paths=set()
        diagnostic_dir=os.environ.get('PC_AEGG_CONTOUR_DIAGNOSTIC_DIR')
        progress_path=None
        if diagnostic_dir:
            directory=Path(diagnostic_dir)
            directory.mkdir(parents=True,exist_ok=True)
            phase=('fine' if abs(self.edge_mm-float(self.config.global_ar_cap_audit_max_edge_mm))<1e-9
                   else 'coarse')
            progress_path=directory/f'{phase}_repair_progress.csv'
            progress_path.unlink(missing_ok=True)

        def record_progress(stage, step):
            if progress_path is None:
                return
            radii=np.asarray(measured['radius'][bad],dtype=float)
            pd.DataFrame([dict(stage=stage,step=step,accepted_walls=accepted,
                               exceeded_count=len(radii),
                               maximum_radius_mm=float(np.max(radii)) if len(radii) else None,
                               minimum_radius_mm=float(np.min(radii)) if len(radii) else None,
                               median_radius_mm=float(np.median(radii)) if len(radii) else None)]).to_csv(
                progress_path,mode='a',header=not progress_path.exists(),index=False)

        record_progress('initial',0)
        # Each unchanged pore is exhausted only after all current bridge and
        # oriented-wall candidates have been tried once.
        repair_limit = max(self.max_repairs, 512, 8 * initial_bad)
        for step in range(repair_limit):
            if not np.any(bad): break
            candidates=[int(i) for i in np.flatnonzero(bad)
                        if pore_state(measured,int(i)) not in stalled]
            if not candidates: break
            cid=max(candidates,key=lambda i:measured['radius'][i])
            state=pore_state(measured,cid)
            member_faces=np.flatnonzero(measured['face_component_id']==cid)
            if len(member_faces):
                faces=np.asarray(self.carrier.faces,dtype=np.int64)[member_faces]
                clearance=measured['signed_distance_at_vertices'][faces].mean(axis=1)
                center=np.asarray(self.carrier.triangles_center)[member_faces[np.argmax(clearance)]]
            else:
                center=measured['centroids'][cid]
            improved=False
            for a_id,b_id in self.candidate_pairs(center):
                candidate_key=(state,'bridge',tuple(sorted((a_id,b_id))))
                if candidate_key in attempted_paths: continue
                attempted_paths.add(candidate_key)
                old_atoms=self.builder.atoms.copy(deep=True)
                old_edges=self.builder.edges.copy(deep=True)
                old_metrics=self.builder.edge_metrics.copy(deep=True)
                endpoint=old_atoms.drop_duplicates('atom_id').set_index('atom_id')
                wall_length=float(np.linalg.norm(
                    endpoint.loc[a_id,['x','y','z']].to_numpy(float)-
                    endpoint.loc[b_id,['x','y','z']].to_numpy(float)))
                bridge_ok,deviation=self._bridge(a_id,b_id)
                if not bridge_ok:
                    self.logs.append(dict(step=step+1,pore_id=cid,endpoint_a=a_id,
                                          endpoint_b=b_id,accepted=False,reason='surface_fit',
                                          max_surface_deviation_mm=deviation))
                    continue
                after,after_complete,after_bad=self.measure_local(
                    (measured,complete,bad), len(old_edges))
                before_max=float(np.max(measured['radius'][complete]))
                after_max=float(np.max(after['radius'][after_complete]))
                before_excess=float(np.maximum(measured['radius'][complete]-self.acceptance_mm,0).sum())
                after_excess=float(np.maximum(after['radius'][after_complete]-self.acceptance_mm,0).sum())
                safe=((int(np.count_nonzero(after_bad))<int(np.count_nonzero(bad))
                       or after_excess<before_excess-1e-5)
                      and after_max<=before_max+1e-6
                      and after['open_area_total']>=measured['open_area_total']-
                          self._allowed_open_area_loss(measured,wall_length))
                self.logs.append(dict(step=step+1,pore_id=cid,endpoint_a=a_id,
                                      endpoint_b=b_id,accepted=bool(safe),
                                      max_surface_deviation_mm=deviation,
                                      max_radius_before_mm=before_max,max_radius_after_mm=after_max,
                                      exceeded_before=int(np.count_nonzero(bad)),
                                      exceeded_after=int(np.count_nonzero(after_bad))))
                if safe:
                    measured,complete,bad=after,after_complete,after_bad
                    self.used_pairs.add(tuple(sorted((a_id,b_id))))
                    accepted+=1
                    record_progress('accepted_repair',step+1)
                    improved=True
                    break
                self.builder.atoms,self.builder.edges,self.builder.edge_metrics=old_atoms,old_edges,old_metrics
            if not improved:
                paths=[(angle,path) for angle in (0.0,90.0,45.0,135.0)
                       for path in self._transverse_paths(measured,cid,angle_deg=angle)]
                untried=[]
                seen_this_round=set()
                for angle,path in paths:
                    coordinates=np.round(np.asarray(path,dtype=np.float64),6)
                    forward=coordinates.tobytes()
                    reverse=coordinates[::-1].copy().tobytes()
                    path_key=(state,'transverse',hashlib.blake2b(min(forward,reverse),digest_size=16).digest())
                    if path_key not in attempted_paths and path_key not in seen_this_round:
                        seen_this_round.add(path_key)
                        untried.append((angle,path,path_key))
                if not untried:
                    self.logs.append(dict(step=step+1,pore_id=cid,accepted=False,
                                          reason='no_untried_path'))
                for angle,path,path_key in untried:
                    attempted_paths.add(path_key)
                    old_atoms=self.builder.atoms.copy(deep=True)
                    old_edges=self.builder.edges.copy(deep=True)
                    old_metrics=self.builder.edge_metrics.copy(deep=True)
                    wall_ok,wall_length=self._append_transverse_wall(path,cid)
                    if not wall_ok:
                        self.logs.append(dict(step=step+1,pore_id=cid,
                                              method='surface_contour_transverse_path',
                                              direction_deg=angle,
                                              accepted=False,
                                              reason=self.last_rejection_reason or 'wall_attachment',
                                              max_surface_error_mm=self.last_surface_error_mm))
                        self.builder.atoms,self.builder.edges,self.builder.edge_metrics=old_atoms,old_edges,old_metrics
                        continue
                    after,after_complete,after_bad=self.measure_local(
                        (measured,complete,bad), len(old_edges))
                    before_max=float(np.max(measured['radius'][complete]))
                    after_max=float(np.max(after['radius'][after_complete]))
                    before_excess=float(np.maximum(measured['radius'][complete]-self.acceptance_mm,0).sum())
                    after_excess=float(np.maximum(after['radius'][after_complete]-self.acceptance_mm,0).sum())
                    safe=((int(np.count_nonzero(after_bad))<int(np.count_nonzero(bad))
                           or after_excess<before_excess-1e-5)
                          and after_max<=before_max+1e-6
                          and after['open_area_total']>=measured['open_area_total']-
                              self._allowed_open_area_loss(measured,wall_length))
                    rejection=[]
                    if not (int(np.count_nonzero(after_bad))<int(np.count_nonzero(bad))
                            or after_excess<before_excess-1e-5):
                        rejection.append('no_radius_improvement')
                    if after_max>before_max+1e-6:
                        rejection.append('maximum_radius_increased')
                    if (after['open_area_total']<measured['open_area_total']-
                            self._allowed_open_area_loss(measured,wall_length)):
                        rejection.append('open_area_loss')
                    self.logs.append(dict(step=step+1,pore_id=cid,
                                          method='surface_contour_transverse_path',
                                          direction_deg=angle,
                                          wall_length_mm=wall_length,accepted=bool(safe),
                                          reason=';'.join(rejection) if rejection else '',
                                          max_radius_before_mm=before_max,
                                          max_radius_after_mm=after_max,
                                          exceeded_before=int(np.count_nonzero(bad)),
                                          exceeded_after=int(np.count_nonzero(after_bad))))
                    if safe:
                        measured,complete,bad=after,after_complete,after_bad
                        accepted+=1
                        record_progress('accepted_repair',step+1)
                        improved=True
                        break
                    self.builder.atoms,self.builder.edges,self.builder.edge_metrics=old_atoms,old_edges,old_metrics
            if not improved: stalled.add(state)
            elif callable(getattr(self.builder, 'contour_repair_checkpoint', None)):
                self.builder.contour_repair_checkpoint(
                    self.builder, accepted, measured, bad)
        if accepted:
            # Local updates are an optimization. A complete independent
            # measurement remains the final authority for this repair pass.
            verified, verified_complete, verified_bad = self.measure()
            local_max = float(np.max(measured['radius'][complete])) if np.any(complete) else 0.0
            verified_max = (float(np.max(verified['radius'][verified_complete]))
                            if np.any(verified_complete) else 0.0)
            if (int(np.count_nonzero(bad)) != int(np.count_nonzero(verified_bad))
                    or abs(local_max - verified_max) > 1e-6
                    or abs(measured['open_area_total'] - verified['open_area_total']) > 1e-6):
                raise RuntimeError('Local contour update disagrees with independent global audit')
            measured, complete, bad = verified, verified_complete, verified_bad
        record_progress('final_audit',step+1 if initial_bad else 0)
        final_max=float(np.max(measured['radius'][complete])) if np.any(complete) else None
        summary=dict(method='complete_pore_contour_radius_repair',
                     intermediate_measurement='affected_components_only',
                     final_measurement='independent_global_contour_audit',
                     carrier_edge_mm=self.edge_mm,
                     local_contour_refinement_levels=self.local_refine,
                     radius_acceptance_mm=self.acceptance_mm,
                     initial_radius_exceeded_count=initial_bad,
                     final_radius_exceeded_count=int(np.count_nonzero(bad)),
                     initial_maximum_radius_mm=initial_max,final_maximum_radius_mm=final_max,
                     added_walls=accepted,completed=not np.any(bad),attempts=self.logs)
        self.builder.contour_radius_repair_summary=summary
        if diagnostic_dir:
            pd.DataFrame(self.logs).to_csv(directory/f'{phase}_repair_attempts.csv',index=False)
            remaining=[]
            for cid in np.flatnonzero(bad):
                xyz=np.asarray(measured['centroids'][cid],dtype=float)
                remaining.append(dict(pore_id=int(cid),radius_mm=float(measured['radius'][cid]),
                                      area_mm2=float(measured['area'][cid]),
                                      x_mm=float(xyz[0]),y_mm=float(xyz[1]),z_mm=float(xyz[2])))
            pd.DataFrame(remaining,columns=['pore_id','radius_mm','area_mm2','x_mm','y_mm','z_mm']).to_csv(
                directory/f'{phase}_remaining_pores.csv',index=False)
            termination=('passed' if not np.any(bad) else
                         'no_eligible_candidate' if all(pore_state(measured,int(cid)) in stalled
                                                        for cid in np.flatnonzero(bad)) else
                         'repair_iteration_limit')
            pd.Series(dict(initial_exceeded=initial_bad,remaining_exceeded=int(np.count_nonzero(bad)),
                           accepted_walls=accepted,repair_limit=repair_limit,
                           attempted_pores=len(stalled)+accepted,termination=termination,
                           acceptance_radius_mm=self.acceptance_mm)).to_json(
                directory/f'{phase}_repair_summary.json',indent=2)
        if np.any(bad):
            raise RuntimeError(f"Contour radius repair left {int(np.count_nonzero(bad))} complete pore(s) above {self.acceptance_mm:.6f} mm")
        self.builder.curr_vertices=self.builder.atoms[['x','y','z']].to_numpy(float).tolist()
        self.builder.curr_edges=list(self.builder.edges[['source','target']].itertuples(index=False,name=None))
        return summary


def apply_global_contour_radius_repair(builder):
    fine = ContourRadiusRepair(builder)
    initial = fine.measure()
    if np.any(initial[2]):
        # Search on a smaller carrier with a conservative target, then make
        # the final decision using the configured fine carrier and refinement.
        coarse = ContourRadiusRepair(
            builder,
            audit_edge_mm=2.0 * fine.edge_mm,
            repair_target_mm=0.96 * fine.acceptance_mm,
            local_refine=2,
        )
        try:
            coarse.run()
        except RuntimeError:
            # The fine pass below remains the only acceptance decision.
            pass
        fine = ContourRadiusRepair(builder)
        return fine.run()
    return fine.run(initial)
