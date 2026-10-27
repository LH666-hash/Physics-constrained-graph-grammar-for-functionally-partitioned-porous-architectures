from __future__ import annotations
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

class RodDistanceQuery:
    """Reuse the rod index across repeated contour-refinement queries."""
    def __init__(self, starts, ends, radii, edge_mm):
        self.starts = starts
        self.ab = ends - starts
        self.half = .5 * np.linalg.norm(self.ab, axis=1)
        self.den = np.maximum(np.einsum('ij,ij->i', self.ab, self.ab), 1e-24)
        self.radii = radii
        self.edge_mm = edge_mm
        self.tree = cKDTree(.5 * (starts + ends))
        self.reach = float(np.max(self.half + radii)) + edge_mm

    def __call__(self, points, chunk=5000):
        out = np.full(len(points), self.reach, dtype=np.float64)
        for start in range(0, len(points), chunk):
            pts = np.asarray(points[start:start + chunk], dtype=np.float64)
            pairs = cKDTree(pts).sparse_distance_matrix(self.tree, self.reach, output_type='coo_matrix')
            if len(pairs.data) == 0:
                continue
            row, col = pairs.row, pairs.col
            keep = pairs.data <= self.half[col] + self.radii[col] + self.edge_mm
            row, col = row[keep], col[keep]
            if not len(row):
                continue
            t = np.einsum('ij,ij->i', pts[row] - self.starts[col], self.ab[col]) / self.den[col]
            closest = self.starts[col] + np.clip(t, 0, 1)[:, None] * self.ab[col]
            d = np.linalg.norm(pts[row] - closest, axis=1) - self.radii[col]
            local = np.full(len(pts), self.reach, dtype=np.float64)
            np.minimum.at(local, row, d)
            out[start:start + len(pts)] = local
        return out


def signed_distance_to_rods(points, starts, ends, radii, edge_mm, chunk=5000):
    """Minimum (distance to rod centreline minus physical rod radius)."""
    return RodDistanceQuery(starts, ends, radii, edge_mm)(points, chunk)


def triangle_measure(tri, s):
    """Open area and within-triangle contour for a linear signed field."""
    pos = s > 0
    count = pos.sum(axis=1)
    full_area = .5 * np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    open_area = np.where(count == 3, full_area, 0.)
    contour = np.zeros(len(tri), dtype=np.float64)

    for npositive in (1, 2):
        ids = np.flatnonzero(count == npositive)
        if not len(ids):
            continue
        mask = pos[ids]
        pivot = np.argmax(mask, axis=1) if npositive == 1 else np.argmin(mask, axis=1)
        other1 = (pivot + 1) % 3
        other2 = (pivot + 2) % 3
        row = np.arange(len(ids))
        sp = s[ids, pivot]
        s1, s2 = s[ids, other1], s[ids, other2]
        t1 = sp / (sp - s1)
        t2 = sp / (sp - s2)
        triangle_fraction = t1 * t2
        open_area[ids] = full_area[ids] * (
            triangle_fraction if npositive == 1 else 1. - triangle_fraction
        )
        v0 = tri[ids, pivot]
        cross1 = v0 + t1[:, None] * (tri[ids, other1] - v0)
        cross2 = v0 + t2[:, None] * (tri[ids, other2] - v0)
        contour[ids] = np.linalg.norm(cross1 - cross2, axis=1)
    return open_area, contour, count, full_area


def refine_boundary_measure(tri, phi, count, signed_fn, levels, band_mm):
    """Subdivide only mixed triangles; retain their original component IDs."""
    total_area, total_wet, _, _ = triangle_measure(tri, phi)
    # The signed distance to a union of rods is 1-Lipschitz. A whole triangle
    # with same-sign vertices may still hide a curved contour, so also refine
    # every face whose nearest vertex lies within one face diameter of zero.
    candidate = np.flatnonzero(((count > 0) & (count < 3)) |
                           (np.min(np.abs(phi),axis=1) <= band_mm))
    total_area[candidate] = 0.
    total_wet[candidate] = 0.
    if not len(candidate):
        return total_area, total_wet
    # Process independent parent-face blocks so large carriers stay within RAM.
    for first in range(0, len(candidate), 20000):
        parent = candidate[first:first+20000]
        current_tri = tri[parent]
        current_phi = phi[parent]
        for level in range(levels):
            a,b,c = current_tri[:,0],current_tri[:,1],current_tri[:,2]
            pa,pb,pc = current_phi[:,0],current_phi[:,1],current_phi[:,2]
            ab,bc,ca = .5*(a+b),.5*(b+c),.5*(c+a)
            mid_phi=signed_fn(np.concatenate((ab,bc,ca)))+1e-8
            pab,pbc,pca=np.split(mid_phi,3)
            children=np.concatenate((
                np.stack((a,ab,ca),axis=1),
                np.stack((ab,b,bc),axis=1),
                np.stack((ca,bc,c),axis=1),
                np.stack((ab,bc,ca),axis=1),
            ))
            values=np.concatenate((
                np.stack((pa,pab,pca),axis=1),
                np.stack((pab,pb,pbc),axis=1),
                np.stack((pca,pbc,pc),axis=1),
                np.stack((pab,pbc,pca),axis=1),
            ))
            owners=np.tile(parent,4)
            a_child,p_child,c_child,_=triangle_measure(children,values)
            if level == levels-1:
                np.add.at(total_area,owners,a_child)
                np.add.at(total_wet,owners,p_child)
            else:
                child_band=band_mm/(2**(level+1))
                keep=((c_child>0)&(c_child<3)) | (np.min(np.abs(values),axis=1)<=child_band)
                done=(c_child==3)&~keep
                np.add.at(total_area,owners[done],a_child[done])
                current_tri=children[keep]
                current_phi=values[keep]
                parent=owners[keep]
    return total_area,total_wet


def contour_metrics(mesh: trimesh.Trimesh, phi: np.ndarray, center=None,
                    signed_fn=None, local_refine=0,
                    physical_boundary_edges=None):
    """Per connected open component, measure clipped area and interior contour."""
    # Shift the signed level by 10 nanometres to avoid a zero contour exactly
    # coinciding with carrier edges. This is far below any reported tolerance.
    phi = phi + 1e-8
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    s = phi[faces]
    tri = verts[faces]
    open_area, contour, count, full_area = triangle_measure(tri, s)
    if local_refine:
        assert signed_fn is not None
        edge_lengths=np.stack((np.linalg.norm(tri[:,1]-tri[:,0],axis=1),
                               np.linalg.norm(tri[:,2]-tri[:,1],axis=1),
                               np.linalg.norm(tri[:,0]-tri[:,2],axis=1)))
        open_area,contour=refine_boundary_measure(
            tri,s,count,signed_fn,local_refine,float(np.max(edge_lengths)))

    active = open_area > 1e-15
    active_ids = np.flatnonzero(active)
    idx = np.full(len(faces), -1, dtype=np.int64)
    idx[active_ids] = np.arange(len(active_ids), dtype=np.int64)
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    shared = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    both = active[adj[:,0]] & active[adj[:,1]]
    # A curved pore can cross a shared edge even when its two mesh vertices
    # are solid. Inspect its interior before declaring neighbouring faces
    # disconnected; use the same eight-part resolution as local_refine=3.
    edge_open = (phi[shared[:,0]] > 0) | (phi[shared[:,1]] > 0)
    if local_refine:
        ambiguous=np.flatnonzero(both & ~edge_open)
        if len(ambiguous):
            p0=verts[shared[ambiguous,0]]
            p1=verts[shared[ambiguous,1]]
            found=np.zeros(len(ambiguous),dtype=bool)
            for k in range(1,2**local_refine):
                t=k/(2**local_refine)
                found |= signed_fn((1-t)*p0+t*p1)>0
                if np.all(found): break
            edge_open[ambiguous]=found
    both &= edge_open
    rows = np.r_[idx[adj[both,0]], idx[adj[both,1]]]
    cols = np.r_[idx[adj[both,1]], idx[adj[both,0]]]
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                       shape=(len(active_ids), len(active_ids))).tocsr()
    ncomp, label = connected_components(graph, directed=False)
    area = np.bincount(label, weights=open_area[active_ids], minlength=ncomp)
    wet = np.bincount(label, weights=contour[active_ids], minlength=ncomp)
    face_center = tri[active_ids].mean(axis=1)
    centroids = np.column_stack([
        np.divide(np.bincount(label,weights=open_area[active_ids]*face_center[:,d],minlength=ncomp),
                  area,out=np.full(ncomp,np.nan),where=area>0) for d in range(3)])
    # Keep the archived 3-D PCA definition, with each carrier triangle weighted
    # by its newly measured open area rather than its old binary face area.
    means=np.zeros((ncomp,3))
    second=np.zeros((ncomp,3,3))
    aw=open_area[active_ids]/3.0
    for corner in range(3):
        xyz=tri[active_ids,corner]
        for d in range(3):
            means[:,d]+=np.bincount(label,weights=aw*xyz[:,d],minlength=ncomp)
            for e in range(d,3):
                second[:,d,e]+=np.bincount(label,weights=aw*xyz[:,d]*xyz[:,e],minlength=ncomp)
    means=np.divide(means,area[:,None],out=np.full_like(means,np.nan),where=area[:,None]>0)
    second=np.divide(second,area[:,None,None],out=np.full_like(second,np.nan),where=area[:,None,None]>0)
    for d in range(3):
        for e in range(d): second[:,d,e]=second[:,e,d]
    cov=second-means[:,:,None]*means[:,None,:]
    eig=np.linalg.eigvalsh(np.nan_to_num(cov,nan=0.0))
    aspect=np.sqrt(np.divide(np.maximum(eig[:,2],0),eig[:,1],
                             out=np.full(ncomp,np.inf),where=eig[:,1]>1e-18))

    # Mark open components that touch a true carrier boundary.
    inv = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    edge_count = np.bincount(inv, minlength=len(mesh.edges_unique))
    boundary_occ = np.flatnonzero(edge_count[inv] == 1)
    if physical_boundary_edges is not None and len(boundary_occ):
        boundary_pairs = np.asarray(mesh.edges_unique, dtype=np.int64)[inv[boundary_occ]]
        radix = len(mesh.vertices)
        boundary_keys = boundary_pairs[:, 0] * radix + boundary_pairs[:, 1]
        boundary_occ = boundary_occ[np.isin(boundary_keys, physical_boundary_edges)]
    touching = np.zeros(ncomp, dtype=bool)
    if len(boundary_occ):
        f = boundary_occ // 3
        edge = np.asarray(mesh.edges_unique, dtype=np.int64)[inv[boundary_occ]]
        edge_open = (phi[edge[:,0]] > 0) | (phi[edge[:,1]] > 0)
        if local_refine:
            ambiguous=np.flatnonzero(~edge_open)
            if len(ambiguous):
                p0,p1=verts[edge[ambiguous,0]],verts[edge[ambiguous,1]]
                found=np.zeros(len(ambiguous),dtype=bool)
                for k in range(1,2**local_refine):
                    t=k/(2**local_refine)
                    found |= signed_fn((1-t)*p0+t*p1)>0
                    if np.all(found): break
                edge_open[ambiguous]=found
        valid = active[f] & edge_open
        touching[label[idx[f[valid]]]] = True

    req = np.divide(2*area, wet, out=np.full(ncomp,np.inf), where=wet>1e-12)
    shape = np.divide(4*np.pi*area, wet**2, out=np.full(ncomp,np.nan), where=wet>1e-12)
    result = {'area':area,'perimeter':wet,'radius':req,'shape':shape,'aspect':aspect,
              'centroids':centroids,'touches_boundary':touching,'component_count':ncomp,
              'face_count':np.bincount(label,minlength=ncomp),
              'open_area_face':open_area,'carrier_area_face':full_area,
              'open_area_total':float(np.sum(open_area)),
              'carrier_area_total':float(np.sum(full_area))}
    face_component_id=np.full(len(faces),-1,dtype=np.int32)
    face_component_id[active_ids]=label
    result['face_component_id']=face_component_id
    if center is not None:
        face_center = tri.mean(axis=1)
        f = int(np.argmin(np.linalg.norm(face_center - center,axis=1)))
        assert active[f], 'Requested centre is not open'
        result['center_component'] = int(label[idx[f]])
    return result


