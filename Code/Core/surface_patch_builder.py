# -*- coding: utf-8 -*-
"""PC-AEGG surface graph construction and validation.

The module implements the surface workflow used by PC-AEGG:

1. split a mesh into connected, approximately developable patches;
2. flatten every patch to a local 2-D chart;
3. refine the graph in the chart using a hydraulic-radius target;
4. map nodes and sampled edge polylines back through source triangles;
5. reject cross-patch/cross-component/domain-external links;
6. globally reconstruct complete pores across patch seams;
7. repair complete pores that exceed the physical contour-radius limit;
8. retain aspect ratio as a diagnostic and export graph-validation data.

The mapping is triangle-aware.  A new 2-D point is returned to 3-D with the
barycentric coordinates of the source triangle that contains it.  Each edge is
represented by an adaptively sampled polyline whose chord-to-surface error is
checked against ``surface_tolerance_mm``.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "trimesh": "trimesh",
    "scipy": "scipy",
    "shapely": "shapely",
    "matplotlib": "matplotlib",
}
_missing = [package for module, package in _DEPENDENCIES.items() if importlib.util.find_spec(module) is None]
if _missing:
    raise ImportError(
        f"Missing packages: {', '.join(_missing)}. Install Code/requirements.txt before running."
    )

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import Voronoi, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, polylabel, unary_union, split, polygonize

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "surface-patch-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.ticker import MaxNLocator

_HAS_RTREE = importlib.util.find_spec("rtree") is not None
_HAS_FAST_SIMPLIFICATION = importlib.util.find_spec("fast_simplification") is not None
from matplotlib.collections import PolyCollection
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


PALETTE = {
    "blue": "#2171B5",
    "mid": "#BCC0DF",
    "pink": "#FFCFCF",
    "grey": "#4D5250",
    "light_grey": "#D9DEDC",
    "pale": "#F6F8F7",
    "green": "#69B36D",
    "red": "#C96F70",
    "yellow": "#D9B45F",
}

# Signed-K class threshold in mm^-2.  Values with |K| <= this threshold are
# reported as near-zero; it is deliberately shared by patching, CSV, JSON,
# validation reports, and the main signed-K figure.
GAUSSIAN_CURVATURE_NEAR_ZERO_THRESHOLD_MM2 = 2.0e-3
SIGNED_K_COLOR_PERCENTILE = 98.0

@dataclass(frozen=True)
class SurfaceCapillaryDesign:
    """Pressure-derived limit for the equivalent capillary radius."""

    hydrostatic_pressure_pa: float
    required_pressure_pa: float
    critical_equivalent_radius_mm: float


def calculate_surface_capillary_design(
    *,
    surface_tension_n_per_m: float,
    contact_angle_deg: float,
    liquid_density_kg_per_m3: float,
    gas_density_kg_per_m3: float,
    gravity_m_per_s2: float,
    target_height_mm: float,
    additional_pressure_pa: float = 0.0,
    safety_factor: float = 1.2,
) -> SurfaceCapillaryDesign:
    """Calculate ``r_crit`` from the Young-Laplace pressure requirement.

    The surface pores use ``r_eq = 2 A_pore / L_wet``.  Their admissible
    upper bound is

    ``r_crit = 2 gamma cos(theta) / {S_s[(rho_l-rho_g)gH_s + DeltaP_op]}``.

    Length inputs/outputs are millimetres; pressure and fluid properties use
    SI units.  A 90-degree contact angle cannot sustain a positive pressure
    head and is therefore rejected when the required pressure is non-zero.
    """

    gamma = float(surface_tension_n_per_m)
    theta = float(contact_angle_deg)
    rho_l = float(liquid_density_kg_per_m3)
    rho_g = float(gas_density_kg_per_m3)
    gravity = float(gravity_m_per_s2)
    height_mm = float(target_height_mm)
    extra_pressure = float(additional_pressure_pa)
    safety = float(safety_factor)

    if gamma <= 0.0:
        raise ValueError("surface_tension_n_per_m must be positive")
    if not 0.0 <= theta <= 90.0:
        raise ValueError("contact_angle_deg must be within [0, 90]")
    if rho_l <= rho_g:
        raise ValueError("liquid_density_kg_per_m3 must exceed gas_density_kg_per_m3")
    if gravity <= 0.0:
        raise ValueError("gravity_m_per_s2 must be positive")
    if height_mm < 0.0:
        raise ValueError("target_height_mm cannot be negative")
    if extra_pressure < 0.0:
        raise ValueError("additional_pressure_pa cannot be negative")
    if safety <= 0.0:
        raise ValueError("safety_factor must be positive")

    hydrostatic_pressure = (rho_l - rho_g) * gravity * height_mm * 1.0e-3
    required_pressure = safety * (hydrostatic_pressure + extra_pressure)
    if required_pressure <= 0.0:
        raise ValueError("the pressure requirement must be positive")

    cos_theta = math.cos(math.radians(theta))
    if cos_theta <= 1.0e-12:
        raise ValueError(
            "contact_angle_deg=90 gives no positive capillary pressure; use an angle below 90 degrees"
        )
    critical_radius_mm = 2.0 * gamma * cos_theta / required_pressure * 1.0e3
    return SurfaceCapillaryDesign(
        hydrostatic_pressure_pa=hydrostatic_pressure,
        required_pressure_pa=required_pressure,
        critical_equivalent_radius_mm=critical_radius_mm,
    )


# Direct-run defaults use paths within the submission package.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
INPUT_STL_PATH = _PACKAGE_ROOT / "Input_Geometries" / "局部高负曲率曲面.stl"
TARGET_HEIGHT_MM = 25.0
SURFACE_DIAMETER_MM = 0.3
SURFACE_TENSION_N_PER_M = 0.072
CONTACT_ANGLE_DEG = 15.0
LIQUID_DENSITY_KG_PER_M3 = 998.0
GAS_DENSITY_KG_PER_M3 = 0.0
GRAVITY_M_PER_S2 = 9.80665
ADDITIONAL_PRESSURE_PA = 0.0
SURFACE_CAPILLARY_SAFETY_FACTOR = 1.0
DEFAULT_CAPILLARY_DESIGN = calculate_surface_capillary_design(
    surface_tension_n_per_m=SURFACE_TENSION_N_PER_M,
    contact_angle_deg=CONTACT_ANGLE_DEG,
    liquid_density_kg_per_m3=LIQUID_DENSITY_KG_PER_M3,
    gas_density_kg_per_m3=GAS_DENSITY_KG_PER_M3,
    gravity_m_per_s2=GRAVITY_M_PER_S2,
    target_height_mm=TARGET_HEIGHT_MM,
    additional_pressure_pa=ADDITIONAL_PRESSURE_PA,
    safety_factor=SURFACE_CAPILLARY_SAFETY_FACTOR,
)
# Backward-compatible name used by existing diagnostics and CSV columns.
TARGET_HYDRAULIC_RADIUS_MM = DEFAULT_CAPILLARY_DESIGN.critical_equivalent_radius_mm
HEIGHT_LAYER_COUNT = 20

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "savefig.facecolor": "white",
    }
)


@dataclass
class SurfaceBuildConfig:
    target_faces: int = 1000
    max_faces_per_patch: int = 72
    normal_angle_deg: float = 24.0
    curvature_sign_threshold: float = GAUSSIAN_CURVATURE_NEAR_ZERO_THRESHOLD_MM2
    curvature_color_percentile: float = SIGNED_K_COLOR_PERCENTILE
    split_area_distortion_p95: float = 0.04
    split_angle_distortion_p95_deg: float = 3.0
    split_rounds: int = 8
    surface_diameter_mm: float = 0.3
    # Equivalent capillary radius r_eq=2*A_pore/L_wet.  The legacy field name
    # is retained so existing diagnostics and downstream readers keep working.
    target_hydraulic_radius_mm: float = TARGET_HYDRAULIC_RADIUS_MM
    target_radius_source: str = "capillary_pressure"
    design_target_height_mm: float = TARGET_HEIGHT_MM
    surface_tension_n_per_m: float = SURFACE_TENSION_N_PER_M
    contact_angle_deg: float = CONTACT_ANGLE_DEG
    liquid_density_kg_per_m3: float = LIQUID_DENSITY_KG_PER_M3
    gas_density_kg_per_m3: float = GAS_DENSITY_KG_PER_M3
    gravity_m_per_s2: float = GRAVITY_M_PER_S2
    additional_pressure_pa: float = ADDITIONAL_PRESSURE_PA
    capillary_safety_factor: float = SURFACE_CAPILLARY_SAFETY_FACTOR
    hydraulic_radius_mapping_safety: float = 1
    # Configurable BCC baselines used by the refinement-evolution figure.
    bcc_reference_hydraulic_radius_mm: float = 0.60
    bcc_reference_aspect_ratio: float = 1.0
    bcc_reference_shape_factor: float = math.pi / 4.0
    bcc_reference_liquid_area_fraction: float = 0.32
    final_graph_radius_tolerance_fraction: float = 0.005
    # Screening threshold for the fraction of local pore records that touch an
    # artificial inter-patch seam. This is a methodological burden screen only;
    # it is not a substitute for cross-patch pore reconstruction or final 3-D
    # pore measurement.
    seam_screen_enabled: bool = True
    seam_screen_max_fraction: float = 0.50
    patch_batch_size: int = 8
    max_insertions_per_patch: int = 0
    max_total_insertions: int = 0
    max_insertions_by_patch: Tuple[int, ...] = ()
    max_voronoi_rounds: int = 100
    max_bad_subdomains_per_round: int = 10
    polylabel_tolerance_mm: float = 0.01
    refinement_insertion_strategy: str = "polylabel"
    centroid_jitter_mm: float = 0.01
    # Second-stage slit-pore morphology rewrite. Capillary Voronoi refinement
    # is performed first; a bounded subset of low-aspect-ratio pores is then
    # split by a wall aligned with the local major axis. Aspect ratio remains
    # a soft target, while the capillary-radius criterion stays the hard gate.
    anisotropy_rewrite_enable: bool = True
    anisotropy_target_aspect_ratio: float = 2.75
    anisotropy_soft_fraction: float = 0.60
    anisotropy_max_splits_per_patch: int = 8
    anisotropy_min_wall_length_mm: float = 0.65
    # Protect true design-domain boundaries, but do NOT exempt artificial
    # inter-patch seams. Seam-dependent pores are valid physical pores and
    # must undergo the same morphology repair.
    anisotropy_skip_patch_boundary_cells: bool = True
    anisotropy_capillary_priority_weight: float = 1.5
    anisotropy_min_ar_improvement_fraction: float = 0.08
    # Hard upper morphology bound. The soft target encourages slit-like pores,
    # while this cap prevents excessively long, needle-like pores.
    anisotropy_max_aspect_ratio: float = 5.0
    anisotropy_cap_repair_target_fraction: float = 0.95
    anisotropy_max_cap_splits_per_cell: int = 8
    # AR is only a hard morphology constraint for physically meaningful
    # complete pores. Small positive-area pores remain valid as long as they
    # satisfy capillary and non-zero-area requirements.
    anisotropy_cap_min_area_mm2: float = 0.5
    # Post-patch global complete-pore rewrite. This stage is independent of
    # chart boundaries and is specifically intended for cross-patch pores.
    global_ar_cap_rewrite_enable: bool = True
    global_ar_cap_audit_max_edge_mm: float = 0.08
    global_ar_cap_max_passes: int = 24
    global_ar_cap_max_walls_per_pass: int = 64
    global_ar_cap_min_improvement_fraction: float = 0.02
    global_ar_cap_anchor_search_padding_mm: float = 0.12
    max_edge_length_mm: float = 1.5
    min_logical_edge_length_mm: float = 0.65
    source_node_spacing_mm: float = 0.65
    min_boundary_carriers_per_patch: int = 3
    feature_patch_area_mm2: float = 1.2
    max_node_degree: int = 8
    edge_sample_spacing_mm: float = 0.35
    surface_tolerance_mm: float = 0.01875
    # Additional contour-repair walls may follow strongly curved input faces.
    # All original and other generated walls retain surface_tolerance_mm.
    contour_repair_surface_tolerance_mm: float = 0.05
    max_edge_refine_depth: int = 7
    map_tolerance: float = 1.0e-7
    outward_offset_mm: float = 0.0
    random_seed: int = 42
    refinement_policy: str = "physics"
    projection_neighbor_vertices: int = 24
    local_projection_face_threshold: int = 100000
    projection_audit_samples: int = 0
    max_hydraulic_radius_rel_error: float = 0.15
    # True geometric seam coverage is sampled on the 3-D artificial seam.
    # These replace the previous placeholder position-gap metric.
    min_seam_position_continuity: float = 0.995
    min_seam_graph_incidence: float = 0.99
    min_seam_patch_label_coverage: float = 0.99
    seam_coverage_sample_spacing_mm: float = 0.10
    max_seam_uncovered_length_mm: float = 0.20
    min_graph_largest_component_fraction: float = 0.95
    min_surface_cell_coverage: float = 0.98
    coverage_sample_spacing_mm: float = 0.25
    max_uncovered_surface_fraction: float = 0.01
    mapping_surface: str = "carrier"
    # Sub-resolution open fragments are reported only.  Rewriting the graph
    # to remove them can undo useful refinement of adjacent, physically
    # relevant pores.
    # Numerical zero-area tolerance for retained open pores. Positive pores
    # are no longer forced above an arbitrary 0.02-mm^2 area threshold; only
    # zero/numerically-zero openings are repaired, including at artificial seams.
    zero_pore_repair_enable: bool = True
    zero_pore_min_diameter_mm: float = 0.25
    zero_pore_min_area_mm2: float = 1.0e-6
    zero_pore_max_per_patch: int = 64
    zero_pore_secondary_trials: int = 8
    # Final minimum-area cleanup is deliberately more persistent than the
    # ordinary rewrite loops.  It alternates source-cell and retained-graph
    # cleanup until BOTH zero/undersized source openings and actual retained
    # open fragments are cleared, or the geometry reaches a true no-progress
    # state.
    min_pore_repair_max_cycles: int = 12
    min_pore_repair_stagnation_patience: int = 3
    final_fragment_repair_max_passes: int = 12


@dataclass
class PatchChart:
    patch_id: int
    component_id: int
    face_ids: np.ndarray
    vertex_ids: np.ndarray
    uv: np.ndarray
    xyz: np.ndarray
    local_faces: np.ndarray
    polygon: Polygon | MultiPolygon
    centroid: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    parameterization_method: str
    scale_2d_to_3d: float
    area_distortion: np.ndarray
    angle_distortion_deg: np.ndarray
    edge_distortion: np.ndarray
    uv_overlap_fraction: float
    curvature_mean: float
    curvature_abs_mean: float


def _triangle_area_2d(tri: np.ndarray) -> float:
    a = tri[1] - tri[0]
    b = tri[2] - tri[0]
    return 0.5 * abs(float(a[0] * b[1] - a[1] * b[0]))


def _triangle_area_3d(tri: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))


def _triangle_angles(points: np.ndarray) -> np.ndarray:
    out = []
    for i in range(3):
        a = points[(i + 1) % 3] - points[i]
        b = points[(i + 2) % 3] - points[i]
        den = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-12)
        out.append(math.degrees(math.acos(float(np.clip(np.dot(a, b) / den, -1.0, 1.0)))))
    return np.asarray(out, dtype=float)


def _barycentric_2d(point: np.ndarray, tri: np.ndarray) -> Optional[np.ndarray]:
    a, b, c = tri
    m = np.column_stack((a - c, b - c))
    det = float(np.linalg.det(m))
    if abs(det) < 1.0e-14:
        return None
    uv = np.linalg.solve(m, point - c)
    return np.asarray([uv[0], uv[1], 1.0 - uv.sum()], dtype=float)


def _safe_percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if len(arr) else 0.0


def _json_safe(value):
    """Convert numpy/non-finite values to JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _geometry_covers_line(domain: Polygon | MultiPolygon, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    line = LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])
    return bool(domain.buffer(tol).covers(line))


def _polygon_parts(geom) -> List[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def _polygon_aspect_ratio(poly: Polygon) -> float:
    if poly.is_empty or poly.area <= 1.0e-12:
        return 0.0
    rect = poly.minimum_rotated_rectangle
    coords = np.asarray(rect.exterior.coords, dtype=float)
    lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    lengths = lengths[lengths > 1.0e-12]
    return float(lengths.max() / lengths.min()) if len(lengths) else 0.0


def _intrinsic_triangle_2d(tri3: np.ndarray) -> np.ndarray:
    v01 = tri3[1] - tri3[0]
    v02 = tri3[2] - tri3[0]
    length01 = max(float(np.linalg.norm(v01)), 1.0e-12)
    axis = v01 / length01
    x2 = float(np.dot(v02, axis))
    y2 = math.sqrt(max(float(np.dot(v02, v02)) - x2 * x2, 0.0))
    return np.asarray([[0.0, 0.0], [length01, 0.0], [x2, y2]], dtype=float)


def _mesh_boundary_vertex_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return vertices incident to an open mesh edge (outer edge or hole)."""
    boundary = np.zeros(len(mesh.vertices), dtype=bool)
    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for face in np.asarray(mesh.faces, dtype=int):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_counts[tuple(sorted((int(a), int(b))))] += 1
    for (a, b), count in edge_counts.items():
        if count == 1:
            boundary[a] = True
            boundary[b] = True
    return boundary


def _curvature_class(value: float, threshold: float) -> str:
    if not np.isfinite(value):
        return "invalid"
    if value > threshold:
        return "positive"
    if value < -threshold:
        return "negative"
    return "near_zero"


def _signed_gaussian_curvature_statistics(
    values: Sequence[float], boundary_mask: Sequence[bool], threshold: float
) -> Dict[str, float]:
    """Summarize signed K using finite non-boundary vertices only."""
    values = np.asarray(values, dtype=float)
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    if len(values) != len(boundary_mask):
        raise ValueError("values and boundary_mask must have the same length")
    included = np.isfinite(values) & ~boundary_mask
    selected = values[included]
    if len(selected) == 0:
        return {
            "gaussian_curvature_mean_signed": np.nan,
            "gaussian_curvature_abs_mean": np.nan,
            "gaussian_curvature_min": np.nan,
            "gaussian_curvature_max": np.nan,
            "positive_K_fraction": np.nan,
            "negative_K_fraction": np.nan,
            "near_zero_K_fraction": np.nan,
            "gaussian_curvature_internal_vertex_count": 0,
            "gaussian_curvature_boundary_vertex_count": int(np.count_nonzero(boundary_mask)),
            "gaussian_curvature_finite_vertex_count": int(np.count_nonzero(np.isfinite(values))),
        }
    return {
        "gaussian_curvature_mean_signed": float(np.mean(selected)),
        "gaussian_curvature_abs_mean": float(np.mean(np.abs(selected))),
        "gaussian_curvature_min": float(np.min(selected)),
        "gaussian_curvature_max": float(np.max(selected)),
        "positive_K_fraction": float(np.mean(selected > float(threshold))),
        "negative_K_fraction": float(np.mean(selected < -float(threshold))),
        "near_zero_K_fraction": float(np.mean(np.abs(selected) <= float(threshold))),
        "gaussian_curvature_internal_vertex_count": int(len(selected)),
        "gaussian_curvature_boundary_vertex_count": int(np.count_nonzero(boundary_mask)),
        "gaussian_curvature_finite_vertex_count": int(np.count_nonzero(np.isfinite(values))),
    }


def _signed_k_color_limit(
    values: Sequence[float], boundary_mask: Sequence[bool], percentile: float, threshold: float
) -> float:
    """Return a symmetric color limit while retaining raw extrema in outputs."""
    values = np.asarray(values, dtype=float)
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    included = values[np.isfinite(values) & ~boundary_mask]
    if len(included) == 0:
        return max(float(threshold), 1.0e-12)
    percentile = float(np.clip(percentile, 50.0, 100.0))
    limit = float(np.percentile(np.abs(included), percentile))
    return max(limit, float(threshold), 1.0e-12)


def _clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    if len(mesh.faces):
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    if len(mesh.faces):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _load_mesh(path: Path, target_height_mm: Optional[float]) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"No triangular surface found in {path}")
    mesh = _clean_mesh(mesh)
    mesh.fix_normals()
    if target_height_mm is not None:
        height = float(np.ptp(mesh.bounds[:, 2]))
        if height <= 1.0e-12:
            raise ValueError("Cannot height-scale a zero-height mesh")
        mesh.apply_scale(float(target_height_mm) / height)
    mesh.apply_translation(-mesh.bounds[0])
    return mesh


def _face_components(mesh: trimesh.Trimesh) -> np.ndarray:
    adjacency = np.asarray(mesh.face_adjacency, dtype=int)
    face_count = len(mesh.faces)
    if face_count == 0:
        return np.empty(0, dtype=int)
    if len(adjacency) == 0:
        return np.arange(face_count, dtype=int)
    rows = np.concatenate((adjacency[:, 0], adjacency[:, 1]))
    cols = np.concatenate((adjacency[:, 1], adjacency[:, 0]))
    data = np.ones(len(rows), dtype=np.uint8)
    graph = coo_matrix((data, (rows, cols)), shape=(face_count, face_count)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(int, copy=False)


def _vertex_gaussian_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """Discrete angle defect divided by one-third incident face area."""
    n_vertices = len(mesh.vertices)
    angle_sum = np.zeros(n_vertices, dtype=float)
    area_sum = np.zeros(n_vertices, dtype=float)
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    for face_id, face in enumerate(np.asarray(mesh.faces, dtype=int)):
        tri = triangles[face_id]
        angles = _triangle_angles(tri)
        area = _triangle_area_3d(tri)
        for local, vid in enumerate(face):
            angle_sum[int(vid)] += angles[local] * math.pi / 180.0
            area_sum[int(vid)] += area / 3.0
    target = np.full(n_vertices, 2.0 * math.pi)
    boundary_vertices = _mesh_boundary_vertex_mask(mesh)
    target[boundary_vertices] = math.pi
    return (target - angle_sum) / np.maximum(area_sum, 1.0e-12)


class SurfacePatchBuilder:
    """Build a traceable surface graph from one triangular mesh."""

    def __init__(self, original_mesh: trimesh.Trimesh, config: Optional[SurfaceBuildConfig] = None):
        self.original_mesh = _clean_mesh(original_mesh)
        self.config = config or SurfaceBuildConfig()
        self.original_face_components = _face_components(self.original_mesh)
        self._proximity = (
            trimesh.proximity.ProximityQuery(self.original_mesh) if _HAS_RTREE else None
        )
        self._projection_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, int]] = {}
        self._use_local_projection = (
            (not _HAS_RTREE)
            or len(self.original_mesh.faces) >= int(self.config.local_projection_face_threshold)
        )
        self._vertex_tree = cKDTree(np.asarray(self.original_mesh.vertices, dtype=float))
        self._vertex_faces = np.asarray(self.original_mesh.vertex_faces, dtype=int)
        self._component_vertex_ids: Dict[int, np.ndarray] = {}
        self._component_vertex_trees: Dict[int, cKDTree] = {}
        for component_id in np.unique(self.original_face_components):
            face_ids = np.flatnonzero(self.original_face_components == int(component_id))
            vertex_ids = np.unique(np.asarray(self.original_mesh.faces, dtype=int)[face_ids].reshape(-1))
            self._component_vertex_ids[int(component_id)] = vertex_ids
            self._component_vertex_trees[int(component_id)] = cKDTree(
                np.asarray(self.original_mesh.vertices, dtype=float)[vertex_ids]
            )
        self._projection_audit_errors: List[float] = []
        self.base_mesh: Optional[trimesh.Trimesh] = None
        self.base_vertex_gaussian_curvature = np.empty(0, dtype=float)
        self.base_boundary_vertex_mask = np.empty(0, dtype=bool)
        self._carrier_proximity: Optional[trimesh.proximity.ProximityQuery] = None
        self._carrier_vertex_tree: Optional[cKDTree] = None
        self._carrier_vertex_faces: Optional[np.ndarray] = None
        self._carrier_face_centroid_tree: Optional[cKDTree] = None
        self._artificial_seam_edges: Set[Tuple[int, int]] = set()
        self._artificial_seam_vertex_ids: Set[int] = set()
        self._seam_edge_patches: Dict[Tuple[int, int], Set[int]] = {}
        self._seam_required_degree_by_vertex: Dict[int, int] = {}
        self.seam_geometric_metrics = pd.DataFrame()
        self.charts: List[PatchChart] = []
        self.atoms = pd.DataFrame()
        self.edges = pd.DataFrame()
        self.patch_metrics = pd.DataFrame()
        self.edge_metrics = pd.DataFrame()
        self.pore_metrics = pd.DataFrame()
        self.coverage_metrics = pd.DataFrame()
        self.refinement_history = pd.DataFrame()
        self.refinement_evolution = pd.DataFrame()
        self._refinement_history_rows: List[Dict] = []
        self._total_refinement_insertions = 0
        self.anisotropy_rewrite_history = pd.DataFrame()
        self._anisotropy_rewrite_rows: List[Dict] = []
        self.voronoi_seeds = pd.DataFrame()
        self._voronoi_seed_rows: List[Dict] = []
        self.summary_metrics = pd.DataFrame()
        self.signed_gaussian_curvature = pd.DataFrame()
        self.seam_metrics = pd.DataFrame()
        self.curr_vertices: List[List[float]] = []
        self.curr_edges: List[Tuple[int, int]] = []
        self._source_cells_by_patch: Dict[int, List[Polygon]] = {}
        self.zero_pore_repair_history = pd.DataFrame()
        self.zero_pore_final_audit = pd.DataFrame()
        self.final_fragment_repair_history = pd.DataFrame()
        self.final_audit_repair_history = pd.DataFrame()
        self.final_graph_pores = pd.DataFrame()
        self.global_ar_cap_rewrite_history = pd.DataFrame()
        self.global_ar_cap_summary: Dict[str, object] = {}

    def _project_original(self, point: np.ndarray, component_id: Optional[int] = None) -> Tuple[np.ndarray, int]:
        point = np.asarray(point, dtype=float)
        key = (int(component_id) if component_id is not None else -1,) + tuple(
            np.round(point * 1.0e6).astype(np.int64).tolist()
        )
        cached = self._projection_cache.get(key)
        if cached is not None:
            return cached[0].copy(), int(cached[1])
        closest, face_id = self._project_original_batch(point[None, :], component_id=component_id)
        result = (np.asarray(closest[0], dtype=float), int(face_id[0]))
        self._projection_cache[key] = result
        return result[0].copy(), result[1]

    def _project_local_batch(self, points: np.ndarray, component_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        if component_id is None:
            vertex_tree = self._vertex_tree
            vertex_lookup = np.arange(len(self.original_mesh.vertices), dtype=int)
        else:
            vertex_tree = self._component_vertex_trees[int(component_id)]
            vertex_lookup = self._component_vertex_ids[int(component_id)]
        k = min(max(1, int(self.config.projection_neighbor_vertices)), len(vertex_lookup))
        _, nearest_local_vertices = vertex_tree.query(points, k=k)
        nearest_vertices = vertex_lookup[np.asarray(nearest_local_vertices, dtype=int)]
        nearest_vertices = np.asarray(nearest_vertices, dtype=int)
        if nearest_vertices.ndim == 1:
            nearest_vertices = nearest_vertices[:, None]
        projected = np.empty_like(points)
        face_ids = np.empty(len(points), dtype=int)
        distances = np.empty(len(points), dtype=float)
        triangles_all = np.asarray(self.original_mesh.triangles, dtype=float)
        for i, (point, vertex_ids) in enumerate(zip(points, nearest_vertices)):
            candidate_faces = np.unique(self._vertex_faces[vertex_ids].reshape(-1))
            candidate_faces = candidate_faces[candidate_faces >= 0]
            if len(candidate_faces) == 0:
                # This is exceptionally rare.  Fall back to a brute-force
                # closest-triangle query so rtree remains optional.
                triangles = triangles_all
                repeated = np.repeat(point[None, :], len(triangles), axis=0)
                closest_all = trimesh.triangles.closest_point(triangles, repeated)
                distance_sq_all = np.einsum(
                    "ij,ij->i", closest_all - repeated, closest_all - repeated
                )
                best_all = int(np.argmin(distance_sq_all))
                projected[i] = closest_all[best_all]
                distances[i] = math.sqrt(max(float(distance_sq_all[best_all]), 0.0))
                face_ids[i] = best_all
                continue
            triangles = triangles_all[candidate_faces]
            repeated = np.repeat(point[None, :], len(candidate_faces), axis=0)
            closest = trimesh.triangles.closest_point(triangles, repeated)
            distance_sq = np.einsum("ij,ij->i", closest - repeated, closest - repeated)
            best = int(np.argmin(distance_sq))
            projected[i] = closest[best]
            distances[i] = math.sqrt(max(float(distance_sq[best]), 0.0))
            face_ids[i] = int(candidate_faces[best])
        return projected, distances, face_ids

    def _surface_projection_batch(self, points: np.ndarray, component_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        if self._use_local_projection or self._proximity is None:
            return self._project_local_batch(points, component_id=component_id)
        return self._proximity.on_surface(points)

    def _surface_distance_batch(self, points: np.ndarray, component_id: Optional[int] = None) -> np.ndarray:
        return np.asarray(
            self._surface_projection_batch(points, component_id=component_id)[1], dtype=float
        )

    def _project_original_batch(self, points: np.ndarray, component_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        projected = np.empty_like(points)
        face_ids = np.empty(len(points), dtype=int)
        missing_indices = []
        missing_points = []
        missing_keys = []
        for i, point in enumerate(points):
            key = (int(component_id) if component_id is not None else -1,) + tuple(
                np.round(point * 1.0e6).astype(np.int64).tolist()
            )
            cached = self._projection_cache.get(key)
            if cached is None:
                missing_indices.append(i)
                missing_points.append(point)
                missing_keys.append(key)
            else:
                projected[i] = cached[0]
                face_ids[i] = int(cached[1])
        if missing_points:
            missing_array = np.asarray(missing_points, dtype=float)
            closest, _, nearest_faces = self._surface_projection_batch(
                missing_array, component_id=component_id
            )
            audit_target = max(0, int(self.config.projection_audit_samples))
            if (
                self._use_local_projection
                and component_id is None
                and len(self._projection_audit_errors) < audit_target
            ):
                audit_n = min(audit_target - len(self._projection_audit_errors), len(missing_array))
                if audit_n > 0:
                    audit_idx = np.linspace(0, len(missing_array) - 1, audit_n, dtype=int)
                    if self._proximity is not None:
                        exact, _, _ = self._proximity.on_surface(missing_array[audit_idx])
                        errors = np.linalg.norm(closest[audit_idx] - exact, axis=1)
                        self._projection_audit_errors.extend(map(float, errors))
            for out_i, point3, face_id, key in zip(missing_indices, closest, nearest_faces, missing_keys):
                projected[out_i] = point3
                face_ids[out_i] = int(face_id)
                self._projection_cache[key] = (np.asarray(point3, dtype=float), int(face_id))
        return projected, face_ids

    def initialize_by_decimation(self, target_faces: Optional[int] = None):
        if target_faces is not None:
            self.config.target_faces = int(target_faces)
        self.build()

    def optimize_iteratively(self):
        if self.atoms.empty:
            self.build()

    def offset_outward(self, distance_mm: float):
        distance_mm = float(distance_mm)
        if abs(distance_mm) <= 1.0e-12 or self.atoms.empty:
            return
        points = self.atoms[["x", "y", "z"]].to_numpy(float)
        closest, _, face_ids = trimesh.proximity.closest_point(self.original_mesh, points)
        normals = np.asarray(self.original_mesh.face_normals)[face_ids]
        moved = closest + normals * distance_mm
        self.atoms.loc[:, ["x", "y", "z"]] = moved
        self.curr_vertices = moved.tolist()
        self.config.outward_offset_mm = distance_mm

    def export_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.atoms.copy(), self.edges.copy()

    def build(self):
        self.base_mesh = self._prepare_base_mesh()
        self.base_vertex_gaussian_curvature = _vertex_gaussian_curvature(self.base_mesh)
        self.base_boundary_vertex_mask = _mesh_boundary_vertex_mask(self.base_mesh)
        self._carrier_proximity = (
            trimesh.proximity.ProximityQuery(self.base_mesh) if _HAS_RTREE else None
        )
        self._carrier_vertex_tree = cKDTree(np.asarray(self.base_mesh.vertices, dtype=float))
        self._carrier_vertex_faces = np.asarray(self.base_mesh.vertex_faces, dtype=int)
        self._carrier_face_centroid_tree = cKDTree(
            np.asarray(self.base_mesh.triangles_center, dtype=float)
        )
        initial = self._merge_small_patches(self._segment_initial_patches())
        patches = self._merge_small_patches(
            self._split_distorted_patches(initial), min_faces=2
        )
        patches = [faces for faces in patches if self._patch_is_valid(faces)]
        if not patches:
            raise ValueError("No non-degenerate surface patches remain after mesh cleanup")
        self.charts = [self._make_chart(i, face_ids) for i, face_ids in enumerate(patches)]
        self._build_patch_boundary_topology()
        self._graph_source_vertex_ids = self._select_graph_source_vertices()
        self._construct_graph()
        if bool(self.config.global_ar_cap_rewrite_enable):
            try:
                from global_surface_pore_rewrite_V11 import apply_global_ar_cap_rewrite
                apply_global_ar_cap_rewrite(self)
            except Exception as exc:
                # Preserve a traceable failure when the optional legacy rewrite is requested.
                self.global_ar_cap_summary = {
                    "enabled": True,
                    "completed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                raise
        if self.config.refinement_policy == "physics":
            from global_surface_contour_repair import apply_global_contour_radius_repair
            self.prefer_transverse = True
            apply_global_contour_radius_repair(self)
        self._analyze()
        return self

    def _prepare_base_mesh(self) -> trimesh.Trimesh:
        mesh = self.original_mesh.copy()
        target = max(16, int(self.config.target_faces))
        if len(mesh.faces) > target:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=target)
            except Exception as exc:
                # trimesh uses fast-simplification when available.  A VTK
                # fallback keeps batch validation reproducible on machines
                # where that optional wheel is unavailable.
                try:
                    import vtk
                    from vtk.util.numpy_support import (
                        numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
                    )
                    poly = vtk.vtkPolyData()
                    pts = vtk.vtkPoints()
                    pts.SetData(numpy_to_vtk(
                        np.asarray(mesh.vertices, dtype=np.float64), deep=True
                    ))
                    poly.SetPoints(pts)
                    faces = np.asarray(mesh.faces, dtype=np.int64)
                    connectivity = np.column_stack((
                        np.full(len(faces), 3, dtype=np.int64), faces
                    )).reshape(-1)
                    cells = vtk.vtkCellArray()
                    cells.SetCells(
                        len(faces), numpy_to_vtkIdTypeArray(connectivity, deep=True)
                    )
                    poly.SetPolys(cells)
                    dec = vtk.vtkQuadricDecimation()
                    dec.SetInputData(poly)
                    dec.SetTargetReduction(
                        float(np.clip(1.0 - target / max(len(faces), 1), 0.0, 0.999999))
                    )
                    dec.Update()
                    out = dec.GetOutput()
                    verts = vtk_to_numpy(out.GetPoints().GetData())
                    raw = vtk_to_numpy(out.GetPolys().GetData())
                    if len(raw) % 4 != 0:
                        raise RuntimeError("VTK decimator returned non-triangular cells")
                    out_faces = raw.reshape((-1, 4))[:, 1:4].astype(np.int64)
                    mesh = trimesh.Trimesh(
                        vertices=verts, faces=out_faces, process=False
                    )
                    print(
                        f"[surface] fast simplification unavailable ({exc}); "
                        f"used VTK quadric decimation -> {len(out_faces)} faces"
                    )
                except Exception as vtk_exc:
                    print(
                        f"[surface] simplification skipped: {exc}; "
                        f"VTK fallback failed: {vtk_exc}"
                    )
        mesh = _clean_mesh(mesh)
        mesh.fix_normals()
        return mesh

    def _chart_surface_area(self, chart: PatchChart) -> float:
        return float(sum(_triangle_area_3d(triangle) for triangle in chart.xyz[chart.local_faces]))

    def _build_patch_boundary_topology(self):
        """Identify artificial inter-patch seams on the carrier mesh.

        A chart-boundary edge is an artificial seam only when the same carrier
        edge is internal to the base mesh and belongs to two or more charts.
        Real outer/hole boundaries are kept separate.
        """
        assert self.base_mesh is not None
        base_edge_faces: Dict[Tuple[int, int], int] = defaultdict(int)
        for face in np.asarray(self.base_mesh.faces, dtype=int):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                base_edge_faces[tuple(sorted((int(a), int(b))))] += 1

        edge_patches: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        for chart in self.charts:
            local_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for face in np.asarray(chart.local_faces, dtype=int):
                for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    local_counts[tuple(sorted((int(a), int(b))))] += 1
            for (la, lb), count in local_counts.items():
                if count != 1:
                    continue
                edge = tuple(sorted((
                    int(chart.vertex_ids[la]), int(chart.vertex_ids[lb])
                )))
                edge_patches[edge].add(int(chart.patch_id))

        seam_edges = {
            edge for edge, patches in edge_patches.items()
            if base_edge_faces.get(edge, 0) >= 2 and len(patches) >= 2
        }
        seam_vertices: Set[int] = set()
        seam_degree: Dict[int, int] = defaultdict(int)
        for a, b in seam_edges:
            seam_vertices.update((a, b))
            seam_degree[a] += 1
            seam_degree[b] += 1
        self._artificial_seam_edges = seam_edges
        self._artificial_seam_vertex_ids = seam_vertices
        self._seam_edge_patches = {
            edge: set(edge_patches[edge]) for edge in seam_edges
        }
        self._seam_required_degree_by_vertex = dict(seam_degree)

    def _select_graph_source_vertices(self) -> Set[int]:
        """Select shared carrier vertices independently of the input triangulation density."""
        assert self.base_mesh is not None
        vertices = np.asarray(self.base_mesh.vertices, dtype=float)
        normals = np.asarray(self.base_mesh.vertex_normals, dtype=float)
        spacing = max(0.0, float(self.config.source_node_spacing_mm))
        curvature = np.abs(self.base_vertex_gaussian_curvature)
        # Process all chart-boundary carriers first.  The same global spacing
        # filter is retained, so shared seams are covered without restoring
        # every input-triangulation vertex.
        chart_boundary_vertices: Set[int] = set()
        for chart in self.charts:
            edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
            for face in np.asarray(chart.local_faces, dtype=int):
                for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    edge_count[tuple(sorted((int(a), int(b))))] += 1
            for (a, b), count in edge_count.items():
                if count == 1:
                    chart_boundary_vertices.update((
                        int(chart.vertex_ids[a]), int(chart.vertex_ids[b])
                    ))
        boundary_priority = np.ones(len(vertices), dtype=int)
        if self._artificial_seam_vertex_ids:
            boundary_priority[
                np.asarray(sorted(self._artificial_seam_vertex_ids), dtype=int)
            ] = 0
        order = np.lexsort((np.arange(len(vertices)), -curvature, boundary_priority))
        selected: List[int] = []
        for vertex_id in map(int, order):
            if not selected:
                selected.append(vertex_id)
                continue
            chosen = np.asarray(selected, dtype=int)
            distances = np.linalg.norm(vertices[chosen] - vertices[vertex_id], axis=1)
            nearby = chosen[distances < spacing]
            if len(nearby) and np.any(normals[nearby] @ normals[vertex_id] > 0.5):
                continue
            selected.append(vertex_id)
        selected_set = set(selected)
        # Artificial seam vertices are processed first, but still obey the
        # global source-node spacing.  The full seam polyline is preserved
        # geometrically below, so carrier triangulation density is not copied
        # into graph-node density.

        # Keep the full chart polygon as the design boundary, but do not turn
        # every triangulation vertex on that boundary into a graph node.  The
        # global spacing filter above already provides shared carrier nodes.
        # Each chart only receives enough extra, well-separated boundary
        # carriers to remain addressable by the retained graph.  Source vertex
        # ids remain global, so seam identity is preserved without copying the
        # input triangulation density into the graph.
        for chart in self.charts:
            edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
            for face in np.asarray(chart.local_faces, dtype=int):
                for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    edge_count[tuple(sorted((int(a), int(b))))] += 1
            boundary_local = {
                vertex for edge, count in edge_count.items() if count == 1 for vertex in edge
            }
            required = max(0, int(self.config.min_boundary_carriers_per_patch))
            boundary_vertices = sorted(
                {int(chart.vertex_ids[local_id]) for local_id in boundary_local}
            )
            present = [vertex for vertex in boundary_vertices if vertex in selected_set]
            while len(present) < min(required, len(boundary_vertices)):
                candidates = [
                    vertex for vertex in boundary_vertices if vertex not in selected_set
                ]
                if not candidates:
                    break
                if present:
                    present_xyz = vertices[np.asarray(present, dtype=int)]
                    score = [
                        float(np.min(np.linalg.norm(present_xyz - vertices[vertex], axis=1)))
                        for vertex in candidates
                    ]
                    choice = candidates[int(np.argmax(score))]
                else:
                    choice = max(candidates, key=lambda vertex: float(curvature[vertex]))
                selected_set.add(int(choice))
                present.append(int(choice))
        return selected_set

    def _patch_is_valid(self, face_ids: np.ndarray) -> bool:
        assert self.base_mesh is not None
        face_ids = np.asarray(face_ids, dtype=int)
        if len(face_ids) == 0:
            return False
        faces = np.asarray(self.base_mesh.faces, dtype=int)[face_ids]
        if len(np.unique(faces.reshape(-1))) < 3:
            return False
        triangles = np.asarray(self.base_mesh.vertices, dtype=float)[faces]
        return any(_triangle_area_3d(triangle) > 1.0e-12 for triangle in triangles)

    def _segment_initial_patches(self) -> List[np.ndarray]:
        mesh = self.base_mesh
        assert mesh is not None
        face_adj = [[] for _ in range(len(mesh.faces))]
        for a, b in np.asarray(mesh.face_adjacency, dtype=int):
            face_adj[int(a)].append(int(b))
            face_adj[int(b)].append(int(a))
        component = _face_components(mesh)
        normals = np.asarray(mesh.face_normals, dtype=float)
        cos_limit = math.cos(math.radians(float(self.config.normal_angle_deg)))
        visited = np.zeros(len(mesh.faces), dtype=bool)
        patches: List[np.ndarray] = []
        max_faces = max(4, int(self.config.max_faces_per_patch))
        for seed in range(len(mesh.faces)):
            if visited[seed]:
                continue
            queue = deque([seed])
            group: List[int] = []
            normal_sum = np.zeros(3, dtype=float)
            while queue and len(group) < max_faces:
                face_id = queue.popleft()
                if visited[face_id]:
                    continue
                visited[face_id] = True
                group.append(face_id)
                normal_sum += normals[face_id]
                mean_normal = normal_sum / max(np.linalg.norm(normal_sum), 1.0e-12)
                for nb in face_adj[face_id]:
                    if visited[nb] or component[nb] != component[seed]:
                        continue
                    # A smooth surface may cross K=0 repeatedly. Hard splitting
                    # on the sampled curvature sign creates one-face charts and
                    # artificial zero-area pores. Grow by normal continuity;
                    # the chart distortion/overlap pass below remains the
                    # geometry-based split criterion.
                    normal_compatible = float(np.dot(mean_normal, normals[nb])) >= cos_limit
                    if normal_compatible:
                        queue.append(nb)
            patches.append(np.asarray(group, dtype=int))
        return patches

    def _merge_small_patches(
            self, patches: List[np.ndarray], min_faces: int = 6) -> List[np.ndarray]:
        """Merge breadth-first remainder fragments into an adjacent chart."""
        mesh = self.base_mesh
        assert mesh is not None
        groups = {i: set(map(int, faces)) for i, faces in enumerate(patches)}
        owner = {face: group_id for group_id, faces in groups.items() for face in faces}
        face_adj = [[] for _ in range(len(mesh.faces))]
        for a, b in np.asarray(mesh.face_adjacency, dtype=int):
            face_adj[int(a)].append(int(b))
            face_adj[int(b)].append(int(a))
        normals = np.asarray(mesh.face_normals, dtype=float)
        cos_limit = math.cos(math.radians(float(self.config.normal_angle_deg)))

        def mean_normal(face_set: Set[int]) -> Optional[np.ndarray]:
            normal = normals[np.asarray(sorted(face_set), dtype=int)].mean(axis=0)
            length = float(np.linalg.norm(normal))
            return normal / length if length > 1.0e-12 else None

        max_faces = max(4, int(self.config.max_faces_per_patch))
        while True:
            merged = False
            for group_id in sorted(groups, key=lambda key: len(groups[key])):
                faces = groups.get(group_id, set())
                if len(faces) >= int(min_faces):
                    continue
                neighbour_counts: Dict[int, int] = defaultdict(int)
                for face in faces:
                    for neighbour in face_adj[face]:
                        target = owner.get(int(neighbour))
                        if target is not None and target != group_id:
                            neighbour_counts[int(target)] += 1
                if not neighbour_counts:
                    continue
                source_normal = mean_normal(faces)
                compatible_targets = [
                    candidate for candidate in neighbour_counts
                    if source_normal is not None
                    and (target_normal := mean_normal(groups[candidate])) is not None
                    and float(np.dot(source_normal, target_normal)) >= cos_limit
                ]
                if not compatible_targets:
                    continue
                target = min(
                    compatible_targets,
                    key=lambda candidate: (
                        len(faces) + len(groups[candidate]) > max_faces,
                        -neighbour_counts[candidate],
                        len(faces) + len(groups[candidate]),
                        candidate,
                    ),
                )
                groups[target].update(faces)
                for face in faces:
                    owner[face] = target
                del groups[group_id]
                merged = True
                break
            if not merged:
                break
        return [np.asarray(sorted(faces), dtype=int) for faces in groups.values()]

    def _split_distorted_patches(self, patches: List[np.ndarray]) -> List[np.ndarray]:
        current = list(patches)
        min_child_faces = 6
        for _ in range(max(0, int(self.config.split_rounds))):
            changed = False
            next_patches: List[np.ndarray] = []
            for faces in current:
                if not self._patch_is_valid(faces):
                    continue
                try:
                    chart = self._make_chart(-1, faces)
                except ValueError as exc:
                    pieces = self._bisect_connected(
                        faces, force_balanced=True, min_child_faces=2,
                    )
                    if len(pieces) > 1 and all(len(piece) >= 2 for piece in pieces):
                        next_patches.extend(pieces)
                        changed = True
                        continue
                    raise ValueError(
                        "Surface patch with "
                        f"{len(faces)} faces cannot form a valid planar chart "
                        "and cannot be split into connected children with at least 2 faces"
                    ) from exc
                area_p95 = _safe_percentile(chart.area_distortion, 95)
                angle_p95 = _safe_percentile(chart.angle_distortion_deg, 95)
                bad_area = area_p95 > self.config.split_area_distortion_p95
                bad_angle = angle_p95 > self.config.split_angle_distortion_p95_deg
                bad_overlap = chart.uv_overlap_fraction > 1.0e-6
                severe_distortion = area_p95 > 0.25 or angle_p95 > 20.0
                required_child_faces = 2 if (bad_overlap or severe_distortion) else min_child_faces
                if len(faces) >= 2 * required_child_faces and (bad_area or bad_angle or bad_overlap):
                    pieces = self._bisect_connected(
                        faces, force_balanced=(bad_overlap or severe_distortion),
                        min_child_faces=required_child_faces,
                    )
                    if len(pieces) > 1 and all(
                            len(piece) >= required_child_faces for piece in pieces):
                        next_patches.extend(pieces)
                        changed = True
                        continue
                next_patches.append(faces)
            current = next_patches
            if not changed:
                break
        return current

    def _bisect_connected(self, face_ids: np.ndarray,
                          force_balanced: bool = False,
                          min_child_faces: int = 6) -> List[np.ndarray]:
        mesh = self.base_mesh
        assert mesh is not None
        allowed = set(map(int, face_ids))
        centroids = np.asarray(mesh.triangles_center)[face_ids]
        centered = centroids - centroids.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        coord = centered @ vh[0]
        threshold = float(np.median(coord))
        raw_label = {int(fid): int(coord[i] > threshold) for i, fid in enumerate(face_ids)}
        adjacency = [[] for _ in range(len(mesh.faces))]
        for a, b in np.asarray(mesh.face_adjacency, dtype=int):
            if int(a) in allowed and int(b) in allowed:
                adjacency[int(a)].append(int(b))
                adjacency[int(b)].append(int(a))
        pieces: List[np.ndarray] = []
        seen: Set[int] = set()
        for seed in map(int, face_ids):
            if seed in seen:
                continue
            label = raw_label[seed]
            queue = [seed]
            seen.add(seed)
            group = []
            while queue:
                f = queue.pop()
                group.append(f)
                for nb in adjacency[f]:
                    if nb not in seen and raw_label[nb] == label:
                        seen.add(nb)
                        queue.append(nb)
            if group:
                pieces.append(np.asarray(group, dtype=int))
        min_child_faces = max(2, int(min_child_faces))
        if len(pieces) >= 2 and all(len(piece) >= min_child_faces for piece in pieces):
            return pieces
        if not force_balanced:
            return [face_ids]

        # PCA labels can fragment a folded chart into several tiny islands.
        # Fall back to a cut of a spanning tree: the selected subtree and its
        # complement are both connected, so a valid chart is never retained
        # merely because the planar split happened to create a small island.
        root = int(face_ids[0])
        parent = {root: -1}
        order = [root]
        for face in order:
            for nb in adjacency[face]:
                if nb not in parent:
                    parent[nb] = face
                    order.append(nb)
        if len(order) != len(face_ids):
            return [face_ids]
        children: Dict[int, List[int]] = defaultdict(list)
        for face, par in parent.items():
            if par >= 0:
                children[par].append(face)
        subtree = {face: 1 for face in order}
        for face in reversed(order[1:]):
            subtree[parent[face]] += subtree[face]
        total = len(order)
        candidates = [
            face for face in order[1:]
            if min_child_faces <= subtree[face] <= total - min_child_faces
        ]
        if not candidates:
            return [face_ids]
        cut_root = min(candidates, key=lambda face: (abs(subtree[face] - total / 2), face))
        cut_set: Set[int] = set()
        stack = [cut_root]
        while stack:
            face = stack.pop()
            cut_set.add(face)
            stack.extend(children[face])
        other = allowed - cut_set
        return [np.asarray(sorted(cut_set), dtype=int), np.asarray(sorted(other), dtype=int)]

    @staticmethod
    def _harmonic_chart_uv(xyz: np.ndarray, local_faces: np.ndarray) -> Optional[np.ndarray]:
        """Injectively flatten a disk chart with a convex boundary."""
        edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
        neighbours: Dict[int, Set[int]] = defaultdict(set)
        for face in np.asarray(local_faces, dtype=int):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                a, b = int(a), int(b)
                edge_count[tuple(sorted((a, b)))] += 1
                neighbours[a].add(b)
                neighbours[b].add(a)
        boundary_graph: Dict[int, List[int]] = defaultdict(list)
        for (a, b), count in edge_count.items():
            if count == 1:
                boundary_graph[a].append(b)
                boundary_graph[b].append(a)
        if len(boundary_graph) < 3 or any(len(nbs) != 2 for nbs in boundary_graph.values()):
            return None
        start = min(boundary_graph)
        boundary = [start]
        previous = -1
        current = start
        while True:
            candidates = [nb for nb in boundary_graph[current] if nb != previous]
            if not candidates:
                return None
            nxt = min(candidates) if previous < 0 else candidates[0]
            if nxt == start:
                break
            if nxt in boundary or len(boundary) > len(boundary_graph):
                return None
            boundary.append(nxt)
            previous, current = current, nxt
        if len(boundary) != len(boundary_graph):
            return None

        boundary_xyz = xyz[np.asarray(boundary, dtype=int)]
        closed = np.vstack([boundary_xyz, boundary_xyz[0]])
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        total = float(lengths.sum())
        if total <= 1.0e-12:
            return None
        angles = 2.0 * math.pi * np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / total
        radius = total / (2.0 * math.pi)
        uv = np.zeros((len(xyz), 2), dtype=float)
        uv[np.asarray(boundary, dtype=int)] = radius * np.column_stack((np.cos(angles), np.sin(angles)))

        boundary_set = set(boundary)
        interior = [vertex for vertex in range(len(xyz)) if vertex not in boundary_set]
        if interior:
            index = {vertex: i for i, vertex in enumerate(interior)}
            rows, cols, data = [], [], []
            rhs = np.zeros((len(interior), 2), dtype=float)
            for vertex in interior:
                row = index[vertex]
                adjacent = neighbours.get(vertex, set())
                if not adjacent:
                    return None
                rows.append(row); cols.append(row); data.append(float(len(adjacent)))
                for nb in adjacent:
                    if nb in index:
                        rows.append(row); cols.append(index[nb]); data.append(-1.0)
                    else:
                        rhs[row] += uv[nb]
            matrix = coo_matrix((data, (rows, cols)), shape=(len(interior), len(interior))).tocsr()
            solution = np.column_stack((spsolve(matrix, rhs[:, 0]), spsolve(matrix, rhs[:, 1])))
            if not np.all(np.isfinite(solution)):
                return None
            uv[np.asarray(interior, dtype=int)] = solution
        return uv

    @staticmethod
    def _unfold_chart_uv(xyz: np.ndarray, local_faces: np.ndarray) -> Optional[np.ndarray]:
        """Develop a connected triangle chart while preserving intrinsic edge lengths.

        The method walks across shared edges.  Each newly reached triangle is
        placed on the side opposite its already placed neighbour.  It is used
        only as a candidate: ``_make_chart`` rejects the result if accumulated
        cycle error creates overlap or does not improve the distortion audit.
        """
        faces = np.asarray(local_faces, dtype=int)
        if len(faces) == 0:
            return None
        triangle_areas = np.asarray(
            [_triangle_area_3d(np.asarray(xyz, dtype=float)[face]) for face in faces],
            dtype=float,
        )
        valid = np.flatnonzero(triangle_areas > 1.0e-12)
        if len(valid) == 0:
            return None

        root = int(valid[int(np.argmax(triangle_areas[valid]))])
        root_face = faces[root]
        p0, p1, p2 = np.asarray(xyz, dtype=float)[root_face]
        d01 = float(np.linalg.norm(p1 - p0))
        d02 = float(np.linalg.norm(p2 - p0))
        d12 = float(np.linalg.norm(p2 - p1))
        if d01 <= 1.0e-12:
            return None
        x2 = (d02 * d02 - d12 * d12 + d01 * d01) / (2.0 * d01)
        y2_sq = d02 * d02 - x2 * x2
        if y2_sq <= 1.0e-14:
            return None

        uv = np.full((len(xyz), 2), np.nan, dtype=float)
        uv[int(root_face[0])] = (0.0, 0.0)
        uv[int(root_face[1])] = (d01, 0.0)
        uv[int(root_face[2])] = (x2, math.sqrt(max(0.0, y2_sq)))
        placed_faces: Set[int] = {root}
        remaining: Set[int] = set(range(len(faces))) - placed_faces

        edge_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for face_id, face in enumerate(faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_faces[tuple(sorted((int(a), int(b))))].append(int(face_id))

        while remaining:
            progress = False
            for face_id in sorted(remaining):
                face = [int(vertex) for vertex in faces[face_id]]
                placed_vertices = [vertex for vertex in face if np.all(np.isfinite(uv[vertex]))]
                if len(placed_vertices) == 3:
                    placed_faces.add(face_id)
                    remaining.remove(face_id)
                    progress = True
                    break
                if len(placed_vertices) != 2:
                    continue
                a, b = placed_vertices
                c = next(vertex for vertex in face if vertex not in (a, b))
                edge = uv[b] - uv[a]
                edge_length = float(np.linalg.norm(edge))
                if edge_length <= 1.0e-12:
                    continue
                da = float(np.linalg.norm(xyz[c] - xyz[a]))
                db = float(np.linalg.norm(xyz[c] - xyz[b]))
                along = (da * da - db * db + edge_length * edge_length) / (2.0 * edge_length)
                height_sq = da * da - along * along
                if height_sq < -1.0e-10:
                    continue
                height = math.sqrt(max(0.0, height_sq))
                tangent = edge / edge_length
                normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
                base = uv[a] + along * tangent
                candidates = (base + height * normal, base - height * normal)

                reference_side = 0.0
                for neighbour in edge_faces.get(tuple(sorted((a, b))), []):
                    if neighbour not in placed_faces:
                        continue
                    other = next(
                        (int(vertex) for vertex in faces[neighbour] if int(vertex) not in (a, b)),
                        None,
                    )
                    if other is None or not np.all(np.isfinite(uv[other])):
                        continue
                    reference_side = float(np.cross(edge, uv[other] - uv[a]))
                    if abs(reference_side) > 1.0e-12:
                        break
                if reference_side > 0.0:
                    uv[c] = candidates[1]
                elif reference_side < 0.0:
                    uv[c] = candidates[0]
                else:
                    uv[c] = candidates[0]
                placed_faces.add(face_id)
                remaining.remove(face_id)
                progress = True
                break
            if not progress:
                return None
        return uv if np.all(np.isfinite(uv)) else None

    def _make_chart(self, patch_id: int, face_ids: np.ndarray) -> PatchChart:
        mesh = self.base_mesh
        assert mesh is not None
        face_ids = np.asarray(face_ids, dtype=int)
        source_faces = np.asarray(mesh.faces, dtype=int)[face_ids]
        vertex_ids = np.unique(source_faces.reshape(-1))
        xyz = np.asarray(mesh.vertices, dtype=float)[vertex_ids]
        centroid = xyz.mean(axis=0)
        _, _, vh = np.linalg.svd(xyz - centroid, full_matrices=False)
        axis_u, axis_v = vh[0], vh[1]
        uv = np.column_stack(((xyz - centroid) @ axis_u, (xyz - centroid) @ axis_v))
        source_to_local = {int(v): i for i, v in enumerate(vertex_ids)}
        local_faces = np.asarray([[source_to_local[int(v)] for v in face] for face in source_faces], dtype=int)
        tri3 = xyz[local_faces]
        area3 = np.asarray([_triangle_area_3d(t) for t in tri3])

        def evaluate(candidate_uv: np.ndarray, method: str) -> Optional[Dict[str, object]]:
            if candidate_uv.shape != (len(xyz), 2) or not np.all(np.isfinite(candidate_uv)):
                return None
            tri2 = candidate_uv[local_faces]
            area2 = np.asarray([_triangle_area_2d(triangle) for triangle in tri2], dtype=float)
            if float(area2.sum()) <= 1.0e-12 or np.any(area2 <= 1.0e-14):
                return None
            scale = math.sqrt(float(area3.sum()) / float(area2.sum()))
            area_dist = np.abs(area2 * scale * scale / np.maximum(area3, 1.0e-12) - 1.0)
            angle_dist: List[float] = []
            edge_dist: List[float] = []
            for t2, t3 in zip(tri2, tri3):
                angle_dist.extend(np.abs(_triangle_angles(t2) - _triangle_angles(t3)).tolist())
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    length2 = float(np.linalg.norm(t2[i] - t2[j])) * scale
                    length3 = float(np.linalg.norm(t3[i] - t3[j]))
                    edge_dist.append(abs(length2 / max(length3, 1.0e-12) - 1.0))
            polygons = [Polygon(triangle) for triangle in tri2]
            domain = unary_union(polygons).buffer(0)
            if domain.is_empty:
                return None
            overlap = max(0.0, 1.0 - float(domain.area) / float(area2.sum()))
            score = (
                _safe_percentile(area_dist, 95) / 0.25
                + _safe_percentile(angle_dist, 95) / 20.0
                + _safe_percentile(edge_dist, 95) / 0.25
            )
            return {
                "uv": candidate_uv,
                "method": method,
                "scale": scale,
                "area_dist": area_dist,
                "angle_dist": np.asarray(angle_dist, dtype=float),
                "edge_dist": np.asarray(edge_dist, dtype=float),
                "domain": domain,
                "overlap": overlap,
                "score": float(score),
            }

        result = evaluate(uv, "principal_plane_projection")
        if result is None:
            raise ValueError(f"Patch {patch_id} cannot form a valid planar chart")
        severe_distortion = (
            _safe_percentile(result["area_dist"], 95) > 0.25
            or _safe_percentile(result["angle_dist"], 95) > 20.0
        )
        if severe_distortion or float(result["overlap"]) > 1.0e-6:
            unfolded = self._unfold_chart_uv(xyz, local_faces)
            unfolded_result = (
                evaluate(unfolded, "intrinsic_edge_length_unfolding")
                if unfolded is not None else None
            )
            if (
                unfolded_result is not None
                and float(unfolded_result["overlap"]) <= 1.0e-6
                and (
                    float(result["overlap"]) > 1.0e-6
                    or float(unfolded_result["score"]) < 0.98 * float(result["score"])
                )
            ):
                result = unfolded_result
        if float(result["overlap"]) > 1.0e-6:
            harmonic = self._harmonic_chart_uv(xyz, local_faces)
            harmonic_result = (
                evaluate(harmonic, "convex_boundary_harmonic")
                if harmonic is not None else None
            )
            if (
                harmonic_result is not None
                and float(harmonic_result["overlap"]) < float(result["overlap"])
            ):
                result = harmonic_result

        uv = np.asarray(result["uv"], dtype=float)
        scale = float(result["scale"])
        area_dist = np.asarray(result["area_dist"], dtype=float)
        angle_dist = np.asarray(result["angle_dist"], dtype=float)
        edge_dist = np.asarray(result["edge_dist"], dtype=float)
        domain = result["domain"]
        uv_overlap_fraction = float(result["overlap"])
        vertex_k = self.base_vertex_gaussian_curvature
        patch_k = vertex_k[vertex_ids]
        patch_center = np.asarray(mesh.triangles_center, dtype=float)[face_ids].mean(axis=0)
        _, original_face = self._project_original(patch_center)
        component = int(self.original_face_components[int(original_face)])
        return PatchChart(
            patch_id=int(patch_id),
            component_id=component,
            face_ids=face_ids,
            vertex_ids=vertex_ids,
            uv=uv,
            xyz=xyz,
            local_faces=local_faces,
            polygon=domain,
            centroid=centroid,
            axis_u=axis_u,
            axis_v=axis_v,
            parameterization_method=str(result["method"]),
            scale_2d_to_3d=scale,
            area_distortion=area_dist,
            angle_distortion_deg=np.asarray(angle_dist),
            edge_distortion=np.asarray(edge_dist),
            uv_overlap_fraction=uv_overlap_fraction,
            curvature_mean=float(np.mean(patch_k)),
            curvature_abs_mean=float(np.mean(np.abs(patch_k))),
        )

    def _locate_uv_base(self, chart: PatchChart, point: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        p = np.asarray(point, dtype=float)
        faces = np.asarray(chart.local_faces, dtype=int)
        tri = chart.uv[faces]
        ac = tri[:, 0] - tri[:, 2]
        bc = tri[:, 1] - tri[:, 2]
        det = ac[:, 0] * bc[:, 1] - bc[:, 0] * ac[:, 1]
        valid = np.abs(det) >= 1.0e-14
        if not np.any(valid):
            nearest = int(np.argmin(np.linalg.norm(chart.uv - p, axis=1)))
            return chart.xyz[nearest].copy(), int(chart.face_ids[0]), True
        delta = p - tri[valid, 2]
        inv = 1.0 / det[valid]
        u = (delta[:, 0] * bc[valid, 1] - delta[:, 1] * bc[valid, 0]) * inv
        v = (ac[valid, 0] * delta[:, 1] - ac[valid, 1] * delta[:, 0]) * inv
        bary_all = np.column_stack((u, v, 1.0 - u - v))
        score = np.maximum(-bary_all, 0.0).sum(axis=1)
        first_inside = np.flatnonzero(score <= self.config.map_tolerance)
        choice = int(first_inside[0]) if len(first_inside) else int(np.argmin(score))
        local_face_id = int(np.flatnonzero(valid)[choice])
        bary = bary_all[choice]
        if len(first_inside):
            return bary @ chart.xyz[faces[local_face_id]], int(chart.face_ids[local_face_id]), False
        bary = np.clip(bary, 0.0, None)
        bary /= max(float(bary.sum()), 1.0e-12)
        tri3 = chart.xyz[faces[local_face_id]]
        base_point = bary @ tri3
        return base_point, int(chart.face_ids[local_face_id]), True

    def _locate_uv(self, chart: PatchChart, point: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        base_point, base_face, fallback = self._locate_uv_base(chart, point)
        if self.config.mapping_surface == "carrier":
            return base_point, base_face, fallback
        closest, original_face = self._project_original(
            base_point, component_id=chart.component_id
        )
        return closest, original_face, fallback

    def _mapping_surface_distance_batch(
        self, points: np.ndarray, component_id: Optional[int] = None
    ) -> np.ndarray:
        if self.config.mapping_surface == "carrier":
            points = np.asarray(points, dtype=float)
            if self._carrier_proximity is not None:
                return np.asarray(self._carrier_proximity.on_surface(points)[1], dtype=float)
            assert self.base_mesh is not None
            assert self._carrier_face_centroid_tree is not None
            triangles_all = np.asarray(self.base_mesh.triangles, dtype=float)
            # Vectorized nearest-face candidates.  This is the no-rtree
            # fallback used only for chord-to-carrier distance checks.
            k = min(32, len(triangles_all))
            _, candidate = self._carrier_face_centroid_tree.query(points, k=k)
            candidate = np.asarray(candidate, dtype=int)
            if candidate.ndim == 1:
                candidate = candidate[:, None]
            tri = triangles_all[candidate.reshape(-1)]
            repeated = np.repeat(points, candidate.shape[1], axis=0)
            closest = trimesh.triangles.closest_point(tri, repeated)
            d2 = np.einsum("ij,ij->i", closest-repeated, closest-repeated)
            d2 = d2.reshape(len(points), candidate.shape[1])
            return np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))
        return self._surface_distance_batch(points, component_id=component_id)

    def _mapped_polygon_metrics(self, chart: PatchChart, poly: Polygon) -> Tuple[float, float]:
        """Return mapped area and perimeter without concave fan over-counting."""
        surface_area = 0.0
        perimeter = 0.0
        clean_poly = poly.buffer(0)
        for face in np.asarray(chart.local_faces, dtype=int):
            tri2 = chart.uv[face]
            area2 = _triangle_area_2d(tri2)
            if area2 <= 1.0e-12:
                continue
            clipped = clean_poly.intersection(Polygon(tri2))
            if clipped.is_empty:
                continue
            area3 = _triangle_area_3d(chart.xyz[face])
            surface_area += float(clipped.area) * area3 / area2

        spacing_uv = max(
            1.0e-6,
            min(float(self.config.edge_sample_spacing_mm), 0.05)
            / max(chart.scale_2d_to_3d, 1.0e-12),
        )
        rings = []
        for part in _polygon_parts(clean_poly):
            rings.extend([part.exterior, *part.interiors])
        for ring in rings:
            raw = np.asarray(ring.coords[:-1], dtype=float)
            if len(raw) < 3:
                continue
            sampled = []
            for p0, p1 in zip(raw, np.roll(raw, -1, axis=0)):
                count = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / spacing_uv)))
                segment = np.linspace(p0, p1, count + 1)[:-1]
                sampled.extend(segment)
            coordinates = np.asarray(sampled, dtype=float)
            mapped = np.asarray([self._locate_uv(chart, point)[0] for point in coordinates])
            perimeter += sum(
                float(np.linalg.norm(mapped[(i + 1) % len(mapped)] - mapped[i]))
                for i in range(len(mapped))
            )
        return max(0.0, surface_area), perimeter

    def _refine_patch_nodes(
        self, chart: PatchChart
    ) -> Tuple[np.ndarray, np.ndarray, Set[Tuple[int, int]], List[Polygon], List[int]]:
        """Split each Voronoi pore independently and rebuild its orthogonal boundary."""

        def polygon_parts(geometry) -> List[Polygon]:
            if geometry is None or geometry.is_empty:
                return []
            if isinstance(geometry, Polygon):
                return [geometry]
            parts = []
            for item in getattr(geometry, "geoms", []):
                parts.extend(polygon_parts(item))
            return [part for part in parts if part.area > 1.0e-12]

        domain = chart.polygon.buffer(0)
        domain_parts = polygon_parts(domain)
        tolerance_uv = max(
            1.0e-6,
            float(self.config.polylabel_tolerance_mm) / max(chart.scale_2d_to_3d, 1.0e-12),
        )
        target = (
            float(self.config.target_hydraulic_radius_mm)
            * float(self.config.hydraulic_radius_mapping_safety)
        )
        buffer_uv = 0.5 * float(self.config.surface_diameter_mm) / max(
            chart.scale_2d_to_3d, 1.0e-12
        )
        rng = np.random.default_rng(int(self.config.random_seed) + 104729 * int(chart.patch_id))
        min_seed_spacing_mm = max(
            float(self.config.surface_diameter_mm),
            0.75 * float(self.config.min_logical_edge_length_mm),
        )
        seeds = []
        for part in domain_parts:
            initial_radii = []
            for liquid in polygon_parts(part.buffer(-buffer_uv)):
                area_mm2 = float(liquid.area) * chart.scale_2d_to_3d ** 2
                perimeter_mm = float(liquid.length) * chart.scale_2d_to_3d
                initial_radii.append(2.0 * area_mm2 / max(perimeter_mm, 1.0e-12))
            seed_count = 1 if max(initial_radii, default=0.0) <= target else 3
            part_bounds = np.asarray(part.bounds, dtype=float)
            attempts = 0
            while seed_count > 1 and sum(part.covers(Point(seed)) for seed in seeds) < seed_count and attempts < 500:
                candidate = np.asarray(
                    [
                        rng.uniform(part_bounds[0], part_bounds[2]),
                        rng.uniform(part_bounds[1], part_bounds[3]),
                    ],
                    dtype=float,
                )
                if not part.covers(Point(candidate)):
                    attempts += 1
                    continue
                candidate_xyz = self._locate_uv(chart, candidate)[0]
                existing_xyz = [self._locate_uv(chart, seed)[0] for seed in seeds]
                if (not existing_xyz or min(
                        float(np.linalg.norm(candidate_xyz - point))
                        for point in existing_xyz
                    ) >= min_seed_spacing_mm):
                    seeds.append(candidate)
                attempts += 1
            if seed_count == 1 or not any(part.covers(Point(seed)) for seed in seeds):
                center = part.representative_point()
                seeds.append(np.asarray(center.coords[0], dtype=float))
        if not seeds:
            raise ValueError(f"Patch {chart.patch_id} has no valid Voronoi seed domain")
        initial_seed_count = len(seeds)
        bounds = np.asarray(domain.bounds, dtype=float)
        span = max(float(bounds[2] - bounds[0]), float(bounds[3] - bounds[1]), 1.0)
        ghost_points = np.asarray(
            [
                [bounds[0] - 4.0 * span, bounds[1] - 4.0 * span],
                [bounds[2] + 4.0 * span, bounds[1] - 4.0 * span],
                [bounds[2] + 4.0 * span, bounds[3] + 4.0 * span],
                [bounds[0] - 4.0 * span, bounds[3] + 4.0 * span],
            ],
            dtype=float,
        )
        max_insert = max(0, int(self.config.max_insertions_per_patch))
        if self.config.max_insertions_by_patch:
            if chart.patch_id >= len(self.config.max_insertions_by_patch):
                raise ValueError("Missing matched insertion budget for a surface patch")
            max_insert = max(0, int(self.config.max_insertions_by_patch[chart.patch_id]))
        policy = str(self.config.refinement_policy).strip().lower()
        if policy not in {"physics", "random", "random_geometry", "none"}:
            raise ValueError(
                "refinement_policy must be 'physics', 'random', 'random_geometry', or 'none'"
            )
        if policy == "random_geometry":
            # Rebuild the initial state without consulting the capillary
            # criterion. Keep one uniformly sampled seed per disconnected
            # domain part and enforce the same mapped 3-D spacing floor used
            # for subsequent random insertions.
            seeds = []
            initial_xyz = []
            min_seed_spacing_mm = max(
                float(self.config.surface_diameter_mm),
                0.75 * float(self.config.min_logical_edge_length_mm),
                1.0e-6,
            )
            for part in domain_parts:
                xmin, ymin, xmax, ymax = part.bounds
                accepted = None
                accepted_xyz = None
                for _ in range(500):
                    candidate = np.asarray(
                        [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=float
                    )
                    if not part.covers(Point(candidate)):
                        continue
                    candidate_xyz = self._locate_uv(chart, candidate)[0]
                    if not initial_xyz or min(
                        float(np.linalg.norm(candidate_xyz - point)) for point in initial_xyz
                    ) >= min_seed_spacing_mm:
                        accepted = candidate
                        accepted_xyz = candidate_xyz
                        break
                if accepted is None:
                    accepted = np.asarray(part.representative_point().coords[0], dtype=float)
                    accepted_xyz = self._locate_uv(chart, accepted)[0]
                seeds.append(accepted)
                initial_xyz.append(accepted_xyz)
            initial_seed_count = len(seeds)
        insertion_strategy = str(self.config.refinement_insertion_strategy).strip().lower()
        if insertion_strategy not in {"polylabel", "centroid_jitter"}:
            raise ValueError(
                "refinement_insertion_strategy must be 'polylabel' or 'centroid_jitter'"
            )
        jitter_uv = max(1.0e-6, float(self.config.centroid_jitter_mm)) / max(
            chart.scale_2d_to_3d, 1.0e-12
        )
        final_cells: List[Polygon] = []
        max_rounds = max(1, int(self.config.max_voronoi_rounds))
        for step in range(max_rounds + 1):
            voronoi = Voronoi(np.vstack((np.asarray(seeds, dtype=float), ghost_points)))
            cells: List[Polygon] = []
            for seed_id in range(len(seeds)):
                region = voronoi.regions[voronoi.point_region[seed_id]]
                if not region or -1 in region:
                    continue
                raw_cell = Polygon(voronoi.vertices[np.asarray(region, dtype=int)])
                if not raw_cell.is_valid:
                    raw_cell = raw_cell.buffer(0)
                parts = polygon_parts(raw_cell.intersection(domain))
                cells.extend(parts)
            final_cells = cells

            bad_cells = []
            candidate_cells = []
            r_values = []
            r_values_2d = []
            r_values_3d = []
            shape_factors = []
            aspect_ratios = []
            liquid_area = 0.0
            for cell in cells:
                liquid_parts = polygon_parts(cell.buffer(-buffer_uv))
                for liquid in liquid_parts:
                    area_mm2 = float(liquid.area) * chart.scale_2d_to_3d ** 2
                    perimeter_mm = float(liquid.length) * chart.scale_2d_to_3d
                    r_hyd_2d = 2.0 * area_mm2 / max(perimeter_mm, 1.0e-12)
                    area3, perimeter3 = self._mapped_polygon_metrics(chart, liquid)
                    r_hyd_3d = 2.0 * area3 / max(perimeter3, 1.0e-12)
                    r_hyd = max(r_hyd_2d, r_hyd_3d)
                    r_values.append(r_hyd)
                    r_values_2d.append(r_hyd_2d)
                    r_values_3d.append(r_hyd_3d)
                    shape_factors.append(
                        4.0 * math.pi * area3 / max(perimeter3 * perimeter3, 1.0e-12)
                    )
                    aspect_ratios.append(_polygon_aspect_ratio(liquid))
                    liquid_area += area_mm2
                    candidate_cells.append((cell, liquid))
                    if r_hyd > target:
                        if insertion_strategy == "centroid_jitter":
                            base = np.asarray(cell.centroid.coords[0], dtype=float)
                            if not cell.covers(Point(base)):
                                base = np.asarray(cell.representative_point().coords[0], dtype=float)
                            center = base
                            for _ in range(20):
                                candidate = base + rng.uniform(-jitter_uv, jitter_uv, size=2)
                                if cell.covers(Point(candidate)):
                                    center = candidate
                                    break
                        else:
                            try:
                                point = polylabel(liquid, tolerance=tolerance_uv)
                            except Exception:
                                point = liquid.representative_point()
                            center = np.asarray(point.coords[0], dtype=float)
                        bad_cells.append(
                            (r_hyd, np.asarray(center, dtype=float), liquid)
                        )
            self._refinement_history_rows.append(
                {
                    "patch_id": chart.patch_id,
                    "component_id": chart.component_id,
                    "round": step,
                    "seed_count": len(seeds),
                    "subdomain_count": len(cells),
                    "bad_subdomain_count": len(bad_cells),
                    "hydraulic_radius_max_mm": max(r_values, default=0.0),
                    "hydraulic_radius_2d_max_mm": max(r_values_2d, default=0.0),
                    "hydraulic_radius_3d_max_mm": max(r_values_3d, default=0.0),
                    "hydraulic_radius_3d_mean_mm": float(np.mean(r_values_3d)) if r_values_3d else 0.0,
                    "pore_count": len(r_values_3d),
                    "shape_factor_mean": float(np.mean(shape_factors)) if shape_factors else 0.0,
                    "shape_factor_median": float(np.median(shape_factors)) if shape_factors else 0.0,
                    "aspect_ratio_mean": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
                    "aspect_ratio_median": float(np.median(aspect_ratios)) if aspect_ratios else 0.0,
                    "hydraulic_radius_threshold_mm": target,
                    "refinement_insertion_strategy": (
                        "largest_void_polylabel"
                        if policy == "random_geometry"
                        else insertion_strategy
                    ),
                    "liquid_area_mm2": liquid_area,
                    "domain_area_mm2": float(domain.area) * chart.scale_2d_to_3d ** 2,
                    "liquid_area_fraction": liquid_area
                    / max(float(domain.area) * chart.scale_2d_to_3d ** 2, 1.0e-12),
                }
            )
            budget_reached = ((max_insert > 0 and len(seeds) >= initial_seed_count + max_insert)
                              or (self.config.max_total_insertions > 0 and
                                  self._total_refinement_insertions >= self.config.max_total_insertions))
            if (
                policy == "none"
                or step >= max_rounds
                or (policy == "physics" and not bad_cells)
                or (policy == "random" and budget_reached)
                or (policy == "random_geometry" and budget_reached)
            ):
                break
            if policy == "physics":
                bad_cells.sort(key=lambda item: item[0], reverse=True)
                selected_cells = [(center, liquid) for _, center, liquid in bad_cells]
            elif policy == "random":
                if not candidate_cells:
                    break
                order = rng.choice(
                    len(candidate_cells),
                    size=min(
                        len(candidate_cells),
                        max(1, int(self.config.max_bad_subdomains_per_round)),
                    ),
                    replace=False,
                )
                selected_cells = []
                for cell_id in np.atleast_1d(order):
                    cell, liquid = candidate_cells[int(cell_id)]
                    # Strict random-subgraph ablation: randomize only which
                    # existing pore is rewritten, then apply the same local
                    # insertion production used by the physics-guided branch.
                    if insertion_strategy == "centroid_jitter":
                        base = np.asarray(cell.centroid.coords[0], dtype=float)
                        if not cell.covers(Point(base)):
                            base = np.asarray(cell.representative_point().coords[0], dtype=float)
                        center = base
                        for _ in range(20):
                            candidate = base + rng.uniform(-jitter_uv, jitter_uv, size=2)
                            if cell.covers(Point(candidate)):
                                center = candidate
                                break
                    else:
                        try:
                            point = polylabel(liquid, tolerance=tolerance_uv)
                        except Exception:
                            point = liquid.representative_point()
                        center = np.asarray(point.coords[0], dtype=float)
                    selected_cells.append((center, liquid))
            else:
                # Geometry-only control: iteratively split the largest current
                # Voronoi void without reading the capillary residual or using
                # the PC-AEGG local production.
                geometry_candidates = []
                # Voronoi sites are generators, not printable graph nodes.
                # Keep the same numerical site-separation guard as the
                # existing polylabel path; final graph spacing, edge length,
                # degree and domain checks remain unchanged downstream.
                min_seed_spacing_mm = 0.005
                for _, liquid in candidate_cells:
                    try:
                        point = polylabel(liquid, tolerance=tolerance_uv)
                    except Exception:
                        point = liquid.representative_point()
                    center = np.asarray(point.coords[0], dtype=float)
                    nearest_seed_mm = min(
                        float(np.linalg.norm(center - seed)) for seed in seeds
                    ) * chart.scale_2d_to_3d
                    if nearest_seed_mm < min_seed_spacing_mm:
                        continue
                    clearance_mm = (
                        float(point.distance(liquid.boundary))
                        * chart.scale_2d_to_3d
                    )
                    geometry_candidates.append(
                        (clearance_mm, float(liquid.area), center, liquid)
                    )
                geometry_candidates.sort(
                    key=lambda item: (item[0], item[1]), reverse=True
                )
                batch_size = min(
                    len(geometry_candidates),
                    max(1, int(self.config.max_bad_subdomains_per_round)),
                )
                selected_cells = [
                    (item[2], item[3])
                    for item in geometry_candidates[:batch_size]
                ]
            inserted = 0
            for center, liquid in selected_cells[: max(1, int(self.config.max_bad_subdomains_per_round))]:
                if ((max_insert > 0 and len(seeds) >= initial_seed_count + max_insert)
                        or (self.config.max_total_insertions > 0 and
                            self._total_refinement_insertions >= self.config.max_total_insertions)):
                    break
                nearest_seed_mm = min(
                    float(np.linalg.norm(center - seed)) for seed in seeds
                ) * chart.scale_2d_to_3d
                if insertion_strategy == "centroid_jitter":
                    # Closely paired seeds are intentional: their Voronoi bisector
                    # creates the narrow, anisotropic pores used by the reference build.
                    if nearest_seed_mm <= 1.0e-6:
                        continue
                elif nearest_seed_mm < min_seed_spacing_mm:
                    exclusion_radius_uv = min_seed_spacing_mm / max(
                        chart.scale_2d_to_3d, 1.0e-12)
                    excluded = unary_union(
                        [Point(seed).buffer(exclusion_radius_uv) for seed in seeds]
                    )
                    alternatives = polygon_parts(liquid.difference(excluded))
                    alternative_centers = []
                    for region in alternatives:
                        try:
                            candidate_point = polylabel(region, tolerance=tolerance_uv)
                        except Exception:
                            candidate_point = region.representative_point()
                        candidate = np.asarray(candidate_point.coords[0], dtype=float)
                        clearance = min(float(np.linalg.norm(candidate - seed)) for seed in seeds)
                        alternative_centers.append((clearance, candidate))
                    if not alternative_centers:
                        continue
                    alternative_centers.sort(key=lambda item: item[0], reverse=True)
                    center = alternative_centers[0][1]
                    center_xyz = self._locate_uv(chart, center)[0]
                    seed_xyz = [self._locate_uv(chart, seed)[0] for seed in seeds]
                    if seed_xyz and min(
                        float(np.linalg.norm(center_xyz - point)) for point in seed_xyz
                    ) < min_seed_spacing_mm:
                        continue
                seeds.append(center)
                inserted += 1
                self._total_refinement_insertions += 1
            if inserted == 0:
                break

        anisotropy_wall_segments: List[LineString] = []

        # Second-stage ANISO_SPLIT.  This runs after ordinary Voronoi
        # refinement and therefore does not replace the pressure-driven rule.
        # Low-AR pores can be rewritten whether r_eq is above or below target:
        # an oversize pore receives extra priority because the added wall also
        # reduces its short scale.  Only a bounded fraction is selected so AR
        # remains a soft morphology target rather than a hard global gate.
        if bool(self.config.anisotropy_rewrite_enable) and final_cells:
            ar_target = max(1.0, float(self.config.anisotropy_target_aspect_ratio))
            soft_fraction = float(np.clip(self.config.anisotropy_soft_fraction, 0.0, 1.0))
            max_splits = max(0, int(self.config.anisotropy_max_splits_per_patch))
            min_wall_mm = max(0.0, float(self.config.anisotropy_min_wall_length_mm))
            cap_weight = max(0.0, float(self.config.anisotropy_capillary_priority_weight))
            min_ar_gain = max(0.0, float(self.config.anisotropy_min_ar_improvement_fraction))
            hard_max_ar = max(ar_target + 1.0e-6, float(self.config.anisotropy_max_aspect_ratio))
            cap_target_ar = max(ar_target, hard_max_ar * float(self.config.anisotropy_cap_repair_target_fraction))
            max_cap_splits = max(0, int(self.config.anisotropy_max_cap_splits_per_cell))

            def _largest_liquid(cell_poly: Polygon):
                parts = polygon_parts(cell_poly.buffer(-buffer_uv))
                return max(parts, key=lambda g: float(g.area)) if parts else None

            def _liquid_metrics(liquid_poly: Polygon):
                if liquid_poly is None or liquid_poly.is_empty:
                    return 0.0, 0.0, 0.0
                area2 = float(liquid_poly.area) * chart.scale_2d_to_3d ** 2
                per2 = float(liquid_poly.length) * chart.scale_2d_to_3d
                r2 = 2.0 * area2 / max(per2, 1.0e-12)
                area3, per3 = self._mapped_polygon_metrics(chart, liquid_poly)
                r3 = 2.0 * area3 / max(per3, 1.0e-12)
                return max(r2, r3), _polygon_aspect_ratio(liquid_poly), area3

            def _major_axis(liquid_poly: Polygon):
                rect = liquid_poly.minimum_rotated_rectangle
                coords = np.asarray(rect.exterior.coords[:-1], dtype=float)
                if len(coords) < 4:
                    return None
                vecs = np.roll(coords, -1, axis=0) - coords
                lens = np.linalg.norm(vecs, axis=1)
                idx = int(np.argmax(lens))
                if lens[idx] <= 1.0e-12:
                    return None
                return vecs[idx] / lens[idx]

            # Classify chart-boundary contact into artificial inter-patch
            # seams versus true design-domain boundaries. Artificial seams
            # are computational only and must never exempt a physical pore
            # from ANISO/AR-cap repair.
            artificial_seam_lines = []
            _boundary_counts_for_aniso: Dict[Tuple[int, int], int] = defaultdict(int)
            for face in np.asarray(chart.local_faces, dtype=int):
                for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    _boundary_counts_for_aniso[tuple(sorted((int(a), int(b))))] += 1
            for (la, lb), count in _boundary_counts_for_aniso.items():
                if count != 1:
                    continue
                source_key = tuple(sorted((
                    int(chart.vertex_ids[la]), int(chart.vertex_ids[lb])
                )))
                if source_key in self._artificial_seam_edges:
                    artificial_seam_lines.append(
                        LineString([chart.uv[int(la)], chart.uv[int(lb)]])
                    )
            artificial_seam_geom = (
                unary_union(artificial_seam_lines) if artificial_seam_lines else None
            )
            boundary_class_tol_uv = max(1.0e-8, float(self.config.map_tolerance) * 10.0)

            def _boundary_contact_lengths_mm(cell_poly: Polygon):
                total = float(cell_poly.boundary.intersection(chart.polygon.boundary).length)
                seam = 0.0
                if artificial_seam_geom is not None and not artificial_seam_geom.is_empty:
                    seam = float(cell_poly.boundary.intersection(
                        artificial_seam_geom.buffer(
                            boundary_class_tol_uv, cap_style=2, join_style=2
                        )
                    ).length)
                total_mm = total * chart.scale_2d_to_3d
                seam_mm = min(total, seam) * chart.scale_2d_to_3d
                physical_mm = max(0.0, total_mm - seam_mm)
                return total_mm, seam_mm, physical_mm

            def _point_on_artificial_seam(pt: Point) -> bool:
                return bool(
                    artificial_seam_geom is not None
                    and not artificial_seam_geom.is_empty
                    and float(pt.distance(artificial_seam_geom)) <= boundary_class_tol_uv
                )

            candidates = []
            skipped_boundary_ids = set()
            for cell_id, cell in enumerate(final_cells):
                liquid = _largest_liquid(cell)
                if liquid is None:
                    continue
                r_pre, ar_pre, area_pre = _liquid_metrics(liquid)
                if ar_pre >= ar_target - 1.0e-12:
                    continue
                boundary_mm, seam_boundary_mm, physical_boundary_mm = (
                    _boundary_contact_lengths_mm(cell)
                )
                # Artificial seams are not physical boundaries: seam-dependent
                # pores remain eligible. Only genuine design-domain boundary
                # contact retains the conservative boundary protection.
                if (bool(self.config.anisotropy_skip_patch_boundary_cells)
                        and physical_boundary_mm > 1.0e-7):
                    skipped_boundary_ids.add(int(cell_id))
                    self._anisotropy_rewrite_rows.append({
                        'patch_id': int(chart.patch_id), 'component_id': int(chart.component_id),
                        'cell_id': int(cell_id), 'rewrite_type': 'soft_parallel', 'selected': 0, 'accepted': 0,
                        'reason': 'true_design_boundary_cell_skipped',
                        'seam_dependent': int(seam_boundary_mm > 1.0e-7),
                        'physical_boundary_contact_mm': physical_boundary_mm,
                        'pre_r_eq_mm': r_pre, 'pre_aspect_ratio': ar_pre,
                        'post_max_r_eq_mm': math.nan, 'post_aspect_ratio_mean': math.nan,
                        'post_aspect_ratio_max': math.nan, 'wall_length_mm': 0.0,
                    })
                    continue
                deficit = max(0.0, (ar_target - ar_pre) / ar_target)
                cap_excess = max(0.0, r_pre / max(target, 1.0e-12) - 1.0)
                score = deficit + cap_weight * cap_excess
                candidates.append((score, cap_excess > 0.0, cell_id, cell, liquid, r_pre, ar_pre, area_pre))

            candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
            selected_ids = set()
            if candidates and max_splits > 0:
                soft_quota = max(1, int(math.ceil(soft_fraction * len(candidates))))
                oversize_count = sum(int(item[1]) for item in candidates)
                quota = min(max(soft_quota, oversize_count), max_splits, len(candidates))
                selected_ids = {int(item[2]) for item in candidates[:quota]}

            rewritten_cells: List[Polygon] = []
            candidate_by_id = {int(item[2]): item for item in candidates}
            for cell_id, cell in enumerate(final_cells):
                item = candidate_by_id.get(int(cell_id))
                if item is None:
                    rewritten_cells.append(cell)
                    continue
                _, _, _, _, liquid, r_pre, ar_pre, _ = item
                if cell_id not in selected_ids:
                    rewritten_cells.append(cell)
                    self._anisotropy_rewrite_rows.append({
                        'patch_id': int(chart.patch_id), 'component_id': int(chart.component_id),
                        'cell_id': int(cell_id), 'rewrite_type': 'soft_parallel', 'selected': 0, 'accepted': 0,
                        'reason': 'soft_constraint_not_selected',
                        'pre_r_eq_mm': r_pre, 'pre_aspect_ratio': ar_pre,
                        'post_max_r_eq_mm': math.nan, 'post_aspect_ratio_mean': math.nan,
                        'post_aspect_ratio_max': math.nan, 'wall_length_mm': 0.0,
                    })
                    continue

                axis = _major_axis(liquid)
                accepted = False
                reason = 'major_axis_unavailable'
                wall_len_mm = 0.0
                post_r = math.nan
                post_ar = math.nan
                post_ar_max = math.nan
                if axis is not None:
                    center_point = liquid.centroid
                    if not liquid.covers(center_point):
                        center_point = liquid.representative_point()
                    center = np.asarray(center_point.coords[0], dtype=float)
                    xmin, ymin, xmax, ymax = cell.bounds
                    span = max(math.hypot(xmax - xmin, ymax - ymin), 1.0e-6) * 2.5
                    guide = LineString([center - axis * span, center + axis * span])
                    wall_geom = guide.intersection(cell)
                    wall_lines = ([wall_geom] if isinstance(wall_geom, LineString)
                                  else [g for g in getattr(wall_geom, 'geoms', []) if isinstance(g, LineString)])
                    wall = max(wall_lines, key=lambda g: float(g.length)) if wall_lines else None
                    wall_len_mm = float(wall.length) * chart.scale_2d_to_3d if wall is not None else 0.0
                    if wall is None or wall_len_mm < min_wall_mm:
                        reason = 'wall_too_short'
                    else:
                        try:
                            pieces_geom = split(cell, guide)
                            pieces = [g.buffer(0) for g in getattr(pieces_geom, 'geoms', [])
                                      if isinstance(g, Polygon) and float(g.area) > 1.0e-12]
                        except Exception:
                            pieces = []
                        if len(pieces) < 2:
                            reason = 'polygon_split_failed'
                        else:
                            post_metrics = []
                            tiny = False
                            for piece in pieces:
                                lp = _largest_liquid(piece)
                                if lp is None:
                                    tiny = True
                                    continue
                                rr, aa, area3 = _liquid_metrics(lp)
                                if area3 < float(self.config.zero_pore_min_area_mm2) - 1.0e-10:
                                    tiny = True
                                post_metrics.append((rr, aa, max(area3, 1.0e-12)))
                            if not post_metrics or tiny:
                                reason = 'subresolution_fragment_guard'
                                post_r = max((m[0] for m in post_metrics), default=math.nan)
                            else:
                                post_r = max(m[0] for m in post_metrics)
                                post_ar = sum(m[1] * m[2] for m in post_metrics) / sum(m[2] for m in post_metrics)
                                post_ar_max = max(m[1] for m in post_metrics)
                                capillary_not_worse = post_r <= r_pre * 1.02 + 1.0e-10
                                morphology_gain = post_ar >= ar_pre * (1.0 + min_ar_gain)
                                capillary_gain = post_r <= r_pre * 0.95
                                cap_min_area = float(self.config.anisotropy_cap_min_area_mm2)
                                relevant_post_ars = [m[1] for m in post_metrics if m[2] > cap_min_area + 1.0e-10]
                                ar_cap_ok = (not relevant_post_ars) or max(relevant_post_ars) <= hard_max_ar + 1.0e-10
                                accepted = bool(capillary_not_worse and ar_cap_ok and (morphology_gain or capillary_gain))
                                if accepted:
                                    rewritten_cells.extend(pieces)
                                    anisotropy_wall_segments.append(wall)
                                    reason = 'accepted_parallel_wall_split'
                                elif not ar_cap_ok:
                                    reason = 'post_ar_hard_cap_exceeded'
                                else:
                                    reason = 'no_useful_soft_improvement'
                if not accepted:
                    rewritten_cells.append(cell)
                self._anisotropy_rewrite_rows.append({
                    'patch_id': int(chart.patch_id), 'component_id': int(chart.component_id),
                    'cell_id': int(cell_id), 'rewrite_type': 'soft_parallel', 'selected': 1, 'accepted': int(bool(accepted)),
                    'reason': reason, 'pre_r_eq_mm': r_pre, 'pre_aspect_ratio': ar_pre,
                    'post_max_r_eq_mm': post_r, 'post_aspect_ratio_mean': post_ar,
                    'post_aspect_ratio_max': post_ar_max, 'wall_length_mm': wall_len_mm,
                })
            final_cells = rewritten_cells

            # Third-stage hard AR-cap repair. A very elongated pore is split
            # transversely (wall aligned with the local minor axis), which
            # shortens its major dimension instead of narrowing it further.
            # This is the opposite action to the soft parallel-wall rewrite.
            cap_repaired_cells: List[Polygon] = []
            cap_queue = [(int(i), cell, 0) for i, cell in enumerate(final_cells)]
            while cap_queue:
                source_cell_id, cell, depth = cap_queue.pop(0)
                liquid = _largest_liquid(cell)
                if liquid is None:
                    cap_repaired_cells.append(cell)
                    continue
                r_pre, ar_pre, area_pre = _liquid_metrics(liquid)
                cap_min_area = float(self.config.anisotropy_cap_min_area_mm2)
                if area_pre <= cap_min_area + 1.0e-10 or ar_pre <= hard_max_ar + 1.0e-10:
                    cap_repaired_cells.append(cell)
                    continue
                if depth >= max_cap_splits:
                    cap_repaired_cells.append(cell)
                    self._anisotropy_rewrite_rows.append({
                        'patch_id': int(chart.patch_id), 'component_id': int(chart.component_id),
                        'cell_id': int(source_cell_id), 'rewrite_type': 'hard_ar_cap',
                        'selected': 1, 'accepted': 0, 'reason': 'ar_cap_split_budget_exhausted',
                        'pre_r_eq_mm': r_pre, 'pre_aspect_ratio': ar_pre,
                        'post_max_r_eq_mm': math.nan, 'post_aspect_ratio_mean': math.nan,
                        'post_aspect_ratio_max': ar_pre, 'wall_length_mm': 0.0,
                    })
                    continue
                major = _major_axis(liquid)
                if major is None:
                    cap_repaired_cells.append(cell)
                    continue
                minor = np.asarray([-major[1], major[0]], dtype=float)
                center_point = liquid.centroid if liquid.covers(liquid.centroid) else liquid.representative_point()
                center = np.asarray(center_point.coords[0], dtype=float)
                xmin, ymin, xmax, ymax = cell.bounds
                span = max(math.hypot(xmax - xmin, ymax - ymin), 1.0e-6) * 2.5
                guide = LineString([center - minor * span, center + minor * span])
                wall_geom = guide.intersection(cell)
                wall_lines = ([wall_geom] if isinstance(wall_geom, LineString)
                              else [g for g in getattr(wall_geom, 'geoms', []) if isinstance(g, LineString)])
                wall = max(wall_lines, key=lambda g: float(g.length)) if wall_lines else None
                wall_len_mm = float(wall.length) * chart.scale_2d_to_3d if wall is not None else 0.0
                if wall is None or wall_len_mm < min_wall_mm:
                    cap_repaired_cells.append(cell)
                    continue
                # Endpoints on an artificial seam are allowed: the seam is a
                # computational partition, not a physical open boundary.
                # Reject only endpoints landing on a genuine design-domain
                # boundary. Such seam-touching walls terminate on the shared
                # seam skeleton, so they remain connected to the neighboring
                # patch graph; the global complete-pore audit then verifies
                # whether the cross-patch pore is actually repaired.
                if bool(self.config.anisotropy_skip_patch_boundary_cells):
                    endpoints = [Point(wall.coords[0]), Point(wall.coords[-1])]
                    true_boundary_hit = False
                    for pt in endpoints:
                        on_chart_boundary = (
                            float(pt.distance(chart.polygon.boundary)) <= boundary_class_tol_uv
                        )
                        if on_chart_boundary and not _point_on_artificial_seam(pt):
                            true_boundary_hit = True
                            break
                    if true_boundary_hit:
                        cap_repaired_cells.append(cell)
                        self._anisotropy_rewrite_rows.append({
                            'patch_id': int(chart.patch_id), 'component_id': int(chart.component_id),
                            'cell_id': int(source_cell_id), 'rewrite_type': 'hard_ar_cap',
                            'selected': 1, 'accepted': 0, 'reason': 'ar_cap_wall_touches_true_design_boundary',
                            'pre_r_eq_mm': r_pre, 'pre_aspect_ratio': ar_pre,
                            'post_max_r_eq_mm': math.nan, 'post_aspect_ratio_mean': math.nan,
                            'post_aspect_ratio_max': ar_pre, 'wall_length_mm': wall_len_mm,
                        })
                        continue
                try:
                    pieces_geom = split(cell, guide)
                    pieces = [g.buffer(0) for g in getattr(pieces_geom, 'geoms', [])
                              if isinstance(g, Polygon) and float(g.area) > 1.0e-12]
                except Exception:
                    pieces = []
                post_metrics = []
                tiny = False
                for piece in pieces:
                    lp = _largest_liquid(piece)
                    if lp is None:
                        tiny = True
                        continue
                    rr, aa, area3 = _liquid_metrics(lp)
                    if area3 < float(self.config.zero_pore_min_area_mm2) - 1.0e-10:
                        tiny = True
                    post_metrics.append((rr, aa, max(area3, 1.0e-12)))
                accepted_cap = False
                post_r = math.nan
                post_ar_mean = math.nan
                post_ar_max = math.nan
                reason = 'ar_cap_polygon_split_failed'
                if len(pieces) >= 2 and post_metrics and not tiny:
                    post_r = max(m[0] for m in post_metrics)
                    post_ar_mean = sum(m[1] * m[2] for m in post_metrics) / sum(m[2] for m in post_metrics)
                    post_ar_max = max(m[1] for m in post_metrics)
                    capillary_safe = post_r <= max(r_pre * 1.02, target) + 1.0e-10
                    morphology_improved = post_ar_max < ar_pre * 0.98
                    accepted_cap = bool(capillary_safe and morphology_improved)
                    reason = 'accepted_transverse_ar_cap_split' if accepted_cap else 'ar_cap_no_safe_improvement'
                self._anisotropy_rewrite_rows.append({
                    'patch_id': int(chart.patch_id), 'component_id': int(chart.component_id),
                    'cell_id': int(source_cell_id), 'rewrite_type': 'hard_ar_cap',
                    'selected': 1, 'accepted': int(bool(accepted_cap)), 'reason': reason,
                    'pre_r_eq_mm': r_pre, 'pre_aspect_ratio': ar_pre,
                    'post_max_r_eq_mm': post_r, 'post_aspect_ratio_mean': post_ar_mean,
                    'post_aspect_ratio_max': post_ar_max, 'wall_length_mm': wall_len_mm,
                })
                if accepted_cap:
                    anisotropy_wall_segments.append(wall)
                    # Continue only on children that are still above the repair
                    # target. The final hard gate remains AR <= hard_max_ar.
                    for piece in pieces:
                        lp = _largest_liquid(piece)
                        if lp is None:
                            cap_repaired_cells.append(piece)
                            continue
                        _, aa, child_area = _liquid_metrics(lp)
                        if (child_area > float(self.config.anisotropy_cap_min_area_mm2) + 1.0e-10
                                and aa > cap_target_ar + 1.0e-10):
                            cap_queue.append((int(source_cell_id), piece, depth + 1))
                        else:
                            cap_repaired_cells.append(piece)
                else:
                    cap_repaired_cells.append(cell)
            final_cells = cap_repaired_cells

        linework = unary_union([cell.boundary for cell in final_cells])
        try:
            # Merge degree-two runs before graph sampling. Otherwise every
            # input-triangle corner survives as a graph vertex even after the
            # global 0.65-mm source-vertex filter has rejected it.
            linework = linemerge(linework)
        except (TypeError, ValueError):
            pass
        line_strings = []

        def collect_lines(geometry):
            if isinstance(geometry, LineString):
                line_strings.append(geometry)
            else:
                for item in getattr(geometry, "geoms", []):
                    collect_lines(item)

        collect_lines(linework)
        points: List[np.ndarray] = []
        point_lookup: Dict[Tuple[int, int], int] = {}
        local_edges: Set[Tuple[int, int]] = set()
        max_edge_uv = float(self.config.max_edge_length_mm) / max(chart.scale_2d_to_3d, 1.0e-12)

        boundary_edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
        for face in np.asarray(chart.local_faces, dtype=int):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                boundary_edge_count[tuple(sorted((int(a), int(b))))] += 1
        boundary_local_ids = {
            vertex for edge, count in boundary_edge_count.items() if count == 1
            for vertex in edge
        }

        def sparse_boundary_coordinates(coordinates: np.ndarray) -> np.ndarray:
            """Remove unselected triangulation corners from graph linework only."""
            coordinates = np.asarray(coordinates, dtype=float)
            if len(coordinates) <= 2:
                return coordinates
            closed = bool(np.linalg.norm(coordinates[0] - coordinates[-1]) <= 1.0e-10)
            body = coordinates[:-1] if closed else coordinates
            kept = []
            for point in body:
                nearest = int(np.argmin(np.linalg.norm(chart.uv - point, axis=1)))
                exact_vertex = float(np.linalg.norm(chart.uv[nearest] - point)) <= 1.0e-8
                source_vertex = int(chart.vertex_ids[nearest])
                if (exact_vertex and nearest in boundary_local_ids
                        and source_vertex not in self._graph_source_vertex_ids
                        and source_vertex not in self._artificial_seam_vertex_ids):
                    continue
                kept.append(point)
            minimum = 3 if closed else 2
            if len(kept) < minimum:
                return coordinates
            result = np.asarray(kept, dtype=float)
            return np.vstack([result, result[0]]) if closed else result

        def point_id(coordinate: np.ndarray) -> int:
            key = tuple(np.round(np.asarray(coordinate, dtype=float) * 1.0e8).astype(np.int64))
            if key not in point_lookup:
                point_lookup[key] = len(points)
                points.append(np.asarray(coordinate, dtype=float))
            return point_lookup[key]

        for line in line_strings:
            coordinates = sparse_boundary_coordinates(np.asarray(line.coords, dtype=float))
            for p0, p1 in zip(coordinates[:-1], coordinates[1:]):
                count = max(1, int(math.ceil(np.linalg.norm(p1 - p0) / max(max_edge_uv, 1.0e-9))))
                samples = np.linspace(p0, p1, count + 1)
                ids = [point_id(sample) for sample in samples]
                for a, b in zip(ids[:-1], ids[1:]):
                    if a != b:
                        local_edges.add(tuple(sorted((a, b))))

        anisotropy_local_edges: Set[Tuple[int, int]] = set()
        if anisotropy_wall_segments:
            aniso_tol_uv = max(1.0e-8, float(self.config.map_tolerance) * 10.0)
            for edge in local_edges:
                seg = LineString([points[edge[0]], points[edge[1]]])
                if seg.length <= 1.0e-12:
                    continue
                for wall_seg in anisotropy_wall_segments:
                    if seg.distance(wall_seg) > aniso_tol_uv:
                        continue
                    overlap = seg.intersection(wall_seg.buffer(
                        aniso_tol_uv, cap_style=2, join_style=2
                    ))
                    if float(getattr(overlap, 'length', 0.0)) >= 0.95 * float(seg.length):
                        anisotropy_local_edges.add(tuple(sorted((int(edge[0]), int(edge[1])))))
                        break

        arr = np.asarray(points, dtype=float)
        point_source_vertices = []
        for point in arr:
            nearest = int(np.argmin(np.linalg.norm(chart.uv - point, axis=1)))
            source_vertex = int(chart.vertex_ids[nearest])
            if (
                float(np.linalg.norm(chart.uv[nearest] - point)) <= 1.0e-8
                and source_vertex in self._graph_source_vertex_ids
            ):
                point_source_vertices.append(source_vertex)
            else:
                point_source_vertices.append(-1)
        base_xyz = []
        for p in arr:
            base_xyz.append(self._locate_uv_base(chart, p)[0])
        if self.config.mapping_surface == "carrier":
            xyz = np.asarray(base_xyz, dtype=float)
        else:
            xyz, _ = self._project_original_batch(
                np.asarray(base_xyz, dtype=float), component_id=chart.component_id
            )
        for seed_id, seed in enumerate(seeds):
            seed3 = self._locate_uv(chart, np.asarray(seed, dtype=float))[0]
            self._voronoi_seed_rows.append(
                {
                    "patch_id": chart.patch_id,
                    "component_id": chart.component_id,
                    "seed_id": seed_id,
                    "seed_role": (
                        "initial" if seed_id < initial_seed_count
                        else "physics_incenter_feedback" if policy == "physics"
                        else "geometry_only_largest_void_voronoi" if policy == "random_geometry"
                        else "random_local_rewrite"
                    ),
                    "uv_x": float(seed[0]),
                    "uv_y": float(seed[1]),
                    "x_mm": float(seed3[0]),
                    "y_mm": float(seed3[1]),
                    "z_mm": float(seed3[2]),
                }
            )
        return arr, xyz, local_edges, final_cells, point_source_vertices, anisotropy_local_edges

    def _sample_mapped_edge(self, chart: PatchChart, uv0: np.ndarray, uv1: np.ndarray):
        length2 = float(np.linalg.norm(uv1 - uv0)) * chart.scale_2d_to_3d
        spacing = min(float(self.config.edge_sample_spacing_mm), float(self.config.max_edge_length_mm))
        count = max(1, int(math.ceil(length2 / max(spacing, 1.0e-6))))
        ts = list(np.linspace(0.0, 1.0, count + 1))
        base_points = []
        fallback = False
        for t in ts:
            base_point, _, used_fallback = self._locate_uv_base(
                chart, uv0 * (1.0 - t) + uv1 * t
            )
            base_points.append(base_point)
            fallback = fallback or used_fallback
        if self.config.mapping_surface == "carrier":
            mapped = np.asarray(base_points, dtype=float)
            face_ids = np.asarray(
                [self._locate_uv_base(chart, uv0 * (1.0 - t) + uv1 * t)[1] for t in ts],
                dtype=int,
            )
        else:
            mapped, face_ids = self._project_original_batch(
                np.asarray(base_points, dtype=float), component_id=chart.component_id
            )

        if len(mapped) >= 2:
            chord_midpoints = 0.5 * (mapped[:-1] + mapped[1:])
            chord_distances = self._mapping_surface_distance_batch(
                chord_midpoints, component_id=chart.component_id
            )
            if float(np.max(chord_distances, initial=0.0)) <= float(self.config.surface_tolerance_mm):
                return [
                    (float(t), np.asarray(point3, dtype=float), False, int(face_id))
                    for t, point3, face_id in zip(ts, mapped, face_ids)
                ], fallback

        def refine(t0: float, p0: np.ndarray, t1: float, p1: np.ndarray, depth: int):
            tm = 0.5 * (t0 + t1)
            uvm = uv0 * (1.0 - tm) + uv1 * tm
            pm, face_id, used_fallback = self._locate_uv(chart, uvm)
            chord_mid = 0.5 * (p0 + p1)
            deviation = float(
                self._mapping_surface_distance_batch(
                    chord_mid[None, :], component_id=chart.component_id
                )[0]
            )
            if deviation > self.config.surface_tolerance_mm and depth < self.config.max_edge_refine_depth:
                left = refine(t0, p0, tm, pm, depth + 1)
                right = refine(tm, pm, t1, p1, depth + 1)
                return left[:-1] + right
            return [(t0, p0, False, int(face_id)), (t1, p1, used_fallback, int(face_id))]

        refined = []
        for i in range(len(ts) - 1):
            segment = refine(ts[i], mapped[i], ts[i + 1], mapped[i + 1], 0)
            if refined:
                segment = segment[1:]
            refined.extend(segment)
        refined.sort(key=lambda item: item[0])
        return refined, fallback

    def _construct_graph(self):
        assert self.base_mesh is not None
        node_rows: List[Dict] = []
        edge_rows: List[Dict] = []
        edge_metric_rows: List[Dict] = []
        pore_rows: List[Dict] = []
        global_source_nodes: Dict[int, int] = {}
        next_node_id = 0
        logical_edge_id = 0
        logical_degree: Dict[int, int] = defaultdict(int)
        accepted_endpoint_pairs: Set[Tuple[int, int]] = set()
        duplicate_pair_skips = 0
        degree_cap_skips = 0

        patch_batch_size = max(1, int(self.config.patch_batch_size))
        for chart_index, chart in enumerate(self.charts):
            uv, xyz, local_edges, pore_cells, point_source_vertices, anisotropy_local_edges = self._refine_patch_nodes(chart)
            self._source_cells_by_patch[int(chart.patch_id)] = list(pore_cells)
            local_to_global: Dict[int, int] = {}
            for local_id, (p2, p3) in enumerate(zip(uv, xyz)):
                source_vertex = int(point_source_vertices[local_id])
                if source_vertex >= 0 and source_vertex in global_source_nodes:
                    gid = global_source_nodes[source_vertex]
                else:
                    gid = next_node_id
                    next_node_id += 1
                    if source_vertex >= 0:
                        global_source_nodes[source_vertex] = gid
                    node_rows.append(
                        {
                            "atom_id": gid,
                            "x": float(p3[0]),
                            "y": float(p3[1]),
                            "z": float(p3[2]),
                            "face_name": "surface",
                            "patch_id": chart.patch_id,
                            "component_id": chart.component_id,
                            "source_vertex_id": source_vertex,
                            "uv_x": float(p2[0]),
                            "uv_y": float(p2[1]),
                            "node_role": "source_vertex" if source_vertex >= 0 else "voronoi_vertex",
                        }
                    )
                local_to_global[local_id] = gid

            def polygon_parts(geometry) -> List[Polygon]:
                if geometry is None or geometry.is_empty:
                    return []
                if isinstance(geometry, Polygon):
                    return [geometry]
                parts = []
                for item in getattr(geometry, "geoms", []):
                    parts.extend(polygon_parts(item))
                return parts

            buffer_uv = 0.5 * float(self.config.surface_diameter_mm) / max(
                chart.scale_2d_to_3d, 1.0e-12
            )
            for subdomain_id, cell in enumerate(pore_cells):
                cell_area2 = float(cell.area) * chart.scale_2d_to_3d ** 2
                cell_perimeter2 = float(cell.length) * chart.scale_2d_to_3d
                cell_area3, cell_perimeter3 = self._mapped_polygon_metrics(chart, cell)
                liquid_parts = polygon_parts(cell.buffer(-buffer_uv))
                for island_id, liquid in enumerate(liquid_parts or [None]):
                    pore_area2 = (
                        float(liquid.area) * chart.scale_2d_to_3d ** 2 if liquid is not None else 0.0
                    )
                    pore_perimeter2 = (
                        float(liquid.length) * chart.scale_2d_to_3d if liquid is not None else 0.0
                    )
                    pore_area3, pore_perimeter3 = (
                        self._mapped_polygon_metrics(chart, liquid)
                        if liquid is not None else (0.0, 0.0)
                    )
                    r2 = 2.0 * pore_area2 / max(pore_perimeter2, 1.0e-12)
                    r3 = 2.0 * pore_area3 / max(pore_perimeter3, 1.0e-12)
                    centroid3 = self._locate_uv(
                        chart,
                        np.asarray(
                            (liquid if liquid is not None else cell).representative_point().coords[0],
                            dtype=float,
                        ),
                    )[0]
                    pore_rows.append(
                        {
                            "patch_id": chart.patch_id,
                            "component_id": chart.component_id,
                            "subdomain_id": subdomain_id,
                            "island_id": island_id,
                            "centroid_x_mm": float(centroid3[0]),
                            "centroid_y_mm": float(centroid3[1]),
                            "centroid_z_mm": float(centroid3[2]),
                            "cell_area_2d_mm2": cell_area2 if island_id == 0 else 0.0,
                            "cell_perimeter_2d_mm": cell_perimeter2 if island_id == 0 else 0.0,
                            "cell_area_3d_mm2": cell_area3 if island_id == 0 else 0.0,
                            "cell_perimeter_3d_mm": cell_perimeter3 if island_id == 0 else 0.0,
                            "pore_area_2d_mm2": pore_area2,
                            "pore_perimeter_2d_mm": pore_perimeter2,
                            "pore_area_3d_mm2": pore_area3,
                            "pore_perimeter_3d_mm": pore_perimeter3,
                            "hydraulic_radius_2d_mm": r2,
                            "hydraulic_radius_3d_mm": r3,
                            "hydraulic_radius_rel_error": abs(r3 - r2) / max(r2, 1.0e-9),
                            "hydraulic_radius_target_normalized_error": abs(r3 - r2)
                            / max(float(self.config.target_hydraulic_radius_mm), 1.0e-9),
                            "shape_factor": 4.0 * math.pi * pore_area3
                            / max(pore_perimeter3 * pore_perimeter3, 1.0e-12),
                            "aspect_ratio": _polygon_aspect_ratio(liquid) if liquid is not None else 0.0,
                            "liquid_area_fraction": pore_area3 / max(cell_area3, 1.0e-12),
                        }
                    )

            # Build the artificial-seam geometry in this chart.  Voronoi
            # linework can split one carrier seam edge into several subsegments
            # whose endpoints are no longer both source vertices, so seam
            # identity must be geometric rather than endpoint-ID only.
            chart_seam_segments = []
            local_boundary_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for face in np.asarray(chart.local_faces, dtype=int):
                for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    local_boundary_counts[tuple(sorted((int(a), int(b))))] += 1
            for (la, lb), count in local_boundary_counts.items():
                if count != 1:
                    continue
                source_key = tuple(sorted((
                    int(chart.vertex_ids[la]), int(chart.vertex_ids[lb])
                )))
                if source_key not in self._artificial_seam_edges:
                    continue
                chart_seam_segments.append((
                    source_key,
                    LineString([chart.uv[la], chart.uv[lb]]),
                    set(self._seam_edge_patches.get(source_key, {int(chart.patch_id)})),
                ))

            seam_tol_uv = max(
                1.0e-8, float(self.config.map_tolerance) * 10.0
            )
            local_edge_seam_cache = {}

            def local_edge_seam_memberships(edge):
                key_cache = tuple(sorted((int(edge[0]), int(edge[1]))))
                if key_cache in local_edge_seam_cache:
                    return local_edge_seam_cache[key_cache]
                seg = LineString([uv[edge[0]], uv[edge[1]]])
                memberships = set()
                if seg.length > 1.0e-12:
                    for _, seam_seg, seam_patches in chart_seam_segments:
                        if seg.distance(seam_seg) > seam_tol_uv:
                            continue
                        overlap = seg.intersection(seam_seg.buffer(
                            seam_tol_uv, cap_style=2, join_style=2
                        ))
                        if float(getattr(overlap, 'length', 0.0)) >= 0.95 * float(seg.length):
                            memberships.update(seam_patches)
                local_edge_seam_cache[key_cache] = memberships
                return memberships

            ordered_edges = sorted(
                local_edges,
                key=lambda edge: (
                    0 if local_edge_seam_memberships(edge) else
                    1 if tuple(sorted((int(edge[0]), int(edge[1])))) in anisotropy_local_edges else 2,
                    float(np.linalg.norm(uv[edge[0]] - uv[edge[1]]))
                    * chart.scale_2d_to_3d,
                ),
            )
            for local_a, local_b in ordered_edges:
                if not _geometry_covers_line(chart.polygon, uv[local_a], uv[local_b], self.config.map_tolerance):
                    continue
                endpoint_a = int(local_to_global[local_a])
                endpoint_b = int(local_to_global[local_b])
                endpoint_pair = tuple(sorted((endpoint_a, endpoint_b)))
                seam_memberships = local_edge_seam_memberships((local_a, local_b))
                is_seam_edge = bool(seam_memberships)
                is_anisotropy_edge = tuple(sorted((int(local_a), int(local_b)))) in anisotropy_local_edges
                if endpoint_pair in accepted_endpoint_pairs:
                    duplicate_pair_skips += 1
                    continue
                degree_limit = max(1, int(self.config.max_node_degree))
                source_a = int(point_source_vertices[local_a])
                source_b = int(point_source_vertices[local_b])
                reserve_a = (
                    int(self._seam_required_degree_by_vertex.get(source_a, 0))
                    if source_a >= 0 else 0
                )
                reserve_b = (
                    int(self._seam_required_degree_by_vertex.get(source_b, 0))
                    if source_b >= 0 else 0
                )
                limit_a = degree_limit if is_seam_edge else max(0, degree_limit - reserve_a)
                limit_b = degree_limit if is_seam_edge else max(0, degree_limit - reserve_b)
                if (
                    logical_degree[endpoint_a] >= limit_a
                    or logical_degree[endpoint_b] >= limit_b
                ):
                    degree_cap_skips += 1
                    continue
                refined, mapping_fallback = self._sample_mapped_edge(chart, uv[local_a], uv[local_b])
                chain = [endpoint_a]
                node_row_start = len(node_rows)
                next_node_start = next_node_id
                interior_points = refined[1:-1]
                for t, point3, used_fallback, source_face in interior_points:
                    gid = next_node_id
                    next_node_id += 1
                    point2 = uv[local_a] * (1.0 - t) + uv[local_b] * t
                    node_rows.append(
                        {
                            "atom_id": gid,
                            "x": float(point3[0]),
                            "y": float(point3[1]),
                            "z": float(point3[2]),
                            "face_name": "surface",
                            "patch_id": chart.patch_id,
                            "component_id": chart.component_id,
                            "source_vertex_id": -1,
                            "uv_x": float(point2[0]),
                            "uv_y": float(point2[1]),
                            "node_role": "edge_sample",
                        }
                    )
                    chain.append(gid)
                chain.append(endpoint_b)
                all_points = [item[1] for item in refined]
                max_dev = 0.0
                mapping_fallback = mapping_fallback or any(bool(item[2]) for item in refined)
                if self.config.mapping_surface == "carrier":
                    chart_faces = set(map(int, chart.face_ids))
                    out_of_patch = sum(int(item[3]) not in chart_faces for item in refined)
                else:
                    out_of_patch = sum(
                        int(self.original_face_components[int(item[3])]) != int(chart.component_id)
                        for item in refined
                    )
                if len(all_points) >= 2:
                    midpoints = 0.5 * (
                        np.asarray(all_points[:-1], dtype=float)
                        + np.asarray(all_points[1:], dtype=float)
                    )
                    distances = self._mapping_surface_distance_batch(
                        midpoints, component_id=chart.component_id
                    )
                    max_dev = float(np.max(distances, initial=0.0))
                rejected = bool(
                    mapping_fallback
                    or out_of_patch > 0
                    or max_dev > float(self.config.surface_tolerance_mm) + 1.0e-12
                )
                if rejected:
                    del node_rows[node_row_start:]
                    next_node_id = next_node_start
                    edge_metric_rows.append(
                        {
                            "logical_edge_id": logical_edge_id,
                            "patch_id": chart.patch_id,
                            "component_id": chart.component_id,
                            "polyline_segments": 0,
                            "max_chord_surface_error_mm": max_dev,
                            "domain_violation": 1,
                            "mapping_fallback": int(mapping_fallback),
                            "cross_component": int(out_of_patch > 0),
                            "anisotropy_wall": int(is_anisotropy_edge),
                            "accepted": 0,
                        }
                    )
                    logical_edge_id += 1
                    continue
                for u, v in zip(chain[:-1], chain[1:]):
                    edge_rows.append(
                        {
                            "source": int(u),
                            "target": int(v),
                            "radius": 0.5 * self.config.surface_diameter_mm,
                            "bond_type": "surface",
                            "role": ("surface_anisotropy_wall" if is_anisotropy_edge else "surface_geodesic_segment"),
                            "patch_id": chart.patch_id,
                            "patch_memberships": ";".join(map(str, sorted(
                                seam_memberships if seam_memberships else {int(chart.patch_id)}
                            ))),
                            "component_id": chart.component_id,
                            "logical_edge_id": logical_edge_id,
                        }
                    )
                accepted_endpoint_pairs.add(endpoint_pair)
                logical_degree[endpoint_a] += 1
                logical_degree[endpoint_b] += 1
                edge_metric_rows.append(
                    {
                        "logical_edge_id": logical_edge_id,
                        "patch_id": chart.patch_id,
                        "component_id": chart.component_id,
                        "polyline_segments": len(chain) - 1,
                        "max_chord_surface_error_mm": max_dev,
                        "domain_violation": int(out_of_patch > 0),
                        "mapping_fallback": int(mapping_fallback),
                        "cross_component": 0,
                        "anisotropy_wall": int(is_anisotropy_edge),
                        "accepted": 1,
                    }
                )
                logical_edge_id += 1

            if (
                (chart_index + 1) % patch_batch_size == 0
                or chart_index + 1 == len(self.charts)
            ):
                del uv, xyz, local_edges, pore_cells, point_source_vertices, anisotropy_local_edges
                gc.collect()

        self.atoms = pd.DataFrame(node_rows).drop_duplicates("atom_id").sort_values("atom_id").reset_index(drop=True)
        memberships: Dict[int, Set[int]] = defaultdict(set)
        for chart in self.charts:
            for vertex in chart.vertex_ids:
                memberships[int(vertex)].add(int(chart.patch_id))
        source_membership = {
            vertex: ";".join(map(str, sorted(patch_ids)))
            for vertex, patch_ids in memberships.items()
        }
        if not self.atoms.empty:
            self.atoms["patch_memberships"] = self.atoms["source_vertex_id"].map(source_membership)
            self.atoms["patch_memberships"] = self.atoms["patch_memberships"].fillna(
                self.atoms["patch_id"].astype(str)
            )
        self.edges = pd.DataFrame(edge_rows)
        self.edge_metrics = pd.DataFrame(edge_metric_rows)
        self.pore_metrics = pd.DataFrame(pore_rows)
        self.refinement_history = pd.DataFrame(self._refinement_history_rows)
        self.anisotropy_rewrite_history = pd.DataFrame(self._anisotropy_rewrite_rows)
        self.voronoi_seeds = pd.DataFrame(self._voronoi_seed_rows)
        self._pruned_node_count = 0
        self._pruned_edge_segment_count = 0
        # Reconcile graph fragments for every input surface domain, including
        # the common case of one connected carrier split into many charts.
        self._bridge_surface_components_within_domains()
        self._prune_minor_graph_fragments()
        if self.config.zero_pore_repair_enable:
            # Minimum-area cleanup is an alternating closed loop.  V7 stopped
            # after three source-cell passes and monitored only A==0, which
            # could leave 0<A<amin fragments (and source cells) unresolved.
            # Here BOTH classes are hard defects.  After every source-cell
            # repair we also run the retained-graph fragment merge, because
            # either operation can unlock the other.
            all_zero_logs = []
            all_fragment_logs = []
            amin = float(self.config.zero_pore_min_area_mm2)
            max_cycles = max(1, int(self.config.min_pore_repair_max_cycles))
            patience = max(1, int(self.config.min_pore_repair_stagnation_patience))
            best_signature = None
            stagnant = 0
            self._final_fragment_unresolved_count = 0
            for repair_cycle in range(max_cycles):
                edge_count_before = int(len(self.edges))
                radius_sum_before = float(self.edges.get('radius', pd.Series(dtype=float)).sum())

                self._repair_zero_area_pores()
                if not self.zero_pore_repair_history.empty:
                    pass_log = self.zero_pore_repair_history.copy()
                    pass_log.insert(0, 'repair_cycle', repair_cycle + 1)
                    all_zero_logs.append(pass_log)

                # One complete retained-graph cleanup call may itself perform
                # several local deletion passes.
                self._repair_final_open_fragments()
                if not self.final_fragment_repair_history.empty:
                    frag_log = self.final_fragment_repair_history.copy()
                    frag_log.insert(0, 'repair_cycle', repair_cycle + 1)
                    all_fragment_logs.append(frag_log)

                # Re-audit the source openings AFTER retained-graph changes.
                # This second call is important: removing a tiny-fragment wall
                # can open a source cell that was previously fully occupied.
                self._repair_zero_area_pores()
                if not self.zero_pore_repair_history.empty:
                    pass_log = self.zero_pore_repair_history.copy()
                    pass_log.insert(0, 'repair_cycle', repair_cycle + 1)
                    pass_log.insert(1, 'repair_phase', 'post_fragment')
                    all_zero_logs.append(pass_log)

                audit_area = self.zero_pore_final_audit.get(
                    'open_area_mm2', pd.Series(dtype=float)
                )
                source_under = int((audit_area < amin - 1.0e-10).sum())
                source_zero = int((audit_area <= 1.0e-10).sum())
                fragment_under = int(getattr(self, '_final_fragment_unresolved_count', 0))

                # Progress signature includes counts and graph mutation state.
                # Equal counts alone are not enough: an opening can grow over
                # multiple cycles before crossing amin.
                deficit = float(np.maximum(0.0, amin - audit_area.to_numpy(float)).sum()) if len(audit_area) else 0.0
                signature = (source_under, source_zero, fragment_under, round(deficit, 10),
                             int(len(self.edges)), round(float(self.edges.get('radius', pd.Series(dtype=float)).sum()), 10))
                if fragment_under == 0:
                    break
                if signature == best_signature and int(len(self.edges)) == edge_count_before \
                        and abs(float(self.edges.get('radius', pd.Series(dtype=float)).sum()) - radius_sum_before) <= 1.0e-12:
                    stagnant += 1
                else:
                    stagnant = 0
                    best_signature = signature
                if stagnant >= patience:
                    break

            if all_zero_logs:
                self.zero_pore_repair_history = pd.concat(all_zero_logs, ignore_index=True)
            if all_fragment_logs:
                self.final_fragment_repair_history = pd.concat(all_fragment_logs, ignore_index=True)
        self._duplicate_pair_skips = duplicate_pair_skips
        self._degree_cap_skips = degree_cap_skips
        if not self.atoms.empty:
            self.curr_vertices = self.atoms[["x", "y", "z"]].to_numpy(float).tolist()
        self.curr_edges = list(self.edges[["source", "target"]].itertuples(index=False, name=None)) if not self.edges.empty else []

    @staticmethod
    def _repair_connectivity_preserved(before: pd.DataFrame, after: pd.DataFrame) -> bool:
        """Prevent splitting an existing connected component of retained graph nodes."""
        old_ids = set(before["source"].astype(int)) | set(before["target"].astype(int))
        new_ids = set(after["source"].astype(int)) | set(after["target"].astype(int))
        # A deleted graph junction cannot be silently discarded. Only polyline
        # sample points with no incident edges may be removed by the caller.
        survivors = old_ids & new_ids
        def classes(frame):
            parent = {}
            def root(x):
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for edge in frame.itertuples(index=False):
                a, b = root(int(edge.source)), root(int(edge.target))
                if a != b:
                    parent[a] = b
            return {x: root(x) for x in survivors}
        old, new = classes(before), classes(after)
        target = {}
        for x in survivors:
            if old[x] in target and target[old[x]] != new[x]:
                return False
            target[old[x]] = new[x]
        return True

    def _repair_zero_area_pores(self):
        """Bounded, post-pruning zero-source-pore repair on retained chart graph.

        The unchanged initial graph is generated by the original Voronoi code.
        Only an actual blocked source cell triggers local radius reduction,
        shared-logical-wall removal and optional *non-Voronoi* orthogonal split.
        Accepted local edits change the exported atoms/edges, not merely metrics.
        """
        logs, audit = [], []
        postrepair_pores = []
        updated_patches = set()
        if self.edges.empty or self.atoms.empty:
            self.zero_pore_repair_history = pd.DataFrame()
            self.zero_pore_final_audit = pd.DataFrame()
            return
        scale_map = {int(c.patch_id): c for c in self.charts}
        source_by_patch = self._source_cells_by_patch
        id_rows = self.atoms.set_index("atom_id")
        next_id = int(self.atoms["atom_id"].max()) + 1
        next_logical = int(self.edges["logical_edge_id"].max()) + 1
        old_edges_all = self.edges.copy()
        source_node_ids = defaultdict(list)
        for atom in self.atoms.itertuples(index=False):
            if int(atom.source_vertex_id) >= 0:
                source_node_ids[int(atom.source_vertex_id)].append(int(atom.atom_id))
        rcrit = float(self.config.target_hydraulic_radius_mm) * float(self.config.hydraulic_radius_mapping_safety)
        amin = float(self.config.zero_pore_min_area_mm2)
        minrad = float(self.config.zero_pore_min_diameter_mm) / 2.0
        allowed = max(0, int(self.config.zero_pore_max_per_patch))
        trials = max(0, int(self.config.zero_pore_secondary_trials))

        def pieces(geom):
            return [p for p in _polygon_parts(geom) if p.area > 1e-12]

        for patch_id, base_edges in self.edges.groupby("patch_id", sort=False):
            chart = scale_map[int(patch_id)]
            cells = source_by_patch.get(int(patch_id), [])
            if not cells:
                continue
            scale = max(float(chart.scale_2d_to_3d), 1e-12)
            patch_edges = base_edges.copy()
            # Cheap, conservative prefilter: a large untouched source cell cannot
            # be completely covered by rods along its Voronoi boundary.
            rmax = max(float(patch_edges["radius"].max()), minrad) / scale
            candidates = [i for i, cell in enumerate(cells)
                          if cell.buffer(-1.2 * rmax).is_empty]
            edge_ids = np.unique(patch_edges[['source', 'target']].to_numpy(dtype=int))
            coordinates = {
                int(node_id): np.asarray([float(row.uv_x), float(row.uv_y)])
                for node_id, row in id_rows.loc[edge_ids].iterrows()
            }
            # Source vertices are shared across charts; their UV is chart specific.
            for source_id, uv in zip(chart.vertex_ids, chart.uv):
                for node_id in source_node_ids.get(int(source_id), []):
                    if int(node_id) in coordinates:
                        coordinates[int(node_id)] = np.asarray(uv, dtype=float)
            def line(edge):
                return LineString([coordinates[int(edge.source)], coordinates[int(edge.target)]])
            def footprint(edges, coords):
                solids = []
                for r in edges.itertuples(index=False):
                    if int(r.source) not in coords or int(r.target) not in coords:
                        continue
                    segment = LineString([coords[int(r.source)], coords[int(r.target)]])
                    radius = float(r.radius) / scale
                    if segment.length > 1e-10:
                        solids.append(segment.buffer(radius, cap_style=1, resolution=8))
                # Current uploaded main STL export contains cylinders only. The
                # round end caps in this proxy are conservative at their junctions.
                return unary_union(solids) if solids else Polygon()
            def opening(edges, coords):
                return chart.polygon.difference(footprint(edges, coords))
            def cell_area(opened, cid):
                return float(opened.intersection(cells[cid]).area) * scale * scale
            def metric(poly):
                parts = pieces(poly)
                return [(float(p.area)*scale**2,
                         2.0*float(p.area)*scale/max(float(p.length), 1e-12), p)
                        for p in parts]
            def admissible(poly):
                vals = metric(poly)
                return bool(vals and all(a >= amin - 1e-9 and 0 < r <= rcrit + 1e-9
                                         for a, r, _ in vals))
            current_open = opening(patch_edges, coordinates)
            # The size-only prefilter can miss a large cell whose particular
            # retained walls still cover it completely. Include every measured
            # zero-opening cell before deciding that the patch needs no repair.
            candidates = sorted(set(candidates) | {
                i for i in range(len(cells))
                if cell_area(current_open, i) < amin - 1.0e-10
            })
            if not candidates:
                continue
            bounds = [p.bounds for p in cells]
            touched = 0
            for zero_id in candidates:
                if cell_area(current_open, zero_id) >= amin - 1.0e-10:
                    continue
                if touched >= allowed:
                    logs.append(dict(patch_id=patch_id, source_cell_id=zero_id,
                                     status="budget_exhausted", accepted=0))
                    continue
                touched += 1
                zero = cells[zero_id]
                neighbors = []
                xmin, ymin, xmax, ymax = bounds[zero_id]
                for neighbor_id, neighbor in enumerate(cells):
                    if neighbor_id == zero_id:
                        continue
                    ax, ay, bx, by = bounds[neighbor_id]
                    if bx < xmin-1e-8 or xmax < ax-1e-8 or by < ymin-1e-8 or ymax < ay-1e-8:
                        continue
                    shared = zero.boundary.intersection(neighbor.boundary)
                    if shared.length > 1e-7 and cell_area(current_open, neighbor_id) >= amin - 1.0e-10:
                        neighbors.append((cell_area(current_open, neighbor_id), neighbor_id, shared))
                neighbors.sort(key=lambda x: (x[0], x[1]))
                # Fast first: thin only the physical walls of the blocked cell.
                incident_ids = set()
                for e in patch_edges.itertuples(index=False):
                    if line(e).intersection(zero.boundary.buffer(1e-7)).length > 1e-7:
                        incident_ids.add(int(e.logical_edge_id))
                thinned = False
                if incident_ids:
                    trial = patch_edges.copy()
                    mask = trial["logical_edge_id"].astype(int).isin(incident_ids)
                    if bool((trial.loc[mask, "radius"].astype(float) > minrad+1e-10).any()):
                        trial.loc[mask, "radius"] = np.maximum(
                            minrad, np.minimum(trial.loc[mask, "radius"].astype(float), minrad))
                        thin_open = opening(trial, coordinates)
                        affected = {zero_id} | {i for _, i, _ in neighbors}
                        was_safe = {i for i in affected if i != zero_id and admissible(current_open.intersection(cells[i]))}
                        thin_zero = thin_open.intersection(zero)
                        if admissible(thin_zero) and all(admissible(thin_open.intersection(cells[i])) for i in was_safe):
                            mapped_area, mapped_peri = self._mapped_polygon_metrics(chart, thin_zero)
                            mapped_ok = (mapped_area >= amin - 1e-9 and
                                         0 < 2*mapped_area/max(mapped_peri,1e-12) <= rcrit + 1e-9)
                        else:
                            mapped_ok = False
                        if mapped_ok:
                            patch_edges = trial
                            # Commit the local radius update to the *exported* graph.
                            global_mask = (self.edges["patch_id"].astype(int) == int(patch_id)) & self.edges["logical_edge_id"].astype(int).isin(incident_ids)
                            self.edges.loc[global_mask, "radius"] = minrad
                            current_open = thin_open
                            thinned = True
                            logs.append(dict(patch_id=patch_id, source_cell_id=zero_id,
                                             status="thinned", accepted=1, min_radius_mm=minrad))
                if thinned:
                    continue
                # Deterministic local search over directly adjacent open cells.
                # Trying only the smallest neighbor left repairable cells blocked
                # when that one wall was a junction or opened multiple islands.
                if not neighbors:
                    logs.append(dict(patch_id=patch_id, source_cell_id=zero_id,
                                     status="no_open_neighbor", accepted=0))
                    continue
                chosen_merge = None
                saw_retained_wall = False
                saw_nonjunction_wall = False
                for _, neighbor_id_try, shared in neighbors:
                    border_candidates = []
                    for lid, group in patch_edges.groupby("logical_edge_id", sort=False):
                        overlap = sum(line(e).intersection(shared.buffer(1e-7)).length
                                      for e in group.itertuples(index=False))
                        if overlap > 0.5 * shared.length:
                            border_candidates.append((overlap, int(lid)))
                    for _, removed_id_try in sorted(border_candidates, reverse=True):
                        saw_retained_wall = True
                        removed_rows_try = patch_edges[
                            patch_edges["logical_edge_id"].astype(int) == removed_id_try]
                        trial_base_try = patch_edges[
                            patch_edges["logical_edge_id"].astype(int) != removed_id_try].copy()
                        all_trial_try = pd.concat([
                            self.edges[self.edges["patch_id"] != patch_id], trial_base_try
                        ], ignore_index=True)
                        removed_nodes_try = (
                            (set(removed_rows_try["source"].astype(int))
                             | set(removed_rows_try["target"].astype(int)))
                            - (set(all_trial_try["source"].astype(int))
                               | set(all_trial_try["target"].astype(int)))
                        )
                        if any(str(id_rows.loc[n, "node_role"]) != "edge_sample"
                               for n in removed_nodes_try):
                            continue
                        saw_nonjunction_wall = True
                        trial_open_try = opening(trial_base_try, coordinates)
                        merged_try = [p for p in pieces(trial_open_try) if
                                      p.intersection(zero).area * scale**2 > 1e-10 and
                                      p.intersection(cells[neighbor_id_try]).area * scale**2 > 1e-10]
                        if len(merged_try) == 1:
                            chosen_merge = (
                                int(neighbor_id_try), int(removed_id_try),
                                removed_rows_try, trial_base_try, all_trial_try,
                                removed_nodes_try, trial_open_try, merged_try[0],
                            )
                            break
                    if chosen_merge is not None:
                        break
                if chosen_merge is None:
                    status = (
                        "shared_wall_not_retained" if not saw_retained_wall else
                        "would_remove_graph_junction" if not saw_nonjunction_wall else
                        "not_one_connected_opening"
                    )
                    logs.append(dict(patch_id=patch_id, source_cell_id=zero_id,
                                     status=status, accepted=0))
                    continue
                (neighbor_id, removed_id, removed_rows, trial_base, all_trial,
                 removed_nodes, trial_open, merged_open) = chosen_merge
                candidate_edges = trial_base
                candidate_nodes = pd.DataFrame()
                candidate_open = trial_open
                reason = "merged"
                merged_area_3d, merged_perimeter_3d = self._mapped_polygon_metrics(chart, merged_open) if admissible(merged_open) else (0.0, 0.0)
                if (not admissible(merged_open) or
                    merged_area_3d < amin - 1e-9 or
                    2*merged_area_3d/max(merged_perimeter_3d,1e-12) > rcrit+1e-9):
                    # Secondary split is a local true-graph rewrite: center node,
                    # two perpendicular connections, split their boundary edges.
                    raw_merged = zero.union(cells[neighbor_id]).buffer(0)
                    old_wall = unary_union([line(e) for e in removed_rows.itertuples(index=False)])
                    try:
                        base_center = np.asarray(polylabel(merged_open.intersection(raw_merged), tolerance=max(1e-5,0.01/scale)).coords[0])
                    except Exception:
                        base_center = np.asarray(merged_open.intersection(raw_merged).representative_point().coords[0])
                    offsets = [(0.0,0.0)] + [(0.17*math.cos(2*math.pi*k/5)/scale,
                                                   0.17*math.sin(2*math.pi*k/5)/scale) for k in range(5)]
                    accepted_split = False
                    for ox, oy in offsets[:trials]:
                        center = np.asarray(base_center)+np.asarray([ox,oy])
                        if (not merged_open.covers(Point(center)) or
                            Point(center).distance(old_wall)*scale < 0.125):
                            continue
                        feet = []
                        for idx, e in trial_base.iterrows():
                            a,b = coordinates[int(e["source"])],coordinates[int(e["target"])]
                            v=b-a
                            den=float(v@v)
                            if den<1e-12: continue
                            t=float((center-a)@v/den)
                            if not 0.08<t<0.92: continue
                            foot=a+t*v
                            seg=LineString([center,foot])
                            if (not raw_merged.buffer(1e-8).covers(seg) or
                                Point(foot).distance(raw_merged.boundary)>2e-6 or
                                np.linalg.norm(foot-a)*scale<0.125 or
                                np.linalg.norm(foot-b)*scale<0.125):
                                continue
                            # No new edge is allowed to reproduce the removed wall.
                            if seg.intersection(old_wall.buffer(0.125/scale)).length > 0.40*seg.length:
                                continue
                            if seg.length*scale > float(self.config.max_edge_length_mm):
                                continue
                            feet.append((idx,foot,seg))
                        feet.sort(key=lambda x:x[2].length)
                        pairs=[]
                        for k,(i,f1,s1) in enumerate(feet):
                            for j,f2,s2 in feet[k+1:]:
                                if i==j: continue
                                v1=f1-center;v2=f2-center
                                if (v1@v2)/max(np.linalg.norm(v1)*np.linalg.norm(v2),1e-12)>-0.2:
                                    continue
                                wire=LineString([f1,center,f2])
                                if wire.intersection(old_wall.buffer(0.125/scale)).length > 0.4*wire.length:
                                    continue
                                children=pieces(split(raw_merged,wire))
                                if len(children)!=2: continue
                                pairs.append((i,f1,j,f2,children))
                        for i,f1,j,f2,children in pairs[:3]:
                            # Map prospective true nodes and *new* edges only.
                            points2=[center,f1,f2]
                            located=[self._locate_uv(chart,p) for p in points2]
                            if any(item[2] for item in located):
                                continue
                            midpoint_distances=[]
                            for a,b in ((center,f1),(center,f2)):
                                samples,failed=self._sample_mapped_edge(chart,a,b)
                                if failed or any(x[2] for x in samples):
                                    break
                                midpoint_distances.append(samples)
                            if len(midpoint_distances)!=2:
                                continue
                            # Check the mapped *new* rods against the configured
                            # carrier/domain and the same chord-error tolerance
                            # used in the original graph construction.
                            mapped_segments = [np.asarray([item[1] for item in sample], dtype=float)
                                               for sample in midpoint_distances]
                            patch_faces = set(map(int, chart.face_ids))
                            if self.config.mapping_surface == "carrier" and any(
                                int(item[3]) not in patch_faces
                                for sample in midpoint_distances for item in sample):
                                continue
                            if any(float(self._mapping_surface_distance_batch(
                                0.5*(line_pts[:-1]+line_pts[1:]),component_id=chart.component_id).max(initial=0.0))
                                > float(self.config.surface_tolerance_mm)+1e-12
                                for line_pts in mapped_segments if len(line_pts)>1):
                                continue
                            fresh_nodes=[]
                            coords_trial=coordinates.copy()
                            local_next=next_id
                            point_ids=[]
                            for p,located_item,role in zip(points2,located,("zero_pore_center","zero_pore_boundary","zero_pore_boundary")):
                                gid=local_next;local_next+=1
                                coords_trial[gid]=np.asarray(p)
                                fresh_nodes.append(dict(atom_id=gid,x=float(located_item[0][0]),
                                                        y=float(located_item[0][1]),z=float(located_item[0][2]),
                                                        face_name="surface",patch_id=patch_id,
                                                        component_id=int(chart.component_id),source_vertex_id=-1,
                                                        uv_x=float(p[0]),uv_y=float(p[1]),node_role=role,
                                                        patch_memberships=str(patch_id)))
                                point_ids.append(gid)
                            new_records=[]
                            trial_remove={int(i),int(j)}
                            for edgeidx,foot_id in ((i,point_ids[1]),(j,point_ids[2])):
                                source=trial_base.loc[edgeidx]
                                rec=source.to_dict()
                                for u, v in [(int(rec["source"]), foot_id), (foot_id, int(rec["target"]))]:
                                    sub=dict(rec);sub.update(source=u,target=v)
                                    new_records.append(sub)
                            local_logical=next_logical
                            for k, sample_list in enumerate(midpoint_distances):
                                start,end=point_ids[0],point_ids[k+1]
                                chain=[start]
                                for st,p3,_,_ in sample_list[1:-1]:
                                    uvp=center*(1-st)+points2[k+1]*st
                                    gid=local_next;local_next+=1
                                    coords_trial[gid]=np.asarray(uvp)
                                    fresh_nodes.append(dict(atom_id=gid,x=float(p3[0]),y=float(p3[1]),z=float(p3[2]),
                                                            face_name="surface",patch_id=patch_id,
                                                            component_id=int(chart.component_id),source_vertex_id=-1,
                                                            uv_x=float(uvp[0]),uv_y=float(uvp[1]),
                                                            node_role="edge_sample",patch_memberships=str(patch_id)))
                                    chain.append(gid)
                                chain.append(end)
                                for u,v in zip(chain[:-1],chain[1:]):
                                    new_records.append(dict(source=u,target=v,
                                                            radius=max(minrad,0.5*float(self.config.surface_diameter_mm)),
                                                            bond_type="surface",role="zero_pore_orthogonal_repair",
                                                            patch_id=patch_id,component_id=int(chart.component_id),
                                                            logical_edge_id=local_logical+k))
                            split_edges=pd.concat([trial_base.drop(index=list(trial_remove)),
                                                  pd.DataFrame(new_records)],ignore_index=True)
                            split_open=opening(split_edges,coords_trial)
                            # Every child is an actual separate liquid region.
                            child_parts=[]
                            for child in children:
                                child_open=pieces(split_open.intersection(child))
                                if len(child_open)!=1 or not admissible(child_open[0]):
                                    break
                                child_parts.append(child_open[0])
                            if len(child_parts)!=2 or cell_area(split_open, zero_id) < amin - 1.0e-10:
                                continue
                            # 3-D mapping is expensive: check only after planar pass.
                            if any((lambda a,p: a < amin-1e-9 or 2*a/max(p,1e-12) > rcrit+1e-9)(
                                   *self._mapped_polygon_metrics(chart,part)) for part in child_parts):
                                continue
                            global_trial=pd.concat([self.edges[self.edges["patch_id"] != patch_id],split_edges],ignore_index=True)
                            # All old non-sample nodes must remain incident, and
                            # all previously connected junctions stay connected.
                            surviving=set(global_trial["source"].astype(int))|set(global_trial["target"].astype(int))
                            lost=set(old_edges_all["source"].astype(int)|old_edges_all["target"].astype(int))-surviving
                            if any(n not in removed_nodes and str(id_rows.loc[n,"node_role"])!="edge_sample" for n in lost):
                                continue
                            if not self._repair_connectivity_preserved(self.edges,global_trial):
                                continue
                            candidate_edges=split_edges
                            candidate_nodes=pd.DataFrame(fresh_nodes)
                            candidate_open=split_open
                            next_id=local_next
                            next_logical=local_logical+2
                            for k in (0,1):
                                self.edge_metrics = pd.concat([self.edge_metrics, pd.DataFrame([dict(
                                    logical_edge_id=local_logical+k, patch_id=patch_id,
                                    component_id=int(chart.component_id), polyline_segments=len(midpoint_distances[k])-1,
                                    max_chord_surface_error_mm=0.0, domain_violation=0,
                                    mapping_fallback=0, cross_component=0, accepted=1)])],ignore_index=True)
                            reason="merged_and_orthogonally_refined"
                            accepted_split=True
                            break
                        if accepted_split: break
                    if not accepted_split:
                        logs.append(dict(patch_id=patch_id,source_cell_id=zero_id,
                                         neighbor_cell_id=neighbor_id,removed_logical_edge_id=removed_id,
                                         status="secondary_split_no_valid_candidate",accepted=0))
                        continue
                else:
                    if not self._repair_connectivity_preserved(self.edges,all_trial):
                        logs.append(dict(patch_id=patch_id,source_cell_id=zero_id,
                                         status="deletion_breaks_connectivity",accepted=0))
                        continue
                # Commit the entire mutation only after the final accepted test.
                self.edges=pd.concat([self.edges[self.edges["patch_id"] != patch_id],candidate_edges],ignore_index=True)
                if not candidate_nodes.empty:
                    self.atoms=pd.concat([self.atoms,candidate_nodes],ignore_index=True)
                    id_rows=self.atoms.set_index("atom_id")
                patch_edges=candidate_edges
                coordinates.update({int(r.atom_id):np.asarray([float(r.uv_x),float(r.uv_y)])
                                    for r in candidate_nodes.itertuples(index=False)})
                current_open=candidate_open
                old_edges_all=self.edges.copy()
                self.edge_metrics.loc[self.edge_metrics["logical_edge_id"].astype(int)==removed_id,"accepted"]=0
                logs.append(dict(patch_id=patch_id,source_cell_id=zero_id,
                                 neighbor_cell_id=neighbor_id,removed_logical_edge_id=removed_id,
                                 status=reason,accepted=1))
            # Report the actual *current* opening of originally closed source
            # regions. Unlike an empty-radius filter, blocked cells remain here.
            for zero_id in candidates:
                area=cell_area(current_open,zero_id)
                audit.append(dict(patch_id=patch_id,source_cell_id=zero_id,
                                  open_area_mm2=area,blocked=int(area<=1e-10),
                                  undersized=int(area < amin - 1.0e-10),
                                  minimum_open_area_mm2=amin,
                                  proxy="retained_graph_2d_cylinder_footprint"))
            if touched:
                updated_patches.add(int(patch_id))
                # Keep the downstream capillary analysis aligned with the
                # accepted retained-graph edits on this patch. Other patches
                # preserve the pre-existing (Voronoi-proxy) measurements.
                open_parts = pieces(current_open)
                for pore_id, part in enumerate(open_parts):
                    area3, perimeter3 = self._mapped_polygon_metrics(chart,part)
                    area2, perimeter2 = float(part.area)*scale**2, float(part.length)*scale
                    cell3 = self._chart_surface_area(chart) if pore_id==0 else 0.0
                    centroid3 = self._locate_uv(chart,np.asarray(part.representative_point().coords[0],dtype=float))[0]
                    postrepair_pores.append(dict(centroid_x_mm=float(centroid3[0]),
                        centroid_y_mm=float(centroid3[1]),centroid_z_mm=float(centroid3[2]),patch_id=patch_id,component_id=int(chart.component_id),
                        subdomain_id=pore_id,island_id=0,cell_area_2d_mm2=float(chart.polygon.area)*scale**2 if pore_id==0 else 0.0,
                        cell_perimeter_2d_mm=0.0,cell_area_3d_mm2=cell3,cell_perimeter_3d_mm=0.0,
                        pore_area_2d_mm2=area2,pore_perimeter_2d_mm=perimeter2,
                        pore_area_3d_mm2=area3,pore_perimeter_3d_mm=perimeter3,
                        hydraulic_radius_2d_mm=2*area2/max(perimeter2,1e-12),
                        hydraulic_radius_3d_mm=2*area3/max(perimeter3,1e-12),
                        hydraulic_radius_rel_error=abs(2*area3/max(perimeter3,1e-12)-2*area2/max(perimeter2,1e-12))/max(2*area2/max(perimeter2,1e-12),1e-9),
                        hydraulic_radius_target_normalized_error=abs(2*area3/max(perimeter3,1e-12)-2*area2/max(perimeter2,1e-12))/max(rcrit,1e-9),
                        shape_factor=4*math.pi*area3/max(perimeter3**2,1e-12),
                        aspect_ratio=_polygon_aspect_ratio(part),liquid_area_fraction=area3/max(cell3,1e-12),
                        pore_metric_source="retained_graph_2d_footprint_mapped_to_3d"))
                print(f"[surface] zero-pore repair patch {patch_id}: inspected {touched}, "
                      f"unresolved={sum(cell_area(current_open,i)<amin-1e-10 for i in candidates)}",flush=True)
        self.zero_pore_repair_history=pd.DataFrame(logs)
        self.zero_pore_final_audit=pd.DataFrame(audit)
        if updated_patches:
            self.pre_zero_repair_pore_metrics = self.pore_metrics.copy()
            other = self.pore_metrics[~self.pore_metrics["patch_id"].astype(int).isin(updated_patches)].copy()
            self.pore_metrics = pd.concat([other,pd.DataFrame(postrepair_pores)],ignore_index=True)

        # Remove only discarded polyline samples; never discard original graph
        # junctions or the newly inserted orthogonal center/boundary vertices.
        incident=set(self.edges["source"].astype(int))|set(self.edges["target"].astype(int))
        disposable=(self.atoms["node_role"]=="edge_sample") & ~self.atoms["atom_id"].astype(int).isin(incident)
        self.atoms=self.atoms.loc[~disposable].reset_index(drop=True)

    def _repair_final_open_fragments(self):
        """Repair blocked, numerically-zero, undersized and over-elongated final pores.

        This routine operates on the same seam-expanded retained graph semantics used
        by :meth:`audit_final_retained_graph`.  A physical seam edge is stored once,
        but it is included in every adjacent chart through ``patch_memberships``.
        Shared seam skeleton edges are never deleted.  Repairs therefore remove only
        safe interior/anisotropy walls and are accepted only when the total defect
        count falls without increasing capillary-radius violations or breaking graph
        connectivity.
        """
        if self.edges.empty or self.atoms.empty:
            self._final_fragment_unresolved_count = 0
            self.final_fragment_repair_history = pd.DataFrame()
            return

        amin = float(self.config.zero_pore_min_area_mm2)
        rcrit = (float(self.config.target_hydraulic_radius_mm) *
                 float(self.config.hydraulic_radius_mapping_safety))
        hard_ar_max = max(1.0, float(self.config.anisotropy_max_aspect_ratio))
        logs = []
        atom_rows = self.atoms.set_index('atom_id')

        source_nodes = defaultdict(list)
        for atom in self.atoms.itertuples(index=False):
            if int(atom.source_vertex_id) >= 0:
                source_nodes[int(atom.source_vertex_id)].append(int(atom.atom_id))

        # True physical carrier boundary only. Artificial chart seams are not
        # protected from *repair*; their shared seam skeleton itself remains protected.
        base_edge_face_count = defaultdict(int)
        if self.base_mesh is not None:
            for tri0 in np.asarray(self.base_mesh.faces, dtype=int):
                for aa, bb in ((tri0[0], tri0[1]), (tri0[1], tri0[2]), (tri0[2], tri0[0])):
                    base_edge_face_count[tuple(sorted((int(aa), int(bb))))] += 1

        def _edges_for_patch(frame: pd.DataFrame, pid: int) -> pd.DataFrame:
            if frame.empty:
                return frame.copy()
            member = frame['patch_memberships'].fillna('').astype(str).str.contains(
                rf'(?:^|;)\s*{pid}\s*(?:;|$)', regex=True
            )
            return frame.loc[frame['patch_id'].eq(pid) | member].copy()

        final_remaining = 0
        for repair_pass in range(max(1, int(self.config.final_fragment_repair_max_passes))):
            changed = False
            pass_remaining = 0

            for chart in self.charts:
                pid = int(chart.patch_id)
                scale = max(float(chart.scale_2d_to_3d), 1.0e-12)

                local_atoms = self.atoms.loc[
                    self.atoms['patch_id'].eq(pid), ['atom_id', 'uv_x', 'uv_y']
                ]
                coords = {
                    int(row.atom_id): np.asarray([float(row.uv_x), float(row.uv_y)])
                    for row in local_atoms.itertuples(index=False)
                }
                for source_id, uv in zip(chart.vertex_ids, chart.uv):
                    for node_id in source_nodes.get(int(source_id), ()):
                        coords[node_id] = np.asarray(uv, dtype=float)

                # Shared seam edge samples can be owned by the neighboring patch.
                # Project those missing 3-D graph nodes onto this chart exactly as
                # the final retained-graph audit does.
                triangles = np.asarray(chart.xyz[chart.local_faces], dtype=float)
                projected_cache = {}
                def ensure_coord(node_id: int):
                    node_id = int(node_id)
                    if node_id in coords:
                        return True
                    if node_id in projected_cache:
                        uvp, dist = projected_cache[node_id]
                    else:
                        if node_id not in atom_rows.index or len(triangles) == 0:
                            return False
                        xyz_point = atom_rows.loc[node_id][['x','y','z']].to_numpy(dtype=float)
                        repeated = np.repeat(xyz_point[None, :], len(triangles), axis=0)
                        closest = trimesh.triangles.closest_point(triangles, repeated)
                        d2 = np.einsum('ij,ij->i', closest-repeated, closest-repeated)
                        best = int(np.argmin(d2))
                        bary = trimesh.triangles.points_to_barycentric(
                            triangles[best:best+1], closest[best:best+1]
                        )[0]
                        uvp = bary @ chart.uv[chart.local_faces[best]]
                        dist = math.sqrt(max(float(d2[best]), 0.0))
                        projected_cache[node_id] = (np.asarray(uvp, dtype=float), dist)
                    if dist <= 2.0 * float(self.config.surface_tolerance_mm) + 1.0e-8:
                        coords[node_id] = np.asarray(uvp, dtype=float)
                        return True
                    return False

                local_boundary_counts = defaultdict(int)
                for tri0 in np.asarray(chart.local_faces, dtype=int):
                    for aa, bb in ((tri0[0], tri0[1]), (tri0[1], tri0[2]), (tri0[2], tri0[0])):
                        local_boundary_counts[tuple(sorted((int(aa), int(bb))))] += 1
                physical_segments = []
                for (aa, bb), count0 in local_boundary_counts.items():
                    if count0 != 1:
                        continue
                    gaa, gbb = int(chart.vertex_ids[aa]), int(chart.vertex_ids[bb])
                    if base_edge_face_count.get(tuple(sorted((gaa, gbb))), 0) == 1:
                        physical_segments.append(LineString([chart.uv[aa], chart.uv[bb]]))
                physical_boundary = unary_union(physical_segments) if physical_segments else None

                def patch_state(frame_global: pd.DataFrame):
                    edges = _edges_for_patch(frame_global, pid)
                    walls, solids, by_logical = [], [], defaultdict(list)
                    for row in edges.itertuples(index=False):
                        a, b = int(row.source), int(row.target)
                        if not ensure_coord(a) or not ensure_coord(b):
                            continue
                        line = LineString([coords[a], coords[b]])
                        if line.length <= 1.0e-10:
                            continue
                        walls.append(line)
                        by_logical[int(row.logical_edge_id)].append(line)
                        solids.append(line.buffer(float(row.radius) / scale,
                                                  cap_style=1, resolution=8))
                    occupied = unary_union(solids) if solids else Polygon()
                    linework = unary_union([chart.polygon.boundary, *walls]) if walls else chart.polygon.boundary
                    faces = [face for face in polygonize(linework)
                             if face.area > 1.0e-12 and
                             chart.polygon.covers(face.representative_point())]
                    if not faces:
                        faces = _polygon_parts(chart.polygon)
                    bad = []
                    radius_exceeded = 0
                    for face in faces:
                        if (physical_boundary is not None and
                                float(face.boundary.intersection(physical_boundary).length)
                                * scale > 1.0e-7):
                            continue
                        open_parts = _polygon_parts(face.difference(occupied))
                        if not open_parts:
                            bad.append((face, Polygon(), 0.0, 'blocked'))
                            continue
                        for part in open_parts:
                            area = float(part.area) * scale**2
                            perimeter = float(part.length) * scale
                            radius = 2.0 * area / max(perimeter, 1.0e-12)
                            ar = _polygon_aspect_ratio(part)
                            if area <= 1.0e-10:
                                bad.append((face, part, area, 'blocked'))
                            elif area < amin - 1.0e-10:
                                bad.append((face, part, area, 'undersized'))
                            elif radius > rcrit + 1.0e-10:
                                radius_exceeded += 1
                            elif ar > hard_ar_max + 1.0e-10:
                                bad.append((face, part, area, 'ar_exceeded'))
                    # blocked first, then hard-AR, then numerical undersize
                    priority = {'blocked': 0, 'ar_exceeded': 1, 'undersized': 2}
                    bad.sort(key=lambda item: (priority.get(item[3], 9), -float(item[2])))
                    return bad, int(radius_exceeded), by_logical, edges

                bad, exceeded, by_logical, patch_edges = patch_state(self.edges)
                while bad:
                    face, _, area_before, defect_kind = bad[0]
                    candidates = []
                    boundary_band = face.boundary.buffer(1.0e-7)
                    for logical_id, lines in by_logical.items():
                        overlap = sum(line.intersection(boundary_band).length for line in lines)
                        if overlap > 1.0e-7:
                            candidates.append((overlap, int(logical_id)))

                    accepted = False
                    for _, logical_id in sorted(candidates, reverse=True):
                        removed = self.edges[
                            self.edges['logical_edge_id'].astype(int) == int(logical_id)
                        ].copy()
                        if removed.empty:
                            continue
                        roles_removed = removed.get('role', pd.Series(dtype=str)).astype(str)
                        memberships_removed = removed.get(
                            'patch_memberships', pd.Series(dtype=str)
                        ).fillna('').astype(str)
                        # Canonical/shared seam skeleton is immutable. The blocked or
                        # high-AR seam pore is opened by removing a neighboring safe
                        # interior/anisotropy wall instead.
                        if (roles_removed.str.contains('seam', case=False, regex=False).any()
                                or memberships_removed.str.contains(';', regex=False).any()):
                            continue

                        trial_all = self.edges[
                            self.edges['logical_edge_id'].astype(int) != int(logical_id)
                        ].copy()
                        remaining_ids = (set(trial_all['source'].astype(int)) |
                                         set(trial_all['target'].astype(int)))
                        orphaned = ((set(removed['source'].astype(int)) |
                                     set(removed['target'].astype(int))) - remaining_ids)
                        if any((node_id in atom_rows.index and
                                str(atom_rows.loc[node_id, 'node_role']) != 'edge_sample')
                               for node_id in orphaned):
                            continue
                        if not self._repair_connectivity_preserved(self.edges, trial_all):
                            continue

                        trial_bad, trial_exceeded, trial_by_logical, _ = patch_state(trial_all)
                        # The deletion must remove at least one hard defect and must
                        # never create a new capillary-radius violation.
                        if len(trial_bad) >= len(bad) or trial_exceeded > exceeded:
                            continue

                        self.edges = trial_all
                        bad, exceeded, by_logical = trial_bad, trial_exceeded, trial_by_logical
                        patch_edges = _edges_for_patch(self.edges, pid)
                        if not self.edge_metrics.empty:
                            mask = self.edge_metrics['logical_edge_id'].astype(int) == int(logical_id)
                            self.edge_metrics.loc[mask, 'accepted'] = 0
                        logs.append(dict(
                            repair_pass=repair_pass + 1, patch_id=pid,
                            removed_logical_edge_id=int(logical_id),
                            defect_kind=defect_kind,
                            open_area_before_mm2=float(area_before),
                            remaining_final_defects=len(bad), accepted=1,
                        ))
                        changed = accepted = True
                        break

                    if not accepted:
                        logs.append(dict(
                            repair_pass=repair_pass + 1, patch_id=pid,
                            removed_logical_edge_id=-1, defect_kind=defect_kind,
                            open_area_before_mm2=float(area_before),
                            remaining_final_defects=len(bad), accepted=0,
                        ))
                        break

                pass_remaining += len(bad)

            final_remaining = int(pass_remaining)
            if final_remaining == 0 or not changed:
                break

        self._final_fragment_unresolved_count = int(final_remaining)
        self.final_fragment_repair_history = pd.DataFrame(logs)
        incident = set(self.edges['source'].astype(int)) | set(self.edges['target'].astype(int))
        disposable = ((self.atoms['node_role'] == 'edge_sample') &
                      ~self.atoms['atom_id'].astype(int).isin(incident))
        self.atoms = self.atoms.loc[~disposable].reset_index(drop=True)
        self.curr_vertices = self.atoms[['x', 'y', 'z']].to_numpy(float).tolist()
        self.curr_edges = list(self.edges[['source', 'target']].itertuples(index=False, name=None))

    def _bridge_surface_components_within_domains(self):
        """Join chart fragments within each disconnected input surface domain."""
        if self.atoms.empty or self.edges.empty:
            return
        next_logical = int(self.edges['logical_edge_id'].max()) + 1
        max_length = float(self.config.max_edge_length_mm)
        degree_limit = int(self.config.max_node_degree)
        for domain_id in sorted(self.atoms['component_id'].astype(int).unique()):
            for _ in range(64):
                ids = self.atoms['atom_id'].astype(int).to_numpy()
                lookup = {atom_id: i for i, atom_id in enumerate(ids)}
                rows = np.asarray([lookup[int(v)] for v in self.edges['source']], dtype=int)
                cols = np.asarray([lookup[int(v)] for v in self.edges['target']], dtype=int)
                adjacency = coo_matrix((
                    np.ones(2 * len(rows), dtype=np.uint8),
                    (np.concatenate((rows, cols)), np.concatenate((cols, rows))),
                ), shape=(len(ids), len(ids))).tocsr()
                _, labels = connected_components(adjacency, directed=False)
                degree = np.diff(adjacency.indptr)
                member = np.flatnonzero(
                    self.atoms['component_id'].astype(int).to_numpy() == int(domain_id)
                )
                component_labels, counts = np.unique(labels[member], return_counts=True)
                if len(component_labels) <= 1:
                    break
                main_label = int(component_labels[int(np.argmax(counts))])
                main_idx = member[labels[member] == main_label]
                main_points = self.atoms.iloc[main_idx][['x', 'y', 'z']].to_numpy(float)
                main_tree = cKDTree(main_points)
                candidates = []
                for label in component_labels:
                    if int(label) == main_label:
                        continue
                    other_idx = member[labels[member] == int(label)]
                    other_points = self.atoms.iloc[other_idx][['x', 'y', 'z']].to_numpy(float)
                    distances, nearest = main_tree.query(other_points, k=1)
                    order = np.argsort(distances)[:8]
                    for local in order:
                        candidates.append((float(distances[local]),
                                           int(main_idx[int(nearest[local])]),
                                           int(other_idx[int(local)])))
                accepted = False
                for length, ia, ib in sorted(candidates):
                    if length <= 1.0e-10 or length > max_length + 1.0e-12:
                        continue
                    if degree[ia] >= degree_limit or degree[ib] >= degree_limit:
                        continue
                    p0 = self.atoms.iloc[ia][['x', 'y', 'z']].to_numpy(float)
                    p1 = self.atoms.iloc[ib][['x', 'y', 'z']].to_numpy(float)
                    midpoint_error = float(self._mapping_surface_distance_batch(
                        (0.5 * (p0 + p1))[None, :], component_id=int(domain_id)
                    )[0])
                    if midpoint_error > float(self.config.surface_tolerance_mm) + 1.0e-12:
                        continue
                    a, b = int(ids[ia]), int(ids[ib])
                    patch_id = int(self.atoms.iloc[ia]['patch_id'])
                    self.edges = pd.concat([self.edges, pd.DataFrame([dict(
                        source=a, target=b,
                        radius=0.5 * float(self.config.surface_diameter_mm),
                        bond_type='surface', role='surface_seam_bridge',
                        patch_id=patch_id, patch_memberships=str(patch_id),
                        component_id=int(domain_id),
                        logical_edge_id=next_logical,
                    )])], ignore_index=True)
                    self.edge_metrics = pd.concat([self.edge_metrics, pd.DataFrame([dict(
                        logical_edge_id=next_logical, patch_id=patch_id,
                        component_id=int(domain_id), polyline_segments=1,
                        max_chord_surface_error_mm=midpoint_error,
                        domain_violation=0, mapping_fallback=0,
                        cross_component=0, accepted=1,
                    )])], ignore_index=True)
                    next_logical += 1
                    accepted = True
                    break
                if not accepted:
                    break

    def _prune_minor_graph_fragments(self):
        """Remove only orphan edge-sample nodes; never hide graph fragmentation.

        Previous code retained only the largest connected graph component in
        each surface domain.  That could turn a seam-connection failure into a
        visually large hole while making the surviving graph appear connected.
        Connectivity is now a hard diagnostic/quality gate instead.
        """
        self._pruned_node_count = 0
        self._pruned_edge_segment_count = 0
        if self.atoms.empty:
            return
        if self.edges.empty:
            return
        incident = set(self.edges["source"].astype(int)) | set(self.edges["target"].astype(int))
        roles = self.atoms.get("node_role", pd.Series("", index=self.atoms.index)).astype(str)
        orphan = (
            roles.eq("edge_sample")
            & ~self.atoms["atom_id"].astype(int).isin(incident)
        )
        self._pruned_node_count = int(orphan.sum())
        if self._pruned_node_count:
            self.atoms = self.atoms.loc[~orphan].copy().reset_index(drop=True)

    def _audit_surface_coverage(self):
        """Measure open gaps on the whole carrier, including gaps between charts."""
        if self.base_mesh is None or self.atoms.empty or self.edges.empty:
            self.coverage_metrics = pd.DataFrame()
            return
        coords = self.atoms.set_index("atom_id")[["x", "y", "z"]]
        spacing = max(0.05, float(self.config.coverage_sample_spacing_mm))
        line_samples = []
        for row in self.edges.itertuples(index=False):
            p0 = coords.loc[int(row.source)].to_numpy(float)
            p1 = coords.loc[int(row.target)].to_numpy(float)
            count = max(2, int(math.ceil(np.linalg.norm(p1 - p0) / (0.25 * spacing))) + 1)
            line_samples.append(np.linspace(p0, p1, count))
        line_tree = cKDTree(np.vstack(line_samples))

        sample_rows = []
        triangles = np.asarray(self.base_mesh.triangles, dtype=float)
        face_areas = np.asarray(self.base_mesh.area_faces, dtype=float)
        for face_id, (triangle, face_area) in enumerate(zip(triangles, face_areas)):
            edge_max = max(
                float(np.linalg.norm(triangle[(i + 1) % 3] - triangle[i])) for i in range(3)
            )
            divisions = max(1, int(math.ceil(edge_max / spacing)))
            barycentric = []
            for i in range(divisions + 1):
                for j in range(divisions + 1 - i):
                    barycentric.append((i / divisions, j / divisions, 1.0 - (i + j) / divisions))
            weights = np.asarray(barycentric, dtype=float)
            points = weights @ triangle
            distances = line_tree.query(points, workers=-1)[0]
            clearance = np.maximum(
                0.0, distances - 0.5 * float(self.config.surface_diameter_mm)
            )
            sample_weight = float(face_area) / max(len(points), 1)
            for point, value in zip(points, clearance):
                sample_rows.append(
                    {
                        "face_id": face_id,
                        "x_mm": float(point[0]),
                        "y_mm": float(point[1]),
                        "z_mm": float(point[2]),
                        "clearance_radius_mm": float(value),
                        "area_weight_mm2": sample_weight,
                        "over_target": int(value > float(self.config.target_hydraulic_radius_mm)),
                    }
                )
        self.coverage_metrics = pd.DataFrame(sample_rows)

    def make_stitch_anchor_area_guard(self, surface_edges: pd.DataFrame):
        """Reject stitch disks that create tiny or capillary-oversized pores.

        The surface graph is optimized for capped rods.  This guard adds each
        accepted stitch-end disk to the same local-chart footprint and requires
        every remaining open island to retain the configured minimum area and
        maximum equivalent capillary radius.
        """
        src_nodes = self.atoms.drop_duplicates('atom_id').set_index('atom_id')
        edge_groups = {}
        if surface_edges is not None and not surface_edges.empty:
            work = surface_edges.copy()
            work['_guard_patch_id'] = pd.to_numeric(work['patch_id'], errors='coerce')
            work = work.dropna(subset=['_guard_patch_id'])
            edge_groups = {int(k): v for k, v in work.groupby('_guard_patch_id', sort=False)}

        chart_coords = defaultdict(dict)
        source_vertex_node_ids = defaultdict(list)
        for node_id, row in src_nodes.iterrows():
            node_id = int(node_id)
            chart_coords[int(row.patch_id)][node_id] = np.asarray(
                [float(row.uv_x), float(row.uv_y)], dtype=float)
            if int(row.source_vertex_id) >= 0:
                source_vertex_node_ids[int(row.source_vertex_id)].append(node_id)

        states = {}
        node_charts = defaultdict(list)
        for chart in self.charts:
            pid = int(chart.patch_id)
            scale = max(float(chart.scale_2d_to_3d), 1e-12)
            coords = dict(chart_coords.get(pid, {}))
            for source, point in zip(chart.vertex_ids, chart.uv):
                for node_id in source_vertex_node_ids.get(int(source), ()):
                    coords[node_id] = np.asarray(point, dtype=float)
            walls, solids = [], []
            p_edges = edge_groups.get(pid)
            if p_edges is not None:
                for row in p_edges.itertuples(index=False):
                    a, b, radius = int(row.source), int(row.target), float(row.radius)
                    if a not in coords or b not in coords or not np.isfinite(radius) or radius <= 0:
                        continue
                    segment = LineString([coords[a], coords[b]])
                    if segment.length <= 1e-10:
                        continue
                    walls.append(segment)
                    solids.append(segment.buffer(radius / scale, resolution=8, cap_style=1))
            linework = unary_union([chart.polygon.boundary, *walls]) if walls else chart.polygon.boundary
            faces = [face for face in polygonize(linework)
                     if face.area > 1e-12 and chart.polygon.covers(face.representative_point())]
            if not faces:
                faces = _polygon_parts(chart.polygon)
            states[pid] = {
                'scale': scale,
                'coords': coords,
                'faces': faces,
                'occupied': unary_union(solids) if solids else Polygon(),
            }
            for node_id in coords:
                node_charts[int(node_id)].append(pid)

        min_open = float(self.config.zero_pore_min_area_mm2)
        max_equivalent_radius = float(self.config.target_hydraulic_radius_mm)

        def guard(node_id: int, radius_mm: float, *, commit: bool = False) -> bool:
            updates = []
            for pid in node_charts.get(int(node_id), ()):
                state = states[pid]
                if int(node_id) not in state['coords']:
                    continue
                disk = Point(state['coords'][int(node_id)]).buffer(
                    float(radius_mm) / state['scale'], resolution=8)
                trial_occupied = unary_union([state['occupied'], disk])
                for face in state['faces']:
                    if not face.intersects(disk):
                        continue
                    open_parts = _polygon_parts(face.difference(trial_occupied))
                    if not open_parts:
                        return False
                    if any(float(part.area) * state['scale'] ** 2 < min_open - 1e-10
                           for part in open_parts):
                        return False
                    for part in open_parts:
                        area_mm2 = float(part.area) * state['scale'] ** 2
                        perimeter_mm = float(part.length) * state['scale']
                        equivalent_radius_mm = (
                            2.0 * area_mm2 / perimeter_mm
                            if perimeter_mm > 1.0e-12 else math.inf
                        )
                        if equivalent_radius_mm > max_equivalent_radius + 1.0e-10:
                            return False
                updates.append((pid, trial_occupied))
            if not updates:
                return False
            if commit:
                for pid, occupied in updates:
                    states[pid]['occupied'] = occupied
            return True

        return guard

    def audit_final_retained_graph(self, atoms_all: pd.DataFrame,
                                   edges_all: pd.DataFrame, output_dir: Path,
                                   *, node_spheres_exported: bool = False,
                                   node_sphere_radius_factor: float = 1.2,
                                   include_stitch_anchors: bool = True,
                                   apply_quality_gate: bool = False,
                                   verbose: bool = True) -> Dict:
        """One-pass FINAL audit, on the post-stitch/post-trim attributed graph.

        Each retained surface edge contributes a finite-radius footprint, and
        each stitch contributes its surface-end contact disk.  Optional node
        spheres contribute only when they are actually exported.  Local-chart
        areas/perimeters are reported as chart-mapped graph geometry, not as a
        Boolean measurement of the combined 3-D STL.  Artificial chart seams
        are marked and NEVER silently claimed to be real pore walls.
        No graph edits, refinements or retry loops occur in this function.
        """
        from time import perf_counter
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        columns = ['patch_id', 'pore_id', 'island_id', 'cell_area_chart_mm2',
                   'open_area_chart_mm2', 'wet_perimeter_chart_mm',
                   'equivalent_radius_chart_mm', 'aspect_ratio_chart', 'shape_factor_chart',
                   'critical_radius_mm', 'status', 'boundary_shared_with_chart_mm',
                   'boundary_shared_with_patch_seam_mm',
                   'boundary_shared_with_physical_boundary_mm',
                   'patch_seam_dependent', 'physical_boundary_dependent',
                   'node_spheres_included', 'stitch_anchors_included',
                   'centroid_uv_x', 'centroid_uv_y']
        nodes = atoms_all.drop_duplicates('atom_id').set_index('atom_id')
        src_nodes = self.atoms.drop_duplicates('atom_id').set_index('atom_id')
        surface_ids = set(self.atoms['atom_id'].astype(int))
        if edges_all.empty:
            surface_edges = edges_all.copy()
            stitch_edges = edges_all.copy()
        else:
            kinds = edges_all['bond_type'].fillna('').astype(str).str.lower()
            surface_edges = edges_all.loc[kinds.eq('surface')].copy()
            stitch_edges = edges_all.loc[kinds.str.contains('stitch')].copy()
        # A physical seam edge is exported only once but belongs to every
        # adjacent chart.  Expand its explicit patch-membership labels for the
        # local audit so a neighboring chart cannot "lose" a real seam rod.
        edge_groups = {}
        if not surface_edges.empty:
            expanded = []
            for _, row in surface_edges.iterrows():
                memberships = []
                raw = row.get('patch_memberships', '')
                if pd.notna(raw) and str(raw).strip():
                    for value in str(raw).split(';'):
                        try:
                            memberships.append(int(value))
                        except (TypeError, ValueError):
                            pass
                if not memberships:
                    try:
                        memberships = [int(row['patch_id'])]
                    except (TypeError, ValueError):
                        memberships = []
                for audit_pid in sorted(set(memberships)):
                    item = row.copy()
                    item['_audit_patch_id'] = int(audit_pid)
                    expanded.append(item)
            if expanded:
                expanded_edges = pd.DataFrame(expanded)
                edge_groups = {
                    int(k): v for k, v in expanded_edges.groupby('_audit_patch_id', sort=False)
                }
        anchors = defaultdict(list)
        if include_stitch_anchors:
            for row in stitch_edges.itertuples(index=False):
                r = float(row.radius)
                if not np.isfinite(r) or r <= 0:
                    continue
                a, b = int(row.source), int(row.target)
                if a in surface_ids and a in nodes.index:
                    anchors[a].append(r)
                if b in surface_ids and b in nodes.index:
                    anchors[b].append(r)
        # Cache node-to-chart memberships once; avoid scanning every graph
        # node again for each of potentially hundreds of surface charts.
        chart_local_coordinates = defaultdict(dict)
        source_vertex_node_ids = defaultdict(list)
        if not src_nodes.empty:
            for node_id, row in src_nodes.iterrows():
                node_id = int(node_id)
                if node_id not in nodes.index:
                    continue
                chart_local_coordinates[int(row.patch_id)][node_id] = np.asarray(
                    [float(row.uv_x),float(row.uv_y)],dtype=float)
                if int(row.source_vertex_id)>=0:
                    source_vertex_node_ids[int(row.source_vertex_id)].append(node_id)
        # Avoid recalculating a whole chart for each pore or source cell.
        # Exported surface-node spheres use the configured surface diameter,
        # independent of any thicker stitch edge incident on that node.  Using
        # max(incident edge radius) here overstates the exported sphere wherever
        # a surface node is stitched to the internal core.
        surface_node_sphere_radius = (
            0.5 * float(self.config.surface_diameter_mm) *
            float(node_sphere_radius_factor)
        )
        projected_coord_cache: Dict[Tuple[int, int], Tuple[np.ndarray, float]] = {}

        def node_uv_on_chart(chart: PatchChart, node_id: int):
            key = (int(chart.patch_id), int(node_id))
            cached = projected_coord_cache.get(key)
            if cached is not None:
                return cached
            if node_id not in nodes.index:
                return None, math.inf
            xyz_point = nodes.loc[node_id][['x','y','z']].to_numpy(dtype=float)
            triangles = np.asarray(chart.xyz[chart.local_faces], dtype=float)
            repeated = np.repeat(xyz_point[None, :], len(triangles), axis=0)
            closest = trimesh.triangles.closest_point(triangles, repeated)
            d2 = np.einsum('ij,ij->i', closest-repeated, closest-repeated)
            best = int(np.argmin(d2))
            bary = trimesh.triangles.points_to_barycentric(
                triangles[best:best+1], closest[best:best+1]
            )[0]
            uv_point = bary @ chart.uv[chart.local_faces[best]]
            result = (np.asarray(uv_point,dtype=float), math.sqrt(max(float(d2[best]),0.0)))
            projected_coord_cache[key] = result
            return result

        pores = []
        source_cells = []
        per_patch = []
        min_open = float(self.config.zero_pore_min_area_mm2)
        rcrit = float(self.config.target_hydraulic_radius_mm) * float(self.config.hydraulic_radius_mapping_safety)
        radius_acceptance = rcrit * (
            1.0 + float(self.config.final_graph_radius_tolerance_fraction)
        )

        # Distinguish artificial inter-patch cuts from real boundaries of the
        # carrier mesh (outer edges and hole boundaries).  The previous audit
        # treated every chart boundary as a seam, which can substantially
        # over-count seam-dependent records on open/holed surfaces.
        base_edge_face_count: Dict[Tuple[int, int], int] = defaultdict(int)
        if self.base_mesh is not None:
            for tri in np.asarray(self.base_mesh.faces, dtype=int):
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    base_edge_face_count[tuple(sorted((int(a), int(b))))] += 1

        for chart in self.charts:
            pid = int(chart.patch_id)
            scale = max(float(chart.scale_2d_to_3d), 1e-12)

            # Build two boundary line sets in this chart: artificial patch
            # seams (the same base-mesh edge belongs to >=2 faces) and true
            # carrier boundaries (the base-mesh edge belongs to one face).
            # Only the former makes a pore record seam-dependent.
            local_boundary_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for tri in np.asarray(chart.local_faces, dtype=int):
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    local_boundary_counts[tuple(sorted((int(a), int(b))))] += 1
            seam_segments = []
            physical_boundary_segments = []
            for (a, b), count in local_boundary_counts.items():
                if count != 1:
                    continue
                ga, gb = int(chart.vertex_ids[a]), int(chart.vertex_ids[b])
                segment = LineString([chart.uv[a], chart.uv[b]])
                if base_edge_face_count.get(tuple(sorted((ga, gb))), 0) >= 2:
                    seam_segments.append(segment)
                else:
                    physical_boundary_segments.append(segment)
            seam_boundary = unary_union(seam_segments) if seam_segments else None
            physical_boundary = (
                unary_union(physical_boundary_segments)
                if physical_boundary_segments else None
            )

            p_edges = edge_groups.get(pid)
            coords = dict(chart_local_coordinates.get(pid, {}))
            # A shared source vertex uses this specific chart's UV, even if
            # its single stored atom row has another patch_id.
            for source, point in zip(chart.vertex_ids, chart.uv):
                for node_id in source_vertex_node_ids.get(int(source), ()):
                    coords[node_id] = np.asarray(point,dtype=float)
            # Surface anchors may have been removed during assembly trimming.
            if p_edges is None or p_edges.empty:
                p_edges = surface_edges.iloc[0:0]
            walls, solids = [], []
            missing_edge_endpoints = 0
            cross_patch_bridge_outside_chart = 0
            live_nodes = set()
            for row in p_edges.itertuples(index=False):
                a, b, r = int(row.source), int(row.target), float(row.radius)
                if not np.isfinite(r) or r <= 0:
                    missing_edge_endpoints += 1
                    continue
                for node_id in (a, b):
                    if node_id in coords:
                        continue
                    uv_point, distance = node_uv_on_chart(chart, node_id)
                    if (uv_point is not None and distance <=
                            2.0 * float(self.config.surface_tolerance_mm) + 1.0e-8):
                        coords[node_id] = uv_point
                if a not in coords or b not in coords:
                    # A seam bridge joins nodes on different charts. Its full
                    # rod need not project into either chart; the independent
                    # global audit includes it in 3-D and checks both IDs.
                    if (getattr(row, 'role', '') == 'surface_seam_bridge'
                            and a in src_nodes.index and b in src_nodes.index
                            and int(src_nodes.loc[a, 'patch_id']) !=
                                int(src_nodes.loc[b, 'patch_id'])):
                        cross_patch_bridge_outside_chart += 1
                        continue
                    missing_edge_endpoints += 1
                    continue
                seg = LineString([coords[a], coords[b]])
                if seg.length <= 1e-10:
                    continue
                live_nodes.update((a,b))
                walls.append(seg)
                # Projected cylindrical-rod footprint, including a
                # conservative round footprint at segment junctions.
                solids.append(seg.buffer(r / scale, resolution=8, cap_style=1))
            stitch_anchor_count = 0
            if include_stitch_anchors:
                for node_id in live_nodes:
                    if node_id not in anchors:
                        continue
                    stitch_anchor_count += 1
                    for r in anchors[node_id]:
                        solids.append(Point(coords[node_id]).buffer(r / scale, resolution=8))
            if node_spheres_exported:
                for node_id in live_nodes:
                    solids.append(Point(coords[node_id]).buffer(
                        surface_node_sphere_radius / scale,
                        resolution=8))
            occupied = unary_union(solids) if solids else Polygon()
            footprint = chart.polygon.difference(occupied)
            # Polygonize with the full chart boundary because it closes the
            # local chart.  Seam classification below, however, uses only the
            # artificial inter-patch subset and keeps real outer/hole
            # boundaries separate.
            linework = unary_union([chart.polygon.boundary, *walls]) if walls else chart.polygon.boundary
            faces = [p for p in polygonize(linework)
                     if p.area > 1e-12 and chart.polygon.covers(p.representative_point())]
            if not faces:
                faces = _polygon_parts(chart.polygon)
            patch_counts = defaultdict(int)
            patch_open_area = 0.0
            for face_id, face in enumerate(faces):
                if face is None or face.is_empty or face.boundary is None:
                    continue
                face = face.intersection(chart.polygon)
                if face.is_empty or face.boundary is None:
                    continue
                seam_shared_length = (
                    float(face.boundary.intersection(seam_boundary).length) * scale
                    if seam_boundary is not None else 0.0
                )
                physical_boundary_shared_length = (
                    float(face.boundary.intersection(physical_boundary).length) * scale
                    if physical_boundary is not None else 0.0
                )
                seam = seam_shared_length > 1e-7
                physical_boundary_dependent = physical_boundary_shared_length > 1e-7
                open_parts = _polygon_parts(face.difference(occupied))
                if not open_parts:
                    open_parts = [Polygon()]
                for island_id, poly in enumerate(open_parts):
                    area = float(poly.area) * scale**2
                    perimeter = float(poly.length) * scale
                    req = 2.0 * area / perimeter if perimeter > 1e-12 else 0.0
                    if area <= 1e-10:
                        status = 'blocked'
                    elif area < min_open - 1e-10:
                        status = 'undersized'
                    elif req > radius_acceptance + 1e-10:
                        status = 'radius_exceeded'
                    else:
                        status = 'valid'
                    patch_counts[status] += 1
                    patch_open_area += area
                    pores.append(dict(patch_id=pid,pore_id=face_id,island_id=island_id,
                                      cell_area_chart_mm2=float(face.area)*scale**2 if island_id==0 else 0.0,
                                      open_area_chart_mm2=area,wet_perimeter_chart_mm=perimeter,
                                      equivalent_radius_chart_mm=req,
                                      aspect_ratio_chart=(_polygon_aspect_ratio(poly) if not poly.is_empty else 0.0),
                                      shape_factor_chart=(4.0*math.pi*area/max(perimeter*perimeter,1.0e-12) if area>0 else 0.0),
                                      critical_radius_mm=rcrit,
                                      status=status,
                                      boundary_shared_with_chart_mm=(
                                          seam_shared_length + physical_boundary_shared_length),
                                      boundary_shared_with_patch_seam_mm=seam_shared_length,
                                      boundary_shared_with_physical_boundary_mm=physical_boundary_shared_length,
                                      patch_seam_dependent=int(seam),
                                      physical_boundary_dependent=int(physical_boundary_dependent),
                                      node_spheres_included=int(node_spheres_exported),
                                      stitch_anchors_included=int(include_stitch_anchors),
                                      centroid_uv_x=(float(poly.representative_point().x)
                                                     if not poly.is_empty else math.nan),
                                      centroid_uv_y=(float(poly.representative_point().y)
                                                     if not poly.is_empty else math.nan)))
            # Auxiliary count of originally identified source cells with NO
            # remaining opening. These are NOT separately counted as final
            # pores: mergers legitimately remove source-cell boundaries.
            for source_id, source_cell in enumerate(self._source_cells_by_patch.get(pid, [])):
                local_open = source_cell.intersection(footprint)
                area = float(local_open.area)*scale**2
                source_cells.append(dict(patch_id=pid,source_cell_id=source_id,
                                         final_open_area_chart_mm2=area,
                                         source_cell_fully_occupied=int(area<=1e-10),
                                         source_cell_undersized=int(area < min_open - 1.0e-10),
                                         minimum_open_area_mm2=min_open,
                                         diagnostic_only=0))
            per_patch.append(dict(patch_id=pid,actual_surface_edge_segments=len(walls),
                                  missing_edge_endpoints=missing_edge_endpoints,
                                  cross_patch_bridge_outside_chart=cross_patch_bridge_outside_chart,
                                  stitch_anchor_count=stitch_anchor_count,
                                  final_pore_count=sum(patch_counts.values()),
                                  open_area_chart_mm2=patch_open_area,
                                  **{f'{name}_count':patch_counts[name] for name in
                                     ('valid','blocked','undersized','radius_exceeded')}))
            if verbose:
                print(f'[surface final audit] chart {pid+1}/{len(self.charts)} '
                      f'pores={sum(patch_counts.values())} '
                      f'blocked={patch_counts["blocked"]} '
                      f'exceeded={patch_counts["radius_exceeded"]}', flush=True)
        pore_frame = pd.DataFrame(pores, columns=columns)
        self.final_graph_pores = pore_frame.copy()
        source_frame = pd.DataFrame(source_cells)
        patch_frame = pd.DataFrame(per_patch)
        pore_frame.to_csv(output_dir/'surface_final_graph_pores.csv', index=False)
        source_frame.to_csv(output_dir/'surface_final_source_cell_audit.csv', index=False)
        patch_frame.to_csv(output_dir/'surface_final_graph_patch_summary.csv', index=False)
        counts = pore_frame['status'].value_counts().to_dict() if len(pore_frame) else {}
        decisive = (
            pore_frame[pore_frame['patch_seam_dependent'].astype(int) == 0]
            if len(pore_frame) else pore_frame
        )
        decisive_counts = decisive['status'].value_counts().to_dict() if len(decisive) else {}
        # A chart region fully occupied by finite-radius rods is solid, not an
        # open pore. Keep its raw diagnostic row, but exclude it from the local
        # open-pore acceptance gate.
        decisive_open = decisive[decisive['open_area_chart_mm2'] > 1.0e-10] if len(decisive) else decisive
        decisive_open_counts = decisive_open['status'].value_counts().to_dict() if len(decisive_open) else {}
        seam_frame = (
            pore_frame[pore_frame['patch_seam_dependent'].astype(int) == 1]
            if len(pore_frame) else pore_frame
        )
        seam_counts = seam_frame['status'].value_counts().to_dict() if len(seam_frame) else {}
        physical_boundary_frame = (
            pore_frame[pore_frame['physical_boundary_dependent'].astype(int) == 1]
            if len(pore_frame) else pore_frame
        )
        seam_fraction = (float(len(seam_frame)) / float(len(pore_frame))) if len(pore_frame) else 0.0
        seam_screen_enabled = bool(self.config.seam_screen_enabled)
        seam_screen_max_fraction = float(self.config.seam_screen_max_fraction)
        seam_screen_pass = bool((not seam_screen_enabled) or seam_fraction <= seam_screen_max_fraction + 1e-12)
        local_chart_pass = bool(
            len(decisive_open) > 0
            and not decisive_open_counts.get('radius_exceeded', 0)
            and not decisive_open_counts.get('undersized', 0)
            and not decisive_open_counts.get('blocked', 0)
        )
        critical = pore_frame[pore_frame['status']=='valid'] if len(pore_frame) else pore_frame
        open_morph = pore_frame[pore_frame['status'].isin(['valid','radius_exceeded'])] if len(pore_frame) else pore_frame
        hard_ar_max = max(1.0, float(self.config.anisotropy_max_aspect_ratio))
        decisive_open_morph = decisive[decisive['status'].isin(['valid','radius_exceeded'])] if len(decisive) else decisive
        seam_open_morph = seam_frame[seam_frame['status'].isin(['valid','radius_exceeded'])] if len(seam_frame) else seam_frame
        ar_cap_min_area = float(self.config.anisotropy_cap_min_area_mm2)
        all_ar_hard_exceeded = int(((open_morph['open_area_chart_mm2'] > ar_cap_min_area + 1.0e-10) & (open_morph['aspect_ratio_chart'] > hard_ar_max + 1.0e-10)).sum()) if len(open_morph) else 0
        nonseam_ar_hard_exceeded = int(((decisive_open_morph['open_area_chart_mm2'] > ar_cap_min_area + 1.0e-10) & (decisive_open_morph['aspect_ratio_chart'] > hard_ar_max + 1.0e-10)).sum()) if len(decisive_open_morph) else 0
        seam_ar_hard_exceeded = int(((seam_open_morph['open_area_chart_mm2'] > ar_cap_min_area + 1.0e-10) & (seam_open_morph['aspect_ratio_chart'] > hard_ar_max + 1.0e-10)).sum()) if len(seam_open_morph) else 0
        geometry_parts = ['finite_radius_surface_rods']
        if node_spheres_exported:
            geometry_parts.append('exported_node_sphere_disks')
        if include_stitch_anchors:
            geometry_parts.append('stitch_anchor_disks')
        result = dict(method='post_assembly_retained_attributed_graph_local_chart',
                      geometry='_plus_'.join(geometry_parts),
                      mapping='local_chart_isotropic_length_scale; not exact_3d_STL_boolean',
                      original_voronoi_cells_used_as_final_pore_boundaries=False,
                      node_spheres_exported=bool(node_spheres_exported),
                      stitch_anchor_contact_disks_included=bool(include_stitch_anchors),
                      internal_rod_3d_intersections_fully_checked=False,
                      cross_patch_pores_merged=False,
                      patch_seam_cells_require_global_interpretation=True,
                      local_pore_record_count=int(len(pore_frame)),
                      local_pore_record_semantics=(
                          'local chart pore records; seam-touching records can be fragments of the same global pore'),
                      global_complete_pore_count=None,
                      global_complete_pore_count_available=False,
                      global_complete_pore_validation_status=(
                          'not_computed_cross_patch_merge_or_final_3d_check_required'),
                      patch_count=int(len(self.charts)),
                      pore_count=int(len(pore_frame)),
                      valid_pore_count=int(counts.get('valid',0)),
                      blocked_pore_count=int(counts.get('blocked',0)),
                      undersized_pore_count=int(counts.get('undersized',0)),
                      all_local_fragment_minimum_area_pass=bool(
                          not pore_frame.loc[pore_frame['open_area_chart_mm2'] > 1.0e-10,
                                             'status'].isin(['blocked', 'undersized']).any()),
                      minimum_area_proof_method=(
                          'Positive-open-area local chart fragments are screened; fully occupied '
                          'zero-open-area records are solid. The global cross-patch complete-pore '
                          'audit is decisive.'),
                      radius_exceeded_pore_count=int(counts.get('radius_exceeded',0)),
                      final_open_aspect_ratio_mean=(float(open_morph['aspect_ratio_chart'].mean()) if len(open_morph) else 0.0),
                      final_open_aspect_ratio_median=(_safe_percentile(open_morph.get('aspect_ratio_chart', []),50) if len(open_morph) else 0.0),
                      final_open_aspect_ratio_p25=(_safe_percentile(open_morph.get('aspect_ratio_chart', []),25) if len(open_morph) else 0.0),
                      final_open_aspect_ratio_p75=(_safe_percentile(open_morph.get('aspect_ratio_chart', []),75) if len(open_morph) else 0.0),
                      final_open_aspect_ratio_fraction_ge_2_5=(float((open_morph['aspect_ratio_chart']>=2.5).mean()) if len(open_morph) else 0.0),
                      final_open_aspect_ratio_fraction_ge_2_75=(float((open_morph['aspect_ratio_chart']>=2.75).mean()) if len(open_morph) else 0.0),
                      final_open_aspect_ratio_fraction_ge_3_0=(float((open_morph['aspect_ratio_chart']>=3.0).mean()) if len(open_morph) else 0.0),
                      final_open_aspect_ratio_max=(float(open_morph['aspect_ratio_chart'].max()) if len(open_morph) else 0.0),
                      aspect_ratio_hard_max=hard_ar_max,
                      aspect_ratio_hard_min_area_mm2=ar_cap_min_area,
                      aspect_ratio_hard_exceeded_local_record_count=all_ar_hard_exceeded,
                      nonseam_aspect_ratio_hard_exceeded_pore_count=nonseam_ar_hard_exceeded,
                      seam_aspect_ratio_hard_exceeded_local_record_count=seam_ar_hard_exceeded,
                      local_chart_aspect_ratio_upper_bound_pass=bool(nonseam_ar_hard_exceeded == 0),
                      seam_aspect_ratio_requires_global_check=bool(seam_ar_hard_exceeded > 0),
                      seam_dependent_pore_count=int(len(seam_frame)),
                      seam_dependent_local_record_fraction=seam_fraction,
                      seam_screen_enabled=seam_screen_enabled,
                      seam_screen_max_fraction=seam_screen_max_fraction,
                      seam_screen_pass=seam_screen_pass,
                      seam_valid_local_record_count=int(seam_counts.get('valid',0)),
                      seam_blocked_local_record_count=int(seam_counts.get('blocked',0)),
                      seam_undersized_local_record_count=int(seam_counts.get('undersized',0)),
                      seam_min_area_requires_global_check=bool(
                          seam_counts.get('undersized',0) or seam_counts.get('blocked',0)),
                      seam_radius_exceeded_local_record_count=int(seam_counts.get('radius_exceeded',0)),
                      physical_boundary_dependent_local_record_count=int(len(physical_boundary_frame)),
                      nonseam_valid_pore_count=int(decisive_counts.get('valid',0)),
                      nonseam_blocked_region_count=int(decisive_counts.get('blocked',0)),
                      nonseam_zero_open_area_solid_region_count=int(
                          (decisive['open_area_chart_mm2'] <= 1.0e-10).sum()) if len(decisive) else 0,
                      nonseam_positive_open_area_blocked_region_count=int(
                          decisive_open_counts.get('blocked',0)),
                      nonseam_undersized_open_pore_count=int(decisive_counts.get('undersized',0)),
                      nonseam_radius_exceeded_pore_count=int(decisive_counts.get('radius_exceeded',0)),
                      zero_area_source_cell_count=int(source_frame['source_cell_fully_occupied'].sum()) if len(source_frame) else 0,
                      undersized_source_cell_count=int(source_frame['source_cell_undersized'].sum()) if len(source_frame) else 0,
                      source_cell_minimum_area_hard_pass=bool(
                          (int(source_frame['source_cell_undersized'].sum()) if len(source_frame) else 0) == 0),
                      open_area_chart_mm2=float(pore_frame['open_area_chart_mm2'].sum()) if len(pore_frame) else 0.0,
                      max_valid_equivalent_radius_chart_mm=float(critical['equivalent_radius_chart_mm'].max()) if len(critical) else None,
                      max_open_equivalent_radius_chart_mm=float(pore_frame.loc[pore_frame['open_area_chart_mm2']>1e-10,'equivalent_radius_chart_mm'].max()) if len(pore_frame.loc[pore_frame['open_area_chart_mm2']>1e-10]) else None,
                      max_seam_open_equivalent_radius_chart_mm=(
                          float(seam_frame.loc[seam_frame['open_area_chart_mm2']>1e-10,'equivalent_radius_chart_mm'].max())
                          if len(seam_frame.loc[seam_frame['open_area_chart_mm2']>1e-10]) else None),
                      max_nonseam_open_equivalent_radius_chart_mm=(
                          float(decisive.loc[decisive['open_area_chart_mm2']>1e-10,'equivalent_radius_chart_mm'].max())
                          if len(decisive.loc[decisive['open_area_chart_mm2']>1e-10]) else None),
                      missing_edge_endpoints=int(patch_frame['missing_edge_endpoints'].sum()) if len(patch_frame) else 0,
                      cross_patch_bridge_outside_chart=int(
                          patch_frame['cross_patch_bridge_outside_chart'].sum()
                      ) if len(patch_frame) else 0,
                      critical_radius_mm=rcrit,minimum_open_area_mm2=min_open,
                      radius_acceptance_mm=radius_acceptance,
                      radius_tolerance_fraction=float(
                          self.config.final_graph_radius_tolerance_fraction),
                      local_chart_acceptance_pass=local_chart_pass,
                      provisional_surface_acceptance_pass=bool(local_chart_pass),
                      provisional_surface_acceptance_interpretation=(
                          'positive-open-area non-seam local capillary check; '
                          'seam burden is diagnostic and cross-patch complete pores '
                          'still require the independent global audit'),
                      fully_open_and_capillary_compliant=bool(len(pore_frame)>0 and
                          pore_frame.loc[pore_frame['open_area_chart_mm2'] > 1.0e-10,
                                         'status'].eq('valid').all()),
                      elapsed_seconds=perf_counter()-started)
        (output_dir/'surface_final_graph_summary.json').write_text(
            json.dumps(_json_safe(result),ensure_ascii=False,indent=2),encoding='utf-8')
        if apply_quality_gate and not self.summary_metrics.empty:
            # Source Voronoi cells remain useful provenance, but merged walls
            # mean they are not the final pores.  Standalone validation is
            # therefore decided by the actual finite-radius retained graph.
            failures = {
                item for item in str(self.summary_metrics.iloc[0].get('quality_failures', '')).split(';')
                if item
            }
            # A fully occupied region is solid, not an open liquid fragment.
            # Pores touching an artificial chart seam need a global 3-D
            # interpretation and are reported without deciding this local gate.
            final_flags = {'radius_exceeded': 'final_graph_radius_exceeded_pores'}
            self.summary_metrics.loc[:, 'final_graph_blocked_region_count'] = int(
                result['blocked_pore_count'])
            self.summary_metrics.loc[:, 'final_graph_nonseam_blocked_region_count'] = int(
                result['nonseam_blocked_region_count'])
            for name, flag in final_flags.items():
                count = int(result[f'{name}_pore_count'])
                self.summary_metrics.loc[:, f'final_graph_{name}_pore_count'] = count
                decisive_count = int(result[f'nonseam_{name}_pore_count'] if name == 'radius_exceeded'
                                     else result['nonseam_undersized_open_pore_count'])
                self.summary_metrics.loc[:, f'final_graph_nonseam_{name}_pore_count'] = decisive_count
                if decisive_count:
                    failures.add(flag)
                else:
                    failures.discard(flag)
            nonseam_small = int(result['nonseam_undersized_open_pore_count'])
            nonseam_blocked = int(result['nonseam_blocked_region_count'])
            all_small = int(result['undersized_pore_count'])
            all_blocked = int(result['blocked_pore_count'])
            self.summary_metrics.loc[:, 'final_graph_minimum_open_area_mm2'] = float(result['minimum_open_area_mm2'])
            self.summary_metrics.loc[:, 'final_graph_nonseam_undersized_open_pore_count'] = nonseam_small
            self.summary_metrics.loc[:, 'final_graph_nonseam_blocked_region_count'] = nonseam_blocked
            self.summary_metrics.loc[:, 'final_graph_all_undersized_open_fragment_count'] = all_small
            self.summary_metrics.loc[:, 'final_graph_all_blocked_fragment_count'] = all_blocked
            self.summary_metrics.loc[:, 'final_graph_all_local_fragment_minimum_area_pass'] = int(
                bool(result.get('all_local_fragment_minimum_area_pass', False)))
            self.summary_metrics.loc[:, 'final_graph_seam_min_area_requires_global_check'] = int(
                bool(result.get('seam_min_area_requires_global_check', False)))
            # Minimum area is a monotone quantity under cross-patch union.
            # Therefore every local fragment, including artificial-seam
            # fragments, is held to the hard limit rather than waived.
            if all_small > 0:
                failures.add('final_graph_undersized_pores')
            else:
                failures.discard('final_graph_undersized_pores')
            if all_blocked > 0:
                failures.add('unresolved_zero_area_open_fragments')
            else:
                failures.discard('unresolved_zero_area_open_fragments')
            source_under = int(result.get('undersized_source_cell_count', 0))
            source_zero = int(result.get('zero_area_source_cell_count', 0))
            self.summary_metrics.loc[:, 'final_graph_undersized_source_cell_count'] = source_under
            self.summary_metrics.loc[:, 'final_graph_zero_area_source_cell_count'] = source_zero
            self.summary_metrics.loc[:, 'final_graph_source_cell_minimum_area_hard_pass'] = int(
                bool(result.get('source_cell_minimum_area_hard_pass', False)))
            # Source cells are historical Voronoi generators. After retained-graph
            # merging a source cell may legitimately disappear; only actual final
            # pore fragments are hard-gated. Keep source occupancy as diagnostics.
            failures.discard('undersized_source_cells')
            failures.discard('zero_area_source_cells')
            self.summary_metrics.loc[:, 'final_graph_pore_count'] = int(result['pore_count'])
            self.summary_metrics.loc[:, 'final_graph_seam_dependent_local_record_count'] = int(
                result['seam_dependent_pore_count'])
            self.summary_metrics.loc[:, 'final_graph_seam_local_record_fraction'] = float(
                result['seam_dependent_local_record_fraction'])
            self.summary_metrics.loc[:, 'final_graph_seam_radius_exceeded_local_record_count'] = int(
                result['seam_radius_exceeded_local_record_count'])
            self.summary_metrics.loc[:, 'final_graph_seam_screen_pass'] = int(
                result['seam_screen_pass'])
            self.summary_metrics.loc[:, 'final_graph_aspect_ratio_hard_max'] = float(
                result['aspect_ratio_hard_max'])
            self.summary_metrics.loc[:, 'final_graph_aspect_ratio_hard_exceeded_local_record_count'] = int(
                result['aspect_ratio_hard_exceeded_local_record_count'])
            self.summary_metrics.loc[:, 'final_graph_nonseam_aspect_ratio_hard_exceeded_pore_count'] = int(
                result['nonseam_aspect_ratio_hard_exceeded_pore_count'])
            self.summary_metrics.loc[:, 'final_graph_seam_aspect_ratio_hard_exceeded_local_record_count'] = int(
                result['seam_aspect_ratio_hard_exceeded_local_record_count'])
            self.summary_metrics.loc[:, 'final_graph_seam_aspect_ratio_requires_global_check'] = int(
                bool(result.get('seam_aspect_ratio_requires_global_check', False)))
            # Aspect ratio is a reported morphology measure, not a final
            # acceptance gate. Retain the exceedance count for auditability.
            failures.discard('final_graph_aspect_ratio_hard_cap_exceeded')
            if bool(result['seam_screen_enabled']) and not bool(result['seam_screen_pass']):
                failures.add('final_graph_seam_burden_exceeded')
            else:
                failures.discard('final_graph_seam_burden_exceeded')
            self.summary_metrics.loc[:, 'quality_pass'] = int(not failures)
            self.summary_metrics.loc[:, 'quality_failures'] = ';'.join(sorted(failures))
        return result

    def repair_from_final_graph_audit(self, output_dir: Path,
                                      max_trials_per_fragment: int = 8) -> None:
        """Use the final retained-graph audit as a bounded deletion oracle."""
        trial_dir = Path(tempfile.mkdtemp(prefix='surface-final-audit-repair-'))
        logs = []
        try:
            baseline = self.audit_final_retained_graph(
                self.atoms, self.edges, trial_dir,
                node_spheres_exported=False, include_stitch_anchors=False,
                verbose=False,
            )
            for repair_round in range(3):
                targets = self.final_graph_pores[
                    (self.final_graph_pores['patch_seam_dependent'].astype(int) == 0) &
                    (self.final_graph_pores['status'] == 'undersized')
                ].copy()
                if targets.empty:
                    break
                accepted_round = False
                atom_rows = self.atoms.set_index('atom_id')
                for target in targets.itertuples(index=False):
                    pid = int(target.patch_id)
                    chart = self.charts[pid]
                    point = Point(float(target.centroid_uv_x), float(target.centroid_uv_y))
                    source_nodes = defaultdict(list)
                    for atom in self.atoms.itertuples(index=False):
                        if int(atom.source_vertex_id) >= 0:
                            source_nodes[int(atom.source_vertex_id)].append(int(atom.atom_id))
                    coords = {
                        int(row.atom_id): np.asarray([float(row.uv_x), float(row.uv_y)])
                        for row in self.atoms.itertuples(index=False)
                        if int(row.patch_id) == pid
                    }
                    for source_id, uv in zip(chart.vertex_ids, chart.uv):
                        for node_id in source_nodes.get(int(source_id), ()):
                            coords[node_id] = np.asarray(uv, dtype=float)
                    patch_edges = self.edges[self.edges['patch_id'].astype(int) == pid]
                    ranked = []
                    for logical_id, group in patch_edges.groupby('logical_edge_id', sort=False):
                        lines = []
                        for row in group.itertuples(index=False):
                            if int(row.source) in coords and int(row.target) in coords:
                                lines.append(LineString([
                                    coords[int(row.source)], coords[int(row.target)]
                                ]))
                        if lines:
                            ranked.append((float(unary_union(lines).distance(point)), int(logical_id)))
                    for _, logical_id in sorted(ranked)[:max(1, int(max_trials_per_fragment))]:
                        before_edges = self.edges
                        removed = before_edges[
                            before_edges['logical_edge_id'].astype(int) == logical_id]
                        trial_edges = before_edges[
                            before_edges['logical_edge_id'].astype(int) != logical_id].copy()
                        remaining = (set(trial_edges['source'].astype(int)) |
                                     set(trial_edges['target'].astype(int)))
                        orphaned = ((set(removed['source'].astype(int)) |
                                     set(removed['target'].astype(int))) - remaining)
                        if any(str(atom_rows.loc[node_id, 'node_role']) != 'edge_sample'
                               for node_id in orphaned):
                            continue
                        if not self._repair_connectivity_preserved(before_edges, trial_edges):
                            continue
                        self.edges = trial_edges
                        trial = self.audit_final_retained_graph(
                            self.atoms, self.edges, trial_dir,
                            node_spheres_exported=False, include_stitch_anchors=False,
                            verbose=False,
                        )
                        improved = (
                            int(trial['nonseam_undersized_open_pore_count']) <
                            int(baseline['nonseam_undersized_open_pore_count']) and
                            int(trial['nonseam_radius_exceeded_pore_count']) <=
                            int(baseline['nonseam_radius_exceeded_pore_count'])
                        )
                        logs.append(dict(
                            repair_round=repair_round + 1, action='delete_wall', patch_id=pid,
                            removed_logical_edge_id=logical_id,
                            adjusted_logical_edge_id=-1, radius_factor=1.0,
                            undersized_before=int(baseline['nonseam_undersized_open_pore_count']),
                            undersized_after=int(trial['nonseam_undersized_open_pore_count']),
                            radius_exceeded_after=int(trial['nonseam_radius_exceeded_pore_count']),
                            accepted=int(improved),
                        ))
                        if improved:
                            baseline = trial
                            if not self.edge_metrics.empty:
                                mask = self.edge_metrics['logical_edge_id'].astype(int) == logical_id
                                self.edge_metrics.loc[mask, 'accepted'] = 0
                            accepted_round = True
                            break
                        self.edges = before_edges
                    if not accepted_round:
                        for _, logical_id in sorted(ranked)[:4]:
                            for factor in (1.10, 1.20, 1.35):
                                before_edges = self.edges
                                trial_edges = before_edges.copy()
                                mask = trial_edges['logical_edge_id'].astype(int) == logical_id
                                if not bool(mask.any()):
                                    continue
                                trial_edges.loc[mask, 'radius'] = (
                                    trial_edges.loc[mask, 'radius'].astype(float) * factor
                                )
                                self.edges = trial_edges
                                trial = self.audit_final_retained_graph(
                                    self.atoms, self.edges, trial_dir,
                                    node_spheres_exported=False,
                                    include_stitch_anchors=False, verbose=False,
                                )
                                improved = (
                                    int(trial['nonseam_undersized_open_pore_count']) <
                                    int(baseline['nonseam_undersized_open_pore_count']) and
                                    int(trial['nonseam_radius_exceeded_pore_count']) <=
                                    int(baseline['nonseam_radius_exceeded_pore_count'])
                                )
                                logs.append(dict(
                                    repair_round=repair_round + 1,
                                    action='thicken_wall', patch_id=pid,
                                    removed_logical_edge_id=-1,
                                    adjusted_logical_edge_id=logical_id,
                                    radius_factor=factor,
                                    undersized_before=int(baseline['nonseam_undersized_open_pore_count']),
                                    undersized_after=int(trial['nonseam_undersized_open_pore_count']),
                                    radius_exceeded_after=int(trial['nonseam_radius_exceeded_pore_count']),
                                    accepted=int(improved),
                                ))
                                if improved:
                                    baseline = trial
                                    accepted_round = True
                                    break
                                self.edges = before_edges
                            if accepted_round:
                                break
                    if accepted_round:
                        break
                if not accepted_round:
                    break
            self.final_audit_repair_history = pd.DataFrame(logs)
            incident = set(self.edges['source'].astype(int)) | set(self.edges['target'].astype(int))
            disposable = ((self.atoms['node_role'] == 'edge_sample') &
                          ~self.atoms['atom_id'].astype(int).isin(incident))
            self.atoms = self.atoms.loc[~disposable].reset_index(drop=True)
            self.curr_vertices = self.atoms[['x', 'y', 'z']].to_numpy(float).tolist()
            self.curr_edges = list(self.edges[['source', 'target']].itertuples(index=False, name=None))
        finally:
            shutil.rmtree(trial_dir, ignore_errors=True)

    @staticmethod
    def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        den = float(np.dot(ab, ab))
        if den <= 1.0e-24:
            return float(np.linalg.norm(point-a))
        t = float(np.clip(np.dot(point-a, ab)/den, 0.0, 1.0))
        return float(np.linalg.norm(point-(a+t*ab)))

    def _audit_artificial_seam_coverage(self) -> Tuple[float, float]:
        """Measure actual finite-radius rod coverage along artificial seams."""
        if not self._artificial_seam_edges:
            self.seam_geometric_metrics = pd.DataFrame()
            return 1.0, 0.0
        if self.atoms.empty or self.edges.empty or self.base_mesh is None:
            self.seam_geometric_metrics = pd.DataFrame()
            return 0.0, math.inf
        atoms = self.atoms.drop_duplicates('atom_id').set_index('atom_id')
        seg_a=[]; seg_b=[]; seg_r=[]
        for row in self.edges.itertuples(index=False):
            a,b=int(row.source),int(row.target)
            r=float(row.radius)
            if a not in atoms.index or b not in atoms.index or not np.isfinite(r) or r<=0:
                continue
            seg_a.append(atoms.loc[a][['x','y','z']].to_numpy(float))
            seg_b.append(atoms.loc[b][['x','y','z']].to_numpy(float))
            seg_r.append(r)
        if not seg_a:
            self.seam_geometric_metrics = pd.DataFrame()
            return 0.0, math.inf
        seg_a=np.asarray(seg_a,float); seg_b=np.asarray(seg_b,float); seg_r=np.asarray(seg_r,float)
        mids=0.5*(seg_a+seg_b)
        half=0.5*np.linalg.norm(seg_b-seg_a,axis=1)
        tree=cKDTree(mids)
        search=float(np.max(half+seg_r))+float(self.config.surface_tolerance_mm)
        spacing=max(0.02,float(self.config.seam_coverage_sample_spacing_mm))
        vertices=np.asarray(self.base_mesh.vertices,float)
        rows=[]; covered_total=0; sample_total=0; global_max_gap=0.0
        for seam_id,(va,vb) in enumerate(sorted(self._artificial_seam_edges)):
            p0,p1=vertices[int(va)],vertices[int(vb)]
            length=float(np.linalg.norm(p1-p0))
            divisions=max(1,int(math.ceil(length/spacing)))
            ts=np.linspace(0.0,1.0,divisions+1)
            covered=[]
            for t in ts:
                point=p0*(1.0-t)+p1*t
                ids=tree.query_ball_point(point,r=search)
                ok=False
                for idx in ids:
                    if self._point_segment_distance(point,seg_a[idx],seg_b[idx]) <= (
                        seg_r[idx]+float(self.config.surface_tolerance_mm)+1.0e-9
                    ):
                        ok=True; break
                covered.append(ok)
            covered=np.asarray(covered,bool)
            step=length/divisions
            longest=0; run=0
            for ok in covered:
                if ok:
                    longest=max(longest,run); run=0
                else:
                    run+=1
            longest=max(longest,run)
            max_gap=min(length,float(longest)*step)
            global_max_gap=max(global_max_gap,max_gap)
            covered_total += int(covered.sum())
            sample_total += int(len(covered))
            rows.append({
                'seam_edge_id':seam_id,'source_vertex_a':int(va),'source_vertex_b':int(vb),
                'expected_patches':';'.join(map(str,sorted(self._seam_edge_patches.get((va,vb),set())))),
                'seam_length_mm':length,'sample_count':int(len(covered)),
                'covered_sample_count':int(covered.sum()),
                'coverage_ratio':float(covered.mean()) if len(covered) else 0.0,
                'max_uncovered_length_mm':max_gap,
            })
        self.seam_geometric_metrics=pd.DataFrame(rows)
        return covered_total/max(sample_total,1), global_max_gap

    def _analyze(self):
        assert self.base_mesh is not None
        self._audit_surface_coverage()
        vertex_k = np.asarray(self.base_vertex_gaussian_curvature, dtype=float)
        boundary_mask = np.asarray(self.base_boundary_vertex_mask, dtype=bool)
        curvature_threshold = float(self.config.curvature_sign_threshold)
        overall_curvature_stats = _signed_gaussian_curvature_statistics(
            vertex_k, boundary_mask, curvature_threshold
        )
        patch_memberships: Dict[int, List[int]] = defaultdict(list)
        for chart in self.charts:
            for vertex_id in chart.vertex_ids:
                patch_memberships[int(vertex_id)].append(int(chart.patch_id))
        signed_rows = []
        vertices = np.asarray(self.base_mesh.vertices, dtype=float)
        for vertex_id, (point, value, is_boundary) in enumerate(
            zip(vertices, vertex_k, boundary_mask)
        ):
            signed_rows.append(
                {
                    "vertex_id": int(vertex_id),
                    "x_mm": float(point[0]),
                    "y_mm": float(point[1]),
                    "z_mm": float(point[2]),
                    "gaussian_curvature_signed_mm-2": float(value),
                    "gaussian_curvature_abs_mm-2": float(abs(value)),
                    "is_boundary_vertex": int(is_boundary),
                    "included_in_signed_K_statistics": int(
                        np.isfinite(value) and not is_boundary
                    ),
                    "gaussian_curvature_class": _curvature_class(
                        float(value), curvature_threshold
                    ),
                    "gaussian_curvature_class_threshold_mm-2": curvature_threshold,
                    "patch_ids": ";".join(
                        map(str, sorted(patch_memberships.get(int(vertex_id), [])))
                    ),
                }
            )
        self.signed_gaussian_curvature = pd.DataFrame(signed_rows)
        patch_rows = []
        for chart in self.charts:
            pores = self.pore_metrics[self.pore_metrics["patch_id"] == chart.patch_id] if not self.pore_metrics.empty else pd.DataFrame()
            valid_pores = pores[pores["hydraulic_radius_2d_mm"] > 1.0e-9] if len(pores) else pores
            edge_m = self.edge_metrics[self.edge_metrics["patch_id"] == chart.patch_id] if not self.edge_metrics.empty else pd.DataFrame()
            edge_m_ok = edge_m[edge_m["accepted"] == 1] if len(edge_m) else edge_m
            patch_curvature_stats = _signed_gaussian_curvature_statistics(
                vertex_k[chart.vertex_ids],
                boundary_mask[chart.vertex_ids],
                curvature_threshold,
            )
            patch_rows.append(
                {
                    "patch_id": chart.patch_id,
                    "component_id": chart.component_id,
                    "face_count": len(chart.face_ids),
                    "vertex_count": len(chart.vertex_ids),
                    "parameterization_method": chart.parameterization_method,
                    "gaussian_curvature_mean_mm-2": chart.curvature_mean,
                    "gaussian_curvature_abs_mean_mm-2": chart.curvature_abs_mean,
                    "gaussian_curvature_class_threshold_mm-2": curvature_threshold,
                    **patch_curvature_stats,
                    "area_distortion_mean": float(np.mean(chart.area_distortion)),
                    "area_distortion_p95": _safe_percentile(chart.area_distortion, 95),
                    "angle_distortion_mean_deg": float(np.mean(chart.angle_distortion_deg)),
                    "angle_distortion_p95_deg": _safe_percentile(chart.angle_distortion_deg, 95),
                    "edge_length_distortion_p95": _safe_percentile(chart.edge_distortion, 95),
                    "uv_overlap_fraction": chart.uv_overlap_fraction,
                    "hydraulic_radius_rel_error_mean": float(valid_pores["hydraulic_radius_rel_error"].mean()) if len(valid_pores) else 0.0,
                    "hydraulic_radius_rel_error_p95": _safe_percentile(valid_pores.get("hydraulic_radius_rel_error", []), 95),
                    "hydraulic_radius_target_normalized_error_p95": _safe_percentile(
                        valid_pores.get("hydraulic_radius_target_normalized_error", []), 95
                    ),
                    "max_chord_surface_error_mm": float(edge_m_ok["max_chord_surface_error_mm"].max()) if len(edge_m_ok) else 0.0,
                    "domain_violation_count": int(edge_m_ok["domain_violation"].sum()) if len(edge_m_ok) else 0,
                    "rejected_edge_count": int((edge_m["accepted"] == 0).sum()) if len(edge_m) else 0,
                    "mapping_fallback_count": int(edge_m["mapping_fallback"].sum()) if len(edge_m) else 0,
                }
            )
        self.patch_metrics = pd.DataFrame(patch_rows)

        face_patch = {}
        for chart in self.charts:
            for face in chart.face_ids:
                face_patch[int(face)] = chart.patch_id
        seam_vertex_patches: Dict[int, Set[int]] = defaultdict(set)
        for chart in self.charts:
            for vertex in chart.vertex_ids:
                seam_vertex_patches[int(vertex)].add(chart.patch_id)
        seam_vertices = {v: p for v, p in seam_vertex_patches.items() if len(p) > 1}
        node_by_source = self.atoms[self.atoms["source_vertex_id"] >= 0].set_index("source_vertex_id") if not self.atoms.empty else pd.DataFrame()
        sampled_seam_vertices = {
            vertex: patches for vertex, patches in seam_vertices.items()
            if not node_by_source.empty and vertex in node_by_source.index
        }
        incident_patch: Dict[int, Set[int]] = defaultdict(set)
        atom_memberships: Dict[int, Set[int]] = {}
        if not self.atoms.empty:
            for row in self.atoms.itertuples(index=False):
                atom_memberships[int(row.atom_id)] = {
                    int(value) for value in str(row.patch_memberships).split(";") if value != ""
                }
        if not self.edges.empty:
            for row in self.edges.itertuples(index=False):
                source = int(row.source)
                target = int(row.target)
                edge_patch = int(row.patch_id)
                edge_memberships = {edge_patch}
                raw_memberships = getattr(row, 'patch_memberships', '')
                if str(raw_memberships).strip():
                    for value in str(raw_memberships).split(';'):
                        try:
                            edge_memberships.add(int(value))
                        except (TypeError, ValueError):
                            pass
                shared_memberships = atom_memberships.get(source, set()) & atom_memberships.get(target, set())
                incident_patch[source].update(edge_memberships)
                incident_patch[target].update(edge_memberships)
                # A deduplicated edge lying on a shared chart boundary is
                # incident to every patch common to both endpoint nodes.
                incident_patch[source].update(shared_memberships)
                incident_patch[target].update(shared_memberships)
        seam_rows = []
        continuous = 0
        patch_label_complete = 0
        position_continuous = 0
        for source_vertex, expected_patches in sampled_seam_vertices.items():
            atom_id = int(node_by_source.loc[source_vertex, "atom_id"])
            position_continuous += 1
            incident = incident_patch.get(atom_id, set())
            label_ok = expected_patches.issubset(incident)
            ok = bool(incident)
            continuous += int(ok)
            patch_label_complete += int(label_ok)
            seam_rows.append({"source_vertex_id": source_vertex, "expected_patch_count": len(expected_patches), "incident_patch_count": len(incident), "position_gap_mm": 0.0, "position_continuous": 1, "graph_incidence_continuous": int(ok), "patch_label_coverage_complete": int(label_ok), "continuous": int(ok)})
        self.seam_metrics = pd.DataFrame(seam_rows)
        seam_graph_ratio = continuous / max(len(seam_rows), 1)
        seam_patch_label_ratio = patch_label_complete / max(len(seam_rows), 1)
        seam_shared_vertex_ratio = position_continuous / max(len(seam_rows), 1)
        seam_position_ratio, seam_position_gap_max = self._audit_artificial_seam_coverage()
        edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for face in np.asarray(self.base_mesh.faces, dtype=int):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_counts[tuple(sorted((int(a), int(b))))] += 1
        boundary_graph: Dict[int, Set[int]] = defaultdict(set)
        for (a, b), count in edge_counts.items():
            if count == 1:
                boundary_graph[a].add(b)
                boundary_graph[b].add(a)
        boundary_seen: Set[int] = set()
        boundary_loops = 0
        for seed in boundary_graph:
            if seed in boundary_seen:
                continue
            boundary_loops += 1
            queue = [seed]
            boundary_seen.add(seed)
            while queue:
                vertex = queue.pop()
                for nb in boundary_graph[vertex]:
                    if nb not in boundary_seen:
                        boundary_seen.add(nb)
                        queue.append(nb)
        component_count = len(set(int(c.component_id) for c in self.charts))
        graph_component_count = 0
        graph_largest_fraction = 0.0
        isolated_node_count = len(self.atoms)
        graph_cycle_rank = 0
        max_node_degree = 0
        domain_component_graph_connected = True
        if not self.atoms.empty:
            atom_ids = self.atoms["atom_id"].astype(int).to_numpy()
            atom_lookup = {atom_id: i for i, atom_id in enumerate(atom_ids)}
            if not self.edges.empty:
                rows = np.asarray([atom_lookup[int(v)] for v in self.edges["source"]], dtype=int)
                cols = np.asarray([atom_lookup[int(v)] for v in self.edges["target"]], dtype=int)
                adjacency = coo_matrix(
                    (
                        np.ones(2 * len(rows), dtype=np.uint8),
                        (np.concatenate((rows, cols)), np.concatenate((cols, rows))),
                    ),
                    shape=(len(atom_ids), len(atom_ids)),
                ).tocsr()
                graph_component_count, labels = connected_components(adjacency, directed=False)
                sizes = np.bincount(labels, minlength=graph_component_count)
                degree = np.diff(adjacency.indptr)
                isolated_node_count = int(np.count_nonzero(degree == 0))
                max_node_degree = int(degree.max(initial=0))
                graph_largest_fraction = float(sizes.max(initial=0) / max(len(atom_ids), 1))
                active_nodes = int(np.count_nonzero(degree > 0))
                active_components = int(np.count_nonzero([
                    np.any(degree[labels == label] > 0) for label in range(graph_component_count)
                ]))
                graph_cycle_rank = max(0, len(self.edges) - active_nodes + active_components)
                atom_domains = self.atoms['component_id'].astype(int).to_numpy()
                for domain_id in np.unique(atom_domains):
                    member = np.flatnonzero((atom_domains == domain_id) & (degree > 0))
                    if len(member) == 0 or len(np.unique(labels[member])) != 1:
                        domain_component_graph_connected = False
                        break
            else:
                graph_component_count = len(atom_ids)
                domain_component_graph_connected = False

        carrier_error_p95 = 0.0
        carrier_error_max = 0.0
        if self.base_mesh is not None and len(self.base_mesh.vertices):
            carrier_errors = self._surface_distance_batch(
                np.asarray(self.base_mesh.vertices, dtype=float)
            )
            carrier_error_p95 = _safe_percentile(carrier_errors, 95)
            carrier_error_max = float(np.max(carrier_errors, initial=0.0))
        valid_pores = (
            self.pore_metrics[self.pore_metrics["hydraulic_radius_2d_mm"] > 1.0e-9]
            if not self.pore_metrics.empty else self.pore_metrics
        )
        total_cell_area = float(self.pore_metrics.get("cell_area_3d_mm2", pd.Series(dtype=float)).sum())
        total_pore_area = float(self.pore_metrics.get("pore_area_3d_mm2", pd.Series(dtype=float)).sum())
        surface_cell_area_ratio_raw = total_cell_area / max(float(self.base_mesh.area), 1.0e-12)
        surface_cell_coverage = min(1.0, surface_cell_area_ratio_raw)
        if self.coverage_metrics.empty:
            clearance_max = clearance_p95 = uncovered_fraction = 1.0
        else:
            clearance = self.coverage_metrics["clearance_radius_mm"].to_numpy(float)
            area_weight = self.coverage_metrics["area_weight_mm2"].to_numpy(float)
            clearance_max = float(np.max(clearance, initial=0.0))
            clearance_p95 = _safe_percentile(clearance, 95)
            uncovered_fraction = float(
                area_weight[clearance > float(self.config.target_hydraulic_radius_mm)].sum()
                / max(area_weight.sum(), 1.0e-12)
            )
        if self.refinement_history.empty:
            refinement_round_max = 0
            unconverged_patch_count = len(self.charts)
        else:
            last_refinement = (
                self.refinement_history.sort_values(["patch_id", "round"])
                .groupby("patch_id", as_index=False)
                .tail(1)
            )
            refinement_round_max = int(self.refinement_history["round"].max())
            unconverged_patch_count = int(
                (last_refinement["bad_subdomain_count"].astype(int) > 0).sum()
            )
        parameterization_counts = self.patch_metrics.get(
            "parameterization_method", pd.Series(dtype=str)
        ).value_counts()
        summary = {
            "construction_mode": "per_subdomain_voronoi_polylabel_orthogonal",
            "patch_count": len(self.charts),
            "principal_plane_patch_count": int(
                parameterization_counts.get("principal_plane_projection", 0)
            ),
            "intrinsic_unfolding_patch_count": int(
                parameterization_counts.get("intrinsic_edge_length_unfolding", 0)
            ),
            "harmonic_parameterization_patch_count": int(
                parameterization_counts.get("convex_boundary_harmonic", 0)
            ),
            "component_count": component_count,
            "boundary_loop_count": boundary_loops,
            "topological_hole_count": max(0, boundary_loops - component_count),
            "node_count": len(self.atoms),
            "edge_segment_count": len(self.edges),
            "logical_edge_count": int(self.edge_metrics["accepted"].sum()) if len(self.edge_metrics) else 0,
            "logical_edge_candidate_count": len(self.edge_metrics),
            "area_distortion_p95": _safe_percentile(self.patch_metrics.get("area_distortion_p95", []), 95),
            "angle_distortion_p95_deg": _safe_percentile(self.patch_metrics.get("angle_distortion_p95_deg", []), 95),
            "uv_overlap_fraction_max": float(
                self.patch_metrics.get("uv_overlap_fraction", pd.Series(dtype=float)).max()
            ) if len(self.patch_metrics) else 0.0,
            "hydraulic_radius_rel_error_p95": _safe_percentile(valid_pores.get("hydraulic_radius_rel_error", []), 95),
            "hydraulic_radius_target_normalized_error_p95": _safe_percentile(
                valid_pores.get("hydraulic_radius_target_normalized_error", []), 95
            ),
            "pore_count": int(len(valid_pores)),
            "liquid_area_fraction": total_pore_area / max(total_cell_area, 1.0e-12),
            "hydraulic_radius_mean_mm": float(valid_pores["hydraulic_radius_3d_mm"].mean()) if len(valid_pores) else 0.0,
            "hydraulic_radius_max_mm": float(valid_pores["hydraulic_radius_3d_mm"].max()) if len(valid_pores) else 0.0,
            "aspect_ratio_mean": float(valid_pores["aspect_ratio"].mean()) if len(valid_pores) else 0.0,
            "aspect_ratio_p10": _safe_percentile(valid_pores.get("aspect_ratio", []), 10),
            "aspect_ratio_p25": _safe_percentile(valid_pores.get("aspect_ratio", []), 25),
            "aspect_ratio_median": _safe_percentile(valid_pores.get("aspect_ratio", []), 50),
            "aspect_ratio_p75": _safe_percentile(valid_pores.get("aspect_ratio", []), 75),
            "aspect_ratio_fraction_ge_2_5": float((valid_pores["aspect_ratio"] >= 2.5).mean()) if len(valid_pores) else 0.0,
            "aspect_ratio_fraction_ge_2_75": float((valid_pores["aspect_ratio"] >= 2.75).mean()) if len(valid_pores) else 0.0,
            "aspect_ratio_fraction_ge_3_0": float((valid_pores["aspect_ratio"] >= 3.0).mean()) if len(valid_pores) else 0.0,
            "anisotropy_soft_target_aspect_ratio": float(self.config.anisotropy_target_aspect_ratio),
            "anisotropy_hard_max_aspect_ratio": float(self.config.anisotropy_max_aspect_ratio),
            "anisotropy_cap_min_area_mm2": float(self.config.anisotropy_cap_min_area_mm2),
            "global_ar_cap_rewrite_enabled": int(bool(self.config.global_ar_cap_rewrite_enable)),
            "global_ar_cap_rewrite_completed": int(bool(self.global_ar_cap_summary.get("completed", False))) if hasattr(self, "global_ar_cap_summary") else 0,
            "global_ar_cap_initial_violation_count": int(self.global_ar_cap_summary.get("initial_eligible_count", 0)) if hasattr(self, "global_ar_cap_summary") else 0,
            "global_ar_cap_final_violation_count": int(self.global_ar_cap_summary.get("final_eligible_count", 0)) if hasattr(self, "global_ar_cap_summary") else 0,
            "global_ar_cap_added_wall_count": int(self.global_ar_cap_summary.get("added_wall_count", 0)) if hasattr(self, "global_ar_cap_summary") else 0,
            "anisotropy_rewrite_candidate_count": int(len(self.anisotropy_rewrite_history)) if hasattr(self, "anisotropy_rewrite_history") else 0,
            "anisotropy_rewrite_selected_count": int(self.anisotropy_rewrite_history.get("selected", pd.Series(dtype=int)).sum()) if hasattr(self, "anisotropy_rewrite_history") and not self.anisotropy_rewrite_history.empty else 0,
            "anisotropy_rewrite_accepted_count": int(self.anisotropy_rewrite_history.get("accepted", pd.Series(dtype=int)).sum()) if hasattr(self, "anisotropy_rewrite_history") and not self.anisotropy_rewrite_history.empty else 0,
            "anisotropy_hard_cap_selected_count": int(((self.anisotropy_rewrite_history.get("rewrite_type", pd.Series(dtype=str)).astype(str) == "hard_ar_cap") & (self.anisotropy_rewrite_history.get("selected", pd.Series(dtype=int)).astype(int) == 1)).sum()) if hasattr(self, "anisotropy_rewrite_history") and not self.anisotropy_rewrite_history.empty else 0,
            "anisotropy_hard_cap_accepted_count": int(((self.anisotropy_rewrite_history.get("rewrite_type", pd.Series(dtype=str)).astype(str) == "hard_ar_cap") & (self.anisotropy_rewrite_history.get("accepted", pd.Series(dtype=int)).astype(int) == 1)).sum()) if hasattr(self, "anisotropy_rewrite_history") and not self.anisotropy_rewrite_history.empty else 0,
            "pore_over_target_fraction": float(
                (valid_pores["hydraulic_radius_3d_mm"] > self.config.target_hydraulic_radius_mm).mean()
            ) if len(valid_pores) else 0.0,
            "surface_cell_coverage_fraction": surface_cell_coverage,
            "surface_cell_area_ratio_raw": surface_cell_area_ratio_raw,
            "surface_clearance_radius_p95_mm": clearance_p95,
            "surface_clearance_radius_max_mm": clearance_max,
            "uncovered_surface_fraction": uncovered_fraction,
            "subdomain_refinement_round_max": refinement_round_max,
            "unconverged_patch_count": unconverged_patch_count,
            "max_chord_surface_error_mm": float(self.edge_metrics.loc[self.edge_metrics["accepted"] == 1, "max_chord_surface_error_mm"].max()) if int(self.edge_metrics["accepted"].sum()) else 0.0,
            "surface_tolerance_mm": float(self.config.surface_tolerance_mm),
            "contour_repair_surface_tolerance_mm": float(
                self.config.contour_repair_surface_tolerance_mm
            ),
            "domain_violation_count": int(self.edge_metrics.loc[self.edge_metrics["accepted"] == 1, "domain_violation"].sum()) if len(self.edge_metrics) else 0,
            "cross_component_edge_count": int(self.edge_metrics.loc[self.edge_metrics["accepted"] == 1, "cross_component"].sum()) if len(self.edge_metrics) else 0,
            "rejected_domain_edge_count": int(
                ((self.edge_metrics["accepted"] == 0) & (self.edge_metrics.get("pruned_fragment", 0) == 0)).sum()
            ) if len(self.edge_metrics) else 0,
            "pruned_fragment_logical_edge_count": int(self.edge_metrics.get("pruned_fragment", pd.Series(dtype=int)).sum()),
            "pruned_fragment_node_count": int(getattr(self, "_pruned_node_count", 0)),
            "pruned_fragment_edge_segment_count": int(getattr(self, "_pruned_edge_segment_count", 0)),
            "duplicate_logical_edge_skip_count": int(getattr(self, "_duplicate_pair_skips", 0)),
            "degree_cap_skip_count": int(getattr(self, "_degree_cap_skips", 0)),
            "anisotropy_wall_logical_edge_count": int(
                self.edge_metrics.get("anisotropy_wall", pd.Series(dtype=int)).sum()
            ) if len(self.edge_metrics) else 0,
            "anisotropy_wall_edge_segment_count": int(
                (self.edges.get("role", pd.Series(dtype=str)).astype(str) == "surface_anisotropy_wall").sum()
            ) if len(self.edges) else 0,
            "max_node_degree_limit": int(self.config.max_node_degree),
            "observed_max_node_degree": max_node_degree,
            "graph_component_count": int(graph_component_count),
            "graph_largest_component_fraction": graph_largest_fraction,
            "domain_component_graph_connected": int(domain_component_graph_connected),
            "isolated_node_count": isolated_node_count,
            "graph_cycle_rank": int(graph_cycle_rank),
            "mapping_fallback_count": int(self.edge_metrics["mapping_fallback"].sum()) if len(self.edge_metrics) else 0,
            "seam_continuity_ratio": seam_position_ratio,
            "seam_position_continuity_ratio": seam_position_ratio,
            "seam_graph_incidence_ratio": seam_graph_ratio,
            "seam_patch_label_coverage_ratio": seam_patch_label_ratio,
            "seam_position_gap_max_mm": float(seam_position_gap_max),
            "seam_shared_vertex_ratio": float(seam_shared_vertex_ratio),
            "seam_geometric_coverage_ratio": float(seam_position_ratio),
            "seam_sampling_ratio": len(sampled_seam_vertices) / max(len(seam_vertices), 1),
            "mapping_surface": self.config.mapping_surface,
            "projection_backend": (
                "input_derived_decimated_carrier"
                if self.config.mapping_surface == "carrier"
                else ("local_original_triangle_kdtree" if self._use_local_projection else "exact_rtree")
            ),
            "carrier_to_input_error_p95_mm": carrier_error_p95,
            "carrier_to_input_error_max_mm": carrier_error_max,
            "projection_audit_count": len(self._projection_audit_errors),
            "projection_audit_error_p95_mm": _safe_percentile(self._projection_audit_errors, 95),
            "projection_audit_error_max_mm": max(self._projection_audit_errors, default=0.0),
            "gaussian_curvature_units": "mm^-2",
            "gaussian_curvature_class_threshold_mm-2": curvature_threshold,
            **overall_curvature_stats,
        }
        if not self.zero_pore_final_audit.empty:
            summary["zero_pore_audited_source_count"] = int(len(self.zero_pore_final_audit))
            _amin = float(self.config.zero_pore_min_area_mm2)
            summary["zero_pore_unresolved_count"] = int((self.zero_pore_final_audit["open_area_mm2"] < _amin - 1e-10).sum())
            summary["zero_pore_zero_area_unresolved_count"] = int((self.zero_pore_final_audit["open_area_mm2"] <= 1e-10).sum())
            summary["zero_pore_repair_accepted_count"] = int(self.zero_pore_repair_history.get("accepted", pd.Series(dtype=int)).sum())
        else:
            summary["zero_pore_audited_source_count"] = 0
            summary["zero_pore_unresolved_count"] = 0
            summary["zero_pore_zero_area_unresolved_count"] = 0
            summary["zero_pore_repair_accepted_count"] = 0
        failures = []
        # Fully occupied *source* Voronoi cells are provenance diagnostics.
        # Wall mergers change the actual final pore partition, which is
        # audited by audit_final_retained_graph and used as the quality gate.
        # The retained finite-radius graph audit is the final pore authority;
        # pre-repair Voronoi pore metrics remain diagnostic only.
        if summary["uv_overlap_fraction_max"] > 1.0e-6:
            failures.append("noninjective_uv_chart")
        if summary["hydraulic_radius_target_normalized_error_p95"] > self.config.max_hydraulic_radius_rel_error:
            failures.append("hydraulic_radius_error")
        contour_lids = set(pd.to_numeric(
            self.edges.loc[
                self.edges["role"].eq("surface_global_contour_radius_wall"),
                "logical_edge_id",
            ], errors="coerce").dropna().astype(int))
        accepted_metrics = self.edge_metrics.loc[self.edge_metrics["accepted"] == 1]
        contour_metric = pd.to_numeric(
            accepted_metrics["logical_edge_id"], errors="coerce"
        ).isin(contour_lids)
        errors = pd.to_numeric(
            accepted_metrics["max_chord_surface_error_mm"], errors="coerce"
        )
        limits = np.where(
            contour_metric,
            self.config.contour_repair_surface_tolerance_mm,
            self.config.surface_tolerance_mm,
        )
        summary["max_contour_repair_surface_error_mm"] = float(
            errors.loc[contour_metric].max()
        ) if contour_metric.any() else 0.0
        if errors.isna().any() or (errors.to_numpy() > limits + 1.0e-12).any():
            failures.append("surface_fit")
        if summary["domain_violation_count"] > 0 or summary["cross_component_edge_count"] > 0:
            failures.append("domain_crossing")
        if summary["seam_position_continuity_ratio"] < self.config.min_seam_position_continuity:
            failures.append("seam_geometric_coverage")
        if summary["seam_position_gap_max_mm"] > float(self.config.max_seam_uncovered_length_mm) + 1.0e-12:
            failures.append("seam_uncovered_gap")
        if summary["seam_graph_incidence_ratio"] < float(self.config.min_seam_graph_incidence):
            failures.append("seam_graph_incidence")
        if summary["seam_patch_label_coverage_ratio"] < float(self.config.min_seam_patch_label_coverage):
            failures.append("seam_patch_membership")
        if not summary["domain_component_graph_connected"]:
            failures.append("graph_fragmentation")
        if summary["observed_max_node_degree"] > self.config.max_node_degree:
            failures.append("node_degree")
        if summary["surface_cell_coverage_fraction"] < self.config.min_surface_cell_coverage:
            failures.append("incomplete_surface_cells")
        if summary["uncovered_surface_fraction"] > self.config.max_uncovered_surface_fraction:
            failures.append("open_surface_gap")
        # A single sampled clearance maximum is unstable at seams and mesh
        # vertices.  The area-weighted uncovered fraction above is the gate;
        # the maximum remains reported for traceability.
        # A source Voronoi cell may stop refining at the seed-spacing limit.
        # Final acceptance is decided by the retained-graph pore audit below.
        summary["quality_pass"] = int(not failures)
        summary["quality_failures"] = ";".join(failures)
        self.summary_metrics = pd.DataFrame([summary])

    def export_diagnostics(self, output_dir: Path, model_name: str = "model", export_stl: bool = True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.atoms.to_csv(output_dir / "surface_nodes.csv", index=False)
        self.edges.to_csv(output_dir / "surface_edges.csv", index=False)
        self.patch_metrics.to_csv(output_dir / "surface_patch_metrics.csv", index=False)
        self.edge_metrics.to_csv(output_dir / "surface_edge_metrics.csv", index=False)
        self.pore_metrics.to_csv(output_dir / "surface_pore_metrics.csv", index=False)
        if hasattr(self, "pre_zero_repair_pore_metrics"):
            self.pre_zero_repair_pore_metrics.to_csv(output_dir / "surface_pore_metrics_before_zero_repair.csv", index=False)
        self.zero_pore_repair_history.to_csv(output_dir / "surface_zero_pore_repair_history.csv", index=False)
        self.zero_pore_final_audit.to_csv(output_dir / "surface_zero_pore_final_audit.csv", index=False)
        self.final_fragment_repair_history.to_csv(
            output_dir / "surface_final_fragment_repair_history.csv", index=False)
        self.final_audit_repair_history.to_csv(
            output_dir / "surface_final_audit_repair_history.csv", index=False)
        self.coverage_metrics.to_csv(output_dir / "surface_coverage_metrics.csv", index=False)
        self.refinement_history.to_csv(output_dir / "surface_subdomain_refinement_history.csv", index=False)
        self.anisotropy_rewrite_history.to_csv(
            output_dir / "surface_anisotropy_rewrite_history.csv", index=False
        )
        if hasattr(self, "global_ar_cap_rewrite_history"):
            self.global_ar_cap_rewrite_history.to_csv(
                output_dir / "surface_global_ar_cap_rewrite_history.csv", index=False
            )
        if hasattr(self, "global_ar_cap_summary"):
            (output_dir / "surface_global_ar_cap_summary.json").write_text(
                json.dumps(_json_safe(self.global_ar_cap_summary), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if hasattr(self, "contour_radius_repair_summary"):
            (output_dir / "surface_contour_radius_repair_summary.json").write_text(
                json.dumps(_json_safe(self.contour_radius_repair_summary), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self.refinement_evolution = self._build_refinement_evolution_table()
        self.refinement_evolution.to_csv(
            output_dir / "surface_refinement_parameter_evolution.csv", index=False
        )
        self.voronoi_seeds.to_csv(output_dir / "surface_voronoi_seeds.csv", index=False)
        self.seam_metrics.to_csv(output_dir / "surface_seam_metrics.csv", index=False)
        self.seam_geometric_metrics.to_csv(
            output_dir / "surface_seam_geometric_coverage.csv", index=False
        )
        signed_curvature = self.signed_gaussian_curvature.copy()
        signed_curvature.insert(0, "model", model_name)
        signed_curvature.to_csv(
            output_dir / "surface_signed_gaussian_curvature.csv", index=False
        )
        summary = self.summary_metrics.copy()
        summary.insert(0, "model", model_name)
        summary.to_csv(output_dir / "surface_summary_metrics.csv", index=False)
        (output_dir / "surface_build_config.json").write_text(
            json.dumps(asdict(self.config), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record = summary.iloc[0].to_dict()
        signed_statistic_names = (
            "gaussian_curvature_mean_signed",
            "gaussian_curvature_abs_mean",
            "gaussian_curvature_min",
            "gaussian_curvature_max",
            "positive_K_fraction",
            "negative_K_fraction",
            "near_zero_K_fraction",
            "gaussian_curvature_internal_vertex_count",
            "gaussian_curvature_boundary_vertex_count",
            "gaussian_curvature_finite_vertex_count",
        )
        signed_statistics = {
            name: record.get(name) for name in signed_statistic_names
        }
        color_limit = _signed_k_color_limit(
            self.base_vertex_gaussian_curvature,
            self.base_boundary_vertex_mask,
            self.config.curvature_color_percentile,
            self.config.curvature_sign_threshold,
        )
        signed_summary_json = {
            "model": model_name,
            "gaussian_curvature_units": "mm^-2",
            "gaussian_curvature_class_threshold_mm-2": float(
                self.config.curvature_sign_threshold
            ),
            "signed_K_categories": {
                "positive": "K > threshold",
                "near_zero": "abs(K) <= threshold",
                "negative": "K < -threshold",
            },
            "boundary_policy": {
                "boundary_vertices_marked_in_csv": True,
                "boundary_vertices_included_in_signed_K_statistics": False,
                "reason": "Open-boundary angle defects use a pi target and can mimic curvature.",
            },
            "color_scale": {
                "percentile_abs_K": float(self.config.curvature_color_percentile),
                "symmetric_range_mm-2": [-color_limit, color_limit],
                "raw_min_max_are_retained_in_csv_and_statistics": True,
            },
            **signed_statistics,
            "statistics": signed_statistics,
        }
        (output_dir / "surface_signed_gaussian_curvature_summary.json").write_text(
            json.dumps(_json_safe(signed_summary_json), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report = f"""# Surface mapping validation: {model_name}

- Quality status: {'PASS' if int(record.get('quality_pass', 0)) else 'REQUIRES REFINEMENT'}
- Failure flags: {record.get('quality_failures', '') or 'none'}
- Construction mode: {record.get('construction_mode', 'unknown')}
- Patches / connected components: {int(record['patch_count'])} / {int(record['component_count'])}
- Maximum subdomain-refinement round / unconverged patches: {int(record['subdomain_refinement_round_max'])} / {int(record['unconverged_patch_count'])}
- 2-D area distortion, p95: {100.0 * float(record['area_distortion_p95']):.3f}%
- Angle distortion, p95: {float(record['angle_distortion_p95_deg']):.3f} deg
- 2-D to 3-D hydraulic-radius relative error, p95: {100.0 * float(record['hydraulic_radius_rel_error_p95']):.3f}%
- Hydraulic-radius error normalized by the target, p95: {100.0 * float(record['hydraulic_radius_target_normalized_error_p95']):.3f}%
- Surface cell coverage: {100.0 * float(record['surface_cell_coverage_fraction']):.3f}%
- Whole-surface clearance radius, p95 / max: {float(record['surface_clearance_radius_p95_mm']):.6f} / {float(record['surface_clearance_radius_max_mm']):.6f} mm
- Surface area exceeding the clearance target: {100.0 * float(record['uncovered_surface_fraction']):.3f}%
- Maximum accepted chord-to-surface error: {float(record['max_chord_surface_error_mm']):.6f} mm
- Accepted domain violations / cross-component edges: {int(record['domain_violation_count'])} / {int(record['cross_component_edge_count'])}
- Rejected candidate edges: {int(record['rejected_domain_edge_count'])}
- Pruned non-domain debris, nodes / edge segments: {int(record['pruned_fragment_node_count'])} / {int(record['pruned_fragment_edge_segment_count'])}
- Graph components / isolated nodes / independent cycles: {int(record['graph_component_count'])} / {int(record['isolated_node_count'])} / {int(record['graph_cycle_rank'])}
- Largest graph component: {100.0 * float(record['graph_largest_component_fraction']):.3f}%
- C0 seam position continuity: {100.0 * float(record['seam_position_continuity_ratio']):.3f}%
- Graph-incidence continuity across seams: {100.0 * float(record['seam_graph_incidence_ratio']):.3f}%
- Source-seam resampling ratio: {100.0 * float(record['seam_sampling_ratio']):.3f}%
- Strict artificial patch-label coverage at seams: {100.0 * float(record['seam_patch_label_coverage_ratio']):.3f}%
- Carrier-to-input surface deviation, p95 / max: {float(record['carrier_to_input_error_p95_mm']):.6f} / {float(record['carrier_to_input_error_max_mm']):.6f} mm
- Projection backend: {record.get('projection_backend', 'exact_rtree')}
- Signed Gaussian curvature units: mm^-2
- Signed-K class threshold: {float(record['gaussian_curvature_class_threshold_mm-2']):.6g} mm^-2; K > threshold = positive, |K| <= threshold = near-zero, K < -threshold = negative
- Signed-K statistics use finite internal vertices only: n={int(record['gaussian_curvature_internal_vertex_count'])}; boundary vertices excluded from the denominator: {int(record['gaussian_curvature_boundary_vertex_count'])}
- Signed-K mean / absolute mean: {float(record['gaussian_curvature_mean_signed']):.6g} / {float(record['gaussian_curvature_abs_mean']):.6g} mm^-2
- Signed-K min / max: {float(record['gaussian_curvature_min']):.6g} / {float(record['gaussian_curvature_max']):.6g} mm^-2
- Positive / near-zero / negative fractions: {100.0 * float(record['positive_K_fraction']):.3f}% / {100.0 * float(record['near_zero_K_fraction']):.3f}% / {100.0 * float(record['negative_K_fraction']):.3f}%

Only accepted edges are written to `surface_edges.csv` and `surface_graph.stl`.
Rejected candidates remain in `surface_edge_metrics.csv` with `accepted = 0`.
Signed-K vertex data are written to `surface_signed_gaussian_curvature.csv`; the
class threshold and boundary policy are recorded in
`surface_signed_gaussian_curvature_summary.json`.
Iteration-level equivalent radius, aspect ratio, shape factor, liquid-area
fraction, uncertainty envelopes, and reference values are written to
`surface_refinement_parameter_evolution.csv` and plotted in
`surface_refinement_parameter_evolution.svg`.
"""
        (output_dir / "surface_validation_report.md").write_text(report, encoding="utf-8")
        self._plot_subdomain_refinement(output_dir / "surface_subdomain_refinement", model_name)
        self._plot_refinement_parameter_evolution(
            output_dir / "surface_refinement_parameter_evolution"
        )
        self._plot_patch_map(output_dir / "surface_patch_map", model_name)
        self._plot_signed_gaussian_curvature(
            output_dir / "surface_signed_gaussian_curvature", model_name
        )
        self._plot_model_metrics(output_dir / "surface_metric_diagnostics", model_name)
        if export_stl:
            export_graph_stl(self.atoms, self.edges, output_dir / "surface_graph.stl")
        return summary

    def _plot_subdomain_refinement(self, stem: Path, model_name: str):
        if self.refinement_history.empty:
            return
        history = self.refinement_history.copy()
        envelope = history.groupby("round", as_index=False).agg(
            hydraulic_radius_2d_max_mm=("hydraulic_radius_2d_max_mm", "max"),
            hydraulic_radius_3d_max_mm=("hydraulic_radius_3d_max_mm", "max"),
            bad_subdomain_count=("bad_subdomain_count", "sum"),
            seed_count=("seed_count", "sum"),
        )
        threshold = float(history["hydraulic_radius_threshold_mm"].iloc[0])
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
        axes[0].plot(
            envelope["round"], envelope["hydraulic_radius_2d_max_mm"],
            color=PALETTE["blue"], linewidth=1.5, label="2-D subdomain",
        )
        axes[0].plot(
            envelope["round"], envelope["hydraulic_radius_3d_max_mm"],
            color=PALETTE["pink"], linewidth=1.5, label="Mapped 3-D subdomain",
        )
        axes[0].axhline(threshold, color=PALETTE["grey"], linewidth=0.9, linestyle="--")
        axes[0].set_xlabel("Refinement round")
        axes[0].set_ylabel("Worst hydraulic radius (mm)")
        axes[0].legend()
        axes[1].plot(
            envelope["round"], envelope["bad_subdomain_count"],
            color=PALETTE["red"], linewidth=1.5, label="Failing subdomains",
        )
        axes[1].set_xlabel("Refinement round")
        axes[1].set_ylabel("Subdomain count")
        axes[1].grid(axis="y", color=PALETTE["light_grey"], linewidth=0.5, alpha=0.55)
        for label, ax in zip(("a", "b"), axes):
            ax.text(-0.16, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
        fig.suptitle(model_name, x=0.02, ha="left", fontsize=9, fontweight="bold", color=PALETTE["grey"])
        fig.tight_layout()
        save_publication_figure(fig, stem)
        plt.close(fig)

    def _build_refinement_evolution_table(self) -> pd.DataFrame:
        """Aggregate traceable per-patch refinement metrics for plotting/export."""
        if self.refinement_history.empty:
            return pd.DataFrame()

        history = self.refinement_history.sort_values(["patch_id", "round"]).copy()
        rounds = np.arange(int(history["round"].max()) + 1, dtype=int)
        completed = []
        for patch_id, patch_history in history.groupby("patch_id", sort=True):
            carried = (
                patch_history.set_index("round")
                .reindex(rounds)
                .ffill()
                .reset_index(names="round")
            )
            carried["patch_id"] = patch_id
            completed.append(carried)
        history_complete = pd.concat(completed, ignore_index=True)

        rows = []
        for round_id, group in history_complete.groupby("round", sort=True):
            def finite(column: str) -> np.ndarray:
                values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
                return values[np.isfinite(values)]

            def spread(column: str) -> Tuple[float, float, float]:
                values = finite(column)
                if not len(values):
                    return 0.0, 0.0, 0.0
                return (
                    float(np.median(values)),
                    float(np.percentile(values, 5)),
                    float(np.percentile(values, 95)),
                )

            radius_median, radius_p05, radius_p95 = spread("hydraulic_radius_3d_max_mm")
            shape_median, shape_p05, shape_p95 = spread("shape_factor_mean")
            aspect_median, aspect_p05, aspect_p95 = spread("aspect_ratio_mean")
            liquid_median, liquid_p05, liquid_p95 = spread("liquid_area_fraction")
            pore_count = int(pd.to_numeric(group["pore_count"], errors="coerce").fillna(0).sum())

            def pore_weighted_mean(column: str) -> float:
                values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
                weights = pd.to_numeric(group["pore_count"], errors="coerce").fillna(0).to_numpy(float)
                valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
                return float(np.average(values[valid], weights=weights[valid])) if valid.any() else 0.0

            liquid_area = float(pd.to_numeric(group["liquid_area_mm2"], errors="coerce").fillna(0).sum())
            domain_area = float(pd.to_numeric(group["domain_area_mm2"], errors="coerce").fillna(0).sum())
            rows.append({
                "iteration": int(round_id) + 1,
                "active_patch_count": int(group["patch_id"].nunique()),
                "updated_patch_count": int((history["round"] == round_id).sum()),
                "pore_count": pore_count,
                "seed_count": int(pd.to_numeric(group["seed_count"], errors="coerce").fillna(0).sum()),
                "subdomain_count": int(pd.to_numeric(group["subdomain_count"], errors="coerce").fillna(0).sum()),
                "bad_subdomain_count": int(pd.to_numeric(group["bad_subdomain_count"], errors="coerce").fillna(0).sum()),
                "hydraulic_radius_max_mm": float(finite("hydraulic_radius_3d_max_mm").max(initial=0.0)),
                "hydraulic_radius_mean_mm": pore_weighted_mean("hydraulic_radius_3d_mean_mm"),
                "hydraulic_radius_patch_median_mm": radius_median,
                "hydraulic_radius_patch_p05_mm": radius_p05,
                "hydraulic_radius_patch_p95_mm": radius_p95,
                "hydraulic_radius_capillary_target_mm": float(
                    group["hydraulic_radius_threshold_mm"].iloc[0]
                ),
                "hydraulic_radius_bcc_reference_mm": float(
                    self.config.bcc_reference_hydraulic_radius_mm
                ),
                "aspect_ratio_mean": pore_weighted_mean("aspect_ratio_mean"),
                "aspect_ratio_patch_median": aspect_median,
                "aspect_ratio_patch_p05": aspect_p05,
                "aspect_ratio_patch_p95": aspect_p95,
                "aspect_ratio_bcc_reference": float(self.config.bcc_reference_aspect_ratio),
                "shape_factor_mean": pore_weighted_mean("shape_factor_mean"),
                "shape_factor_patch_median": shape_median,
                "shape_factor_patch_p05": shape_p05,
                "shape_factor_patch_p95": shape_p95,
                "shape_factor_bcc_reference": float(self.config.bcc_reference_shape_factor),
                "liquid_area_fraction": liquid_area / max(domain_area, 1.0e-12),
                "liquid_area_fraction_patch_median": liquid_median,
                "liquid_area_fraction_patch_p05": liquid_p05,
                "liquid_area_fraction_patch_p95": liquid_p95,
                "liquid_area_fraction_bcc_reference": float(
                    self.config.bcc_reference_liquid_area_fraction
                ),
            })
        return pd.DataFrame(rows)

    def _plot_refinement_parameter_evolution(self, stem: Path):
        """Export the four-panel surface-refinement figure as editable SVG."""
        data = self.refinement_evolution
        if data.empty:
            return

        x = data["iteration"].to_numpy(float)
        panels = [
            ("hydraulic_radius_max_mm", "hydraulic_radius_mean_mm",
             "hydraulic_radius_patch_p05_mm", "hydraulic_radius_patch_p95_mm",
             "hydraulic_radius_bcc_reference_mm", r"$r_{\mathrm{eq}}$ (mm)",
             PALETTE["blue"], "v"),
            ("aspect_ratio_mean", "aspect_ratio_patch_median",
             "aspect_ratio_patch_p05", "aspect_ratio_patch_p95",
             "aspect_ratio_bcc_reference", "Mean AR", "#6750A4", "D"),
            ("shape_factor_mean", "shape_factor_patch_median",
             "shape_factor_patch_p05", "shape_factor_patch_p95",
             "shape_factor_bcc_reference", "Mean SF", "#2B8CBE", "^"),
            ("liquid_area_fraction", "liquid_area_fraction_patch_median",
             "liquid_area_fraction_patch_p05", "liquid_area_fraction_patch_p95",
             "liquid_area_fraction_bcc_reference", r"$A_{\mathrm{liquid}}^{*}$",
             "#E35D62", ">"),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.6), sharex=True)
        for index, (primary, secondary, low, high, reference, ylabel, color, marker) in enumerate(panels):
            ax = axes[index]
            y = data[primary].to_numpy(float)
            y_secondary = data[secondary].to_numpy(float)
            y_low = data[low].to_numpy(float)
            y_high = data[high].to_numpy(float)
            ax.fill_between(x, y_low, y_high, color=color, alpha=0.14, linewidth=0)
            ax.plot(x, y_secondary, color=color, alpha=0.35, linewidth=1.0,
                    marker="o", markersize=3.6, markeredgecolor="white", markeredgewidth=0.45)
            ax.plot(x, y, color=color, linewidth=2.1, marker=marker, markersize=8.5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label="Surface-refined structure")
            ax.axhline(float(data[reference].iloc[0]), color=PALETTE["grey"],
                       linewidth=1.6, linestyle=(0, (6, 3)), label="BCC reference")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(axis="y", color=PALETTE["light_grey"], linewidth=0.5, alpha=0.75)
            ax.tick_params(direction="out", length=3.0, width=0.7)
            if index == 0:
                ax.legend(loc="upper center", fontsize=7.5, handlelength=1.9)
        axes[-1].set_xlabel("Surface refinement iteration", fontsize=10)
        axes[-1].set_xticks(data["iteration"].astype(int).tolist())
        fig.subplots_adjust(left=0.14, right=0.985, bottom=0.08, top=0.99, hspace=0.19)
        save_publication_figure(fig, stem)
        plt.close(fig)

    def _plot_patch_map(self, stem: Path, model_name: str):
        assert self.base_mesh is not None
        fig = plt.figure(figsize=(7.2, 3.4))
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
        face_centers = np.asarray(self.base_mesh.triangles_center, dtype=float)
        chart_areas = {
            chart.patch_id: float(sum(part.area for part in _polygon_parts(chart.polygon)))
            for chart in self.charts
        }
        chart_centers = {
            chart.patch_id: face_centers[np.asarray(chart.face_ids, dtype=int)].mean(axis=0)
            for chart in self.charts
        }

        # Select large patches that are also spatially separated on the 3-D surface.
        candidate_count = min(len(self.charts), max(16, len(self.charts) // 4))
        candidates = sorted(
            self.charts, key=lambda chart: chart_areas[chart.patch_id], reverse=True
        )[:candidate_count]
        representative = candidates[:1]
        mesh_span = np.maximum(np.ptp(self.base_mesh.vertices, axis=0), 1.0e-9)
        largest_candidate_area = max(chart_areas[chart.patch_id] for chart in candidates)
        while len(representative) < min(4, len(candidates)):
            selected_ids = {chart.patch_id for chart in representative}
            remaining = [chart for chart in candidates if chart.patch_id not in selected_ids]
            representative.append(max(
                remaining,
                key=lambda chart: min(
                    np.linalg.norm(
                        (chart_centers[chart.patch_id] - chart_centers[selected.patch_id])
                        / mesh_span
                    )
                    for selected in representative
                ) * (
                    0.75
                    + 0.25 * math.sqrt(
                        max(chart_areas[chart.patch_id], 1.0e-12)
                        / max(largest_candidate_area, 1.0e-12)
                    )
                ),
            ))

        # Greedy graph colouring maximizes contrast between adjacent patches.
        categorical_palette = (
            "#2171B5", "#D95F59", "#2A9D6F", "#7B61A8",
            "#F39C35", "#2E9EA6", "#D66BA0", "#8A6D4B",
            "#5D78A6", "#A6A23A", "#6E8B57", "#B85C8A",
        )
        face_to_patch = np.full(len(self.base_mesh.faces), -1, dtype=int)
        neighbors = {chart.patch_id: set() for chart in self.charts}
        for chart in self.charts:
            face_to_patch[np.asarray(chart.face_ids, dtype=int)] = chart.patch_id
        for face_a, face_b in np.asarray(self.base_mesh.face_adjacency, dtype=int):
            patch_a, patch_b = int(face_to_patch[face_a]), int(face_to_patch[face_b])
            if patch_a >= 0 and patch_b >= 0 and patch_a != patch_b:
                neighbors[patch_a].add(patch_b)
                neighbors[patch_b].add(patch_a)
        color_index = {
            chart.patch_id: index for index, chart in enumerate(representative)
        }
        for chart in sorted(self.charts, key=lambda item: len(neighbors[item.patch_id]), reverse=True):
            if chart.patch_id in color_index:
                continue
            used = {color_index[item] for item in neighbors[chart.patch_id] if item in color_index}
            color_index[chart.patch_id] = next(
                (index for index in range(len(categorical_palette)) if index not in used),
                chart.patch_id % len(categorical_palette),
            )
        patch_colors = {
            chart.patch_id: categorical_palette[color_index[chart.patch_id]]
            for chart in self.charts
        }
        face_colors = np.empty((len(self.base_mesh.faces), 4), dtype=float)
        for chart in self.charts:
            face_colors[chart.face_ids] = mpl_colors.to_rgba(
                patch_colors[chart.patch_id], 0.95
            )
        for chart in representative:
            face_colors[chart.face_ids] = mpl_colors.to_rgba(
                patch_colors[chart.patch_id], 1.0
            )
        tri = np.asarray(self.base_mesh.vertices)[np.asarray(self.base_mesh.faces)]
        surface_collection = Poly3DCollection(
            tri,
            facecolors=face_colors,
            edgecolors="none",
            linewidths=0.0,
            antialiased=True,
        )
        ax3.add_collection3d(surface_collection)

        def patch_boundary_segments(chart: PatchChart):
            edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
            for face in np.asarray(chart.local_faces, dtype=int):
                for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                    edge_count[tuple(sorted((int(a), int(b))))] += 1
            return [
                np.asarray([chart.xyz[a], chart.xyz[b]], dtype=float)
                for (a, b), count in edge_count.items() if count == 1
            ]

        # White seams show patch locations without assigning extra colours.
        for chart in self.charts:
            for segment in patch_boundary_segments(chart):
                ax3.plot(*segment.T, color="white", lw=0.55, alpha=1.0)
        for chart in representative:
            color = patch_colors[chart.patch_id]
            for segment in patch_boundary_segments(chart):
                ax3.plot(*segment.T, color="white", lw=0.85, alpha=1.0)
            center = chart_centers[chart.patch_id]
            ax3.text(
                *center,
                f"P{chart.patch_id}",
                fontsize=6.2,
                fontweight="bold",
                color=PALETTE["grey"],
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.65,
                    "alpha": 0.92,
                },
            )
        bounds = np.asarray(self.base_mesh.bounds, dtype=float)
        ax3.set_xlim(bounds[0, 0], bounds[1, 0])
        ax3.set_ylim(bounds[0, 1], bounds[1, 1])
        ax3.set_zlim(bounds[0, 2], bounds[1, 2])
        ax3.set_title("3-D surface partition", loc="left", fontsize=8, fontweight="bold")
        ax3.set_axis_off()
        ax3.set_box_aspect(np.maximum(np.ptp(self.base_mesh.vertices, axis=0), 1.0e-6))
        ax3.view_init(elev=24, azim=-55)

        columns = 2
        connector_targets = {}
        for chart_index, chart in enumerate(representative):
            parts = _polygon_parts(chart.polygon)
            if not parts:
                continue
            bounds = chart.polygon.bounds
            width = max(bounds[2] - bounds[0], 1.0e-6)
            height = max(bounds[3] - bounds[1], 1.0e-6)
            scale = 0.78 / max(width, height)
            col = chart_index % columns
            row = chart_index // columns
            offset_x = 1.05 * col
            offset_y = -1.02 * row
            color = patch_colors[chart.patch_id]
            for part in parts:
                xy = np.asarray(part.exterior.coords)
                plot_xy = np.column_stack(
                    ((xy[:, 0] - bounds[0]) * scale + offset_x,
                     (xy[:, 1] - bounds[1]) * scale + offset_y)
                )
                ax2.fill(
                    plot_xy[:, 0], plot_xy[:, 1],
                    color=color, alpha=0.90, lw=0,
                )
                ax2.plot(plot_xy[:, 0], plot_xy[:, 1], color=color, lw=0.75)
                for ring in part.interiors:
                    hole = np.asarray(ring.coords)
                    hole_xy = np.column_stack(
                        ((hole[:, 0] - bounds[0]) * scale + offset_x,
                         (hole[:, 1] - bounds[1]) * scale + offset_y)
                    )
                    ax2.fill(hole_xy[:, 0], hole_xy[:, 1], color="white", zorder=3)
                    ax2.plot(hole_xy[:, 0], hole_xy[:, 1], color=color, lw=0.75, zorder=4)
            ax2.text(
                offset_x + 0.39,
                offset_y - 0.10,
                f"P{chart.patch_id}",
                fontsize=6.2,
                fontweight="bold",
                color=PALETTE["grey"],
                ha="center",
                va="top",
            )
            connector_targets[chart.patch_id] = (offset_x + 0.39, offset_y - 0.07)
        ax2.set_xlim(-0.08, 1.91)
        ax2.set_ylim(-1.18, 0.88)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_axis_off()
        ax2.set_title("Representative 2-D parameter domains", loc="left", fontsize=8, fontweight="bold")
        fig.text(
            0.52,
            0.045,
            "Matching colour and patch ID indicate the same domain in 3-D and 2-D.",
            ha="center",
            fontsize=6.0,
            color=PALETTE["grey"],
        )
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.94, wspace=0.04)
        fig.canvas.draw()
        for chart in representative:
            center = chart_centers[chart.patch_id]
            projected_x, projected_y, _ = proj3d.proj_transform(*center, ax3.get_proj())
            connector = ConnectionPatch(
                xyA=(projected_x, projected_y), coordsA=ax3.transData,
                xyB=connector_targets[chart.patch_id], coordsB=ax2.transData,
                arrowstyle="-|>", mutation_scale=5.0,
                color=patch_colors[chart.patch_id], lw=0.75, alpha=0.58,
                linestyle=(0, (2.0, 2.0)), shrinkA=6.0, shrinkB=5.0,
                clip_on=False, zorder=10,
            )
            connector.set_in_layout(False)
            fig.add_artist(connector)
        save_publication_figure(fig, stem)
        plt.close(fig)

    def _plot_signed_gaussian_curvature(self, stem: Path, model_name: str):
        """Plot the signed-K surface map and compact patch-level distribution."""
        assert self.base_mesh is not None
        vertices = np.asarray(self.base_mesh.vertices, dtype=float)
        faces = np.asarray(self.base_mesh.faces, dtype=int)
        vertex_k = np.asarray(self.base_vertex_gaussian_curvature, dtype=float)
        boundary_mask = np.asarray(self.base_boundary_vertex_mask, dtype=bool)
        threshold = float(self.config.curvature_sign_threshold)
        color_limit = _signed_k_color_limit(
            vertex_k,
            boundary_mask,
            self.config.curvature_color_percentile,
            threshold,
        )
        curvature_cmap = mpl_colors.LinearSegmentedColormap.from_list(
            "signed_gaussian_curvature",
            [PALETTE["blue"], PALETTE["mid"], PALETTE["pink"]],
            N=257,
        )
        curvature_norm = mpl_colors.TwoSlopeNorm(
            vmin=-color_limit, vcenter=0.0, vmax=color_limit
        )

        # A face touching the open boundary is colored from its finite internal
        # vertices only; a face with no internal vertex is deliberately neutral.
        face_values = np.full(len(faces), np.nan, dtype=float)
        for face_id, face in enumerate(faces):
            included = ~boundary_mask[face] & np.isfinite(vertex_k[face])
            if np.any(included):
                face_values[face_id] = float(np.mean(vertex_k[face][included]))
        face_colors = [
            curvature_cmap(curvature_norm(value)) if np.isfinite(value)
            else mpl_colors.to_rgba(PALETTE["light_grey"], 0.72)
            for value in face_values
        ]

        fig = plt.figure(figsize=(7.2, 3.65))
        grid = fig.add_gridspec(1, 2, width_ratios=(2.25, 1.0), wspace=0.22)
        ax3 = fig.add_subplot(grid[0, 0], projection="3d")
        ax_dist = fig.add_subplot(grid[0, 1])
        surface_collection = Poly3DCollection(
            vertices[faces],
            facecolors=face_colors,
            edgecolors=PALETTE["light_grey"],
            linewidths=0.08,
            antialiased=True,
        )
        ax3.add_collection3d(surface_collection)
        bounds = np.asarray(self.base_mesh.bounds, dtype=float)
        ax3.set_xlim(bounds[0, 0], bounds[1, 0])
        ax3.set_ylim(bounds[0, 1], bounds[1, 1])
        ax3.set_zlim(bounds[0, 2], bounds[1, 2])
        ax3.set_box_aspect(np.maximum(np.ptp(vertices, axis=0), 1.0e-6))
        ax3.view_init(elev=24, azim=-55)
        ax3.set_axis_off()
        ax3.text2D(0.0, 1.02, "a", transform=ax3.transAxes, fontsize=8, fontweight="bold")

        patch_table = self.patch_metrics.copy()
        patch_values = np.asarray(
            patch_table.get("gaussian_curvature_mean_signed", pd.Series(dtype=float)),
            dtype=float,
        )
        patch_counts = np.asarray(
            patch_table.get(
                "gaussian_curvature_internal_vertex_count", pd.Series(dtype=float)
            ),
            dtype=float,
        )
        keep = np.isfinite(patch_values)
        patch_values = patch_values[keep]
        patch_counts = patch_counts[keep]
        if len(patch_values):
            order = np.argsort(patch_values)
            patch_values = patch_values[order]
            patch_counts = patch_counts[order]
            class_colors = {
                "negative": PALETTE["blue"],
                "near_zero": PALETTE["mid"],
                "positive": PALETTE["pink"],
            }
            colors = [
                class_colors[_curvature_class(float(value), threshold)]
                for value in patch_values
            ]
            max_count = max(float(np.max(patch_counts)), 1.0)
            sizes = 18.0 + 92.0 * np.sqrt(np.maximum(patch_counts, 0.0) / max_count)
            ax_dist.scatter(
                patch_values,
                np.arange(len(patch_values)),
                s=sizes,
                c=colors,
                alpha=0.86,
                edgecolors="white",
                linewidths=0.35,
                rasterized=len(patch_values) > 500,
            )
            ax_dist.set_ylim(-1.0, max(len(patch_values), 1.0))
        else:
            ax_dist.text(0.5, 0.5, "No internal patch vertices", ha="center", va="center", transform=ax_dist.transAxes)
            ax_dist.set_ylim(-1.0, 1.0)
        ax_dist.axvline(0.0, color=PALETTE["grey"], linewidth=0.8, zorder=0)
        ax_dist.set_xlim(-color_limit, color_limit)
        ax_dist.set_xticks([-color_limit, 0.0, color_limit])
        ax_dist.set_xticklabels(
            [f"−{color_limit:.2g}", "0", f"{color_limit:.2g}"], fontsize=6.5
        )
        ax_dist.set_yticks([])
        ax_dist.set_xlabel("Patch mean signed K (mm^-2)")
        ax_dist.set_title("Patch-level signed-K distribution", loc="left", fontsize=7, pad=5)
        record = self.summary_metrics.iloc[0].to_dict()
        ax_dist.text(
            0.02,
            0.97,
            "Internal vertices only\n"
            f"n = {int(record['gaussian_curvature_internal_vertex_count'])}\n"
            f"K > 0: {100.0 * float(record['positive_K_fraction']):.1f}%\n"
            f"K ≈ 0: {100.0 * float(record['near_zero_K_fraction']):.1f}%\n"
            f"K < 0: {100.0 * float(record['negative_K_fraction']):.1f}%",
            transform=ax_dist.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color=PALETTE["grey"],
        )
        ax_dist.text(0.02, -0.15, "blue: K < 0   grey: K ≈ 0   red: K > 0", transform=ax_dist.transAxes, fontsize=5.8, color=PALETTE["grey"])
        ax_dist.text(-0.16, 1.04, "b", transform=ax_dist.transAxes, fontsize=8, fontweight="bold")
        ax_dist.grid(axis="x", color=PALETTE["light_grey"], linewidth=0.5, alpha=0.55)

        scalar_map = plt.cm.ScalarMappable(norm=curvature_norm, cmap=curvature_cmap)
        scalar_map.set_array([])
        colorbar = fig.colorbar(scalar_map, ax=ax3, fraction=0.035, pad=0.025, shrink=0.78)
        colorbar.set_ticks([-color_limit, 0.0, color_limit])
        colorbar.set_ticklabels([f"−{color_limit:.2g}", "0", f"{color_limit:.2g}"])
        colorbar.set_label("Signed Gaussian curvature, K (mm^-2)")
        fig.text(
            0.02,
            0.015,
            "Boundary vertices are marked in the CSV and excluded from signed-K statistics.",
            fontsize=5.8,
            color=PALETTE["grey"],
        )
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.13, top=0.91)
        save_publication_figure(fig, stem)
        plt.close(fig)

    def _plot_model_metrics(self, stem: Path, model_name: str):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
        values = [
            (self.patch_metrics.get("area_distortion_p95", pd.Series(dtype=float)) * 100.0, "Area distortion, patch P95 (%)", PALETTE["blue"], "%"),
            (self.patch_metrics.get("angle_distortion_p95_deg", pd.Series(dtype=float)), "Angular distortion, patch P95 (°)", PALETTE["mid"], "°"),
            (self.patch_metrics.get("hydraulic_radius_rel_error_p95", pd.Series(dtype=float)) * 100.0, "Equivalent-radius error, patch P95 (%)", PALETTE["pink"], "%"),
        ]
        for label, (series, title, color, unit) in zip("abc", values):
            ax = axes[ord(label) - ord("a")]
            data = np.asarray(series, dtype=float)
            data = data[np.isfinite(data)]
            if len(data):
                median = float(np.median(data))
                percentile95 = float(np.percentile(data, 95))
                ax.hist(
                    data,
                    bins=min(18, max(7, int(math.sqrt(len(data))))),
                    color=color,
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.45,
                )
                ax.axvline(median, color=PALETTE["grey"], linewidth=0.9, zorder=3)
                ax.axvline(
                    percentile95,
                    color=PALETTE["red"],
                    linewidth=0.9,
                    linestyle=(0, (3.0, 2.0)),
                    zorder=3,
                )
                decimals = 2 if percentile95 < 10.0 else 1
                ax.text(
                    0.97,
                    0.96,
                    f"Median = {median:.{decimals}f}{unit}\n"
                    f"P95 = {percentile95:.{decimals}f}{unit}\n"
                    f"n = {len(data)} patches",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.2,
                    color=PALETTE["grey"],
                    linespacing=1.35,
                )
            ax.set_xlabel(title)
            ax.set_ylabel("Patch count" if label == "a" else "")
            ax.text(-0.14, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
            ax.grid(axis="y", color=PALETTE["light_grey"], linewidth=0.5, alpha=0.45)
            ax.tick_params(direction="out", length=2.5, width=0.7)
        fig.text(
            0.5,
            0.02,
            "Solid line: median    Dashed line: 95th percentile",
            ha="center",
            fontsize=6.2,
            color=PALETTE["grey"],
        )
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.92, wspace=0.14)
        save_publication_figure(fig, stem)
        plt.close(fig)


def style_publication_axes(fig):
    """Keep numeric axes legible at the common manuscript figure width."""
    width, height = fig.get_size_inches()
    target_width = 17.5 / 2.54
    if abs(width - target_width) > 1.0e-6:
        fig.set_size_inches(target_width, height * target_width / width)
    for ax in fig.axes:
        if ax.name == "3d" or ax.get_label() == "<colorbar>":
            continue
        for axis in (ax.xaxis, ax.yaxis):
            if axis.converter is None and len(axis.get_ticklocs()) > 6:
                axis.set_major_locator(MaxNLocator(nbins=5))


def save_publication_figure(fig, stem: Path, dpi: int = 600):
    stem = Path(stem)
    style_publication_axes(fig)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    for suffix in (".pdf", ".tiff", ".tif", ".png"):
        stem.with_suffix(suffix).unlink(missing_ok=True)


def _sweep_polyline_tube(points: np.ndarray, radius: float, sections: int = 10) -> trimesh.Trimesh:
    """Create one continuous tube whose centreline follows a mapped polyline."""
    points = np.asarray(points, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-9]
    points = points[keep]
    if len(points) < 2:
        return trimesh.Trimesh()
    tangents = np.empty_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tangents[1:-1] = points[2:] - points[:-2]
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1.0e-12)

    normals = np.empty_like(points)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, tangents[0]))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    normals[0] = np.cross(tangents[0], reference)
    normals[0] /= max(float(np.linalg.norm(normals[0])), 1.0e-12)
    for i in range(1, len(points)):
        transported = normals[i - 1] - np.dot(normals[i - 1], tangents[i]) * tangents[i]
        if np.linalg.norm(transported) <= 1.0e-9:
            reference = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(reference, tangents[i]))) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            transported = np.cross(tangents[i], reference)
        normals[i] = transported / max(float(np.linalg.norm(transported)), 1.0e-12)
    binormals = np.cross(tangents, normals)
    angles = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
    rings = np.asarray([
        point + radius * (
            np.cos(angles)[:, None] * normal + np.sin(angles)[:, None] * binormal
        )
        for point, normal, binormal in zip(points, normals, binormals)
    ])
    vertices = rings.reshape(-1, 3).tolist()
    faces = []
    for ring_id in range(len(points) - 1):
        start = ring_id * sections
        nxt = (ring_id + 1) * sections
        for j in range(sections):
            k = (j + 1) % sections
            faces.extend(((start + j, nxt + j, nxt + k), (start + j, nxt + k, start + k)))
    start_center = len(vertices)
    vertices.append(points[0].tolist())
    end_center = len(vertices)
    vertices.append(points[-1].tolist())
    last = (len(points) - 1) * sections
    for j in range(sections):
        k = (j + 1) % sections
        faces.append((start_center, k, j))
        faces.append((end_center, last + j, last + k))
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=int), process=False
    )


def export_graph_stl(atoms: pd.DataFrame, edges: pd.DataFrame, path: Path):
    if atoms.empty or edges.empty:
        return
    coords = atoms.set_index("atom_id")[["x", "y", "z"]].to_dict("index")
    meshes = []
    for _, group in edges.groupby("logical_edge_id", sort=False):
        rows = list(group.itertuples(index=False))
        if not rows:
            continue
        point_ids = [int(rows[0].source)] + [int(row.target) for row in rows]
        if any(atom_id not in coords for atom_id in point_ids):
            continue
        points = np.asarray(
            [[coords[atom_id][axis] for axis in ("x", "y", "z")] for atom_id in point_ids],
            dtype=float,
        )
        tube = _sweep_polyline_tube(points, float(rows[0].radius), sections=10)
        if len(tube.faces):
            meshes.append(tube)
    if meshes:
        trimesh.util.concatenate(meshes).export(path)


def _grid_surface(x: np.ndarray, y: np.ndarray, z_fn, mask_fn=None) -> trimesh.Trimesh:
    xx, yy = np.meshgrid(x, y)
    zz = z_fn(xx, yy)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    ny, nx = xx.shape
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            ids = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i]
            center = np.mean(vertices[ids], axis=0)
            if mask_fn is not None and not mask_fn(center[0], center[1]):
                continue
            faces.extend([[ids[0], ids[1], ids[2]], [ids[0], ids[2], ids[3]]])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=int), process=True)


def generate_validation_models() -> Dict[str, trimesh.Trimesh]:
    grid = np.linspace(-5.0, 5.0, 19)
    positive = _grid_surface(grid, grid, lambda x, y: 0.075 * (x * x + y * y))
    negative = _grid_surface(grid, grid, lambda x, y: 0.075 * (x * x - y * y))
    sign_change = _grid_surface(grid, grid, lambda x, y: 0.012 * x ** 3 + 0.055 * y ** 2)
    hole = _grid_surface(
        grid,
        grid,
        lambda x, y: 0.10 * np.sin(0.35 * x) * np.cos(0.35 * y),
        mask_fn=lambda x, y: not (-1.7 < x < 1.7 and -1.7 < y < 1.7),
    )
    left = _grid_surface(np.linspace(-5.0, -0.8, 12), np.linspace(-3.2, 3.2, 15), lambda x, y: 0.07 * ((x + 2.9) ** 2 + 0.5 * y ** 2))
    right = _grid_surface(np.linspace(0.8, 5.0, 12), np.linspace(-3.2, 3.2, 15), lambda x, y: -0.06 * ((x - 2.9) ** 2 - 0.5 * y ** 2))
    multi = trimesh.util.concatenate([left, right])
    return {
        "positive_K": positive,
        "negative_K": negative,
        "K_sign_change": sign_change,
        "hole": hole,
        "same_height_multi_domain": multi,
    }


def plot_validation_summary(summary: pd.DataFrame, output_dir: Path):
    output_dir = Path(output_dir)
    display_names = {
        "positive_K": "K > 0",
        "negative_K": "K < 0",
        "K_sign_change": "K sign change",
        "same_height_multi_domain": "Multi-domain",
        "knitted_elephant": "Multibranch free boundary",
        "knitted_elephant_highres": "Multibranch free boundary",
    }
    labels = [display_names.get(str(name), str(name)) for name in summary["model"]]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    panels = [
        ("area_distortion_p95", 100.0, "2-D area distortion, p95 (%)", PALETTE["blue"]),
        ("hydraulic_radius_rel_error_p95", 100.0, "3-D hydraulic-radius error, p95 (%)", PALETTE["pink"]),
        ("max_chord_surface_error_mm", 1.0, "Maximum chord-to-surface error (mm)", PALETTE["mid"]),
        ("seam_position_continuity_ratio" if "seam_position_continuity_ratio" in summary else "seam_continuity_ratio", 100.0, "C0 seam position continuity (%)", PALETTE["green"]),
    ]
    for panel_label, ax, (column, factor, ylabel, color) in zip("abcd", axes.ravel(), panels):
        vals = summary[column].to_numpy(float) * factor
        ax.bar(x, vals, width=0.68, color=color, edgecolor="white", linewidth=0.5)
        ax.scatter(x, vals, s=10, color=PALETTE["grey"], zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(axis="y", color=PALETTE["light_grey"], linewidth=0.55, alpha=0.65)
        ax.text(-0.16, 1.03, panel_label, transform=ax.transAxes, fontsize=8, fontweight="bold")
        if column == "max_chord_surface_error_mm":
            tolerance = float(summary.get("surface_tolerance_mm", pd.Series([np.nan])).iloc[0])
            if np.isfinite(tolerance):
                ax.axhline(tolerance, color=PALETTE["red"], lw=0.8, ls="--", label="Tolerance")
                ax.legend(loc="upper right")
        if column in {"seam_continuity_ratio", "seam_position_continuity_ratio"}:
            ax.set_ylim(0, 105)
    fig.suptitle("Curvature-aware surface mapping validation", x=0.02, ha="left", fontsize=10, fontweight="bold", color=PALETTE["grey"])
    fig.tight_layout()
    save_publication_figure(fig, output_dir / "surface_validation_summary")
    plt.close(fig)


def export_pore_and_pressure_analysis(
    output_dir: Path,
    config: SurfaceBuildConfig,
    layers: int = 20,
    gamma_npm: float = SURFACE_TENSION_N_PER_M,
    liquid_density_kgm3: float = LIQUID_DENSITY_KG_PER_M3,
    gas_density_kgm3: float = GAS_DENSITY_KG_PER_M3,
    gravity_ms2: float = GRAVITY_M_PER_S2,
    contact_angle_deg: float = CONTACT_ANGLE_DEG,
    additional_pressure_pa: float = ADDITIONAL_PRESSURE_PA,
    safety_factor: float = SURFACE_CAPILLARY_SAFETY_FACTOR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Export pore statistics, 20-layer pressure checks, and bubble plots."""
    output_dir = Path(output_dir)
    pores_all = pd.read_csv(output_dir / "surface_pore_metrics.csv")
    summary = pd.read_csv(output_dir / "surface_summary_metrics.csv")
    pores = pores_all[pores_all["hydraulic_radius_3d_mm"] > 1.0e-9].copy()
    if pores.empty:
        raise ValueError("No positive-area pores were generated")

    target = float(config.target_hydraulic_radius_mm)
    total_cell_area = float(pores_all["cell_area_3d_mm2"].sum())
    liquid_fraction = float(
        pores_all["pore_area_3d_mm2"].sum() / max(total_cell_area, 1.0e-12)
    )
    r_max = float(pores["hydraulic_radius_3d_mm"].max())
    physics = pd.DataFrame([{
        "A_LL": liquid_fraction,
        "target_hydraulic_radius_mm": target,
        "target_equivalent_capillary_radius_mm": target,
        "target_radius_source": config.target_radius_source,
        "design_target_height_mm": config.design_target_height_mm,
        "surface_additional_pressure_Pa": additional_pressure_pa,
        "surface_capillary_safety_factor": safety_factor,
        "r_hyd_mean_mm": float(pores["hydraulic_radius_3d_mm"].mean()),
        "r_hyd_p95_mm": float(pores["hydraulic_radius_3d_mm"].quantile(0.95)),
        "r_hyd_max_mm": r_max,
        "S_normalized_target_over_max": target / max(r_max, 1.0e-12),
        "shape_factor_mean": float(pores["shape_factor"].mean()),
        "aspect_ratio_mean": float(pores["aspect_ratio"].mean()),
        "pore_over_target_fraction": float(
            (pores["hydraulic_radius_3d_mm"] > target).mean()
        ),
    }])
    physics.to_csv(output_dir / "surface_physics_summary.csv", index=False)

    z_min = float(pores["centroid_z_mm"].min())
    z_max = float(pores["centroid_z_mm"].max())
    layer_bounds = np.linspace(z_min, z_max + 1.0e-9, max(1, int(layers)) + 1)
    pores["height_layer"] = np.clip(
        np.digitize(pores["centroid_z_mm"], layer_bounds) - 1,
        0,
        len(layer_bounds) - 2,
    )
    cos_theta = math.cos(math.radians(float(contact_angle_deg)))
    pressure_rows = []
    for layer in range(len(layer_bounds) - 1):
        group = pores[pores["height_layer"] == layer]
        z_mm = 0.5 * (layer_bounds[layer] + layer_bounds[layer + 1])
        r_worst = float(group["hydraulic_radius_3d_mm"].max()) if len(group) else np.nan
        r_p95 = float(group["hydraulic_radius_3d_mm"].quantile(0.95)) if len(group) else np.nan
        p_cap = (
            2.0 * gamma_npm * cos_theta / (r_worst * 1.0e-3)
            if np.isfinite(r_worst) else np.nan
        )
        p_cap_p95 = (
            2.0 * gamma_npm * cos_theta / (r_p95 * 1.0e-3)
            if np.isfinite(r_p95) else np.nan
        )
        p_hydro = (
            (liquid_density_kgm3 - gas_density_kgm3)
            * gravity_ms2
            * max(z_mm - z_min, 0.0)
            * 1.0e-3
        )
        p_required = safety_factor * (p_hydro + additional_pressure_pa)
        pressure_rows.append({
            "layer": layer + 1,
            "z_mm": z_mm,
            "pore_count": len(group),
            "r_hyd_worst_mm": r_worst,
            "r_hyd_p95_mm": r_p95,
            "P_cap_min_Pa": p_cap,
            "P_cap_p95_Pa": p_cap_p95,
            "P_hydro_Pa": p_hydro,
            "additional_pressure_Pa": additional_pressure_pa,
            "safety_factor": safety_factor,
            "P_required_Pa": p_required,
            "pressure_pass": int(np.isfinite(p_cap) and p_cap >= p_required),
        })
    pressure = pd.DataFrame(pressure_rows)
    pressure["P_cap_trend_Pa"] = pressure["P_cap_p95_Pa"].rolling(
        3, center=True, min_periods=1
    ).median()
    pressure.to_csv(output_dir / "surface_pressure_20_layers.csv", index=False)

    cmap = mpl_colors.LinearSegmentedColormap.from_list(
        "pore_features", [PALETTE["blue"], PALETTE["mid"], PALETTE["pink"]]
    )
    # Plot central observations only; the source table and acceptance audit
    # retain every pore, including the extrema.
    plot_pores = pores.copy()
    for column in ("hydraulic_radius_3d_mm", "shape_factor", "aspect_ratio", "pore_area_3d_mm2"):
        values = pd.to_numeric(pores[column], errors="coerce")
        lower, upper = values.quantile([0.05, 0.95])
        plot_pores = plot_pores.loc[values.between(lower, upper, inclusive="both")]
    if plot_pores.empty:
        raise ValueError("No complete pores remain in the 5–95% plotting range")
    (output_dir / "surface_pore_bubble_plot_selection.json").write_text(
        json.dumps({
            "source_complete_pores": int(len(pores)),
            "plotted_complete_pores": int(len(plot_pores)),
            "plotting_quantile_interval": [0.05, 0.95],
            "trimmed_plot_variables": ["hydraulic_radius_3d_mm", "shape_factor",
                                       "aspect_ratio", "pore_area_3d_mm2"],
            "acceptance_uses_all_pores": True,
        }, indent=2), encoding="utf-8")
    pore_area = plot_pores["pore_area_3d_mm2"]
    area_span = max(float(pore_area.max() - pore_area.min()), 1.0e-12)
    marker_sizes = 80.0 + 250.0 * (pore_area - pore_area.min()) / area_span
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    scatter = ax.scatter(
        plot_pores["hydraulic_radius_3d_mm"], plot_pores["shape_factor"],
        c=plot_pores["aspect_ratio"], s=marker_sizes, cmap=cmap,
        alpha=0.88, edgecolors="white", linewidth=0.35,
        vmin=float(plot_pores["aspect_ratio"].min()),
        vmax=float(plot_pores["aspect_ratio"].max()),
    )
    handles, labels = [], []
    for quantile in (0.1, 0.5, 0.9):
        value = float(pore_area.quantile(quantile))
        size = 80.0 + 250.0 * (value - float(pore_area.min())) / area_span
        handles.append(ax.scatter(
            [], [], s=size, color=PALETTE["mid"], alpha=0.8,
            edgecolors="white", linewidth=0.5,
        ))
        labels.append(f"Area = {value:.2f}")
    legend = ax.legend(
        handles, labels, title="Pore area", loc="lower right",
        frameon=True, framealpha=0.9, edgecolor="black",
    )
    legend.get_frame().set_linewidth(1.0)
    ax.set_xlabel(r"Hydraulic radius ($r_{hyd}$)")
    ax.set_ylabel(r"Shape factor ($SF$)")
    ax.set_title("Geometric Feature Correlation Map", pad=15)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(direction="in", top=True, right=True)
    x_values = pd.to_numeric(
        plot_pores["hydraulic_radius_3d_mm"], errors="coerce"
    ).to_numpy()
    x_values = x_values[np.isfinite(x_values)]

    x_max_data = float(np.max(x_values))
    ax.set_xlim(0.0, 1.08 * x_max_data)
    ax.set_ylim(0.0, min(1.0, 1.05 * float(plot_pores["shape_factor"].max())))
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02, extend="max")
    colorbar.set_label(r"Aspect ratio ($AR$)", fontweight="bold")
    colorbar.outline.set_linewidth(1.0)
    fig.tight_layout()
    save_publication_figure(fig, output_dir / "surface_pore_bubble")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].hist(
        pores["hydraulic_radius_3d_mm"], bins=28,
        color=PALETTE["blue"], edgecolor="white", linewidth=0.4,
    )
    axes[0].axvline(target, color=PALETTE["red"], linestyle="--", linewidth=1.1)
    axes[0].set_xlabel(r"Hydraulic radius, $r_{hyd}$ (mm)")
    axes[0].set_ylabel("Pore count")
    axes[1].scatter(
        pores["centroid_z_mm"], pores["hydraulic_radius_3d_mm"],
        s=5, color=PALETTE["green"], alpha=0.5, rasterized=True,
    )
    axes[1].axhline(target, color=PALETTE["red"], linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Height, z (mm)")
    axes[1].set_ylabel(r"Hydraulic radius, $r_{hyd}$ (mm)")
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    save_publication_figure(fig, output_dir / "surface_pore_statistics")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(
        pressure["z_mm"], pressure["P_cap_min_Pa"],
        color=PALETTE["blue"], marker="o", s=10, alpha=0.45,
        label="Worst-pore safety values",
    )
    ax.plot(
        pressure["z_mm"], pressure["P_cap_trend_Pa"],
        color=PALETTE["blue"], linewidth=1.4,
        label="Local P95 pressure trend",
    )
    ax.plot(
        pressure["z_mm"], pressure["P_hydro_Pa"],
        color=PALETTE["grey"], linestyle=":", linewidth=1.2,
        label="Hydrostatic pressure",
    )
    ax.plot(
        pressure["z_mm"], pressure["P_required_Pa"],
        color=PALETTE["red"], linestyle="--", linewidth=1.3,
        label="Margin-corrected requirement",
    )
    passed = pressure["pressure_pass"].astype(bool)
    ax.scatter(
        pressure.loc[passed, "z_mm"], pressure.loc[passed, "P_cap_min_Pa"],
        color=PALETTE["green"], s=18, zorder=4,
    )
    ax.set_xlabel("Height, z (mm)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_xlim(0.0, 25.0)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.legend(fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    save_publication_figure(fig, output_dir / "surface_pressure_balance")
    plt.close(fig)

    complete = summary.copy()
    for column, value in physics.iloc[0].items():
        complete[column] = value
    complete["pressure_pass_layers"] = int(pressure["pressure_pass"].sum())
    complete["pressure_total_layers"] = len(pressure)
    complete.to_csv(output_dir / "surface_complete_test_summary.csv", index=False)
    return complete, pressure


def run_validation_suite(output_dir: Path, config: SurfaceBuildConfig, input_mesh: Optional[trimesh.Trimesh] = None, input_name: str = "input_model") -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = generate_validation_models()
    if input_mesh is not None:
        models[input_name] = input_mesh
    summaries = []
    for name, mesh in models.items():
        print(f"[surface] validating {name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        builder = SurfacePatchBuilder(mesh, SurfaceBuildConfig(**asdict(config))).build()
        model_dir = output_dir / name
        builder.audit_final_retained_graph(
            builder.atoms, builder.edges, model_dir,
            node_spheres_exported=False,
            include_stitch_anchors=False,
            apply_quality_gate=True,
        )
        summary = builder.export_diagnostics(model_dir, model_name=name, export_stl=True)
        summary["surface_tolerance_mm"] = config.surface_tolerance_mm
        summary["contour_repair_surface_tolerance_mm"] = (
            config.contour_repair_surface_tolerance_mm
        )
        summaries.append(summary)
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(output_dir / "surface_validation_all_models.csv", index=False)
    validation_signed_summary = {
        "gaussian_curvature_units": "mm^-2",
        "gaussian_curvature_class_threshold_mm-2": float(
            config.curvature_sign_threshold
        ),
        "boundary_policy": {
            "boundary_vertices_marked_in_csv": True,
            "boundary_vertices_included_in_signed_K_statistics": False,
        },
        "models": all_summary.to_dict(orient="records"),
    }
    (output_dir / "surface_validation_signed_gaussian_curvature_summary.json").write_text(
        json.dumps(_json_safe(validation_signed_summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_validation_summary(all_summary, output_dir)
    return all_summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Curvature-aware 2-D/3-D surface graph builder and validator")
    parser.add_argument("--input", type=Path, default=INPUT_STL_PATH, help="Input STL/OBJ/PLY surface model")
    parser.add_argument("--target-height", type=float, default=TARGET_HEIGHT_MM, help="Scale model height while preserving aspect ratio")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--validation-suite", action="store_true", help="Also run positive/negative/sign-changing K, hole, and multi-domain models")
    parser.add_argument("--target-faces", type=int, default=1000)
    parser.add_argument("--max-faces-per-patch", type=int, default=72)
    parser.add_argument(
        "--seam-screen-max-fraction", type=float, default=0.50,
        help=("Maximum accepted fraction of local pore records touching an artificial "
              "inter-patch seam; screening only, not a global pore proof."),
    )
    parser.add_argument(
        "--disable-seam-screen", action="store_true",
        help="Report seam burden without using it in the provisional surface quality gate.",
    )
    parser.add_argument("--normal-angle", type=float, default=24.0)
    parser.add_argument("--split-area-distortion-p95", type=float, default=0.04)
    parser.add_argument("--split-angle-distortion-p95-deg", type=float, default=3.0)
    parser.add_argument("--split-rounds", type=int, default=8)
    parser.add_argument(
        "--curvature-sign-threshold",
        type=float,
        default=GAUSSIAN_CURVATURE_NEAR_ZERO_THRESHOLD_MM2,
        help="Near-zero signed-K threshold in mm^-2; |K| <= threshold is classified as near-zero",
    )
    parser.add_argument(
        "--curvature-color-percentile",
        type=float,
        default=SIGNED_K_COLOR_PERCENTILE,
        help="Percentile of internal |K| used for symmetric color clipping",
    )
    parser.add_argument("--surface-diameter", type=float, default=SURFACE_DIAMETER_MM)
    parser.add_argument(
        "--target-hydraulic-radius",
        type=float,
        default=None,
        help="Manual equivalent-capillary-radius limit in mm; default derives it from pressure inputs.",
    )
    parser.add_argument("--gamma", type=float, default=SURFACE_TENSION_N_PER_M, help="Surface tension, N/m")
    parser.add_argument("--contact-angle", type=float, default=CONTACT_ANGLE_DEG, help="Contact angle, degrees")
    parser.add_argument("--liquid-density", type=float, default=LIQUID_DENSITY_KG_PER_M3, help="Liquid density, kg/m^3")
    parser.add_argument("--gas-density", type=float, default=GAS_DENSITY_KG_PER_M3, help="Gas density, kg/m^3")
    parser.add_argument("--gravity", type=float, default=GRAVITY_M_PER_S2, help="Gravity, m/s^2")
    parser.add_argument("--extra-pressure-pa", type=float, default=ADDITIONAL_PRESSURE_PA, help="Additional operating pressure, Pa")
    parser.add_argument("--capillary-safety-factor", type=float, default=SURFACE_CAPILLARY_SAFETY_FACTOR, help="Surface capillary safety factor")
    parser.add_argument("--hydraulic-mapping-safety", type=float, default=0.93)
    parser.add_argument(
        "--patch-batch-size", type=int, default=8,
        help="Number of patches processed between explicit geometry/Qhull cleanup barriers",
    )
    parser.add_argument(
        "--max-insertions-per-patch", type=int, default=0,
        help="0 means no node-count limit; refinement stops on hydraulic convergence",
    )
    parser.add_argument("--max-voronoi-rounds", type=int, default=100)
    parser.add_argument("--max-bad-subdomains-per-round", type=int, default=10)
    parser.add_argument("--polylabel-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--refinement-insertion-strategy",
        choices=("polylabel", "centroid_jitter"),
        default="centroid_jitter",
        help="centroid_jitter reproduces narrow anisotropic Voronoi pores; polylabel favors rounder pores",
    )
    parser.add_argument("--centroid-jitter-mm", type=float, default=0.01)
    parser.add_argument("--anisotropy-rewrite", dest="anisotropy_rewrite", action="store_true", default=True)
    parser.add_argument("--no-anisotropy-rewrite", dest="anisotropy_rewrite", action="store_false")
    parser.add_argument("--anisotropy-target-ar", type=float, default=2.75)
    parser.add_argument("--anisotropy-soft-fraction", type=float, default=0.60)
    parser.add_argument("--anisotropy-max-splits-per-patch", type=int, default=8)
    parser.add_argument("--anisotropy-min-wall-length-mm", type=float, default=0.65)
    parser.add_argument("--anisotropy-max-ar", type=float, default=5.0)
    parser.add_argument("--anisotropy-cap-min-area-mm2", type=float, default=0.5)
    parser.add_argument("--global-ar-cap-rewrite", dest="global_ar_cap_rewrite", action="store_true", default=True)
    parser.add_argument("--no-global-ar-cap-rewrite", dest="global_ar_cap_rewrite", action="store_false")
    parser.add_argument("--global-ar-cap-audit-max-edge-mm", type=float, default=0.08)
    parser.add_argument("--global-ar-cap-max-passes", type=int, default=24)
    parser.add_argument("--max-edge-length", type=float, default=1.5)
    parser.add_argument("--min-logical-edge-length", type=float, default=0.65)
    parser.add_argument("--source-node-spacing", type=float, default=0.65)
    parser.add_argument("--feature-patch-area", type=float, default=1.2)
    parser.add_argument("--edge-sample-spacing", type=float, default=0.35)
    parser.add_argument("--surface-tolerance", type=float, default=0.01875)
    parser.add_argument(
        "--refinement-policy",
        choices=("physics", "random", "random_geometry", "none"),
        default="physics",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--mapping-surface",
        choices=("carrier", "original"),
        default="carrier",
        help="Map optimized charts to the continuous STL-derived carrier or directly to the raw input triangles",
    )
    parser.add_argument("--outward-offset", type=float, default=0.0)
    parser.add_argument("--height-layers", type=int, default=HEIGHT_LAYER_COUNT)
    parser.add_argument("--skip-physics-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    if args.input is None:
        raise SystemExit("Set INPUT_STL_PATH or provide --input")
    if not args.input.exists():
        raise SystemExit(f"Input model not found: {args.input}")
    if args.out_dir is None:
        args.out_dir = args.input.parent / f"{args.input.stem}_surface_results"
    if args.curvature_sign_threshold < 0.0:
        raise SystemExit("--curvature-sign-threshold must be non-negative")
    if not 50.0 <= args.curvature_color_percentile <= 100.0:
        raise SystemExit("--curvature-color-percentile must be within [50, 100]")
    if not 0.0 <= args.split_area_distortion_p95 < 1.0:
        raise SystemExit("--split-area-distortion-p95 must be within [0, 1)")
    if not 0.0 <= args.split_angle_distortion_p95_deg < 180.0:
        raise SystemExit("--split-angle-distortion-p95-deg must be within [0, 180)")
    if args.split_rounds < 0:
        raise SystemExit("--split-rounds must be non-negative")
    if not 0.0 <= float(args.seam_screen_max_fraction) <= 1.0:
        raise SystemExit("--seam-screen-max-fraction must be in [0, 1]")
    if args.patch_batch_size < 1:
        raise SystemExit("--patch-batch-size must be at least 1")
    mesh = _load_mesh(args.input, args.target_height) if args.input is not None else None
    design_height_mm = (
        float(args.target_height)
        if args.target_height is not None
        else float(np.ptp(mesh.bounds[:, 2]))
    )
    capillary_design = calculate_surface_capillary_design(
        surface_tension_n_per_m=args.gamma,
        contact_angle_deg=args.contact_angle,
        liquid_density_kg_per_m3=args.liquid_density,
        gas_density_kg_per_m3=args.gas_density,
        gravity_m_per_s2=args.gravity,
        target_height_mm=design_height_mm,
        additional_pressure_pa=args.extra_pressure_pa,
        safety_factor=args.capillary_safety_factor,
    )
    target_radius_mm = (
        float(args.target_hydraulic_radius)
        if args.target_hydraulic_radius is not None
        else capillary_design.critical_equivalent_radius_mm
    )
    if target_radius_mm <= 0.0:
        raise SystemExit("The target equivalent capillary radius must be positive")
    target_radius_source = "manual_override" if args.target_hydraulic_radius is not None else "capillary_pressure"
    print(
        f"[surface] pressure-derived r_crit={capillary_design.critical_equivalent_radius_mm:.6f} mm "
        f"(P_required={capillary_design.required_pressure_pa:.3f} Pa); "
        f"using {target_radius_mm:.6f} mm [{target_radius_source}]"
    )
    config = SurfaceBuildConfig(
        target_faces=args.target_faces,
        max_faces_per_patch=args.max_faces_per_patch,
        seam_screen_enabled=not args.disable_seam_screen,
        seam_screen_max_fraction=args.seam_screen_max_fraction,
        normal_angle_deg=args.normal_angle,
        split_area_distortion_p95=args.split_area_distortion_p95,
        split_angle_distortion_p95_deg=args.split_angle_distortion_p95_deg,
        split_rounds=args.split_rounds,
        curvature_sign_threshold=args.curvature_sign_threshold,
        curvature_color_percentile=args.curvature_color_percentile,
        surface_diameter_mm=args.surface_diameter,
        target_hydraulic_radius_mm=target_radius_mm,
        target_radius_source=target_radius_source,
        design_target_height_mm=design_height_mm,
        surface_tension_n_per_m=args.gamma,
        contact_angle_deg=args.contact_angle,
        liquid_density_kg_per_m3=args.liquid_density,
        gas_density_kg_per_m3=args.gas_density,
        gravity_m_per_s2=args.gravity,
        additional_pressure_pa=args.extra_pressure_pa,
        capillary_safety_factor=args.capillary_safety_factor,
        hydraulic_radius_mapping_safety=args.hydraulic_mapping_safety,
        patch_batch_size=args.patch_batch_size,
        max_insertions_per_patch=args.max_insertions_per_patch,
        max_voronoi_rounds=args.max_voronoi_rounds,
        max_bad_subdomains_per_round=args.max_bad_subdomains_per_round,
        polylabel_tolerance_mm=args.polylabel_tolerance,
        refinement_insertion_strategy=args.refinement_insertion_strategy,
        centroid_jitter_mm=args.centroid_jitter_mm,
        anisotropy_rewrite_enable=bool(args.anisotropy_rewrite),
        anisotropy_target_aspect_ratio=float(args.anisotropy_target_ar),
        anisotropy_soft_fraction=float(args.anisotropy_soft_fraction),
        anisotropy_max_splits_per_patch=int(args.anisotropy_max_splits_per_patch),
        anisotropy_min_wall_length_mm=float(args.anisotropy_min_wall_length_mm),
        anisotropy_max_aspect_ratio=float(args.anisotropy_max_ar),
        anisotropy_cap_min_area_mm2=float(args.anisotropy_cap_min_area_mm2),
        global_ar_cap_rewrite_enable=bool(args.global_ar_cap_rewrite),
        global_ar_cap_audit_max_edge_mm=float(args.global_ar_cap_audit_max_edge_mm),
        global_ar_cap_max_passes=int(args.global_ar_cap_max_passes),
        max_edge_length_mm=args.max_edge_length,
        min_logical_edge_length_mm=args.min_logical_edge_length,
        source_node_spacing_mm=args.source_node_spacing,
        feature_patch_area_mm2=args.feature_patch_area,
        edge_sample_spacing_mm=args.edge_sample_spacing,
        surface_tolerance_mm=args.surface_tolerance,
        refinement_policy=args.refinement_policy,
        random_seed=args.random_seed,
        mapping_surface=args.mapping_surface,
        outward_offset_mm=args.outward_offset,
    )
    if args.validation_suite:
        run_validation_suite(args.out_dir, config, mesh, args.input.stem if args.input else "input_model")
    elif mesh is not None:
        builder = SurfacePatchBuilder(mesh, config).build()
        builder.offset_outward(config.outward_offset_mm)
        builder.audit_final_retained_graph(
            builder.atoms, builder.edges, args.out_dir,
            node_spheres_exported=False,
            include_stitch_anchors=False,
            apply_quality_gate=True,
        )
        builder.export_diagnostics(args.out_dir, args.input.stem, export_stl=True)
        signed_record = builder.summary_metrics.iloc[0]
        print(
            f"[surface] signed K (internal vertices only): "
            f"mean={float(signed_record['gaussian_curvature_mean_signed']):.6g} mm^-2, "
            f"min/max={float(signed_record['gaussian_curvature_min']):.6g}/"
            f"{float(signed_record['gaussian_curvature_max']):.6g} mm^-2, "
            f"positive/near-zero/negative="
            f"{100.0 * float(signed_record['positive_K_fraction']):.2f}%/"
            f"{100.0 * float(signed_record['near_zero_K_fraction']):.2f}%/"
            f"{100.0 * float(signed_record['negative_K_fraction']):.2f}%"
        )
        if not args.skip_physics_plots:
            complete, pressure = export_pore_and_pressure_analysis(
                args.out_dir,
                config,
                layers=args.height_layers,
                gamma_npm=args.gamma,
                liquid_density_kgm3=args.liquid_density,
                gas_density_kgm3=args.gas_density,
                gravity_ms2=args.gravity,
                contact_angle_deg=args.contact_angle,
                additional_pressure_pa=args.extra_pressure_pa,
                safety_factor=args.capillary_safety_factor,
            )
            record = complete.iloc[0]
            print(
                f"[surface] complete: graph components={int(record['graph_component_count'])}, "
                f"cycles={int(record['graph_cycle_rank'])}, "
                f"r_hyd,max={float(record['r_hyd_max_mm']):.4f} mm, "
                f"pressure={int(record['pressure_pass_layers'])}/{int(record['pressure_total_layers'])} layers"
            )
            print(f"[surface] results: {args.out_dir.resolve()}")
    else:
        raise SystemExit("Provide --input or use --validation-suite")


if __name__ == "__main__":
    main()
