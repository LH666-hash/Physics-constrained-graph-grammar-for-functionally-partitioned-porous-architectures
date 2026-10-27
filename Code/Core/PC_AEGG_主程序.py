"""PC-AEGG dual-domain graph construction and physics validation workflow."""

from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, os, tempfile, time, math, random, warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree, Delaunay

from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon, GeometryCollection, \
    box as shapely_box
from shapely.ops import polylabel, unary_union, polygonize
from shapely import affinity

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-pc-aegg"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from tqdm import tqdm
from joblib import Parallel, delayed

from surface_patch_builder import (
    SurfaceBuildConfig,
    SurfacePatchBuilder,
    calculate_surface_capillary_design,
    export_pore_and_pressure_analysis,
    style_publication_axes,
)
from capillary_viscous_filling import (
    apply_state_updated_viscous_profile,
    section_radius4_sum_mm4,
)
from global_surface_pore_audit import global_audit as run_global_surface_pore_audit

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
})

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
IN_STL = _PACKAGE_ROOT / "Input_Geometries" / "长方体.stl"
OUT_DIR = _PACKAGE_ROOT.parent / "PC_AEGG_reproduction"
TARGET_HEIGHT_MM = 22.5

GAMMA_NPM = 0.072
CONTACT_ANGLE_DEG = 15
RHO_LIQ = 998
RHO_GAS = 0.0
G_GRAV = 9.80665

SAFETY_FACTOR = 1
SAFETY_FACTOR_BOTTOM = 1
SAFETY_FACTOR_TOP = 1
FLUID_VISCOSITY_PA_S = 0.266e-3

FLOW_RESISTANCE_CALIBRATION_FACTOR = 1.0
VISC_REF_Q_M3_S = 2.0e-7
TARGET_FILL_TIME_S = 0.0
SLICE_DZ_MM = 1.0

PRESSURE_SLICE_COUNT = max(1, int(math.ceil(float(TARGET_HEIGHT_MM) / float(SLICE_DZ_MM))))

FLOW_SECTION_MIN_COMPONENT_AREA_MM2 = 0.02

SURFACE_DESIGN_ENABLE = True
SURFACE_DIAM_MM = 0.4
INITIAL_FACE_COUNT = 3000
SURFACE_MAX_FACES_PER_PATCH = 72
SURFACE_ANISOTROPY_TARGET_AR = 2.75
SURFACE_AR_CAP_MAX = 5.0
SURFACE_AR_CAP_MIN_AREA_MM2 = 0.5
SURFACE_MIN_PORE_AREA_MM2 = 1.0e-6
SURFACE_GLOBAL_AR_CAP_REWRITE_ENABLE = True
SURFACE_GLOBAL_AUDIT_MAX_EDGE_MM = 0.08
SURFACE_SEAM_MAX_FRACTION = 0.50
SURFACE_FACE_SCAN_START = 6000
SURFACE_FACE_SCAN_STOP = 1000
SURFACE_FACE_SCAN_STEP = 500
MAX_SPLIT_ROUNDS = 1
SURFACE_EXTRA_PRESSURE_PA = 0.0
SURFACE_CAPILLARY_SAFETY_FACTOR = 1.0
SURFACE_RADIUS_SOURCE = "capillary_pressure"
_DEFAULT_SURFACE_CAPILLARY_DESIGN = calculate_surface_capillary_design(
    surface_tension_n_per_m=GAMMA_NPM,
    contact_angle_deg=CONTACT_ANGLE_DEG,
    liquid_density_kg_per_m3=RHO_LIQ,
    gas_density_kg_per_m3=RHO_GAS,
    gravity_m_per_s2=G_GRAV,
    target_height_mm=TARGET_HEIGHT_MM,
    additional_pressure_pa=SURFACE_EXTRA_PRESSURE_PA,
    safety_factor=SURFACE_CAPILLARY_SAFETY_FACTOR,
)
R_EFF_LIMIT_MM = _DEFAULT_SURFACE_CAPILLARY_DESIGN.critical_equivalent_radius_mm


def _effective_surface_split_rounds() -> int:
    """Return the split-round budget actually used by the surface builder."""
    return max(6, int(MAX_SPLIT_ROUNDS))


def _surface_capillary_design():
    return calculate_surface_capillary_design(
        surface_tension_n_per_m=GAMMA_NPM,
        contact_angle_deg=CONTACT_ANGLE_DEG,
        liquid_density_kg_per_m3=RHO_LIQ,
        gas_density_kg_per_m3=RHO_GAS,
        gravity_m_per_s2=G_GRAV,
        target_height_mm=TARGET_HEIGHT_MM,
        additional_pressure_pa=SURFACE_EXTRA_PRESSURE_PA,
        safety_factor=SURFACE_CAPILLARY_SAFETY_FACTOR,
    )


INTERNAL_PITCH_BOTTOM_MM = 0.8
INTERNAL_PITCH_TOP_MM = 0.8
INTERNAL_DIAM_BOTTOM_MM = 0.40
INTERNAL_DIAM_TOP_MM = 0.40
Z_SAME_TOLERANCE_MM = 0.2
SEARCH_RADIUS_FACTOR = 2.2

INTERNAL_BUILD_MODE = "dynamic"
INTERNAL_DYNAMIC_LAYERS = 20
INTERNAL_DYNAMIC_SEED = 42
INTERNAL_MATCHED_RANDOM_POSITION = False
INTERNAL_MATCHED_RANDOM_NODE_BUDGET = 0
INTERNAL_MATCHED_RANDOM_EDGE_BUDGET = 0
INTERNAL_DYNAMIC_MIN_NODES_PER_LAYER = 20
INTERNAL_DYNAMIC_MAX_NODES_PER_LAYER = 0
INTERNAL_DYNAMIC_BOTTOM_NODE_TARGET = 32
INTERNAL_DYNAMIC_TOP_NODE_TARGET = 110
INTERNAL_DYNAMIC_DENSITY_FACTOR = 0.09

INTERNAL_DYNAMIC_INITIAL_DENSITY_MULTIPLIER = 0.9
INTERNAL_DYNAMIC_INITIAL_DENSITY_SOURCE = "adaptive_default"
INTERNAL_DYNAMIC_AREA_TARGET_EXPONENT = 0.62
INTERNAL_DYNAMIC_AREA_TARGET_MIN_SCALE = 0.20
INTERNAL_DYNAMIC_MIN_NODES_PER_COMPONENT = 1
INTERNAL_DYNAMIC_MIN_COMPONENT_AREA_MM2 = 2.0
INTERNAL_DYNAMIC_COMPONENT_TRIES_BASE = 120
INTERNAL_DYNAMIC_COMPONENT_TRIES_PER_NODE = 50
INTERNAL_DYNAMIC_COMPONENT_CAPACITY_FACTOR = 0.70
INTERNAL_DYNAMIC_STEP1_FRACTION = 0.60
INTERNAL_DYNAMIC_STEP2_FRACTION = 0.40
INTERNAL_DYNAMIC_MAX_TRIES_PER_LAYER = 2800
INTERNAL_DYNAMIC_CANDIDATES = 56
INTERNAL_DYNAMIC_MIN_DIST_FACTOR = 0.26
INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR = 0.08
INTERNAL_DYNAMIC_NEIGHBOR_K = 56
INTERNAL_DYNAMIC_SEARCH_FACTOR = 4.2
INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW = 8
INTERNAL_DYNAMIC_FORCE_SAME_LAYER_EDGES = 2
INTERNAL_DYNAMIC_FORCE_LOWER_EDGES = 3
INTERNAL_DYNAMIC_FORCE_UPPER_EDGES = 3
INTERNAL_DYNAMIC_NORMAL_SAME_EDGES = 2
INTERNAL_DYNAMIC_NORMAL_LOWER_EDGES = 3
INTERNAL_DYNAMIC_NORMAL_UPPER_EDGES = 3
INTERNAL_DYNAMIC_LAYER_Z_JITTER_FRAC = 0.5
INTERNAL_DYNAMIC_LAYER_INTRA_MIN_DZ_MM = 0.06
INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR = 3.8
INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW = 2
INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC = 4.2
INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG = 20.0

INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM = 2.5
INTERNAL_HORIZONTAL_BRIDGE_ENABLE = True
INTERNAL_REQUIRE_BOTTOM_SUPPORT = False
INTERNAL_DYNAMIC_ALLOW_EDGELESS_FOR_MIN = True

INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_ENABLE = True
INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_REJECTIONS = 8
INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_FRACTION = 0.10
INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MAX_TARGETS_PER_LAYER = 24
INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_DEDUP_FACTOR = 0.40

INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_PEERS = 6
INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_LENGTH_MM = 3.50
INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_DEGREE_RESERVE = 8
INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_FRACTION = 0.15
INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_MIN_NODES = 8
INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_MAX_NODES = 18
INTERNAL_DYNAMIC_PHYSICS_DZ_MM = max(1.0, SLICE_DZ_MM)

INTERNAL_DYNAMIC_GLOBAL_FEEDBACK_SPACING_MM = SLICE_DZ_MM
INTERNAL_DYNAMIC_UPPER_FEEDBACK_DEPTH_MM = 5.0
INTERNAL_DYNAMIC_UPPER_FEEDBACK_SPACING_MM = SLICE_DZ_MM
INTERNAL_DYNAMIC_FAILURE_BAND_MERGE_MM = 0.15

INTERNAL_DYNAMIC_PHYS_GAPFILL_ENABLE = True
INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_TOTAL_ADDS = 0
INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_ADDS_PER_INTERVAL = 120
INTERNAL_DYNAMIC_PHYS_GAPFILL_MIN_Z_FRAC = 0.04
INTERNAL_DYNAMIC_PHYS_GAPFILL_BASE_BIAS = 0.90
INTERNAL_DYNAMIC_PHYS_GAPFILL_RELDEF_CAP = 3.0

INTERNAL_DYNAMIC_SMARTFILL_MAX_PRESSURE_EVALUATIONS = 1500
INTERNAL_DYNAMIC_SMARTFILL_PER_ROUND = 2
INTERNAL_DYNAMIC_SMARTFILL_MAX_TOTAL_ADDS = 0
INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND = 40
INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_SLICE = 0

INTERNAL_DYNAMIC_SMARTFILL_MAX_SAFETY_ROUNDS = 0
INTERNAL_DYNAMIC_SMARTFILL_MAX_TRIALS_PER_SLICE = 0

INTERNAL_DYNAMIC_SMARTFILL_MAX_ACCEPTED_PER_SLICE = 0
INTERNAL_DYNAMIC_SMARTFILL_MIN_DEFICIT_DROP_PA = 0.05
INTERNAL_DYNAMIC_SMARTFILL_MIN_REL_DEFICIT_DROP = 0.005

INTERNAL_DYNAMIC_SMARTFILL_HALF_GAIN_MARGIN_PA = -20.0
INTERNAL_DYNAMIC_SMARTFILL_NEAR_PASS_GAIN_FACTOR = 0.5
INTERNAL_DYNAMIC_SMARTFILL_TOP_FAIL_SLICES = 25

INTERNAL_DYNAMIC_CHANNEL_REPAIR_MAX_EVALUATIONS = 12

INTERNAL_DYNAMIC_VISCOUS_RELIEF_ENABLE = True
INTERNAL_DYNAMIC_VISCOUS_RELIEF_MAX_EDGE_EVALUATIONS_PER_SEARCH = 12
INTERNAL_DYNAMIC_VISCOUS_RELIEF_MAX_FAILED_SEARCHES_PER_BAND = 5
INTERNAL_DYNAMIC_VISCOUS_RELIEF_PATH_TOP_SEGMENTS = 6
INTERNAL_DYNAMIC_VISCOUS_RELIEF_MOBILITY_MEDIAN_RATIO = 0.75
INTERNAL_DYNAMIC_VISCOUS_RELIEF_MIN_VISCOUS_SHARE = 0.55
INTERNAL_DYNAMIC_VISCOUS_RELIEF_MIN_PRIOR_INSERTION_FAILURES = 2
INTERNAL_DYNAMIC_TOP_GEOMETRY_SCAN_LIMIT = 32
INTERNAL_DYNAMIC_SUPPORT_REPAIR_MAX_EDGES = 160

INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE = 2

INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE = 2

INTERNAL_DYNAMIC_SMARTFILL_SUPPORT_FIRST_ENABLE = False
INTERNAL_TOP_SUPPORT_FIRST_ENABLE = False
INTERNAL_TOP_SUPPORT_CANDIDATES_PER_INSERT = 24
INTERNAL_TOP_SUPPORT_ANCHORS_TO_SEARCH = 48

INTERNAL_TOP_SUPPORT_ROOF_PROBE_OFFSETS_MM = (0.25, 0.10, 0.05)

INTERNAL_TOP_COVERAGE_DIAGNOSTIC_OFFSETS_MM = (0.25, 0.10, 0.05)

INTERNAL_DYNAMIC_SMARTFILL_INCREMENTAL_PRESSURE_ENABLE = True

INTERNAL_DYNAMIC_LOCAL_GEOMETRY_REPAIR_ENABLE = True
INTERNAL_DYNAMIC_MOVE_NODE_MAX_TRIALS_PER_BAND = 20
INTERNAL_DYNAMIC_MOVE_NODE_MAX_CANDIDATES_PER_SEARCH = 5
INTERNAL_DYNAMIC_MOVE_NODE_MAX_DISTANCE_PITCH_FACTOR = 0.35
INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_TRIALS_PER_BAND = 12
INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_CANDIDATES_PER_SEARCH = 5
INTERNAL_DYNAMIC_THICKEN_EDGE_STEP_FACTOR = 1.05
INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_RADIUS_FACTOR = 1.20
INTERNAL_DYNAMIC_LOCAL_REPAIR_MAX_VISCOUS_SHARE = 0.55

INTERNAL_DYNAMIC_SMARTFILL_PARTIAL_SALVAGE_ENABLE = True
INTERNAL_DYNAMIC_SMARTFILL_MAX_SALVAGE_EVALUATIONS_PER_ROUND = 80
INTERNAL_DYNAMIC_MIN_LIQUID_FRAC = 0.03
INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2 = 2.0
INTERNAL_DYNAMIC_OVERFILL_POROSITY_FLOOR = 0.10
INTERNAL_DYNAMIC_TOP_POROSITY_FLOOR = 0.10
INTERNAL_DYNAMIC_TOP_GUARD_START_FRAC = 0.82
INTERNAL_DYNAMIC_TOP_MAX_ADDS_PER_SLICE = 0

INTERNAL_DYNAMIC_TOP_MAX_EDGES_PER_NODE = INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW

INTERNAL_DYNAMIC_TOP_SAME_LAYER_EDGES = 9

INTERNAL_DYNAMIC_TOP_PREFILL_BRIDGES_PER_SLICE = 8
INTERNAL_DYNAMIC_TOP_PREFILL_MAX_TRIALS_PER_SLICE = 4

INTERNAL_TOP_RELAX_ACTIVE_HEIGHT_MM = 1.50
INTERNAL_TOP_RELAX_ENDPOINT_HEIGHT_MM = 2.50
INTERNAL_TOP_RELAX_MIN_ELEVATION_DEG = 0.0
INTERNAL_TOP_RELAX_MAX_EDGE_LENGTH_MM = 2.00
INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO = 1.25
INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO = 1.15
INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO = 1.25
INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM = 5.0
INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO = 0.02
INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_THRESHOLD = 0.20
INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_MAX_ROUNDS = 120
INTERNAL_DYNAMIC_SMOOTH_PRUNE_MAX_REMOVALS = 64
INTERNAL_DYNAMIC_SMOOTH_PRUNE_BATCH_SIZE = 18
INTERNAL_DYNAMIC_FINAL_TRIM_MAX_EVALUATIONS = 200
INTERNAL_DYNAMIC_PRUNE_MAX_SECONDS = 600.0
INTERNAL_DYNAMIC_PRUNE_MAX_STALL_EVALUATIONS = 8
INTERNAL_DYNAMIC_PRUNE_MIN_RELATIVE_IMPROVEMENT = 1.0e-4
INTERNAL_DYNAMIC_XZ_BINS = 12
INITIAL_SKELETON_CHECKPOINT_LOAD: Optional[Path] = None
INITIAL_SKELETON_CHECKPOINT_SAVE: Optional[Path] = None
PRE_SMARTFILL_CHECKPOINT_LOAD: Optional[Path] = None
PRE_SMARTFILL_CHECKPOINT_SAVE: Optional[Path] = None
CHECKPOINT_STOP_AFTER: Optional[str] = None
CHECKPOINT_IGNORE_CONFIG = False

FIGURE_HEX_COLORS = ['#2171B5', '#BCC0DF', '#FFCFCF']
FIGURE_BLUE = '#2171B5'
FIGURE_MID = '#BCC0DF'
FIGURE_PINK = '#FFCFCF'
FIGURE_RED = '#D86C6C'
FIGURE_GREY = '#666666'
FIGURE_CMAP = LinearSegmentedColormap.from_list("MyCustomMap", FIGURE_HEX_COLORS)

STITCH_DISTANCE_MM = 2.0
SURFACE_OUTWARD_OFFSET_MM = 0.0
STITCH_ANCHOR_SPACING_MM = 1.50
STITCH_RELAXED_DISTANCE_MM = 2.40
STITCH_CANDIDATE_K = 48
STITCH_ALTERNATIVE_ANCHORS = 12

BOTTOM_PILLAR_ENABLE = True
BOTTOM_PILLAR_ROWS = 4
BOTTOM_PILLAR_MAX_ANCHORS = 36
BOTTOM_PILLAR_ANCHOR_SPACING_MM = 1.00
BOTTOM_PILLAR_HEIGHT_LAYERS = 2.0
BOTTOM_PILLAR_RADIUS_FACTOR = 1.0
BOTTOM_PILLAR_MIN_COMPONENT_AREA_MM2 = 2.0

NODE_DIAM_RATIO = 1.2
STL_ROD_RADIAL_SECTIONS = 24
STL_NODE_SPHERE_SUBDIVISIONS = 1


def log(msg):
    prefix = time.strftime("[%H:%M:%S] ")
    try:
        print(prefix, msg, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(getattr(os, "device_encoding", lambda _: None)(1), "lower", lambda: None)() or "utf-8"
        safe = str(msg).encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(prefix, safe, flush=True)


class WorkflowCheckpointStop(RuntimeError):
    """Successful early stop requested immediately after saving a checkpoint."""


def _checkpoint_json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"checkpoint value is not JSON serializable: {type(value).__name__}")


_INITIAL_CHECKPOINT_CONFIG_NAMES = (
    "TARGET_HEIGHT_MM", "INTERNAL_DYNAMIC_LAYERS", "INTERNAL_DYNAMIC_SEED",
    "INTERNAL_DYNAMIC_MIN_NODES_PER_LAYER", "INTERNAL_DYNAMIC_MAX_NODES_PER_LAYER",
    "INTERNAL_DYNAMIC_BOTTOM_NODE_TARGET", "INTERNAL_DYNAMIC_TOP_NODE_TARGET",
    "INTERNAL_DYNAMIC_DENSITY_FACTOR", "INTERNAL_DYNAMIC_INITIAL_DENSITY_MULTIPLIER",
    "INTERNAL_DYNAMIC_AREA_TARGET_EXPONENT", "INTERNAL_DYNAMIC_AREA_TARGET_MIN_SCALE",
    "INTERNAL_DYNAMIC_MIN_NODES_PER_COMPONENT", "INTERNAL_DYNAMIC_MIN_COMPONENT_AREA_MM2",
    "INTERNAL_DYNAMIC_COMPONENT_TRIES_BASE", "INTERNAL_DYNAMIC_COMPONENT_TRIES_PER_NODE",
    "INTERNAL_DYNAMIC_COMPONENT_CAPACITY_FACTOR", "INTERNAL_DYNAMIC_STEP1_FRACTION",
    "INTERNAL_DYNAMIC_MAX_TRIES_PER_LAYER", "INTERNAL_DYNAMIC_CANDIDATES",
    "INTERNAL_DYNAMIC_MIN_DIST_FACTOR", "INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR",
    "INTERNAL_DYNAMIC_NEIGHBOR_K", "INTERNAL_DYNAMIC_SEARCH_FACTOR",
    "INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW", "INTERNAL_DYNAMIC_FORCE_SAME_LAYER_EDGES",
    "INTERNAL_DYNAMIC_FORCE_LOWER_EDGES", "INTERNAL_DYNAMIC_FORCE_UPPER_EDGES",
    "INTERNAL_DYNAMIC_NORMAL_SAME_EDGES", "INTERNAL_DYNAMIC_NORMAL_LOWER_EDGES",
    "INTERNAL_DYNAMIC_NORMAL_UPPER_EDGES", "INTERNAL_DYNAMIC_LAYER_Z_JITTER_FRAC",
    "INTERNAL_DYNAMIC_LAYER_INTRA_MIN_DZ_MM", "INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR",
    "INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW", "INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC",
    "INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG", "INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM",
    "INTERNAL_HORIZONTAL_BRIDGE_ENABLE", "INTERNAL_DYNAMIC_ALLOW_EDGELESS_FOR_MIN",
    "INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_ENABLE",
    "INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_REJECTIONS",
    "INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_FRACTION",
    "INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MAX_TARGETS_PER_LAYER",
    "INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_DEDUP_FACTOR",
    "INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_PEERS",
    "INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_LENGTH_MM",
    "INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_DEGREE_RESERVE",
    "INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_FRACTION",
    "INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_MIN_NODES",
    "INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_MAX_NODES",
    "INTERNAL_PITCH_BOTTOM_MM", "INTERNAL_PITCH_TOP_MM",
    "INTERNAL_DIAM_BOTTOM_MM", "INTERNAL_DIAM_TOP_MM",
)

_PRE_SMARTFILL_EXTRA_CONFIG_NAMES = (
    "BOTTOM_PILLAR_ENABLE", "BOTTOM_PILLAR_ROWS", "BOTTOM_PILLAR_MAX_ANCHORS",
    "BOTTOM_PILLAR_ANCHOR_SPACING_MM", "BOTTOM_PILLAR_HEIGHT_LAYERS",
    "BOTTOM_PILLAR_RADIUS_FACTOR", "BOTTOM_PILLAR_MIN_COMPONENT_AREA_MM2",
    "INTERNAL_DYNAMIC_PHYS_GAPFILL_ENABLE", "INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_ADDS_PER_INTERVAL",
    "INTERNAL_DYNAMIC_PHYS_GAPFILL_MIN_Z_FRAC", "INTERNAL_DYNAMIC_PHYS_GAPFILL_BASE_BIAS",
    "INTERNAL_DYNAMIC_PHYS_GAPFILL_RELDEF_CAP", "INTERNAL_DYNAMIC_SUPPORT_REPAIR_MAX_EDGES",
    "INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE", "INTERNAL_REQUIRE_BOTTOM_SUPPORT",
    "INTERNAL_DYNAMIC_MIN_LIQUID_FRAC", "INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2",
    "INTERNAL_DYNAMIC_OVERFILL_POROSITY_FLOOR", "INTERNAL_DYNAMIC_TOP_POROSITY_FLOOR",
    "INTERNAL_DYNAMIC_TOP_GUARD_START_FRAC", "INTERNAL_DYNAMIC_CHANNEL_REPAIR_MAX_EVALUATIONS",
    "GAMMA_NPM", "CONTACT_ANGLE_DEG", "RHO_LIQ",
    "RHO_GAS", "G_GRAV", "SAFETY_FACTOR", "SAFETY_FACTOR_BOTTOM", "SAFETY_FACTOR_TOP",
    "FLUID_VISCOSITY_PA_S", "FLOW_RESISTANCE_CALIBRATION_FACTOR", "VISC_REF_Q_M3_S",
    "TARGET_FILL_TIME_S", "SLICE_DZ_MM",
    "INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO", "INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO",
    "INTERNAL_DYNAMIC_LOCAL_GEOMETRY_REPAIR_ENABLE",
    "INTERNAL_DYNAMIC_MOVE_NODE_MAX_TRIALS_PER_BAND",
    "INTERNAL_DYNAMIC_MOVE_NODE_MAX_CANDIDATES_PER_SEARCH",
    "INTERNAL_DYNAMIC_MOVE_NODE_MAX_DISTANCE_PITCH_FACTOR",
    "INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_TRIALS_PER_BAND",
    "INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_CANDIDATES_PER_SEARCH",
    "INTERNAL_DYNAMIC_THICKEN_EDGE_STEP_FACTOR",
    "INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_RADIUS_FACTOR",
    "INTERNAL_DYNAMIC_LOCAL_REPAIR_MAX_VISCOUS_SHARE",
)


def _checkpoint_config(stage: str) -> Dict:
    names = list(_INITIAL_CHECKPOINT_CONFIG_NAMES)
    if stage == "pre-smartfill":
        names.extend(_PRE_SMARTFILL_EXTRA_CONFIG_NAMES)
    return {name: globals()[name] for name in names}


def _checkpoint_mesh_signature(mesh, start_id: int) -> Dict:
    payload = {
        "bounds_mm": np.round(np.asarray(mesh.bounds, dtype=float), 9).tolist(),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "start_id": int(start_id),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _save_internal_checkpoint(path: Path, stage: str, mesh, start_id: int, *,
                              all_atoms: pd.DataFrame, all_edges: List[Dict],
                              internal_ids: Set[int], supported_internal_ids: Set[int],
                              layer_nodes: List[List[int]], next_id: int, global_step: int,
                              process_log: List[Dict], internal_log: List[Dict],
                              initial_rejection_counts: Dict[str, int],
                              initial_rejection_details: Optional[List[Dict]] = None) -> None:
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    all_atoms.to_csv(folder / "atoms.csv", index=False)
    pd.DataFrame(_edge_records(all_edges)).to_csv(folder / "edges.csv", index=False)
    state = {
        "format_version": 1, "stage": stage,
        "mesh": _checkpoint_mesh_signature(mesh, start_id),
        "config": _checkpoint_config(stage),
        "internal_ids": sorted(map(int, internal_ids)),
        "supported_internal_ids": sorted(map(int, supported_internal_ids)),
        "layer_nodes": [[int(node_id) for node_id in nodes] for nodes in layer_nodes],
        "next_id": int(next_id), "global_step": int(global_step),
        "process_log": process_log, "internal_log": internal_log,
        "initial_rejection_counts": initial_rejection_counts,
        "initial_rejection_details": list(initial_rejection_details or []),
    }
    (folder / "checkpoint.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=_checkpoint_json_value),
        encoding="utf-8")
    log(f"  Saved {stage} checkpoint: {folder}")


def _load_internal_checkpoint(path: Path, expected_stage: str, mesh, start_id: int,
                              *, allow_internal_id_remap: bool = False) -> Dict:
    folder = Path(path)
    manifest_path, atoms_path, edges_path = (
        folder / "checkpoint.json", folder / "atoms.csv", folder / "edges.csv")
    missing = [str(p) for p in (manifest_path, atoms_path, edges_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("checkpoint is incomplete; missing: " + ", ".join(missing))
    state = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(state.get("format_version", -1)) != 1 or state.get("stage") != expected_stage:
        raise ValueError(
            f"checkpoint stage/version mismatch: expected {expected_stage!r} format 1, "
            f"found {state.get('stage')!r} format {state.get('format_version')!r}")
    saved_mesh = state.get("mesh", {})
    saved_start_id = int(saved_mesh.get("start_id", start_id))
    saved_start_signature = _checkpoint_mesh_signature(mesh, saved_start_id)
    if saved_mesh.get("sha256") != saved_start_signature["sha256"]:
        raise ValueError("checkpoint does not match the scaled input mesh")
    expected_config = _checkpoint_config(expected_stage)
    if state.get("config") != expected_config:
        saved = state.get("config", {})
        changed = sorted(name for name in set(saved) | set(expected_config)
                         if saved.get(name) != expected_config.get(name))
        if not CHECKPOINT_IGNORE_CONFIG:
            raise ValueError(
                "checkpoint pre-stage configuration differs for: " + ", ".join(changed)
                + "; use matching options or --checkpoint-ignore-config")
        log("  WARNING: checkpoint configuration mismatch ignored for: " + ", ".join(changed))
    atoms, edges_df = pd.read_csv(atoms_path), pd.read_csv(edges_path)
    id_offset = int(start_id) - saved_start_id
    if id_offset:
        if not allow_internal_id_remap:
            raise ValueError(
                "checkpoint graph start_id differs; enable surface-aware internal ID remapping")
        face_names = atoms.get("face_name", pd.Series(dtype=str)).astype(str).str.lower()
        bond_types = edges_df.get("bond_type", pd.Series(dtype=str)).astype(str).str.lower()
        if face_names.str.contains("surface", na=False).any() or bond_types.isin(
                {"surface", "stitch", "surface_internal"}).any():
            raise ValueError(
                "checkpoint already contains surface/stitch elements and cannot be rebased safely")
        atoms["atom_id"] = pd.to_numeric(atoms["atom_id"], errors="raise").astype(int) + id_offset
        for col in ("source", "target"):
            if col in edges_df.columns:
                edges_df[col] = pd.to_numeric(edges_df[col], errors="raise").astype(int) + id_offset
        state["internal_ids"] = [int(node_id) + id_offset
                                 for node_id in state.get("internal_ids", [])]
        state["supported_internal_ids"] = [int(node_id) + id_offset
                                           for node_id in state.get("supported_internal_ids", [])]
        state["layer_nodes"] = [
            [int(node_id) + id_offset for node_id in nodes]
            for nodes in state.get("layer_nodes", [])]
        state["next_id"] = int(state["next_id"]) + id_offset
        state["id_remap_offset"] = id_offset
        log(f"  Rebased surface-free checkpoint IDs by +{id_offset} "
            f"(saved start={saved_start_id}, surface-aware start={start_id}).")
    state["all_atoms"], state["all_edges"] = atoms, _edge_records(edges_df)
    log(f"  Loaded {expected_stage} checkpoint: {folder} "
        f"(atoms={len(atoms)}, edges={len(edges_df)})")
    return state


def _save_svg_only(fig, path: Path) -> Path:
    """Save one editable SVG and remove obsolete duplicate raster/PDF exports."""
    stem = Path(path).with_suffix("")
    svg_path = stem.with_suffix(".svg")
    style_publication_axes(fig)
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    for suffix in (".png", ".pdf", ".tiff", ".tif"):
        stem.with_suffix(suffix).unlink(missing_ok=True)
    return svg_path


def load_and_scale_mesh(path: Path, target_h_mm: float | None):
    if not path.exists():
        raise FileNotFoundError(f"Input STL not found: {path}")
    else:
        log(f"Loading mesh: {path.name}...")
        mesh = trimesh.load(str(path), force='mesh')

    mesh.merge_vertices()
    mesh.fix_normals()

    min_x, min_y, min_z = mesh.bounds[0]
    mesh.apply_translation([-min_x, -min_y, -min_z])
    log(f"Aligned mesh bounding box minimum to origin (0, 0, 0).")

    if target_h_mm is not None:
        zmin, zmax = mesh.bounds[0, 2], mesh.bounds[1, 2]
        scale = target_h_mm / max(zmax - zmin, 1e-9)
        mesh.apply_scale(scale)
        log(f"Scaled to Height: {target_h_mm:.2f}mm")

    return mesh, 1.0


def snap_points_to_surface(mesh, points):
    if len(points) == 0: return np.array([])
    closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, points)
    return closest


def _safety_factor_at_z(z: float, z_bounds: Optional[Tuple[float, float]] = None) -> float:
    if z_bounds is None:
        return float(SAFETY_FACTOR)
    zmin, zmax = float(z_bounds[0]), float(z_bounds[1])
    t = 0.0 if zmax <= zmin + 1e-12 else (float(z) - zmin) / (zmax - zmin)
    t = float(np.clip(t, 0.0, 1.0))
    return float(SAFETY_FACTOR_BOTTOM * (1.0 - t) + SAFETY_FACTOR_TOP * t)


def barycentric_map_3d(p2d, tri2d, tri3d):
    x, y = p2d
    x1, y1 = tri2d[0]
    x2, y2 = tri2d[1]
    x3, y3 = tri2d[2]

    detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(detT) < 1e-9: return np.mean(tri3d, axis=0)

    u = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / detT
    v = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / detT
    w = 1.0 - u - v

    p3d = u * tri3d[0] + v * tri3d[1] + w * tri3d[2]
    return p3d


def analyze_triangle_pore(p0, p1, p2):
    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p0 - p2)

    if a < 1e-6 or b < 1e-6 or c < 1e-6: return 0.0, None

    s = (a + b + c) / 2.0
    area_sq = max(0, s * (s - a) * (s - b) * (s - c))
    raw_area = np.sqrt(area_sq)
    raw_peri = a + b + c
    if raw_peri < 1e-9: return 0.0, None

    raw_r_eff = (2.0 * raw_area) / raw_peri
    if raw_r_eff < R_EFF_LIMIT_MM:
        return raw_r_eff, None

    p2_x = (a ** 2 + c ** 2 - b ** 2) / (2 * a)
    term = max(0, c ** 2 - p2_x ** 2)
    p2_y = np.sqrt(term)

    tri_2d = [(0.0, 0.0), (a, 0.0), (p2_x, p2_y)]
    poly = Polygon(tri_2d)

    poly_liq = poly.buffer(-SURFACE_DIAM_MM / 2.0, resolution=4)

    if poly_liq.is_empty:
        return 0.0, None

    real_area = poly_liq.area
    real_peri = poly_liq.length
    real_r_eff = (2.0 * real_area) / real_peri if real_peri > 1e-9 else 0.0

    split_pt_3d = None
    if real_r_eff > R_EFF_LIMIT_MM:
        try:
            pt_ins = polylabel(poly_liq, tolerance=0.01)
        except:
            pt_ins = poly_liq.centroid

        split_pt_3d = barycentric_map_3d(
            [pt_ins.x, pt_ins.y],
            tri_2d,
            np.array([p0, p1, p2])
        )

    return real_r_eff, split_pt_3d


class SurfaceMeshSystem:
    def __init__(self, original_mesh):
        self.original_mesh = original_mesh
        self.curr_vertices = []
        self.curr_faces = []
        self.curr_edges = None

    def initialize_by_decimation(self, target_faces):
        try:
            self.initialize_by_curvature_patches(target_faces)
            return
        except Exception as exc:
            log(f"  -> Curvature patch surface failed, fallback to decimation: {exc}")
        log(f"Initializing Surface by Decimation ({target_faces} faces)...")
        simplified = self.original_mesh.simplify_quadric_decimation(face_count=target_faces)
        self.curr_vertices = simplified.vertices.tolist()
        self.curr_faces = simplified.faces.tolist()
        self.curr_edges = None
        log(f"  -> Surface Nodes: {len(self.curr_vertices)}, Faces: {len(self.curr_faces)}")

    def initialize_by_curvature_patches(self, target_faces):
        log(f"Initializing Surface by curvature patches ({target_faces} base faces)...")
        base = self.original_mesh.simplify_quadric_decimation(face_count=max(int(target_faces), 80))
        verts = np.asarray(base.vertices, dtype=float)
        faces = np.asarray(base.faces, dtype=int)
        normals = np.asarray(base.face_normals, dtype=float)
        adjacency = [[] for _ in range(len(faces))]
        for a, b in np.asarray(base.face_adjacency, dtype=int):
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
        cos_thr = math.cos(math.radians(22.5))
        visited = np.zeros(len(faces), dtype=bool)
        patches = []
        for seed in range(len(faces)):
            if visited[seed]:
                continue
            queue = [seed]
            visited[seed] = True
            group = []
            while queue:
                f = queue.pop()
                group.append(f)
                for nb in adjacency[f]:
                    if visited[nb]:
                        continue
                    if float(np.dot(normals[f], normals[nb])) >= cos_thr:
                        visited[nb] = True
                        queue.append(nb)
            patches.append(group)

        all_points = []
        all_edges = set()
        point_key_to_id: Dict[Tuple[int, int, int], int] = {}

        def add_point(p):
            key = tuple(np.round(np.asarray(p, dtype=float) * 10000).astype(int).tolist())
            if key in point_key_to_id:
                return point_key_to_id[key]
            point_key_to_id[key] = len(all_points)
            all_points.append(np.asarray(p, dtype=float))
            return point_key_to_id[key]

        for patch in patches:
            patch_faces = faces[np.asarray(patch, dtype=int)]
            patch_vertex_ids = sorted(set(int(v) for tri in patch_faces for v in tri))
            pts3 = verts[patch_vertex_ids]
            if len(pts3) < 3:
                continue
            centroid = pts3.mean(axis=0)
            uu, ss, vh = np.linalg.svd(pts3 - centroid, full_matrices=False)
            ax0, ax1 = vh[0], vh[1]
            pts2 = np.column_stack(((pts3 - centroid) @ ax0, (pts3 - centroid) @ ax1))
            edge_count: Dict[Tuple[int, int], int] = {}
            for tri in patch_faces:
                for a, b in [(int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))]:
                    e = tuple(sorted((a, b)))
                    edge_count[e] = edge_count.get(e, 0) + 1
            boundary_edges = [e for e, c in edge_count.items() if c == 1]
            id_to_local = {vid: i for i, vid in enumerate(patch_vertex_ids)}
            lines = []
            for a, b in boundary_edges:
                if a in id_to_local and b in id_to_local:
                    lines.append(LineString([tuple(pts2[id_to_local[a]]), tuple(pts2[id_to_local[b]])]))
            polys = list(polygonize(lines))
            if not polys:
                try:
                    hull = MultiPoint([Point(float(x), float(y)) for x, y in pts2]).convex_hull
                    polys = [hull] if isinstance(hull, Polygon) else []
                except Exception:
                    polys = []
            if not polys:
                continue
            poly2 = max(polys, key=lambda p: p.area)
            node2 = [tuple(v) for v in pts2]
            liq = poly2.buffer(-SURFACE_DIAM_MM / 2.0, resolution=4)
            step = 0
            while (not liq.is_empty) and step < MAX_SPLIT_ROUNDS * 8:
                step += 1
                try:
                    c = polylabel(liq, tolerance=0.03)
                except Exception:
                    c = liq.representative_point()
                if liq.boundary.distance(c) <= R_EFF_LIMIT_MM:
                    break
                node2.append((float(c.x), float(c.y)))
                liq = liq.difference(Point(float(c.x), float(c.y)).buffer(SURFACE_DIAM_MM / 2.0, resolution=8))
            node2_arr = np.asarray(node2, dtype=float)
            if len(node2_arr) < 3:
                continue
            try:
                tri = Delaunay(node2_arr)
                local_edges = set()
                for simplex in tri.simplices:
                    for ia, ib in [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]:
                        a2, b2 = node2_arr[int(ia)], node2_arr[int(ib)]
                        seg = LineString([tuple(a2), tuple(b2)])
                        if poly2.buffer(1e-7).covers(seg):
                            local_edges.add(tuple(sorted((int(ia), int(ib)))))
            except Exception:
                local_edges = set()
                for i in range(len(node2_arr) - 1):
                    local_edges.add((i, i + 1))
            local_to_global = []
            for p2 in node2_arr:
                p3 = centroid + p2[0] * ax0 + p2[1] * ax1
                local_to_global.append(add_point(p3))
            for ia, ib in local_edges:
                if ia != ib:
                    all_edges.add(tuple(sorted((local_to_global[ia], local_to_global[ib]))))

        if not all_points or not all_edges:
            raise RuntimeError("no curvature patch edges generated")
        snapped = snap_points_to_surface(self.original_mesh, np.asarray(all_points, dtype=float))
        self.curr_vertices = snapped.tolist()
        self.curr_faces = []
        self.curr_edges = sorted(all_edges)
        log(f"  -> Curvature patches: {len(patches)} patches, Surface Nodes: {len(self.curr_vertices)}, Edges: {len(self.curr_edges)}")

    def optimize_iteratively(self):
        if self.curr_edges is not None:
            log("  -> Curvature-patch surface already built in 2D; skip triangle split optimization.")
            return
        for round_idx in range(MAX_SPLIT_ROUNDS):
            bad_faces_indices = []
            split_points_3d = []
            stats = []

            coords = np.array(self.curr_vertices)

            for f_idx, face in enumerate(self.curr_faces):
                u, v, w = face
                p0, p1, p2 = coords[u], coords[v], coords[w]
                r_eff, split_pt = analyze_triangle_pore(p0, p1, p2)
                stats.append(r_eff)
                if split_pt is not None:
                    bad_faces_indices.append(f_idx)
                    split_points_3d.append(split_pt)

            max_r = max(stats) if stats else 0
            bad_count = len(bad_faces_indices)
            log(f"  Round {round_idx + 1}: Tris={len(self.curr_faces)}, Max r_eff={max_r:.4f}mm, Bad={bad_count}")

            if bad_count == 0:
                log("  >> Converged!")
                break

            new_pts_arr = np.array(split_points_3d)
            snapped_pts = snap_points_to_surface(self.original_mesh, new_pts_arr)

            start_id = len(self.curr_vertices)
            self.curr_vertices.extend(snapped_pts.tolist())

            new_faces_list = []
            bad_indices_set = set(bad_faces_indices)
            face_idx_to_new_pt = {old_idx: start_id + k for k, old_idx in enumerate(bad_faces_indices)}

            for f_idx, face in enumerate(self.curr_faces):
                if f_idx in bad_indices_set:
                    u, v, w = face
                    c = face_idx_to_new_pt[f_idx]
                    new_faces_list.append([u, v, c])
                    new_faces_list.append([v, w, c])
                    new_faces_list.append([w, u, c])
                else:
                    new_faces_list.append(face)

            self.curr_faces = new_faces_list

    def offset_outward(self, distance_mm):
        """Move surface nodes outward along the nearest source-face normals."""
        if distance_mm <= 0 or not self.curr_vertices:
            return

        coords = np.asarray(self.curr_vertices, dtype=float)
        closest, _, triangle_ids = trimesh.proximity.closest_point(
            self.original_mesh, coords
        )
        normals = np.asarray(self.original_mesh.face_normals[triangle_ids], dtype=float)
        normal_lengths = np.linalg.norm(normals, axis=1)
        valid = np.isfinite(normal_lengths) & (normal_lengths > 1e-12)

        unit_normals = np.zeros_like(normals)
        unit_normals[valid] = normals[valid] / normal_lengths[valid, None]
        moved = closest.copy()
        moved[valid] += unit_normals[valid] * distance_mm
        moved[~valid] = coords[~valid]

        self.curr_vertices = moved.tolist()
        log(
            f"  -> Surface offset outward by {distance_mm:.3f}mm "
            f"({valid.sum()}/{len(coords)} nodes)."
        )

    def export_data(self):
        atoms = pd.DataFrame(self.curr_vertices, columns=['x', 'y', 'z'])
        atoms['atom_id'] = np.arange(len(atoms))
        atoms['face_name'] = 'surface'

        edges_set = set()
        if self.curr_edges is not None:
            edges_set.update(tuple(sorted((int(u), int(v)))) for u, v in self.curr_edges)
        else:
            for f in self.curr_faces:
                u, v, w = sorted(f)
                edges_set.add(tuple(sorted((u, v))))
                edges_set.add(tuple(sorted((v, w))))
                edges_set.add(tuple(sorted((w, u))))

        edges_list = [{'source': u, 'target': v, 'radius': SURFACE_DIAM_MM / 2.0, 'bond_type': 'surface'} for u, v in
                      edges_set]
        edges = pd.DataFrame(edges_list)
        return atoms, edges


def _build_internal_structure_voxel_legacy(mesh, start_id, mm_per_unit):
    log("生成内部结构...")
    z_min, z_max = mesh.bounds[0, 2], mesh.bounds[1, 2]
    total_h = z_max - z_min

    pitch_bot, pitch_top = INTERNAL_PITCH_BOTTOM_MM, INTERNAL_PITCH_TOP_MM
    diam_bot, diam_top = INTERNAL_DIAM_BOTTOM_MM, INTERNAL_DIAM_TOP_MM

    vg = mesh.voxelized(pitch=pitch_top * 0.8).fill()
    raw_pts = vg.points
    if len(raw_pts) > 0: raw_pts += np.random.uniform(-0.1, 0.1, raw_pts.shape)

    keep_indices = []
    if len(raw_pts) > 0:
        h_norm = np.clip((raw_pts[:, 2] - z_min) / (z_max - z_min), 0, 1)
        exclusion_radii = (pitch_bot * (1 - h_norm) + pitch_top * h_norm) * 0.7
        tree = cKDTree(raw_pts)
        active = np.ones(len(raw_pts), dtype=bool)
        rng = np.random.RandomState(42)
        order = rng.permutation(len(raw_pts))
        for idx in order:
            if not active[idx]: continue
            keep_indices.append(idx)
            active[tree.query_ball_point(raw_pts[idx], exclusion_radii[idx])] = False

    pts = raw_pts[keep_indices] if len(keep_indices) > 0 else np.array([])
    if len(pts) == 0: return pd.DataFrame(), pd.DataFrame()

    kdt = cKDTree(pts)
    edges = set()
    h_norm_all = np.clip((pts[:, 2] - z_min) / total_h, 0, 1)
    local_pitch_all = pitch_bot * (1 - h_norm_all) + pitch_top * h_norm_all

    for i in range(len(pts)):
        z_curr = pts[i][2]
        r_search = local_pitch_all[i] * SEARCH_RADIUS_FACTOR
        neighbors_idx = kdt.query_ball_point(pts[i], r_search)
        cands_below, cands_same = [], []
        for nb in neighbors_idx:
            if nb == i: continue
            dz = z_curr - pts[nb][2]
            if abs(dz) < Z_SAME_TOLERANCE_MM:
                cands_same.append((np.linalg.norm(pts[i] - pts[nb]), nb))
            elif dz > Z_SAME_TOLERANCE_MM:
                cands_below.append((np.linalg.norm(pts[i] - pts[nb]), nb))

        cands_below.sort()
        for _, nb in cands_below[:2]: edges.add(tuple(sorted((i, nb))))
        cands_same.sort()
        for _, nb in cands_same[:1]: edges.add(tuple(sorted((i, nb))))

    ids = np.arange(start_id, start_id + len(pts))
    atoms = pd.DataFrame({"atom_id": ids, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "face_name": "internal"})
    pts_dict = {i: pts[i] for i in range(len(pts))}
    edge_list = []
    for u, v in edges:
        z_mid = (pts_dict[u][2] + pts_dict[v][2]) / 2.0
        h_edge = np.clip((z_mid - z_min) / total_h, 0, 1)
        r_val = (diam_bot * (1 - h_edge) + diam_top * h_edge) / 2.0
        edge_list.append({"source": ids[u], "target": ids[v], "radius": r_val, "bond_type": "internal"})

    return atoms, pd.DataFrame(edge_list)


def build_internal_structure_v2(mesh, start_id, mm_per_unit,
                                atoms_surf: Optional[pd.DataFrame] = None,
                                edges_surf: Optional[pd.DataFrame] = None,
                                output_dir: Optional[Path] = None):
    if str(INTERNAL_BUILD_MODE).lower() == "voxel":
        return _build_internal_structure_voxel_legacy(mesh, start_id, mm_per_unit)

    try:
        return build_internal_structure_dynamic(
            mesh=mesh,
            start_id=start_id,
            atoms_surf=atoms_surf,
            edges_surf=edges_surf,
            output_dir=output_dir or OUT_DIR,
        )
    except WorkflowCheckpointStop:
        raise
    except Exception as exc:
        log(f"Dynamic internal build failed ({exc}); no automatic method-changing fallback was used.")
        raise RuntimeError("dynamic internal build failed") from exc


import random


def _select_surface_anchors(coords, spacing):
    """Deterministic greedy sampling with full spacing coverage."""
    if len(coords) == 0:
        return np.array([], dtype=int)
    if spacing <= 0:
        return np.arange(len(coords), dtype=int)

    cell_size = float(spacing)
    buckets = {}
    selected = []

    for idx, point in enumerate(coords):
        cell = tuple(np.floor(point / cell_size).astype(int))
        keep = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nearby = buckets.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz), ()
                    )
                    if any(
                            np.linalg.norm(point - coords[j]) < spacing
                            for j in nearby
                    ):
                        keep = False
                        break
                if not keep:
                    break
            if not keep:
                break
        if keep:
            selected.append(idx)
            buckets.setdefault(cell, []).append(idx)

    return np.asarray(selected, dtype=int)


def _surface_graph_components(atoms_surf, edges_surf):
    """Return actual retained surface connected components and membership."""
    ids = set(atoms_surf['atom_id'].astype(int))
    adj = {i: [] for i in ids}
    for e in _edge_records(edges_surf):
        if str(e.get('bond_type', '')) != 'surface':
            continue
        u, v = int(e['source']), int(e['target'])
        if u in adj and v in adj and u != v:
            adj[u].append(v)
            adj[v].append(u)

    active = {i for i in ids if adj[i]}
    labels, components = {}, []
    for start in sorted(active):
        if start in labels:
            continue
        index = len(components)
        group, pending = set(), [start]
        labels[start] = index
        while pending:
            u = pending.pop()
            group.add(u)
            for v in adj[u]:
                if v not in labels:
                    labels[v] = index
                    pending.append(v)
        components.append(group)
    return labels, components


def _stitch_print_class(surface_xyz, internal_xyz, bridge_max_mm):
    """PRINT angle relative to build Z; never use the surface inward normal."""
    delta = np.asarray(internal_xyz, float) - np.asarray(surface_xyz, float)
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance) or distance <= 1.0e-8:
        return 'rejected', 0.0, distance
    dz = abs(float(delta[2]))
    dxy = float(np.linalg.norm(delta[:2]))
    elevation = math.degrees(math.atan2(dz, max(dxy, 1.0e-12)))
    if elevation + 1e-8 >= float(INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG):
        return 'inclined', elevation, distance
    if (INTERNAL_HORIZONTAL_BRIDGE_ENABLE
            and distance < float(bridge_max_mm) - 1e-9):
        return 'short_bridge', elevation, distance
    return 'rejected', elevation, distance


def _stitch_centerline_screen(mesh, cache, surface_xyz, internal_xyz,
                              *, surface_tolerance_mm=0.05):
    """Three sample points with bounded mesh-carrier tolerance at skin only.

    A carrier-chart surface anchor may differ slightly from the original STL.
    The inside samples must lie strictly inside the ORIGINAL mesh slice.
    """
    a, b = np.asarray(surface_xyz, float), np.asarray(internal_xyz, float)
    for t in (0.0, 0.5, 1.0):
        xyz = (1.0 - t) * a + t * b
        domain = _domain_at_z_cached(mesh, float(xyz[2]), cache)
        point = Point(float(xyz[0]), float(xyz[1]))
        if domain.is_empty:
            return False
        if t == 0.0:
            if not domain.buffer(max(0.0, float(surface_tolerance_mm))).covers(point):
                return False
        elif not domain.covers(point):
            return False
    return True


def _stitch_radius_xy_screen(mesh, cache, surface_xyz, internal_xyz, radius_mm):
    """Conservative 2D section-envelope screen at stitch midpoint/internal end.

    The surface end is intentionally attached to the skin, so it is exempt
    from the interior clearance requirement. This tests lateral clearance
    of the rod radius on sampled horizontal sections; it is NOT a certified
    exact swept-volume containment test on arbitrary freeform meshes.
    """
    a, b = np.asarray(surface_xyz, float), np.asarray(internal_xyz, float)
    r = max(0.0, float(radius_mm))
    for t in (0.5, 1.0):
        xyz = a * (1.0 - t) + b * t
        z = float(xyz[2])
        domain = _domain_at_z_cached(mesh, z, cache)
        point = Point(float(xyz[0]), float(xyz[1]))

        if not any(poly.buffer(-max(0.0, r - 1.0e-5)).covers(point)
                   for poly in _as_polygons(domain)):
            return False
    return True


def stitch_surface_to_core_optimized(atoms_surf, atoms_int, r_stitch,
                                     anchor_spacing=STITCH_ANCHOR_SPACING_MM,
                                     existing_edges: Optional[pd.DataFrame] = None,
                                     max_internal_degree: int = INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW,
                                     *, surface_edges: Optional[pd.DataFrame] = None,
                                     mesh=None, z_bounds=None,
                                     max_surface_degree: int = 8,
                                     relaxed_distance_mm=None,
                                     candidate_k=None,
                                     alternative_anchors=None,
                                     surface_anchor_area_guard=None,
                                     audit_dir=None):
    """Constrained dual-domain stitching with bounded fallback and explicit failure.

    1. Both endpoints checked against their actual FINAL logical degrees.
    2. All stitch directions use build-Z overhang angle, or a short supported
       horizontal bridge; no surface-normal angle substitution.
    3. Broaden k-nearest search, then allow bounded distance expansion, then
       try nearby ALTERNATE anchors on the SAME retained surface component.
    4. Only bottom-supported internal vertices are legal attachment targets.
       Domain centreline is checked on accepted candidates if mesh is supplied.
    5. Audit and mark components which cannot be attached; never force an
       invalid edge just to report a successful dual-domain merger.
    """
    columns = ['source', 'target', 'radius', 'bond_type', 'role', 'stitch_search_tier',
               'stitch_angle_deg_above_horizontal', 'stitch_print_class',
               'stitch_length_mm', 'stitch_domain_checked', 'stitch_radius_xy_screen_pass']
    started = time.perf_counter()
    if atoms_surf.empty or atoms_int.empty:
        return pd.DataFrame(columns=columns)
    surface = atoms_surf.sort_values('atom_id').reset_index(drop=True)
    internal = atoms_int.sort_values('atom_id').reset_index(drop=True)
    surf_ids = surface['atom_id'].to_numpy(dtype=int)
    surf_xyz = surface[['x', 'y', 'z']].to_numpy(float)
    int_ids = internal['atom_id'].to_numpy(dtype=int)
    int_xyz = internal[['x', 'y', 'z']].to_numpy(float)
    surf_at = {int(i): int(j) for j, i in enumerate(surf_ids)}
    int_at = {int(i): int(j) for j, i in enumerate(int_ids)}
    strict_length = float(r_stitch)
    relaxed_length = max(strict_length, float(STITCH_RELAXED_DISTANCE_MM
                                              if relaxed_distance_mm is None else relaxed_distance_mm))
    nearest_k = max(8, int(STITCH_CANDIDATE_K if candidate_k is None else candidate_k))
    alt_k = max(0, int(STITCH_ALTERNATIVE_ANCHORS if alternative_anchors is None
                       else alternative_anchors))
    if strict_length <= 0 or relaxed_length <= 0:
        raise ValueError('stitch search radius must be positive')
    internal_existing = existing_edges if existing_edges is not None else pd.DataFrame()
    support = _internal_support_state(internal, internal_existing, z_bounds=z_bounds)
    legal_int = support['supported'] & support['root_connected']
    if not legal_int:
        log('  Dual stitch: no bottom-supported internal endpoints; do not force stitches.')
    surf_degree = {int(i): 0 for i in surf_ids}
    int_degree = {int(i): 0 for i in int_ids}
    all_existing_pairs = set()
    for e in _edge_records(pd.concat([x for x in (surface_edges, existing_edges)
                                      if x is not None and not x.empty], ignore_index=True)
                           if any(x is not None and not x.empty
                                  for x in (surface_edges, existing_edges))
                           else []):
        u, v = int(e['source']), int(e['target'])
        all_existing_pairs.add((min(u, v), max(u, v)))
        if u in surf_degree: surf_degree[u] += 1
        if v in surf_degree: surf_degree[v] += 1
        if u in int_degree: int_degree[u] += 1
        if v in int_degree: int_degree[v] += 1
    labels, components = _surface_graph_components(surface,
                                                   surface_edges if surface_edges is not None else pd.DataFrame())

    anchor_idx = _select_surface_anchors(surf_xyz, float(anchor_spacing))
    if not len(components):
        components = [set(map(int, surf_ids))]
        labels = {int(i): 0 for i in surf_ids}
    tree_all = cKDTree(int_xyz)
    surf_tree = cKDTree(surf_xyz)
    domain_cache = {}
    edges, attempts = [], []
    paired = set()
    failed_anchors = set()
    component_stitches = {i: 0 for i in range(len(components))}
    reject_count = {'no_supported_target': 0, 'angle_or_span': 0, 'degree': 0,
                    'domain': 0, 'radial_envelope': 0, 'duplicate': 0,
                    'local_pore_geometry': 0}
    bridge_limit = float(INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM)

    def try_anchor(sid, *, replacement=False):
        sid = int(sid)
        if sid not in labels or sid in paired or sid in failed_anchors:
            return False
        s_idx = surf_at[sid]
        if max_surface_degree > 0 and surf_degree[sid] >= max_surface_degree:
            reject_count['degree'] += 1
            return False
        if (surface_anchor_area_guard is not None and
                not surface_anchor_area_guard(
                    sid, INTERNAL_DIAM_TOP_MM / 2.0, commit=False)):
            reject_count['local_pore_geometry'] += 1
            failed_anchors.add(sid)
            attempts.append({'original_anchor_id': sid, 'actual_anchor_id': None,
                             'component_id': int(labels[sid]),
                             'result': 'local_pore_geometry_rejected',
                             'tier': 'surface_open_area_and_radius_guard'})
            return False

        k0 = min(8, len(int_ids))
        tried = set()
        batches = ((k0, strict_length, 'strict'),
                   (min(len(int_ids), nearest_k), strict_length, 'expanded_neighbors'),
                   (min(len(int_ids), nearest_k), relaxed_length, 'relaxed_distance'))
        for k, radius, tier in batches:
            if radius <= 0 or not k: continue
            dists, idxs = tree_all.query(surf_xyz[s_idx], k=k, distance_upper_bound=radius)
            options = sorted(zip(np.atleast_1d(dists), np.atleast_1d(idxs)),
                             key=lambda item: (float(item[0]), int(item[1])))
            for d, j in options:
                j = int(j)
                if not np.isfinite(d) or j >= len(int_ids): continue
                pair = (sid, int(int_ids[j]))
                if pair in tried: continue
                tried.add(pair)
                if pair[1] not in legal_int:
                    reject_count['no_supported_target'] += 1;
                    continue
                if (min(pair), max(pair)) in all_existing_pairs:
                    reject_count['duplicate'] += 1;
                    continue
                if max_internal_degree > 0 and int_degree[pair[1]] >= max_internal_degree:
                    reject_count['degree'] += 1;
                    continue
                classification, elevation, length = _stitch_print_class(
                    surf_xyz[s_idx], int_xyz[j], bridge_limit)
                if classification == 'rejected':
                    reject_count['angle_or_span'] += 1;
                    continue

                if mesh is not None:
                    if not _stitch_centerline_screen(
                            mesh, domain_cache, surf_xyz[s_idx], int_xyz[j]):
                        reject_count['domain'] += 1;
                        continue
                    if not _stitch_radius_xy_screen(
                            mesh, domain_cache, surf_xyz[s_idx], int_xyz[j],
                            INTERNAL_DIAM_TOP_MM / 2.0):
                        reject_count['radial_envelope'] += 1;
                        continue
                if (surface_anchor_area_guard is not None and
                        not surface_anchor_area_guard(
                            sid, INTERNAL_DIAM_TOP_MM / 2.0, commit=True)):
                    reject_count['local_pore_geometry'] += 1
                    failed_anchors.add(sid)
                    return False
                surf_degree[sid] += 1;
                int_degree[pair[1]] += 1
                all_existing_pairs.add((min(pair), max(pair)))
                paired.add(sid)
                component_stitches[labels[sid]] += 1
                edges.append({'source': sid, 'target': pair[1],
                              'radius': INTERNAL_DIAM_TOP_MM / 2.0, 'bond_type': 'stitch',
                              'role': 'dual_domain_stitch',
                              'stitch_search_tier': ('alternative_anchor_' if replacement else '') + tier,
                              'stitch_angle_deg_above_horizontal': float(elevation),
                              'stitch_print_class': classification,
                              'stitch_length_mm': float(length),
                              'stitch_domain_checked': bool(mesh is not None),
                              'stitch_radius_xy_screen_pass': bool(mesh is not None)})
                attempts.append({'original_anchor_id': sid, 'actual_anchor_id': sid,
                                 'component_id': int(labels[sid]), 'result': 'accepted',
                                 'tier': tier})
                return True
        failed_anchors.add(sid)
        attempts.append({'original_anchor_id': sid, 'actual_anchor_id': None,
                         'component_id': int(labels[sid]), 'result': 'unmatched',
                         'tier': 'all_bounded_search_stages_failed'})
        return False

    unmatched = []
    for i in anchor_idx:
        sid = int(surf_ids[int(i)])
        if sid not in labels: continue
        if not try_anchor(sid): unmatched.append(sid)

    for sid in unmatched:
        center = surf_xyz[surf_at[sid]]
        near = surf_tree.query_ball_point(center, r=max(float(anchor_spacing), 0.1))
        near = sorted((int(j) for j in near if int(surf_ids[j]) != sid and
                       labels.get(int(surf_ids[j])) == labels[sid]),
                      key=lambda j: (float(np.linalg.norm(surf_xyz[j] - center)), int(surf_ids[j])))
        for j in near[:alt_k]:
            if try_anchor(int(surf_ids[j]), replacement=True): break

    for cid, group in enumerate(components):
        if component_stitches[cid]: continue
        candidates = sorted((sid for sid in group if sid in surf_at),
                            key=lambda sid: (float(tree_all.query(surf_xyz[surf_at[sid]], k=1)[0]), sid))
        for sid in candidates[:max(1, alt_k)]:
            if try_anchor(sid, replacement=True): break
    unattached = [int(i) for i, n in component_stitches.items() if not n]
    stats = {'selected_anchor_count': int(len(anchor_idx)), 'stitch_count': len(edges),
             'surface_component_count': len(components), 'unattached_surface_components': unattached,
             'max_standard_length_mm': strict_length, 'max_fallback_length_mm': relaxed_length,
             'provisional_bridge_max_length_mm': bridge_limit,
             'fallback_long_edge_count': sum(float(e['stitch_length_mm']) > strict_length + 1e-9 for e in edges),
             'short_bridge_count': sum(e['stitch_print_class'] == 'short_bridge' for e in edges),
             'near_horizontal_short_bridge_max_length_mm_provisional': bridge_limit,
             'reject_counts': reject_count, 'geometry_checked': bool(mesh is not None),
             'geometry_scope': 'three_centerline_samples_plus_2D_radius_clearance_at_midpoint_and_internal_end',
             'surface_carrier_tolerance_mm': 0.05,
             'section_cache_count': len(domain_cache),
             'elapsed_seconds': float(time.perf_counter() - started)}
    log('  Dual stitch: ' + json.dumps(stats, ensure_ascii=False))
    result = pd.DataFrame(edges, columns=columns)
    result.attrs['stitch_build_audit'] = stats
    if audit_dir is not None:
        folder = Path(audit_dir);
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'dual_stitch_build.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                       encoding='utf-8')
        pd.DataFrame(attempts, columns=['original_anchor_id', 'actual_anchor_id',
                                        'component_id', 'result', 'tier']).to_csv(folder / 'dual_stitch_attempts.csv',
                                                                                  index=False)
    return result


def analyze_surface_physics_detailed(atoms_df, edges_df, output_dir: Path):
    log("正在执行表面物理与几何分析 (Surface Physics)...")

    surf_atoms = atoms_df[atoms_df['face_name'] == 'surface'].copy()
    if surf_atoms.empty:
        log("  -> Error: No surface atoms found.")
        return

    atom_id_map = {row.atom_id: i for i, row in surf_atoms.reset_index().iterrows()}
    coords = surf_atoms[['x', 'y', 'z']].values
    surf_edges = edges_df[edges_df['bond_type'] == 'surface'].copy()

    adj = {i: set() for i in range(len(coords))}
    edge_radii = {}

    for r in surf_edges.itertuples():
        if r.source in atom_id_map and r.target in atom_id_map:
            u, v = atom_id_map[r.source], atom_id_map[r.target]
            adj[u].add(v);
            adj[v].add(u)
            rad = getattr(r, 'radius', SURFACE_DIAM_MM / 2.0)
            edge_radii[tuple(sorted((u, v)))] = rad

    triangles = set()
    for u in adj:
        for v in adj[u]:
            if u < v:
                common = adj[u].intersection(adj[v])
                for w in common:
                    if v < w: triangles.add((u, v, w))

    stats = []
    cos_theta = math.cos(math.radians(CONTACT_ANGLE_DEG))

    for tri in tqdm(triangles, desc="Analyzing Surface Physics"):
        p0, p1, p2 = coords[tri[0]], coords[tri[1]], coords[tri[2]]
        z_center = (p0[2] + p1[2] + p2[2]) / 3.0

        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p0 - p2)
        s = (a + b + c) / 2.0
        area_sq = max(0, s * (s - a) * (s - b) * (s - c))
        raw_area = np.sqrt(area_sq)

        if raw_area < 1e-9: continue

        p2_x = (a ** 2 + c ** 2 - b ** 2) / (2 * a)
        term = max(0, c ** 2 - p2_x ** 2)
        p2_y = np.sqrt(term)
        tri_2d = [(0.0, 0.0), (a, 0.0), (p2_x, p2_y)]
        poly_base = Polygon(tri_2d)

        r01 = edge_radii.get(tuple(sorted((tri[0], tri[1]))), SURFACE_DIAM_MM / 2.0)
        r12 = edge_radii.get(tuple(sorted((tri[1], tri[2]))), SURFACE_DIAM_MM / 2.0)
        r20 = edge_radii.get(tuple(sorted((tri[2], tri[0]))), SURFACE_DIAM_MM / 2.0)
        avg_r = (r01 + r12 + r20) / 3.0

        poly_liq = poly_base.buffer(-avg_r, resolution=8)

        r_eff = 0.0;
        p_cap = 99999.0;
        sf = 0.0;
        ar = 1.0

        if not poly_liq.is_empty:
            area_ll = poly_liq.area
            perimeter = poly_liq.length
            if perimeter > 1e-9:
                r_eff = (2.0 * area_ll) / perimeter
                p_cap = (2 * GAMMA_NPM * cos_theta) / (r_eff * 1e-3)
                sf = (4 * np.pi * area_ll) / (perimeter ** 2)
            try:
                rect = poly_liq.minimum_rotated_rectangle
                coords_rect = rect.exterior.coords
                d1 = math.hypot(coords_rect[1][0] - coords_rect[0][0], coords_rect[1][1] - coords_rect[0][1])
                d2 = math.hypot(coords_rect[2][0] - coords_rect[1][0], coords_rect[2][1] - coords_rect[1][1])
                if min(d1, d2) > 1e-9: ar = max(d1, d2) / min(d1, d2)
            except:
                pass

        p_hydro = (RHO_LIQ - RHO_GAS) * G_GRAV * (z_center * 1e-3)
        p_required = SURFACE_CAPILLARY_SAFETY_FACTOR * (
                p_hydro + SURFACE_EXTRA_PRESSURE_PA
        )
        safety_ratio = p_cap / max(p_required, 1e-6)

        stats.append({
            "z_mm": z_center, "r_eff_mm": r_eff, "P_cap_Pa": p_cap,
            "P_hydro_Pa": p_hydro,
            "P_extra_Pa": SURFACE_EXTRA_PRESSURE_PA,
            "surface_safety_factor": SURFACE_CAPILLARY_SAFETY_FACTOR,
            "P_required_Pa": p_required,
            "Safety": safety_ratio, "ShapeFactor": sf, "AspectRatio": ar
        })

    df = pd.DataFrame(stats)
    csv_path = output_dir / "surface_physics_detail.csv"
    df.to_csv(csv_path, index=False)

    if len(df) > 0:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        sc = ax1.scatter(df["r_eff_mm"], df["z_mm"], c=df["Safety"], cmap='RdYlGn', s=10, alpha=0.6, vmin=0.8, vmax=2.0)
        ax1.set_title("Surface Pore Size Gradient");
        ax1.invert_xaxis()
        plt.colorbar(sc, ax=ax1, label="Safety")

        ax2 = fig.add_subplot(gs[0, 1])
        df_sorted = df.sort_values("z_mm")
        ax2.plot(df_sorted["z_mm"], df_sorted["P_hydro_Pa"], color='grey', linestyle=':', label='Hydrostatic')
        ax2.plot(df_sorted["z_mm"], df_sorted["P_required_Pa"], 'r--', label='Required')
        ax2.scatter(df["z_mm"], df["P_cap_Pa"], s=5, color='green', alpha=0.2, label='Capillary')
        ax2.legend()

        plt.tight_layout()
        _save_svg_only(plt.gcf(), output_dir / "surface_physics_report.svg")
        plt.close()


@dataclass
class EdgeSeg:
    p0: np.ndarray
    p1: np.ndarray
    r: float


def _get_domain_poly_at_z(mesh, z):
    """精确获取大象在 z 高度的截面 (支持多区域分离)"""
    try:
        slice_3d = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if slice_3d is None: return Polygon()
        rings = []
        for path_3d in slice_3d.discrete:
            path_2d = path_3d[:, 0:2]
            if len(path_2d) > 2:
                if not np.allclose(path_2d[0], path_2d[-1]):
                    path_2d = np.vstack([path_2d, path_2d[0]])
                poly = Polygon(path_2d)
                if poly.is_valid and poly.area > 1e-9:
                    rings.append(poly)
                elif not poly.is_valid:
                    fixed = poly.buffer(0)
                    if isinstance(fixed, Polygon) and fixed.area > 1e-9:
                        rings.append(fixed)
                    elif isinstance(fixed, (MultiPolygon, GeometryCollection)):
                        rings.extend([g for g in fixed.geoms if isinstance(g, Polygon) and g.area > 1e-9])
        if not rings: return Polygon()

        rings.sort(key=lambda p: p.area, reverse=True)
        reps = [p.representative_point() for p in rings]
        solids, holes = [], []
        for i, poly in enumerate(rings):
            depth = 0
            for j, other in enumerate(rings):
                if i != j and other.area > poly.area and other.covers(reps[i]):
                    depth += 1
            if depth % 2 == 0:
                solids.append(poly)
            else:
                holes.append(poly)
        domain = unary_union(solids) if solids else Polygon()
        if holes and not domain.is_empty:
            domain = domain.difference(unary_union(holes))
        return domain.buffer(0) if not domain.is_empty else Polygon()
    except:
        return Polygon()


def _generate_true_solid_geometry_at_z_full(nodes_data: np.ndarray,
                                            edges_df: pd.DataFrame,
                                            atoms_df: pd.DataFrame,
                                            z_plane: float) -> GeometryCollection:
    shapes = []
    pos_map = {row.atom_id: (row.x, row.y, row.z) for row in atoms_df.itertuples(index=False)}

    for r in edges_df.itertuples(index=False):
        if r.source not in pos_map or r.target not in pos_map: continue
        p0 = np.array(pos_map[r.source])
        p1 = np.array(pos_map[r.target])
        rad = r.radius
        zmin, zmax = min(p0[2], p1[2]), max(p0[2], p1[2])
        if not (zmin - rad <= z_plane <= zmax + rad): continue

        dx, dy, dz = p1 - p0
        Lxy = math.hypot(dx, dy)
        L3d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if L3d < 1e-12: continue

        if abs(dz) <= max(1e-6, 1e-3 * L3d, 0.05 * rad):
            dmin = min(abs(p0[2] - z_plane), abs(p1[2] - z_plane))
            if (z_plane - p0[2]) * (z_plane - p1[2]) <= 0: dmin = 0.0
            if dmin < rad:
                w = math.sqrt(max(0.0, rad ** 2 - dmin ** 2))
                shapes.append(
                    LineString([p0[:2], p1[:2]]).buffer(w, resolution=16) if w > 1e-9 else Point(p0[:2]).buffer(rad,
                                                                                                                resolution=16))
            continue

        t = (z_plane - p0[2]) / dz
        if not (0.0 <= t <= 1.0): continue
        pt = p0 + t * (p1 - p0)
        uz = abs(dz) / L3d
        a = min(rad / uz, max(rad, 0.5 * (Lxy + 2 * rad))) if uz > 1e-12 else rad
        base = Point(pt[0], pt[1]).buffer(rad, resolution=16)
        angle = math.degrees(math.atan2(dy, dx))
        g = affinity.rotate(base, -angle, origin='centroid')
        g = affinity.scale(g, xfact=a / rad, yfact=1.0, origin='centroid')
        g = affinity.rotate(g, angle, origin='centroid')
        shapes.append(g)

    if nodes_data is not None and len(nodes_data) > 0:
        dz_arr = np.abs(nodes_data[:, 2] - z_plane)
        mask = dz_arr < nodes_data[:, 3]
        active_nodes = nodes_data[mask]
        for n in active_nodes:
            r_slice = math.sqrt(max(0.0, n[3] ** 2 - (n[2] - z_plane) ** 2))
            if r_slice > 1e-9:
                shapes.append(Point(n[0], n[1]).buffer(r_slice, resolution=16))

    if not shapes: return GeometryCollection()
    try:
        merged = unary_union(shapes)
        return merged.buffer(0) if not merged.is_valid else merged
    except:
        return unary_union([s.buffer(0) for s in shapes])


def _robust_internal_wetted_perimeter(solid_geometry, domain_geometry) -> float:
    """Return the liquid-solid interface length without silently converting errors to zero."""
    if solid_geometry is None or solid_geometry.is_empty:
        return 0.0
    try:
        inner_domain = domain_geometry.buffer(-1.0e-6)
    except Exception:
        inner_domain = domain_geometry
    candidates = [solid_geometry]
    try:
        repaired = solid_geometry.buffer(0)
        if not repaired.is_empty:
            candidates.append(repaired)
    except Exception:
        pass
    for candidate in candidates:
        try:
            length = float(candidate.boundary.intersection(inner_domain).length)
            if np.isfinite(length) and length > 1.0e-10:
                return length
        except Exception:
            continue

    for candidate in reversed(candidates):
        try:
            liquid = domain_geometry.difference(candidate)
            length = float(liquid.boundary.intersection(inner_domain).length)
            if np.isfinite(length) and length > 1.0e-10:
                return length
        except Exception:
            continue
    raise RuntimeError("non-empty solid section has no computable internal wetted perimeter")


def process_domain_slice(z_mid, domain_poly, nodes_np, edges_df, atoms_df):
    """
    [修复版] 并行分析单元
    接收预计算好的 domain_poly (Shapely Polygon)，不再接收 mesh 对象，避免 PicklingError。
    """

    if domain_poly.is_empty: return None

    solid_poly = _generate_true_solid_geometry_at_z_full(nodes_np, edges_df, atoms_df, z_mid)

    real_solid = solid_poly.intersection(domain_poly)

    A_solid = real_solid.area if not real_solid.is_empty else 0.0
    P_wet = _robust_internal_wetted_perimeter(real_solid, domain_poly)

    domain_area = domain_poly.area

    A_liquid = max(1e-9, domain_area - A_solid)
    try:
        liquid_geometry = domain_poly.difference(real_solid)
    except Exception:
        liquid_geometry = domain_poly
    radius4_sum_mm4, n_flow_components = section_radius4_sum_mm4(
        liquid_geometry, min_area_mm2=FLOW_SECTION_MIN_COMPONENT_AREA_MM2
    )

    P_hydro = RHO_LIQ * G_GRAV * (z_mid * 1e-3)
    cos_th = math.cos(math.radians(CONTACT_ANGLE_DEG))

    P_cap = 0.0
    r_eff = 0.0

    porosity = A_liquid / max(domain_area, 1e-12)
    if A_liquid > 1e-9 and P_wet > 1e-9:
        r_eff = (2.0 * A_liquid) / P_wet if P_wet > 1e-9 else 0.0

        if r_eff > 1e-9:
            P_cap = (GAMMA_NPM * cos_th * (P_wet * 1e-3)) / (A_liquid * 1e-6)
    z_bounds_local = (
    float(atoms_df["z"].min()), float(atoms_df["z"].max())) if atoms_df is not None and not atoms_df.empty else (
    0.0, 1.0)
    safety_factor_z = _safety_factor_at_z(z_mid, z_bounds_local)
    return {
        "z_mm": z_mid,
        "A_domain_mm2": domain_area,
        "A_solid_mm2": A_solid,
        "A_liq_mm2": A_liquid,
        "axial_radius4_sum_mm4": radius4_sum_mm4,
        "n_flow_components": n_flow_components,
        "porosity": porosity,
        "P_wet_mm": P_wet,
        "r_eff_mm": r_eff,
        "P_cap_Pa": P_cap,
        "P_hydro_Pa": P_hydro,
        "safety_factor_z": safety_factor_z,
        "fluid_viscosity_Pa_s": FLUID_VISCOSITY_PA_S,
    }


def run_full_analysis(mesh, atoms_df, edges_df, nodes_np, zs):
    """
    [修复版] 启动并行全切片分析
    先在主线程计算 Mesh 轮廓多边形，再传给子进程。
    """
    log("预计算大象轮廓 (Pre-calculating contours)...")

    domain_polys = []
    for z in tqdm(zs, desc="Slicing Contours"):
        poly = _get_domain_poly_at_z(mesh, z)
        domain_polys.append(poly)

    log("开始并行物理计算 (Parallel Physics Calculation)...")

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_domain_slice)(z, poly, nodes_np, edges_df, atoms_df)
        for z, poly in tqdm(zip(zs, domain_polys), total=len(zs), desc="Running Physics Slicing")
    )

    profile = pd.DataFrame([r for r in results if r is not None])
    z_values = np.asarray(zs, dtype=float)
    analysis_dz = float(SLICE_DZ_MM)
    if not profile.empty:
        z_bounds_local = (float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2]))
        centres, widths = _pressure_feedback_grid(z_bounds_local)
        width_by_center = {round(float(z), 9): float(w) for z, w in zip(centres, widths)}
        z_col = "z_mid_mm" if "z_mid_mm" in profile.columns else "z_mm"
        profile["slice_dz_mm"] = [
            width_by_center.get(round(float(z), 9), analysis_dz)
            for z in profile[z_col].to_numpy(dtype=float)
        ]
    return apply_state_updated_viscous_profile(
        profile,
        viscosity_pa_s=FLUID_VISCOSITY_PA_S,
        default_dz_mm=analysis_dz,
        reference_flow_m3_s=VISC_REF_Q_M3_S,
        target_fill_time_s=TARGET_FILL_TIME_S,
    )


def _as_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 1e-9 else []
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polys = []
        for g in geom.geoms:
            polys.extend(_as_polygons(g))
        return [p for p in polys if p.area > 1e-9]
    return []


def _edge_records(edges_any) -> List[Dict]:
    if edges_any is None:
        return []
    rows = edges_any.to_dict("records") if isinstance(edges_any, pd.DataFrame) else list(edges_any)
    clean = []
    for row in rows:
        rec = dict(row)
        rec.setdefault("bond_type", rec.get("role", "internal"))
        rec.setdefault("role", rec.get("bond_type", "internal"))
        rec.setdefault("radius", INTERNAL_DIAM_BOTTOM_MM / 2.0)
        clean.append(rec)
    return clean


def _dedup_edges_local(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df is None or edges_df.empty:
        return pd.DataFrame(columns=["source", "target", "radius", "bond_type", "role"])
    df = edges_df.copy()
    for col, default in [("radius", INTERNAL_DIAM_BOTTOM_MM / 2.0), ("bond_type", "internal"), ("role", "internal")]:
        if col not in df.columns:
            df[col] = default
    df = df[df["source"] != df["target"]].copy()
    if df.empty:
        return df
    df["__a"] = df[["source", "target"]].min(axis=1)
    df["__b"] = df[["source", "target"]].max(axis=1)
    keep_cols = [c for c in df.columns if c not in {"__a", "__b"}]
    df = df.drop_duplicates(subset=["__a", "__b", "bond_type", "role"]).copy()
    return df[keep_cols].reset_index(drop=True)


def _domain_at_z_cached(mesh, z: float, cache: Dict[float, object]):
    key = round(float(z), 4)
    if key in cache:
        return cache[key]
    geom = _get_domain_poly_at_z(mesh, float(z))
    if geom is None or geom.is_empty:
        zmin, zmax = mesh.bounds[0, 2], mesh.bounds[1, 2]
        dz = max((zmax - zmin) * 0.001, 1e-3)
        for shift in (-dz, dz, -2.0 * dz, 2.0 * dz):
            zz = min(max(float(z) + shift, zmin + 1e-6), zmax - 1e-6)
            geom = _get_domain_poly_at_z(mesh, zz)
            if geom is not None and not geom.is_empty:
                break
    if geom is None:
        geom = Polygon()
    try:
        geom = geom.buffer(0) if not geom.is_valid else geom
    except Exception:
        geom = Polygon()
    cache[key] = geom
    return geom


def _point_component_index(domain, x: float, y: float) -> Optional[int]:
    pt = Point(float(x), float(y))
    polys = sorted(_as_polygons(domain), key=lambda p: p.area, reverse=True)
    for idx, poly in enumerate(polys):
        if poly.covers(pt):
            return int(idx)
    return None


def _edge_centerline_stays_in_domain(mesh, cache: Dict[float, object],
                                     p0: Tuple[float, float, float],
                                     p1: Tuple[float, float, float],
                                     samples: int = 3) -> bool:
    x0, y0, z0 = map(float, p0)
    x1, y1, z1 = map(float, p1)
    dz = abs(z1 - z0)
    dxy = math.hypot(x1 - x0, y1 - y0)
    if dz <= max(1e-4, 0.10 * max(dxy, 1e-9)):
        z_mid = 0.5 * (z0 + z1)
        domain = _domain_at_z_cached(mesh, z_mid, cache)
        line = LineString([(x0, y0), (x1, y1)])
        for poly in _as_polygons(domain):
            if poly.covers(Point(x0, y0)) and poly.covers(Point(x1, y1)):
                return bool(poly.covers(line))
        return False
    n = max(3, int(samples))
    for t in np.linspace(0.0, 1.0, n):
        x = x0 * (1.0 - t) + x1 * t
        y = y0 * (1.0 - t) + y1 * t
        z = z0 * (1.0 - t) + z1 * t
        domain = _domain_at_z_cached(mesh, z, cache)
        if domain.is_empty or not domain.covers(Point(float(x), float(y))):
            return False
    return True


def _local_internal_radius(z: float, z_bounds: Tuple[float, float]) -> float:
    zmin, zmax = z_bounds
    t = float(np.clip((float(z) - zmin) / max(zmax - zmin, 1e-9), 0.0, 1.0))
    diam = INTERNAL_DIAM_BOTTOM_MM * (1.0 - t) + INTERNAL_DIAM_TOP_MM * t
    return float(diam / 2.0)


def _local_internal_pitch(z: float, z_bounds: Tuple[float, float]) -> float:
    zmin, zmax = z_bounds
    t = float(np.clip((float(z) - zmin) / max(zmax - zmin, 1e-9), 0.0, 1.0))
    return float(INTERNAL_PITCH_BOTTOM_MM * (1.0 - t) + INTERNAL_PITCH_TOP_MM * t)


def _make_nodes_np_for_analysis(atoms_df: pd.DataFrame, z_bounds: Tuple[float, float]) -> np.ndarray:
    if atoms_df is None or atoms_df.empty:
        return np.zeros((0, 4), dtype=float)
    zmin, zmax = z_bounds
    zh = max(zmax - zmin, 1e-9)
    rows = []
    for row in atoms_df.itertuples(index=False):
        face_name = str(getattr(row, "face_name", "internal"))
        if "surface" in face_name:

            continue
        else:
            h = float(np.clip((float(row.z) - zmin) / zh, 0.0, 1.0))
            diam = INTERNAL_DIAM_BOTTOM_MM * (1.0 - h) + INTERNAL_DIAM_TOP_MM * h
            radius = (diam / 2.0) * NODE_DIAM_RATIO
        rows.append([float(row.x), float(row.y), float(row.z), float(radius)])
    return np.asarray(rows, dtype=float)


def _sample_xy_in_domain(domain, rng: np.random.Generator, max_tries: int = 250) -> Optional[Tuple[float, float]]:
    polys = _as_polygons(domain)
    if not polys:
        return None
    areas = np.asarray([max(p.area, 0.0) for p in polys], dtype=float)
    if float(areas.sum()) > 1e-12:
        poly = polys[int(rng.choice(len(polys), p=areas / float(areas.sum())))]
    else:
        poly = max(polys, key=lambda p: p.area)
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_tries):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if poly.covers(Point(x, y)):
            return x, y
    p = poly.representative_point()
    return float(p.x), float(p.y)


def _best_candidate_in_stl_domain(mesh, cache: Dict[float, object], z0: float, z1: float,
                                  rng: np.random.Generator, atoms_df: pd.DataFrame,
                                  n_candidates: int = INTERNAL_DYNAMIC_CANDIDATES,
                                  z_hint: Optional[float] = None,
                                  domain_hint=None):
    coords = atoms_df[["x", "y", "z"]].to_numpy(dtype=float) if atoms_df is not None and not atoms_df.empty else None
    tree = cKDTree(coords) if coords is not None and len(coords) > 0 else None
    best, best_score = None, -1.0
    use_hint = domain_hint is not None and not domain_hint.is_empty and z_hint is not None
    for _ in range(max(1, int(n_candidates))):
        if use_hint:
            layer_h = max(float(z1 - z0), 1.0e-9)
            z_band = float(INTERNAL_DYNAMIC_LAYER_Z_JITTER_FRAC) * layer_h
            z = float(np.clip(float(z_hint) + rng.uniform(-z_band, z_band), z0 + 1e-6, z1 - 1e-6))
            xy = _sample_xy_in_domain(domain_hint, rng)
        else:
            z = float(rng.uniform(z0, z1))
            xy = _sample_xy_in_domain(_domain_at_z_cached(mesh, z, cache), rng)
        if xy is None:
            continue
        x, y = xy
        if tree is None:
            return x, y, z
        dist, _ = tree.query([x, y, z], k=1)
        if float(dist) > best_score:
            best_score = float(dist)
            best = (x, y, z)
    return best


def _solid_geometry_dynamic(atoms_df: pd.DataFrame, edges_records: List[Dict],
                            z_bounds: Tuple[float, float], z_plane: float):
    edges_df = pd.DataFrame(edges_records)
    if edges_df.empty:
        edges_df = pd.DataFrame(columns=["source", "target", "radius", "bond_type", "role"])
    nodes_np = _make_nodes_np_for_analysis(atoms_df, z_bounds)
    return _generate_true_solid_geometry_at_z_full(nodes_np, edges_df, atoms_df, z_plane)


def _internal_candidate_geometry(atoms_df: pd.DataFrame, edges_records: List[Dict],
                                 z_bounds: Tuple[float, float], z_plane: float):
    if atoms_df is None or atoms_df.empty:
        return GeometryCollection()
    names = atoms_df["face_name"].astype(str) if "face_name" in atoms_df.columns else pd.Series("",
                                                                                                index=atoms_df.index)
    internal_atoms = atoms_df[~names.str.contains("Surface", case=False, na=False)].copy()
    if internal_atoms.empty:
        return GeometryCollection()
    internal_ids_local = set(internal_atoms["atom_id"].astype(int).tolist())
    internal_edges = [
        e for e in edges_records
        if str(e.get("bond_type", "")) in {"internal", "stitch"}
           and int(e.get("source", -1)) in internal_ids_local
           and int(e.get("target", -1)) in internal_ids_local
    ]
    return _solid_geometry_dynamic(internal_atoms, internal_edges, z_bounds, z_plane)


def _find_worst_pore_center_stl(mesh, cache: Dict[float, object], atoms_df: pd.DataFrame,
                                edges_records: List[Dict], z_bounds: Tuple[float, float], z_plane: float,
                                domain_override=None):
    domain = domain_override if domain_override is not None else _domain_at_z_cached(mesh, z_plane, cache)
    if domain.is_empty:
        return None, 0.0
    solid = _internal_candidate_geometry(atoms_df, edges_records, z_bounds, z_plane)
    try:
        liquid = domain.difference(solid.intersection(domain)) if solid is not None and not solid.is_empty else domain
    except Exception:
        liquid = domain
    polys = _as_polygons(liquid)
    if not polys:
        return None, 0.0
    pore = max(polys, key=lambda p: p.area)
    try:
        center = polylabel(pore, tolerance=0.05)
    except Exception:
        center = pore.representative_point()
    return (float(center.x), float(center.y)), float(pore.area)


def _component_worst_pore_candidates(mesh, cache: Dict[float, object], atoms_df: pd.DataFrame,
                                     edges_records: List[Dict], z_bounds: Tuple[float, float], z_plane: float):
    domain = _domain_at_z_cached(mesh, z_plane, cache)
    out = []
    for comp_idx, poly in enumerate(sorted(_as_polygons(domain), key=lambda p: p.area, reverse=True)):
        if float(poly.area) < INTERNAL_DYNAMIC_MIN_COMPONENT_AREA_MM2:
            continue
        xy, score = _find_worst_pore_center_stl(
            mesh, cache, atoms_df, edges_records, z_bounds, z_plane, domain_override=poly
        )
        if xy is not None:
            out.append(
                {"component": int(comp_idx), "domain": poly, "xy": xy, "score": float(score), "area": float(poly.area)})
    return out


def _top_pore_candidates(mesh, cache, atoms_df, edges_records, z_bounds, z_plane, limit=8):
    """Distinct liquid pockets at the pressure section, largest first."""
    domain = _domain_at_z_cached(mesh, z_plane, cache)
    solid = _internal_candidate_geometry(atoms_df, edges_records, z_bounds, z_plane)
    liquid = domain.difference(solid.intersection(domain)) if solid is not None and not solid.is_empty else domain
    pockets = sorted(_as_polygons(liquid), key=lambda p: p.area, reverse=True)
    candidates = []
    for pocket in pockets[:limit]:
        center = polylabel(pocket, tolerance=0.05)
        candidates.append({"xy": (float(center.x), float(center.y)),
                           "score": float(pocket.area), "domain": domain, "component": 0})
        if len(candidates) >= limit:
            break
    if len(candidates) < limit and pockets:
        pocket = pockets[0]
        xmin, ymin, xmax, ymax = pocket.bounds
        for xfrac, yfrac in ((.25, .25), (.75, .25), (.25, .75), (.75, .75),
                             (.5, .25), (.5, .75), (.25, .5), (.75, .5)):
            x, y = xmin + xfrac * (xmax - xmin), ymin + yfrac * (ymax - ymin)
            if pocket.covers(Point(x, y)) and all(
                    math.hypot(x - c['xy'][0], y - c['xy'][1]) > 0.2 for c in candidates):
                candidates.append({"xy": (x, y), "score": float(pocket.area),
                                   "domain": domain, "component": 0})
            if len(candidates) >= limit:
                break
    return candidates


_PRESSURE_RAW_SECTION_COLS = (
    "z_mm", "z_mid_mm", "A_domain_mm2", "A_solid_mm2", "A_liq_mm2",
    "axial_radius4_sum_mm4", "n_flow_components", "porosity", "P_wet_mm",
    "r_eff_mm", "P_cap_Pa", "P_hydro_Pa", "safety_factor_z",
    "fluid_viscosity_Pa_s", "slice_dz_mm",
)


def _pressure_feedback_grid(z_bounds: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return fixed-thickness pressure bins as (centres, widths) in millimetres.

    Full bins use ``SLICE_DZ_MM`` exactly.  If the total design height is not an
    integer multiple of the nominal spacing, only the final bin is shortened so
    the grid still terminates at the actual top surface.
    """
    zmin, zmax = map(float, z_bounds)
    height = max(0.0, zmax - zmin)
    dz = max(1.0e-6, float(SLICE_DZ_MM))
    if height <= 1.0e-12:
        return np.asarray([zmin], dtype=float), np.asarray([dz], dtype=float)

    n_full = int(math.floor(height / dz + 1.0e-12))
    edges = zmin + np.arange(n_full + 1, dtype=float) * dz
    if len(edges) == 0:
        edges = np.asarray([zmin], dtype=float)
    if edges[-1] < zmax - 1.0e-9:
        edges = np.r_[edges, zmax]
    else:
        edges[-1] = zmax
    if len(edges) < 2:
        edges = np.asarray([zmin, zmax], dtype=float)
    widths = np.diff(edges)
    centres = edges[:-1] + 0.5 * widths
    return centres.astype(float), widths.astype(float)


def _pressure_feedback_spacing(z_bounds: Tuple[float, float]) -> float:
    """Nominal pressure-slice thickness; retained for legacy callers/reporting."""
    return max(1.0e-6, float(SLICE_DZ_MM))


def _pressure_feedback_heights(z_bounds: Tuple[float, float]) -> np.ndarray:
    """Return fixed-1-mm-bin centres shared by construction and acceptance."""
    centres, _ = _pressure_feedback_grid(z_bounds)
    return centres


def _pressure_profile_stl(mesh, atoms_df: pd.DataFrame, edges_records: List[Dict],
                          z_bounds: Tuple[float, float], dz: float,
                          cache: Optional[Dict[float, object]] = None,
                          start_z: Optional[float] = None,
                          *, reuse_profile: Optional[pd.DataFrame] = None,
                          changed_z_intervals: Optional[List[Tuple[float, float]]] = None,
                          reuse_stats: Optional[Dict] = None) -> pd.DataFrame:
    cache = cache if cache is not None else {}
    zmin, zmax = z_bounds
    dz = max(1e-6, float(dz))
    if start_z is None:
        start_z = zmin
    edges_df = pd.DataFrame(edges_records)
    if edges_df.empty:
        edges_df = pd.DataFrame(columns=["source", "target", "radius", "bond_type", "role"])
    nodes_np = _make_nodes_np_for_analysis(atoms_df, z_bounds)
    cos_th = math.cos(math.radians(CONTACT_ANGLE_DEG))

    saved_rows = {}
    if (reuse_profile is not None and not reuse_profile.empty
            and changed_z_intervals is not None):
        raw_snapshot = reuse_profile.attrs.get("_pressure_raw_section_rows")
        if raw_snapshot is not None:
            saved_rows = {float(rec["z_mm"]): rec for rec in raw_snapshot}
        elif (set(_PRESSURE_RAW_SECTION_COLS).issubset(reuse_profile.columns)
              and not reuse_profile["z_mm"].duplicated().any()):
            saved_rows = {float(rec["z_mm"]): rec
                          for rec in reuse_profile.loc[:, _PRESSURE_RAW_SECTION_COLS].to_dict("records")}
    n_reused = n_recomputed = 0
    rows = []
    feedback_step = _pressure_feedback_spacing(z_bounds)

    coarse_centres_arr, coarse_widths_arr = _pressure_feedback_grid(z_bounds)
    coarse_centres = coarse_centres_arr.tolist()
    width_by_center = {round(float(z), 9): float(w)
                       for z, w in zip(coarse_centres_arr, coarse_widths_arr)}
    valid_probe_min = float(zmin) + 1.0e-6
    valid_probe_max = float(zmax) - 1.0e-6
    for z_mid in sorted(set(
            round(float(z), 9) for z in coarse_centres
            if valid_probe_min < float(z) < valid_probe_max)):
        old_row = saved_rows.get(float(z_mid))
        if (old_row is not None and not any(
                lo - 1.0e-9 <= z_mid <= hi + 1.0e-9
                for lo, hi in (changed_z_intervals or ())
        )):
            rows.append(old_row)
            n_reused += 1
            continue
        domain = _domain_at_z_cached(mesh, z_mid, cache)
        if domain.is_empty:
            continue
        n_recomputed += 1
        solid = _generate_true_solid_geometry_at_z_full(nodes_np, edges_df, atoms_df, z_mid)
        try:
            clipped = solid.intersection(domain) if solid is not None and not solid.is_empty else GeometryCollection()
        except Exception:
            clipped = GeometryCollection()
        A_solid = float(clipped.area) if not clipped.is_empty else 0.0
        A_domain = float(domain.area)
        A_liq = max(1e-9, A_domain - A_solid)
        try:
            liquid_geometry = domain.difference(clipped)
        except Exception:
            liquid_geometry = domain
        radius4_sum_mm4, n_flow_components = section_radius4_sum_mm4(
            liquid_geometry, min_area_mm2=FLOW_SECTION_MIN_COMPONENT_AREA_MM2
        )
        P_wet = _robust_internal_wetted_perimeter(clipped, domain)
        P_hydro = RHO_LIQ * G_GRAV * (max(0.0, z_mid - zmin) * 1e-3)
        P_cap = (GAMMA_NPM * cos_th * (P_wet * 1e-3)) / (A_liq * 1e-6) if A_liq > 1e-9 else 0.0
        r_eff = float((2.0 * A_liq / P_wet) if P_wet > 1e-9 else 0.0)
        safety_factor_z = _safety_factor_at_z(z_mid, z_bounds)
        rows.append({
            "z_mm": float(z_mid),
            "z_mid_mm": float(z_mid),
            "A_domain_mm2": A_domain,
            "A_solid_mm2": A_solid,
            "A_liq_mm2": A_liq,
            "axial_radius4_sum_mm4": radius4_sum_mm4,
            "n_flow_components": n_flow_components,
            "porosity": float(A_liq / max(A_domain, 1e-12)),
            "P_wet_mm": P_wet,
            "r_eff_mm": r_eff,
            "P_cap_Pa": float(P_cap),
            "P_hydro_Pa": float(P_hydro),
            "safety_factor_z": float(safety_factor_z),
            "fluid_viscosity_Pa_s": FLUID_VISCOSITY_PA_S,
            "slice_dz_mm": float(width_by_center.get(round(float(z_mid), 9), feedback_step)),
        })
    if reuse_stats is not None:
        reuse_stats["section_rows_reused"] = reuse_stats.get("section_rows_reused", 0) + n_reused
        reuse_stats["section_rows_recomputed"] = reuse_stats.get("section_rows_recomputed", 0) + n_recomputed
    result = apply_state_updated_viscous_profile(
        pd.DataFrame(rows),
        viscosity_pa_s=FLUID_VISCOSITY_PA_S,
        default_dz_mm=feedback_step,
        reference_flow_m3_s=VISC_REF_Q_M3_S,
        target_fill_time_s=TARGET_FILL_TIME_S,
    )

    result.attrs["_pressure_raw_section_rows"] = rows
    return result


def _capillary_ok_stl(mesh, atoms_df: pd.DataFrame, edges_records: List[Dict],
                      z_bounds: Tuple[float, float], dz: float,
                      cache: Optional[Dict[float, object]] = None,
                      *, reuse_profile: Optional[pd.DataFrame] = None,
                      changed_z_intervals: Optional[List[Tuple[float, float]]] = None,
                      reuse_stats: Optional[Dict] = None):
    df = _pressure_profile_stl(
        mesh, atoms_df, edges_records, z_bounds, dz, cache=cache,
        reuse_profile=reuse_profile, changed_z_intervals=changed_z_intervals,
        reuse_stats=reuse_stats,
    )
    if df.empty:
        return False, float(z_bounds[0]), 0.0, 0.0, df
    fail = df[_capillary_margin_values(df) < 0.0].sort_values("z_mm")
    if fail.empty:
        last = df.sort_values("z_mm").iloc[-1]
        return True, float(last["z_mm"]), float(last["P_cap_Pa"]), float(last["P_hydro_Pa"]), df
    row = fail.iloc[0]
    return False, float(row["z_mm"]), float(row["P_cap_Pa"]), float(row["P_hydro_Pa"]), df


def _smartfill_trial_capillary_ok_stl(mesh, atoms_after: pd.DataFrame, edges_after: List[Dict],
                                      atoms_before: pd.DataFrame, edges_before: List[Dict],
                                      profile_before: pd.DataFrame,
                                      z_bounds: Tuple[float, float], dz: float,
                                      cache: Dict[float, object],
                                      stats: Optional[Dict] = None):
    """Exact SmartFill pressure trial: reuse only sections outside changed solids.

    Fast path requires an UNMODIFIED prefix of both node and edge tables.
    Any deletion, reorder, coordinate edit or edge edit falls back to the
    existing full-slice calculation. No cached hydraulic integrals are reused.
    Rejected trials never mutate profile_before or its cached raw rows.
    """
    start = time.perf_counter()
    intervals: List[Tuple[float, float]] = []
    fast = bool(INTERNAL_DYNAMIC_SMARTFILL_INCREMENTAL_PRESSURE_ENABLE)
    if fast:
        fast = bool(
            profile_before is not None and not profile_before.empty
            and len(atoms_after) >= len(atoms_before)
            and len(edges_after) >= len(edges_before)
            and atoms_after.iloc[:len(atoms_before)].equals(atoms_before)
            and edges_after[:len(edges_before)] == edges_before
        )
    if fast:
        added = atoms_after.iloc[len(atoms_before):]
        before_xyz = {int(r.atom_id): (float(r.x), float(r.y), float(r.z))
                      for r in atoms_before.itertuples(index=False)}
        all_xyz = dict(before_xyz)
        extra_nodes = _make_nodes_np_for_analysis(added, z_bounds)
        for row, node in zip(added.itertuples(index=False), extra_nodes):
            ident = int(row.atom_id)
            if ident in all_xyz or not np.isfinite(node).all():
                fast = False
                break
            all_xyz[ident] = (float(row.x), float(row.y), float(row.z))

            intervals.append((float(node[2] - node[3]), float(node[2] + node[3])))
        if fast:
            for edge in edges_after[len(edges_before):]:
                u, v = int(edge.get("source", -1)), int(edge.get("target", -1))
                radius = float(edge.get("radius", float("nan")))
                if (u not in all_xyz or v not in all_xyz
                        or not np.isfinite(radius) or radius <= 0.0):
                    fast = False
                    break
                zu, zv = all_xyz[u][2], all_xyz[v][2]

                intervals.append((min(zu, zv) - radius, max(zu, zv) + radius))
    if stats is not None:
        key = "incremental_trials" if fast else "full_fallback_trials"
        stats[key] = int(stats.get(key, 0)) + 1
    result = _capillary_ok_stl(
        mesh, atoms_after, edges_after, z_bounds, dz, cache=cache,
        reuse_profile=profile_before if fast else None,
        changed_z_intervals=intervals if fast else None,
        reuse_stats=stats,
    )
    if stats is not None:
        stats["trial_pressure_seconds"] = stats.get("trial_pressure_seconds", 0.0) + (time.perf_counter() - start)
    return result


def _profile_quality_score(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return float("inf")
    work = df.sort_values("z_mid_mm" if "z_mid_mm" in df.columns else "z_mm").copy()
    penalty = 0.0
    lower_margins = _capillary_margin_values(work)
    penalty += float(np.square(np.clip(-lower_margins, 0.0, None)).sum()) * 100.0
    por = work.get("porosity", pd.Series(np.ones(len(work), dtype=float))).to_numpy(dtype=float)
    z_col = "z_mid_mm" if "z_mid_mm" in work.columns else "z_mm"
    z_bounds = (float(work[z_col].min()), float(work[z_col].max())) if z_col in work.columns and len(work) else (
    0.0, 1.0)
    por_floor = work[z_col].map(lambda z: _porosity_floor_at_z(float(z), z_bounds)).to_numpy(
        dtype=float) if z_col in work.columns else np.full(len(work), INTERNAL_DYNAMIC_MIN_LIQUID_FRAC, dtype=float)
    aliq = work.get("A_liq_mm2",
                    pd.Series(np.full(len(work), INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2, dtype=float))).to_numpy(
        dtype=float)
    penalty += float(np.square(np.clip(por_floor - por, 0.0, None)).sum()) * 1.0e6
    penalty += float(np.square(np.clip(INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2 - aliq, 0.0, None)).sum()) * 1.0e3
    cap = np.maximum(work["P_cap_Pa"].to_numpy(dtype=float), 1e-9)
    if "P_required_Pa" in work.columns:
        required = np.maximum(work["P_required_Pa"].to_numpy(dtype=float), 1.0e-9)
        surplus_ratio = cap / required - _capillary_trim_ratio()
        band_mask = work[z_col].to_numpy(dtype=float) >= float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM)
        penalty += float(np.square(np.clip(surplus_ratio[band_mask], 0.0, None)).sum()) * 1.0e6
    if len(cap) >= 2:
        ratios = np.maximum(cap[1:] / cap[:-1], cap[:-1] / cap[1:])
        penalty += float(np.square(np.clip(ratios - INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO, 0.0, None)).sum()) * 1.0e4
        log_cap = np.log(cap)
        penalty += float(np.square(np.clip(-np.diff(log_cap), 0.0, None)).sum()) * 5.0e4
        if len(cap) >= 3:
            penalty += float(np.square(np.diff(log_cap, n=2)).sum()) * 5.0e3
    return float(penalty)


def _pressure_z_values(df: pd.DataFrame) -> np.ndarray:
    if df is None or df.empty:
        return np.asarray([], dtype=float)
    z_col = "z_mid_mm" if "z_mid_mm" in df.columns else "z_mm"
    if z_col not in df.columns:
        return np.full(len(df), np.inf, dtype=float)
    return pd.to_numeric(df[z_col], errors="coerce").fillna(np.inf).to_numpy(dtype=float)


def _capillary_lower_ratio_values(
        df: pd.DataFrame, *, include_assembly_reserve: bool = True) -> np.ndarray:
    """Return the per-slice capillary lower-bound ratio.

    During internal construction and pre-stitch trimming, the configured assembly
    reserve is added to the normal lower bound at and above the pressure band
    start.  Final assembled acceptance explicitly disables this reserve and uses
    the reported physical/design lower bound alone.
    """
    if df is None or df.empty:
        return np.asarray([], dtype=float)
    z_values = _pressure_z_values(df)
    ratio = float(INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO)
    if include_assembly_reserve:
        ratio += float(INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO)
    ratios = np.full(len(df), ratio, dtype=float)

    ratios[z_values < float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM)] = 1.0
    return ratios


def _capillary_margin_values(
        df: pd.DataFrame, *, include_assembly_reserve: bool = True) -> np.ndarray:
    """Return margins above the selected lower bound."""
    if df is None or df.empty:
        return np.asarray([], dtype=float)
    if {"P_cap_Pa", "P_required_Pa"}.issubset(df.columns):
        cap = pd.to_numeric(df["P_cap_Pa"], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
        required = pd.to_numeric(df["P_required_Pa"], errors="coerce").fillna(np.inf).to_numpy(dtype=float)
        lower_ratio = _capillary_lower_ratio_values(
            df, include_assembly_reserve=include_assembly_reserve)
        return cap - lower_ratio * required
    margin_col = "safety_margin" if "safety_margin" in df.columns else "Safety_Margin"
    if margin_col not in df.columns:
        return np.asarray([], dtype=float)
    return pd.to_numeric(df[margin_col], errors="coerce").fillna(-np.inf).to_numpy(dtype=float)


def _capillary_trim_ratio() -> float:
    """Upper trimming target; assembly reserve belongs to the lower bound."""
    return max(
        float(INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO)
        + float(INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO),
        float(INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO),
    )


def _capillary_profile_passes(
        df: pd.DataFrame, *, include_assembly_reserve: bool = True) -> bool:
    margins = _capillary_margin_values(
        df, include_assembly_reserve=include_assembly_reserve)
    return bool(len(margins) > 0 and np.all(margins >= 0.0))


def _transport_reach_height_mm(df: pd.DataFrame, *, use_design_reserve: bool) -> float:
    """Return the contiguous transport reach from the bottom of the pressure grid.

    The returned height is the lower face of the first failing pressure bin.  If
    every bin passes, it is the upper face of the last bin.  ``use_design_reserve``
    selects either the pre-stitch production-rule lower bound (1.00 below
    the band start and 1.20 + assembly reserve above it) or the ordinary
    fixed-flow pressure budget ``P_cap - P_required``.
    """
    if df is None or df.empty:
        return 0.0
    work = df.sort_values("z_mid_mm" if "z_mid_mm" in df.columns else "z_mm").copy()
    z_col = "z_mid_mm" if "z_mid_mm" in work.columns else "z_mm"
    z = work[z_col].to_numpy(dtype=float)
    widths = (work["slice_dz_mm"].to_numpy(dtype=float)
              if "slice_dz_mm" in work.columns
              else np.full(len(work), float(SLICE_DZ_MM), dtype=float))
    if use_design_reserve:
        margin = _capillary_margin_values(work)
    else:
        if "transport_margin_Pa" in work.columns:
            margin = work["transport_margin_Pa"].to_numpy(dtype=float)
        elif "P_required_base_Pa" in work.columns:
            margin = (work["P_cap_Pa"].to_numpy(dtype=float)
                      - work["P_required_base_Pa"].to_numpy(dtype=float))
        else:
            margin = (work["P_cap_Pa"].to_numpy(dtype=float)
                      - work["P_required_Pa"].to_numpy(dtype=float))
    # An undefined pressure result cannot establish transport through that bin.
    failed = np.flatnonzero(~np.isfinite(margin) | (margin < 0.0))
    if len(failed):
        i = int(failed[0])
        return float(max(0.0, z[i] - 0.5 * widths[i]))
    return float(z[-1] + 0.5 * widths[-1])


def _viscous_mobility_values(df: pd.DataFrame) -> np.ndarray:
    """Return the geometry-aware capillary-viscous mobility at every slice."""
    if df is None or df.empty:
        return np.asarray([], dtype=float)
    if "capillary_viscous_mobility_Pa_m3" in df.columns:
        values = pd.to_numeric(
            df["capillary_viscous_mobility_Pa_m3"], errors="coerce"
        ).to_numpy(dtype=float)
    elif {"available_pressure_head_Pa", "R_geom_cumulative_m-3"}.issubset(df.columns):
        head = pd.to_numeric(
            df["available_pressure_head_Pa"], errors="coerce"
        ).to_numpy(dtype=float)
        resistance = pd.to_numeric(
            df["R_geom_cumulative_m-3"], errors="coerce"
        ).to_numpy(dtype=float)
        values = np.maximum(head, 0.0) / np.maximum(resistance, 1.0e-300)
    else:
        return np.zeros(len(df), dtype=float)
    return np.where(np.isfinite(values) & (values > 0.0), values, 0.0)


def _minimum_viscous_mobility(df: pd.DataFrame) -> float:
    values = _viscous_mobility_values(df)
    return float(np.min(values)) if len(values) else 0.0


def _pressure_failure_mechanism(df: pd.DataFrame, z_target: float) -> Dict[str, float]:
    """Diagnose whether a failed slice is capillary- or viscous-dominated.

    Diagnose the physical character of a failed slice.  Viscous dominance is
    based on the fraction of the two explicit resistance terms (hydrostatic +
    viscous) carried by viscous loss, together with an abnormal mobility/local-
    resistance bottleneck.  The timing of edge deletion is controlled separately
    in SmartFill: normal insertion must first accumulate ineffective searches.
    """
    result = {
        "mode": "capillary", "z_mm": float(z_target),
        "P_cap_Pa": 0.0, "P_hydro_Pa": 0.0, "P_viscous_Pa": 0.0,
        "viscous_share": 0.0, "hydro_only_margin_Pa": -float("inf"),
        "mobility_ratio_to_median": float("inf"),
        "segment_resistance_ratio_to_median": 0.0,
    }
    if df is None or df.empty:
        return result
    z_col = "z_mid_mm" if "z_mid_mm" in df.columns else "z_mm"
    if z_col not in df.columns:
        return result
    idx = (pd.to_numeric(df[z_col], errors="coerce") - float(z_target)).abs().idxmin()
    row = df.loc[idx]
    p_cap = float(row.get("P_cap_Pa", 0.0))
    p_hydro = float(row.get("P_hydro_Pa", 0.0))
    p_visc = float(row.get("P_visc_dynamic_Pa", row.get("P_viscous_Pa", 0.0)))
    p_required_base = float(row.get("P_required_base_Pa", p_hydro + p_visc))
    lower_ratio = float(_capillary_lower_ratio_values(df.loc[[idx]])[0])
    hydro_only_margin = p_cap - lower_ratio * p_hydro
    viscous_share = max(0.0, p_visc) / max(max(0.0, p_hydro) + max(0.0, p_visc), 1.0e-9)

    mobility = _viscous_mobility_values(df)
    mobility_i = float(mobility[df.index.get_loc(idx)]) if len(mobility) else 0.0
    positive = mobility[np.isfinite(mobility) & (mobility > 0.0)]
    mobility_median = float(np.median(positive)) if len(positive) else 0.0
    mobility_ratio = (mobility_i / max(mobility_median, 1.0e-300)
                      if mobility_median > 0.0 else float("inf"))

    segment_ratio = 0.0
    if "R_geom_segment_m-3" in df.columns:
        seg = pd.to_numeric(df["R_geom_segment_m-3"], errors="coerce").to_numpy(float)
        finite_pos = seg[np.isfinite(seg) & (seg > 0.0)]
        seg_i = float(row.get("R_geom_segment_m-3", 0.0))
        if len(finite_pos):
            segment_ratio = seg_i / max(float(np.median(finite_pos)), 1.0e-300)

    mobility_bottleneck = (
            mobility_ratio <= float(INTERNAL_DYNAMIC_VISCOUS_RELIEF_MOBILITY_MEDIAN_RATIO)
            or segment_ratio >= 1.50
    )
    viscous_dominated = bool(
        INTERNAL_DYNAMIC_VISCOUS_RELIEF_ENABLE
        and viscous_share >= float(INTERNAL_DYNAMIC_VISCOUS_RELIEF_MIN_VISCOUS_SHARE)
        and mobility_bottleneck
    )
    result.update({
        "mode": "viscous" if viscous_dominated else "capillary",
        "P_cap_Pa": p_cap, "P_hydro_Pa": p_hydro, "P_viscous_Pa": p_visc,
        "viscous_share": viscous_share, "hydro_only_margin_Pa": hydro_only_margin,
        "mobility_ratio_to_median": mobility_ratio,
        "segment_resistance_ratio_to_median": segment_ratio,
    })
    return result


def _meaningful_decrease(before: float, after: float) -> bool:
    minimum = max(1.0e-12, abs(float(before)) * float(INTERNAL_DYNAMIC_PRUNE_MIN_RELATIVE_IMPROVEMENT))
    return bool(np.isfinite(after) and float(after) <= float(before) - minimum)


def _prune_limit_reason(started: float, evaluations: int, stalled: int,
                        max_evaluations: int) -> str:
    if int(evaluations) >= max(0, int(max_evaluations)):
        return "evaluation limit"
    if time.perf_counter() - float(started) >= float(INTERNAL_DYNAMIC_PRUNE_MAX_SECONDS):
        return "time limit"
    if int(stalled) >= int(INTERNAL_DYNAMIC_PRUNE_MAX_STALL_EVALUATIONS):
        return "no significant progress"
    return ""


def _internal_edge_print_class(a, b, *, max_bridge_mm=None) -> str:
    """Global INTERNAL screen using the configured SLA elevation minimum.

    An inclined rod (including a short steep rod) may propagate bottom support.
    A sub-threshold short rod is a non-supporting bridge: both endpoints must
    independently have a qualified bottom-support path. This classification
    applies to bottom, middle and top; all remaining geometry/degree checks stay.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return "rejected"
    delta = b - a
    span = float(np.linalg.norm(delta))
    if span <= 1.0e-9:
        return "rejected"
    dz, dxy = abs(float(delta[2])), float(np.linalg.norm(delta[:2]))
    elevation = math.degrees(math.atan2(dz, max(dxy, 1.0e-12)))
    if elevation + 1.0e-8 >= float(INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG):
        return "inclined"
    limit = (float(INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM)
             if max_bridge_mm is None else float(max_bridge_mm))

    if INTERNAL_HORIZONTAL_BRIDGE_ENABLE and span < limit - 1.0e-9:
        return "short_bridge"
    return "rejected"


def _top_relaxed_internal_edge_class(a, b, *, z_bounds, smartfill_node_z=None):
    """Top-only, any-angle short internal rod: full 3-D span <=2.0 mm.

    Both endpoints must lie within the final 2.5 mm; one endpoint must reach
    the final 1.5 mm. This function ONLY classifies a potential link. Callers
    must additionally require independent strict-angle support of both ends;
    it never supplies the first downward support for a newly inserted node.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 'rejected'
    zmax = float(z_bounds[1])
    if (min(float(a[2]), float(b[2])) < zmax - float(INTERNAL_TOP_RELAX_ENDPOINT_HEIGHT_MM) - 1e-9
            or max(float(a[2]), float(b[2])) < zmax - float(INTERNAL_TOP_RELAX_ACTIVE_HEIGHT_MM) - 1e-9):
        return 'rejected'
    if (smartfill_node_z is not None and
            float(smartfill_node_z) < zmax - float(INTERNAL_TOP_RELAX_ACTIVE_HEIGHT_MM) - 1e-9):
        return 'rejected'
    length = float(np.linalg.norm(b - a))
    if (INTERNAL_HORIZONTAL_BRIDGE_ENABLE and
            1.0e-9 < length <= min(float(INTERNAL_TOP_RELAX_MAX_EDGE_LENGTH_MM),
                                   float(INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM)) + 1e-9):
        return 'top_relaxed'
    return 'rejected'


def _internal_edge_class_with_top_exception(a, b, edge, z_bounds):
    """Honor explicitly tagged, geometry-checked manufacturing exceptions."""
    if (str(edge.get('role', '')) == 'bottom_pillar_to_core' and
            float(np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))) <= 1.0e-8):
        return 'coincident_joint'
    if str(edge.get('role', '')) == 'layer_peer_relaxed' and z_bounds is not None:

        low, high = map(float, z_bounds)
        layer_h = max(high - low, 1.0e-9) / max(1, int(INTERNAL_DYNAMIC_LAYERS))
        pa, pb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        ia = min(int(INTERNAL_DYNAMIC_LAYERS) - 1,
                 max(0, int(math.floor((float(pa[2]) - low) / layer_h))))
        ib = min(int(INTERNAL_DYNAMIC_LAYERS) - 1,
                 max(0, int(math.floor((float(pb[2]) - low) / layer_h))))
        length = float(np.linalg.norm(pb - pa))
        if (ia == ib and 1.0e-9 < length
                <= float(INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_LENGTH_MM) + 1.0e-9):
            return 'abnormal_layer_peer'
    strict = _internal_edge_print_class(a, b)
    if strict != 'rejected':
        return strict
    if (str(edge.get('role', '')) == 'top_relaxed' and z_bounds is not None and
            _top_relaxed_internal_edge_class(a, b, z_bounds=z_bounds) == 'top_relaxed'):
        return 'top_relaxed'
    return 'rejected'


def _internal_support_state(atoms_df, edges_any, *, z_bounds=None):
    """Linear-time print-support audit of INTERNAL nodes only.

    Only strict-angle inclined rods propagate support upward. Sub-threshold
    short bridges and top-only any-angle short links do NOT create the first anchor; both
    endpoints must receive downward support independently.
    An undirected root-reachability check independently catches split islands.
    Bottom roots occupy the first nominal layer; an actual substrate/AM process
    must be verified separately. A print-support proxy is NOT a liquid-path test.
    """
    if atoms_df is None or atoms_df.empty:
        return {"nodes": set(), "roots": set(), "supported": set(), "root_connected": set(),
                "degree": {}, "classes": {}, "edge_count": 0, "bridge_bad": set()}
    names = atoms_df.get('face_name', pd.Series('', index=atoms_df.index)).astype(str)
    internal = atoms_df[~names.str.contains('surface', case=False, regex=False)].copy()
    xyz = {int(r.atom_id): np.asarray([r.x, r.y, r.z], float)
           for r in internal.itertuples(index=False)}
    if not xyz:
        return {"nodes": set(), "roots": set(), "supported": set(), "root_connected": set(),
                "degree": {}, "classes": {}, "edge_count": 0, "bridge_bad": set()}
    low = min(float(p[2]) for p in xyz.values())
    high = max(float(p[2]) for p in xyz.values())
    if z_bounds is not None:
        low, high = float(z_bounds[0]), float(z_bounds[1])
    layer_h = max(0.0, high - low) / max(1, int(INTERNAL_DYNAMIC_LAYERS))
    roots = {i for i, p in xyz.items() if p[2] <= low + layer_h + 1.0e-6}
    if not roots:
        roots = {min(xyz, key=lambda i: xyz[i][2])}
    upward = {i: [] for i in xyz}
    neighbors = {i: [] for i in xyz}
    degree = dict.fromkeys(xyz, 0)
    classes = {}
    raw = _edge_records(edges_any)
    count = 0
    for e in raw:
        if str(e.get('bond_type', '')) != 'internal':
            continue
        u, v = int(e.get('source', -1)), int(e.get('target', -1))
        if u not in xyz or v not in xyz or u == v:
            continue
        count += 1
        degree[u] += 1;
        degree[v] += 1
        neighbors[u].append(v);
        neighbors[v].append(u)
        cls = _internal_edge_class_with_top_exception(xyz[u], xyz[v], e, (low, high))
        classes[(min(u, v), max(u, v))] = cls
        if cls == 'inclined':
            lower, upper = (u, v) if xyz[u][2] <= xyz[v][2] else (v, u)
            upward[lower].append(upper)
        elif cls == 'coincident_joint':
            upward[u].append(v)
            upward[v].append(u)
    supported = set(roots)
    q = deque(roots)
    while q:
        for v in upward[q.popleft()]:
            if v not in supported:
                supported.add(v);
                q.append(v)
    root_connected = set(roots)
    q = deque(roots)
    while q:
        for v in neighbors[q.popleft()]:
            if v not in root_connected:
                root_connected.add(v);
                q.append(v)
    bridge_bad = {pair for pair, cls in classes.items()
                  if cls in ('short_bridge', 'top_relaxed', 'abnormal_layer_peer')
                  and (pair[0] not in supported or pair[1] not in supported)}
    if not INTERNAL_REQUIRE_BOTTOM_SUPPORT:
        supported = set(xyz)
        root_connected = set(xyz)
        bridge_bad = set()
    return {'nodes': set(xyz), 'roots': roots, 'supported': supported,
            'root_connected': root_connected, 'degree': degree, 'classes': classes,
            'edge_count': count, 'bridge_bad': bridge_bad}


def _dual_root_reachable(atoms_df, edges_any, root_ids):
    """BFS on actual retained SURFACE + INTERNAL + STITCH logical edges."""
    ids = set(atoms_df['atom_id'].astype(int)) if atoms_df is not None and not atoms_df.empty else set()
    if not ids: return set()
    adj = {i: [] for i in ids}
    for e in _edge_records(edges_any):
        if str(e.get('bond_type', '')) not in {'surface', 'internal', 'stitch'}:
            continue
        u, v = int(e['source']), int(e['target'])
        if u in adj and v in adj and u != v:
            adj[u].append(v);
            adj[v].append(u)
    reached = set(ids & set(root_ids))
    q = deque(reached)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in reached:
                reached.add(v);
                q.append(v)
    return reached


def _deletion_preserves_dual_attachment(before_atoms, before_edges, after_atoms, after_edges, *, root_ids):
    """Never erase prior cross-domain bottom attachment in post-stitch pruning.

    The test is skipped before any stitch exists; otherwise it preserves all
    RETAINED surface nodes which previously reached a bottom root. It also
    rejects removing the last stitch even when endpoints have degree > 1.
    """
    before = _edge_records(before_edges)
    if not any(str(e.get('bond_type', '')) == 'stitch' for e in before):
        return True
    after = _edge_records(after_edges)
    if not any(str(e.get('bond_type', '')) == 'stitch' for e in after):
        return False
    before_ids = set(before_atoms['atom_id'].astype(int))
    after_ids = set(after_atoms['atom_id'].astype(int))
    surfdf = before_atoms[before_atoms['face_name'].astype(str).str.contains('surface', case=False)]
    labels, components = _surface_graph_components(surfdf,
                                                   before_edges if isinstance(before_edges,
                                                                              pd.DataFrame) else pd.DataFrame(before))
    old_attached = {labels[int(e['source'])] if int(e['source']) in labels else
                    labels[int(e['target'])] for e in before
                    if str(e.get('bond_type', '')) == 'stitch' and
                    (int(e['source']) in labels or int(e['target']) in labels)}
    new_attached = {labels[int(e['source'])] if int(e['source']) in labels else
                    labels[int(e['target'])] for e in after
                    if str(e.get('bond_type', '')) == 'stitch' and
                    (int(e['source']) in labels or int(e['target']) in labels)}
    if not old_attached <= new_attached:
        return False
    name = before_atoms.set_index('atom_id')['face_name'].astype(str).to_dict()
    old_root = _dual_root_reachable(before_atoms, before, root_ids)
    formerly_attached = {i for i in (old_root & before_ids & after_ids)
                         if 'surface' in name.get(i, '').lower()}
    new_root = _dual_root_reachable(after_atoms, after, root_ids)
    return formerly_attached <= new_root


def _deletion_preserves_internal_support(before_atoms, before_edges, after_atoms, after_edges,
                                         *, z_bounds=None) -> bool:
    """Cheap O(V+E) prefilter BEFORE expensive hydraulic profile recomputation.

    Prevents new loss of previous vertical anchors or bottom-root connectivity;
    pre-existing unsupported nodes are reported, not silently 'fixed' by pruning.
    """
    old = _internal_support_state(before_atoms, before_edges, z_bounds=z_bounds)
    new = _internal_support_state(after_atoms, after_edges, z_bounds=z_bounds)
    kept = new['nodes']
    if not (old['supported'] & kept) <= new['supported']:
        return False
    if not (old['root_connected'] & kept) <= new['root_connected']:
        return False
    if new['bridge_bad'] - old['bridge_bad']:
        return False

    old_pairs = {(min(int(e.get('source', -1)), int(e.get('target', -1))),
                  max(int(e.get('source', -1)), int(e.get('target', -1))))
                 for e in _edge_records(before_edges) if str(e.get('bond_type', '')) == 'internal'}
    new_pairs = {(min(int(e.get('source', -1)), int(e.get('target', -1))),
                  max(int(e.get('source', -1)), int(e.get('target', -1))))
                 for e in _edge_records(after_edges) if str(e.get('bond_type', '')) == 'internal'}
    affected = {i for pair in old_pairs - new_pairs for i in pair if i in kept}
    if affected - new['supported']:
        return False

    if any(old['degree'].get(i, 0) > 0 and new['degree'].get(i, 0) == 0 for i in kept):
        return False

    seed_ids = set()
    if 'local_seed' in after_atoms.columns:
        seed_ids = set(after_atoms.loc[
                           after_atoms['local_seed'].fillna(False).astype(bool),
                           'atom_id',
                       ].astype(int))
    coords = {int(r.atom_id): float(r.z) for r in after_atoms.itertuples(index=False)
              if int(r.atom_id) in kept}
    lower_linked = set()
    for pair, edge_class in new['classes'].items():
        if edge_class == 'rejected' or pair[0] not in coords or pair[1] not in coords:
            continue
        u, v = pair
        if coords[u] < coords[v] - 1.0e-9:
            lower_linked.add(v)
        elif coords[v] < coords[u] - 1.0e-9:
            lower_linked.add(u)
    if set(kept) - set(new['roots']) - seed_ids - lower_linked:
        return False

    if not _deletion_preserves_dual_attachment(
            before_atoms, before_edges, after_atoms, after_edges, root_ids=old['roots']):
        return False
    return True


def audit_final_dual_domain(atoms_df, edges_df, *, z_bounds=None, strict_length_mm=None,
                            relaxed_length_mm=None, output_dir=None):
    """Read-only finished-assembly audit; does not relabel failure as success."""
    edges = _edge_records(edges_df)
    names = atoms_df.set_index('atom_id')['face_name'].astype(str).to_dict()
    coords = {int(r.atom_id): np.array([r.x, r.y, r.z], float)
              for r in atoms_df.itertuples(index=False)}
    surfaces = {i for i, n in names.items() if 'surface' in n.lower()}
    intdf = atoms_df[~atoms_df['face_name'].astype(str).str.contains('surface', case=False)]
    internal = _internal_support_state(intdf, edges, z_bounds=z_bounds)
    labels, comps = _surface_graph_components(atoms_df[atoms_df['atom_id'].isin(surfaces)], edges_df)
    roots_reached = _dual_root_reachable(atoms_df, edges, internal['roots'])
    degree = {i: 0 for i in coords}
    stitch_rows = []
    strict = float(STITCH_DISTANCE_MM if strict_length_mm is None else strict_length_mm)
    relaxed = float(STITCH_RELAXED_DISTANCE_MM if relaxed_length_mm is None else relaxed_length_mm)
    for e in edges:
        u, v = int(e['source']), int(e['target'])
        if u in degree and v in degree:
            degree[u] += 1;
            degree[v] += 1
        if str(e.get('bond_type', '')) != 'stitch': continue
        sid, iid = (u, v) if u in surfaces else (v, u)
        if sid not in surfaces or iid not in internal['nodes']:
            stitch_rows.append(dict(surface_id=sid, internal_id=iid, valid=False, reason='wrong_domains'))
            continue
        cls, angle, length = _stitch_print_class(coords[sid], coords[iid],
                                                 INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM)
        failures = []
        if cls == 'rejected': failures.append('overhang_or_span')
        if length > relaxed + 1e-9: failures.append('distance')
        if iid not in internal['supported'] or iid not in internal['root_connected']:
            failures.append('internal_support')
        if sid not in roots_reached: failures.append('no_root_path')
        if not bool(e.get('stitch_domain_checked', False)):
            failures.append('domain_not_checked')
        if not bool(e.get('stitch_radius_xy_screen_pass', False)):
            failures.append('radius_xy_not_screened')
        stitch_rows.append(dict(surface_id=sid, internal_id=iid, valid=not failures,
                                reason='|'.join(failures), angle_deg_above_horizontal=angle,
                                length_mm=length, print_class=cls,
                                relaxed_distance=length > strict + 1e-9,
                                surface_degree=degree.get(sid, 0), internal_degree=degree.get(iid, 0)))
    cover = {cid: bool(component & roots_reached) for cid, component in enumerate(comps)}

    unattached = [cid for cid, yes in cover.items() if not yes]
    repair_nodes = {int(endpoint) for e in edges
                    if str(e.get('role', '')) == 'support_repair'
                    for endpoint in (e.get('source', -1), e.get('target', -1))}
    abnormal_peer_nodes = {int(endpoint) for e in edges
                           if str(e.get('bond_type', '')) == 'internal'
                           and str(e.get('role', '')) == 'layer_peer_relaxed'
                           for endpoint in (e.get('source', -1), e.get('target', -1))}
    top_pressure_support_nodes = set()
    for e in edges:
        if str(e.get('bond_type', '')) != 'internal':
            continue
        u, v = int(e.get('source', -1)), int(e.get('target', -1))
        if names.get(u, '').startswith('SMART_FILL_LINEAR') or names.get(v, '').startswith('SMART_FILL_LINEAR'):
            top_pressure_support_nodes.update((u, v))

    def allowed_degree(node_id):
        routine = int(INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW)
        if node_id in internal['nodes']:
            endpoint_reserve = max(
                0,
                int(INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE),
                int(INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE),
            )
            if node_id in abnormal_peer_nodes:
                endpoint_reserve = max(
                    endpoint_reserve,
                    int(INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_DEGREE_RESERVE),
                )
            routine += endpoint_reserve
        return routine

    degree_bad = [i for i, d in degree.items() if d > allowed_degree(i)]
    unsupported_int = internal['nodes'] - internal['supported']
    bad_stitch = sum(not r['valid'] for r in stitch_rows)
    result = dict(surface_component_count=len(comps),
                  unattached_surface_component_ids=unattached,
                  anchored_surface_component_count=len(comps) - len(unattached),
                  stitch_count=len(stitch_rows), invalid_stitch_count=bad_stitch,
                  relaxed_distance_stitch_count=sum(bool(r.get('relaxed_distance')) for r in stitch_rows),
                  short_bridge_stitch_count=sum(r.get('print_class') == 'short_bridge' for r in stitch_rows),
                  routine_degree_limit=int(INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW),
                  top_pressure_support_degree_limit=int(
                      INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                      + INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE),
                  all_internal_nodes_have_new_node_inbound_reserve=True,
                  new_node_inbound_reserve=int(INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE),
                  support_repair_degree_reserve=int(INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE),
                  support_repair_absolute_max_degree=int(
                      INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                      + INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE),
                  abnormal_layer_peer_endpoint_degree_limit=int(
                      INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                      + max(0, INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE,
                            INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE,
                            INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_DEGREE_RESERVE)),
                  internal_nodes_using_shared_reserve_count=sum(
                      i in internal['nodes'] and d > INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                      for i, d in degree.items()),
                  degree_over_limit_node_ids=degree_bad,
                  downward_unsupported_internal_count=len(unsupported_int),
                  dual_domain_pass=bool(comps and not unattached and stitch_rows and
                                        not bad_stitch and not degree_bad and not unsupported_int))
    if output_dir is not None:
        folder = Path(output_dir);
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(stitch_rows).to_csv(folder / 'dual_stitch_final_edges.csv', index=False)
        (folder / 'dual_stitch_final_audit.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def export_internal_support_audit(atoms_df, edges_df, output_dir, *, z_bounds=None):
    """Read-only final internal printability proxy; no liquid connectivity claim."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = _internal_support_state(atoms_df, edges_df, z_bounds=z_bounds)
    cls_counts = {x: list(state['classes'].values()).count(x)
                  for x in ('inclined', 'short_bridge', 'abnormal_layer_peer',
                            'top_relaxed', 'coincident_joint', 'rejected')}
    seed_ids = set()
    if 'local_seed' in atoms_df.columns:
        seed_ids = set(atoms_df.loc[
                           atoms_df['local_seed'].fillna(False).astype(bool), 'atom_id'].astype(int))
    coord_all = {int(r.atom_id): np.asarray([r.x, r.y, r.z], float)
                 for r in atoms_df.itertuples(index=False)
                 if int(r.atom_id) in state['nodes']}
    local_lower_ids = set()
    abnormal_peer_ids = set()
    for pair, edge_class in state['classes'].items():
        if edge_class == 'rejected' or pair[0] not in coord_all or pair[1] not in coord_all:
            continue
        if edge_class == 'abnormal_layer_peer':
            abnormal_peer_ids.update(pair)
        u, v = pair
        if coord_all[u][2] < coord_all[v][2] - 1.0e-9:
            local_lower_ids.add(v)
        elif coord_all[v][2] < coord_all[u][2] - 1.0e-9:
            local_lower_ids.add(u)
    rows = []
    for i in sorted(state['nodes']):
        rows.append({'atom_id': i, 'bottom_root': i in state['roots'],
                     'downward_supported': i in state['supported'],
                     'bottom_root_connected': i in state['root_connected'],
                     'local_seed': i in seed_ids,
                     'has_local_lower_connection': i in local_lower_ids,
                     'internal_degree': state['degree'].get(i, 0)})
    pd.DataFrame(rows, columns=['atom_id', 'bottom_root', 'downward_supported',
                                'bottom_root_connected', 'local_seed', 'has_local_lower_connection',
                                'internal_degree']).to_csv(
        output_dir / 'internal_print_support_nodes.csv', index=False)
    summary = {'scope': 'internal solid graph only; NOT fluid percolation or validated AM span',
               'build_direction': [0, 0, 1],
               'inclined_min_elevation_from_horizontal_deg': float(INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG),
               'any_angle_internal_short_edge_max_3d_length_mm_provisional': float(
                   INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM),
               'top_any_angle_short_edge_max_3d_length_mm_provisional': min(
                   float(INTERNAL_TOP_RELAX_MAX_EDGE_LENGTH_MM), float(INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM)),
               'bottom_support_required': bool(INTERNAL_REQUIRE_BOTTOM_SUPPORT),
               'short_edges_need_independent_endpoint_support': False,
               'bridge_length_experimentally_validated': False,
               'top_any_angle_short_edge_min_elevation_deg': 0.0,
               'top_relaxed_max_full_length_mm_provisional': float(INTERNAL_TOP_RELAX_MAX_EDGE_LENGTH_MM),
               'top_relaxed_3d_am_experimentally_validated': False,
               'strict_configured_angle_screen_pass': False,
               'strict_45deg_screen_pass': False,
               'internal_node_count': len(state['nodes']),
               'internal_edge_count': state['edge_count'],
               'isolated_internal_node_count': sum(
                   int(state['degree'].get(i, 0) == 0) for i in state['nodes']),

               'nonseed_without_local_lower_connection_count': len(
                   set(state['nodes']) - set(state['roots']) - seed_ids - local_lower_ids - abnormal_peer_ids),
               'edge_class_counts': cls_counts,
               'bottom_root_count': len(state['roots']),
               'downward_unsupported_node_count': len(state['nodes'] - state['supported']),
               'root_disconnected_node_count': len(state['nodes'] - state['root_connected']),
               'non_supporting_edge_without_two_supported_ends_count': len(state['bridge_bad'])}
    summary['abnormal_layer_peer_node_count'] = len(abnormal_peer_ids)
    summary['abnormal_layer_exception_present'] = bool(abnormal_peer_ids)
    summary['abnormal_layer_exception_policy'] = (
        'Allowed only for state-detected persistent no-lower-connection layers; '
        'nodes must retain at least one same-layer graph edge and all final pressure checks must pass.'
    )
    summary['print_support_screen_pass'] = bool(
        summary['downward_unsupported_node_count'] == 0
        and summary['root_disconnected_node_count'] == 0
        and summary['non_supporting_edge_without_two_supported_ends_count'] == 0
        and summary['isolated_internal_node_count'] == 0
        and summary['nonseed_without_local_lower_connection_count'] == 0
        and summary['edge_class_counts']['rejected'] == 0
    )
    summary['strict_configured_angle_screen_pass'] = bool(
        summary['print_support_screen_pass'] and summary['edge_class_counts']['top_relaxed'] == 0
        and summary['edge_class_counts']['short_bridge'] == 0
        and summary['edge_class_counts']['abnormal_layer_peer'] == 0
    )
    summary['strict_45deg_screen_pass'] = bool(
        float(INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG) >= 45.0
        and summary['strict_configured_angle_screen_pass'])
    coord = {int(r.atom_id): np.array([r.x, r.y, r.z], float)
             for r in atoms_df.itertuples(index=False)
             if int(r.atom_id) in state['nodes']}
    edge_rows = []
    for e in _edge_records(edges_df):
        if str(e.get('bond_type', '')) != 'internal':
            continue
        u, v = int(e.get('source', -1)), int(e.get('target', -1))
        if u not in coord or v not in coord:
            continue
        delta = coord[v] - coord[u]
        length = float(np.linalg.norm(delta))
        dxy = float(np.linalg.norm(delta[:2]));
        dz = abs(float(delta[2]))
        cls = _internal_edge_class_with_top_exception(coord[u], coord[v], e, z_bounds)
        edge_rows.append({'source': u, 'target': v, 'role': e.get('role', ''),
                          'diameter_mm': 2.0 * float(e.get('radius', INTERNAL_DIAM_TOP_MM / 2)),
                          'length_mm': length, 'horizontal_span_mm': dxy, 'delta_z_mm': dz,
                          'angle_from_vertical_deg': math.degrees(math.atan2(dxy, max(dz, 1e-12))),
                          'manufacturing_screen_class': cls,
                          'endpoints_down_supported': u in state['supported'] and v in state['supported'],
                          'bridge_end_support_pass': cls not in (
                          'short_bridge', 'top_relaxed', 'abnormal_layer_peer') or
                                                     (u in state['supported'] and v in state['supported'])})
    pd.DataFrame(edge_rows, columns=['source', 'target', 'role', 'diameter_mm', 'length_mm',
                                     'horizontal_span_mm', 'delta_z_mm', 'angle_from_vertical_deg',
                                     'manufacturing_screen_class',
                                     'endpoints_down_supported', 'bridge_end_support_pass']).to_csv(
        output_dir / 'internal_print_support_edges.csv', index=False)
    (output_dir / 'internal_print_support_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def audit_abnormal_layer_connectivity(atoms_df, edges_df, *, z_bounds, output_dir=None):
    """Audit the explicit abnormal-layer exception without treating its existence as failure."""
    atoms = atoms_df.copy() if atoms_df is not None else pd.DataFrame()
    present_ids = set(atoms['atom_id'].astype(int)) if not atoms.empty else set()
    abnormal_ids = set()
    if not atoms.empty and 'peer_only_relaxed' in atoms.columns:
        abnormal_ids.update(atoms.loc[
                                atoms['peer_only_relaxed'].fillna(False).astype(bool), 'atom_id'
                            ].astype(int))

    coord = {int(r.atom_id): np.asarray([r.x, r.y, r.z], dtype=float)
             for r in atoms.itertuples(index=False)}
    valid_degree = {i: 0 for i in abnormal_ids}
    retained_edges = 0
    invalid_edges = 0
    edge_rows = []
    for edge in _edge_records(edges_df):
        if (str(edge.get('bond_type', '')) != 'internal'
                or str(edge.get('role', '')) != 'layer_peer_relaxed'):
            continue
        u, v = int(edge.get('source', -1)), int(edge.get('target', -1))
        if u not in coord or v not in coord:
            continue
        abnormal_ids.add(u)
        valid_degree.setdefault(u, 0)
        cls = _internal_edge_class_with_top_exception(coord[u], coord[v], edge, z_bounds)
        valid = cls != 'rejected'
        retained_edges += 1
        invalid_edges += int(not valid)
        if valid:
            if u in valid_degree:
                valid_degree[u] += 1
            if v in valid_degree:
                valid_degree[v] += 1
        edge_rows.append({
            'source': u, 'target': v,
            'length_mm': float(np.linalg.norm(coord[v] - coord[u])),
            'manufacturing_screen_class': cls, 'valid': bool(valid),
        })

    abnormal_ids &= present_ids
    valid_degree = {i: int(valid_degree.get(i, 0)) for i in abnormal_ids}
    disconnected = sorted(i for i in abnormal_ids if valid_degree.get(i, 0) == 0)
    low, high = map(float, z_bounds)
    layer_h = max(high - low, 1.0e-9) / max(1, int(INTERNAL_DYNAMIC_LAYERS))
    abnormal_layers = sorted({
        min(int(INTERNAL_DYNAMIC_LAYERS) - 1,
            max(0, int(math.floor((float(coord[i][2]) - low) / layer_h)))) + 1
        for i in abnormal_ids if i in coord
    })
    result = {
        'abnormal_layer_exception_present': bool(abnormal_ids),
        'abnormal_layer_numbers_1_based': abnormal_layers,
        'abnormal_node_count': len(abnormal_ids),
        'retained_abnormal_peer_edge_count': int(retained_edges),
        'invalid_abnormal_peer_edge_count': int(invalid_edges),
        'disconnected_abnormal_node_count': len(disconnected),
        'disconnected_abnormal_node_ids': disconnected,
        'abnormal_layer_connectivity_pass': bool(not disconnected and invalid_edges == 0),
        'acceptance_interpretation': (
            'The presence of an abnormal layer is permitted. It passes this gate only when every '
            'retained abnormal node has a valid same-layer peer connection; the final pressure gate '
            'is evaluated separately and must also pass.'
        ),
        'manufacturing_exception': {
            'maximum_peer_edges_per_new_node': int(INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_PEERS),
            'maximum_full_3d_length_mm': float(INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_LENGTH_MM),
            'minimum_elevation_deg': 0.0,
            'endpoint_degree_reserve': int(INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_DEGREE_RESERVE),
            'final_peer_endpoint_degree_limit': int(
                INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                + max(0, INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE,
                      INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE,
                      INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_DEGREE_RESERVE)),
            'ordinary_layers_unchanged': True,
            'stl_centreline_and_same_component_checks_retained': True,
        },
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(edge_rows, columns=[
            'source', 'target', 'length_mm', 'manufacturing_screen_class', 'valid'
        ]).to_csv(out / 'abnormal_layer_peer_edges.csv', index=False)
        (out / 'abnormal_layer_connectivity.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def trim_final_internal_pressure(mesh, atoms_df: pd.DataFrame, edges_df: pd.DataFrame,
                                 z_bounds: Tuple[float, float], batch_size: int = 12):
    """Final INTERNAL-only surplus-pressure pruning, before surface stitching.

    Invoke after bottom print pillars are added. This routine changes only
    the internal graph; the stitched assembly is subsequently read-only.
    """
    started = time.perf_counter()
    atoms_work = atoms_df.copy().reset_index(drop=True)
    records = _edge_records(edges_df)
    cache: Dict[float, object] = {}
    profile = _pressure_profile_stl(
        mesh, atoms_work, records, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=cache,
    )
    removed_nodes = 0
    removed_edges = 0
    rejected_edges: Set[Tuple[int, int, str]] = set()
    trial_batch_size = max(1, int(batch_size))
    edge_evaluation = 0
    total_evaluations = 0
    stalled_evaluations = 0
    stop_reason = ""

    def upper_rank(df: pd.DataFrame) -> Tuple[int, float, float]:
        if df is None or df.empty:
            return (10 ** 9, float("inf"), float("inf"))
        z_col = "z_mid_mm" if "z_mid_mm" in df.columns else "z_mm"
        work = df[pd.to_numeric(df[z_col], errors="coerce") >= float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM)]
        if work.empty:
            return (0, 0.0, 0.0)
        ratio = work["P_cap_Pa"].to_numpy(dtype=float) / np.maximum(
            work["P_required_Pa"].to_numpy(dtype=float), 1e-12,
        )
        surplus = np.clip(ratio - float(INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO), 0.0, None)
        return int(np.count_nonzero(surplus > 1e-9)), float(surplus.sum()), float(surplus.max())

    def upper_improves(before: Tuple[int, float, float], after: Tuple[int, float, float]) -> bool:
        if after[0] != before[0]:
            return after[0] < before[0]
        return _meaningful_decrease(before[1], after[1])

    rejected_nodes: Set[int] = set()
    node_batch_size = max(1, int(batch_size))
    node_evaluation = 0
    if not _capillary_profile_passes(profile):
        stop_reason = "lower capillary bound failed before trim"
        log("  InternalPressureTrim skipped: internal structure has not passed the lower capillary bound.")
    while not stop_reason:
        stop_reason = _prune_limit_reason(
            started, total_evaluations, stalled_evaluations,
            INTERNAL_DYNAMIC_FINAL_TRIM_MAX_EVALUATIONS,
        )
        if stop_reason:
            break
        rank_before = upper_rank(profile)
        if rank_before[0] == 0:
            break
        z_col = "z_mid_mm" if "z_mid_mm" in profile.columns else "z_mm"
        band = profile[
            pd.to_numeric(profile[z_col], errors="coerce") >= float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM)].copy()
        band["upper_surplus"] = (
                band["P_cap_Pa"].astype(float) / np.maximum(band["P_required_Pa"].astype(float), 1e-12)
                - float(INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO)
        )
        issues = band[band["upper_surplus"] > 1e-9].sort_values("upper_surplus", ascending=False)
        chosen_nodes: List[int] = []
        chosen_heights: List[float] = []
        names = atoms_work.get("face_name", pd.Series("", index=atoms_work.index)).astype(str)
        removable = ~names.str.contains("surface|bottom_pillar", case=False, regex=True)
        for issue in issues.itertuples(index=False):
            z_issue = float(getattr(issue, z_col))
            candidates = atoms_work[
                removable
                & ~atoms_work["atom_id"].astype(int).isin(rejected_nodes)
                & ~atoms_work["atom_id"].astype(int).isin(chosen_nodes)
                ].copy()
            if candidates.empty:
                continue
            candidate_names = candidates.get("face_name", pd.Series("", index=candidates.index)).astype(str)
            candidates["delete_priority"] = np.select(
                [candidate_names.str.contains("SMART_FILL|SMART_BRIDGE", regex=True),
                 candidate_names.str.contains("PHYS_GAP", regex=False)],
                [0, 1], default=2,
            )
            candidates["dist_z"] = np.abs(candidates["z"].astype(float) - z_issue)
            row = candidates.sort_values(
                ["delete_priority", "dist_z", "atom_id"], ascending=[True, True, False]
            ).iloc[0]
            chosen_nodes.append(int(row["atom_id"]))
            chosen_heights.append(z_issue)
            if len(chosen_nodes) >= node_batch_size:
                break
        if not chosen_nodes:
            stop_reason = "no removable node"
            break
        node_evaluation += 1
        total_evaluations += 1
        chosen_set = set(chosen_nodes)
        trial_atoms = atoms_work[~atoms_work["atom_id"].astype(int).isin(chosen_set)].reset_index(drop=True)
        trial_records = [
            edge for edge in records
            if int(edge.get("source", -1)) not in chosen_set and int(edge.get("target", -1)) not in chosen_set
        ]
        if not _deletion_preserves_internal_support(
                atoms_work, records, trial_atoms, trial_records, z_bounds=z_bounds
        ):
            rejected_nodes.update(chosen_set)
            node_batch_size = max(1, len(chosen_set) // 2) if len(chosen_set) > 1 else max(1, int(batch_size))
            continue
        log(
            f"  InternalPressureTrim node eval {node_evaluation} "
            f"(total {total_evaluations}/{INTERNAL_DYNAMIC_FINAL_TRIM_MAX_EVALUATIONS}): "
            f"evaluating {len(chosen_set)}-node deletion."
        )
        trial_profile = _pressure_profile_stl(
            mesh, trial_atoms, trial_records, z_bounds,
            INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=cache,
        )
        rank_after = upper_rank(trial_profile)
        if _capillary_profile_passes(trial_profile) and upper_improves(rank_before, rank_after):
            atoms_work = trial_atoms
            records = trial_records
            profile = trial_profile
            removed_nodes += len(chosen_set)
            node_batch_size = max(1, int(batch_size))
            stalled_evaluations = 0
            log(
                f"  InternalPressureTrim node eval {node_evaluation}: removed {len(chosen_set)} nodes across "
                f"z={min(chosen_heights):.2f}-{max(chosen_heights):.2f}mm; "
                f"remaining upper-band violations={upper_rank(profile)[0]}."
            )
            continue
        stalled_evaluations += 1
        if len(chosen_set) > 1:
            node_batch_size = max(1, len(chosen_set) // 2)
            log(
                f"  InternalPressureTrim node eval {node_evaluation}: rejected {len(chosen_set)}-node batch; "
                f"retrying with batch size {node_batch_size}."
            )
            continue
        rejected_nodes.update(chosen_set)
        node_batch_size = max(1, int(batch_size))

    atom_xyz = atoms_work.set_index("atom_id")[["x", "y", "z"]].to_dict("index")

    if stop_reason in {"no removable node", "no significant progress"}:
        log(f"  InternalPressureTrim switching from nodes to edges after {stop_reason}.")
        stop_reason = ""
        stalled_evaluations = 0

    while not stop_reason:
        stop_reason = _prune_limit_reason(
            started, total_evaluations, stalled_evaluations,
            INTERNAL_DYNAMIC_FINAL_TRIM_MAX_EVALUATIONS,
        )
        if stop_reason:
            break
        rank_before = upper_rank(profile)
        if rank_before[0] == 0:
            stop_reason = ""
            break
        z_col = "z_mid_mm" if "z_mid_mm" in profile.columns else "z_mm"
        band = profile[
            pd.to_numeric(profile[z_col], errors="coerce") >= float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM)].copy()
        band["upper_surplus"] = (
                band["P_cap_Pa"].astype(float) / np.maximum(band["P_required_Pa"].astype(float), 1e-12)
                - float(INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO)
        )
        issues = band[band["upper_surplus"] > 1e-9].sort_values("upper_surplus", ascending=False)

        degree: Dict[int, int] = {}
        for edge in records:
            u, v = int(edge.get("source", -1)), int(edge.get("target", -1))
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
        chosen_indices: List[int] = []
        chosen_keys: Set[Tuple[int, int, str]] = set()
        chosen_heights: List[float] = []
        issue_rows = list(issues.itertuples(index=False))
        issue_heights = sorted({float(getattr(issue, z_col)) for issue in issue_rows})
        candidate_by_height = {height: [] for height in issue_heights}
        issue_height_array = np.asarray(issue_heights, dtype=float)
        for edge_idx, edge in enumerate(records):
            bond_type = str(edge.get("bond_type", ""))
            if bond_type not in {"internal", "stitch"} or "bottom_pillar" in str(edge.get("role", "")).lower():
                continue
            u, v = int(edge.get("source", -1)), int(edge.get("target", -1))
            key = (min(u, v), max(u, v), bond_type)
            if key in rejected_edges or degree.get(u, 0) <= 1 or degree.get(v, 0) <= 1:
                continue
            if u not in atom_xyz or v not in atom_xyz:
                continue
            zu, zv = float(atom_xyz[u]["z"]), float(atom_xyz[v]["z"])
            radius = float(edge.get("radius", INTERNAL_DIAM_TOP_MM / 2.0))
            left = int(np.searchsorted(issue_height_array, min(zu, zv) - radius, side="left"))
            right = int(np.searchsorted(issue_height_array, max(zu, zv) + radius, side="right"))
            z_mid = 0.5 * (zu + zv)
            for z_issue in issue_heights[left:right]:
                candidate_by_height[z_issue].append(
                    (0 if bond_type == "internal" else 1, abs(z_mid - z_issue), radius, edge_idx, key)
                )
        for candidates in candidate_by_height.values():
            candidates.sort()
        for issue in issue_rows:
            z_issue = float(getattr(issue, z_col))
            candidate = next(
                (item for item in candidate_by_height[z_issue] if item[4] not in chosen_keys),
                None,
            )
            if candidate is not None:
                _, _, _, edge_idx, key = candidate
                chosen_indices.append(edge_idx)
                chosen_keys.add(key)
                chosen_heights.append(z_issue)
            if len(chosen_indices) >= trial_batch_size:
                break

        if not chosen_indices:
            log("  InternalPressureTrim stopped: no safe removable edge remains in an over-limit slice.")
            stop_reason = "no removable edge"
            break
        edge_evaluation += 1
        total_evaluations += 1
        chosen_index_set = set(chosen_indices)
        trial_records = [edge for idx, edge in enumerate(records) if idx not in chosen_index_set]
        if not _deletion_preserves_internal_support(
                atoms_work, records, atoms_work, trial_records, z_bounds=z_bounds
        ):
            if len(chosen_index_set) > 1:
                trial_batch_size = max(1, len(chosen_index_set) // 2)
            else:
                rejected_edges.update(chosen_keys)
            continue
        log(
            f"  InternalPressureTrim edge eval {edge_evaluation} "
            f"(total {total_evaluations}/{INTERNAL_DYNAMIC_FINAL_TRIM_MAX_EVALUATIONS}): "
            f"evaluating {len(chosen_index_set)}-edge deletion."
        )
        trial_profile = _pressure_profile_stl(
            mesh, atoms_work, trial_records, z_bounds,
            INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=cache,
        )
        rank_after = upper_rank(trial_profile)
        if _capillary_profile_passes(trial_profile) and upper_improves(rank_before, rank_after):
            records = trial_records
            profile = trial_profile
            removed_edges += len(chosen_index_set)
            trial_batch_size = max(1, int(batch_size))
            stalled_evaluations = 0
            log(
                f"  InternalPressureTrim edge eval {edge_evaluation}: removed {len(chosen_index_set)} edges across "
                f"z={min(chosen_heights):.2f}-{max(chosen_heights):.2f}mm; "
                f"remaining upper-band violations={rank_after[0]}."
            )
            continue
        stalled_evaluations += 1
        if len(chosen_index_set) > 1:
            trial_batch_size = max(1, len(chosen_index_set) // 2)
            log(
                f"  InternalPressureTrim edge eval {edge_evaluation}: rejected {len(chosen_index_set)}-edge batch; "
                f"retrying with batch size {trial_batch_size}."
            )
            continue
        rejected_edges.update(chosen_keys)
        trial_batch_size = max(1, int(batch_size))

    final_rank = upper_rank(profile)
    lower_passed = _capillary_profile_passes(profile)
    if lower_passed and final_rank[0] == 0:
        status = "passed"
        stop_reason = ""
    elif not lower_passed:
        status = "lower_bound_failed"
        stop_reason = stop_reason or "lower capillary bound failed"
    else:
        status = "incomplete"
        stop_reason = stop_reason or "upper-band violations remain"
    profile.attrs["internal_pressure_trim"] = {
        "status": status,
        "stop_reason": stop_reason,
        "evaluations": int(total_evaluations),
        "elapsed_seconds": float(time.perf_counter() - started),
        "removed_nodes": int(removed_nodes),
        "removed_edges": int(removed_edges),
        "remaining_upper_band_violations": int(final_rank[0]),
        "lower_bound_passed": bool(lower_passed),
    }
    log(
        f"  InternalPressureTrim finished: status={status}, reason={stop_reason or 'target reached'}, "
        f"evaluations={total_evaluations}, remaining upper-band violations={final_rank[0]}."
    )

    return (
        atoms_work,
        _dedup_edges_local(pd.DataFrame(records)),
        profile,
        int(removed_nodes),
        int(removed_edges),
    )


def _capillary_safety_rank(df: pd.DataFrame) -> Tuple[int, float, float]:
    """Lexicographic safety rank: fewer failures, less deficit, better worst slice."""
    margins = _capillary_margin_values(df)
    if len(margins) == 0:
        return (10 ** 9, float("inf"), float("inf"))
    deficits = np.clip(-margins, 0.0, None)
    return (
        int(np.count_nonzero(margins < 0.0)),
        float(deficits.sum()),
        -float(np.min(margins)),
    )


def _profile_trial_improves(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """Improve the complete lower-bound rank without breaking physical safety."""
    if before is None or after is None or len(before) != len(after):
        return False
    z_col = "z_mid_mm" if "z_mid_mm" in before.columns else "z_mm"
    if z_col not in after.columns or not np.allclose(
            before[z_col].to_numpy(dtype=float), after[z_col].to_numpy(dtype=float),
            rtol=0.0, atol=1.0e-9, equal_nan=False):
        return False
    before_margins = _capillary_margin_values(before)
    after_margins = _capillary_margin_values(after)
    if not _capillary_profile_passes(before):
        if not _capillary_safety_rank(after) < _capillary_safety_rank(before):
            return False

        before_physical = (
                before["P_cap_Pa"].to_numpy(dtype=float)
                - before["P_required_Pa"].to_numpy(dtype=float)
        )
        after_physical = (
                after["P_cap_Pa"].to_numpy(dtype=float)
                - after["P_required_Pa"].to_numpy(dtype=float)
        )
        return not np.any(
            (before_physical >= -1.0e-9) & (after_physical < -1.0e-9)
        )
    previously_safe = before_margins >= -1.0e-9
    if np.any(after_margins[previously_safe] < -1.0e-9):
        return False
    return _capillary_profile_passes(after) and _profile_quality_score(after) < _profile_quality_score(before)


def _adaptive_repair_budget(failed, previous_budget, max_budget):
    """Size a transaction from current domain capacity and pressure shortage.

    The CLI maximum is a resource ceiling, never a model-specific quota.
    Geometric feasibility and exact pressure gain subsequently backtrack it.
    """
    if failed.empty:
        return 0
    capacity = failed.get('A_liq_mm2', pd.Series(1.0, index=failed.index)).clip(lower=0)
    porosity = failed.get('porosity', pd.Series(1.0, index=failed.index))
    floor = failed.get('porosity_floor', pd.Series(0.0, index=failed.index))
    room = ((porosity - floor) / porosity.clip(lower=1e-9)).clip(0, 1)
    deficit = failed.get('lower_deficit_Pa', pd.Series(1.0, index=failed.index)).clip(lower=0)
    required = failed.get('P_required_Pa', pd.Series(1.0, index=failed.index)).clip(lower=1e-9)
    demand = float(np.sum(np.sqrt(capacity * room) * (deficit / required).clip(0, 1)))
    proposed = max(2, int(math.ceil(math.sqrt(len(failed)) + demand)))
    return min(max(2, int(max_budget)), max(2, int(previous_budget)), proposed)


def _smartfill_weighted_allocation(weights, budget, per_slice_cap):
    """Give every scheduled height one proposal; allocate spare slots by deficit weight.

    The per-height cap and the transaction budget are hard search limits. A
    transaction cannot cover more independent heights than its node budget;
    _gradient_repair_schedule visits the remaining heights on subsequent rounds.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    if n == 0:
        return np.zeros(0, dtype=int)
    budget = max(0, int(budget))
    caps = np.asarray(per_slice_cap, dtype=int)
    if caps.ndim == 0:
        caps = np.full(n, max(1, int(caps)), dtype=int)
    else:
        caps = np.maximum(1, np.broadcast_to(caps, (n,)).copy())
    if budget < n:
        raise ValueError("SmartFill transaction budget must cover every scheduled height")
    alloc = np.ones(n, dtype=int)
    free = min(budget - n, int(np.sum(caps - 1)))
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if not np.any(w):
        w[:] = 1.0

    while free > 0:
        eligible = np.flatnonzero(alloc < caps)
        if not len(eligible):
            break
        local_w = w[eligible]
        if not np.any(local_w):
            local_w = np.ones(len(eligible), dtype=float)

        quotas = free * local_w / float(local_w.sum())
        floors = np.minimum(np.floor(quotas).astype(int), caps[eligible] - alloc[eligible])
        if np.any(floors):
            alloc[eligible] += floors
            free -= int(floors.sum())
            continue
        idx = int(eligible[int(np.argmax(quotas))])
        alloc[idx] += 1
        free -= 1
    return alloc


def _gradient_repair_schedule(failed, profile, budget, visits, z_bounds):
    """Schedule contiguous physical failure bands over their complete Z range.

    Fine feedback deliberately places many probes inside one physical failure
    interval. Treating every probe as a separate repair target caused a single
    interval to consume a whole transaction and retained many redundant nodes.
    Adjacent failed probes are therefore merged into a band. The band's weight
    integrates its deficit, while its insertion cell spans the complete failed
    interval. Later placement stratifies nodes through that cell.
    """
    if failed.empty:
        return failed.copy()
    source = failed.sort_values("z_mm").copy()
    probes = np.unique(profile.z_mm.to_numpy(dtype=float))
    probe_steps = np.diff(probes)
    probe_steps = probe_steps[probe_steps > 1.0e-8]
    typical_step = float(np.median(probe_steps)) if len(probe_steps) else float(INTERNAL_DYNAMIC_PHYSICS_DZ_MM)
    merge_gap = max(float(INTERNAL_DYNAMIC_FAILURE_BAND_MERGE_MM), 1.51 * typical_step)
    source["_band_id"] = (source["z_mm"].diff().fillna(0.0) > merge_gap).cumsum()

    band_rows = []
    for _, group in source.groupby("_band_id", sort=True):
        group = group.sort_values("z_mm")

        worst_idx = group["lower_margin_Pa"].idxmin() if "lower_margin_Pa" in group else group.index[0]
        row = group.loc[worst_idx].copy()
        if "weight" in group:
            weights = np.clip(group["weight"].to_numpy(dtype=float), 0.0, None)
        elif "lower_deficit_Pa" in group:
            weights = np.clip(group["lower_deficit_Pa"].to_numpy(dtype=float), 0.0, None)
        elif "deficit_Pa" in group:
            weights = np.clip(group["deficit_Pa"].to_numpy(dtype=float), 0.0, None)
        else:
            weights = np.ones(len(group), dtype=float)
        if not np.any(weights):
            weights = np.ones(len(group), dtype=float)
        z_values = group["z_mm"].to_numpy(dtype=float)

        worst_failure_z = float(group.loc[worst_idx, "z_mm"])
        row["z_mm"] = float(np.average(z_values, weights=weights))
        row["worst_failure_z_mm"] = worst_failure_z
        row["weight"] = float(np.sum(weights))
        lo_probe, hi_probe = float(z_values.min()), float(z_values.max())
        lo_i = int(np.searchsorted(probes, lo_probe, side="left"))
        hi_i = int(np.searchsorted(probes, hi_probe, side="right")) - 1
        lo = (float(z_bounds[0]) if lo_i <= 0
              else 0.5 * (float(probes[lo_i - 1]) + lo_probe))
        hi = (float(z_bounds[1]) if hi_i >= len(probes) - 1
              else 0.5 * (hi_probe + float(probes[hi_i + 1])))
        row["failure_band_z_min_mm"] = lo_probe
        row["failure_band_z_max_mm"] = hi_probe
        row["insertion_cell_lo_mm"] = float(max(lo, z_bounds[0] + 1.0e-4))
        row["insertion_cell_hi_mm"] = float(min(hi, z_bounds[1] - 1.0e-4))

        worst_probe_i = int(np.argmin(np.abs(probes - worst_failure_z)))
        local_lo = (float(z_bounds[0]) if worst_probe_i <= 0
                    else 0.5 * (float(probes[worst_probe_i - 1]) + worst_failure_z))
        local_hi = (float(z_bounds[1]) if worst_probe_i >= len(probes) - 1
                    else 0.5 * (worst_failure_z + float(probes[worst_probe_i + 1])))
        row["local_insertion_cell_lo_mm"] = float(max(local_lo, z_bounds[0] + 1.0e-4))
        row["local_insertion_cell_hi_mm"] = float(min(local_hi, z_bounds[1] - 1.0e-4))
        row["failure_band_key_mm"] = round(2.0 * 0.5 * (lo + hi)) / 2.0
        row["failure_probe_count"] = int(len(group))

        span = max(typical_step, float(hi - lo))
        local_pitch = _local_internal_pitch(float(row["z_mm"]), z_bounds)
        vertical_step = max(typical_step, 0.5 * local_pitch)
        row["adaptive_add_cap"] = max(2, min(
            int(INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND),
            int(math.ceil(span / max(vertical_step, 1.0e-6))) + 1,
        ))
        band_rows.append(row)
    rows = pd.DataFrame(band_rows).reset_index(drop=True)

    rows["_visits"] = [visits.get(round(float(k), 6), 0)
                       for k in rows["worst_failure_z_mm"]]
    budget = max(2, int(budget))
    chosen = []
    for _, group in rows.groupby("_visits", sort=True):
        room = budget - len(chosen)
        if room <= 0:
            break
        if len(group) <= room:
            chosen.extend(group.index.tolist())
        else:

            remaining = group.copy()
            targets = np.linspace(float(group.z_mm.min()), float(group.z_mm.max()), room)
            for target in targets:
                idx = (remaining.z_mm - target).abs().idxmin()
                chosen.append(idx)
                remaining = remaining.drop(idx)
    selected = rows.loc[chosen].copy()
    targets = []
    cell_lows = []
    cell_highs = []
    for j, z in enumerate(selected.z_mm.to_numpy(dtype=float)):
        lo = float(selected.iloc[j].get("local_insertion_cell_lo_mm", selected.iloc[j]["insertion_cell_lo_mm"]))
        hi = float(selected.iloc[j].get("local_insertion_cell_hi_mm", selected.iloc[j]["insertion_cell_hi_mm"]))

        phase = 0.1 + 0.8 * ((0.25 + 0.6180339887498949 *
                              visits.get(round(float(selected.iloc[j]["worst_failure_z_mm"]), 6), 0)
                              + 0.5 * j) % 1.0)
        targets.append(float(np.clip(lo + phase * (hi - lo), z_bounds[0] + 1e-4, z_bounds[1] - 1e-4)))
        cell_lows.append(float(max(lo, z_bounds[0] + 1e-4)))
        cell_highs.append(float(min(hi, z_bounds[1] - 1e-4)))
    selected["insertion_z_mm"] = targets
    selected["insertion_cell_lo_mm"] = cell_lows
    selected["insertion_cell_hi_mm"] = cell_highs
    return selected.sort_values("insertion_z_mm").reset_index(drop=True)


def _smartfill_insertion_gain_factor(before: pd.DataFrame) -> float:
    """Scale insertion-only improvement floors once transport nears passage."""
    if before is None or before.empty:
        return 1.0
    if "transport_margin_Pa" in before.columns:
        margin = pd.to_numeric(
            before["transport_margin_Pa"], errors="coerce").to_numpy(float)
    else:
        margin = _capillary_margin_values(
            before, include_assembly_reserve=False)
    finite = margin[np.isfinite(margin)]
    if (len(finite) and float(np.min(finite))
            >= float(INTERNAL_DYNAMIC_SMARTFILL_HALF_GAIN_MARGIN_PA)):
        return float(INTERNAL_DYNAMIC_SMARTFILL_NEAR_PASS_GAIN_FACTOR)
    return 1.0


def _smartfill_meaningful_safety_gain(before: pd.DataFrame, after: pd.DataFrame,
                                      *, insertion_trial: bool = False) -> bool:
    """Reject near-zero one-slice deficit changes that otherwise permit endless fill.

    A feasible after-profile always passes. More general multi-slice trials retain
    the original ranking; this guard only applies to the observed one-failing-slice stall.
    """
    if not _profile_trial_improves(before, after):
        return False
    old_rank = _capillary_safety_rank(before)
    new_rank = _capillary_safety_rank(after)
    if new_rank[0] == 0:
        return True
    if new_rank[0] < old_rank[0]:
        return True

    gain_factor = (_smartfill_insertion_gain_factor(before)
                   if insertion_trial else 1.0)
    minimum = gain_factor * max(
        float(INTERNAL_DYNAMIC_SMARTFILL_MIN_DEFICIT_DROP_PA),
        float(INTERNAL_DYNAMIC_SMARTFILL_MIN_REL_DEFICIT_DROP) * old_rank[2],
    )
    return bool(old_rank[1] - new_rank[1] >= minimum)


def _smartfill_local_safe_gain(before: pd.DataFrame, after: pd.DataFrame,
                               *, insertion_trial: bool = False) -> bool:
    """Accept an incremental gain without weakening the original physical limit.

    Equal failure counts may hide a narrow local repair; accept it only when
    the old worst failed row becomes measurably better, total deficit does not
    grow, and no originally physically passing row is driven below requirement.
    """
    if _smartfill_meaningful_safety_gain(
            before, after, insertion_trial=insertion_trial):
        return True
    if before is None or after is None or len(before) != len(after):
        return False
    z_col = "z_mid_mm" if "z_mid_mm" in before else "z_mm"
    if z_col not in after or not np.allclose(
            before[z_col].to_numpy(float), after[z_col].to_numpy(float),
            rtol=0.0, atol=1e-9, equal_nan=False):
        return False
    mb, ma = _capillary_margin_values(before), _capillary_margin_values(after)
    if not np.isfinite(ma).all() or not np.isfinite(mb).all():
        return False
    rb, ra = _capillary_safety_rank(before), _capillary_safety_rank(after)
    if rb[0] == 0 or ra[0] != rb[0] or ra[1] > rb[1] + 1e-9:
        return False
    before_physical = before["P_cap_Pa"].to_numpy(float) - before["P_required_Pa"].to_numpy(float)
    after_physical = after["P_cap_Pa"].to_numpy(float) - after["P_required_Pa"].to_numpy(float)
    if np.any((before_physical >= -1e-9) & (after_physical < -1e-9)):
        return False
    failed = mb < 0.0
    if not failed.any():
        return False
    gain_factor = (_smartfill_insertion_gain_factor(before)
                   if insertion_trial else 1.0)
    min_gain = gain_factor * max(
        float(INTERNAL_DYNAMIC_SMARTFILL_MIN_DEFICIT_DROP_PA),
        float(INTERNAL_DYNAMIC_SMARTFILL_MIN_REL_DEFICIT_DROP)
        * float(-np.min(mb[failed])))

    worst_old = int(np.argmin(mb))
    return bool(ma[worst_old] - mb[worst_old] >= min_gain and
                np.min(ma) >= np.min(mb) - 1e-9)


def _smartfill_filter_transaction(atoms_before: pd.DataFrame, edges_before: List[Dict],
                                  atoms_trial: pd.DataFrame, edges_trial: List[Dict],
                                  keep_new_ids: Set[int]):
    """Build a prefix-preserving graph from a full trial; omit edges to rolled-back nodes.

    Newly reconciled original-original edges are intentionally NOT carried over:
    reconciliation is rerun after acceptance with the actual retained graph.
    """
    base_ids = set(atoms_before["atom_id"].astype(int))
    keep = base_ids | {int(i) for i in keep_new_ids}
    extra = atoms_trial.iloc[len(atoms_before):]
    extra = extra[extra["atom_id"].isin(keep_new_ids)]
    atoms = pd.concat([atoms_before, extra], ignore_index=True)
    appended = [e for e in edges_trial[len(edges_before):]
                if int(e.get("source", -1)) in keep and int(e.get("target", -1)) in keep
                and (int(e.get("source", -1)) not in base_ids or
                     int(e.get("target", -1)) not in base_ids)]
    return atoms, list(edges_before) + appended


def _smartfill_subset_supported(atoms: pd.DataFrame, edges: List[Dict],
                                new_ids: Set[int], z_bounds) -> bool:
    """Refuse any partial graph that strands a trial node or breaks a short bridge."""
    status = _internal_support_state(atoms, edges, z_bounds=z_bounds)
    if not new_ids.issubset(status["supported"] & status["root_connected"]):
        return False
    if status["bridge_bad"]:
        return False
    for pair, cls in status["classes"].items():
        if cls == "rejected" and (set(pair) & new_ids):
            return False
    return True


def _profile_smooth_deficits(df: pd.DataFrame) -> pd.DataFrame:
    """Return slices that form a capillary-pressure dip relative to neighbours."""
    if df is None or df.empty or "P_cap_Pa" not in df.columns:
        return pd.DataFrame()
    work = df.sort_values("z_mid_mm" if "z_mid_mm" in df.columns else "z_mm").copy()
    cap = np.maximum(work["P_cap_Pa"].to_numpy(dtype=float), 1.0e-9)
    target = cap.copy()
    limit = max(float(INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO), 1.0)
    if len(cap) >= 2:
        target[1:] = np.maximum(target[1:], cap[:-1] / limit)
    if len(cap) >= 3:
        target[1:-1] = np.maximum(target[1:-1], np.minimum(cap[:-2], cap[2:]) / limit)
    work["smooth_target_Pa"] = target
    work["smooth_deficit_Pa"] = np.clip(target - cap, 0.0, None)
    work["smooth_rel_def"] = work["smooth_deficit_Pa"] / np.maximum(target, 1.0e-9)
    return work[work["smooth_deficit_Pa"] > 1.0e-6].copy()


def _low_porosity_excess_rows(
        df: pd.DataFrame,
        threshold: Optional[float] = None) -> pd.DataFrame:
    """Slices eligible for deletion: low porosity with surplus capillary pressure."""
    required = {"porosity", "P_cap_Pa", "P_required_Pa"}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    work = df.copy()
    limit = float(
        INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_THRESHOLD
        if threshold is None else threshold
    )
    work["capillary_excess_Pa"] = work["P_cap_Pa"] - work["P_required_Pa"]
    work["porosity_deficit"] = np.clip(limit - work["porosity"], 0.0, None)
    issues = work[
        (work["porosity"] < limit)
        & (work["capillary_excess_Pa"] > 0.0)
        ].copy()
    return issues.sort_values(
        ["porosity", "capillary_excess_Pa"], ascending=[True, False]
    )


def _porosity_floor_at_z(z_mm: float, z_bounds: Tuple[float, float]) -> float:
    zmin, zmax = map(float, z_bounds)
    total_h = max(zmax - zmin, 1.0e-9)
    t = float(np.clip((float(z_mm) - zmin) / total_h, 0.0, 1.0))
    base = max(float(INTERNAL_DYNAMIC_MIN_LIQUID_FRAC), float(INTERNAL_DYNAMIC_OVERFILL_POROSITY_FLOOR))
    top_start = float(np.clip(INTERNAL_DYNAMIC_TOP_GUARD_START_FRAC, 0.0, 0.99))
    if t <= top_start:
        return base
    ramp = (t - top_start) / max(1.0 - top_start, 1.0e-9)
    return float(base * (1.0 - ramp) + float(INTERNAL_DYNAMIC_TOP_POROSITY_FLOOR) * ramp)


def _plot_gradient_line(ax, x_values, y_values, linewidth: float = 3.0, alpha: float = 1.0):
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[valid], y_arr[valid]
    if len(x_arr) < 2:
        if len(x_arr) == 1:
            ax.scatter(x_arr, y_arr, s=18, color=FIGURE_BLUE, zorder=3)
        return None
    points = np.array([x_arr, y_arr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(vmin=float(np.nanmin(y_arr)), vmax=float(np.nanmax(y_arr)))
    lc = LineCollection(segments, cmap=FIGURE_CMAP, norm=norm, linewidth=linewidth, alpha=alpha)
    lc.set_array(y_arr[:-1])
    ax.add_collection(lc)
    ax.set_xlim(float(np.nanmin(x_arr)), float(np.nanmax(x_arr)))
    pad = max(1.0e-9, 0.05 * (float(np.nanmax(y_arr)) - float(np.nanmin(y_arr))))
    ax.set_ylim(float(np.nanmin(y_arr)) - pad, float(np.nanmax(y_arr)) + pad)
    return lc


def _style_internal_axis(ax, show_top_right: bool = True):
    ax.spines['top'].set_visible(bool(show_top_right))
    ax.spines['right'].set_visible(bool(show_top_right))
    ax.tick_params(direction='in', top=bool(show_top_right), right=bool(show_top_right), length=5, pad=6)
    ax.grid(False)


def _profile_overfill_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.sort_values("z_mid_mm" if "z_mid_mm" in df.columns else "z_mm").copy()
    z_col = "z_mid_mm" if "z_mid_mm" in work.columns else "z_mm"
    z_bounds = (float(work[z_col].min()), float(work[z_col].max()))
    work["porosity_floor"] = work[z_col].map(lambda z: _porosity_floor_at_z(float(z), z_bounds))
    work["low_porosity"] = work["porosity"] < work["porosity_floor"]
    work["low_liquid_area"] = work["A_liq_mm2"] < INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2
    cap = np.maximum(work["P_cap_Pa"].to_numpy(dtype=float), 1e-9)
    required = np.maximum(work["P_required_Pa"].to_numpy(dtype=float), 1.0e-9)
    work["capillary_ratio"] = cap / required
    band_mask = work[z_col] >= float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM)
    work["capillary_surplus"] = band_mask & (work["capillary_ratio"] > _capillary_trim_ratio())
    spike = np.zeros(len(work), dtype=bool)
    if len(cap) >= 3:
        for i in range(1, len(cap) - 1):
            neighbor = max(min(cap[i - 1], cap[i + 1]), 1e-9)
            spike[i] = cap[i] / neighbor > INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO
    work["cap_spike"] = spike
    work.loc[~band_mask, "cap_spike"] = False
    issues = work[
        work["low_porosity"] | work["low_liquid_area"]
        | work["cap_spike"] | work["capillary_surplus"]
        ].copy()
    if issues.empty:
        return issues
    issues["issue_score"] = (
            np.clip(issues["porosity_floor"] - issues["porosity"], 0.0, None) * 1.0e5
            + np.clip(INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2 - issues["A_liq_mm2"], 0.0, None) * 1.0e3
            + issues["cap_spike"].astype(float) * issues["P_cap_Pa"].clip(lower=0.0)
            + np.clip(
        issues["P_cap_Pa"] - _capillary_trim_ratio() * issues["P_required_Pa"],
        0.0, None,
    ) * 1.0e3
    )
    return issues.sort_values(["issue_score", z_col], ascending=[False, False])


def _allocate_targets_to_components(domain, total_target: int,
                                    z_mid: Optional[float] = None,
                                    z_bounds: Optional[Tuple[float, float]] = None) -> List[Tuple[Polygon, int]]:
    all_polys = sorted(_as_polygons(domain), key=lambda p: p.area, reverse=True)
    polys = [p for p in all_polys if float(p.area) >= INTERNAL_DYNAMIC_MIN_COMPONENT_AREA_MM2]
    if not polys and all_polys:
        polys = [all_polys[0]]
    if not polys:
        return []
    total_target = max(0, int(total_target))
    min_each = max(0, int(INTERNAL_DYNAMIC_MIN_NODES_PER_COMPONENT))
    if total_target <= 0:
        return [(p, 0) for p in polys]
    areas = np.asarray([max(float(p.area), 0.0) for p in polys], dtype=float)
    if float(areas.sum()) <= 1e-12:
        return [(p, min_each) for p in polys]
    base = np.zeros(len(polys), dtype=int)
    if min_each > 0:
        base[:] = min_each
    if int(base.sum()) > total_target:
        order = np.argsort(-areas)
        base[:] = 0
        for idx in order[:total_target]:
            base[int(idx)] = 1
    remaining = max(0, total_target - int(base.sum()))
    if remaining > 0:
        raw = areas / float(areas.sum()) * float(remaining)
        add = np.floor(raw).astype(int)
        frac_order = np.argsort(-(raw - add))
        for idx in frac_order:
            if int(add.sum()) >= remaining:
                break
            add[int(idx)] += 1
        base += add
    if z_mid is not None and z_bounds is not None:
        pitch = max(_local_internal_pitch(float(z_mid), z_bounds), 1e-6)
        caps = np.asarray([
            max(min_each, int(math.ceil(
                float(poly.area) / max(pitch * pitch * INTERNAL_DYNAMIC_COMPONENT_CAPACITY_FACTOR, 1e-9))))
            for poly in polys
        ], dtype=int)
        base = np.minimum(base, caps)
    return [(poly, int(n)) for poly, n in zip(polys, base)]


def build_internal_structure_dynamic(mesh, start_id: int,
                                     atoms_surf: Optional[pd.DataFrame] = None,
                                     edges_surf: Optional[pd.DataFrame] = None,
                                     output_dir: Optional[Path] = None):
    log("Generating internal structure with STL-domain physics feedback...")
    output_dir = output_dir or OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(INTERNAL_DYNAMIC_SEED)
    # Independent stream: switching the position control cannot change the
    # initial-skeleton sampling sequence or the ordinary proposal RNG.
    position_rng = np.random.default_rng(int(INTERNAL_DYNAMIC_SEED) + 104729)
    random.seed(INTERNAL_DYNAMIC_SEED)
    zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    z_bounds = (zmin, zmax)
    total_h = max(zmax - zmin, 1e-9)
    n_layers = max(1, int(INTERNAL_DYNAMIC_LAYERS))
    layer_z = np.linspace(zmin, zmax, n_layers + 1)
    domain_cache: Dict[float, object] = {}

    all_atoms = atoms_surf.copy().reset_index(
        drop=True) if atoms_surf is not None and not atoms_surf.empty else pd.DataFrame(
        columns=["atom_id", "x", "y", "z", "face_name"])
    all_edges = _edge_records(edges_surf)
    next_id = int(start_id)
    internal_ids: Set[int] = set()

    supported_internal_ids: Set[int] = set()
    layer_nodes: List[List[int]] = [[] for _ in range(n_layers)]
    process_log: List[Dict] = []
    internal_log: List[Dict] = []
    global_step = 0
    smartfill_rejections: List[Dict] = []
    smartfill_placement_log: List[Dict] = []
    initial_rejection_counts: Dict[str, int] = {}
    initial_rejection_details: List[Dict] = []
    initial_peer_relaxed_layers: Set[int] = set()

    layer_domains, layer_areas = [], []
    for k in range(n_layers):
        z_mid = float(0.5 * (layer_z[k] + layer_z[k + 1]))
        domain = _domain_at_z_cached(mesh, z_mid, domain_cache)
        layer_domains.append(domain)
        layer_areas.append(float(domain.area) if not domain.is_empty else 0.0)
    if max(layer_areas) <= 1e-9:
        raise ValueError("no valid STL design-domain slices were found")

    loaded_checkpoint_stage: Optional[str] = None
    checkpoint_to_load = None
    has_surface_prefix = False
    if PRE_SMARTFILL_CHECKPOINT_LOAD is not None:
        loaded_checkpoint_stage = "pre-smartfill"
        checkpoint_to_load = PRE_SMARTFILL_CHECKPOINT_LOAD
    elif INITIAL_SKELETON_CHECKPOINT_LOAD is not None:
        loaded_checkpoint_stage = "initial-skeleton"
        checkpoint_to_load = INITIAL_SKELETON_CHECKPOINT_LOAD
    if checkpoint_to_load is not None:
        surface_atoms_prefix = all_atoms.copy().reset_index(drop=True)
        surface_edges_prefix = [dict(edge) for edge in all_edges]
        has_surface_prefix = not surface_atoms_prefix.empty
        restored = _load_internal_checkpoint(
            checkpoint_to_load, loaded_checkpoint_stage, mesh, int(start_id),
            allow_internal_id_remap=has_surface_prefix)
        restored_atoms = restored["all_atoms"].copy().reset_index(drop=True)
        restored_edges = [dict(edge) for edge in restored["all_edges"]]
        if has_surface_prefix:
            all_atoms = pd.concat(
                [surface_atoms_prefix, restored_atoms], ignore_index=True, sort=False)
            all_edges = surface_edges_prefix + restored_edges
            log(f"  Combined checkpoint interior with surface prefix: "
                f"surface_atoms={len(surface_atoms_prefix)}, "
                f"surface_edges={len(surface_edges_prefix)}.")
        else:
            all_atoms = restored_atoms
            all_edges = restored_edges
        internal_ids = set(map(int, restored.get("internal_ids", [])))
        supported_internal_ids = set(map(int, restored.get("supported_internal_ids", [])))
        layer_nodes = [list(map(int, nodes)) for nodes in restored.get("layer_nodes", [])]
        if len(layer_nodes) != n_layers:
            raise ValueError(
                f"checkpoint layer count is {len(layer_nodes)}, expected {n_layers}")
        next_id = int(restored["next_id"])
        global_step = int(restored.get("global_step", 0))
        process_log = list(restored.get("process_log", []))
        internal_log = list(restored.get("internal_log", []))
        initial_rejection_counts = dict(restored.get("initial_rejection_counts", {}))
        initial_rejection_details = list(restored.get("initial_rejection_details", []))

    def target_nodes_for_layer(k: int) -> int:
        z_mid = float(0.5 * (layer_z[k] + layer_z[k + 1]))
        t = 0.0 if n_layers <= 1 else float(k) / float(n_layers - 1)
        grad_target = INTERNAL_DYNAMIC_BOTTOM_NODE_TARGET * (1.0 - t) + INTERNAL_DYNAMIC_TOP_NODE_TARGET * t
        area_scale = layer_areas[k] / max(max(layer_areas), 1e-9)
        grad_target *= max(INTERNAL_DYNAMIC_AREA_TARGET_MIN_SCALE,
                           float(area_scale) ** max(0.0, INTERNAL_DYNAMIC_AREA_TARGET_EXPONENT))
        area_target = (layer_areas[k] / max(_local_internal_pitch(z_mid, z_bounds) ** 2, 1e-9))
        area_target *= INTERNAL_DYNAMIC_DENSITY_FACTOR
        target = max(float(grad_target), float(area_target), float(INTERNAL_DYNAMIC_MIN_NODES_PER_LAYER))
        target *= float(INTERNAL_DYNAMIC_INITIAL_DENSITY_MULTIPLIER)
        target = max(round(target), INTERNAL_DYNAMIC_MIN_NODES_PER_LAYER)
        if INTERNAL_DYNAMIC_MAX_NODES_PER_LAYER and INTERNAL_DYNAMIC_MAX_NODES_PER_LAYER > 0:
            target = min(target, INTERNAL_DYNAMIC_MAX_NODES_PER_LAYER)
        return int(target)

    def stratified_z_values(z0: float, z1: float, count: int) -> np.ndarray:
        """Spread a quota through an overlapping height band to blur layer seams."""
        count = max(1, int(count))
        layer_h = max(float(z1 - z0), 1.0e-9)
        overlap = 0.5 * float(np.clip(INTERNAL_DYNAMIC_LAYER_Z_JITTER_FRAC, 0.0, 1.0)) * layer_h
        sample_z0 = max(float(zmin), float(z0) - overlap)
        sample_z1 = min(float(zmax), float(z1) + overlap)
        phase = (np.arange(count, dtype=float) + rng.uniform(0.15, 0.85, count)) / count
        values = sample_z0 + phase * max(sample_z1 - sample_z0, 1.0e-9)

        return np.sort(values)

    def exact_lower_support_anchors(x: float, y: float, z: float,
                                    allow_inbound_reserve: bool = False):
        """Return legal supported lower endpoints without moving the target."""
        if not internal_ids:
            return []
        internal_df = all_atoms[all_atoms['atom_id'].isin(internal_ids)].copy()
        degree = {int(i): 0 for i in internal_ids}
        for edge in all_edges:
            if str(edge.get('bond_type', '')) != 'internal':
                continue
            for node_id in (int(edge.get('source', -1)), int(edge.get('target', -1))):
                if node_id in degree:
                    degree[node_id] += 1
        routine_limit = int(INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW)
        endpoint_limit = routine_limit + (
            max(0, int(INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE),
                int(INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE))
            if allow_inbound_reserve else 0)
        layer_h = total_h / max(n_layers, 1)
        pitch = _local_internal_pitch(z, z_bounds)
        max_len = max(layer_h, pitch) * INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR
        max_dz = INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW * layer_h + 1.0e-6
        out = []
        for row in internal_df.itertuples(index=False):
            aid, za = int(row.atom_id), float(row.z)
            if aid not in supported_internal_ids or za >= float(z) - 1.0e-9:
                continue
            if endpoint_limit > 0 and degree.get(aid, 0) >= endpoint_limit:
                continue
            dz = float(z) - za
            dxy = math.hypot(float(x) - float(row.x), float(y) - float(row.y))
            span = math.hypot(dxy, dz)
            if (dz > max_dz or span > max_len + 1.0e-9
                    or dxy > pitch * INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC + 1.0e-9):
                continue
            if _internal_edge_print_class(
                    (float(x), float(y), float(z)),
                    (float(row.x), float(row.y), za)) != 'inclined':
                continue
            if not _edge_centerline_stays_in_domain(
                    mesh, domain_cache, (float(x), float(y), float(z)),
                    (float(row.x), float(row.y), za), samples=3):
                continue
            out.append((span, degree.get(aid, 0), aid))
        return sorted(out)

    def best_connectable_initial_candidate(z0: float, z1: float, layer_idx: int,
                                           z_hint: float, domain_hint,
                                           n_candidates: int,
                                           min_spacing_factor: float):
        """Construct support discs first, then sample only their legal union."""
        internal_df = all_atoms[all_atoms['atom_id'].isin(internal_ids)].copy()
        xyz = (internal_df[['x', 'y', 'z']].to_numpy(float)
               if not internal_df.empty else np.empty((0, 3), float))
        tree = cKDTree(xyz) if len(xyz) else None
        z_try = float(np.clip(z_hint, zmin + 1.0e-6, zmax - 1.0e-6))
        domain_z = _domain_at_z_cached(mesh, z_try, domain_cache)
        active = domain_z
        if domain_hint is not None and not domain_hint.is_empty:
            active = domain_hint.intersection(domain_z)
        if active.is_empty:
            return None, 'support_region_no_component_overlap'
        degree = {int(i): 0 for i in internal_ids}
        for edge in all_edges:
            if str(edge.get('bond_type', '')) != 'internal':
                continue
            for node_id in (int(edge.get('source', -1)), int(edge.get('target', -1))):
                if node_id in degree:
                    degree[node_id] += 1
        layer_h = total_h / max(n_layers, 1)
        pitch = _local_internal_pitch(z_try, z_bounds)
        max_len = max(layer_h, pitch) * INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR
        max_dz = INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW * layer_h + 1.0e-6
        lower = [r for r in internal_df.itertuples(index=False)
                 if int(r.atom_id) in supported_internal_ids and float(r.z) < z_try - 1.0e-9]
        if not lower:
            return None, 'support_region_no_supported_lower_node'
        degree_open = [r for r in lower if degree.get(int(r.atom_id), 0) < int(INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW)]
        if not degree_open:
            return None, 'support_region_lower_degree_full'
        within_height = [r for r in degree_open if z_try - float(r.z) <= max_dz]
        if not within_height:
            return None, 'support_region_beyond_two_layers'
        regions = []
        anchor_specs = []
        positive_reach = 0
        angle_tan = math.tan(math.radians(max(
            float(INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG), 1.0e-6)))
        for row in within_height:
            dz = z_try - float(row.z)
            length_reach = math.sqrt(max(0.0, max_len ** 2 - dz ** 2))
            reach = min(dz / angle_tan, length_reach,
                        pitch * INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC)
            if reach <= 1.0e-6:
                continue
            positive_reach += 1
            anchor_specs.append((
                int(row.atom_id), float(row.x), float(row.y), float(row.z),
                float(0.985 * reach), degree.get(int(row.atom_id), 0)))

        if not anchor_specs:
            return None, 'support_region_angle_or_length_blocked'

        anchor_arr = np.asarray([[a[1], a[2], a[3], a[4]] for a in anchor_specs], float)
        fast_best = None
        fast_best_score = -1.0
        fast_trials = min(max(1, int(n_candidates)), max(
            10, int(math.ceil(3.0 * math.sqrt(max(float(active.area) / (pitch * pitch), 1.0))))))
        for _ in range(fast_trials):
            xy = _sample_xy_in_domain(active, rng)
            if xy is None:
                continue
            if tree is not None:
                score = float(tree.query([xy[0], xy[1], z_try], k=1)[0])
                if score < max(float(min_spacing_factor) * pitch, 1.0e-6):
                    continue
            else:
                score = float('inf')
            dxy = np.hypot(anchor_arr[:, 0] - float(xy[0]),
                           anchor_arr[:, 1] - float(xy[1]))
            feasible = np.flatnonzero(dxy <= anchor_arr[:, 3] + 1.0e-9)
            if not len(feasible):
                continue
            order = sorted(feasible.tolist(), key=lambda j: (
                math.hypot(float(dxy[j]), z_try - anchor_arr[j, 2]),
                anchor_specs[j][5], anchor_specs[j][0]))
            for j in order[:6]:
                anchor_id, ax, ay, az, _reach, _degree = anchor_specs[j]
                if not _edge_centerline_stays_in_domain(
                        mesh, domain_cache, (float(xy[0]), float(xy[1]), z_try),
                        (ax, ay, az), samples=5):
                    continue
                if score > fast_best_score:
                    fast_best_score = score
                    fast_best = (float(xy[0]), float(xy[1]), z_try, int(anchor_id))
                break
        if fast_best is not None:
            return fast_best, 'candidate_ready_fast'

        for anchor_id, ax, ay, az, reach, _degree in anchor_specs:
            anchor_point = Point(ax, ay)
            if float(active.distance(anchor_point)) >= reach:
                continue
            region = active.intersection(anchor_point.buffer(reach, resolution=8))
            if region.is_empty or float(region.area) <= 1.0e-9:
                continue
            regions.append((int(anchor_id), (ax, ay, az), region, float(region.area)))
        if not regions:
            return None, ('support_region_angle_or_length_blocked'
                          if positive_reach == 0
                          else 'support_region_disconnected_component')

        best = None
        best_score = -1.0
        rejected_spacing = rejected_visibility = 0
        weights = np.asarray([item[3] for item in regions], float)
        weights /= max(float(weights.sum()), 1.0e-12)

        effective_candidates = min(
            max(1, int(n_candidates)),
            max(12, int(math.ceil(2.0 * math.sqrt(len(regions)) + 6.0))))
        for _ in range(effective_candidates):
            region_idx = int(rng.choice(len(regions), p=weights))
            anchor_id, anchor_xyz, region, _area = regions[region_idx]
            xy = _sample_xy_in_domain(region, rng)
            if xy is None:
                continue
            if tree is not None:
                score = float(tree.query([xy[0], xy[1], z_try], k=1)[0])
                if score < max(float(min_spacing_factor) * pitch, 1.0e-6):
                    rejected_spacing += 1
                    continue
            else:
                score = float('inf')
            if not _edge_centerline_stays_in_domain(
                    mesh, domain_cache, (float(xy[0]), float(xy[1]), z_try),
                    anchor_xyz, samples=5):
                rejected_visibility += 1
                continue
            if score > best_score:
                best_score = score
                best = (float(xy[0]), float(xy[1]), z_try, anchor_id)
        if best is not None:
            return best, 'candidate_ready'
        if rejected_visibility and rejected_visibility >= rejected_spacing:
            return None, 'support_region_edge_outside_stl'
        if rejected_spacing:
            return None, 'support_region_min_spacing_exhausted'
        return None, 'support_region_sampling_exhausted'

    def add_node_to_layer(z0: float, z1: float, layer_idx: int, force_insert: bool = False,
                          label: str = "internal", specific_z: Optional[float] = None,
                          specific_xy: Optional[Tuple[float, float]] = None,
                          allow_unanchored: bool = False,
                          domain_override=None, required_support_id: Optional[int] = None,
                          allow_layer_peer_without_lower: bool = False):
        def reject(reason):
            if str(label).startswith("L"):
                initial_rejection_counts[str(reason)] = (
                        initial_rejection_counts.get(str(reason), 0) + 1)

                rx = float(specific_xy[0]) if specific_xy is not None else None
                ry = float(specific_xy[1]) if specific_xy is not None else None
                rz = float(specific_z) if specific_z is not None else None
                try:
                    if rx is None:
                        rx = float(x)
                    if ry is None:
                        ry = float(y)
                    if rz is None:
                        rz = float(z)
                except (NameError, UnboundLocalError, TypeError, ValueError):
                    pass
                initial_rejection_details.append({
                    "layer_idx": int(layer_idx), "label": str(label),
                    "reason": str(reason), "x_mm": rx, "y_mm": ry, "z_mm": rz,
                    "force_insert": bool(force_insert),
                })
            if str(label).startswith('SMART_'):
                smartfill_rejections.append({
                    'label': str(label), 'reason': str(reason),
                    'z_mm': float(specific_z) if specific_z is not None else None,
                    'x_mm': float(specific_xy[0]) if specific_xy is not None else None,
                    'y_mm': float(specific_xy[1]) if specific_xy is not None else None,
                    'requested_lower_support_id': required_support_id,
                })
            return False, None

        nonlocal next_id, all_atoms, all_edges, global_step

        if (INTERNAL_MATCHED_RANDOM_POSITION and INTERNAL_MATCHED_RANDOM_NODE_BUDGET > 0
                and len(all_atoms) >= INTERNAL_MATCHED_RANDOM_NODE_BUDGET):
            return reject('matched_node_budget_exhausted')

        if specific_xy is not None:
            z = float(specific_z if specific_z is not None else rng.uniform(z0, z1))
            x, y = float(specific_xy[0]), float(specific_xy[1])
        else:
            candidate_z0, candidate_z1 = float(z0), float(z1)
            candidate_hint = 0.5 * (float(z0) + float(z1))
            candidate_domain = domain_override if domain_override is not None else (
                layer_domains[layer_idx] if 0 <= layer_idx < len(layer_domains) else None
            )
            if specific_z is not None:
                candidate_hint = float(np.clip(specific_z, z0 + 1.0e-6, z1 - 1.0e-6))
                eps = max(1.0e-6, 1.0e-4 * max(float(z1 - z0), 1.0e-9))
                candidate_z0 = max(float(z0), candidate_hint - eps)
                candidate_z1 = min(float(z1), candidate_hint + eps)
                candidate_domain = _domain_at_z_cached(mesh, candidate_hint, domain_cache)
            candidate = _best_candidate_in_stl_domain(
                mesh, domain_cache, candidate_z0, candidate_z1, rng,
                all_atoms[all_atoms["atom_id"].isin(internal_ids)].copy() if internal_ids else pd.DataFrame(),
                n_candidates=INTERNAL_DYNAMIC_CANDIDATES * (2 if force_insert else 1),
                z_hint=candidate_hint,
                domain_hint=candidate_domain,
            )
            if candidate is None:
                return reject("no_sampling_candidate")
            x, y, z = candidate
            if specific_z is not None:
                z = float(specific_z)

        z = float(np.clip(z, zmin + 1e-6, zmax - 1e-6))
        domain = _domain_at_z_cached(mesh, z, domain_cache)
        if domain.is_empty:
            return reject("outside_design_domain")
        active_domain = domain
        if domain_override is not None and not domain_override.is_empty:
            active_domain = domain_override.intersection(domain)
            if active_domain.is_empty:
                active_domain = domain
        if not active_domain.covers(Point(x, y)):
            xy = _sample_xy_in_domain(active_domain, rng)
            if xy is None:
                return reject("outside_design_domain")
            x, y = xy

        local_pitch = _local_internal_pitch(z, z_bounds)
        min_factor = INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR if force_insert else INTERNAL_DYNAMIC_MIN_DIST_FACTOR
        if internal_ids:
            coords = all_atoms[all_atoms["atom_id"].isin(internal_ids)][["x", "y", "z"]].to_numpy(dtype=float)
            d_near, _ = cKDTree(coords).query([x, y, z], k=1)
            if float(d_near) < local_pitch * min_factor:
                return reject("min_distance")

        internal_df = all_atoms[all_atoms["atom_id"].isin(internal_ids)].copy()

        def internal_degree_map():
            deg = {int(i): 0 for i in internal_ids}
            for edge in all_edges:
                if str(edge.get("bond_type", "")) != "internal":
                    continue
                u = int(edge.get("source", -1))
                v = int(edge.get("target", -1))
                if u in deg:
                    deg[u] += 1
                if v in deg:
                    deg[v] += 1
            return deg

        degree_now = internal_degree_map()
        top_smart_relax = (
                str(label).startswith('SMART_')
                and float(z) >= float(zmax) - float(INTERNAL_TOP_RELAX_ACTIVE_HEIGHT_MM) - 1e-9
        )
        active_degree_limit = (
            max(int(INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW),
                int(INTERNAL_DYNAMIC_TOP_MAX_EDGES_PER_NODE))
            if top_smart_relax else int(INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW)
        )
        is_later_new_node = str(label).startswith('SMART_')

        existing_target_degree_limit = active_degree_limit + (
            max(0,
                int(INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE),
                int(INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE))
            if is_later_new_node else 0
        )
        support_anchor_degree_limit = existing_target_degree_limit
        raw_neighbors = []
        if not internal_df.empty:
            coords = internal_df[["x", "y", "z"]].to_numpy(dtype=float)
            ids = internal_df["atom_id"].to_numpy(dtype=int)
            z_vals = internal_df["z"].to_numpy(dtype=float)
            kq = min(INTERNAL_DYNAMIC_NEIGHBOR_K, len(coords))
            layer_h = max(float(z1 - z0), 1e-9)
            max_edge_length = max(layer_h, local_pitch) * INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR
            search_radius = min(max(layer_h, local_pitch) * INTERNAL_DYNAMIC_SEARCH_FACTOR, max_edge_length)
            dists, idxs = cKDTree(coords).query([x, y, z], k=kq, distance_upper_bound=search_radius)
            query_pairs = list(zip(np.atleast_1d(dists), np.atleast_1d(idxs)))

            if required_support_id is not None:
                match = np.flatnonzero(ids == int(required_support_id))
                if len(match):
                    j = int(match[0])
                    query_pairs.append((float(np.linalg.norm(coords[j] - np.array([x, y, z]))), j))
            visited_targets = set()
            for d_val, idx in query_pairs:
                if not np.isfinite(d_val):
                    continue
                idx = int(idx)
                if 0 <= idx < len(ids):
                    target_id = int(ids[idx])
                    if target_id in visited_targets:
                        continue
                    visited_targets.add(target_id)
                    target_degree_limit = existing_target_degree_limit
                    if target_degree_limit > 0 and degree_now.get(target_id, 0) >= target_degree_limit:
                        continue
                    target_z = float(z_vals[idx])
                    d3 = float(d_val)
                    if d3 > max_edge_length:
                        continue
                    tx, ty, _tz = coords[idx]
                    dz_abs = abs(target_z - z)
                    dxy = math.hypot(float(tx) - x, float(ty) - y)
                    edge_class = _internal_edge_print_class((x, y, z), (float(tx), float(ty), target_z))
                    if (edge_class == 'rejected' and top_smart_relax):
                        edge_class = _top_relaxed_internal_edge_class(
                            (x, y, z), (float(tx), float(ty), target_z),
                            z_bounds=z_bounds, smartfill_node_z=z,
                        )
                    if edge_class == 'rejected':
                        continue
                    if dz_abs > INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW * layer_h + 1e-6:
                        continue
                    if dxy > local_pitch * INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC:
                        continue
                    if edge_class == 'short_bridge':

                        domain_mid = _domain_at_z_cached(mesh, 0.5 * (z + target_z), domain_cache)
                        if (_point_component_index(domain_mid, x, y) is None
                                or _point_component_index(domain_mid, x, y) !=
                                _point_component_index(domain_mid, float(tx), float(ty))):
                            continue
                    if not _edge_centerline_stays_in_domain(
                            mesh, domain_cache, (x, y, z), (float(tx), float(ty), target_z), samples=3,
                    ):
                        continue
                    raw_neighbors.append((d3, target_id, target_z, edge_class))

        new_root = z <= zmin + total_h / max(n_layers, 1) + 1.0e-6

        edges_to_create = []

        def select_spread(bucket, count):
            if count <= 0 or not bucket:
                return []
            bucket.sort(key=lambda q: q[0])
            if count == 1 or len(bucket) == 1:
                return [bucket[0]]
            selected = [bucket[0]]
            if len(bucket) > 1:
                selected.append(bucket[-1])
            idx = 1
            while len(selected) < count and idx < len(bucket) - 1:
                selected.append(bucket[idx])
                idx += 1
            return selected[:count]

        if force_insert:
            flat_same, lower, upper = [], [], []
            min_intra_dz = min(
                max(float(INTERNAL_DYNAMIC_LAYER_INTRA_MIN_DZ_MM), 0.0),
                0.35 * max(float(z1 - z0), 1.0e-9),
            )
            for item in raw_neighbors:
                d_val, target_id, target_z, edge_class = item
                dz_rel = target_z - z
                dz_abs = abs(dz_rel)

                if dz_abs < min_intra_dz:
                    flat_same.append(item)
                elif dz_rel < 0.0:
                    lower.append(item)
                else:
                    upper.append(item)

            z_frac_here = (float(z) - float(zmin)) / max(float(total_h), 1.0e-9)
            top_smart = (
                    str(label).startswith("SMART_")
                    and z_frac_here >= float(INTERNAL_DYNAMIC_TOP_GUARD_START_FRAC)
            )
            same_quota = int(INTERNAL_DYNAMIC_FORCE_SAME_LAYER_EDGES)
            if top_smart:
                same_quota = max(same_quota, int(INTERNAL_DYNAMIC_TOP_SAME_LAYER_EDGES))

            supported_lower = [
                item for item in lower
                if item[3] == 'inclined' and int(item[1]) in supported_internal_ids
            ]
            if required_support_id is not None:
                preferred = next((item for item in supported_lower
                                  if int(item[1]) == int(required_support_id)), None)
                if preferred is None:
                    return reject('no_qualified_edge_selected')

                other_lower = [item for item in supported_lower
                               if int(item[1]) != int(required_support_id)]
                selected_lower = [preferred] + select_spread(
                    other_lower, max(0, int(INTERNAL_DYNAMIC_FORCE_LOWER_EDGES) - 1))
            else:
                selected_lower = select_spread(supported_lower, INTERNAL_DYNAMIC_FORCE_LOWER_EDGES)
            selected_lower_ids = {int(item[1]) for item in selected_lower}
            if len(selected_lower) < int(INTERNAL_DYNAMIC_FORCE_LOWER_EDGES):
                remaining_lower = [item for item in lower if int(item[1]) not in selected_lower_ids]
                selected_lower.extend(select_spread(
                    remaining_lower, int(INTERNAL_DYNAMIC_FORCE_LOWER_EDGES) - len(selected_lower)
                ))

            edges_to_create.extend(
                (item, "support_repair" if required_support_id is not None
                                           and int(item[1]) == int(required_support_id) else "layer_inter")
                for item in selected_lower
            )
            selected_same = select_spread(flat_same, same_quota)
            edges_to_create.extend((item, "layer_intra") for item in selected_same)
            edges_to_create.extend(
                (item, "layer_inter") for item in select_spread(upper, INTERNAL_DYNAMIC_FORCE_UPPER_EDGES))
        else:
            flat_same, lower, upper = [], [], []
            min_intra_dz = min(
                max(float(INTERNAL_DYNAMIC_LAYER_INTRA_MIN_DZ_MM), 0.0),
                0.35 * max(float(z1 - z0), 1.0e-9),
            )
            for item in raw_neighbors:
                d_val, target_id, target_z, edge_class = item
                dz_rel = target_z - z
                dz_abs = abs(dz_rel)
                if dz_abs < min_intra_dz:
                    flat_same.append(item)
                elif dz_rel < 0.0:
                    lower.append(item)
                else:
                    upper.append(item)
            selected_same = select_spread(flat_same, INTERNAL_DYNAMIC_NORMAL_SAME_EDGES)
            edges_to_create.extend((item, "layer_intra") for item in selected_same)
            edges_to_create.extend(
                (item, "layer_inter") for item in select_spread(lower, INTERNAL_DYNAMIC_NORMAL_LOWER_EDGES))
            edges_to_create.extend(
                (item, "layer_inter") for item in select_spread(upper, INTERNAL_DYNAMIC_NORMAL_UPPER_EDGES))

        if active_degree_limit > 0:
            edges_to_create = edges_to_create[:active_degree_limit]

        has_local_lower_connection = any(
            float(target_z) < float(z) - float(min_intra_dz)
            for (_dist, _target_id, target_z, _edge_class), _role in edges_to_create
        )

        if (not allow_unanchored and not has_local_lower_connection
                and str(label).startswith("L")
                and int(layer_idx) not in initial_peer_relaxed_layers):
            prior_layer_rejects = sum(
                1 for d in initial_rejection_details
                if int(d.get("layer_idx", -1)) == int(layer_idx)
                and d.get("reason") == "no_local_lower_connection"
            )
            prospective = int(prior_layer_rejects) + 1
            layer_target = max(1, int(target_nodes_for_layer(layer_idx)))
            rejection_fraction = float(prospective) / float(layer_target)
            if (prospective >= int(INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_REJECTIONS)
                    and rejection_fraction >= float(INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_FRACTION)):
                initial_peer_relaxed_layers.add(int(layer_idx))
                log(
                    f"    Initial peer-only relaxation activated: layer {layer_idx + 1}/{n_layers}, "
                    f"no-lower rejections={prospective}, fraction={rejection_fraction:.3f}."
                )

        peer_relax_active = bool(
            allow_layer_peer_without_lower
            or int(layer_idx) in initial_peer_relaxed_layers
        )

        if (peer_relax_active and not allow_unanchored
                and not has_local_lower_connection and not internal_df.empty):
            peer_limit = float(INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_LENGTH_MM)
            peer_max_count = max(1, int(INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_PEERS))
            existing_targets = {
                int(item[0][1]) for item in edges_to_create
                if item and item[0] is not None
            }
            peer_candidates = []
            peer_coords = internal_df[["x", "y", "z"]].to_numpy(dtype=float)
            peer_ids = internal_df["atom_id"].to_numpy(dtype=int)
            for idx_peer, peer_xyz in enumerate(peer_coords):
                peer_id = int(peer_ids[idx_peer])
                peer_z = float(peer_xyz[2])
                if not (float(z0) - 1.0e-9 <= peer_z <= float(z1) + 1.0e-9):
                    continue
                if peer_id in existing_targets:
                    continue
                abnormal_target_degree_limit = (
                        existing_target_degree_limit
                        + max(0, int(INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_DEGREE_RESERVE))
                )
                if (abnormal_target_degree_limit > 0
                        and degree_now.get(peer_id, 0) >= abnormal_target_degree_limit):
                    continue
                d3 = float(np.linalg.norm(peer_xyz - np.asarray([x, y, z], dtype=float)))
                if not np.isfinite(d3) or d3 <= 1.0e-9 or d3 > peer_limit + 1.0e-9:
                    continue
                dxy = float(np.linalg.norm(peer_xyz[:2] - np.asarray([x, y], dtype=float)))
                if dxy > local_pitch * INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC + 1.0e-9:
                    continue

                domain_mid = _domain_at_z_cached(mesh, 0.5 * (float(z) + peer_z), domain_cache)
                c0 = _point_component_index(domain_mid, float(x), float(y))
                c1 = _point_component_index(domain_mid, float(peer_xyz[0]), float(peer_xyz[1]))
                if c0 is None or c1 is None or c0 != c1:
                    continue
                if not _edge_centerline_stays_in_domain(
                        mesh, domain_cache,
                        (float(x), float(y), float(z)),
                        (float(peer_xyz[0]), float(peer_xyz[1]), peer_z),
                        samples=5):
                    continue
                peer_candidates.append((d3, peer_id, peer_z, "short_bridge"))

            peer_candidates.sort(key=lambda q: q[0])
            selected_peer = peer_candidates[:peer_max_count]
            if selected_peer:

                peer_ids_selected = {int(item[1]) for item in selected_peer}
                peer_entries = [(item, "layer_peer_relaxed") for item in selected_peer]
                ordinary_entries = [
                    entry for entry in edges_to_create
                    if int(entry[0][1]) not in peer_ids_selected
                ]
                edges_to_create = peer_entries + ordinary_entries
                if active_degree_limit > 0:
                    edges_to_create = edges_to_create[:active_degree_limit]

        has_layer_peer_connection = any(
            float(z0) - 1.0e-9 <= float(target_z) <= float(z1) + 1.0e-9
            and str(role) == "layer_peer_relaxed"
            for (_dist, _target_id, target_z, _edge_class), role in edges_to_create
        )
        peer_only_ok = bool(peer_relax_active and has_layer_peer_connection)

        if not allow_unanchored and not has_local_lower_connection and not peer_only_ok:
            if peer_relax_active:
                return reject("no_local_peer_connection")
            return reject("no_local_lower_connection")
        new_supported = bool(edges_to_create) or bool(allow_unanchored) or bool(new_root)
        new_edges = []
        nid = int(next_id)
        for (_, target_id, target_z, edge_class), role in edges_to_create:
            radius = 0.5 * (_local_internal_radius(z, z_bounds) + _local_internal_radius(target_z, z_bounds))
            new_edges.append({"source": nid, "target": int(target_id), "radius": float(radius), "bond_type": "internal",
                              "role": "top_relaxed" if edge_class == "top_relaxed" else role})

        if not new_edges and len(internal_ids) >= 3 and not allow_unanchored:
            return reject('no_qualified_edge_selected')
        if (INTERNAL_MATCHED_RANDOM_POSITION and INTERNAL_MATCHED_RANDOM_EDGE_BUDGET > 0
                and len(all_edges) + len(new_edges) > INTERNAL_MATCHED_RANDOM_EDGE_BUDGET):
            return reject('matched_edge_budget_exhausted')

        row = {"atom_id": nid, "x": float(x), "y": float(y), "z": float(z),
               "face_name": str(label), "local_seed": bool(allow_unanchored),
               "peer_only_relaxed": bool(peer_only_ok and not has_local_lower_connection)}
        all_atoms = pd.concat([all_atoms, pd.DataFrame([row])], ignore_index=True)
        all_edges.extend(new_edges)
        internal_ids.add(nid)
        supported_internal_ids.add(nid)
        next_id += 1
        global_step += 1
        return True, nid

    def add_node_with_adaptive_support_path(z0: float, z1: float, layer_idx: int,
                                            target_z: float, target_xy,
                                            label: str, domain_override=None,
                                            force_insert: bool = True):
        """Commit an immutable target through a bottom-up multi-height path.

        Every segment independently obeys the ordinary two-layer height,
        configured-angle, length, endpoint-degree and STL-centreline constraints.
        The target XY/Z is never projected toward an existing support.
        """
        nonlocal all_atoms, all_edges, internal_ids, layer_nodes, next_id, global_step
        target = (float(target_xy[0]), float(target_xy[1]), float(target_z))
        direct = exact_lower_support_anchors(
            target[0], target[1], target[2], allow_inbound_reserve=True)
        if direct:
            ok, nid = add_node_to_layer(
                z0, z1, layer_idx, force_insert=force_insert, label=label,
                specific_z=target[2], specific_xy=target[:2],
                domain_override=domain_override,
                required_support_id=int(direct[0][2]))
            if ok and nid is not None:
                layer_nodes[layer_idx].append(int(nid))
            return bool(ok), nid, ([int(nid)] if ok and nid is not None else []), 'direct'

        layer_h = total_h / max(n_layers, 1)
        existing_xyz = all_atoms[all_atoms['atom_id'].isin(internal_ids)][['x', 'y', 'z']].to_numpy(float)
        existing_tree = cKDTree(existing_xyz) if len(existing_xyz) else None

        queue = [(target, [target])]
        visited = set()
        solution = None
        solution_anchor = None
        max_states = max(80, int(12 * math.ceil(max(target[2] - zmin, layer_h) / layer_h)))
        while queue and len(visited) < max_states:
            current, path = queue.pop(0)
            key = (round(current[0] / max(0.35 * _local_internal_pitch(current[2], z_bounds), 0.15)),
                   round(current[1] / max(0.35 * _local_internal_pitch(current[2], z_bounds), 0.15)),
                   round(current[2] / max(0.45 * layer_h, 0.1)))
            if key in visited:
                continue
            visited.add(key)
            anchors = exact_lower_support_anchors(
                current[0], current[1], current[2], allow_inbound_reserve=True)
            if anchors:
                solution, solution_anchor = path, int(anchors[0][2])
                break
            if current[2] <= zmin + 0.55 * layer_h or len(path) > n_layers + 2:
                continue
            predecessors = []
            for dz in (1.75 * layer_h, 1.25 * layer_h, 0.75 * layer_h):
                z_lower = float(max(zmin + 0.20 * layer_h, current[2] - dz))
                actual_dz = current[2] - z_lower
                if actual_dz <= 1.0e-6 or actual_dz > 2.0 * layer_h + 1.0e-6:
                    continue
                domain_lower = _domain_at_z_cached(mesh, z_lower, domain_cache)
                if domain_lower.is_empty:
                    continue
                lateral_limit = actual_dz / math.tan(math.radians(max(
                    float(INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG), 1.0e-6)))
                radial_values = (0.0, 0.42 * lateral_limit,
                                 0.78 * lateral_limit, 0.96 * lateral_limit)
                angle_offset = (len(path) * 0.3819660112501051) * 2.0 * math.pi
                for radial in radial_values:
                    angles = (0.0,) if radial <= 1.0e-12 else tuple(
                        angle_offset + 2.0 * math.pi * j / 12.0 for j in range(12))
                    for angle in angles:
                        x_lower = current[0] + radial * math.cos(angle)
                        y_lower = current[1] + radial * math.sin(angle)
                        if not domain_lower.covers(Point(x_lower, y_lower)):
                            continue
                        lower = (float(x_lower), float(y_lower), z_lower)
                        if _internal_edge_print_class(current, lower) != 'inclined':
                            continue
                        pitch_lower = _local_internal_pitch(z_lower, z_bounds)
                        max_len = max(layer_h, pitch_lower) * INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR
                        if math.dist(current, lower) > max_len + 1.0e-9:
                            continue
                        if not _edge_centerline_stays_in_domain(
                                mesh, domain_cache, current, lower, samples=5):
                            continue
                        if existing_tree is not None:
                            nearest = float(existing_tree.query(lower, k=1)[0])
                            if nearest < max(INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR * pitch_lower, 1.0e-6):
                                continue
                        if any(math.dist(lower, p) < max(
                                INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR * pitch_lower, 1.0e-6)
                               for p in path):
                            continue

                        goal = exact_lower_support_anchors(
                            lower[0], lower[1], lower[2], allow_inbound_reserve=True)
                        priority = (0 if goal else 1, -actual_dz, radial)
                        predecessors.append((priority, lower))
            for _, lower in sorted(predecessors, key=lambda item: item[0])[:18]:
                queue.append((lower, path + [lower]))

        if solution is None or solution_anchor is None:
            return False, None, [], 'no_multi_height_support_path'

        atoms_before, edges_before = all_atoms.copy(), list(all_edges)
        ids_before = set(internal_ids)
        layers_before = [list(v) for v in layer_nodes]
        supported_before = set(supported_internal_ids)
        next_before, step_before = int(next_id), int(global_step)
        created = []
        previous_id = int(solution_anchor)

        for point in reversed(solution[1:]):
            li = int(np.clip(np.searchsorted(layer_z, point[2], side='right') - 1, 0, n_layers - 1))
            ok, nid = add_node_to_layer(
                float(layer_z[li]), float(layer_z[li + 1]), li, force_insert=True,
                label='SMART_RELAY', specific_z=point[2], specific_xy=point[:2],
                required_support_id=previous_id)
            if not ok or nid is None:
                break
            layer_nodes[li].append(int(nid));
            created.append(int(nid));
            previous_id = int(nid)
        else:
            ok, nid = add_node_to_layer(
                z0, z1, layer_idx, force_insert=force_insert, label=label,
                specific_z=target[2], specific_xy=target[:2],
                domain_override=domain_override, required_support_id=previous_id)
            if ok and nid is not None:
                layer_nodes[layer_idx].append(int(nid));
                created.append(int(nid))
                return True, int(nid), created, 'relay_path'

        all_atoms, all_edges, internal_ids, layer_nodes = (
            atoms_before, edges_before, ids_before, layers_before)
        supported_internal_ids.clear();
        supported_internal_ids.update(supported_before)
        next_id, global_step = next_before, step_before
        return False, None, [], 'relay_commit_failed'

    def add_smart_bridge_cluster(z0: float, z1: float, layer_idx: int,
                                 z_center: float, xy_center: Tuple[float, float],
                                 max_points: int = 4,
                                 domain_override=None) -> List[int]:
        added_ids: List[int] = []
        z_center = float(np.clip(z_center, z0 + 1e-4, z1 - 1e-4))
        domain = domain_override if domain_override is not None else _domain_at_z_cached(mesh, z_center, domain_cache)
        if domain.is_empty:
            return added_ids
        cx, cy = float(xy_center[0]), float(xy_center[1])
        if not domain.covers(Point(cx, cy)):
            sampled = _sample_xy_in_domain(domain, rng)
            if sampled is None:
                return added_ids
            cx, cy = sampled

        local_pitch = _local_internal_pitch(z_center, z_bounds)
        local_radius = _local_internal_radius(z_center, z_bounds)
        spacing = max(
            local_pitch * max(INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR * 1.25, 0.16),
            local_radius * 2.05,
            0.08,
        )
        base_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        angles = [base_angle + i * 2.0 * math.pi / 6.0 for i in range(6)]
        radii = [0.0, spacing, spacing * 1.45]
        candidates: List[Tuple[float, float]] = []
        for r in radii:
            if r <= 1e-9:
                candidates.append((cx, cy))
                continue
            for a in angles:
                candidates.append((cx + r * math.cos(a), cy + r * math.sin(a)))

        for px, py in candidates:
            if len(added_ids) >= max_points:
                break
            if not domain.covers(Point(px, py)):
                continue
            ok, nid = add_node_to_layer(
                z0, z1, layer_idx,
                force_insert=True,
                label="SMART_BRIDGE",
                specific_z=float(np.clip(
                    z_center + [0.0, -0.30, 0.30, -0.45, 0.45][len(added_ids) % 5] * max(z1 - z0, 1e-6),
                    z0 + 1.0e-4,
                    z1 - 1.0e-4,
                )),
                specific_xy=(px, py),
                allow_unanchored=(len(added_ids) == 0),
                domain_override=domain,
            )
            if ok and nid is not None:
                added_ids.append(int(nid))
        return added_ids

    def add_top_same_layer_bridges(z_center: float, max_edges: int, rejected_pairs: Set[Tuple[int, int]]):
        """Trial short bridges between already-supported nodes near a failing top slice.

        These edges do not create support. Both endpoints must already have a
        downward support path. Candidates are restricted to short-bridge geometry
        and to rods whose mid-height is close enough to affect the failing slice.
        Returns the exact added pairs so a rejected pressure trial can blacklist
        them and avoid repeating the same expensive global evaluation.
        """
        nonlocal all_edges
        if max_edges <= 0 or not internal_ids:
            return []
        internal_df = all_atoms[all_atoms["atom_id"].isin(internal_ids)].copy()
        if len(internal_df) < 2:
            return []

        support_state = _internal_support_state(all_atoms, all_edges, z_bounds=z_bounds)
        supported = set(support_state.get("supported", set()))
        degree = dict(support_state.get("degree", {}))
        ids = internal_df["atom_id"].to_numpy(dtype=int)
        coords = internal_df[["x", "y", "z"]].to_numpy(dtype=float)
        index_of = {int(node_id): i for i, node_id in enumerate(ids)}
        existing = {
            tuple(sorted((int(e.get("source", -1)), int(e.get("target", -1)))))
            for e in all_edges if str(e.get("bond_type", "")) == "internal"
        }
        bin_w = max(float(INTERNAL_DYNAMIC_PHYSICS_DZ_MM), 1.0e-9)
        active_ids = [
            int(node_id) for node_id, xyz in zip(ids, coords)
            if int(node_id) in supported
               and abs(float(xyz[2]) - float(z_center)) <= 0.45 * bin_w + 1.0e-9
               and (INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW <= 0
                    or degree.get(int(node_id), 0) < INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW)
        ]
        if len(active_ids) < 2:
            return []

        active_coords = np.asarray([coords[index_of[i]] for i in active_ids], dtype=float)
        tree = cKDTree(active_coords)
        bridge_limit = float(INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM)
        candidates = []
        for local_i, node_id in enumerate(active_ids):
            for local_j in tree.query_ball_point(active_coords[local_i], r=bridge_limit + 1.0e-9):
                if int(local_j) <= int(local_i):
                    continue
                target_id = int(active_ids[int(local_j)])
                pair = tuple(sorted((int(node_id), target_id)))
                if pair in existing or pair in rejected_pairs:
                    continue
                top_degree_limit = max(
                    int(INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW),
                    int(INTERNAL_DYNAMIC_TOP_MAX_EDGES_PER_NODE),
                )
                if (top_degree_limit > 0 and
                        (degree.get(int(node_id), 0) >= top_degree_limit
                         or degree.get(target_id, 0) >= top_degree_limit)):
                    continue
                a = coords[index_of[int(node_id)]]
                b = coords[index_of[target_id]]
                prefill_class = _internal_edge_print_class(a, b)
                if (prefill_class == 'rejected' and
                        float(z_center) >= float(zmax) - float(INTERNAL_TOP_RELAX_ACTIVE_HEIGHT_MM) - 1e-9):
                    prefill_class = _top_relaxed_internal_edge_class(a, b, z_bounds=z_bounds)
                if not (prefill_class in ('short_bridge', 'top_relaxed')
                        or (prefill_class == 'inclined' and float(np.linalg.norm(b - a)) <= bridge_limit + 1.0e-9)):
                    continue
                z_mid = 0.5 * (float(a[2]) + float(b[2]))
                radius = 0.5 * (
                        _local_internal_radius(float(a[2]), z_bounds)
                        + _local_internal_radius(float(b[2]), z_bounds)
                )

                if not (min(float(a[2]), float(b[2])) - radius - 1.0e-9
                        <= float(z_center) <=
                        max(float(a[2]), float(b[2])) + radius + 1.0e-9):
                    continue
                mid_domain = _domain_at_z_cached(mesh, z_mid, domain_cache)
                if (_point_component_index(mid_domain, float(a[0]), float(a[1])) is None
                        or _point_component_index(mid_domain, float(a[0]), float(a[1])) !=
                        _point_component_index(mid_domain, float(b[0]), float(b[1]))):
                    continue
                if not _edge_centerline_stays_in_domain(
                        mesh, domain_cache, tuple(a), tuple(b), samples=3
                ):
                    continue
                span = float(np.linalg.norm(b - a))

                candidates.append((
                    abs(z_mid - float(z_center)),
                    -span,
                    degree.get(int(node_id), 0) + degree.get(target_id, 0),
                    int(node_id), target_id, float(radius), pair, prefill_class,
                ))

        added_pairs = []
        for _, _, _, u, v, radius, pair, prefill_class in sorted(candidates):
            if len(added_pairs) >= int(max_edges):
                break
            if (INTERNAL_MATCHED_RANDOM_POSITION and INTERNAL_MATCHED_RANDOM_EDGE_BUDGET > 0
                    and len(all_edges) >= INTERNAL_MATCHED_RANDOM_EDGE_BUDGET):
                break
            if (top_degree_limit > 0 and
                    (degree.get(u, 0) >= top_degree_limit
                     or degree.get(v, 0) >= top_degree_limit)):
                continue
            all_edges.append({
                "source": int(u), "target": int(v), "radius": float(radius),
                "bond_type": "internal", "role": (
                    "top_relaxed" if prefill_class == "top_relaxed" else "top_same_bridge_prefill"),
            })
            existing.add(pair)
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            added_pairs.append(pair)
        return added_pairs

    def repair_downward_support(max_edges: int):
        """Add only strict inclined links from unsupported nodes to supported lower nodes."""
        nonlocal all_edges
        added = 0
        while added < int(max_edges):
            state = _internal_support_state(all_atoms, all_edges, z_bounds=z_bounds)
            missing = sorted(state['nodes'] - state['supported'],
                             key=lambda i: float(all_atoms.loc[all_atoms.atom_id == i, 'z'].iloc[0]))
            if not missing:
                break
            xyz = {int(r.atom_id): np.array([r.x, r.y, r.z], float)
                   for r in all_atoms.itertuples(index=False) if int(r.atom_id) in state['nodes']}
            existing = {tuple(sorted((int(e.get('source', -1)), int(e.get('target', -1)))))
                        for e in all_edges if str(e.get('bond_type', '')) == 'internal'}
            progress = False
            for nid in missing:
                a = xyz[nid]
                candidates = []
                for sid in state['supported']:
                    b = xyz[sid]
                    if b[2] >= a[2] - 1.0e-9 or tuple(sorted((nid, sid))) in existing:
                        continue
                    if (INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW > 0 and
                            state['degree'].get(sid, 0) >=
                            INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW +
                            INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE):
                        continue
                    span = float(np.linalg.norm(a - b))
                    local_pitch = _local_internal_pitch(float(a[2]), z_bounds)
                    layer_h = total_h / max(n_layers, 1)
                    if span > max(layer_h, local_pitch) * INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR:
                        continue
                    if _internal_edge_print_class(a, b) != 'inclined':
                        continue
                    if not _edge_centerline_stays_in_domain(mesh, domain_cache, tuple(a), tuple(b), samples=3):
                        continue
                    candidates.append((span, state['degree'].get(sid, 0), int(sid)))
                if not candidates:
                    continue
                _, _, sid = min(candidates)
                radius = 0.5 * (_local_internal_radius(float(a[2]), z_bounds)
                                + _local_internal_radius(float(xyz[sid][2]), z_bounds))
                if (INTERNAL_MATCHED_RANDOM_POSITION and INTERNAL_MATCHED_RANDOM_EDGE_BUDGET > 0
                        and len(all_edges) >= INTERNAL_MATCHED_RANDOM_EDGE_BUDGET):
                    return added
                all_edges.append({'source': int(nid), 'target': sid, 'radius': float(radius),
                                  'bond_type': 'internal', 'role': 'support_repair'})
                added += 1
                progress = True
                break
            if not progress:
                break
        return added

    def remove_invalid_internal_edges():
        """Drop nonmanufacturable edges only when support/root attachment is preserved."""
        nonlocal all_edges
        removed = 0
        while True:
            state = _internal_support_state(all_atoms, all_edges, z_bounds=z_bounds)
            bad = {pair for pair, cls in state['classes'].items() if cls == 'rejected'} | set(state['bridge_bad'])
            if not bad:
                break
            progress = False
            for pair in sorted(bad):
                trial = [e for e in all_edges if not (
                        str(e.get('bond_type', '')) == 'internal' and
                        tuple(sorted((int(e.get('source', -1)), int(e.get('target', -1))))) == pair)]
                if len(trial) == len(all_edges):
                    continue
                if _deletion_preserves_internal_support(
                        all_atoms, all_edges, all_atoms, trial, z_bounds=z_bounds):
                    all_edges = trial
                    removed += 1
                    progress = True
                    break
            if not progress:
                break
        return removed

    def remove_unrepairable_unsupported_nodes() -> int:
        """Delete the residual unsupported set when all supported nodes stay supported."""
        nonlocal all_atoms, all_edges, internal_ids, layer_nodes
        state = _internal_support_state(all_atoms, all_edges, z_bounds=z_bounds)
        doomed = set(state['nodes'] - state['supported'])
        if not doomed:
            return 0
        doomed_rows = all_atoms[all_atoms['atom_id'].astype(int).isin(doomed)]
        log("  Support repair could not retain nodes under the strict angle/span/domain rules: " +
            ", ".join(
                f"{int(row.atom_id)}@z={float(row.z):.3f}"
                for row in doomed_rows.sort_values('z').itertuples(index=False)))
        trial_atoms = all_atoms[
            ~all_atoms['atom_id'].astype(int).isin(doomed)].reset_index(drop=True)
        trial_edges = [e for e in all_edges
                       if int(e.get('source', -1)) not in doomed
                       and int(e.get('target', -1)) not in doomed]
        if not _deletion_preserves_internal_support(
                all_atoms, all_edges, trial_atoms, trial_edges, z_bounds=z_bounds):
            return 0
        all_atoms, all_edges = trial_atoms, trial_edges
        internal_ids.difference_update(doomed)
        for nodes in layer_nodes:
            nodes[:] = [node_id for node_id in nodes if node_id not in doomed]
        return len(doomed)

    def optimise_high_resistance_channel(profile: pd.DataFrame, max_evaluations: int):
        """Greedily remove noncritical solids only when the complete safety rank improves."""
        nonlocal all_atoms, all_edges, internal_ids
        if profile is None or profile.empty or max_evaluations <= 0:
            return profile, 0, 0
        work = profile.copy()
        zcol = 'z_mid_mm' if 'z_mid_mm' in work.columns else 'z_mm'
        mobility = pd.Series(_viscous_mobility_values(work), index=work.index)
        mobility_max = max(float(mobility.max()), 1.0e-300)
        mobility_weakness = 1.0 + np.clip(
            (mobility_max - mobility) / mobility_max, 0.0, 1.0)
        margin = _capillary_margin_values(work)

        failed_mask = margin < 0.0
        rows = work.loc[failed_mask].copy()
        if rows.empty:
            return profile, 0, 0
        required = np.maximum(
            pd.to_numeric(rows.get('P_required_Pa', 1.0), errors='coerce').fillna(1.0).to_numpy(float),
            1.0e-9,
        )
        deficit_scale = 1.0 + np.clip(-margin[rows.index] / required, 0.0, None)
        rows['_priority'] = mobility_weakness.loc[rows.index].to_numpy(float) * deficit_scale
        rows = rows.sort_values('_priority', ascending=False)
        accepted = evaluations = 0
        for row in rows.itertuples(index=False):
            if evaluations >= int(max_evaluations):
                break
            zc = float(getattr(row, zcol))
            state = _internal_support_state(all_atoms, all_edges, z_bounds=z_bounds)
            candidates = all_atoms[
                all_atoms.atom_id.isin(internal_ids)
                & all_atoms.face_name.astype(str).str.contains('internal', case=False, regex=False)
                & ((all_atoms.z - zc).abs() <= 0.45 * float(INTERNAL_DYNAMIC_PHYSICS_DZ_MM))
                ].copy()

            candidates['_degree'] = candidates.atom_id.map(state['degree']).fillna(0)
            for cand in candidates.sort_values('_degree').itertuples(index=False):
                if evaluations >= int(max_evaluations):
                    break
                nid = int(cand.atom_id)
                if nid in state['roots'] or state['degree'].get(nid, 0) <= 1:
                    continue
                atoms_trial = all_atoms[all_atoms.atom_id != nid].reset_index(drop=True)
                edges_trial = [e for e in all_edges
                               if int(e.get('source', -1)) != nid and int(e.get('target', -1)) != nid]
                if not _deletion_preserves_internal_support(
                        all_atoms, all_edges, atoms_trial, edges_trial, z_bounds=z_bounds):
                    continue
                evaluations += 1
                _, _, _, _, trial = _capillary_ok_stl(
                    mesh, atoms_trial, edges_trial, z_bounds,
                    INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache)
                if _profile_trial_improves(profile, trial):
                    all_atoms, all_edges = atoms_trial, edges_trial
                    internal_ids.discard(nid)
                    profile = trial
                    accepted += 1
                    break
        return profile, accepted, evaluations

    def try_viscous_edge_relief(z_center: float, profile: pd.DataFrame,
                                rejected_pairs: Set[Tuple[int, int]],
                                max_evaluations: int):
        """Cumulative-path viscous relief for one failed height.

        A viscous-dominated failure at z_center is not assumed to originate at
        that same horizontal slice.  Viscous loss is cumulative, so candidate
        deletion sites are searched over the complete transport path from the
        bottom to z_center.  Sections with the largest local geometry resistance
        (or, when available, local viscous-pressure increment) receive the
        highest priority.  Within those sections, only noncritical INTERNAL
        edges are tried.  Every deletion must preserve print/support connectivity
        and improve the COMPLETE pressure profile without turning an already-safe
        slice into a failing one.
        """
        nonlocal all_edges, supported_internal_ids
        if profile is None or profile.empty or max_evaluations <= 0:
            return profile, False, 0, {"reason": "disabled_or_empty"}

        z_col = "z_mid_mm" if "z_mid_mm" in profile.columns else "z_mm"
        if z_col not in profile.columns:
            return profile, False, 0, {"reason": "missing_height_column"}

        z_vals = pd.to_numeric(profile[z_col], errors="coerce").to_numpy(float)
        path_mask = np.isfinite(z_vals) & (z_vals <= float(z_center) + 1.0e-9)
        path_rows = profile.loc[path_mask].copy()
        if path_rows.empty:
            return profile, False, 0, {"reason": "empty_cumulative_path"}

        if "R_geom_segment_m-3" in path_rows.columns:
            contribution = pd.to_numeric(
                path_rows["R_geom_segment_m-3"], errors="coerce"
            ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        else:
            visc_col = "P_visc_dynamic_Pa" if "P_visc_dynamic_Pa" in path_rows.columns else (
                "P_viscous_Pa" if "P_viscous_Pa" in path_rows.columns else None
            )
            if visc_col is not None:
                visc = pd.to_numeric(path_rows[visc_col], errors="coerce").fillna(0.0).to_numpy(float)
                contribution = np.maximum(0.0, np.diff(np.r_[0.0, visc]))
            else:
                contribution = np.zeros(len(path_rows), dtype=float)

        path_z = pd.to_numeric(path_rows[z_col], errors="coerce").to_numpy(float)
        finite = np.isfinite(path_z) & np.isfinite(contribution)
        path_z, contribution = path_z[finite], contribution[finite]
        if not len(path_z):
            return profile, False, 0, {"reason": "no_finite_path_sections"}

        n_top = max(1, int(INTERNAL_DYNAMIC_VISCOUS_RELIEF_PATH_TOP_SEGMENTS))
        order = np.argsort(-contribution)
        order = [int(i) for i in order if contribution[int(i)] > 0.0][:n_top]
        if not order:

            full_mobility = _viscous_mobility_values(path_rows)
            if len(full_mobility):
                order = list(np.argsort(full_mobility)[:n_top])
            else:
                order = [int(np.argmin(np.abs(path_z - float(z_center))))]

        target_sections = []
        max_contrib = max(float(np.max(contribution)), 1.0e-300)
        for i in order:
            target_sections.append((float(path_z[i]), float(contribution[i]) / max_contrib))

        coords = {int(r.atom_id): np.asarray([r.x, r.y, r.z], dtype=float)
                  for r in all_atoms.itertuples(index=False)}
        candidates = []
        for edge_idx, edge in enumerate(all_edges):
            if str(edge.get("bond_type", "")) != "internal":
                continue
            role = str(edge.get("role", "")).lower()
            if any(token in role for token in ("pillar", "support")):
                continue
            u, v = int(edge.get("source", -1)), int(edge.get("target", -1))
            if u not in coords or v not in coords or u == v:
                continue
            pair = (min(u, v), max(u, v))
            if pair in rejected_pairs:
                continue
            a, b = coords[u], coords[v]

            radius = float(edge.get("radius", _local_internal_radius(z_center, z_bounds)))
            edge_lo = min(float(a[2]), float(b[2])) - radius
            edge_hi = max(float(a[2]), float(b[2])) + radius
            if edge_lo > float(z_center) + 1.0e-9:
                continue
            length = float(np.linalg.norm(b - a))
            if length <= 1.0e-9:
                continue
            dxy = float(np.linalg.norm((b - a)[:2]))
            horizontal_fraction = dxy / length

            best_path_score = 0.0
            best_path_z = None
            for z_path, resistance_weight in target_sections:
                if edge_lo - 1.0e-9 <= z_path <= edge_hi + 1.0e-9:
                    mid_dist = abs(0.5 * float(a[2] + b[2]) - z_path)
                    local_score = resistance_weight / max(0.05, mid_dist + 0.05)
                    if local_score > best_path_score:
                        best_path_score = local_score
                        best_path_z = z_path
            if best_path_z is None:
                continue
            blockage_score = (radius ** 2) * (1.0 + horizontal_fraction)
            score = blockage_score * best_path_score
            candidates.append((score, pair, edge_idx, edge, float(best_path_z)))

        if not candidates:
            return profile, False, 0, {
                "reason": "no_untried_edge_candidate_on_cumulative_path",
                "target_sections_mm": [round(z, 4) for z, _ in target_sections],
            }

        before_diag = _pressure_failure_mechanism(profile, z_center)
        before_mobility = _viscous_mobility_values(profile)
        before_idx = (pd.to_numeric(profile[z_col], errors="coerce") - float(z_center)).abs().idxmin()
        before_pos = profile.index.get_loc(before_idx)
        before_m = float(before_mobility[before_pos]) if len(before_mobility) else 0.0
        before_margin = float(_capillary_margin_values(profile)[before_pos])
        before_rank = _capillary_safety_rank(profile)

        evaluations = 0
        for _, pair, edge_idx, edge, path_z_selected in sorted(
                candidates, key=lambda item: item[0], reverse=True):
            if evaluations >= int(max_evaluations):
                break
            trial_edges = [dict(e) for j, e in enumerate(all_edges) if j != int(edge_idx)]
            rejected_pairs.add(pair)
            if not _deletion_preserves_internal_support(
                    all_atoms, all_edges, all_atoms, trial_edges, z_bounds=z_bounds):
                continue
            evaluations += 1
            _, _, _, _, trial = _capillary_ok_stl(
                mesh, all_atoms, trial_edges, z_bounds,
                INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache)
            if trial is None or trial.empty or len(trial) != len(profile):
                continue

            trial_mobility = _viscous_mobility_values(trial)
            after_idx = (pd.to_numeric(trial[z_col], errors="coerce") - float(z_center)).abs().idxmin()
            after_pos = trial.index.get_loc(after_idx)
            after_m = float(trial_mobility[after_pos]) if len(trial_mobility) else 0.0
            after_margin = float(_capillary_margin_values(trial)[after_pos])
            after_rank = _capillary_safety_rank(trial)

            target_relief = bool(
                after_m > before_m * (1.0 + 1.0e-4)
                or after_margin > before_margin + float(INTERNAL_DYNAMIC_SMARTFILL_MIN_DEFICIT_DROP_PA)
            )
            global_relief = bool(
                _smartfill_local_safe_gain(profile, trial)
                and after_rank <= before_rank
            )
            if target_relief and global_relief:
                all_edges = trial_edges
                support = _internal_support_state(all_atoms, all_edges, z_bounds=z_bounds)
                supported_internal_ids.clear()
                supported_internal_ids.update(set(map(int, support.get("supported", set()))))
                after_diag = _pressure_failure_mechanism(trial, z_center)
                return trial, True, evaluations, {
                    "reason": "accepted", "removed_edge": pair,
                    "relief_section_z_mm": float(path_z_selected),
                    "target_sections_mm": [round(z, 4) for z, _ in target_sections],
                    "mobility_before": before_m, "mobility_after": after_m,
                    "margin_before_Pa": before_margin, "margin_after_Pa": after_margin,
                    "mode_before": before_diag.get("mode"),
                    "mode_after": after_diag.get("mode"),
                    "viscous_share_after": after_diag.get("viscous_share", 0.0),
                }
        return profile, False, evaluations, {
            "reason": "pressure_gain_insufficient_on_cumulative_path",
            "target_sections_mm": [round(z, 4) for z, _ in target_sections],
        }

    def try_top_supported_batch(z_center: float, profile: pd.DataFrame,
                                stats: Dict, excluded_xy: List[Tuple[float, float]],
                                geometry_scan_limit: int, batch_size: int):
        """Insert a small distinct support-first batch, then run one pressure solve."""
        nonlocal all_atoms, all_edges, internal_ids, layer_nodes, next_id, global_step
        base_atoms, base_edges = all_atoms.copy(), list(all_edges)
        base_ids = set(internal_ids)
        base_layers = [list(v) for v in layer_nodes]
        base_supported = set(supported_internal_ids)
        base_next, base_step = int(next_id), int(global_step)
        z_target = float(np.clip(z_center, zmin + 1.0e-4, zmax - 1.0e-4))
        idx = int(np.clip(np.searchsorted(layer_z, z_target, side='right') - 1, 0, n_layers - 1))
        z0, z1 = float(layer_z[idx]), float(layer_z[idx + 1])

        local_pitch = _local_internal_pitch(z_target, z_bounds)
        exclusion_radius = max(
            2.0 * _local_internal_radius(z_target, z_bounds) * NODE_DIAM_RATIO,
            0.55 * local_pitch,
        )
        local_excluded = list(excluded_xy)
        added_info: List[Dict] = []
        last_reason = 'no_distinct_supported_position'
        for _ in range(max(1, int(batch_size))):
            pores = _top_pore_candidates(
                mesh, domain_cache, all_atoms, all_edges, z_bounds, z_center,
                limit=max(1, int(geometry_scan_limit)))
            selected = None
            for pore in pores:
                xy = (float(pore['xy'][0]), float(pore['xy'][1]))
                if all(math.hypot(xy[0] - px, xy[1] - py) >= exclusion_radius
                       for px, py in local_excluded):
                    selected = (xy, None, pore['domain'])
                if selected is not None:
                    break
            if selected is not None:
                xy, anchor_id, active_domain = selected
                step_atoms, step_edges = all_atoms.copy(), list(all_edges)
                step_ids = set(internal_ids)
                step_layers = [list(v) for v in layer_nodes]
                step_supported = set(supported_internal_ids)
                step_next, step_global = int(next_id), int(global_step)
                ok_add, nid = add_node_to_layer(
                    z0, z1, idx, force_insert=True, label='SMART_FILL_LINEAR',
                    specific_z=z_target, specific_xy=xy, domain_override=active_domain)
                created_ids = [int(nid)] if ok_add and nid is not None else []
                path_status = 'local_connection' if ok_add else 'local_connection_rejected'
                if ok_add and nid is not None:
                    layer_nodes[idx].append(int(nid))
                info = {'node_id': int(nid) if nid is not None else None,
                        'x_mm': float(xy[0]), 'y_mm': float(xy[1]),
                        'support_id': int(anchor_id) if anchor_id is not None else None,
                        'exclusion_radius_mm': float(exclusion_radius),
                        'path_status': path_status,
                        'created_node_ids': [int(i) for i in created_ids]}
                local_excluded.append((float(xy[0]), float(xy[1])))
                if not ok_add:
                    last_reason = 'geometry_rejected'
                    continue

                before_section = _internal_candidate_geometry(
                    step_atoms, step_edges, z_bounds, z_center)
                after_section = _internal_candidate_geometry(
                    all_atoms, all_edges, z_bounds, z_center)
                try:
                    added_section_area = float(after_section.difference(before_section).area)
                except Exception:
                    added_section_area = 0.0
                info['added_section_area_mm2'] = added_section_area
                if added_section_area <= 1.0e-6:
                    all_atoms, all_edges, internal_ids, layer_nodes = (
                        step_atoms, step_edges, step_ids, step_layers)
                    supported_internal_ids.clear();
                    supported_internal_ids.update(step_supported)
                    next_id, global_step = step_next, step_global
                    last_reason = 'no_section_effect'
                    continue
                added_info.append(info)
            if selected is None:
                break

        if not added_info:
            all_atoms, all_edges, internal_ids, layer_nodes = (
                base_atoms, base_edges, base_ids, base_layers)
            supported_internal_ids.clear();
            supported_internal_ids.update(base_supported)
            next_id, global_step = base_next, base_step
            return profile, 'geometry_exhausted', 0, {
                'reason': last_reason, 'positions': [],
                'exclusion_radius_mm': float(exclusion_radius)}

        _, _, _, _, trial = _smartfill_trial_capillary_ok_stl(
            mesh, all_atoms, all_edges, base_atoms, base_edges, profile,
            z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, domain_cache, stats=stats)
        before_row = profile.iloc[(profile['z_mm'] - z_center).abs().argmin()]
        after_row = trial.iloc[(trial['z_mm'] - z_center).abs().argmin()]
        deficit_drop = float(
            _capillary_safety_rank(profile)[1] - _capillary_safety_rank(trial)[1])
        for info in added_info:
            info['deficit_drop_Pa'] = deficit_drop
            smartfill_placement_log.append({
                'failed_slice_z_mm': float(z_center), 'proposed_z_mm': z_target,
                'support_first': True, 'support_id': info['support_id'],
                'x_mm': info['x_mm'], 'y_mm': info['y_mm'],
                'accepted_geometrically': True,
                'P_cap_before_Pa': float(before_row['P_cap_Pa']),
                'P_cap_after_Pa': float(after_row['P_cap_Pa']),
                'P_required_before_Pa': float(before_row['P_required_Pa']),
                'P_required_after_Pa': float(after_row['P_required_Pa']),
                'deficit_drop_Pa': deficit_drop,
            })
        result_info = {'positions': added_info, 'batch_size': len(added_info),
                       'deficit_drop_Pa': deficit_drop}
        if _smartfill_meaningful_safety_gain(
                profile, trial, insertion_trial=True):
            return trial, 'accepted', 1, result_info

        all_atoms, all_edges, internal_ids, layer_nodes = (
            base_atoms, base_edges, base_ids, base_layers)
        supported_internal_ids.clear();
        supported_internal_ids.update(base_supported)
        next_id, global_step = base_next, base_step
        return profile, 'pressure_gain_insufficient', 1, result_info

    def reconcile_cross_layer_edges() -> int:
        """Add the upward links unavailable during bottom-to-top node creation."""
        nonlocal all_edges
        internal_df = all_atoms[all_atoms["atom_id"].isin(internal_ids)].copy()
        if len(internal_df) < 2:
            return 0
        ids = internal_df["atom_id"].to_numpy(dtype=int)
        coords = internal_df[["x", "y", "z"]].to_numpy(dtype=float)
        index_of = {int(node_id): idx for idx, node_id in enumerate(ids)}
        tree = cKDTree(coords)
        existing: Set[Tuple[int, int]] = set()
        degree = {int(node_id): 0 for node_id in ids}
        cross_degree = {int(node_id): 0 for node_id in ids}
        min_cross_dz = 0.18 * total_h / max(n_layers, 1)
        for edge in all_edges:
            if str(edge.get("bond_type", "")) != "internal":
                continue
            u, v = int(edge.get("source", -1)), int(edge.get("target", -1))
            if u not in index_of or v not in index_of:
                continue
            existing.add(tuple(sorted((u, v))))
            degree[u] += 1
            degree[v] += 1
            if abs(float(coords[index_of[u], 2] - coords[index_of[v], 2])) >= min_cross_dz:
                cross_degree[u] += 1
                cross_degree[v] += 1

        desired_cross = max(
            2,
            int(INTERNAL_DYNAMIC_NORMAL_LOWER_EDGES + INTERNAL_DYNAMIC_NORMAL_UPPER_EDGES),
        )
        added = 0
        for i, node_id in enumerate(ids):
            node_id = int(node_id)
            room = desired_cross - cross_degree[node_id]
            if room <= 0 or (
                    INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW > 0
                    and degree[node_id] >= INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
            ):
                continue
            z = float(coords[i, 2])
            pitch = _local_internal_pitch(z, z_bounds)
            layer_h = total_h / max(n_layers, 1)
            max_len = max(layer_h, pitch) * INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR
            kq = min(len(coords), max(2, int(INTERNAL_DYNAMIC_NEIGHBOR_K)))
            dists, idxs = tree.query(coords[i], k=kq, distance_upper_bound=max_len)
            candidates = []
            for dist, j in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
                if not np.isfinite(dist) or int(j) == i or int(j) >= len(ids):
                    continue
                target_id = int(ids[int(j)])
                pair = tuple(sorted((node_id, target_id)))
                dz_abs = abs(float(coords[int(j), 2] - z))
                dxy = float(np.linalg.norm(coords[int(j), :2] - coords[i, :2]))
                if pair in existing or dz_abs < min_cross_dz:
                    continue
                if dz_abs > INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW * layer_h + 1.0e-6:
                    continue
                if dxy > pitch * INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC:
                    continue
                edge_class = _internal_edge_print_class(coords[i], coords[int(j)])
                if edge_class == 'rejected':
                    continue
                if edge_class == 'short_bridge':
                    if node_id not in supported_internal_ids or target_id not in supported_internal_ids:
                        continue
                else:
                    lower_id = node_id if z <= float(coords[int(j), 2]) else target_id
                    if lower_id not in supported_internal_ids:
                        continue
                if INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW > 0 and degree[target_id] >= INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW:
                    continue
                if not _edge_centerline_stays_in_domain(
                        mesh, domain_cache, tuple(coords[i]), tuple(coords[int(j)]), samples=3
                ):
                    continue
                candidates.append((cross_degree[target_id], float(dist), target_id, int(j)))
            for _, _, target_id, j in sorted(candidates)[:room]:
                if INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW > 0 and (
                        degree[node_id] >= INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                        or degree[target_id] >= INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                ):
                    break
                pair = tuple(sorted((node_id, target_id)))
                if pair in existing:
                    continue
                target_z = float(coords[j, 2])
                radius = 0.5 * (_local_internal_radius(z, z_bounds) + _local_internal_radius(target_z, z_bounds))
                if (INTERNAL_MATCHED_RANDOM_POSITION and INTERNAL_MATCHED_RANDOM_EDGE_BUDGET > 0
                        and len(all_edges) >= INTERNAL_MATCHED_RANDOM_EDGE_BUDGET):
                    break
                all_edges.append({
                    "source": node_id,
                    "target": target_id,
                    "radius": float(radius),
                    "bond_type": "internal",
                    "role": "layer_inter",
                })
                existing.add(pair)
                degree[node_id] += 1
                degree[target_id] += 1
                cross_degree[node_id] += 1
                cross_degree[target_id] += 1

                if _internal_edge_print_class(coords[i], coords[j]) == 'inclined':
                    lower_id, upper_id = (node_id, target_id) if z <= target_z else (target_id, node_id)
                    if lower_id in supported_internal_ids:
                        supported_internal_ids.add(upper_id)
                added += 1
        return int(added)

    for k in ([] if loaded_checkpoint_stage is not None else range(n_layers)):
        rejection_before_layer = dict(initial_rejection_counts)
        z0, z1 = float(layer_z[k]), float(layer_z[k + 1])
        z_mid = 0.5 * (z0 + z1)
        if layer_domains[k].is_empty:
            internal_log.append(
                {"layer_idx": k, "z_mid_mm": z_mid, "target_total": 0, "added": 0, "domain_area_mm2": 0.0})
            continue

        target_total = target_nodes_for_layer(k)
        component_targets = _allocate_targets_to_components(layer_domains[k], target_total, z_mid, z_bounds)
        allocated_total = int(sum(n for _, n in component_targets))
        log(f"  Dynamic layer {k + 1}/{n_layers}: z={z_mid:.2f}mm, domain_area={layer_areas[k]:.3f}, components={len(component_targets)}, target={target_total}, allocated={allocated_total}")

        tries = 0
        tries2 = 0
        for comp_idx, (comp_domain, comp_target) in enumerate(component_targets):
            if comp_target <= 0:
                continue
            comp_start_count = len(layer_nodes[k])
            comp_step1 = int(max(1, math.ceil(comp_target * INTERNAL_DYNAMIC_STEP1_FRACTION)))
            z_slots = stratified_z_values(z0, z1, comp_target)
            if k == 0:
                root_lo = float(zmin + 0.05 * (z1 - z0))
                root_hi = float(zmin + 0.95 * (z1 - z0))
                z_slots = np.linspace(root_lo, root_hi, max(1, int(comp_target)))
            comp_try_budget = min(
                INTERNAL_DYNAMIC_MAX_TRIES_PER_LAYER,
                INTERNAL_DYNAMIC_COMPONENT_TRIES_BASE + INTERNAL_DYNAMIC_COMPONENT_TRIES_PER_NODE * max(1,
                                                                                                        int(comp_target)),
            )
            comp_tries = 0
            comp_fail_streak = 0
            while comp_tries < comp_try_budget and (len(layer_nodes[k]) - comp_start_count) < comp_step1:
                comp_tries += 1
                tries += 1
                comp_added_now = len(layer_nodes[k]) - comp_start_count
                z_pick = float(z_slots[(comp_tries - 1) % len(z_slots)])
                if comp_added_now < min(2, int(comp_target)):
                    ok, nid = add_node_to_layer(
                        z0, z1, k, force_insert=False,
                        label=f"L{k + 1}_C{comp_idx}_uni",
                        specific_z=z_pick, allow_unanchored=True,
                        domain_override=comp_domain,
                    )
                else:

                    ok, nid = add_node_to_layer(
                        z0, z1, k, force_insert=False,
                        label=f"L{k + 1}_C{comp_idx}_uni",
                        specific_z=z_pick, allow_unanchored=False,
                        domain_override=comp_domain,
                    )
                    if not ok:
                        candidate, candidate_status = best_connectable_initial_candidate(
                            z0, z1, k, z_pick, comp_domain,
                            INTERNAL_DYNAMIC_CANDIDATES,
                            INTERNAL_DYNAMIC_MIN_DIST_FACTOR,
                        )
                    else:
                        candidate, candidate_status = None, 'fast_candidate_accepted'
                    if not ok and candidate is None:
                        initial_rejection_counts[candidate_status] = (
                                initial_rejection_counts.get(candidate_status, 0) + 1)
                    elif not ok:
                        cx, cy, cz, support_id = candidate
                        ok, nid = add_node_to_layer(
                            z0, z1, k, force_insert=False,
                            label=f"L{k + 1}_C{comp_idx}_uni",
                            specific_z=cz, specific_xy=(cx, cy),
                            allow_unanchored=False, domain_override=comp_domain,
                            required_support_id=int(support_id),
                        )
                if ok:
                    layer_nodes[k].append(int(nid))
                    comp_fail_streak = 0
                else:
                    comp_fail_streak += 1
                    if comp_fail_streak >= max(8, len(z_slots)):
                        break

            comp_force_budget = min(
                INTERNAL_DYNAMIC_MAX_TRIES_PER_LAYER,
                INTERNAL_DYNAMIC_COMPONENT_TRIES_BASE + INTERNAL_DYNAMIC_COMPONENT_TRIES_PER_NODE * max(1,
                                                                                                        int(comp_target)),
            )
            comp_force_tries = 0
            comp_force_fail_streak = 0
            while comp_force_tries < comp_force_budget and (len(layer_nodes[k]) - comp_start_count) < comp_target:
                comp_force_tries += 1
                tries2 += 1
                comp_added_now = len(layer_nodes[k]) - comp_start_count

                z_pick = float(z_slots[(comp_step1 + comp_force_tries - 1) % len(z_slots)])

                if comp_added_now < min(2, int(comp_target)):
                    ok, nid = add_node_to_layer(
                        z0, z1, k, force_insert=True,
                        label=f"L{k + 1}_C{comp_idx}_init",
                        specific_z=z_pick, allow_unanchored=True,
                        domain_override=comp_domain,
                    )
                else:
                    ok, nid = add_node_to_layer(
                        z0, z1, k, force_insert=True,
                        label=f"L{k + 1}_C{comp_idx}_init",
                        specific_z=z_pick, allow_unanchored=False,
                        domain_override=comp_domain,
                    )
                    if not ok:
                        candidate, candidate_status = best_connectable_initial_candidate(
                            z0, z1, k, z_pick, comp_domain,
                            INTERNAL_DYNAMIC_CANDIDATES * 2,
                            INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR,
                        )
                    else:
                        candidate, candidate_status = None, 'fast_candidate_accepted'
                    if not ok and candidate is None:
                        initial_rejection_counts[candidate_status] = (
                                initial_rejection_counts.get(candidate_status, 0) + 1)
                    elif not ok:
                        cx, cy, cz, support_id = candidate
                        ok, nid = add_node_to_layer(
                            z0, z1, k, force_insert=True,
                            label=f"L{k + 1}_C{comp_idx}_init",
                            specific_z=cz, specific_xy=(cx, cy),
                            allow_unanchored=False, domain_override=comp_domain,
                            required_support_id=int(support_id),
                        )
                if ok:
                    layer_nodes[k].append(int(nid))
                    comp_force_fail_streak = 0
                else:
                    comp_force_fail_streak += 1
                    if comp_force_fail_streak >= max(8, len(z_slots)):
                        break

        process_log.append({"step": int(global_step), "layer_idx": int(k), "phase": "STEP1_2_STL_DOMAIN_COMPONENTS",
                            "layer_nodes": int(len(layer_nodes[k])), "target_total": int(target_total),
                            "allocated_total": int(allocated_total), "component_count": int(len(component_targets)),
                            "z0": float(z0), "z1": float(z1), "domain_area_mm2": float(layer_areas[k])})
        internal_log.append({"layer_idx": int(k), "z_mid_mm": float(z_mid), "target_total": int(target_total),
                             "allocated_total": int(allocated_total), "added": int(len(layer_nodes[k])),
                             "component_count": int(len(component_targets)),
                             "component_targets": ";".join(str(n) for _, n in component_targets),
                             "domain_area_mm2": float(layer_areas[k]), "tries_step1": int(tries),
                             "tries_step2": int(tries2)})
        log(f"    Initial quota result: placed={len(layer_nodes[k])}/{allocated_total}, "
            f"shortfall={max(0, allocated_total - len(layer_nodes[k]))}.")
        rejection_delta = {
            reason: count - rejection_before_layer.get(reason, 0)
            for reason, count in initial_rejection_counts.items()
            if count - rejection_before_layer.get(reason, 0) > 0
        }
        internal_log[-1]["initial_rejections"] = dict(rejection_delta)
        internal_log[-1]["initial_rejection_count"] = int(sum(rejection_delta.values()))
        if rejection_delta:
            log(f"    Initial candidate rejections: {json.dumps(rejection_delta, sort_keys=True)}")

    def densify_detected_abnormal_layers() -> int:
        """Add local sampling capacity where a widening triggered peer relaxation."""
        if loaded_checkpoint_stage is not None or not initial_peer_relaxed_layers:
            return 0
        total_added = 0
        for layer_idx in sorted(initial_peer_relaxed_layers):
            base_target = max(1, int(target_nodes_for_layer(layer_idx)))
            extra_target = int(np.clip(
                math.ceil(base_target * float(INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_FRACTION)),
                int(INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_MIN_NODES),
                int(INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_MAX_NODES),
            ))
            added = 0
            tries = 0
            max_tries = max(80, 20 * extra_target)
            z0, z1 = float(layer_z[layer_idx]), float(layer_z[layer_idx + 1])
            while added < extra_target and tries < max_tries:
                tries += 1
                ok, nid = add_node_to_layer(
                    z0, z1, layer_idx, force_insert=True,
                    label=f"ABNORMAL_DENSIFY_L{layer_idx + 1}",
                    allow_unanchored=False,
                    domain_override=layer_domains[layer_idx],
                    allow_layer_peer_without_lower=True,
                )
                if ok and nid is not None:
                    layer_nodes[layer_idx].append(int(nid))
                    added += 1
                    total_added += 1
            log(f"  Abnormal-layer densification: layer {layer_idx + 1}/{n_layers}, "
                f"added={added}/{extra_target}, attempts={tries}, "
                f"peer_limit={INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_LENGTH_MM:.2f}mm.")
            process_log.append({
                'step': int(global_step), 'layer_idx': int(layer_idx),
                'phase': 'STEP1_25_ABNORMAL_LAYER_DENSIFY',
                'base_target': int(base_target), 'extra_target': int(extra_target),
                'added_nodes': int(added), 'attempts': int(tries),
                'peer_limit_mm': float(INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_LENGTH_MM),
            })
        return int(total_added)

    densified_abnormal = densify_detected_abnormal_layers()
    if densified_abnormal:
        log(f"  Abnormal-layer densification added {densified_abnormal} nearby nodes in total.")

    if loaded_checkpoint_stage is None:
        reconciled_initial = reconcile_cross_layer_edges()
        log(f"  Cross-layer reconciliation added {reconciled_initial} edges after initial placement.")
    else:
        log(f"  Initial construction bypassed by {loaded_checkpoint_stage} checkpoint.")

    def repair_persistent_initial_lower_connection_bands() -> int:
        """Repair unsupported initial-skeleton expansion bands before pressure filling.

        Triggering is based only on repeated ``no_local_lower_connection`` rejections
        relative to that layer's requested quota.  No absolute Z range is encoded.
        Rejected XYZ sites are ranked by current spatial under-coverage and de-duplicated.
        In a triggered layer the retry no longer requires a lower-layer anchor: a target
        is retained when it forms at least one otherwise legal edge to an existing node
        in the same quota layer.  No absolute Z range is encoded.
        """
        if (not INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_ENABLE
                or loaded_checkpoint_stage is not None):
            return 0

        details = [d for d in initial_rejection_details
                   if d.get("reason") == "no_local_lower_connection"
                   and d.get("x_mm") is not None and d.get("y_mm") is not None
                   and d.get("z_mm") is not None]
        if not details:
            log("  Initial support-band repair: no recorded lower-connection rejection sites.")
            return 0

        by_layer: Dict[int, List[Dict]] = {}
        for item in details:
            by_layer.setdefault(int(item["layer_idx"]), []).append(item)

        repair_rows: List[Dict] = []
        total_added = 0
        for layer_idx in sorted(by_layer):
            if not (0 <= layer_idx < n_layers):
                continue
            rejects = by_layer[layer_idx]
            target_total = target_nodes_for_layer(layer_idx)
            rejection_count = len(rejects)
            rejection_fraction = rejection_count / max(1, int(target_total))
            if (rejection_count < int(INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_REJECTIONS)
                    or rejection_fraction < float(INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_FRACTION)):
                continue

            current = all_atoms[all_atoms["atom_id"].isin(internal_ids)][["x", "y", "z"]].to_numpy(float)
            tree = cKDTree(current) if len(current) else None
            scored = []
            for item in rejects:
                xyz = np.asarray([float(item["x_mm"]), float(item["y_mm"]), float(item["z_mm"])], dtype=float)
                score = float(tree.query(xyz, k=1)[0]) if tree is not None else float("inf")
                scored.append((score, item))
            scored.sort(key=lambda pair: pair[0], reverse=True)

            pitch = _local_internal_pitch(0.5 * (layer_z[layer_idx] + layer_z[layer_idx + 1]), z_bounds)
            dedup = max(1.0e-6, float(INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_DEDUP_FACTOR) * pitch)
            selected: List[Dict] = []
            for _, item in scored:
                xyz = np.asarray([float(item["x_mm"]), float(item["y_mm"]), float(item["z_mm"])], dtype=float)
                if any(np.linalg.norm(
                        xyz - np.asarray([float(s["x_mm"]), float(s["y_mm"]), float(s["z_mm"])], dtype=float)) < dedup
                       for s in selected):
                    continue
                selected.append(item)
                if len(selected) >= int(INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MAX_TARGETS_PER_LAYER):
                    break

            log(f"  Initial support-band repair: layer {layer_idx + 1}/{n_layers}, "
                f"rejections={rejection_count}, fraction={rejection_fraction:.3f}, "
                f"targets={len(selected)}")
            for item in selected:
                z_try = float(np.clip(float(item["z_mm"]),
                                      float(layer_z[layer_idx]) + 1.0e-6,
                                      float(layer_z[layer_idx + 1]) - 1.0e-6))
                domain_try = _domain_at_z_cached(mesh, z_try, domain_cache)
                point = Point(float(item["x_mm"]), float(item["y_mm"]))
                if domain_try.is_empty or not domain_try.covers(point):
                    repair_rows.append({**item, "trigger_rejections": rejection_count,
                                        "trigger_fraction": rejection_fraction,
                                        "accepted": False, "created_nodes": 0,
                                        "path_type": "rejected_site_outside_current_domain"})
                    continue
                before_ids = len(internal_ids)
                ok, nid = add_node_to_layer(
                    float(layer_z[layer_idx]), float(layer_z[layer_idx + 1]), layer_idx,
                    force_insert=True, label="INITIAL_SUPPORT_REPAIR",
                    specific_z=z_try,
                    specific_xy=(float(item["x_mm"]), float(item["y_mm"])),
                    allow_unanchored=False, domain_override=domain_try,
                    allow_layer_peer_without_lower=True)
                added_now = max(0, len(internal_ids) - before_ids) if ok else 0
                total_added += added_now
                created = [int(nid)] if ok and nid is not None else []
                repair_rows.append({**item, "trigger_rejections": rejection_count,
                                    "trigger_fraction": rejection_fraction,
                                    "accepted": bool(ok), "target_node_id": nid,
                                    "created_nodes": int(added_now),
                                    "created_node_ids": ";".join(map(str, created)),
                                    "path_type": "same_layer_peer_relax" if ok else "no_legal_same_layer_peer"})

        pd.DataFrame(repair_rows).to_csv(
            output_dir / "initial_support_repair_details.csv", index=False)
        if repair_rows:
            accepted_targets = sum(bool(r.get("accepted")) for r in repair_rows)
            log(f"  Initial support-band repair finished: accepted_targets={accepted_targets}/"
                f"{len(repair_rows)}, added_nodes={total_added}.")
            if total_added:
                reconciled = reconcile_cross_layer_edges()
                log(f"  Initial support-band repair reconciliation added {reconciled} edges.")
            process_log.append({
                "step": int(global_step), "layer_idx": -1,
                "phase": "STEP1_3_PERSISTENT_LOWER_CONNECTION_REPAIR",
                "attempted_targets": int(len(repair_rows)),
                "accepted_targets": int(accepted_targets),
                "added_nodes": int(total_added),
                "peer_relaxed_layers": ";".join(str(i + 1) for i in sorted(initial_peer_relaxed_layers)),
            })
        else:
            log("  Initial support-band repair: no layer exceeded the adaptive trigger.")
        return int(total_added)

    repair_persistent_initial_lower_connection_bands()

    if INITIAL_SKELETON_CHECKPOINT_SAVE is not None:
        _save_internal_checkpoint(
            INITIAL_SKELETON_CHECKPOINT_SAVE, "initial-skeleton", mesh, int(start_id),
            all_atoms=all_atoms, all_edges=all_edges, internal_ids=internal_ids,
            supported_internal_ids=supported_internal_ids, layer_nodes=layer_nodes,
            next_id=next_id, global_step=global_step, process_log=process_log,
            internal_log=internal_log, initial_rejection_counts=initial_rejection_counts,
            initial_rejection_details=initial_rejection_details)
        if CHECKPOINT_STOP_AFTER == "initial-skeleton":
            raise WorkflowCheckpointStop("initial-skeleton checkpoint saved")

    def export_snapshot(name: str):
        df = _pressure_profile_stl(mesh, all_atoms, all_edges, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM,
                                   cache=domain_cache)
        if not df.empty:
            band_mask = df["z_mid_mm"] >= float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM)
            df["P_cap_target_lower_Pa"] = (
                    _capillary_lower_ratio_values(
                        df, include_assembly_reserve=True)
                    * df["P_required_Pa"].to_numpy(dtype=float)
            )
            df["P_cap_target_upper_Pa"] = np.where(
                band_mask,
                float(INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO) * df["P_required_Pa"],
                np.nan,
            )
            df["Lower_Band_Margin_Pa"] = df["P_cap_Pa"] - df["P_cap_target_lower_Pa"]
            cols = ["z_mid_mm", "P_wet_mm", "P_cap_Pa", "P_hydro_Pa", "safety_factor_z", "P_visc_dynamic_Pa",
                    "P_visc_ref_Pa", "P_viscous_Pa", "P_viscous_capacity_proxy_Pa", "P_required_base_Pa",
                    "P_required_Pa", "transport_margin_Pa", "R_tr_Pa", "P_cap_target_lower_Pa", "P_cap_target_upper_Pa",
                    "Safety_Margin", "Lower_Band_Margin_Pa", "r_eff_mm", "A_liq_mm2", "porosity",
                    "axial_radius4_sum_mm4", "n_flow_components", "R_geom_segment_m-3", "R_geom_cumulative_m-3",
                    "fluid_viscosity_Pa_s", "reference_flow_rate_m3_s"]
            df[[c for c in cols if c in df.columns]].to_csv(output_dir / name, index=False)

    if loaded_checkpoint_stage != "pre-smartfill":
        export_snapshot("physics_snapshot_after_initial.csv")

    def physics_gapfill_pass() -> int:
        if loaded_checkpoint_stage == "pre-smartfill":
            return 0
        if not INTERNAL_DYNAMIC_PHYS_GAPFILL_ENABLE:
            return 0
        start_z = zmin + total_h * INTERNAL_DYNAMIC_PHYS_GAPFILL_MIN_Z_FRAC
        df = _pressure_profile_stl(mesh, all_atoms, all_edges, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM,
                                   cache=domain_cache, start_z=start_z)
        if df.empty:
            return 0
        df["porosity_floor"] = df["z_mm"].map(lambda z: _porosity_floor_at_z(float(z), z_bounds))
        df["lower_margin_Pa"] = _capillary_margin_values(df)
        df["deficit_Pa"] = np.clip(-df["lower_margin_Pa"], 0.0, None)
        lower_ratio = _capillary_lower_ratio_values(
            df, include_assembly_reserve=True)
        df["rel_def"] = df["deficit_Pa"] / np.maximum(
            lower_ratio * df["P_required_Pa"].to_numpy(dtype=float), 1.0e-9
        )
        fail = df[df["lower_margin_Pa"] < 0.0].sort_values("z_mm").reset_index(drop=True)
        fail = fail[
            (fail["porosity"] > fail["porosity_floor"])
            & (fail["A_liq_mm2"] > INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2)
            ].reset_index(drop=True)
        if fail.empty:
            log("  Physics gapfill: no failing intervals with available porosity.")
            return 0

        dz = float(INTERNAL_DYNAMIC_PHYSICS_DZ_MM)
        z_vals = fail["z_mm"].to_numpy(dtype=float)
        intervals, s = [], 0
        for i in range(1, len(z_vals)):
            if (z_vals[i] - z_vals[i - 1]) > 1.05 * dz:
                intervals.append((s, i - 1))
                s = i
        intervals.append((s, len(z_vals) - 1))

        bottom_density = target_nodes_for_layer(0) / max(total_h / max(n_layers, 1), 1e-9)
        total_added = 0
        for i0, i1 in intervals:
            block = fail.iloc[i0:i1 + 1].copy()
            rel = float(np.clip(block["rel_def"].mean(), 0.0, INTERNAL_DYNAMIC_PHYS_GAPFILL_RELDEF_CAP))
            n_need = int(
                math.ceil(bottom_density * max(dz, len(block) * dz) * (INTERNAL_DYNAMIC_PHYS_GAPFILL_BASE_BIAS + rel)))
            n_need = int(np.clip(n_need, 1, INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_ADDS_PER_INTERVAL))
            block = _gradient_repair_schedule(block, df, max(2, n_need), {}, z_bounds)
            band_caps = block["adaptive_add_cap"].to_numpy(dtype=int)
            band_budget = min(max(len(block), n_need), int(np.sum(band_caps)))
            alloc = _smartfill_weighted_allocation(
                block["weight"].to_numpy(dtype=float), band_budget, band_caps)
            for row_idx, n_add in enumerate(alloc):
                comp_candidates_cache = None
                row = block.iloc[row_idx]
                lo = float(row["insertion_cell_lo_mm"])
                hi = float(row["insertion_cell_hi_mm"])
                phases = ((np.arange(int(n_add), dtype=float) + 0.5)
                          / max(1, int(n_add)))
                z_band_targets = lo + phases * max(hi - lo, 0.0)
                for add_idx in range(int(n_add)):
                    z_try = float(z_band_targets[add_idx])
                    if float(row["porosity"]) <= _porosity_floor_at_z(z_try, z_bounds):
                        continue
                    li = int(np.clip(np.searchsorted(layer_z, z_try, side="right") - 1, 0, n_layers - 1))
                    comp_candidates_cache = (
                        _top_pore_candidates(mesh, domain_cache, all_atoms, all_edges,
                                             z_bounds, z_try)
                        if INTERNAL_MATCHED_RANDOM_POSITION else
                        _component_worst_pore_candidates(mesh, domain_cache, all_atoms, all_edges,
                                                         z_bounds, z_try)
                    )
                    if not comp_candidates_cache:
                        continue
                    cand = (comp_candidates_cache[int(position_rng.integers(len(comp_candidates_cache)))]
                            if INTERNAL_MATCHED_RANDOM_POSITION else
                            comp_candidates_cache[(total_added + int(add_idx)) % len(comp_candidates_cache)])
                    xy = cand["xy"]
                    comp_domain = cand["domain"]
                    ok, nid = add_node_to_layer(
                        float(layer_z[li]), float(layer_z[li + 1]), li, force_insert=True,
                        label=f"SMART_GAP3_C{cand['component']}", specific_z=z_try,
                        specific_xy=xy, domain_override=comp_domain)
                    if ok:
                        layer_nodes[li].append(int(nid))
                        total_added += 1
        log(f"  Physics gapfill added {total_added} nodes.")
        return int(total_added)

    if (loaded_checkpoint_stage != "pre-smartfill"
            and BOTTOM_PILLAR_ENABLE and internal_ids):
        old_internal_ids = set(internal_ids)
        core_atoms = all_atoms[all_atoms["atom_id"].astype(int).isin(old_internal_ids)].copy().reset_index(drop=True)
        core_edges = pd.DataFrame([
            dict(edge) for edge in all_edges
            if str(edge.get("bond_type", "")) == "internal"
               and int(edge.get("source", -1)) in old_internal_ids
               and int(edge.get("target", -1)) in old_internal_ids
        ])
        core_with_pillars, core_edges_with_pillars, next_id = add_bottom_print_pillars(
            mesh, core_atoms, core_edges, int(next_id))
        pillar_ids = (set(core_with_pillars["atom_id"].astype(int).tolist())
                      - old_internal_ids)
        if pillar_ids:
            carrier_atoms = all_atoms[
                ~all_atoms["atom_id"].astype(int).isin(old_internal_ids)
            ].copy()
            all_atoms = pd.concat([carrier_atoms, core_with_pillars], ignore_index=True)
            all_edges = [
                            dict(edge) for edge in all_edges
                            if str(edge.get("bond_type", "")) != "internal"
                        ] + _edge_records(core_edges_with_pillars)
            internal_ids.update(pillar_ids)
            for row in core_with_pillars[
                core_with_pillars["atom_id"].astype(int).isin(pillar_ids)
            ].itertuples(index=False):
                li = int(np.clip(
                    np.searchsorted(layer_z, float(row.z), side="right") - 1,
                    0, n_layers - 1))
                layer_nodes[li].append(int(row.atom_id))
            support_after_pillars = _internal_support_state(
                core_with_pillars, core_edges_with_pillars, z_bounds=z_bounds)
            supported_internal_ids.clear()
            supported_internal_ids.update(
                set(map(int, support_after_pillars.get("supported", set()))))
            process_log.append({
                "step": int(global_step),
                "layer_idx": -1,
                "phase": "STEP3_BOTTOM_PRINT_PILLARS_BEFORE_PRESSURE_REPAIR",
                "added_nodes": int(len(pillar_ids)),
                "added_edges": int(max(0, len(core_edges_with_pillars) - len(core_edges))),
            })
            log(
                f"  Bottom pillars integrated before pressure repair: "
                f"added_nodes={len(pillar_ids)}, "
                f"pre-stitch lower ratio="
                f"{INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO + INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO:.3f}."
            )

    df_gap_before = _pressure_profile_stl(
        mesh, all_atoms, all_edges, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache
    )
    atoms_gap_before = all_atoms.copy()
    edges_gap_before = list(all_edges)
    ids_gap_before = set(internal_ids)
    layers_gap_before = [list(nodes) for nodes in layer_nodes]
    supported_gap_before = set(supported_internal_ids)
    next_id_gap_before = int(next_id)
    global_step_gap_before = int(global_step)
    added_gap = physics_gapfill_pass()
    if added_gap > 0:
        reconcile_cross_layer_edges()
        df_gap_after = _pressure_profile_stl(
            mesh, all_atoms, all_edges, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache
        )
        gap_heights = all_atoms.loc[
            all_atoms['atom_id'].isin(internal_ids - ids_gap_before), 'z'
        ].round(6).nunique()
        if gap_heights < 2 or not _profile_trial_improves(df_gap_before, df_gap_after):
            all_atoms = atoms_gap_before
            all_edges = edges_gap_before
            internal_ids = ids_gap_before
            layer_nodes = layers_gap_before
            supported_internal_ids.clear()
            supported_internal_ids.update(supported_gap_before)
            next_id = next_id_gap_before
            global_step = global_step_gap_before
            added_gap = 0
            log("  Physics gapfill rolled back: insufficient height coverage or no safe pressure gain.")
    process_log.append(
        {"step": int(global_step), "layer_idx": -1, "phase": "STEP3_PHYS_GAPFILL", "added_nodes": int(added_gap)})
    if loaded_checkpoint_stage != "pre-smartfill":
        export_snapshot("physics_snapshot_after_gapfill.csv")

    support_edges_added = (repair_downward_support(int(INTERNAL_DYNAMIC_SUPPORT_REPAIR_MAX_EDGES))
                           if (INTERNAL_REQUIRE_BOTTOM_SUPPORT
                               and loaded_checkpoint_stage != "pre-smartfill") else 0)
    invalid_edges_removed = (remove_invalid_internal_edges()
                             if loaded_checkpoint_stage != "pre-smartfill" else 0)

    if INTERNAL_REQUIRE_BOTTOM_SUPPORT and loaded_checkpoint_stage != "pre-smartfill":
        support_edges_added += repair_downward_support(
            max(0, int(INTERNAL_DYNAMIC_SUPPORT_REPAIR_MAX_EDGES) - support_edges_added))
        unsupported_nodes_removed = remove_unrepairable_unsupported_nodes()
    else:
        unsupported_nodes_removed = 0
    support_now = _internal_support_state(all_atoms, all_edges, z_bounds=z_bounds)
    log(f"  Pre-SmartFill support repair: added_strict_support_edges={support_edges_added}, "
        f"removed_invalid_edges={invalid_edges_removed}, "
        f"removed_unrepairable_nodes={unsupported_nodes_removed}, "
        f"unsupported_remaining={len(support_now['nodes'] - support_now['supported'])}, "
        f"rejected_remaining={sum(v == 'rejected' for v in support_now['classes'].values())}, "
        f"bridge_bad_remaining={len(support_now['bridge_bad'])}.")
    ok, crit_z, p_cap, p_hydro, df_smart = _capillary_ok_stl(mesh, all_atoms, all_edges, z_bounds,
                                                             INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache)
    if loaded_checkpoint_stage == "pre-smartfill" and has_surface_prefix:

        internal_atoms_for_compare = all_atoms[
            all_atoms["atom_id"].isin(internal_ids)
        ].copy().reset_index(drop=True)
        internal_edges_for_compare = [
            dict(edge) for edge in all_edges
            if int(edge.get("source", -1)) in internal_ids
               and int(edge.get("target", -1)) in internal_ids
        ]
        df_without_surface = _pressure_profile_stl(
            mesh, internal_atoms_for_compare, internal_edges_for_compare,
            z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache)
        compare_cols = [
            "z_mm", "r_eff_mm", "P_wet_mm", "P_cap_Pa", "P_hydro_Pa",
            "P_visc_dynamic_Pa", "P_viscous_Pa", "P_required_Pa", "porosity"
        ]
        base = df_without_surface[[c for c in compare_cols
                                   if c in df_without_surface.columns]].copy()
        surf = df_smart[[c for c in compare_cols if c in df_smart.columns]].copy()
        comparison = base.merge(surf, on="z_mm", how="inner",
                                suffixes=("_without_surface", "_with_surface"))
        for metric in ("r_eff_mm", "P_wet_mm", "P_cap_Pa", "P_hydro_Pa",
                       "P_visc_dynamic_Pa", "P_viscous_Pa", "P_required_Pa", "porosity"):
            before_col = f"{metric}_without_surface"
            after_col = f"{metric}_with_surface"
            if before_col in comparison.columns and after_col in comparison.columns:
                comparison[f"delta_{metric}"] = (
                        comparison[after_col] - comparison[before_col])
        for suffix in ("without_surface", "with_surface"):
            cap_col, req_col = f"P_cap_Pa_{suffix}", f"P_required_Pa_{suffix}"
            if cap_col in comparison.columns and req_col in comparison.columns:
                comparison[f"pressure_margin_Pa_{suffix}"] = (
                        comparison[cap_col] - comparison[req_col])
        if {"pressure_margin_Pa_without_surface", "pressure_margin_Pa_with_surface"}.issubset(
                comparison.columns):
            comparison["delta_pressure_margin_Pa"] = (
                    comparison["pressure_margin_Pa_with_surface"]
                    - comparison["pressure_margin_Pa_without_surface"])
        comparison.to_csv(
            output_dir / "surface_pressure_contribution_pre_smartfill.csv", index=False)
        cap_delta = comparison.get("delta_P_cap_Pa", pd.Series(dtype=float))
        margin_delta = comparison.get("delta_pressure_margin_Pa", pd.Series(dtype=float))
        base_margin = comparison.get(
            "pressure_margin_Pa_without_surface", pd.Series(dtype=float))
        surf_margin = comparison.get(
            "pressure_margin_Pa_with_surface", pd.Series(dtype=float))
        summary = {
            "comparison_stage": "pre-smartfill checkpoint, before any new repair",
            "surface_atoms": int(len(surface_atoms_prefix)),
            "surface_edges": int(len(surface_edges_prefix)),
            "height_slices_compared": int(len(comparison)),
            "mean_capillary_change_Pa": float(cap_delta.mean()) if len(cap_delta) else None,
            "max_capillary_gain_Pa": float(cap_delta.max()) if len(cap_delta) else None,
            "min_capillary_change_Pa": float(cap_delta.min()) if len(cap_delta) else None,
            "mean_pressure_margin_change_Pa": float(margin_delta.mean()) if len(margin_delta) else None,
            "worst_margin_without_surface_Pa": float(base_margin.min()) if len(base_margin) else None,
            "worst_margin_with_surface_Pa": float(surf_margin.min()) if len(surf_margin) else None,
            "passing_slices_without_surface": int((base_margin >= 0.0).sum()) if len(base_margin) else None,
            "passing_slices_with_surface": int((surf_margin >= 0.0).sum()) if len(surf_margin) else None,
        }
        (output_dir / "surface_pressure_contribution_pre_smartfill.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log("  Surface pressure contribution recorded before SmartFill: "
            f"mean dP_cap={summary['mean_capillary_change_Pa']:.3f} Pa, "
            f"worst-margin change="
            f"{summary['worst_margin_with_surface_Pa'] - summary['worst_margin_without_surface_Pa']:.3f} Pa.")
    if loaded_checkpoint_stage == "pre-smartfill":
        channel_removed, channel_evaluations = 0, 0
        log("  Coupled channel repair bypassed by pre-smartfill checkpoint.")
    else:
        df_smart, channel_removed, channel_evaluations = optimise_high_resistance_channel(
            df_smart, int(INTERNAL_DYNAMIC_CHANNEL_REPAIR_MAX_EVALUATIONS))
    if channel_evaluations:
        log(f"  Coupled channel repair: removed={channel_removed}, "
            f"exact_evaluations={channel_evaluations}, "
            f"remaining_deficit={_capillary_safety_rank(df_smart)[1]:.3f} Pa.")
    ok = _capillary_profile_passes(df_smart)
    if PRE_SMARTFILL_CHECKPOINT_SAVE is not None:
        _save_internal_checkpoint(
            PRE_SMARTFILL_CHECKPOINT_SAVE, "pre-smartfill", mesh, int(start_id),
            all_atoms=all_atoms, all_edges=all_edges, internal_ids=internal_ids,
            supported_internal_ids=supported_internal_ids, layer_nodes=layer_nodes,
            next_id=next_id, global_step=global_step, process_log=process_log,
            internal_log=internal_log, initial_rejection_counts=initial_rejection_counts,
            initial_rejection_details=initial_rejection_details)
        if CHECKPOINT_STOP_AFTER == "pre-smartfill":
            raise WorkflowCheckpointStop("pre-smartfill checkpoint saved")
    smart_round = 0
    smart_total_added = 0
    smoothing_rounds = 0
    adaptive_round_budget = max(2, int(INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND))
    gain_per_node_ema: Optional[float] = None
    single_trial_failures: Dict[float, int] = {}
    slice_trial_counts: Dict[float, int] = {}
    slice_accepted_counts: Dict[float, int] = {}
    blocked_slices: Set[float] = set()
    blocked_failure_bands: List[Tuple[float, float]] = []
    low_efficiency_streaks: Dict[float, int] = {}
    viscous_edge_rejected_by_band: Dict[float, Set[Tuple[int, int]]] = {}

    viscous_relief_failures: Dict[float, int] = {}
    viscous_relief_removed_edges = 0
    viscous_relief_evaluations = 0
    peak_gain_per_node = 0.0
    smart_pressure_evaluations = 0
    salvage_summary = {"rejected_batches": 0, "salvaged_batches": 0,
                       "salvaged_nodes": 0, "candidate_evaluations": 0,
                       "candidate_pressure_rejections": 0, "candidate_support_rejections": 0}
    smartfill_pressure_cache_stats: Dict = {
        "incremental_trials": 0, "full_fallback_trials": 0,
        "section_rows_reused": 0, "section_rows_recomputed": 0,
        "trial_pressure_seconds": 0.0,
    }
    smartfill_status = "running"
    near_pass_gain_logged = False

    rejected_top_bridge_pairs: Set[Tuple[int, int]] = set()
    top_bridge_trial_counts: Dict[float, int] = {}
    top_excluded_positions: Dict[float, List[Tuple[float, float]]] = {}
    move_node_trials: Dict[float, int] = {}
    thicken_edge_trials: Dict[float, int] = {}
    smartfill_local_geometry_log: List[Dict] = []

    def _local_geometry_profile_acceptable(before_profile, after_profile, z_target):
        if after_profile is None or after_profile.empty:
            return False, 'empty_pressure_profile'
        if not _smartfill_local_safe_gain(
                before_profile, after_profile, insertion_trial=True):
            return False, 'global_pressure_gain_insufficient'
        for row in after_profile.itertuples(index=False):
            z_row = float(getattr(row, 'z_mm'))
            if float(getattr(row, 'porosity')) < _porosity_floor_at_z(z_row, z_bounds) - 1.0e-9:
                return False, 'porosity_floor'
        before_row = before_profile.iloc[
            (before_profile['z_mm'] - float(z_target)).abs().argmin()]
        after_row = after_profile.iloc[
            (after_profile['z_mm'] - float(z_target)).abs().argmin()]
        local_cap_gain = float(after_row['P_cap_Pa']) - float(before_row['P_cap_Pa'])
        if local_cap_gain <= 1.0e-9 and not _capillary_profile_passes(after_profile):
            return False, 'target_capillary_pressure_not_improved'
        diag_after = _pressure_failure_mechanism(after_profile, float(z_target))
        if (float(diag_after.get('viscous_share', 0.0))
                > float(INTERNAL_DYNAMIC_LOCAL_REPAIR_MAX_VISCOUS_SHARE)
                and not _capillary_profile_passes(after_profile)):
            return False, 'viscous_share_limit'
        return True, 'accepted'

    def try_move_node_toward_worst_pore(z_target, profile_before, state_key):
        """Move one existing node toward the largest pore and commit only exact pressure gain."""
        nonlocal all_atoms
        info = {'action': 'MOVE_NODE', 'z_mm': float(z_target), 'accepted': False}
        if not INTERNAL_DYNAMIC_LOCAL_GEOMETRY_REPAIR_ENABLE:
            info['reason'] = 'disabled';
            return profile_before, False, 0, info
        used = int(move_node_trials.get(state_key, 0))
        limit = int(INTERNAL_DYNAMIC_MOVE_NODE_MAX_TRIALS_PER_BAND)
        if used >= limit:
            info['reason'] = 'band_trial_limit';
            return profile_before, False, 0, info
        pores = _component_worst_pore_candidates(
            mesh, domain_cache, all_atoms, all_edges, z_bounds, float(z_target))
        if not pores:
            move_node_trials[state_key] = used + 1
            info['reason'] = 'no_pore';
            return profile_before, False, 0, info
        pore = (pores[int(position_rng.integers(len(pores)))]
                if INTERNAL_MATCHED_RANDOM_POSITION else
                max(pores, key=lambda item: float(item['score'])))
        target_xy = np.asarray(pore['xy'], dtype=float)
        state = _internal_support_state(all_atoms, all_edges, z_bounds=z_bounds)
        candidates = all_atoms[
            all_atoms['atom_id'].astype(int).isin(internal_ids)
            & ((all_atoms['z'].astype(float) - float(z_target)).abs()
               <= 0.65 * float(INTERNAL_DYNAMIC_PHYSICS_DZ_MM) + 1.0e-9)
            ].copy()
        if 'peer_only_relaxed' in candidates.columns:
            candidates = candidates[~candidates['peer_only_relaxed'].fillna(False).astype(bool)]
        candidates = candidates[
            ~candidates['face_name'].astype(str).str.contains('bottom_pillar', case=False, regex=False)]
        if candidates.empty:
            move_node_trials[state_key] = used + 1
            info['reason'] = 'no_movable_node';
            return profile_before, False, 0, info
        candidates['_pore_distance'] = np.hypot(
            candidates['x'].astype(float) - target_xy[0],
            candidates['y'].astype(float) - target_xy[1])
        candidates['_degree'] = candidates['atom_id'].astype(int).map(state['degree']).fillna(0)
        candidates = candidates[(candidates['_degree'] >= 2) & (candidates['_degree'] <= 8)]
        candidates = candidates.sort_values(['_pore_distance', '_degree']).head(
            int(INTERNAL_DYNAMIC_MOVE_NODE_MAX_CANDIDATES_PER_SEARCH))
        coord = {int(r.atom_id): np.asarray([r.x, r.y, r.z], dtype=float)
                 for r in all_atoms.itertuples(index=False)}
        evaluations = 0
        for cand in candidates.itertuples(index=False):
            if int(move_node_trials.get(state_key, 0)) >= limit:
                break
            move_node_trials[state_key] = int(move_node_trials.get(state_key, 0)) + 1
            nid = int(cand.atom_id)
            old = coord[nid].copy()
            delta_xy = target_xy - old[:2]
            distance = float(np.linalg.norm(delta_xy))
            if distance <= 1.0e-8:
                continue
            max_move = (float(INTERNAL_DYNAMIC_MOVE_NODE_MAX_DISTANCE_PITCH_FACTOR)
                        * _local_internal_pitch(float(old[2]), z_bounds))
            step = min(max_move, 0.50 * distance)
            moved = old.copy()
            moved[:2] += delta_xy / distance * step
            domain = _domain_at_z_cached(mesh, float(old[2]), domain_cache)
            if domain.is_empty or not domain.covers(Point(float(moved[0]), float(moved[1]))):
                continue
            others = np.asarray([xyz for ident, xyz in coord.items() if ident != nid], dtype=float)
            if len(others):
                nearest = float(cKDTree(others).query(moved, k=1)[0])
                if nearest < (_local_internal_pitch(float(old[2]), z_bounds)
                              * float(INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR)):
                    continue
            trial_atoms = all_atoms.copy()
            mask = trial_atoms['atom_id'].astype(int) == nid
            trial_atoms.loc[mask, ['x', 'y']] = [float(moved[0]), float(moved[1])]
            trial_coord = dict(coord);
            trial_coord[nid] = moved
            geometry_ok = True
            layer_h = total_h / max(1, n_layers)
            for edge in all_edges:
                u, v = int(edge.get('source', -1)), int(edge.get('target', -1))
                if nid not in (u, v) or u not in trial_coord or v not in trial_coord:
                    continue
                a, b = trial_coord[u], trial_coord[v]
                max_len = max(layer_h, _local_internal_pitch(
                    0.5 * (float(a[2]) + float(b[2])), z_bounds)) * float(
                    INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR)
                if (float(np.linalg.norm(b - a)) > max_len + 1.0e-9
                        or _internal_edge_class_with_top_exception(a, b, edge, z_bounds) == 'rejected'
                        or not _edge_centerline_stays_in_domain(
                            mesh, domain_cache, tuple(a), tuple(b), samples=5)):
                    geometry_ok = False
                    break
            if not geometry_ok:
                continue
            if not _deletion_preserves_internal_support(
                    all_atoms, all_edges, trial_atoms, all_edges, z_bounds=z_bounds):
                continue
            abnormal = audit_abnormal_layer_connectivity(
                trial_atoms, all_edges, z_bounds=z_bounds)
            if not abnormal['abnormal_layer_connectivity_pass']:
                continue
            result = _smartfill_trial_capillary_ok_stl(
                mesh, trial_atoms, all_edges, all_atoms, all_edges, profile_before,
                z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, domain_cache,
                stats=smartfill_pressure_cache_stats)
            evaluations += 1
            accepted, reason = _local_geometry_profile_acceptable(
                profile_before, result[4], z_target)
            event = {**info, 'node_id': nid,
                     'old_x_mm': float(old[0]), 'old_y_mm': float(old[1]),
                     'new_x_mm': float(moved[0]), 'new_y_mm': float(moved[1]),
                     'move_distance_mm': float(step), 'reason': reason,
                     'deficit_before_Pa': float(_capillary_safety_rank(profile_before)[1]),
                     'deficit_after_Pa': float(_capillary_safety_rank(result[4])[1]),
                     'accepted': bool(accepted)}
            smartfill_local_geometry_log.append(event)
            if accepted:
                all_atoms = trial_atoms
                return result[4], True, evaluations, event
        info['reason'] = 'no_acceptable_move'
        return profile_before, False, evaluations, info

    def try_thicken_edge_near_worst_pore(z_target, profile_before, state_key):
        """Increase one local strut radius in a bounded transaction."""
        nonlocal all_edges
        info = {'action': 'THICKEN_EDGE', 'z_mm': float(z_target), 'accepted': False}
        if not INTERNAL_DYNAMIC_LOCAL_GEOMETRY_REPAIR_ENABLE:
            info['reason'] = 'disabled';
            return profile_before, False, 0, info
        used = int(thicken_edge_trials.get(state_key, 0))
        limit = int(INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_TRIALS_PER_BAND)
        if used >= limit:
            info['reason'] = 'band_trial_limit';
            return profile_before, False, 0, info
        pores = _component_worst_pore_candidates(
            mesh, domain_cache, all_atoms, all_edges, z_bounds, float(z_target))
        if not pores:
            thicken_edge_trials[state_key] = used + 1
            info['reason'] = 'no_pore';
            return profile_before, False, 0, info
        pore = (pores[int(position_rng.integers(len(pores)))]
                if INTERNAL_MATCHED_RANDOM_POSITION else
                max(pores, key=lambda item: float(item['score'])))
        point = Point(float(pore['xy'][0]), float(pore['xy'][1]))
        coord = {int(r.atom_id): np.asarray([r.x, r.y, r.z], dtype=float)
                 for r in all_atoms.itertuples(index=False)}
        candidates = []
        for idx, edge in enumerate(all_edges):
            if str(edge.get('bond_type', '')) != 'internal':
                continue
            if str(edge.get('role', '')).startswith('bottom_pillar'):
                continue
            u, v = int(edge.get('source', -1)), int(edge.get('target', -1))
            if u not in coord or v not in coord:
                continue
            a, b = coord[u], coord[v]
            radius = float(edge.get('radius', _local_internal_radius(z_target, z_bounds)))
            if not (min(float(a[2]), float(b[2])) - radius <= float(z_target)
                    <= max(float(a[2]), float(b[2])) + radius):
                continue
            cap = (float(INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_RADIUS_FACTOR)
                   * _local_internal_radius(0.5 * (float(a[2]) + float(b[2])), z_bounds))
            if radius >= cap - 1.0e-9:
                continue
            distance = float(LineString([
                (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
            ]).distance(point))
            candidates.append((distance, -float(np.linalg.norm(b - a)), idx, cap))
        candidates.sort()
        evaluations = 0
        short_list = candidates[:int(INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_CANDIDATES_PER_SEARCH)]

        for batch_size in (3, 2, 1):
            chosen = short_list[:min(batch_size, len(short_list))]
            if not chosen:
                continue
            if int(thicken_edge_trials.get(state_key, 0)) >= limit:
                break
            thicken_edge_trials[state_key] = int(thicken_edge_trials.get(state_key, 0)) + 1
            trial_edges = [dict(edge) for edge in all_edges]
            changed = []
            for _distance, _neg_len, idx, cap in chosen:
                old_radius = float(trial_edges[idx].get('radius', 0.0))
                new_radius = min(float(cap), old_radius * float(
                    INTERNAL_DYNAMIC_THICKEN_EDGE_STEP_FACTOR))
                if new_radius <= old_radius + 1.0e-9:
                    continue
                trial_edges[idx]['radius'] = float(new_radius)
                trial_edges[idx]['local_thickening_factor'] = float(new_radius / old_radius)
                changed.append((idx, float(_distance), old_radius, float(new_radius)))
            if not changed:
                continue
            result = _smartfill_trial_capillary_ok_stl(
                mesh, all_atoms, trial_edges, all_atoms, all_edges, profile_before,
                z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, domain_cache,
                stats=smartfill_pressure_cache_stats)
            evaluations += 1
            accepted, reason = _local_geometry_profile_acceptable(
                profile_before, result[4], z_target)
            primary_idx, primary_distance, old_radius, new_radius = changed[0]
            edge = trial_edges[primary_idx]
            event = {**info, 'source': int(edge.get('source', -1)),
                     'target': int(edge.get('target', -1)),
                     'old_radius_mm': old_radius, 'new_radius_mm': float(new_radius),
                     'pore_distance_mm': float(primary_distance),
                     'edge_count': int(len(changed)),
                     'edges': ';'.join(
                         f"{int(trial_edges[i].get('source', -1))}-{int(trial_edges[i].get('target', -1))}"
                         for i, _d, _old, _new in changed),
                     'reason': reason,
                     'deficit_before_Pa': float(_capillary_safety_rank(profile_before)[1]),
                     'deficit_after_Pa': float(_capillary_safety_rank(result[4])[1]),
                     'accepted': bool(accepted)}
            smartfill_local_geometry_log.append(event)
            if accepted:
                all_edges = trial_edges
                return result[4], True, evaluations, event
        info['reason'] = 'no_acceptable_thickening'
        return profile_before, False, evaluations, info

    while True:
        safety_mode = not _capillary_profile_passes(df_smart)
        insertion_gain_factor = _smartfill_insertion_gain_factor(df_smart)
        if insertion_gain_factor < 1.0 and not near_pass_gain_logged:
            transport_values = pd.to_numeric(
                df_smart.get("transport_margin_Pa", pd.Series(dtype=float)),
                errors="coerce").to_numpy(float)
            finite_transport = transport_values[np.isfinite(transport_values)]
            min_transport = (float(np.min(finite_transport))
                             if len(finite_transport) else float("nan"))
            log(
                f"  SmartFill insertion gain threshold scaled to "
                f"{insertion_gain_factor:.3f} after minimum transport margin reached "
                f"{min_transport:.3f} Pa (trigger "
                f"{INTERNAL_DYNAMIC_SMARTFILL_HALF_GAIN_MARGIN_PA:.3f} Pa).")
            near_pass_gain_logged = True
        if not safety_mode:
            smartfill_status = "passed_feedback_pending_final_fine_audit"
            log("  SmartFill feedback pressure passed; final fine-slice audit remains mandatory.")
            break
        max_pressure_evals = int(INTERNAL_DYNAMIC_SMARTFILL_MAX_PRESSURE_EVALUATIONS)
        if max_pressure_evals > 0 and smart_pressure_evaluations >= max_pressure_evals:
            smartfill_status = "search_budget_exhausted"
            log(
                f"  SmartFill stopped: computational pressure-evaluation budget "
                f"exhausted ({smart_pressure_evaluations}/{max_pressure_evals}). "
                "Unresolved slices remain physically unresolved; this stop is not a pass/fail reclassification."
            )
            break
        smart_round += 1
        smooth_rows = _profile_smooth_deficits(df_smart)
        smooth_cols = ["z_mm", "smooth_target_Pa", "smooth_deficit_Pa", "smooth_rel_def"]
        smooth_cols = [col for col in smooth_cols if col in smooth_rows.columns]
        work_smart = df_smart.merge(
            smooth_rows[smooth_cols], on="z_mm", how="left"
        ) if smooth_cols else df_smart.copy()
        for col in ("smooth_target_Pa", "smooth_deficit_Pa", "smooth_rel_def"):
            if col not in work_smart.columns:
                work_smart[col] = 0.0
            work_smart[col] = work_smart[col].fillna(0.0)
        work_smart["lower_margin_Pa"] = _capillary_margin_values(work_smart)
        work_smart["lower_deficit_Pa"] = np.clip(-work_smart["lower_margin_Pa"], 0.0, None)
        mobility = _viscous_mobility_values(work_smart)
        positive_mobility = mobility[mobility > 0.0]
        mobility_reference = (float(np.max(positive_mobility))
                              if len(positive_mobility) else 0.0)
        work_smart["mobility_weakness"] = (
                1.0 + np.clip(
            (mobility_reference - mobility) / max(mobility_reference, 1.0e-300),
            0.0, 1.0,
        )
        )
        safety_failed = work_smart["lower_margin_Pa"] < 0.0

        target_mask = safety_failed if bool(safety_failed.any()) else (work_smart["smooth_deficit_Pa"] > 0.0)
        eligible_fail = work_smart[
            target_mask
            & (work_smart["A_liq_mm2"] > INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2)
            ].copy()
        eligible_fail["porosity_floor"] = eligible_fail["z_mm"].map(lambda z: _porosity_floor_at_z(float(z), z_bounds))
        eligible_fail = eligible_fail[eligible_fail["porosity"] > eligible_fail["porosity_floor"]].copy()
        if eligible_fail.empty:
            smartfill_status = "physically_blocked" if safety_mode else "passed"
            log("  SmartFill stopped: no failing slice retains safe insertion space.")
            break

        fail = eligible_fail[
            ~eligible_fail["z_mm"].map(lambda z: round(float(z), 6)).isin(blocked_slices)
        ].copy()
        if fail.empty:
            smartfill_status = "five_consecutive_ineffective_per_height" if safety_mode else "passed_feedback_pending_final_fine_audit"
            log("  SmartFill stopped: all remaining candidate slices exhausted their "
                "consecutive no-gain searches; unresolved slices remain in final profile.")
            break
        fail["weight"] = (
                                 fail["lower_deficit_Pa"] * (1.0 + fail["rel_def"].clip(lower=0.0))
                                 + fail["smooth_deficit_Pa"].clip(lower=0.0)
                         ) * fail["mobility_weakness"]

        domain_round_budget = _adaptive_repair_budget(
            fail, adaptive_round_budget, INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND)
        fail = _gradient_repair_schedule(
            fail, work_smart,
            domain_round_budget,
            slice_trial_counts, z_bounds,
        )
        if blocked_failure_bands:
            keep_band = []
            for row in fail.itertuples(index=False):
                lo = float(getattr(row, "insertion_cell_lo_mm"))
                hi = float(getattr(row, "insertion_cell_hi_mm"))
                keep_band.append(not any(
                    min(hi, blocked_hi) > max(lo, blocked_lo) + 1.0e-8
                    for blocked_lo, blocked_hi in blocked_failure_bands
                ))
            fail = fail[np.asarray(keep_band, dtype=bool)].reset_index(drop=True)
        if fail.empty:
            smartfill_status = "physically_blocked_after_adaptive_search"
            log("  SmartFill stopped: every remaining physical failure band exhausted "
                "its pressure-effective repair search.")
            break

        viscous_keys: Set[float] = set()
        viscous_committed = False
        for mech_row in fail.itertuples(index=False):
            z_sched = float(getattr(mech_row, "z_mm"))

            failure_z = float(getattr(mech_row, "worst_failure_z_mm", z_sched))
            state_key = round(failure_z, 6)
            diag = _pressure_failure_mechanism(df_smart, failure_z)
            if diag.get("mode") != "viscous":
                continue
            prior_insert_failures = int(single_trial_failures.get(state_key, 0))
            if prior_insert_failures < int(INTERNAL_DYNAMIC_VISCOUS_RELIEF_MIN_PRIOR_INSERTION_FAILURES):
                continue
            viscous_keys.add(state_key)
            log(
                f"  SmartFill mechanism z={failure_z:.2f}mm: VISCOUS dominated after "
                f"{prior_insert_failures} ineffective insertion searches; "
                f"viscous_share={diag['viscous_share']:.3f}, "
                f"mobility/median={diag['mobility_ratio_to_median']:.3f}, "
                f"hydro-only margin={diag['hydro_only_margin_Pa']:.3f} Pa. "
                f"Switching to cumulative-path resistance relief."
            )
            reject_set = viscous_edge_rejected_by_band.setdefault(state_key, set())
            trial_profile, edge_ok, edge_evals, edge_info = try_viscous_edge_relief(
                failure_z, df_smart, reject_set,
                int(INTERNAL_DYNAMIC_VISCOUS_RELIEF_MAX_EDGE_EVALUATIONS_PER_SEARCH),
            )
            smart_pressure_evaluations += int(edge_evals)
            viscous_relief_evaluations += int(edge_evals)
            slice_trial_counts[state_key] = slice_trial_counts.get(state_key, 0) + 1
            if edge_ok:
                df_smart = trial_profile
                ok = _capillary_profile_passes(df_smart)
                viscous_relief_removed_edges += 1
                slice_accepted_counts[state_key] = slice_accepted_counts.get(state_key, 0) + 1

                viscous_edge_rejected_by_band.clear()
                single_trial_failures.pop(state_key, None)
                viscous_relief_failures.pop(state_key, None)

                reopened_band_count = len(blocked_failure_bands)
                reopened_slice_count = len(blocked_slices)
                blocked_slices.clear()
                blocked_failure_bands.clear()
                single_trial_failures.clear()
                viscous_relief_failures.clear()

                if reopened_band_count or reopened_slice_count:
                    log(
                        f"    SmartFill pressure state changed after accepted VISCOUS_RELIEF; "
                        f"reopened {reopened_band_count} exhausted failure bands "
                        f"({reopened_slice_count} blocked slice keys)."
                    )
                log(
                    f"    + VISCOUS_RELIEF fail_z={failure_z:.2f}mm removed edge "
                    f"{edge_info.get('removed_edge')} at resistance section "
                    f"z={edge_info.get('relief_section_z_mm', failure_z):.2f}mm; mobility "
                    f"{edge_info.get('mobility_before', 0.0):.3e} -> "
                    f"{edge_info.get('mobility_after', 0.0):.3e}, margin "
                    f"{edge_info.get('margin_before_Pa', 0.0):.3f} -> "
                    f"{edge_info.get('margin_after_Pa', 0.0):.3f} Pa; "
                    f"post_mode={edge_info.get('mode_after', 'unknown')}, "
                    f"post_viscous_share={edge_info.get('viscous_share_after', 0.0):.3f}."
                )
                process_log.append({
                    "step": int(global_step), "layer_idx": -1,
                    "phase": "STEP4_VISCOUS_EDGE_RELIEF",
                    "z_mm": float(failure_z),
                    "removed_edge": list(edge_info.get("removed_edge", ())),
                    "pressure_evaluations": int(edge_evals),
                })
                viscous_committed = True
                break
            viscous_relief_failures[state_key] = viscous_relief_failures.get(state_key, 0) + 1
            if viscous_relief_failures[state_key] >= int(INTERNAL_DYNAMIC_VISCOUS_RELIEF_MAX_FAILED_SEARCHES_PER_BAND):
                blocked_slices.add(state_key)
                band_bounds = (float(getattr(mech_row, "local_insertion_cell_lo_mm",
                                             getattr(mech_row, "insertion_cell_lo_mm"))),
                               float(getattr(mech_row, "local_insertion_cell_hi_mm",
                                             getattr(mech_row, "insertion_cell_hi_mm"))))
                if band_bounds not in blocked_failure_bands:
                    blocked_failure_bands.append(band_bounds)
                log(
                    f"  SmartFill skip z={failure_z:.2f}mm: ADD_NODE was exhausted and "
                    f"{int(INTERNAL_DYNAMIC_VISCOUS_RELIEF_MAX_FAILED_SEARCHES_PER_BAND)} ineffective VISCOUS_RELIEF searches also failed; pressure remains unresolved."
                )
            else:
                log(
                    f"    VISCOUS_RELIEF z={failure_z:.2f}mm found no acceptable edge "
                    f"removal ({edge_info.get('reason')}); delete search "
                    f"{viscous_relief_failures[state_key]}/{int(INTERNAL_DYNAMIC_VISCOUS_RELIEF_MAX_FAILED_SEARCHES_PER_BAND)} after "
                    f"{prior_insert_failures} ineffective ADD_NODE transactions."
                )
        if viscous_committed:
            continue
        if viscous_keys:

            if "worst_failure_z_mm" in fail.columns:
                state_series = fail["worst_failure_z_mm"].map(lambda z: round(float(z), 6))
            else:
                state_series = fail["z_mm"].map(lambda z: round(float(z), 6))
            fail = fail[~state_series.isin(viscous_keys)].reset_index(drop=True)
            if fail.empty:
                continue

        local_geometry_committed = False
        if INTERNAL_DYNAMIC_LOCAL_GEOMETRY_REPAIR_ENABLE:
            for local_row in fail.sort_values('weight', ascending=False).itertuples(index=False):
                z_sched = float(getattr(local_row, 'z_mm'))
                failure_z = float(getattr(local_row, 'worst_failure_z_mm', z_sched))
                state_key = round(failure_z, 6)
                diag = _pressure_failure_mechanism(df_smart, failure_z)
                if diag.get('mode') == 'viscous':
                    continue
                if int(single_trial_failures.get(state_key, 0)) < 1:
                    continue
                moved_profile, moved, move_evals, move_info = (
                    try_move_node_toward_worst_pore(failure_z, df_smart, state_key))
                smart_pressure_evaluations += int(move_evals)
                if moved:
                    df_smart = moved_profile
                    ok = _capillary_profile_passes(df_smart)
                    slice_trial_counts[state_key] = slice_trial_counts.get(state_key, 0) + 1
                    slice_accepted_counts[state_key] = slice_accepted_counts.get(state_key, 0) + 1
                    blocked_slices.clear();
                    blocked_failure_bands.clear()
                    single_trial_failures.clear();
                    viscous_relief_failures.clear()
                    move_node_trials.clear();
                    thicken_edge_trials.clear()
                    process_log.append({
                        'step': int(global_step), 'layer_idx': -1,
                        'phase': 'STEP4_MOVE_NODE', 'z_mm': float(failure_z),
                        'node_id': int(move_info.get('node_id', -1)),
                        'move_distance_mm': float(move_info.get('move_distance_mm', 0.0)),
                        'pressure_evaluations': int(move_evals),
                    })
                    log(f"    + MOVE_NODE fail_z={failure_z:.2f}mm node="
                        f"{move_info.get('node_id')} distance="
                        f"{move_info.get('move_distance_mm', 0.0):.3f}mm; deficit "
                        f"{move_info.get('deficit_before_Pa', 0.0):.3f} -> "
                        f"{move_info.get('deficit_after_Pa', 0.0):.3f} Pa.")
                    local_geometry_committed = True
                    break
                if move_evals:
                    log(f"    MOVE_NODE fail_z={failure_z:.2f}mm rejected after "
                        f"{move_evals} exact pressure trial(s); accepted graph unchanged.")
                thick_profile, thickened, thick_evals, thick_info = (
                    try_thicken_edge_near_worst_pore(
                        failure_z, df_smart, state_key))
                smart_pressure_evaluations += int(thick_evals)
                if thickened:
                    df_smart = thick_profile
                    ok = _capillary_profile_passes(df_smart)
                    slice_trial_counts[state_key] = slice_trial_counts.get(state_key, 0) + 1
                    slice_accepted_counts[state_key] = slice_accepted_counts.get(state_key, 0) + 1
                    blocked_slices.clear();
                    blocked_failure_bands.clear()
                    single_trial_failures.clear();
                    viscous_relief_failures.clear()
                    move_node_trials.clear();
                    thicken_edge_trials.clear()
                    process_log.append({
                        'step': int(global_step), 'layer_idx': -1,
                        'phase': 'STEP4_THICKEN_EDGE', 'z_mm': float(failure_z),
                        'source': int(thick_info.get('source', -1)),
                        'target': int(thick_info.get('target', -1)),
                        'old_radius_mm': float(thick_info.get('old_radius_mm', 0.0)),
                        'new_radius_mm': float(thick_info.get('new_radius_mm', 0.0)),
                        'pressure_evaluations': int(thick_evals),
                    })
                    log(f"    + THICKEN_EDGE fail_z={failure_z:.2f}mm edge="
                        f"({thick_info.get('source')},{thick_info.get('target')}) radius="
                        f"{thick_info.get('old_radius_mm', 0.0):.3f} -> "
                        f"{thick_info.get('new_radius_mm', 0.0):.3f}mm; deficit "
                        f"{thick_info.get('deficit_before_Pa', 0.0):.3f} -> "
                        f"{thick_info.get('deficit_after_Pa', 0.0):.3f} Pa.")
                    local_geometry_committed = True
                    break
                if thick_evals:
                    log(f"    THICKEN_EDGE fail_z={failure_z:.2f}mm rejected after "
                        f"{thick_evals} atomic pressure trial(s); accepted graph unchanged.")
        if local_geometry_committed:
            continue

        round_budget = max(1, min(
            int(INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND),
            int(adaptive_round_budget),
        ))

        atoms_before = all_atoms.copy()
        edges_before = list(all_edges)
        ids_before = set(internal_ids)
        layers_before = [list(nodes) for nodes in layer_nodes]
        next_id_before = int(next_id)
        global_step_before = int(global_step)
        smart_total_before = int(smart_total_added)
        supported_before = set(supported_internal_ids)

        slice_cap = max(1, int(INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND))

        round_budget = min(
            int(INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND),
            max(len(fail), int(round_budget)),
        )
        per_slice_caps = np.minimum(
            fail.get("adaptive_add_cap", pd.Series(slice_cap, index=fail.index)).to_numpy(dtype=int),
            slice_cap,
        )

        round_budget = min(int(round_budget), int(np.sum(per_slice_caps)))
        alloc = _smartfill_weighted_allocation(
            fail["weight"].to_numpy(dtype=float), round_budget, per_slice_caps)
        effective_batch_size = int(alloc.sum())
        log("  SmartFill weighted allocation: " + ", ".join(
            f"{float(row.z_mm):.2f}mm x{int(n)} (w={float(row.weight):.2f})"
            for row, n in zip(fail.itertuples(index=False), alloc)))

        round_added = 0
        attempted_heights: List[float] = []
        insertion_attempts_by_height: Dict[float, int] = {}
        trial_positions: List[Dict] = []
        worst_before = float(fail["safety_margin"].min())
        for row_pos, n_add in enumerate(alloc):
            if int(n_add) <= 0:
                continue
            row = fail.iloc[row_pos]
            z_center = float(row["z_mm"])

            failure_z = float(row.get("worst_failure_z_mm", z_center))
            band_key = round(failure_z, 6)
            attempted_heights.append(failure_z)
            bin_w = float(INTERNAL_DYNAMIC_PHYSICS_DZ_MM)
            slice_added = 0
            slice_cap = int(per_slice_caps[row_pos]) if row_pos < len(per_slice_caps) else int(
                INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND)
            last_score = 0.0
            comp_cursor = 0

            n_planned = min(int(n_add), max(0, slice_cap))
            cell_lo = float(
                row.get("local_insertion_cell_lo_mm", row.get("insertion_cell_lo_mm", row["insertion_z_mm"])))
            cell_hi = float(
                row.get("local_insertion_cell_hi_mm", row.get("insertion_cell_hi_mm", row["insertion_z_mm"])))
            if n_planned > 1 and cell_hi > cell_lo + 1.0e-8:
                phase_shift = (0.6180339887498949 * slice_trial_counts.get(
                    band_key, 0)) % 1.0
                phases = ((np.arange(n_planned, dtype=float) + 0.5) / n_planned
                          + phase_shift) % 1.0
                z_targets = cell_lo + np.sort(phases) * (cell_hi - cell_lo)
            else:
                z_targets = np.asarray([float(row["insertion_z_mm"])], dtype=float)
            for _ in range(n_planned):
                if slice_added >= slice_cap:
                    break
                height_key = band_key
                insertion_attempts_by_height[height_key] = insertion_attempts_by_height.get(height_key, 0) + 1
                z_target = float(z_targets[min(_, len(z_targets) - 1)])
                fail_idx = int(np.clip(np.searchsorted(layer_z, z_target, side="right") - 1, 0, n_layers - 1))
                zf0, zf1 = float(layer_z[fail_idx]), float(layer_z[fail_idx + 1])
                is_top_repair = (float(failure_z) >= zmax - float(INTERNAL_TOP_RELAX_ACTIVE_HEIGHT_MM) - 1e-9
                                 and bool(INTERNAL_TOP_SUPPORT_FIRST_ENABLE))
                support_first_repair = bool(
                    INTERNAL_DYNAMIC_SMARTFILL_SUPPORT_FIRST_ENABLE or is_top_repair
                )
                if is_top_repair and len(fail) == 1:

                    fail_idx = int(np.clip(np.searchsorted(layer_z, z_target, side="right") - 1, 0, n_layers - 1))
                    zf0, zf1 = float(layer_z[fail_idx]), float(layer_z[fail_idx + 1])
                    comp_candidates = _top_pore_candidates(
                        mesh, domain_cache, all_atoms, all_edges, z_bounds, failure_z
                    )
                else:
                    comp_candidates = (
                        _top_pore_candidates(mesh, domain_cache, all_atoms, all_edges,
                                             z_bounds, z_target)
                        if INTERNAL_MATCHED_RANDOM_POSITION else
                        _component_worst_pore_candidates(
                            mesh, domain_cache, all_atoms, all_edges, z_bounds, z_target
                        )
                    )
                if not comp_candidates:
                    continue
                key = band_key
                cand = (comp_candidates[int(position_rng.integers(len(comp_candidates)))]
                        if INTERNAL_MATCHED_RANDOM_POSITION else
                        comp_candidates[(comp_cursor + slice_trial_counts.get(key, 0)) % len(comp_candidates)])
                comp_cursor += 1
                xy, score, comp_domain = cand["xy"], cand["score"], cand["domain"]

                candidate_locations = [(xy, None)]
                ok_add, nid = False, None
                if candidate_locations:
                    offset = single_trial_failures.get(key, 0) % len(candidate_locations)
                    candidate_locations = candidate_locations[offset:] + candidate_locations[:offset]
                created_ids = []
                path_status = 'not_attempted'
                for candidate_xy, anchor_id in candidate_locations:
                    ok_add, nid = add_node_to_layer(
                        zf0, zf1, fail_idx, force_insert=True,
                        label=f"SMART_FILL_C{cand['component']}",
                        specific_z=z_target, specific_xy=candidate_xy,
                        domain_override=comp_domain)
                    created_ids = [int(nid)] if ok_add and nid is not None else []
                    path_status = ('local_connection' if ok_add
                                   else 'local_connection_rejected')
                    if ok_add and nid is not None:
                        layer_nodes[fail_idx].append(int(nid))
                    smartfill_placement_log.append({
                        'failed_slice_z_mm': float(failure_z), 'proposed_z_mm': float(z_target),
                        'support_first': bool(support_first_repair), 'support_id': anchor_id,
                        'x_mm': float(candidate_xy[0]), 'y_mm': float(candidate_xy[1]),
                        'accepted_geometrically': bool(ok_add),
                    })
                    trial_positions.append(
                        {'z_mm': float(z_target), 'target_height_mm': height_key, 'x_mm': float(candidate_xy[0]),
                         'y_mm': float(candidate_xy[1]), 'support_id': anchor_id,
                         'inserted': bool(ok_add), 'path_status': path_status})
                    if ok_add:
                        trial_positions[-1]["new_ids"] = [int(i) for i in created_ids]
                        break
                if ok_add:
                    added_count = max(1, len(created_ids))
                    round_added += added_count
                    slice_added += added_count
                    smart_total_added += added_count
                    last_score = float(score)
                else:
                    bridge_room = min(
                        4,
                        slice_cap - slice_added,
                    )
                    if bridge_room > 0 and xy is not None and not support_first_repair:
                        bridge_ids = add_smart_bridge_cluster(zf0, zf1, fail_idx, z_target, xy,
                                                              max_points=int(bridge_room), domain_override=comp_domain)
                        if bridge_ids:
                            trial_positions.append(
                                {"z_mm": float(z_target), "target_height_mm": height_key, "x_mm": float(xy[0]),
                                 "y_mm": float(xy[1]), "inserted": True,
                                 "new_ids": [int(i) for i in bridge_ids],
                                 "kind": "bridge_cluster"})
                            layer_nodes[fail_idx].extend(bridge_ids)
                            added_count = int(len(bridge_ids))
                            round_added += added_count
                            slice_added += added_count
                            smart_total_added += added_count
                            last_score = float(score)
            if slice_added > 0:
                log(f"    + SMART_FILL fail_z={failure_z:.2f}mm x{slice_added}, pore_score={last_score:.3f}")

        attempted_trial_nodes = int(round_added)
        for z_value in set(attempted_heights):
            key = round(float(z_value), 6)
            slice_trial_counts[key] = slice_trial_counts.get(key, 0) + 1

        top_only_smart_trial = bool(attempted_heights) and all(
            float(zv) >= float(zmax) - float(INTERNAL_TOP_RELAX_ACTIVE_HEIGHT_MM) - 1e-9
            for zv in attempted_heights
        )
        reconciled_round = (reconcile_cross_layer_edges()
                            if round_added > 0 and not top_only_smart_trial else 0)
        if round_added > 0:
            ok_trial, crit_z_trial, p_cap_trial, p_hydro_trial, df_trial = _smartfill_trial_capillary_ok_stl(
                mesh, all_atoms, all_edges, atoms_before, edges_before, df_smart,
                z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, domain_cache,
                stats=smartfill_pressure_cache_stats,
            )
            smart_pressure_evaluations += 1
            accepted = (
                bool(_smartfill_meaningful_safety_gain(
                    df_smart, df_trial, insertion_trial=True))
                if safety_mode else bool(_profile_trial_improves(df_smart, df_trial))
            )
            if accepted and safety_mode and not _capillary_profile_passes(df_trial):
                batch_drop = max(0.0, _capillary_safety_rank(df_smart)[1]
                                 - _capillary_safety_rank(df_trial)[1])
                batch_efficiency = batch_drop / max(1, int(round_added))
                adaptive_gain_floor = max(
                    float(INTERNAL_DYNAMIC_SMARTFILL_MIN_DEFICIT_DROP_PA),
                    0.05 * float(peak_gain_per_node),
                )
                if batch_efficiency < adaptive_gain_floor:
                    accepted = False
                    log(f"  SmartFill batch sent to selective salvage: "
                        f"{batch_efficiency:.3f} < {adaptive_gain_floor:.3f} Pa/node.")
        else:
            accepted = False
            ok_trial, crit_z_trial, p_cap_trial, p_hydro_trial, df_trial = (
                ok, crit_z, p_cap, p_hydro, df_smart)
        inserted_heights = {round(p["z_mm"], 6) for p in trial_positions if p["inserted"]}

        salvage_accepted_heights: Set[float] = set()
        salvage_candidate_events: List[Dict] = []
        if (round_added > 0 and not accepted and safety_mode and
                INTERNAL_DYNAMIC_SMARTFILL_PARTIAL_SALVAGE_ENABLE):
            salvage_summary["rejected_batches"] += 1
            full_atoms, full_edges = all_atoms, list(all_edges)

            operations = [p for p in trial_positions if p.get("inserted") and p.get("new_ids")]

            operations.sort(key=lambda p: float(df_smart.iloc[
                                                    (df_smart["z_mm"] - float(p["z_mm"])).abs().argmin()
                                                ].get("safety_margin", 0.0)))
            kept_ids: Set[int] = set()
            current_profile = df_smart
            current_atoms, current_edges = atoms_before, list(edges_before)

            for op in operations:
                new_ids = set(map(int, op["new_ids"]))
                selected = kept_ids | new_ids
                candidate_atoms, candidate_edges = _smartfill_filter_transaction(
                    atoms_before, edges_before, full_atoms, full_edges, selected)
                if not _smartfill_subset_supported(candidate_atoms, candidate_edges, selected, z_bounds):
                    salvage_summary["candidate_support_rejections"] += 1
                    salvage_candidate_events.append({"ids": sorted(new_ids), "accepted": False,
                                                     "reason": "support_or_connectivity"})
                    continue
                result = _smartfill_trial_capillary_ok_stl(
                    mesh, candidate_atoms, candidate_edges, current_atoms, current_edges,
                    current_profile, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM,
                    domain_cache, stats=smartfill_pressure_cache_stats)
                smart_pressure_evaluations += 1
                salvage_summary["candidate_evaluations"] += 1
                candidate_profile = result[4]
                candidate_drop = max(
                    0.0,
                    _capillary_safety_rank(current_profile)[1]
                    - _capillary_safety_rank(candidate_profile)[1],
                )
                candidate_efficiency = candidate_drop / max(1, len(new_ids))
                adaptive_gain_floor = max(
                    float(INTERNAL_DYNAMIC_SMARTFILL_MIN_DEFICIT_DROP_PA),
                    0.05 * float(peak_gain_per_node),
                )
                gain = bool(
                    _smartfill_local_safe_gain(
                        current_profile, candidate_profile, insertion_trial=True)
                    and (_capillary_profile_passes(candidate_profile)
                         or candidate_efficiency >= adaptive_gain_floor)
                )
                salvage_candidate_events.append({"ids": sorted(new_ids), "accepted": bool(gain),
                                                 "reason": "pressure_gain" if gain else "pressure_gain_insufficient",
                                                 "deficit_before_Pa": _capillary_safety_rank(current_profile)[1],
                                                 "deficit_after_Pa": _capillary_safety_rank(candidate_profile)[1],
                                                 "gain_per_node_Pa": candidate_efficiency,
                                                 "adaptive_gain_floor_Pa": adaptive_gain_floor})
                if gain:
                    kept_ids = selected
                    current_atoms, current_edges, current_profile = (
                        candidate_atoms, candidate_edges, candidate_profile)
                    salvage_accepted_heights.add(round(float(op["target_height_mm"]), 6))
                else:
                    salvage_summary["candidate_pressure_rejections"] += 1
            if kept_ids:
                accepted = True
                all_atoms, all_edges = current_atoms, current_edges
                internal_ids = ids_before | kept_ids
                layer_nodes = [list(nodes) for nodes in layers_before]
                for row in current_atoms.iloc[len(atoms_before):].itertuples(index=False):
                    k = int(np.clip(np.searchsorted(layer_z, float(row.z), side="right") - 1,
                                    0, n_layers - 1))
                    layer_nodes[k].append(int(row.atom_id))
                supported_internal_ids.clear()
                supported_internal_ids.update(supported_before)
                supported_internal_ids.update(kept_ids)

                next_id = max(next_id_before, max(kept_ids) + 1)
                global_step = global_step_before + len(kept_ids)
                smart_total_added = smart_total_before + len(kept_ids)
                round_added = len(kept_ids)
                reconciled_round = 0

                ok_trial, crit_z_trial, p_cap_trial, p_hydro_trial, df_trial = _smartfill_trial_capillary_ok_stl(
                    mesh, all_atoms, all_edges, atoms_before, edges_before, df_smart,
                    z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, domain_cache,
                    stats=smartfill_pressure_cache_stats)
                smart_pressure_evaluations += 1
                if not _smartfill_local_safe_gain(
                        df_smart, df_trial, insertion_trial=True):
                    accepted = False
                else:
                    salvage_summary["salvaged_batches"] += 1
                    salvage_summary["salvaged_nodes"] += len(kept_ids)
                    log(f"  SmartFill partial salvage: retained {len(kept_ids)} / {attempted_trial_nodes} nodes, "
                        f"heights={len(salvage_accepted_heights)}")
        rank_before = _capillary_safety_rank(df_smart)
        rank_after = _capillary_safety_rank(df_trial)
        trial_pressure_rows = []
        for z_value in attempted_heights:
            before_row = df_smart.iloc[(df_smart['z_mm'] - z_value).abs().argmin()]
            after_row = df_trial.iloc[(df_trial['z_mm'] - z_value).abs().argmin()]
            trial_pressure_rows.append({
                'z_mm': float(z_value),
                'P_cap_before_Pa': float(before_row['P_cap_Pa']),
                'P_cap_after_Pa': float(after_row['P_cap_Pa']),
                'P_required_before_Pa': float(before_row['P_required_Pa']),
                'P_required_after_Pa': float(after_row['P_required_Pa']),
                'deficit_before_Pa': float(rank_before[1]),
                'deficit_after_Pa': float(rank_after[1]),
            })
        rolled_back = not accepted
        if not accepted:
            all_atoms = atoms_before
            all_edges = edges_before
            internal_ids = ids_before
            layer_nodes = layers_before
            supported_internal_ids.clear()
            supported_internal_ids.update(supported_before)
            next_id = next_id_before
            global_step = global_step_before
            smart_total_added = smart_total_before
            round_added = 0
            reconciled_round = 0
            adaptive_round_budget = max(2, min(int(INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND),
                                               max(2, effective_batch_size // 2)))

            for key, attempt_count in insertion_attempts_by_height.items():
                z_value = key
                single_trial_failures[key] = single_trial_failures.get(key, 0) + 1
                if single_trial_failures[key] >= 5:

                    diag_after_insert = _pressure_failure_mechanism(df_smart, float(z_value))
                    if diag_after_insert.get("mode") == "viscous":
                        log(
                            f"  SmartFill ADD_NODE exhausted z={z_value:.2f}mm after "
                            f"{single_trial_failures[key]} ineffective ADD_NODE transactions; "
                            f"viscous_share={diag_after_insert['viscous_share']:.3f}, "
                            f"mobility/median={diag_after_insert['mobility_ratio_to_median']:.3f}. "
                            "Band remains active for VISCOUS_RELIEF."
                        )
                    else:
                        blocked_slices.add(key)
                        if "worst_failure_z_mm" in fail.columns:
                            nearest = fail.iloc[(fail["worst_failure_z_mm"] - float(key)).abs().argmin()]
                        else:
                            nearest = fail.iloc[(fail["z_mm"] - float(key)).abs().argmin()]
                        band_bounds = (
                            float(nearest.get("local_insertion_cell_lo_mm", nearest["insertion_cell_lo_mm"])),
                            float(nearest.get("local_insertion_cell_hi_mm", nearest["insertion_cell_hi_mm"])),
                        )
                        if band_bounds not in blocked_failure_bands:
                            blocked_failure_bands.append(band_bounds)
                        log(
                            f"  SmartFill skip z={z_value:.2f}mm: ADD_NODE exhausted after "
                            f"{single_trial_failures[key]} ineffective ADD_NODE transactions and the failure "
                            "is not viscous-dominated; pressure remains unresolved."
                        )
            log("  SmartFill trial had no acceptable pressure improvement after partial salvage "
                f"(min_margin_before={-rank_before[2]:.3f} Pa, "
                f"min_margin_after={-rank_after[2]:.3f} Pa).")
        else:
            ok, crit_z, p_cap, p_hydro, df_smart = (
                ok_trial, crit_z_trial, p_cap_trial, p_hydro_trial, df_trial
            )

            retained_nodes = max(1, int(round_added))
            realised_gain = max(0.0, float(rank_before[1] - rank_after[1]))
            gain_per_node = realised_gain / retained_nodes
            peak_gain_per_node = max(float(peak_gain_per_node), float(gain_per_node))
            previous_efficiency = gain_per_node_ema
            if previous_efficiency is None or not np.isfinite(previous_efficiency):
                gain_per_node_ema = gain_per_node
                next_budget = min(
                    int(INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND),
                    retained_nodes + max(1, retained_nodes // 2),
                )
            else:
                response_ratio = gain_per_node / max(previous_efficiency, 1.0e-9)
                if response_ratio < 0.50:

                    next_budget = max(2, int(math.ceil(
                        retained_nodes * max(0.10, response_ratio))))
                elif response_ratio < 0.85:
                    next_budget = max(2, retained_nodes)
                else:
                    next_budget = retained_nodes + max(1, retained_nodes // 3)
                gain_per_node_ema = 0.75 * previous_efficiency + 0.25 * gain_per_node
            adaptive_round_budget = min(
                max(2, int(INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND)),
                max(2, int(next_budget)),
            )
            log(f"  SmartFill adaptive response: {gain_per_node:.3f} Pa/node, "
                f"next transaction budget={adaptive_round_budget}.")

            reopened_band_count = len(blocked_failure_bands)
            reopened_slice_count = len(blocked_slices)
            blocked_slices.clear()
            blocked_failure_bands.clear()

            single_trial_failures.clear()
            viscous_relief_failures.clear()
            viscous_edge_rejected_by_band.clear()

            if reopened_band_count or reopened_slice_count:
                log(
                    f"  SmartFill pressure state changed after accepted repair; reopened "
                    f"{reopened_band_count} exhausted failure bands "
                    f"({reopened_slice_count} blocked slice keys)."
                )
            committed_heights = (salvage_accepted_heights if salvage_accepted_heights else
                                 set(insertion_attempts_by_height))
            for key in committed_heights:
                single_trial_failures.pop(key, None)
                slice_accepted_counts[key] = slice_accepted_counts.get(key, 0) + 1
                low_floor = max(
                    float(INTERNAL_DYNAMIC_SMARTFILL_MIN_DEFICIT_DROP_PA),
                    0.05 * float(peak_gain_per_node),
                )
                if gain_per_node < low_floor:
                    low_efficiency_streaks[key] = low_efficiency_streaks.get(key, 0) + 1
                else:
                    low_efficiency_streaks[key] = 0
                if low_efficiency_streaks[key] >= 6:
                    if "worst_failure_z_mm" in fail.columns:
                        nearest = fail.iloc[(fail["worst_failure_z_mm"] - float(key)).abs().argmin()]
                    else:
                        nearest = fail.iloc[(fail["z_mm"] - float(key)).abs().argmin()]
                    band_bounds = (
                        float(nearest.get("local_insertion_cell_lo_mm", nearest["insertion_cell_lo_mm"])),
                        float(nearest.get("local_insertion_cell_hi_mm", nearest["insertion_cell_hi_mm"])),
                    )
                    if band_bounds not in blocked_failure_bands:
                        blocked_failure_bands.append(band_bounds)
                    log(f"  SmartFill block z={float(key):.2f}mm band after six "
                        f"low-return commits (<{low_floor:.3f} Pa/node).")

            stale_rejected_heights = set(insertion_attempts_by_height) - committed_heights
            if stale_rejected_heights:
                log(
                    f"  SmartFill accepted repair changed the pressure state; discarded "
                    f"stale ineffective-search counts at {len(stale_rejected_heights)} other height(s)."
                )
            smartfill_status = "passed_feedback_pending_final_fine_audit" if ok else "running"
            if safety_mode:
                log(f"  SmartFill accepted: deficit {rank_before[1]:.3f} -> "
                    f"{rank_after[1]:.3f} Pa; remaining_fail_slices={rank_after[0]}")

        process_log.append({
            "step": int(global_step),
            "layer_idx": -1,
            "phase": "STEP4_SMART_FILL_BATCH",
            "smart_round": int(smart_round),
            "fail_slices": int(len(fail)),
            "worst_margin_before": float(worst_before),
            "added": int(round_added),
            "reconciled_edges": int(reconciled_round),
            "rolled_back": int(rolled_back),
            "total_smart_added": int(smart_total_added),
            "adaptive_round_budget": int(adaptive_round_budget),
            "domain_round_budget": int(domain_round_budget),
            "effective_batch_size": int(effective_batch_size),
            "attempted_trial_nodes": int(attempted_trial_nodes),
            "cumulative_pressure_evaluations": int(smart_pressure_evaluations),
            "deficit_before_Pa": float(rank_before[1]),
            "deficit_after_Pa": float(rank_after[1]),
            "gain_per_retained_node_Pa": (
                float(max(0.0, rank_before[1] - rank_after[1]) / max(1, round_added))
                if accepted else 0.0),
            "safety_mode": int(safety_mode),
            "smartfill_status": smartfill_status,
            "trial_positions": json.dumps(trial_positions),
            "salvage_candidate_events": json.dumps(salvage_candidate_events),
            "trial_pressure": json.dumps(trial_pressure_rows),
            "trial_outcome": "accepted" if accepted else (
                "geometry_failed" if attempted_trial_nodes == 0 else "pressure_gain_insufficient"),
        })
        log(f"  SmartFill round {smart_round}: fail_slices={len(fail)}, added={round_added}, total_added={smart_total_added}")
    process_log.append({
        "step": int(global_step),
        "layer_idx": -1,
        "phase": "STEP4_SMART_FILL_STATUS",
        "smart_round": int(smart_round),
        "total_smart_added": int(smart_total_added),
        "pressure_evaluations": int(smart_pressure_evaluations),
        "pressure_evaluation_limit": int(INTERNAL_DYNAMIC_SMARTFILL_MAX_PRESSURE_EVALUATIONS),
        "per_slice_trial_counts": str(slice_trial_counts),
        "per_slice_accepted_counts": str(slice_accepted_counts),
        "per_slice_consecutive_ineffective": str(single_trial_failures),
        "skipped_after_five_failures": str(sorted(blocked_slices)),
        "remaining_failed_slices": int(_capillary_safety_rank(df_smart)[0]),
        "remaining_deficit_Pa": float(_capillary_safety_rank(df_smart)[1]),
        "smartfill_status": smartfill_status,
    })
    (output_dir / "smartfill_partial_salvage_summary.json").write_text(
        json.dumps(salvage_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(smartfill_local_geometry_log).to_csv(
        output_dir / 'smartfill_local_geometry_trials.csv', index=False)
    local_action_summary = {
        'move_node_trial_count': int(sum(
            str(item.get('action')) == 'MOVE_NODE'
            for item in smartfill_local_geometry_log)),
        'move_node_accepted_count': int(sum(
            str(item.get('action')) == 'MOVE_NODE' and bool(item.get('accepted'))
            for item in smartfill_local_geometry_log)),
        'thicken_edge_trial_count': int(sum(
            str(item.get('action')) == 'THICKEN_EDGE'
            for item in smartfill_local_geometry_log)),
        'thicken_edge_accepted_count': int(sum(
            str(item.get('action')) == 'THICKEN_EDGE' and bool(item.get('accepted'))
            for item in smartfill_local_geometry_log)),
        'action_priority': 'ADD_NODE_then_MOVE_NODE_then_THICKEN_EDGE',
        'maximum_allowed_viscous_share': float(
            INTERNAL_DYNAMIC_LOCAL_REPAIR_MAX_VISCOUS_SHARE),
    }
    (output_dir / 'smartfill_local_geometry_summary.json').write_text(
        json.dumps(local_action_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f"  SmartFill finished with status={smartfill_status}, total_added={smart_total_added}, "
        f"salvaged_nodes={salvage_summary['salvaged_nodes']}, "
        f"viscous_relief_removed_edges={viscous_relief_removed_edges}, "
        f"viscous_relief_evaluations={viscous_relief_evaluations}.")
    reject_columns = ['label', 'reason', 'z_mm', 'x_mm', 'y_mm', 'requested_lower_support_id']
    pd.DataFrame(smartfill_rejections, columns=reject_columns).to_csv(
        output_dir / 'smartfill_rejection_details.csv', index=False)
    counts = dict(pd.Series([r['reason'] for r in smartfill_rejections]).value_counts())
    counts_json = {str(k): int(v) for k, v in counts.items()}
    (output_dir / 'smartfill_rejection_summary.json').write_text(json.dumps({
        'reason_counts': counts_json,
        'total_rejected_positions': len(smartfill_rejections),
        'support_first_all_smartfill': bool(INTERNAL_DYNAMIC_SMARTFILL_SUPPORT_FIRST_ENABLE),
        'top_short_edge_relaxation_enabled': bool(INTERNAL_TOP_SUPPORT_FIRST_ENABLE),
        'strict_first_support_min_angle_deg': float(INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(smartfill_placement_log, columns=[
        'failed_slice_z_mm', 'proposed_z_mm', 'support_first', 'support_id',
        'x_mm', 'y_mm', 'accepted_geometrically', 'P_cap_before_Pa', 'P_cap_after_Pa',
        'P_required_before_Pa', 'P_required_after_Pa', 'deficit_drop_Pa',
    ]).to_csv(output_dir / 'smartfill_placement_candidates.csv', index=False)
    log('  SmartFill placement rejected by reason: ' + json.dumps(counts_json, ensure_ascii=False))

    df_smart_full = _pressure_profile_stl(
        mesh, all_atoms, all_edges, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM,
        cache=domain_cache,
    )
    comparison_cols = [c for c in ("z_mm", "A_liq_mm2", "P_wet_mm",
                                   "P_cap_Pa", "P_visc_dynamic_Pa", "P_visc_ref_Pa", "P_viscous_Pa",
                                   "P_viscous_capacity_proxy_Pa", "P_required_base_Pa", "P_required_Pa",
                                   "Safety_Margin")
                       if c in df_smart.columns and c in df_smart_full.columns]
    cache_matches_full = bool(
        len(df_smart) == len(df_smart_full)
        and len(comparison_cols) >= 4
        and np.allclose(
            df_smart[comparison_cols].to_numpy(float),
            df_smart_full[comparison_cols].to_numpy(float),
            rtol=1.0e-9, atol=1.0e-8, equal_nan=True,
        )
    )
    smartfill_pressure_cache_stats["final_full_profile_matches_cached"] = cache_matches_full
    if not cache_matches_full:
        log("  WARNING: incremental SmartFill profile differs from complete pressure "
            "recalculation; using the complete profile for all later stages.")
    df_smart = df_smart_full

    fine_rows = []
    if all_atoms is not None and not all_atoms.empty:
        nodes_for_roof = _make_nodes_np_for_analysis(all_atoms, z_bounds)
        edges_for_roof = pd.DataFrame(all_edges)
        for roof_offset in INTERNAL_TOP_COVERAGE_DIAGNOSTIC_OFFSETS_MM:
            z_roof = float(zmax) - float(roof_offset)
            if not (zmin < z_roof < zmax):
                continue
            domain = _domain_at_z_cached(mesh, z_roof, domain_cache)
            if domain.is_empty:
                continue
            solid = _generate_true_solid_geometry_at_z_full(
                nodes_for_roof, edges_for_roof, all_atoms, z_roof)
            clipped = solid.intersection(domain) if not solid.is_empty else GeometryCollection()
            area = float(domain.area)
            liquid_area = max(1e-9, area - float(clipped.area))
            try:
                wet = float(clipped.boundary.intersection(domain.buffer(-1e-6)).length) if not clipped.is_empty else 0.0
            except Exception:
                wet = 0.0
            cap = (GAMMA_NPM * math.cos(math.radians(CONTACT_ANGLE_DEG)) * wet * 1e-3 / (liquid_area * 1e-6))
            fine_rows.append({'z_mm': z_roof, 'A_liq_mm2': liquid_area,
                              'P_wet_mm': wet, 'P_cap_geometry_only_Pa': cap,
                              'porosity': liquid_area / max(area, 1e-12),
                              'pressure_pass_assessed': False,
                              'note': 'No z-resolved viscous integration; diagnostic only'})
    pd.DataFrame(fine_rows, columns=['z_mm', 'A_liq_mm2', 'P_wet_mm',
                                     'P_cap_geometry_only_Pa', 'porosity', 'pressure_pass_assessed', 'note']).to_csv(
        output_dir / 'top_near_roof_after_smartfill_geometry_diagnostic.csv', index=False)
    (output_dir / "smartfill_pressure_cache_stats.json").write_text(
        json.dumps(smartfill_pressure_cache_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log("  SmartFill pressure sections: "
        f"reused={smartfill_pressure_cache_stats['section_rows_reused']}, "
        f"recomputed={smartfill_pressure_cache_stats['section_rows_recomputed']}, "
        f"fallbacks={smartfill_pressure_cache_stats['full_fallback_trials']}.")

    def relieve_low_porosity_slices(
            df_start: Optional[pd.DataFrame] = None) -> Tuple[int, int, pd.DataFrame]:
        """Delete internal nodes and incident edges while both stop conditions hold."""
        nonlocal all_atoms, all_edges, internal_ids, layer_nodes
        df_now = df_start.copy() if df_start is not None and not df_start.empty else _pressure_profile_stl(
            mesh, all_atoms, all_edges, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache
        )
        threshold = float(INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_THRESHOLD)
        max_rounds = max(0, int(INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_MAX_ROUNDS))
        if max_rounds == 0:
            log("  LowPorosityPrune disabled because its maximum round count is 0.")
            return 0, 0, df_now
        dz_win = max(float(INTERNAL_DYNAMIC_PHYSICS_DZ_MM), total_h / max(n_layers, 1))
        removed_total = 0
        removed_edges_total = 0
        batch_limit = max(1, int(INTERNAL_DYNAMIC_SMOOTH_PRUNE_BATCH_SIZE))
        trial_batch_size = batch_limit

        def issue_score(rows: pd.DataFrame) -> float:
            if rows is None or rows.empty:
                return 0.0
            critical = np.maximum(rows["P_required_Pa"].to_numpy(dtype=float), 1.0)
            excess_rel = rows["capillary_excess_Pa"].to_numpy(dtype=float) / critical
            return float(np.sum(rows["porosity_deficit"].to_numpy(dtype=float) * (1.0 + excess_rel)))

        if not _capillary_profile_passes(df_now):
            log("  LowPorosityPrune skipped: capillary pressure has not passed every slice.")
            return 0, 0, df_now
        mobility_reference_profile = df_now.copy(deep=True)

        rejected_delete_nodes: Set[int] = set()
        started = time.perf_counter()
        stalled_evaluations = 0
        stop_reason = ""
        issues = _low_porosity_excess_rows(df_now, threshold)
        if issues.empty:
            return 0, 0, df_now
        log(
            f"  LowPorosityPrune started: threshold={100.0 * threshold:.1f}%, "
            f"eligible_slices={len(issues)}, max_rounds={max_rounds}."
        )
        round_idx = 0
        while round_idx < max_rounds:
            stop_reason = _prune_limit_reason(started, round_idx, stalled_evaluations, max_rounds)
            if stop_reason:
                break
            round_idx += 1
            round_label = f"{round_idx}/{max_rounds}"
            issues = _low_porosity_excess_rows(df_now, threshold)
            if issues.empty:
                break
            selected: List[int] = []
            for issue in issues.itertuples(index=False):
                z_issue = float(getattr(issue, "z_mid_mm", getattr(issue, "z_mm", zmin)))
                candidates = all_atoms[
                    all_atoms["atom_id"].astype(int).isin(internal_ids)
                    & (np.abs(all_atoms["z"].astype(float) - z_issue) <= dz_win)
                    & ~all_atoms["face_name"].astype(str).str.contains("bottom_pillar", case=False, regex=False)
                    ].copy()
                candidates = candidates[
                    ~candidates["atom_id"].astype(int).isin(selected)
                    & ~candidates["atom_id"].astype(int).isin(rejected_delete_nodes)
                    ]
                if candidates.empty:
                    continue
                names = candidates["face_name"].astype(str)
                candidates["delete_priority"] = np.select(
                    [
                        names.str.contains("SMART_FILL|SMART_BRIDGE", regex=True),
                        names.str.contains("PHYS_GAP", regex=False),
                    ],
                    [0, 1],
                    default=2,
                )
                candidates["dist_z"] = np.abs(candidates["z"].astype(float) - z_issue)
                candidates = candidates.sort_values(
                    ["delete_priority", "dist_z", "atom_id"],
                    ascending=[True, True, False],
                )
                selected.append(int(candidates.iloc[0]["atom_id"]))
                if len(selected) >= trial_batch_size:
                    break

            if not selected:
                log("  LowPorosityPrune stopped: no removable internal nodes near eligible slices.")
                break
            selected_set = set(selected)
            trial_atoms = all_atoms[~all_atoms["atom_id"].astype(int).isin(selected_set)].reset_index(drop=True)
            trial_edges = [
                edge for edge in all_edges
                if int(edge.get("source", -1)) not in selected_set
                   and int(edge.get("target", -1)) not in selected_set
            ]
            if not _deletion_preserves_internal_support(
                    all_atoms, all_edges, trial_atoms, trial_edges, z_bounds=z_bounds
            ):
                rejected_delete_nodes.update(selected_set)
                if len(selected_set) > 1:
                    trial_batch_size = max(1, len(selected_set) // 2)
                continue
            log(f"    LowPorosityPrune {round_label}: evaluating {len(selected_set)}-node deletion.")
            df_trial = _pressure_profile_stl(
                mesh, trial_atoms, trial_edges, z_bounds,
                INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache
            )
            issues_trial = _low_porosity_excess_rows(df_trial, threshold)
            if not _capillary_profile_passes(df_trial):
                stalled_evaluations += 1
                if len(selected_set) > 1:
                    trial_batch_size = max(1, len(selected_set) // 2)
                    log(
                        f"    LowPorosityPrune {round_label}: rejected {len(selected_set)}-node batch "
                        f"because the fixed-flow transport-margin requirement would fail; retrying with batch size {trial_batch_size}."
                    )
                else:
                    rejected_delete_nodes.update(selected_set)
                    log(
                        f"    LowPorosityPrune {round_label}: "
                        "rejected deletion because the fixed-flow transport-margin requirement would fail."
                    )
                continue
            if not _meaningful_decrease(issue_score(issues), issue_score(issues_trial)):
                stalled_evaluations += 1
                if len(selected_set) > 1:
                    trial_batch_size = max(1, len(selected_set) // 2)
                    log(
                        f"    LowPorosityPrune {round_label}: rejected {len(selected_set)}-node batch "
                        f"because porosity did not improve; retrying with batch size {trial_batch_size}."
                    )
                else:
                    rejected_delete_nodes.update(selected_set)
                    log(
                        f"    LowPorosityPrune {round_label}: "
                        "rejected deletion because porosity did not improve."
                    )
                continue

            all_atoms = trial_atoms
            removed_edges_round = len(all_edges) - len(trial_edges)
            all_edges = trial_edges
            internal_ids.difference_update(selected_set)
            for nodes in layer_nodes:
                nodes[:] = [node_id for node_id in nodes if node_id not in selected_set]
            df_now = df_trial
            removed_total += len(selected_set)
            removed_edges_total += removed_edges_round
            trial_batch_size = min(batch_limit, trial_batch_size + 1)
            stalled_evaluations = 0
            remaining = len(issues_trial)
            log(
                f"    LowPorosityPrune {round_label}: "
                f"removed_nodes={len(selected_set)}, removed_edges={removed_edges_round}, "
                f"remaining_slices={remaining}."
            )

        remaining = len(_low_porosity_excess_rows(df_now, threshold))
        if remaining and (stop_reason or round_idx >= max_rounds):
            log(
                f"  LowPorosityPrune stopped by {stop_reason or 'evaluation limit'} with "
                f"{remaining} eligible slices left."
            )
        return int(removed_total), int(removed_edges_total), df_now

    def smooth_prune_overfilled_slices(
            df_start: Optional[pd.DataFrame] = None) -> Tuple[int, int, pd.DataFrame]:
        nonlocal all_atoms, all_edges, internal_ids, layer_nodes
        removed = 0
        removed_edges = 0
        df_now = df_start.copy() if df_start is not None and not df_start.empty else _pressure_profile_stl(
            mesh, all_atoms, all_edges, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache
        )
        if not _capillary_profile_passes(df_now):
            log("  SmoothPrune skipped: capillary pressure has not passed every slice.")
            return 0, 0, df_now
        mobility_reference_profile = df_now.copy(deep=True)
        best_score = _profile_quality_score(df_now)
        dz_win = max(float(INTERNAL_DYNAMIC_PHYSICS_DZ_MM), (zmax - zmin) / max(n_layers, 1))
        rejected_prune_nodes: Set[int] = set()
        consecutive_rejections = 0
        max_attempts = max(0, int(INTERNAL_DYNAMIC_SMOOTH_PRUNE_MAX_REMOVALS))
        batch_limit = max(1, int(INTERNAL_DYNAMIC_SMOOTH_PRUNE_BATCH_SIZE))
        trial_batch_size = batch_limit
        started = time.perf_counter()
        evaluation_count = 0
        stalled_evaluations = 0
        stop_reason = ""
        if max_attempts:
            log(
                f"  SmoothPrune started: target P_cap <= "
                f"{_capillary_trim_ratio():.3f} * P_required before final assembly; "
                f"at most {max_attempts} pressure re-evaluations."
            )
        for attempt in range(max_attempts):
            stop_reason = _prune_limit_reason(started, evaluation_count, stalled_evaluations, max_attempts)
            if stop_reason:
                break
            issues = _profile_overfill_rows(df_now)
            if issues.empty:
                break
            chosen_nodes: List[int] = []
            chosen_z_values: List[float] = []
            for issue in issues.itertuples(index=False):
                z_issue = float(getattr(issue, "z_mid_mm", getattr(issue, "z_mm", zmin)))
                removable = (
                        all_atoms["atom_id"].astype(int).isin(internal_ids)
                        & ~all_atoms["face_name"].astype(str).str.contains("bottom_pillar", case=False, regex=False)
                )
                candidates = all_atoms[
                    removable
                    & all_atoms["face_name"].astype(str).str.contains("SMART_FILL|SMART_BRIDGE", regex=True)
                    & (np.abs(all_atoms["z"].astype(float) - z_issue) <= dz_win)
                    ].copy()
                if candidates.empty:
                    candidates = all_atoms[
                        removable
                        & (np.abs(all_atoms["z"].astype(float) - z_issue) <= dz_win)
                        ].copy()
                if candidates.empty:
                    continue
                candidates = candidates[
                    ~candidates["atom_id"].astype(int).isin(rejected_prune_nodes)
                    & ~candidates["atom_id"].astype(int).isin(chosen_nodes)
                    ].copy()
                if candidates.empty:
                    continue
                candidates["dist_z"] = np.abs(candidates["z"].astype(float) - z_issue)
                candidates = candidates.sort_values(["dist_z", "atom_id"], ascending=[True, False])
                chosen_nodes.append(int(candidates.iloc[0]["atom_id"]))
                chosen_z_values.append(z_issue)
                if len(chosen_nodes) >= trial_batch_size:
                    break
            if not chosen_nodes:
                break

            chosen_set = set(chosen_nodes)
            atoms_prev = all_atoms
            edges_prev = list(all_edges)
            ids_prev = set(internal_ids)
            layer_prev = [list(v) for v in layer_nodes]
            all_atoms = all_atoms[~all_atoms["atom_id"].astype(int).isin(chosen_set)].reset_index(drop=True)
            all_edges = [
                e for e in all_edges
                if int(e.get("source", -1)) not in chosen_set and int(e.get("target", -1)) not in chosen_set
            ]
            internal_ids.difference_update(chosen_set)
            for nodes in layer_nodes:
                nodes[:] = [node_id for node_id in nodes if node_id not in chosen_set]

            if not _deletion_preserves_internal_support(
                    atoms_prev, edges_prev, all_atoms, all_edges, z_bounds=z_bounds
            ):
                all_atoms, all_edges, internal_ids, layer_nodes = atoms_prev, edges_prev, ids_prev, layer_prev
                rejected_prune_nodes.update(chosen_set)
                if len(chosen_set) > 1:
                    trial_batch_size = max(1, len(chosen_set) // 2)
                continue
            evaluation_count += 1
            log(
                f"    SmoothPrune evaluation {evaluation_count}/{max_attempts}: "
                f"evaluating {len(chosen_set)}-node deletion."
            )
            df_trial = _pressure_profile_stl(mesh, all_atoms, all_edges, z_bounds, INTERNAL_DYNAMIC_PHYSICS_DZ_MM,
                                             cache=domain_cache)
            score_trial = _profile_quality_score(df_trial)
            if (_capillary_profile_passes(df_trial)
                    and _meaningful_decrease(best_score, score_trial)):
                removed += len(chosen_set)
                best_score = score_trial
                df_now = df_trial
                consecutive_rejections = 0
                stalled_evaluations = 0
                trial_batch_size = min(batch_limit, trial_batch_size + 1)
                log(
                    f"    SmoothPrune {attempt + 1}/{max_attempts}: "
                    f"removed {len(chosen_set)} nodes across z="
                    f"{min(chosen_z_values):.2f}-{max(chosen_z_values):.2f}mm."
                )
                continue

            all_atoms = atoms_prev
            all_edges = edges_prev
            internal_ids = ids_prev
            layer_nodes = layer_prev
            stalled_evaluations += 1
            if len(chosen_set) > 1:
                trial_batch_size = max(1, len(chosen_set) // 2)
                consecutive_rejections = 0
                log(
                    f"    SmoothPrune {attempt + 1}/{max_attempts}: "
                    f"rejected {len(chosen_set)}-node batch; retrying with batch size {trial_batch_size}."
                )
                continue

            chosen_node = next(iter(chosen_set))
            degree = {}
            for edge in edges_prev:
                u, v = int(edge.get("source", -1)), int(edge.get("target", -1))
                degree[u] = degree.get(u, 0) + 1
                degree[v] = degree.get(v, 0) + 1
            atom_xyz = atoms_prev.set_index("atom_id")[["x", "y", "z"]].to_dict("index")
            edge_candidates = []
            for edge_idx, edge in enumerate(edges_prev):
                u, v = int(edge.get("source", -1)), int(edge.get("target", -1))
                if chosen_node not in (u, v):
                    continue
                if str(edge.get("bond_type", "")) != "internal":
                    continue
                if "bottom_pillar" in str(edge.get("role", "")).lower():
                    continue
                if degree.get(u, 0) <= 1 or degree.get(v, 0) <= 1:
                    continue
                if u not in atom_xyz or v not in atom_xyz:
                    continue
                zu = float(atom_xyz[u]["z"])
                zv = float(atom_xyz[v]["z"])
                z_edge = 0.5 * (zu + zv)
                edge_candidates.append((abs(z_edge - chosen_z_values[0]), -float(edge.get("radius", 0.0)), edge_idx))
            edge_accepted = False
            for _, _, edge_idx in sorted(edge_candidates)[:3]:
                stop_reason = _prune_limit_reason(started, evaluation_count, stalled_evaluations, max_attempts)
                if stop_reason:
                    break
                all_atoms = atoms_prev
                all_edges = [edge for idx, edge in enumerate(edges_prev) if idx != edge_idx]
                internal_ids = ids_prev
                layer_nodes = layer_prev
                if not _deletion_preserves_internal_support(
                        atoms_prev, edges_prev, all_atoms, all_edges, z_bounds=z_bounds
                ):
                    continue
                evaluation_count += 1
                log(
                    f"    SmoothPrune evaluation {evaluation_count}/{max_attempts}: "
                    "evaluating 1-edge deletion."
                )
                df_edge_trial = _pressure_profile_stl(
                    mesh, all_atoms, all_edges, z_bounds,
                    INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache=domain_cache,
                )
                edge_score = _profile_quality_score(df_edge_trial)
                if (_capillary_profile_passes(df_edge_trial)
                        and _meaningful_decrease(best_score, edge_score)):
                    removed_edges += 1
                    best_score = edge_score
                    df_now = df_edge_trial
                    consecutive_rejections = 0
                    stalled_evaluations = 0
                    trial_batch_size = 1
                    edge_accepted = True
                    log(f"    SmoothPrune {attempt + 1}/{max_attempts}: removed 1 internal edge for fine adjustment.")
                    break
                stalled_evaluations += 1
            if edge_accepted:
                continue

            all_atoms = atoms_prev
            all_edges = edges_prev
            internal_ids = ids_prev
            layer_nodes = layer_prev
            if stop_reason:
                break
            rejected_prune_nodes.update(chosen_set)
            consecutive_rejections += 1
            log(f"    SmoothPrune {attempt + 1}/{max_attempts}: rejected {len(chosen_set)}-node batch; no improvement.")
            if consecutive_rejections >= 3:
                log("  SmoothPrune stopped after three consecutive non-improving trials.")
                break
        if stop_reason:
            log(f"  SmoothPrune stopped by {stop_reason} after {evaluation_count} pressure evaluations.")
        return int(removed), int(removed_edges), df_now

    removed_low_porosity, removed_low_porosity_edges, df_smart = relieve_low_porosity_slices(df_smart)
    if removed_low_porosity:
        process_log.append({
            "step": int(global_step),
            "layer_idx": -1,
            "phase": "STEP5_LOW_POROSITY_PRUNE",
            "removed_nodes": int(removed_low_porosity),
            "removed_edges": int(removed_low_porosity_edges),
            "porosity_threshold": float(INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_THRESHOLD),
        })
        log(
            f"  LowPorosityPrune removed {removed_low_porosity} nodes and "
            f"{removed_low_porosity_edges} incident edges."
        )

    atoms_before_smooth = all_atoms.copy(deep=True)
    edges_before_smooth = [dict(edge) for edge in all_edges]
    ids_before_smooth = set(internal_ids)
    layers_before_smooth = [list(nodes) for nodes in layer_nodes]
    profile_before_smooth = df_smart.copy(deep=True)
    removed_smooth, removed_smooth_edges, df_smart = smooth_prune_overfilled_slices(df_smart)
    if removed_smooth or removed_smooth_edges:
        fine_z = _pressure_feedback_heights(z_bounds)
        fine_nodes = _make_nodes_np_for_analysis(all_atoms, z_bounds)
        fine_profile = run_full_analysis(
            mesh, all_atoms, pd.DataFrame(all_edges), fine_nodes, fine_z,
        )
        if not _capillary_profile_passes(fine_profile):
            fine_margin = _capillary_margin_values(fine_profile)
            log(
                "  SmoothPrune rolled back: final fine-grid fixed-flow transport-margin guard failed; "
                f"lower margin would be {float(np.min(fine_margin)):.3f} Pa."
            )
            all_atoms = atoms_before_smooth
            all_edges = edges_before_smooth
            internal_ids = ids_before_smooth
            layer_nodes = layers_before_smooth
            df_smart = profile_before_smooth
            removed_smooth = 0
            removed_smooth_edges = 0
    if removed_smooth or removed_smooth_edges:
        process_log.append({"step": int(global_step), "layer_idx": -1, "phase": "STEP5_SMOOTH_PRUNE",
                            "removed_nodes": int(removed_smooth), "removed_edges": int(removed_smooth_edges)})
        log(f"  SmoothPrune removed {removed_smooth} nodes and {removed_smooth_edges} edges while preserving capillary passage.")

    ok = _capillary_profile_passes(df_smart)

    export_snapshot("physics_snapshot_after_smartfill.csv")
    process_df = pd.DataFrame(process_log)
    process_df.to_csv(output_dir / "evolution_process_forces.csv", index=False)
    initial_layer_df = process_df[
        process_df.get("phase", pd.Series(index=process_df.index, dtype=object))
        == "STEP1_2_STL_DOMAIN_COMPONENTS"
        ].copy()
    if not initial_layer_df.empty:
        initial_layer_df.insert(
            0, "initial_density_multiplier",
            float(INTERNAL_DYNAMIC_INITIAL_DENSITY_MULTIPLIER),
        )
        initial_layer_df.insert(
            1, "density_multiplier_source",
            str(INTERNAL_DYNAMIC_INITIAL_DENSITY_SOURCE),
        )
        initial_layer_df.to_csv(
            output_dir / "initial_skeleton_layer_parameters.csv", index=False
        )
    pd.DataFrame(internal_log).to_csv(output_dir / "evolution_internal.csv", index=False)

    all_edges_df = _dedup_edges_local(pd.DataFrame(all_edges))
    atoms_int = all_atoms[all_atoms["atom_id"].isin(internal_ids)].copy().reset_index(drop=True)
    internal_set = set(atoms_int["atom_id"].astype(int).tolist())
    if all_edges_df.empty or atoms_int.empty:
        edges_int = pd.DataFrame(columns=["source", "target", "radius", "bond_type", "role"])
    else:
        mask = all_edges_df["source"].astype(int).isin(internal_set) & all_edges_df["target"].astype(int).isin(
            internal_set)
        edges_int = all_edges_df[mask].copy().reset_index(drop=True)
    log(f"  Dynamic internal result: nodes={len(atoms_int)}, edges={len(edges_int)}, final_ok={ok}")
    return atoms_int, edges_int


def _plot_pressure_panel(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    z_plot = df["z_mid_mm"].to_numpy(dtype=float)
    # Display the final slice value up to its upper boundary. This is a
    # plotting extension, not an additional pressure calculation.
    z_upper = float(z_plot[-1] + 0.5 * np.median(np.diff(z_plot))) if len(z_plot) > 1 else float(z_plot[-1])
    has_upper = z_upper > z_plot[-1]
    def through_upper(values):
        return np.append(values, values[-1]) if has_upper else values
    if has_upper:
        z_plot = np.append(z_plot, z_upper)
    p_required = through_upper(df["P_required_Pa"].to_numpy(dtype=float))
    p_hydro = through_upper(df["P_hydro_Pa"].to_numpy(dtype=float))

    p_visc = df.get(
        "P_visc_dynamic_Pa",
        df.get("P_viscous_Pa", pd.Series(np.zeros(len(df))))
    ).to_numpy(dtype=float)
    p_visc = through_upper(p_visc)

    p_required_base = df.get(
        "P_required_base_Pa",
        df["P_hydro_Pa"] + df.get("P_visc_dynamic_Pa", df.get("P_viscous_Pa", 0.0)),
    ).to_numpy(dtype=float)
    p_required_base = through_upper(p_required_base)
    p_cap = through_upper(df["P_cap_Pa"].to_numpy(dtype=float))
    ax.plot(z_plot, p_required_base, color="#A9CAE5", lw=1.8,
            label=r"$P_{\mathrm{hydro}}+P_{\mathrm{visc}}$")
    if not np.allclose(p_required, p_required_base):
        ax.plot(z_plot, p_required, color=FIGURE_GREY, lw=1.2, ls="--",
                label=r"$S(P_{\mathrm{hydro}}+P_{\mathrm{visc}})$")
    ax.plot(z_plot, p_hydro, color="#F3B6B6", lw=1.4, label=r"$P_{\mathrm{hydro}}$")
    ax.plot(z_plot, p_visc, color="#BFC0E2", lw=1.4,
            label=r"$P_{\mathrm{visc}}(Q_{\mathrm{ref}})$")
    ax.plot(z_plot, p_cap, color="#55A6D9", lw=1.6, label=r"$P_{\mathrm{cap}}$")
    ax.fill_between(z_plot, p_required, p_cap, where=p_cap >= p_required,
                    color="#73B6DD", alpha=0.42, label=r"$R_{\mathrm{tr}}$")
    ax.fill_between(z_plot, p_required, p_cap, where=p_cap < p_required,
                    color=FIGURE_PINK, alpha=0.55)
    ax.set_xlabel("h (mm)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_xlim(0.0, z_upper)
    if np.isclose(z_upper, 25.0):
        ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", frameon=False, fontsize=8, handlelength=2.8, handletextpad=0.4)
    _style_internal_axis(ax)
    fig.tight_layout()
    _save_svg_only(fig, save_path)
    plt.close(fig)


def _plot_mobility_panel(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    z_plot = df["z_mid_mm"].to_numpy(dtype=float)
    mobility = df["capillary_viscous_mobility_Pa_m3"].to_numpy(dtype=float)
    ax.plot(z_plot, mobility * 1.0e12, color="#2B7A78", lw=1.8)
    bottleneck = int(np.argmin(mobility))
    ax.scatter([z_plot[bottleneck]], [mobility[bottleneck] * 1.0e12],
               color=FIGURE_PINK, s=28, zorder=3, label="Bottleneck")
    ax.set_xlabel("h (mm)")
    ax.set_ylabel(r"Mobility ($10^{-12}$ Pa m$^3$)")
    ax.set_xlim(float(z_plot.min()), float(z_plot.max()))
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", frameon=False, fontsize=8)
    _style_internal_axis(ax)
    fig.tight_layout()
    _save_svg_only(fig, save_path)
    plt.close(fig)


def export_dynamic_analysis_outputs(mesh, atoms_df: pd.DataFrame, edges_df: pd.DataFrame,
                                    nodes_np: np.ndarray, df_res: pd.DataFrame,
                                    output_dir: Path):
    if df_res is None or df_res.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_stem in (
            "full_physics_chart",
            "plot_xz_wetted_perimeter_heatmap",
            "plot_xz_effective_radius_heatmap",
    ):
        for suffix in (".svg", ".png", ".pdf", ".tiff", ".tif"):
            (output_dir / obsolete_stem).with_suffix(suffix).unlink(missing_ok=True)
    df = df_res.sort_values("z_mm").copy()
    df["z_mid_mm"] = df.get("z_mid_mm", df["z_mm"])
    output_dz = (float(np.median(np.diff(df["z_mid_mm"].to_numpy(dtype=float))))
                 if len(df) > 1 else float(SLICE_DZ_MM))
    df = apply_state_updated_viscous_profile(
        df,
        viscosity_pa_s=FLUID_VISCOSITY_PA_S,
        default_dz_mm=output_dz,
        reference_flow_m3_s=VISC_REF_Q_M3_S,
        target_fill_time_s=TARGET_FILL_TIME_S,
    )
    df["fluid_viscosity_Pa_s"] = FLUID_VISCOSITY_PA_S
    if "A_liq_mm2" not in df.columns and {"A_domain_mm2", "A_solid_mm2"}.issubset(df.columns):
        df["A_liq_mm2"] = np.maximum(1e-9, df["A_domain_mm2"] - df["A_solid_mm2"])
    if "safety_factor_z" not in df.columns:
        z_bounds_df = (float(df["z_mid_mm"].min()), float(df["z_mid_mm"].max()))
        df["safety_factor_z"] = df["z_mid_mm"].map(lambda z: _safety_factor_at_z(float(z), z_bounds_df))
    df["Safety_Margin"] = df["P_cap_Pa"] - df["P_required_Pa"]
    band_mask = df["z_mid_mm"] >= float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM)
    df["P_cap_target_lower_Pa"] = (
            _capillary_lower_ratio_values(
                df, include_assembly_reserve=False)
            * df["P_required_Pa"].to_numpy(dtype=float)
    )
    df["P_cap_target_preassembly_Pa"] = (
            _capillary_lower_ratio_values(
                df, include_assembly_reserve=True)
            * df["P_required_Pa"].to_numpy(dtype=float)
    )
    df["P_cap_target_upper_Pa"] = np.where(
        band_mask,
        float(INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO) * df["P_required_Pa"],
        np.nan,
    )
    df["Lower_Band_Margin_Pa"] = df["P_cap_Pa"] - df["P_cap_target_lower_Pa"]
    df["Preassembly_Reserve_Margin_Pa"] = (
            df["P_cap_Pa"] - df["P_cap_target_preassembly_Pa"])
    df["Capillary_Ratio"] = df["P_cap_Pa"] / np.maximum(df["P_required_Pa"], 1.0e-9)
    cap_arr = np.maximum(df["P_cap_Pa"].to_numpy(dtype=float), 1e-9)
    if len(cap_arr) >= 2:
        adjacent_ratios = np.maximum(cap_arr[1:] / cap_arr[:-1], cap_arr[:-1] / cap_arr[1:])
        adjacent_steps = np.abs(np.diff(cap_arr))
        max_adjacent_ratio = float(np.max(adjacent_ratios))
        max_adjacent_step = float(np.max(adjacent_steps))
    else:
        max_adjacent_ratio = 1.0
        max_adjacent_step = 0.0
    smoothness_summary = pd.DataFrame([{
        "capillary_all_pass": _capillary_profile_passes(
            df, include_assembly_reserve=False),
        "preassembly_reserve_pass": _capillary_profile_passes(
            df, include_assembly_reserve=True),
        "worst_safety_margin_Pa": float(df["Safety_Margin"].min()),
        "worst_margin_z_mm": float(df.loc[df["Safety_Margin"].idxmin(), "z_mid_mm"]),
        "max_P_cap_Pa": float(df["P_cap_Pa"].max()),
        "capillary_min_ratio": float(INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO),
        "capillary_target_ratio": float(INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO),
        "capillary_band_start_mm": float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM),
        "slices_below_capillary_min": int((df["P_cap_Pa"] < df["P_cap_target_lower_Pa"]).sum()),
        "slices_above_capillary_target": int((df["P_cap_Pa"] > df["P_cap_target_upper_Pa"]).sum()),
        "max_adjacent_P_cap_step_Pa": max_adjacent_step,
        "max_adjacent_P_cap_ratio": max_adjacent_ratio,
        "min_A_liq_mm2": float(df["A_liq_mm2"].min()),
        "min_porosity": float(df["porosity"].min()) if "porosity" in df.columns else np.nan,
        "smooth_ratio_limit": float(INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO),
        "min_liquid_area_limit_mm2": float(INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2),
        "minimum_capillary_viscous_mobility_Pa_m3": _minimum_viscous_mobility(df),
        "smoothness_ok": bool(max_adjacent_ratio <= INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO and df[
            "A_liq_mm2"].min() >= INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2),
    }])
    force_cols = ["z_mid_mm", "P_wet_mm", "P_cap_Pa", "P_hydro_Pa", "safety_factor_z",
                  "P_visc_dynamic_Pa", "P_visc_ref_Pa", "P_viscous_Pa", "P_viscous_capacity_proxy_Pa",
                  "P_required_base_Pa", "P_required_Pa", "transport_margin_Pa", "R_tr_Pa", "P_cap_target_lower_Pa",
                  "P_cap_target_upper_Pa",
                  "Capillary_Ratio", "Safety_Margin", "Lower_Band_Margin_Pa", "r_eff_mm", "A_liq_mm2",
                  "axial_radius4_sum_mm4", "n_flow_components",
                  "R_geom_segment_m-3", "R_geom_cumulative_m-3", "R_hyd_cumulative_Pa_s_m-3",
                  "P_visc_dynamic_Pa", "P_visc_ref_Pa", "P_viscous_Pa", "P_required_base_Pa", "Q_required_m3_s",
                  "Q_required_source", "reference_flow_rate_m3_s",
                  "available_pressure_head_Pa", "reference_flow_residual_head_Pa", "flow_capacity_ratio",
                  "capillary_viscous_mobility_Pa_m3",
                  "Q_available_proxy_m3_s",
                  "capillary_stall_proxy", "resistance_calibration_factor", "fluid_viscosity_Pa_s"]
    force_cols = [col for col in force_cols if col in df.columns]
    por_cols = ["z_mid_mm", "porosity", "P_wet_mm", "A_domain_mm2", "A_solid_mm2", "A_liq_mm2"]
    df[force_cols].to_csv(output_dir / "structure_force_summary.csv", index=False)
    df[por_cols].to_csv(output_dir / "structure_porosity_profile.csv", index=False)
    smoothness_summary.to_csv(output_dir / "structure_smoothness_summary.csv", index=False)

    if atoms_df is not None and not atoms_df.empty:
        nodes_export = atoms_df.copy().reset_index(drop=True)
        zmin_n, zmax_n = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
        layer_h_n = max((zmax_n - zmin_n) / max(1, INTERNAL_DYNAMIC_LAYERS), 1e-9)
        cache_n: Dict[float, object] = {}

        def node_kind(name: str) -> str:
            return "surface" if "surface" in str(name).lower() else "internal"

        nodes_export["node_kind"] = nodes_export["face_name"].astype(str).map(node_kind)
        nodes_export["dynamic_layer"] = np.clip(
            np.floor((nodes_export["z"].astype(float) - zmin_n) / layer_h_n).astype(int),
            0,
            max(0, INTERNAL_DYNAMIC_LAYERS - 1),
        )
        comp_ids = []
        for row in nodes_export.itertuples(index=False):

            if str(getattr(row, "node_kind", "")) == "surface":
                comp_ids.append(-1)
                continue
            z_node = float(getattr(row, "z"))
            domain_node = _domain_at_z_cached(mesh, z_node, cache_n)
            comp = _point_component_index(domain_node, float(getattr(row, "x")), float(getattr(row, "y")))
            comp_ids.append(-1 if comp is None else int(comp))
        nodes_export["slice_component"] = comp_ids
        nodes_export["slice_component_scope"] = np.where(
            nodes_export["node_kind"].eq("internal"),
            "internal_slice_component",
            "not_applicable_surface_uses_patch_provenance",
        )
        nodes_export.to_csv(output_dir / "structure_nodes.csv", index=False)

    if edges_df is not None and not edges_df.empty:
        _dedup_edges_local(edges_df.copy()).to_csv(output_dir / "structure_edges.csv", index=False)

    if edges_df is not None and not edges_df.empty and atoms_df is not None and not atoms_df.empty:
        atom_map = atoms_df.set_index("atom_id")[["x", "y", "z"]].to_dict("index")
        audit_rows = []
        zmin_a, zmax_a = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
        layer_h_a = max((zmax_a - zmin_a) / max(1, INTERNAL_DYNAMIC_LAYERS), 1e-9)
        for edge in edges_df.itertuples(index=False):
            u = int(getattr(edge, "source"))
            v = int(getattr(edge, "target"))
            if u not in atom_map or v not in atom_map:
                continue
            a, b = atom_map[u], atom_map[v]
            dx = float(a["x"]) - float(b["x"])
            dy = float(a["y"]) - float(b["y"])
            dz_e = float(a["z"]) - float(b["z"])
            dxy = math.hypot(dx, dy)
            d3 = math.sqrt(dx * dx + dy * dy + dz_e * dz_e)
            z_mid_e = 0.5 * (float(a["z"]) + float(b["z"]))
            pitch_e = _local_internal_pitch(z_mid_e, (zmin_a, zmax_a))
            max_len_e = max(layer_h_a, pitch_e) * INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR
            audit_rows.append({
                "source": u,
                "target": v,
                "bond_type": getattr(edge, "bond_type", ""),
                "role": getattr(edge, "role", ""),
                "length_3d_mm": d3,
                "dxy_mm": dxy,
                "dz_mm": abs(dz_e),
                "max_length_mm": max_len_e,
                "over_max_length": bool(d3 > max_len_e + 1e-9),
            })
        pd.DataFrame(audit_rows).to_csv(output_dir / "structure_edge_audit.csv", index=False)

    _plot_pressure_panel(df, output_dir / "plot_forces.svg")
    _plot_mobility_panel(df, output_dir / "plot_capillary_viscous_mobility.svg")

    fig2 = plt.figure(figsize=(8, 5), dpi=600)
    ax2 = plt.gca()
    ax2r = ax2.twinx()
    lc_por = _plot_gradient_line(ax2, df["z_mid_mm"], df["porosity"], linewidth=3.0)
    ax2.axhline(y=INTERNAL_DYNAMIC_OVERFILL_POROSITY_FLOOR, color=FIGURE_GREY, linestyle="--", linewidth=1.4, alpha=0.8)
    ax2r.plot(df["z_mid_mm"], df["P_wet_mm"], label="Wetted perimeter", color=FIGURE_RED, linestyle="-.", linewidth=2.0)
    ax2.set_xlabel("Z (mm)")
    ax2.set_ylabel("Porosity")
    ax2r.set_ylabel("P_wet (mm)")
    _style_internal_axis(ax2)
    _style_internal_axis(ax2r)
    fig2.tight_layout()
    _save_svg_only(fig2, output_dir / "plot_porosity_profile.svg")
    plt.close(fig2)

    _export_topology_profile(atoms_df, edges_df, output_dir, (float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])))
    _export_xz_local_profile_stl(mesh, atoms_df, edges_df, nodes_np, output_dir, profile_df=df)
    _export_process_evolution_plot(output_dir)


def _export_topology_profile(atoms_df: pd.DataFrame, edges_df: pd.DataFrame,
                             output_dir: Path, z_bounds: Tuple[float, float]):
    zmin, zmax = z_bounds
    dz = max(1.0, (zmax - zmin) / 20.0)
    bins = np.arange(zmin, zmax + dz, dz)
    if len(bins) < 2:
        return
    centers = 0.5 * (bins[:-1] + bins[1:])
    atoms = atoms_df.copy()
    atoms["bin_idx"] = pd.cut(atoms["z"], bins=bins, labels=False, include_lowest=True)
    v_counts = atoms.groupby("bin_idx").size().reindex(range(len(centers)), fill_value=0)
    z_map = atoms_df.set_index("atom_id")["z"].to_dict()
    edge_mids = []
    for row in edges_df.itertuples(index=False):
        z1 = z_map.get(row.source)
        z2 = z_map.get(row.target)
        if z1 is not None and z2 is not None:
            edge_mids.append(0.5 * (float(z1) + float(z2)))
    e_bins = pd.cut(pd.Series(edge_mids), bins=bins, labels=False, include_lowest=True)
    e_counts = e_bins.value_counts().reindex(range(len(centers)), fill_value=0)
    df_topo = pd.DataFrame({"z_mid_mm": centers, "V_count": v_counts.values, "E_count": e_counts.values})
    df_topo["Chi_local"] = df_topo["V_count"] - df_topo["E_count"]
    df_topo["Avg_Z"] = np.where(df_topo["V_count"] > 0, 2.0 * df_topo["E_count"] / df_topo["V_count"], 0.0)
    df_topo.to_csv(output_dir / "structure_topology.csv", index=False)

    chi_total = len(atoms_df) - len(edges_df)
    avg_coord = 2.0 * len(edges_df) / max(len(atoms_df), 1)
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, dpi=600)
    axes[0].plot(df_topo["z_mid_mm"], df_topo["V_count"], label="Nodes", color=FIGURE_BLUE, linewidth=2.4)
    axes[0].plot(df_topo["z_mid_mm"], df_topo["E_count"], label="Edges", color=FIGURE_RED, linewidth=2.4)
    axes[0].set_title(f"Topological Elements (global chi={chi_total})")
    axes[0].legend()
    _plot_gradient_line(axes[1], df_topo["z_mid_mm"], df_topo["Chi_local"], linewidth=2.8)
    axes[1].axhline(0, color=FIGURE_GREY, linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("V - E")
    _plot_gradient_line(axes[2], df_topo["z_mid_mm"], df_topo["Avg_Z"], linewidth=2.8)
    axes[2].axhline(avg_coord, color=FIGURE_GREY, linestyle="--", linewidth=1.2)
    axes[2].set_ylabel("Avg coordination")
    axes[2].set_xlabel("Z (mm)")
    for axis in axes:
        _style_internal_axis(axis)
    fig.tight_layout()
    _save_svg_only(fig, output_dir / "plot_topology.svg")
    plt.close(fig)


def _export_xz_local_profile_stl(mesh, atoms_df: pd.DataFrame, edges_df: pd.DataFrame,
                                 nodes_np: np.ndarray, output_dir: Path,
                                 profile_df: Optional[pd.DataFrame] = None):
    zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    xmin, xmax = float(mesh.bounds[0, 0]), float(mesh.bounds[1, 0])
    ymin, ymax = float(mesh.bounds[0, 1]), float(mesh.bounds[1, 1])
    dz = max(INTERNAL_DYNAMIC_PHYSICS_DZ_MM, SLICE_DZ_MM)
    x_edges = np.linspace(xmin, xmax, INTERNAL_DYNAMIC_XZ_BINS + 1)
    cache: Dict[float, object] = {}
    rows = []
    for z0 in tqdm(np.arange(zmin, zmax, dz), desc="XZ local profile"):
        z_mid = float(min(z0 + 0.5 * dz, zmax))
        domain = _domain_at_z_cached(mesh, z_mid, cache)
        if domain.is_empty:
            continue
        solid = _generate_true_solid_geometry_at_z_full(nodes_np, edges_df, atoms_df, z_mid)
        try:
            solid = solid.intersection(domain) if solid is not None and not solid.is_empty else GeometryCollection()
        except Exception:
            solid = GeometryCollection()
        for i in range(INTERNAL_DYNAMIC_XZ_BINS):
            x0, x1 = float(x_edges[i]), float(x_edges[i + 1])
            strip_domain = domain.intersection(shapely_box(x0, ymin, x1, ymax))
            if strip_domain.is_empty:
                continue
            solid_local = solid.intersection(strip_domain) if not solid.is_empty else GeometryCollection()
            A_solid = float(solid_local.area) if not solid_local.is_empty else 0.0
            A_liq = max(1e-9, float(strip_domain.area) - A_solid)
            try:
                P_wet = float(solid_local.boundary.intersection(
                    strip_domain.buffer(-1e-6)).length) if not solid_local.is_empty else 0.0
            except Exception:
                P_wet = 0.0
            rows.append({
                "z_mid_mm": z_mid,
                "x_bin_idx": int(i),
                "x_bin_left_mm": x0,
                "x_bin_right_mm": x1,
                "x_bin_center_mm": 0.5 * (x0 + x1),
                "strip_area_mm2": float(strip_domain.area),
                "A_solid_local_mm2": A_solid,
                "A_liq_local_mm2": A_liq,
                "P_wet_local_mm": P_wet,
                "r_eff_local_mm": float((2.0 * A_liq / P_wet) if P_wet > 1e-9 else 0.0),
            })
    df_xz = pd.DataFrame(rows)
    df_xz.to_csv(output_dir / "structure_xz_local_profile.csv", index=False)
    if df_xz.empty:
        return
    if profile_df is None or profile_df.empty:
        return
    _plot_xz_profile_panel(
        df_xz, profile_df, "P_wet_local_mm", "P_wet_mm",
        "Wetted perimeter (mm)", output_dir / "plot_wetted_perimeter.svg",
    )
    _plot_xz_profile_panel(
        df_xz, profile_df, "r_eff_local_mm", "r_eff_mm",
        "Effective radius (mm)", output_dir / "plot_effective_radius.svg",
        reference_value=R_EFF_LIMIT_MM,
    )


def _plot_xz_profile_panel(
        df: pd.DataFrame,
        profile_df: pd.DataFrame,
        value_col: str,
        line_col: str,
        value_label: str,
        save_path: Path,
        reference_value: Optional[float] = None,
        color_percentile_low: float = 5.0,
        color_percentile_high: float = 95.0,
):
    pivot = df.pivot_table(
        index="x_bin_center_mm",
        columns="z_mid_mm",
        values=value_col,
        aggfunc="mean",
    )

    if pivot.empty:
        return

    x_vals = pivot.index.to_numpy(dtype=float)
    z_vals = pivot.columns.to_numpy(dtype=float)

    raw_values = pivot.to_numpy(dtype=float)
    values = np.ma.masked_invalid(raw_values)

    valid_values = raw_values[np.isfinite(raw_values)]

    if valid_values.size > 0:
        vmin = float(np.percentile(
            valid_values,
            color_percentile_low
        ))
        vmax = float(np.percentile(
            valid_values,
            color_percentile_high
        ))

        if (not np.isfinite(vmin)
                or not np.isfinite(vmax)
                or vmax <= vmin + 1.0e-12):
            vmin = float(np.min(valid_values))
            vmax = float(np.max(valid_values))

            if vmax <= vmin + 1.0e-12:
                vmax = vmin + 1.0e-12
    else:
        vmin = 0.0
        vmax = 1.0

    profile = profile_df.sort_values("z_mid_mm")

    fig, (ax_map, ax_line) = plt.subplots(
        2, 1,
        figsize=(5.2, 4.1),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.05, 0.90],
            "hspace": 0.08,
        },
    )

    image_artist = ax_map.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[
            float(z_vals.min()),
            float(z_vals.max()),
            float(x_vals.min()),
            float(x_vals.max()),
        ],
        cmap=FIGURE_CMAP,
        interpolation="nearest",

        vmin=vmin,
        vmax=vmax,
    )

    ax_map.set_ylabel("x (mm)")
    ax_map.tick_params(labelbottom=False)

    colorbar = fig.colorbar(
        image_artist,
        ax=ax_map,
        pad=0.025,
        fraction=0.045,
        extend="both",
    )
    colorbar.set_label(value_label)

    _plot_gradient_line(
        ax_line,
        profile["z_mid_mm"],
        profile[line_col],
        linewidth=2.8,
    )

    if reference_value is not None:
        ax_line.axhline(
            reference_value,
            color=FIGURE_GREY,
            ls="--",
            lw=1.1,
            alpha=0.8,
        )

    ax_line.set_xlabel("h (mm)")
    ax_line.set_ylabel(value_label)
    ax_line.set_xlim(
        float(z_vals.min()),
        float(z_vals.max()),
    )

    _style_internal_axis(ax_map)
    _style_internal_axis(ax_line)

    fig.subplots_adjust(
        left=0.15,
        right=0.90,
        bottom=0.14,
        top=0.98,
    )

    _save_svg_only(fig, save_path)
    plt.close(fig)


def _export_process_evolution_plot(output_dir: Path):
    csv_path = output_dir / "evolution_process_forces.csv"
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    if df.empty or "step" not in df.columns:
        return
    fig = plt.figure(figsize=(9, 5), dpi=600)
    ax = plt.gca()
    if "layer_nodes" in df.columns:
        d_layer = df.dropna(subset=["layer_nodes"])
        if not d_layer.empty:
            ax.plot(d_layer["step"], d_layer["layer_nodes"], marker="o", label="Layer nodes", color=FIGURE_BLUE,
                    linewidth=2.2)
    if "added" in df.columns:
        d_added = df.dropna(subset=["added"])
        if not d_added.empty:
            ax.plot(d_added["step"], d_added["added"], marker=".", label="Smart fill added", color=FIGURE_RED,
                    linewidth=2.0)
    if "added_nodes" in df.columns:
        d_gap = df.dropna(subset=["added_nodes"])
        if not d_gap.empty:
            ax.scatter(d_gap["step"], d_gap["added_nodes"], label="Physics gapfill added", color=FIGURE_PINK,
                       edgecolors=FIGURE_GREY, linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.set_title("Dynamic Construction Evolution")
    ax.legend()
    _style_internal_axis(ax)
    fig.tight_layout()
    _save_svg_only(fig, output_dir / "plot_process_evolution.svg")
    plt.close(fig)


def export_stl(atoms, edges, path):
    log(f"Exporting STL: {path.name}...")
    if atoms.empty or edges.empty:
        log("  -> Empty data, skipping.")
        return

    pos = atoms.set_index("atom_id")[["x", "y", "z"]].to_dict("index")
    meshes = []

    def _rod(p0, p1, r):
        v = p1 - p0;
        h = np.linalg.norm(v)
        if h < 1e-9: return None
        cyl = trimesh.creation.cylinder(radius=r, height=h, sections=STL_ROD_RADIAL_SECTIONS)
        cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], v))
        cyl.apply_translation((p0 + p1) / 2)
        return cyl

    for r in tqdm(edges.to_dict('records'), desc=f"Meshing {path.stem}"):
        if r['source'] not in pos or r['target'] not in pos: continue
        p1 = np.array([pos[r['source']][k] for k in 'xyz'])
        p2 = np.array([pos[r['target']][k] for k in 'xyz'])
        m = _rod(p1, p2, r['radius'])
        if m: meshes.append(m)

    z_bounds = (0.0, max(float(TARGET_HEIGHT_MM), float(atoms['z'].max())))
    nodes_np = _make_nodes_np_for_analysis(atoms, z_bounds)
    for x, y, z, radius in tqdm(nodes_np, desc=f"Meshing {path.stem} internal nodes"):
        sphere = trimesh.creation.icosphere(
            subdivisions=STL_NODE_SPHERE_SUBDIVISIONS,
            radius=float(radius),
        )
        sphere.apply_translation([float(x), float(y), float(z)])
        meshes.append(sphere)

    if meshes:
        trimesh.util.concatenate(meshes).export(path)
        log("  -> Done.")


def audit_final_stl_file(path: Path, input_domain: trimesh.Trimesh,
                         final_atoms: pd.DataFrame,
                         final_edges: pd.DataFrame) -> Dict:
    """Stream-check the exact exported binary STL; never re-mesh or boolean-union it."""
    path = Path(path)
    result = dict(stl_file=path.name, exists=path.is_file(),
                  binary_file_format_valid=False, finite_vertices=False,
                  bounding_box_within_input_domain_bbox=False,
                  exact_3d_pore_measurement_performed=False,
                  watertight_union_verified=False,
                  node_spheres_exported=False)
    if not path.is_file():
        return result
    length = path.stat().st_size
    if length < 84:
        result['error'] = 'STL shorter than 84 bytes'
        return result
    tri_dtype = np.dtype([('normal', '<f4', (3,)),
                          ('vertex', '<f4', (3, 3)), ('attr', '<u2')])
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    finite = True
    triangles = 0
    with path.open('rb') as fh:
        fh.read(80)
        declared = int(np.frombuffer(fh.read(4), dtype='<u4')[0])
        result['declared_triangle_count'] = declared
        result['actual_file_bytes'] = length
        result['expected_file_bytes'] = 84 + 50 * declared
        if length != 84 + 50 * declared:
            result['error'] = 'Binary STL byte length does not match its declared triangle count'
            return result
        while True:
            blob = fh.read(50 * 25000)
            if not blob:
                break
            if len(blob) % 50:
                result['error'] = 'Incomplete STL triangle block'
                return result
            block = np.frombuffer(blob, dtype=tri_dtype)
            vertices = block['vertex']
            finite = finite and bool(np.isfinite(vertices).all())
            if not finite:
                break
            mins = np.minimum(mins, vertices.min(axis=(0, 1)))
            maxs = np.maximum(maxs, vertices.max(axis=(0, 1)))
            triangles += len(block)
    result['binary_file_format_valid'] = bool(finite and triangles == declared)
    result['finite_vertices'] = bool(finite and triangles == declared)
    result['triangle_count'] = triangles
    result['mesh_bounds_mm'] = [mins.tolist(), maxs.tolist()] if triangles else []
    domain_bounds = np.asarray(input_domain.bounds, dtype=float)
    result['input_domain_bounds_mm'] = domain_bounds.tolist()
    result['bounding_box_within_input_domain_bbox'] = bool(triangles and
                                                           np.all(mins >= domain_bounds[0] - 1e-4) and np.all(
        maxs <= domain_bounds[1] + 1e-4))
    lower_excess = np.maximum(domain_bounds[0] - mins, 0.0) if triangles else np.zeros(3)
    upper_excess = np.maximum(maxs - domain_bounds[1], 0.0) if triangles else np.zeros(3)
    result['finite_radius_boundary_excess_mm'] = {
        'below_input_min_xyz': lower_excess.tolist(),
        'above_input_max_xyz': upper_excess.tolist(),
        'maximum': float(max(lower_excess.max(), upper_excess.max())),
    }
    result['boundary_excess_interpretation'] = (
        'Finite-radius rod end caps and internal node spheres extend around graph '
        'centrelines placed on or near the design boundary. Hydraulic elevation uses '
        'the centreline as its reference, with local wetting-position uncertainty bounded '
        'by the local solid radius; this does not redefine the physical STL envelope.'
    )
    result['graph_node_count'] = int(len(final_atoms))
    result['graph_edge_segment_count'] = int(len(final_edges))
    atom_xyz = final_atoms.set_index('atom_id')[["x", "y", "z"]].to_dict('index')
    exported_rods = 0
    for edge in final_edges.itertuples(index=False):
        p0 = atom_xyz.get(int(edge.source))
        p1 = atom_xyz.get(int(edge.target))
        if p0 is None or p1 is None:
            continue
        if np.linalg.norm(
                np.asarray([p1[k] for k in 'xyz'], dtype=float) -
                np.asarray([p0[k] for k in 'xyz'], dtype=float)
        ) >= 1e-9:
            exported_rods += 1
    rod_facets = int(4 * STL_ROD_RADIAL_SECTIONS * exported_rods)
    face_names = final_atoms.get('face_name', pd.Series('', index=final_atoms.index)).astype(str)
    exported_sphere_count = int((~face_names.str.contains('surface', case=False, regex=False)).sum())
    sphere_facets = int(20 * (4 ** STL_NODE_SPHERE_SUBDIVISIONS) * exported_sphere_count)
    result['exported_nonzero_rod_count'] = exported_rods
    result['exported_internal_node_sphere_count'] = exported_sphere_count
    result['surface_node_spheres_exported'] = False
    result['internal_node_spheres_exported'] = bool(exported_sphere_count > 0)
    result['expected_rod_triangle_count_if_all_valid'] = rod_facets
    result['expected_node_sphere_triangle_count'] = sphere_facets
    result['facets_match_rods_plus_node_spheres'] = bool(triangles == rod_facets + sphere_facets)
    result['node_spheres_exported'] = bool(triangles >= sphere_facets and exported_sphere_count > 0)
    result['export_rod_sections'] = STL_ROD_RADIAL_SECTIONS
    result['export_node_sphere_subdivisions'] = STL_NODE_SPHERE_SUBDIVISIONS

    return result


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Build a capillary graph structure inside an STL-derived, height-scaled design domain."
    )
    parser.add_argument("--stl", type=Path, default=IN_STL, help="Input STL design-domain model.")
    parser.add_argument("--height-mm", type=float, default=TARGET_HEIGHT_MM,
                        help="Target design height in mm; X/Y scale with Z.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory. Default: <stl_stem>_dynamic_out beside the STL.")
    parser.add_argument("--build-mode", choices=["dynamic", "voxel"], default=INTERNAL_BUILD_MODE)
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help="Generate only the internal graph and stop after the internal-only analysis branch.",
    )
    parser.add_argument(
        "--surface-only",
        action="store_true",
        help=("Generate and audit only the surface graph. Internal construction and "
              "dual-domain stitching are skipped."),
    )
    parser.add_argument(
        "--internal-matched-random-position", action="store_true",
        default=INTERNAL_MATCHED_RANDOM_POSITION,
        help=("Randomize feasible local update positions while retaining the same "
              "initial skeleton, physical failure schedule, graph productions, "
              "geometry guards, and acceptance checks."),
    )
    parser.add_argument("--internal-matched-node-budget", type=int, default=0,
                        help="Maximum internal nodes for the matched random-position control; 0 disables.")
    parser.add_argument("--internal-matched-edge-budget", type=int, default=0,
                        help="Maximum internal edges for the matched random-position control; 0 disables.")
    parser.add_argument(
        "--surface-face-scan",
        action="store_true",
        help=("Before the selected workflow, scan descending surface target-face counts "
              "and choose the highest count that passes the local non-seam capillary gate "
              "and the seam-burden screen."),
    )
    checkpoint_load = parser.add_mutually_exclusive_group()
    checkpoint_load.add_argument(
        "--load-initial-skeleton", type=Path, default=INITIAL_SKELETON_CHECKPOINT_LOAD, metavar="DIR",
        help="Resume after initial skeleton placement from a checkpoint directory.")
    checkpoint_load.add_argument(
        "--load-pre-smartfill", type=Path, default=PRE_SMARTFILL_CHECKPOINT_LOAD, metavar="DIR",
        help="Resume immediately before SmartFill from a checkpoint directory.")
    parser.add_argument(
        "--save-initial-skeleton", type=Path, default=INITIAL_SKELETON_CHECKPOINT_SAVE, metavar="DIR",
        help="Save a reusable initial-skeleton checkpoint directory.")
    parser.add_argument(
        "--save-pre-smartfill", type=Path, default=PRE_SMARTFILL_CHECKPOINT_SAVE, metavar="DIR",
        help="Save a reusable checkpoint immediately before SmartFill.")
    parser.add_argument(
        "--stop-after-checkpoint", choices=["initial-skeleton", "pre-smartfill"], default=CHECKPOINT_STOP_AFTER,
        help="Stop successfully after saving the selected checkpoint stage.")
    parser.add_argument(
        "--checkpoint-ignore-config", action="store_true", default=CHECKPOINT_IGNORE_CONFIG,
        help="Allow loading despite pre-stage option differences (unsafe; mesh identity is still enforced).")
    surface_group = parser.add_mutually_exclusive_group()
    surface_group.add_argument(
        "--surface-design", dest="surface_design", action="store_true", default=None,
        help="Enable surface-domain construction and surface-to-core stitching for this run.",
    )
    surface_group.add_argument(
        "--no-surface-design", dest="surface_design", action="store_false",
        help="Disable surface-domain construction and stitching while retaining the full internal workflow.",
    )

    parser.add_argument("--gamma", type=float, default=GAMMA_NPM, help="Surface tension, N/m.")
    parser.add_argument("--contact-angle", type=float, default=CONTACT_ANGLE_DEG, help="Contact angle, degrees.")
    parser.add_argument("--rho", type=float, default=RHO_LIQ, help="Liquid density, kg/m^3.")
    parser.add_argument("--surface-gas-density", type=float, default=RHO_GAS,
                        help="Gas density used by the surface pressure criterion, kg/m^3.")
    parser.add_argument("--gravity", type=float, default=G_GRAV, help="Gravity, m/s^2.")
    parser.add_argument("--safety-factor", type=float, default=SAFETY_FACTOR)
    parser.add_argument("--margin-correction-factor", type=float, default=None,
                        help="Alias for --safety-factor; applied to P_hydro + P_visc(Q_ref).")
    parser.add_argument("--safety-factor-bottom", type=float, default=SAFETY_FACTOR_BOTTOM,
                        help="Bottom safety factor; linearly transitions to --safety-factor-top.")
    parser.add_argument("--safety-factor-top", type=float, default=SAFETY_FACTOR_TOP,
                        help="Top safety factor; linearly transitions from --safety-factor-bottom.")
    parser.add_argument("--viscosity", type=float, default=FLUID_VISCOSITY_PA_S, help="Fluid viscosity, Pa*s.")
    parser.add_argument("--reference-flow-rate", type=float, default=VISC_REF_Q_M3_S,
                        help="Fixed reference flow used to calculate viscous pressure loss, m^3/s.")
    parser.add_argument("--flow-resistance-calibration", type=float,
                        default=FLOW_RESISTANCE_CALIBRATION_FACTOR,
                        help="Deprecated compatibility option. Original viscous model is unscaled; value must remain 1.0.")
    parser.add_argument("--slice-dz", type=float, default=SLICE_DZ_MM,
                        help="Fixed nominal pressure-slice thickness in mm; default 1.0 mm. The last bin may be shorter.")
    parser.add_argument("--pressure-slices", type=int, default=None,
                        help="Deprecated compatibility argument; slice count is derived from design height / --slice-dz.")
    parser.add_argument("--physics-dz", type=float, default=None,
                        help="Deprecated compatibility argument; pressure feedback uses --slice-dz.")

    parser.add_argument("--surface-diam", type=float, default=SURFACE_DIAM_MM)
    parser.add_argument("--surface-faces", type=int, default=INITIAL_FACE_COUNT)
    parser.add_argument(
        "--surface-max-faces-per-patch", type=int, default=SURFACE_MAX_FACES_PER_PATCH,
        help=("Maximum carrier triangles per initial surface patch. Larger values can reduce "
              "artificial seams; distortion checks can still split a patch afterward."),
    )
    parser.add_argument(
        "--surface-seam-max-fraction", type=float, default=SURFACE_SEAM_MAX_FRACTION,
        help=("Maximum accepted fraction of local pore records touching an artificial "
              "inter-patch seam. This is a screening threshold, not a global pore proof."),
    )
    parser.add_argument("--surface-face-scan-start", type=int, default=SURFACE_FACE_SCAN_START)
    parser.add_argument("--surface-face-scan-stop", type=int, default=SURFACE_FACE_SCAN_STOP)
    parser.add_argument("--surface-face-scan-step", type=int, default=SURFACE_FACE_SCAN_STEP)
    parser.add_argument("--surface-rounds", type=int, default=MAX_SPLIT_ROUNDS)
    parser.add_argument(
        "--surface-reff-limit",
        type=float,
        default=None,
        help="Manual equivalent-capillary-radius limit in mm; default derives it from the pressure criterion.",
    )
    parser.add_argument(
        "--surface-extra-pressure-pa",
        type=float,
        default=SURFACE_EXTRA_PRESSURE_PA,
        help="Additional operating pressure in the surface-domain requirement, Pa.",
    )
    parser.add_argument(
        "--surface-safety-factor",
        type=float,
        default=SURFACE_CAPILLARY_SAFETY_FACTOR,
        help="Surface-domain capillary safety factor used to derive the surface capillary limit.",
    )
    parser.add_argument("--surface-offset", type=float, default=SURFACE_OUTWARD_OFFSET_MM)
    parser.add_argument("--stitch-distance", type=float, default=STITCH_DISTANCE_MM)
    parser.add_argument("--stitch-anchor-spacing", type=float, default=STITCH_ANCHOR_SPACING_MM)
    parser.add_argument("--stitch-relaxed-distance", type=float, default=STITCH_RELAXED_DISTANCE_MM,
                        help="Bounded fallback stitch search length; angle and support guards are unchanged.")
    parser.add_argument("--bottom-pillars", dest="bottom_pillars", action="store_true", default=BOTTOM_PILLAR_ENABLE)
    parser.add_argument("--no-bottom-pillars", dest="bottom_pillars", action="store_false")
    parser.add_argument("--bottom-pillar-rows", type=int, default=BOTTOM_PILLAR_ROWS)
    parser.add_argument("--bottom-pillar-max-anchors", type=int, default=BOTTOM_PILLAR_MAX_ANCHORS)
    parser.add_argument("--bottom-pillar-spacing", type=float, default=BOTTOM_PILLAR_ANCHOR_SPACING_MM)
    parser.add_argument("--bottom-pillar-height-layers", type=float, default=BOTTOM_PILLAR_HEIGHT_LAYERS)
    parser.add_argument("--bottom-pillar-radius-factor", type=float, default=BOTTOM_PILLAR_RADIUS_FACTOR)
    parser.add_argument("--bottom-pillar-min-component-area", type=float, default=BOTTOM_PILLAR_MIN_COMPONENT_AREA_MM2)

    parser.add_argument("--layers", type=int, default=INTERNAL_DYNAMIC_LAYERS)
    parser.add_argument("--seed", type=int, default=INTERNAL_DYNAMIC_SEED)
    parser.add_argument("--min-nodes", type=int, default=INTERNAL_DYNAMIC_MIN_NODES_PER_LAYER)
    parser.add_argument("--max-nodes", type=int, default=INTERNAL_DYNAMIC_MAX_NODES_PER_LAYER)
    parser.add_argument("--bottom-nodes", type=int, default=INTERNAL_DYNAMIC_BOTTOM_NODE_TARGET)
    parser.add_argument("--top-nodes", type=int, default=INTERNAL_DYNAMIC_TOP_NODE_TARGET)
    parser.add_argument("--density-factor", type=float, default=INTERNAL_DYNAMIC_DENSITY_FACTOR)
    parser.add_argument(
        "--initial-density-multiplier", type=float, default=None,
        help=("Scale every initial skeleton layer target after the height/area rules; "
              "if omitted, the configured design-domain value is used."),
    )
    parser.add_argument("--area-target-exponent", type=float, default=INTERNAL_DYNAMIC_AREA_TARGET_EXPONENT)
    parser.add_argument("--area-target-min-scale", type=float, default=INTERNAL_DYNAMIC_AREA_TARGET_MIN_SCALE)
    parser.add_argument("--min-component-area", type=float, default=INTERNAL_DYNAMIC_MIN_COMPONENT_AREA_MM2)
    parser.add_argument("--step1-fraction", type=float, default=INTERNAL_DYNAMIC_STEP1_FRACTION)
    parser.add_argument("--max-tries", type=int, default=INTERNAL_DYNAMIC_MAX_TRIES_PER_LAYER)
    parser.add_argument("--candidates", type=int, default=INTERNAL_DYNAMIC_CANDIDATES)
    parser.add_argument("--min-dist-factor", type=float, default=INTERNAL_DYNAMIC_MIN_DIST_FACTOR)
    parser.add_argument("--force-min-dist-factor", type=float, default=INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR)
    parser.add_argument("--neighbor-k", type=int, default=INTERNAL_DYNAMIC_NEIGHBOR_K)
    parser.add_argument("--search-factor", type=float, default=INTERNAL_DYNAMIC_SEARCH_FACTOR)
    parser.add_argument("--max-edges-per-new", type=int, default=INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW)
    parser.add_argument("--max-edge-length-factor", type=float, default=INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR)
    parser.add_argument("--cross-layer-window", type=int, default=INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW)
    parser.add_argument("--cross-layer-max-dxy-frac", type=float, default=INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC)
    parser.add_argument("--cross-layer-min-angle", type=float, default=INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG,
                        help="Minimum inclination above horizontal for rods outside the short-link exception.")
    parser.add_argument("--internal-bridge-max-length", type=float,
                        default=INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM,
                        help="Configured maximum length for all-angle internal short links; process validation is required.")
    parser.add_argument("--disable-internal-short-bridges", action="store_true",
                        help="Disallow below-threshold INTERNAL short-link exceptions at every height.")
    parser.add_argument("--allow-edgeless-for-min", dest="allow_edgeless_for_min", action="store_true",
                        default=INTERNAL_DYNAMIC_ALLOW_EDGELESS_FOR_MIN)
    parser.add_argument("--no-allow-edgeless-for-min", dest="allow_edgeless_for_min", action="store_false")
    parser.add_argument("--force-same-edges", type=int, default=INTERNAL_DYNAMIC_FORCE_SAME_LAYER_EDGES)
    parser.add_argument("--force-lower-edges", type=int, default=INTERNAL_DYNAMIC_FORCE_LOWER_EDGES)
    parser.add_argument("--force-upper-edges", type=int, default=INTERNAL_DYNAMIC_FORCE_UPPER_EDGES)
    parser.add_argument("--normal-same-edges", type=int, default=INTERNAL_DYNAMIC_NORMAL_SAME_EDGES)
    parser.add_argument("--normal-lower-edges", type=int, default=INTERNAL_DYNAMIC_NORMAL_LOWER_EDGES)
    parser.add_argument("--normal-upper-edges", type=int, default=INTERNAL_DYNAMIC_NORMAL_UPPER_EDGES)
    parser.add_argument("--layer-z-jitter-frac", type=float, default=INTERNAL_DYNAMIC_LAYER_Z_JITTER_FRAC)
    parser.add_argument("--layer-intra-min-dz", type=float, default=INTERNAL_DYNAMIC_LAYER_INTRA_MIN_DZ_MM)
    parser.add_argument("--internal-pitch-bottom", type=float, default=INTERNAL_PITCH_BOTTOM_MM)
    parser.add_argument("--internal-pitch-top", type=float, default=INTERNAL_PITCH_TOP_MM)
    parser.add_argument("--internal-diam-bottom", type=float, default=INTERNAL_DIAM_BOTTOM_MM)
    parser.add_argument("--internal-diam-top", type=float, default=INTERNAL_DIAM_TOP_MM)

    parser.add_argument("--gapfill", dest="gapfill", action="store_true", default=INTERNAL_DYNAMIC_PHYS_GAPFILL_ENABLE)
    parser.add_argument("--no-gapfill", dest="gapfill", action="store_false")
    parser.add_argument("--gapfill-max-total", type=int, default=INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_TOTAL_ADDS,
                        help="Deprecated compatibility option; safety gap filling has no global node ceiling.")
    parser.add_argument("--gapfill-max-per-interval", type=int,
                        default=INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_ADDS_PER_INTERVAL)
    parser.add_argument("--gapfill-min-z-frac", type=float, default=INTERNAL_DYNAMIC_PHYS_GAPFILL_MIN_Z_FRAC)
    parser.add_argument("--gapfill-base-bias", type=float, default=INTERNAL_DYNAMIC_PHYS_GAPFILL_BASE_BIAS)
    parser.add_argument("--gapfill-reldef-cap", type=float, default=INTERNAL_DYNAMIC_PHYS_GAPFILL_RELDEF_CAP)
    parser.add_argument("--smartfill-per-round", type=int, default=INTERNAL_DYNAMIC_SMARTFILL_PER_ROUND)
    parser.add_argument("--smartfill-max-total", type=int, default=INTERNAL_DYNAMIC_SMARTFILL_MAX_TOTAL_ADDS,
                        help="Deprecated compatibility option; safety SmartFill has no global node ceiling.")
    parser.add_argument("--smartfill-max-per-round", type=int, default=INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND)
    parser.add_argument("--smartfill-max-per-slice", type=int, default=INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_SLICE)
    parser.add_argument("--smartfill-top-fail-slices", type=int, default=INTERNAL_DYNAMIC_SMARTFILL_TOP_FAIL_SLICES)
    parser.add_argument(
        "--smartfill-half-gain-margin-pa", type=float,
        default=INTERNAL_DYNAMIC_SMARTFILL_HALF_GAIN_MARGIN_PA,
        help="Halve insertion-improvement floors once minimum transport margin reaches this value (Pa).")
    parser.add_argument(
        "--smartfill-near-pass-gain-factor", type=float,
        default=INTERNAL_DYNAMIC_SMARTFILL_NEAR_PASS_GAIN_FACTOR,
        help="Insertion-improvement floor multiplier after the near-pass margin trigger.")
    parser.add_argument("--disable-smartfill-incremental-pressure", action="store_true",
                        help="Run every SmartFill pressure trial with the original full-slice calculation for A/B verification.")
    parser.add_argument("--min-liquid-frac", type=float, default=INTERNAL_DYNAMIC_MIN_LIQUID_FRAC)
    parser.add_argument("--min-liquid-area", type=float, default=INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2)
    parser.add_argument("--overfill-porosity-floor", type=float, default=INTERNAL_DYNAMIC_OVERFILL_POROSITY_FLOOR)
    parser.add_argument("--top-porosity-floor", type=float, default=INTERNAL_DYNAMIC_TOP_POROSITY_FLOOR)
    parser.add_argument("--top-guard-start-frac", type=float, default=INTERNAL_DYNAMIC_TOP_GUARD_START_FRAC)
    parser.add_argument("--top-max-adds-per-slice", type=int, default=INTERNAL_DYNAMIC_TOP_MAX_ADDS_PER_SLICE)
    parser.add_argument("--max-cap-jump-ratio", type=float, default=INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO)
    parser.add_argument(
        "--capillary-min-ratio", type=float,
        default=INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO,
        help="Fill until P_cap reaches this multiple of P_required; must be >= 1.",
    )
    parser.add_argument(
        "--capillary-target-ratio", type=float,
        default=INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO,
        help="Trim excess solid until P_cap is near this multiple of P_required; must be >= 1.",
    )
    parser.add_argument(
        "--capillary-band-start-mm", type=float,
        default=INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM,
        help="Apply the preferred lower/upper pressure band only at and above this height.",
    )
    parser.add_argument(
        "--capillary-assembly-reserve-ratio", type=float,
        default=INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO,
        help="Add this ratio to the pre-stitch lower pressure bound; final assembled acceptance uses --capillary-min-ratio without this reserve.",
    )
    parser.add_argument(
        "--low-porosity-delete-threshold", type=float,
        default=INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_THRESHOLD,
        help="Delete internal nodes only where porosity is below this fraction and P_cap exceeds P_required.",
    )
    parser.add_argument(
        "--low-porosity-delete-rounds", type=int,
        default=INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_MAX_ROUNDS,
        help="Maximum low-porosity deletion evaluations; non-positive values disable this optional stage.",
    )
    parser.add_argument("--smooth-prune-max", type=int, default=INTERNAL_DYNAMIC_SMOOTH_PRUNE_MAX_REMOVALS)
    parser.add_argument("--smooth-prune-batch-size", type=int, default=INTERNAL_DYNAMIC_SMOOTH_PRUNE_BATCH_SIZE)
    parser.add_argument("--final-trim-max-evaluations", type=int, default=INTERNAL_DYNAMIC_FINAL_TRIM_MAX_EVALUATIONS)
    parser.add_argument("--prune-max-seconds", type=float, default=INTERNAL_DYNAMIC_PRUNE_MAX_SECONDS)
    parser.add_argument("--prune-max-stall", type=int, default=INTERNAL_DYNAMIC_PRUNE_MAX_STALL_EVALUATIONS)
    parser.add_argument(
        "--prune-min-relative-improvement", type=float,
        default=INTERNAL_DYNAMIC_PRUNE_MIN_RELATIVE_IMPROVEMENT,
    )
    parser.add_argument("--xz-bins", type=int, default=INTERNAL_DYNAMIC_XZ_BINS)
    return parser.parse_args()


def apply_cli_config(args):
    global IN_STL, OUT_DIR, TARGET_HEIGHT_MM, INTERNAL_BUILD_MODE, SURFACE_DESIGN_ENABLE
    global INITIAL_SKELETON_CHECKPOINT_LOAD, INITIAL_SKELETON_CHECKPOINT_SAVE
    global PRE_SMARTFILL_CHECKPOINT_LOAD, PRE_SMARTFILL_CHECKPOINT_SAVE
    global CHECKPOINT_STOP_AFTER, CHECKPOINT_IGNORE_CONFIG
    global GAMMA_NPM, CONTACT_ANGLE_DEG, RHO_LIQ, RHO_GAS, G_GRAV, SAFETY_FACTOR, SAFETY_FACTOR_BOTTOM, SAFETY_FACTOR_TOP, FLUID_VISCOSITY_PA_S, FLOW_RESISTANCE_CALIBRATION_FACTOR, VISC_REF_Q_M3_S, TARGET_FILL_TIME_S, SLICE_DZ_MM, PRESSURE_SLICE_COUNT
    global SURFACE_DIAM_MM, INITIAL_FACE_COUNT, SURFACE_MAX_FACES_PER_PATCH, SURFACE_SEAM_MAX_FRACTION
    global SURFACE_FACE_SCAN_START, SURFACE_FACE_SCAN_STOP, SURFACE_FACE_SCAN_STEP
    global MAX_SPLIT_ROUNDS, R_EFF_LIMIT_MM, SURFACE_RADIUS_SOURCE
    global SURFACE_EXTRA_PRESSURE_PA, SURFACE_CAPILLARY_SAFETY_FACTOR, SURFACE_OUTWARD_OFFSET_MM, STITCH_DISTANCE_MM, STITCH_ANCHOR_SPACING_MM, STITCH_RELAXED_DISTANCE_MM
    global BOTTOM_PILLAR_ENABLE, BOTTOM_PILLAR_ROWS, BOTTOM_PILLAR_MAX_ANCHORS, BOTTOM_PILLAR_ANCHOR_SPACING_MM
    global BOTTOM_PILLAR_HEIGHT_LAYERS, BOTTOM_PILLAR_RADIUS_FACTOR, BOTTOM_PILLAR_MIN_COMPONENT_AREA_MM2
    global INTERNAL_DYNAMIC_LAYERS, INTERNAL_DYNAMIC_SEED, INTERNAL_DYNAMIC_MIN_NODES_PER_LAYER, INTERNAL_DYNAMIC_MAX_NODES_PER_LAYER
    global INTERNAL_MATCHED_RANDOM_POSITION, INTERNAL_MATCHED_RANDOM_NODE_BUDGET
    global INTERNAL_MATCHED_RANDOM_EDGE_BUDGET
    global INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_ENABLE, INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_REJECTIONS
    global INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_FRACTION, INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MAX_TARGETS_PER_LAYER
    global INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_DEDUP_FACTOR
    global INTERNAL_DYNAMIC_BOTTOM_NODE_TARGET, INTERNAL_DYNAMIC_TOP_NODE_TARGET, INTERNAL_DYNAMIC_DENSITY_FACTOR, INTERNAL_DYNAMIC_INITIAL_DENSITY_MULTIPLIER, INTERNAL_DYNAMIC_INITIAL_DENSITY_SOURCE, INTERNAL_DYNAMIC_AREA_TARGET_EXPONENT, INTERNAL_DYNAMIC_AREA_TARGET_MIN_SCALE, INTERNAL_DYNAMIC_MIN_COMPONENT_AREA_MM2
    global INTERNAL_DYNAMIC_STEP1_FRACTION, INTERNAL_DYNAMIC_MAX_TRIES_PER_LAYER, INTERNAL_DYNAMIC_CANDIDATES
    global INTERNAL_DYNAMIC_MIN_DIST_FACTOR, INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR, INTERNAL_DYNAMIC_NEIGHBOR_K, INTERNAL_DYNAMIC_SEARCH_FACTOR, INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
    global INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR, INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW, INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC, INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG
    global INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM, INTERNAL_HORIZONTAL_BRIDGE_ENABLE
    global INTERNAL_DYNAMIC_ALLOW_EDGELESS_FOR_MIN
    global INTERNAL_DYNAMIC_BOTTOM_NODE_TARGET, INTERNAL_DYNAMIC_TOP_NODE_TARGET, INTERNAL_DYNAMIC_DENSITY_FACTOR, INTERNAL_DYNAMIC_STEP1_FRACTION
    global INTERNAL_DYNAMIC_MAX_TRIES_PER_LAYER, INTERNAL_DYNAMIC_CANDIDATES, INTERNAL_DYNAMIC_MIN_DIST_FACTOR, INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR
    global INTERNAL_DYNAMIC_NEIGHBOR_K, INTERNAL_DYNAMIC_SEARCH_FACTOR, INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
    global INTERNAL_DYNAMIC_FORCE_SAME_LAYER_EDGES, INTERNAL_DYNAMIC_FORCE_LOWER_EDGES, INTERNAL_DYNAMIC_FORCE_UPPER_EDGES
    global INTERNAL_DYNAMIC_NORMAL_SAME_EDGES, INTERNAL_DYNAMIC_NORMAL_LOWER_EDGES, INTERNAL_DYNAMIC_NORMAL_UPPER_EDGES
    global INTERNAL_DYNAMIC_LAYER_Z_JITTER_FRAC, INTERNAL_DYNAMIC_LAYER_INTRA_MIN_DZ_MM, INTERNAL_DYNAMIC_PHYSICS_DZ_MM
    global INTERNAL_DYNAMIC_GLOBAL_FEEDBACK_SPACING_MM, INTERNAL_DYNAMIC_UPPER_FEEDBACK_SPACING_MM
    global INTERNAL_DYNAMIC_PHYS_GAPFILL_ENABLE, INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_TOTAL_ADDS, INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_ADDS_PER_INTERVAL
    global INTERNAL_DYNAMIC_PHYS_GAPFILL_MIN_Z_FRAC, INTERNAL_DYNAMIC_PHYS_GAPFILL_BASE_BIAS, INTERNAL_DYNAMIC_PHYS_GAPFILL_RELDEF_CAP
    global INTERNAL_DYNAMIC_SMARTFILL_PER_ROUND, INTERNAL_DYNAMIC_SMARTFILL_MAX_TOTAL_ADDS
    global INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND, INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_SLICE, INTERNAL_DYNAMIC_SMARTFILL_TOP_FAIL_SLICES
    global INTERNAL_DYNAMIC_SMARTFILL_HALF_GAIN_MARGIN_PA, INTERNAL_DYNAMIC_SMARTFILL_NEAR_PASS_GAIN_FACTOR
    global INTERNAL_DYNAMIC_SMARTFILL_INCREMENTAL_PRESSURE_ENABLE
    global INTERNAL_DYNAMIC_MIN_LIQUID_FRAC, INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2
    global INTERNAL_DYNAMIC_OVERFILL_POROSITY_FLOOR, INTERNAL_DYNAMIC_TOP_POROSITY_FLOOR
    global INTERNAL_DYNAMIC_TOP_GUARD_START_FRAC, INTERNAL_DYNAMIC_TOP_MAX_ADDS_PER_SLICE
    global INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO, INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO, INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO, INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM, INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO, INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_THRESHOLD
    global INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_MAX_ROUNDS, INTERNAL_DYNAMIC_SMOOTH_PRUNE_MAX_REMOVALS, INTERNAL_DYNAMIC_SMOOTH_PRUNE_BATCH_SIZE
    global INTERNAL_DYNAMIC_FINAL_TRIM_MAX_EVALUATIONS, INTERNAL_DYNAMIC_PRUNE_MAX_SECONDS
    global INTERNAL_DYNAMIC_PRUNE_MAX_STALL_EVALUATIONS, INTERNAL_DYNAMIC_PRUNE_MIN_RELATIVE_IMPROVEMENT
    global INTERNAL_DYNAMIC_XZ_BINS
    global INTERNAL_PITCH_BOTTOM_MM, INTERNAL_PITCH_TOP_MM, INTERNAL_DIAM_BOTTOM_MM, INTERNAL_DIAM_TOP_MM

    IN_STL = args.stl
    INITIAL_SKELETON_CHECKPOINT_LOAD = args.load_initial_skeleton
    INITIAL_SKELETON_CHECKPOINT_SAVE = args.save_initial_skeleton
    PRE_SMARTFILL_CHECKPOINT_LOAD = args.load_pre_smartfill
    PRE_SMARTFILL_CHECKPOINT_SAVE = args.save_pre_smartfill
    CHECKPOINT_STOP_AFTER = args.stop_after_checkpoint
    CHECKPOINT_IGNORE_CONFIG = bool(args.checkpoint_ignore_config)
    if PRE_SMARTFILL_CHECKPOINT_LOAD is not None and INITIAL_SKELETON_CHECKPOINT_SAVE is not None:
        raise ValueError("cannot save an initial-skeleton checkpoint after loading pre-SmartFill state")
    if CHECKPOINT_STOP_AFTER == "initial-skeleton" and INITIAL_SKELETON_CHECKPOINT_SAVE is None:
        raise ValueError("--stop-after-checkpoint initial-skeleton requires --save-initial-skeleton DIR")
    if CHECKPOINT_STOP_AFTER == "pre-smartfill" and PRE_SMARTFILL_CHECKPOINT_SAVE is None:
        raise ValueError("--stop-after-checkpoint pre-smartfill requires --save-pre-smartfill DIR")
    TARGET_HEIGHT_MM = args.height_mm
    OUT_DIR = args.out_dir or (IN_STL.with_suffix("").parent / f"{IN_STL.stem}_dynamic_out")
    INTERNAL_BUILD_MODE = args.build_mode
    if args.internal_only and args.surface_only:
        raise ValueError("--internal-only and --surface-only are mutually exclusive")
    if args.internal_only and args.surface_face_scan:
        raise ValueError("--surface-face-scan cannot be combined with --internal-only")
    if args.surface_only and args.surface_design is False:
        raise ValueError("--surface-only cannot be combined with --no-surface-design")
    if args.surface_design is not None:
        SURFACE_DESIGN_ENABLE = bool(args.surface_design)
    if args.surface_only:
        SURFACE_DESIGN_ENABLE = True

    GAMMA_NPM = args.gamma
    CONTACT_ANGLE_DEG = args.contact_angle
    RHO_LIQ = args.rho
    RHO_GAS = args.surface_gas_density
    G_GRAV = args.gravity
    SAFETY_FACTOR = args.margin_correction_factor if args.margin_correction_factor is not None else args.safety_factor
    SAFETY_FACTOR_BOTTOM = args.safety_factor_bottom
    SAFETY_FACTOR_TOP = args.safety_factor_top
    FLUID_VISCOSITY_PA_S = args.viscosity
    if float(args.reference_flow_rate) <= 0.0:
        raise ValueError("--reference-flow-rate must be > 0")
    VISC_REF_Q_M3_S = float(args.reference_flow_rate)
    if not math.isclose(float(args.flow_resistance_calibration), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("--flow-resistance-calibration is deprecated; original viscous model requires 1.0")
    FLOW_RESISTANCE_CALIBRATION_FACTOR = 1.0
    if float(args.slice_dz) <= 0.0:
        raise ValueError("--slice-dz must be > 0")
    SLICE_DZ_MM = float(args.slice_dz)

    PRESSURE_SLICE_COUNT = max(1, int(math.ceil(float(TARGET_HEIGHT_MM) / SLICE_DZ_MM)))
    INTERNAL_DYNAMIC_PHYSICS_DZ_MM = SLICE_DZ_MM
    INTERNAL_DYNAMIC_GLOBAL_FEEDBACK_SPACING_MM = SLICE_DZ_MM
    INTERNAL_DYNAMIC_UPPER_FEEDBACK_SPACING_MM = SLICE_DZ_MM
    SURFACE_DIAM_MM = args.surface_diam
    INITIAL_FACE_COUNT = int(args.surface_faces)
    if INITIAL_FACE_COUNT < 16:
        raise ValueError("--surface-faces must be >= 16")
    SURFACE_MAX_FACES_PER_PATCH = int(args.surface_max_faces_per_patch)
    if SURFACE_MAX_FACES_PER_PATCH < 4:
        raise ValueError("--surface-max-faces-per-patch must be >= 4")
    SURFACE_SEAM_MAX_FRACTION = float(args.surface_seam_max_fraction)
    if not 0.0 <= SURFACE_SEAM_MAX_FRACTION <= 1.0:
        raise ValueError("--surface-seam-max-fraction must be in [0, 1]")
    SURFACE_FACE_SCAN_START = int(args.surface_face_scan_start)
    SURFACE_FACE_SCAN_STOP = int(args.surface_face_scan_stop)
    SURFACE_FACE_SCAN_STEP = int(args.surface_face_scan_step)
    if SURFACE_FACE_SCAN_START < 16 or SURFACE_FACE_SCAN_STOP < 16:
        raise ValueError("surface face-scan bounds must be >= 16")
    if SURFACE_FACE_SCAN_STEP <= 0:
        raise ValueError("--surface-face-scan-step must be > 0")
    if SURFACE_FACE_SCAN_START < SURFACE_FACE_SCAN_STOP:
        raise ValueError("--surface-face-scan-start must be >= --surface-face-scan-stop")
    MAX_SPLIT_ROUNDS = args.surface_rounds
    SURFACE_EXTRA_PRESSURE_PA = args.surface_extra_pressure_pa
    SURFACE_CAPILLARY_SAFETY_FACTOR = args.surface_safety_factor
    capillary_design = _surface_capillary_design()
    if args.surface_reff_limit is None:
        R_EFF_LIMIT_MM = capillary_design.critical_equivalent_radius_mm
        SURFACE_RADIUS_SOURCE = "capillary_pressure"
    else:
        if args.surface_reff_limit <= 0.0:
            raise ValueError("--surface-reff-limit must be positive")
        R_EFF_LIMIT_MM = float(args.surface_reff_limit)
        SURFACE_RADIUS_SOURCE = "manual_override"
    SURFACE_OUTWARD_OFFSET_MM = args.surface_offset
    STITCH_DISTANCE_MM = args.stitch_distance
    STITCH_ANCHOR_SPACING_MM = args.stitch_anchor_spacing
    STITCH_RELAXED_DISTANCE_MM = max(float(args.stitch_relaxed_distance), STITCH_DISTANCE_MM)
    BOTTOM_PILLAR_ENABLE = args.bottom_pillars
    BOTTOM_PILLAR_ROWS = args.bottom_pillar_rows
    BOTTOM_PILLAR_MAX_ANCHORS = args.bottom_pillar_max_anchors
    BOTTOM_PILLAR_ANCHOR_SPACING_MM = args.bottom_pillar_spacing
    BOTTOM_PILLAR_HEIGHT_LAYERS = args.bottom_pillar_height_layers
    BOTTOM_PILLAR_RADIUS_FACTOR = args.bottom_pillar_radius_factor
    BOTTOM_PILLAR_MIN_COMPONENT_AREA_MM2 = args.bottom_pillar_min_component_area

    INTERNAL_DYNAMIC_LAYERS = args.layers
    INTERNAL_DYNAMIC_SEED = args.seed
    INTERNAL_MATCHED_RANDOM_POSITION = bool(args.internal_matched_random_position)
    INTERNAL_MATCHED_RANDOM_NODE_BUDGET = int(args.internal_matched_node_budget)
    INTERNAL_MATCHED_RANDOM_EDGE_BUDGET = int(args.internal_matched_edge_budget)
    if INTERNAL_MATCHED_RANDOM_NODE_BUDGET < 0 or INTERNAL_MATCHED_RANDOM_EDGE_BUDGET < 0:
        raise ValueError("Matched random-position budgets must be nonnegative")
    if (not INTERNAL_MATCHED_RANDOM_POSITION and
            (INTERNAL_MATCHED_RANDOM_NODE_BUDGET or INTERNAL_MATCHED_RANDOM_EDGE_BUDGET)):
        raise ValueError("Matched budgets require --internal-matched-random-position")
    INTERNAL_DYNAMIC_MIN_NODES_PER_LAYER = args.min_nodes
    INTERNAL_DYNAMIC_MAX_NODES_PER_LAYER = args.max_nodes
    INTERNAL_DYNAMIC_BOTTOM_NODE_TARGET = args.bottom_nodes
    INTERNAL_DYNAMIC_TOP_NODE_TARGET = args.top_nodes
    INTERNAL_DYNAMIC_DENSITY_FACTOR = args.density_factor
    if args.initial_density_multiplier is None:
        INTERNAL_DYNAMIC_INITIAL_DENSITY_SOURCE = "configured_design_domain"
    else:
        if args.initial_density_multiplier <= 0.0:
            raise ValueError("--initial-density-multiplier must be positive")
        INTERNAL_DYNAMIC_INITIAL_DENSITY_MULTIPLIER = float(args.initial_density_multiplier)
        INTERNAL_DYNAMIC_INITIAL_DENSITY_SOURCE = "manual_cli"
    INTERNAL_DYNAMIC_AREA_TARGET_EXPONENT = args.area_target_exponent
    INTERNAL_DYNAMIC_AREA_TARGET_MIN_SCALE = args.area_target_min_scale
    INTERNAL_DYNAMIC_MIN_COMPONENT_AREA_MM2 = args.min_component_area
    INTERNAL_DYNAMIC_STEP1_FRACTION = args.step1_fraction
    INTERNAL_DYNAMIC_MAX_TRIES_PER_LAYER = args.max_tries
    INTERNAL_DYNAMIC_CANDIDATES = args.candidates
    INTERNAL_DYNAMIC_MIN_DIST_FACTOR = args.min_dist_factor
    INTERNAL_DYNAMIC_FORCE_MIN_DIST_FACTOR = args.force_min_dist_factor
    INTERNAL_DYNAMIC_NEIGHBOR_K = args.neighbor_k
    INTERNAL_DYNAMIC_SEARCH_FACTOR = args.search_factor
    INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW = args.max_edges_per_new
    INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR = args.max_edge_length_factor
    INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW = args.cross_layer_window
    INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC = args.cross_layer_max_dxy_frac
    INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG = args.cross_layer_min_angle
    if not (0.0 <= INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG <= 90.0):
        raise ValueError('--cross-layer-min-angle must be in [0,90]')
    INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM = float(args.internal_bridge_max_length)
    if not (np.isfinite(INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM) and
            INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM > 0.0):
        raise ValueError('--internal-bridge-max-length must be a positive finite number')
    INTERNAL_HORIZONTAL_BRIDGE_ENABLE = not bool(args.disable_internal_short_bridges)
    INTERNAL_DYNAMIC_ALLOW_EDGELESS_FOR_MIN = args.allow_edgeless_for_min
    INTERNAL_DYNAMIC_FORCE_SAME_LAYER_EDGES = args.force_same_edges
    INTERNAL_DYNAMIC_FORCE_LOWER_EDGES = args.force_lower_edges
    INTERNAL_DYNAMIC_FORCE_UPPER_EDGES = args.force_upper_edges
    INTERNAL_DYNAMIC_NORMAL_SAME_EDGES = args.normal_same_edges
    INTERNAL_DYNAMIC_NORMAL_LOWER_EDGES = args.normal_lower_edges
    INTERNAL_DYNAMIC_NORMAL_UPPER_EDGES = args.normal_upper_edges
    INTERNAL_DYNAMIC_LAYER_Z_JITTER_FRAC = args.layer_z_jitter_frac
    INTERNAL_DYNAMIC_LAYER_INTRA_MIN_DZ_MM = args.layer_intra_min_dz
    INTERNAL_PITCH_BOTTOM_MM = args.internal_pitch_bottom
    INTERNAL_PITCH_TOP_MM = args.internal_pitch_top
    INTERNAL_DIAM_BOTTOM_MM = args.internal_diam_bottom
    INTERNAL_DIAM_TOP_MM = args.internal_diam_top

    INTERNAL_DYNAMIC_PHYS_GAPFILL_ENABLE = args.gapfill
    INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_TOTAL_ADDS = args.gapfill_max_total
    INTERNAL_DYNAMIC_PHYS_GAPFILL_MAX_ADDS_PER_INTERVAL = args.gapfill_max_per_interval
    INTERNAL_DYNAMIC_PHYS_GAPFILL_MIN_Z_FRAC = args.gapfill_min_z_frac
    INTERNAL_DYNAMIC_PHYS_GAPFILL_BASE_BIAS = args.gapfill_base_bias
    INTERNAL_DYNAMIC_PHYS_GAPFILL_RELDEF_CAP = args.gapfill_reldef_cap
    INTERNAL_DYNAMIC_SMARTFILL_PER_ROUND = args.smartfill_per_round
    INTERNAL_DYNAMIC_SMARTFILL_MAX_TOTAL_ADDS = args.smartfill_max_total
    INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND = args.smartfill_max_per_round
    INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_SLICE = args.smartfill_max_per_slice
    INTERNAL_DYNAMIC_SMARTFILL_TOP_FAIL_SLICES = args.smartfill_top_fail_slices
    INTERNAL_DYNAMIC_SMARTFILL_HALF_GAIN_MARGIN_PA = float(args.smartfill_half_gain_margin_pa)
    if not 0.0 < float(args.smartfill_near_pass_gain_factor) <= 1.0:
        raise ValueError("--smartfill-near-pass-gain-factor must be in (0, 1]")
    INTERNAL_DYNAMIC_SMARTFILL_NEAR_PASS_GAIN_FACTOR = float(args.smartfill_near_pass_gain_factor)
    INTERNAL_DYNAMIC_SMARTFILL_INCREMENTAL_PRESSURE_ENABLE = not args.disable_smartfill_incremental_pressure
    INTERNAL_DYNAMIC_MIN_LIQUID_FRAC = args.min_liquid_frac
    INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2 = args.min_liquid_area
    INTERNAL_DYNAMIC_OVERFILL_POROSITY_FLOOR = args.overfill_porosity_floor
    INTERNAL_DYNAMIC_TOP_POROSITY_FLOOR = args.top_porosity_floor
    INTERNAL_DYNAMIC_TOP_GUARD_START_FRAC = args.top_guard_start_frac
    INTERNAL_DYNAMIC_TOP_MAX_ADDS_PER_SLICE = args.top_max_adds_per_slice
    INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO = args.max_cap_jump_ratio
    if float(args.capillary_min_ratio) < 1.0:
        raise ValueError("--capillary-min-ratio must be >= 1")
    if float(args.capillary_target_ratio) < 1.0:
        raise ValueError("--capillary-target-ratio must be >= 1")
    if float(args.capillary_min_ratio) > float(args.capillary_target_ratio):
        raise ValueError("--capillary-min-ratio must not exceed --capillary-target-ratio")
    INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO = float(args.capillary_min_ratio)
    INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO = float(args.capillary_target_ratio)
    if float(args.capillary_band_start_mm) < 0.0:
        raise ValueError("--capillary-band-start-mm must be >= 0")
    INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM = float(args.capillary_band_start_mm)
    if float(args.capillary_assembly_reserve_ratio) < 0.0:
        raise ValueError("--capillary-assembly-reserve-ratio must be >= 0")
    if float(args.capillary_min_ratio) + float(args.capillary_assembly_reserve_ratio) > float(
            args.capillary_target_ratio):
        raise ValueError(
            "pre-stitch lower bound (--capillary-min-ratio + assembly reserve) must not exceed --capillary-target-ratio")
    INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO = float(args.capillary_assembly_reserve_ratio)
    if not 0.0 <= float(args.low_porosity_delete_threshold) < 1.0:
        raise ValueError("--low-porosity-delete-threshold must be in [0, 1)")
    INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_THRESHOLD = float(args.low_porosity_delete_threshold)
    INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_MAX_ROUNDS = max(0, int(args.low_porosity_delete_rounds))
    INTERNAL_DYNAMIC_SMOOTH_PRUNE_MAX_REMOVALS = max(0, int(args.smooth_prune_max))
    INTERNAL_DYNAMIC_SMOOTH_PRUNE_BATCH_SIZE = max(1, int(args.smooth_prune_batch_size))
    if int(args.final_trim_max_evaluations) < 1:
        raise ValueError("--final-trim-max-evaluations must be >= 1")
    if float(args.prune_max_seconds) <= 0.0:
        raise ValueError("--prune-max-seconds must be > 0")
    if int(args.prune_max_stall) < 1:
        raise ValueError("--prune-max-stall must be >= 1")
    if not 0.0 <= float(args.prune_min_relative_improvement) < 1.0:
        raise ValueError("--prune-min-relative-improvement must be in [0, 1)")
    INTERNAL_DYNAMIC_FINAL_TRIM_MAX_EVALUATIONS = int(args.final_trim_max_evaluations)
    INTERNAL_DYNAMIC_PRUNE_MAX_SECONDS = float(args.prune_max_seconds)
    INTERNAL_DYNAMIC_PRUNE_MAX_STALL_EVALUATIONS = int(args.prune_max_stall)
    INTERNAL_DYNAMIC_PRUNE_MIN_RELATIVE_IMPROVEMENT = float(args.prune_min_relative_improvement)
    INTERNAL_DYNAMIC_XZ_BINS = args.xz_bins


def export_run_config(output_dir: Path):
    cfg = {
        "input_stl": str(IN_STL),
        "target_height_mm": TARGET_HEIGHT_MM,
        "output_dir": str(output_dir),
        "build_mode": INTERNAL_BUILD_MODE,
        "internal_matched_random_position": bool(INTERNAL_MATCHED_RANDOM_POSITION),
        "internal_matched_random_node_budget": int(INTERNAL_MATCHED_RANDOM_NODE_BUDGET),
        "internal_matched_random_edge_budget": int(INTERNAL_MATCHED_RANDOM_EDGE_BUDGET),
        "surface_design_enabled": bool(SURFACE_DESIGN_ENABLE),
        "workflow_checkpoints": {
            "load_initial_skeleton": (str(INITIAL_SKELETON_CHECKPOINT_LOAD)
                                      if INITIAL_SKELETON_CHECKPOINT_LOAD else None),
            "save_initial_skeleton": (str(INITIAL_SKELETON_CHECKPOINT_SAVE)
                                      if INITIAL_SKELETON_CHECKPOINT_SAVE else None),
            "load_pre_smartfill": (str(PRE_SMARTFILL_CHECKPOINT_LOAD)
                                   if PRE_SMARTFILL_CHECKPOINT_LOAD else None),
            "save_pre_smartfill": (str(PRE_SMARTFILL_CHECKPOINT_SAVE)
                                   if PRE_SMARTFILL_CHECKPOINT_SAVE else None),
            "stop_after": CHECKPOINT_STOP_AFTER,
            "ignore_config_mismatch": bool(CHECKPOINT_IGNORE_CONFIG),
        },
        "dual_stitching": {"enabled": bool(SURFACE_DESIGN_ENABLE),
                           "standard_length_mm": STITCH_DISTANCE_MM,
                           "relaxed_search_length_mm": STITCH_RELAXED_DISTANCE_MM,
                           "anchor_spacing_mm": STITCH_ANCHOR_SPACING_MM,
                           "angle": "configured_elevation_or_short_link_exception",
                           "surface_normal_angle_used": False},
        "fluid": {
            "surface_tension_N_per_m": GAMMA_NPM,
            "contact_angle_deg": CONTACT_ANGLE_DEG,
            "density_kg_per_m3": RHO_LIQ,
            "gas_density_kg_per_m3": RHO_GAS,
            "gravity_m_per_s2": G_GRAV,
            "safety_factor": SAFETY_FACTOR,
            "margin_correction_factor": SAFETY_FACTOR,
            "safety_factor_bottom": SAFETY_FACTOR_BOTTOM,
            "safety_factor_top": SAFETY_FACTOR_TOP,
            "viscosity_Pa_s": FLUID_VISCOSITY_PA_S,
            "reference_flow_rate_m3_s": VISC_REF_Q_M3_S,
        },
        "quasi_steady_filling": {
            "representation": "single-phase slice-wise fixed-reference-flow capillary pressure budget",
            "acceptance_basis": "fixed-reference-flow pressure budget: P_cap must exceed hydrostatic plus viscous loss, including the configured reserve ratio",
            "reference_flow_rate_m3_s": VISC_REF_Q_M3_S,
            "mobility_metric_role": "diagnostic_and_candidate_priority_only",
            "resistance_calibration_factor": FLOW_RESISTANCE_CALIBRATION_FACTOR,
            "minimum_component_area_mm2": FLOW_SECTION_MIN_COMPONENT_AREA_MM2,
            "pressure_slice_nominal_thickness_mm": SLICE_DZ_MM,
            "pressure_slice_count": PRESSURE_SLICE_COUNT,
            "limitations": (
                "reduced-order fixed-flow pressure-budget model; excludes simultaneous immiscible displacement, "
                "trapped gas and contact-angle hysteresis"
            ),
        },
        "surface": {
            "builder": "curvature_aware_patch_barycentric_v1",
            "diameter_mm": SURFACE_DIAM_MM,
            "initial_face_count": INITIAL_FACE_COUNT,
            "target_face_count": INITIAL_FACE_COUNT,
            "target_face_count_source": "configured_per_design_domain",
            "max_split_rounds": _effective_surface_split_rounds(),
            "requested_split_rounds": int(MAX_SPLIT_ROUNDS),
            "effective_split_rounds": _effective_surface_split_rounds(),
            "r_eff_limit_mm": R_EFF_LIMIT_MM,
            "equivalent_capillary_radius_limit_mm": R_EFF_LIMIT_MM,
            "radius_limit_source": SURFACE_RADIUS_SOURCE,
            "additional_pressure_Pa": SURFACE_EXTRA_PRESSURE_PA,
            "capillary_safety_factor": SURFACE_CAPILLARY_SAFETY_FACTOR,
            "pressure_requirement_Pa": SURFACE_CAPILLARY_SAFETY_FACTOR * (
                    (RHO_LIQ - RHO_GAS) * G_GRAV * TARGET_HEIGHT_MM * 1.0e-3
                    + SURFACE_EXTRA_PRESSURE_PA
            ),
            "outward_offset_mm": SURFACE_OUTWARD_OFFSET_MM,
            "max_faces_per_patch": SURFACE_MAX_FACES_PER_PATCH,
            "seam_screen_max_local_record_fraction": SURFACE_SEAM_MAX_FRACTION,
            "seam_screen_interpretation": (
                "methodological burden screen only; cross-patch pore merging or final 3-D "
                "inspection remains required for globally complete pore claims"
            ),
            "max_edge_segment_mm": max(0.75, min(1.50, 3.0 * float(R_EFF_LIMIT_MM))),
            "surface_fit_tolerance_mm": min(0.025, 0.0625 * float(SURFACE_DIAM_MM)),
            "reject_cross_patch_or_component_edges": True,
            "explicit_surface_node_spheres": False,
            "surface_junction_geometry": "round_capped_rods",
            "stitch_anchor_min_open_area_guard_mm2": 0.04,
            "source_node_spacing_mm": 0.65,
            "min_boundary_carriers_per_patch": 3,
            "boundary_graph_node_policy": (
                "global spacing filter plus at least three well-separated boundary carriers per patch; "
                "the complete patch polygon remains the surface-domain boundary"
            ),
        },
        "internal_dynamic": {
            "layers": INTERNAL_DYNAMIC_LAYERS,
            "seed": INTERNAL_DYNAMIC_SEED,
            "min_nodes_per_layer": INTERNAL_DYNAMIC_MIN_NODES_PER_LAYER,
            "max_nodes_per_layer": INTERNAL_DYNAMIC_MAX_NODES_PER_LAYER,
            "bottom_node_target": INTERNAL_DYNAMIC_BOTTOM_NODE_TARGET,
            "top_node_target": INTERNAL_DYNAMIC_TOP_NODE_TARGET,
            "density_factor": INTERNAL_DYNAMIC_DENSITY_FACTOR,
            "initial_skeleton_density_multiplier": INTERNAL_DYNAMIC_INITIAL_DENSITY_MULTIPLIER,
            "initial_skeleton_density_multiplier_source": INTERNAL_DYNAMIC_INITIAL_DENSITY_SOURCE,
            "initial_skeleton_density_multiplier_effective": float(INTERNAL_DYNAMIC_INITIAL_DENSITY_MULTIPLIER),
            "initial_skeleton_density_rule": (
                "max(height_gradient_target, area_target, minimum_layer_target) "
                "multiplied uniformly before rounding and the optional maximum-layer cap"
            ),
            "area_target_exponent": INTERNAL_DYNAMIC_AREA_TARGET_EXPONENT,
            "area_target_min_scale": INTERNAL_DYNAMIC_AREA_TARGET_MIN_SCALE,
            "min_component_area_mm2": INTERNAL_DYNAMIC_MIN_COMPONENT_AREA_MM2,
            "neighbor_k": INTERNAL_DYNAMIC_NEIGHBOR_K,
            "max_edges_per_new": INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW,
            "max_edge_length_factor": INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR,
            "cross_layer_window": INTERNAL_DYNAMIC_CROSS_LAYER_WINDOW,
            "cross_layer_max_dxy_frac": INTERNAL_DYNAMIC_CROSS_LAYER_MAX_DXY_FRAC,
            "cross_layer_min_angle_deg": INTERNAL_DYNAMIC_CROSS_LAYER_MIN_ANGLE_DEG,
            "internal_bridge_max_length_mm_provisional": INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM,
            "internal_short_edge_any_angle": True,
            "top_smartfill_short_edge_any_angle": True,
            "top_smartfill_short_edge_first_support_allowed": True,
            "top_relaxation_disclosure": (
                "All internal heights use the same configured short-link and elevation rules."
            ),
            "top_any_angle_active_height_mm": INTERNAL_TOP_RELAX_ACTIVE_HEIGHT_MM,
            "top_any_angle_endpoint_height_mm": INTERNAL_TOP_RELAX_ENDPOINT_HEIGHT_MM,
            "top_any_angle_max_3d_length_mm": min(INTERNAL_TOP_RELAX_MAX_EDGE_LENGTH_MM,
                                                  INTERNAL_HORIZONTAL_BRIDGE_MAX_LENGTH_MM),
            "internal_short_bridges_enabled": INTERNAL_HORIZONTAL_BRIDGE_ENABLE,
            "allow_edgeless_for_min": INTERNAL_DYNAMIC_ALLOW_EDGELESS_FOR_MIN,
            "force_same_edges": INTERNAL_DYNAMIC_FORCE_SAME_LAYER_EDGES,
            "force_lower_edges": INTERNAL_DYNAMIC_FORCE_LOWER_EDGES,
            "force_upper_edges": INTERNAL_DYNAMIC_FORCE_UPPER_EDGES,
            "normal_same_edges": INTERNAL_DYNAMIC_NORMAL_SAME_EDGES,
            "normal_lower_edges": INTERNAL_DYNAMIC_NORMAL_LOWER_EDGES,
            "normal_upper_edges": INTERNAL_DYNAMIC_NORMAL_UPPER_EDGES,
            "support_repair_max_edges": INTERNAL_DYNAMIC_SUPPORT_REPAIR_MAX_EDGES,
            "support_repair_degree_reserve": INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE,
            "new_node_inbound_degree_reserve": INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE,
            "new_node_degree_policy": (
                "new-node degree and later inbound reserve are controlled by the recorded numeric fields"
            ),
            "support_repair_absolute_max_degree": (
                    INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                    + max(INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE,
                          INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE)
            ),
            "support_repair_constraints": "same_configured_angle_span_and_domain_rules_as_routine_edges",
            "initial_peer_relax_trigger_min_rejections": INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_REJECTIONS,
            "initial_peer_relax_trigger_min_fraction": INTERNAL_DYNAMIC_INITIAL_SUPPORT_REPAIR_MIN_FRACTION,
            "initial_peer_relax_max_peers_per_new_node": INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_PEERS,
            "initial_peer_relax_max_3d_length_mm": INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_MAX_LENGTH_MM,
            "initial_peer_relax_endpoint_degree_reserve": INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_DEGREE_RESERVE,
            "final_regular_node_degree_limit": (
                INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                + max(INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE,
                      INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE)
            ),
            "final_abnormal_layer_peer_endpoint_degree_limit": (
                INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW
                + max(INTERNAL_DYNAMIC_SUPPORT_REPAIR_DEGREE_RESERVE,
                      INTERNAL_DYNAMIC_NEW_NODE_INBOUND_RESERVE,
                      INTERNAL_DYNAMIC_INITIAL_PEER_RELAX_DEGREE_RESERVE)
            ),
            "final_node_degree_policy": (
                "Regular internal endpoints use the regular limit; retained same-layer "
                "peer endpoints in state-detected abnormal layers use the abnormal endpoint limit."
            ),
            "abnormal_layer_densify_fraction": INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_FRACTION,
            "abnormal_layer_densify_min_nodes": INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_MIN_NODES,
            "abnormal_layer_densify_max_nodes": INTERNAL_DYNAMIC_ABNORMAL_LAYER_DENSIFY_MAX_NODES,
            "initial_peer_relax_angle_rule": "any angle including horizontal, only in state-detected abnormal layers; bounded full length plus STL centreline, same-component and expanded endpoint-degree checks retained",
            "layer_z_jitter_fraction": INTERNAL_DYNAMIC_LAYER_Z_JITTER_FRAC,
            "layer_intra_min_dz_mm": INTERNAL_DYNAMIC_LAYER_INTRA_MIN_DZ_MM,
            "gapfill_enabled": INTERNAL_DYNAMIC_PHYS_GAPFILL_ENABLE,
            "smartfill_max_total": INTERNAL_DYNAMIC_SMARTFILL_MAX_TOTAL_ADDS,
            "smartfill_max_per_round": INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_ROUND,
            "smartfill_max_per_slice": INTERNAL_DYNAMIC_SMARTFILL_MAX_ADDS_PER_SLICE,
            "smartfill_top_fail_slices": INTERNAL_DYNAMIC_SMARTFILL_TOP_FAIL_SLICES,
            "smartfill_half_gain_margin_pa": INTERNAL_DYNAMIC_SMARTFILL_HALF_GAIN_MARGIN_PA,
            "smartfill_near_pass_gain_factor": INTERNAL_DYNAMIC_SMARTFILL_NEAR_PASS_GAIN_FACTOR,
            "smartfill_height_policy": "full_domain_fair_stratified_sweep",
            "smartfill_batch_policy": "domain_capacity_pressure_deficit_and_trial_gain",
            "smartfill_action_priority": "ADD_NODE_then_MOVE_NODE_then_THICKEN_EDGE",
            "local_geometry_repair_enabled": INTERNAL_DYNAMIC_LOCAL_GEOMETRY_REPAIR_ENABLE,
            "move_node_max_trials_per_band": INTERNAL_DYNAMIC_MOVE_NODE_MAX_TRIALS_PER_BAND,
            "move_node_max_candidates_per_search": INTERNAL_DYNAMIC_MOVE_NODE_MAX_CANDIDATES_PER_SEARCH,
            "move_node_max_distance_pitch_factor": INTERNAL_DYNAMIC_MOVE_NODE_MAX_DISTANCE_PITCH_FACTOR,
            "thicken_edge_max_trials_per_band": INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_TRIALS_PER_BAND,
            "thicken_edge_max_candidates_per_search": INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_CANDIDATES_PER_SEARCH,
            "thicken_edge_step_factor": INTERNAL_DYNAMIC_THICKEN_EDGE_STEP_FACTOR,
            "thicken_edge_max_radius_factor": INTERNAL_DYNAMIC_THICKEN_EDGE_MAX_RADIUS_FACTOR,
            "local_geometry_repair_max_viscous_share": INTERNAL_DYNAMIC_LOCAL_REPAIR_MAX_VISCOUS_SHARE,
            "pressure_target_position_policy": "immutable_exact_xyz_no_support_cone_projection",
            "support_cone_xy_fallback_enabled": False,
            "bottom_support_required": bool(INTERNAL_REQUIRE_BOTTOM_SUPPORT),
            "local_lower_connection_required_for_nonseed_nodes": "required except state-detected persistent-rejection initial layers, where one legal same-layer peer edge is sufficient",
            "local_lower_connection_propagates_to_bottom": False,
            "horizontal_short_link_is_valid_support": True,
            "initial_candidate_policy": "full_component_farthest_site; persistent no-lower rejection layers may switch to same-layer-peer connectivity",
            "unreachable_pressure_target_policy": "retain exact target; persistent initial no-lower rejection layers may accept one legal same-layer peer edge instead of a lower anchor",
            "smartfill_minimum_committed_heights": 1,
            "smartfill_partial_salvage_enabled": INTERNAL_DYNAMIC_SMARTFILL_PARTIAL_SALVAGE_ENABLE,
            "failure_band_merge_mm": INTERNAL_DYNAMIC_FAILURE_BAND_MERGE_MM,
            "upper_feedback_depth_mm": INTERNAL_DYNAMIC_UPPER_FEEDBACK_DEPTH_MM,
            "upper_feedback_spacing_mm": INTERNAL_DYNAMIC_UPPER_FEEDBACK_SPACING_MM,
            "global_feedback_spacing_mm": INTERNAL_DYNAMIC_GLOBAL_FEEDBACK_SPACING_MM,
            "feedback_acceptance_grid_policy": "identical_fixed_thickness_bins_for_construction_and_acceptance",
            "feedback_acceptance_slice_count": int(PRESSURE_SLICE_COUNT),
            "feedback_spacing_rule": "fixed_nominal_slice_thickness_with_optional_short_final_bin",
            "multi_node_height_policy": "stratified_across_each_pressure_probe_cell",
            "smartfill_incremental_pressure_enabled": INTERNAL_DYNAMIC_SMARTFILL_INCREMENTAL_PRESSURE_ENABLE,
            "top_support_first_enabled": INTERNAL_TOP_SUPPORT_FIRST_ENABLE,
            "top_support_candidates_per_insert": INTERNAL_TOP_SUPPORT_CANDIDATES_PER_INSERT,
            "top_geometry_scan_limit": INTERNAL_DYNAMIC_TOP_GEOMETRY_SCAN_LIMIT,
            "top_nodes_per_exact_pressure_batch": INTERNAL_DYNAMIC_TOP_MAX_ADDS_PER_SLICE,
            "top_max_edges_per_pressure_repair_node": INTERNAL_DYNAMIC_TOP_MAX_EDGES_PER_NODE,
            "top_same_layer_edge_target": INTERNAL_DYNAMIC_TOP_SAME_LAYER_EDGES,
            "top_prefill_bridges_per_trial": INTERNAL_DYNAMIC_TOP_PREFILL_BRIDGES_PER_SLICE,
            "angle_rule_scope": (
                "all heights use the configured short-link exception and minimum elevation; "
                "no bottom-directed support path is required"
            ),
            "top_distinct_spacing_rule": "max(exported_internal_node_sphere_diameter,0.55*local_pitch)",
            "top_support_roof_probe_offsets_mm": list(INTERNAL_TOP_SUPPORT_ROOF_PROBE_OFFSETS_MM),
            "top_coverage_geometry_only_offsets_mm": list(INTERNAL_TOP_COVERAGE_DIAGNOSTIC_OFFSETS_MM),
            "min_liquid_frac": INTERNAL_DYNAMIC_MIN_LIQUID_FRAC,
            "min_liquid_area_mm2": INTERNAL_DYNAMIC_MIN_LIQUID_AREA_MM2,
            "overfill_porosity_floor": INTERNAL_DYNAMIC_OVERFILL_POROSITY_FLOOR,
            "top_porosity_floor": INTERNAL_DYNAMIC_TOP_POROSITY_FLOOR,
            "top_guard_start_frac": INTERNAL_DYNAMIC_TOP_GUARD_START_FRAC,
            "top_max_adds_per_slice": INTERNAL_DYNAMIC_TOP_MAX_ADDS_PER_SLICE,
            "max_cap_jump_ratio": INTERNAL_DYNAMIC_MAX_CAP_JUMP_RATIO,
            "capillary_min_ratio": INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO,
            "capillary_target_ratio": INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO,
            "capillary_band_start_mm": INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM,
            "capillary_assembly_reserve_ratio": INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO,
            "low_porosity_delete_threshold": INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_THRESHOLD,
            "low_porosity_delete_max_rounds": INTERNAL_DYNAMIC_LOW_POROSITY_DELETE_MAX_ROUNDS,
            "smooth_prune_max_removals": INTERNAL_DYNAMIC_SMOOTH_PRUNE_MAX_REMOVALS,
            "smooth_prune_batch_size": INTERNAL_DYNAMIC_SMOOTH_PRUNE_BATCH_SIZE,
            "final_trim_max_evaluations": INTERNAL_DYNAMIC_FINAL_TRIM_MAX_EVALUATIONS,
            "prune_max_seconds_per_stage": INTERNAL_DYNAMIC_PRUNE_MAX_SECONDS,
            "prune_max_stall_evaluations": INTERNAL_DYNAMIC_PRUNE_MAX_STALL_EVALUATIONS,
            "prune_min_relative_improvement": INTERNAL_DYNAMIC_PRUNE_MIN_RELATIVE_IMPROVEMENT,
            "internal_diam_bottom_mm": INTERNAL_DIAM_BOTTOM_MM,
            "internal_diam_top_mm": INTERNAL_DIAM_TOP_MM,
            "explicit_internal_node_spheres": True,
            "internal_node_sphere_radius_factor": NODE_DIAM_RATIO,
            "bottom_pillars_enabled": BOTTOM_PILLAR_ENABLE,
            "bottom_pillar_rows": BOTTOM_PILLAR_ROWS,
            "bottom_pillar_max_anchors": BOTTOM_PILLAR_MAX_ANCHORS,
            "bottom_pillar_spacing_mm": BOTTOM_PILLAR_ANCHOR_SPACING_MM,
            "bottom_pillar_height_layers": BOTTOM_PILLAR_HEIGHT_LAYERS,
            "bottom_pillar_radius_factor": BOTTOM_PILLAR_RADIUS_FACTOR,
            "bottom_pillar_min_component_area_mm2": BOTTOM_PILLAR_MIN_COMPONENT_AREA_MM2,
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def subdivide_edges_by_length(atoms_df: pd.DataFrame, edges_df: pd.DataFrame,
                              max_length: float, start_id: int,
                              project_mesh: Optional[trimesh.Trimesh] = None) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if atoms_df is None or atoms_df.empty or edges_df is None or edges_df.empty:
        return atoms_df, edges_df, int(start_id)
    max_length = max(float(max_length), 1e-9)
    atoms = atoms_df.copy().reset_index(drop=True)
    atom_map = atoms.set_index("atom_id")[["x", "y", "z"]].to_dict("index")
    new_atoms = []
    new_edges = []
    next_id = int(start_id)
    for edge in edges_df.itertuples(index=False):
        u = int(getattr(edge, "source"))
        v = int(getattr(edge, "target"))
        base = edge._asdict() if hasattr(edge, "_asdict") else dict(edge)
        if u not in atom_map or v not in atom_map:
            new_edges.append(base)
            continue
        p0 = np.array([float(atom_map[u]["x"]), float(atom_map[u]["y"]), float(atom_map[u]["z"])], dtype=float)
        p1 = np.array([float(atom_map[v]["x"]), float(atom_map[v]["y"]), float(atom_map[v]["z"])], dtype=float)
        length = float(np.linalg.norm(p1 - p0))
        n_seg = max(1, int(math.ceil(length / max_length)))
        chain = [u]
        interp_points = []
        for i in range(1, n_seg):
            t = float(i) / float(n_seg)
            p = p0 * (1.0 - t) + p1 * t
            interp_points.append(p)

        if project_mesh is not None and interp_points:
            try:
                projected = snap_points_to_surface(project_mesh, np.asarray(interp_points, dtype=float))
                if len(projected) == len(interp_points):
                    interp_points = [np.asarray(p, dtype=float) for p in projected]
            except Exception as exc:
                log(f"  -> Surface edge projection skipped for one edge: {exc}")

        for p in interp_points:
            nid = next_id
            next_id += 1
            new_atoms.append({
                "atom_id": int(nid),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "face_name": str(base.get("bond_type", "surface")),
            })
            chain.append(int(nid))
        chain.append(v)
        for a_id, b_id in zip(chain[:-1], chain[1:]):
            rec = dict(base)
            rec["source"] = int(a_id)
            rec["target"] = int(b_id)
            new_edges.append(rec)
    if new_atoms:
        atoms = pd.concat([atoms, pd.DataFrame(new_atoms)], ignore_index=True)
    return atoms, pd.DataFrame(new_edges), int(next_id)


def add_bottom_print_pillars(mesh, atoms_int: pd.DataFrame, edges_int: pd.DataFrame,
                             start_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if (not BOTTOM_PILLAR_ENABLE) or atoms_int is None or atoms_int.empty:
        return atoms_int, edges_int, int(start_id)
    zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    total_h = max(zmax - zmin, 1e-9)
    layer_h = total_h / max(1, int(INTERNAL_DYNAMIC_LAYERS))
    support_top = min(zmax - 1e-6, zmin + max(1.0, float(BOTTOM_PILLAR_HEIGHT_LAYERS)) * layer_h)
    cache: Dict[float, object] = {}
    z_ref = min(zmax - 1e-6, zmin + 0.5 * layer_h)
    domain_ref = _domain_at_z_cached(mesh, z_ref, cache)
    comps = [
        p for p in sorted(_as_polygons(domain_ref), key=lambda q: q.area, reverse=True)
        if float(p.area) >= float(BOTTOM_PILLAR_MIN_COMPONENT_AREA_MM2)
    ]
    if not comps:
        return atoms_int, edges_int, int(start_id)

    low_atoms = atoms_int[
        atoms_int["z"].astype(float).between(zmin - 1e-6, support_top + layer_h + 1e-6)
    ].copy()
    if low_atoms.empty:
        return atoms_int, edges_int, int(start_id)

    existing_degree = {int(i): 0 for i in atoms_int["atom_id"].astype(int).tolist()}
    if edges_int is not None and not edges_int.empty:
        for edge in edges_int.itertuples(index=False):
            if str(getattr(edge, "bond_type", "")) != "internal":
                continue
            u, v = int(getattr(edge, "source")), int(getattr(edge, "target"))
            if u in existing_degree:
                existing_degree[u] += 1
            if v in existing_degree:
                existing_degree[v] += 1

    candidates = []
    for row in low_atoms.itertuples(index=False):
        if INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW > 0 and existing_degree.get(int(row.atom_id),
                                                                          0) >= INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW:
            continue
        x, y, z = float(row.x), float(row.y), float(row.z)
        domain_at_node = _domain_at_z_cached(mesh, z, cache)
        comp_idx = _point_component_index(domain_at_node, x, y)
        if comp_idx is None:
            comp_idx = _point_component_index(domain_ref, x, y)
        if comp_idx is None:
            continue
        candidates.append((row, int(comp_idx)))
    candidates.sort(key=lambda item: (float(item[0].z), int(item[1]), float(item[0].x), float(item[0].y)))

    selected = []
    selected_xy: List[Tuple[float, float]] = []
    used_components: Set[int] = set()
    for pass_components_first in (True, False):
        for row, comp_idx in candidates:
            if len(selected) >= int(BOTTOM_PILLAR_MAX_ANCHORS):
                break
            if pass_components_first and comp_idx in used_components:
                continue
            xy = (float(row.x), float(row.y))
            if selected_xy:
                dmin = min(math.hypot(xy[0] - sx, xy[1] - sy) for sx, sy in selected_xy)
                if dmin < float(BOTTOM_PILLAR_ANCHOR_SPACING_MM):
                    continue
            selected.append((row, comp_idx))
            selected_xy.append(xy)
            used_components.add(comp_idx)
        if len(selected) >= int(BOTTOM_PILLAR_MAX_ANCHORS):
            break
    if not selected:
        log("  -> Bottom print pillars: no eligible low-layer anchors found.")
        return atoms_int, edges_int, int(start_id)

    atoms = atoms_int.copy().reset_index(drop=True)
    edges = edges_int.copy().reset_index(drop=True) if edges_int is not None and not edges_int.empty else pd.DataFrame(
        columns=["source", "target", "radius", "bond_type", "role"])
    new_atoms, new_edges = [], []
    nid = int(start_id)
    radius = max(1e-6, _local_internal_radius(zmin + 0.5 * layer_h, (zmin, zmax)) * float(BOTTOM_PILLAR_RADIUS_FACTOR))
    rows = max(2, int(BOTTOM_PILLAR_ROWS))
    max_len = max(layer_h, _local_internal_pitch(zmin, (zmin, zmax))) * INTERNAL_DYNAMIC_MAX_EDGE_LENGTH_FACTOR

    for anchor, comp_idx in selected:
        x, y, z_anchor = float(anchor.x), float(anchor.y), float(anchor.z)
        z_top = min(float(z_anchor), support_top)
        z_list = np.linspace(zmin + 1e-4, max(zmin + 2e-4, z_top), rows)
        col_ids = []
        for z in z_list:
            new_atoms.append({
                "atom_id": int(nid),
                "x": x,
                "y": y,
                "z": float(z),
                "face_name": f"bottom_pillar_C{comp_idx}",
            })
            col_ids.append(int(nid))
            nid += 1
        for a_id, b_id in zip(col_ids[:-1], col_ids[1:]):
            new_edges.append(
                {"source": a_id, "target": b_id, "radius": radius, "bond_type": "internal", "role": "bottom_pillar"})
        top_id = col_ids[-1]
        if math.sqrt((z_anchor - float(z_list[-1])) ** 2) <= max_len + 1e-9:
            new_edges.append({
                "source": top_id,
                "target": int(anchor.atom_id),
                "radius": radius,
                "bond_type": "internal",
                "role": "bottom_pillar_to_core",
            })

    if new_atoms:
        atoms = pd.concat([atoms, pd.DataFrame(new_atoms)], ignore_index=True)
    if new_edges:
        edges = pd.concat([edges, pd.DataFrame(new_edges)], ignore_index=True)
    log(f"  -> Bottom print pillars: anchors={len(selected)}, nodes={len(new_atoms)}, edges={len(new_edges)}, radius={radius:.3f}mm")
    return atoms, edges, int(nid)


def _surface_build_config(target_faces: Optional[int] = None) -> SurfaceBuildConfig:
    """Create the surface-builder configuration used by both production and scans."""
    surface_max_len = max(0.75, min(1.50, 3.0 * float(R_EFF_LIMIT_MM)))
    return SurfaceBuildConfig(
        target_faces=int(INITIAL_FACE_COUNT if target_faces is None else target_faces),
        max_faces_per_patch=int(SURFACE_MAX_FACES_PER_PATCH),
        normal_angle_deg=24.0,
        split_rounds=_effective_surface_split_rounds(),
        surface_diameter_mm=float(SURFACE_DIAM_MM),
        target_hydraulic_radius_mm=float(R_EFF_LIMIT_MM),
        target_radius_source=SURFACE_RADIUS_SOURCE,
        design_target_height_mm=float(TARGET_HEIGHT_MM),
        surface_tension_n_per_m=float(GAMMA_NPM),
        contact_angle_deg=float(CONTACT_ANGLE_DEG),
        liquid_density_kg_per_m3=float(RHO_LIQ),
        gas_density_kg_per_m3=float(RHO_GAS),
        gravity_m_per_s2=float(G_GRAV),
        additional_pressure_pa=float(SURFACE_EXTRA_PRESSURE_PA),
        capillary_safety_factor=float(SURFACE_CAPILLARY_SAFETY_FACTOR),
        seam_screen_enabled=True,
        seam_screen_max_fraction=float(SURFACE_SEAM_MAX_FRACTION),
        anisotropy_rewrite_enable=True,
        anisotropy_target_aspect_ratio=float(SURFACE_ANISOTROPY_TARGET_AR),
        anisotropy_max_aspect_ratio=float(SURFACE_AR_CAP_MAX),
        anisotropy_cap_min_area_mm2=float(SURFACE_AR_CAP_MIN_AREA_MM2),
        zero_pore_min_area_mm2=float(SURFACE_MIN_PORE_AREA_MM2),
        global_ar_cap_rewrite_enable=bool(SURFACE_GLOBAL_AR_CAP_REWRITE_ENABLE),
        global_ar_cap_audit_max_edge_mm=float(SURFACE_GLOBAL_AUDIT_MAX_EDGE_MM),
        max_insertions_per_patch=0,
        max_voronoi_rounds=100,
        refinement_insertion_strategy="polylabel",
        centroid_jitter_mm=0.01,
        max_edge_length_mm=surface_max_len,
        max_node_degree=8,
        edge_sample_spacing_mm=min(0.35, surface_max_len / 3.0),
        surface_tolerance_mm=min(0.025, 0.0625 * float(SURFACE_DIAM_MM)),
        mapping_surface="carrier",
        source_node_spacing_mm=0.65,
        min_boundary_carriers_per_patch=3,
        outward_offset_mm=float(SURFACE_OUTWARD_OFFSET_MM),
        random_seed=int(INTERNAL_DYNAMIC_SEED),
    )


def run_v11_global_surface_pore_audit() -> Dict:
    """Read-only contour audit on the final surface graph.

    The surface builder has already applied the physical contour-radius repair.
    This function independently reconstructs cross-patch complete pores on a
    conformingly refined carrier and does not modify the graph.
    """
    audit_args = argparse.Namespace(
        stl=Path(IN_STL),
        run_dir=Path(OUT_DIR),
        target_height_mm=float(TARGET_HEIGHT_MM),
        global_max_edge_mm=float(SURFACE_GLOBAL_AUDIT_MAX_EDGE_MM),
        solid_vote_fraction=0.5,
        aspect_ratio_max=float(SURFACE_AR_CAP_MAX),
        aspect_ratio_min_area_mm2=float(SURFACE_AR_CAP_MIN_AREA_MM2),
        minimum_pore_area_mm2=float(SURFACE_MIN_PORE_AREA_MM2),
        radius_acceptance_mm=None,
        search_padding_mm=0.02,
        out_dir_name="global_surface_pore_audit",
        export_face_classification=False,
    )
    result = run_global_surface_pore_audit(audit_args)
    log(
        "  Global contour surface audit: "
        f"complete_pores={int(result.get('global_complete_pore_count', 0))}, "
        f"radius_exceeded={int(result.get('global_radius_exceeded_complete_pore_count', 0))}, "
        f"AR_exceeded(A>{SURFACE_AR_CAP_MIN_AREA_MM2:g})="
        f"{int(result.get('global_aspect_ratio_exceeded_complete_pore_count', 0))}, "
        f"pass={bool(result.get('global_complete_pore_acceptance_pass', False))}."
    )
    return result


def _log_surface_seam_audit(audit: Dict, prefix: str = "Surface seam audit") -> None:
    seam_fraction = float(audit.get('seam_dependent_local_record_fraction', 0.0))
    max_seam = audit.get('max_seam_open_equivalent_radius_chart_mm')
    max_nonseam = audit.get('max_nonseam_open_equivalent_radius_chart_mm')
    max_seam_text = 'n/a' if max_seam is None else f'{float(max_seam):.6f} mm'
    max_nonseam_text = 'n/a' if max_nonseam is None else f'{float(max_nonseam):.6f} mm'
    log(
        f"  {prefix}: patches={int(audit.get('patch_count', 0))}, "
        f"local_records={int(audit.get('local_pore_record_count', audit.get('pore_count', 0)))}, "
        f"seam_records={int(audit.get('seam_dependent_pore_count', 0))} "
        f"({100.0 * seam_fraction:.1f}%), "
        f"seam_radius_exceeded_local={int(audit.get('seam_radius_exceeded_local_record_count', 0))}, "
        f"nonseam_radius_exceeded={int(audit.get('nonseam_radius_exceeded_pore_count', 0))}, "
        f"max_seam_r_eq={max_seam_text}, max_nonseam_r_eq={max_nonseam_text}, "
        f"seam_screen_pass={bool(audit.get('seam_screen_pass', False))}."
    )


def run_surface_face_scan(mesh) -> Optional[int]:
    """Scan target face counts downward and select the highest passing candidate.

    The scan is deliberately surface-only and lightweight: no internal graph is
    constructed.  Each candidate is audited as local chart records.  The hard
    local capillary decision uses only non-seam records, while the artificial
    seam burden is screened by SURFACE_SEAM_MAX_FRACTION.  Seam-local oversize
    counts are reported but remain pending global cross-patch interpretation.
    """
    scan_dir = OUT_DIR / 'surface_face_scan'
    scan_dir.mkdir(parents=True, exist_ok=True)
    native_faces = max(16, int(len(mesh.faces)))
    start_faces = min(int(SURFACE_FACE_SCAN_START), native_faces)
    stop_faces = min(int(SURFACE_FACE_SCAN_STOP), start_faces)
    step = max(1, int(SURFACE_FACE_SCAN_STEP))
    candidates = list(range(start_faces, stop_faces - 1, -step))
    if not candidates or candidates[-1] != stop_faces:
        candidates.append(stop_faces)

    rows: List[Dict] = []
    selected: Optional[int] = None
    for target_faces in candidates:
        started = time.perf_counter()
        cfg = _surface_build_config(target_faces)
        builder = SurfacePatchBuilder(mesh, cfg).build()
        candidate_dir = scan_dir / f'faces_{int(target_faces)}'
        audit = builder.audit_final_retained_graph(
            builder.atoms, builder.edges, candidate_dir,
            node_spheres_exported=False,
            include_stitch_anchors=False,
            apply_quality_gate=False,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        row = {
            'requested_target_faces': int(target_faces),
            'effective_base_mesh_faces': int(len(builder.base_mesh.faces)) if builder.base_mesh is not None else None,
            'patch_count': int(len(builder.charts)),
            'surface_node_count': int(len(builder.atoms)),
            'surface_edge_segment_count': int(len(builder.edges)),
            'local_pore_record_count': int(audit.get('local_pore_record_count', audit.get('pore_count', 0))),
            'seam_dependent_local_record_count': int(audit.get('seam_dependent_pore_count', 0)),
            'seam_dependent_local_record_fraction': float(audit.get('seam_dependent_local_record_fraction', 0.0)),
            'physical_boundary_dependent_local_record_count': int(audit.get('physical_boundary_dependent_local_record_count', 0)),
            'nonseam_radius_exceeded_pore_count': int(audit.get('nonseam_radius_exceeded_pore_count', 0)),
            'seam_radius_exceeded_local_record_count': int(audit.get('seam_radius_exceeded_local_record_count', 0)),
            'max_nonseam_open_equivalent_radius_chart_mm': audit.get('max_nonseam_open_equivalent_radius_chart_mm'),
            'max_seam_open_equivalent_radius_chart_mm': audit.get('max_seam_open_equivalent_radius_chart_mm'),
            'local_chart_acceptance_pass': bool(audit.get('local_chart_acceptance_pass', False)),
            'seam_screen_pass': bool(audit.get('seam_screen_pass', False)),
            'provisional_surface_acceptance_pass': bool(audit.get('provisional_surface_acceptance_pass', False)),
            'elapsed_seconds': float(elapsed),
        }
        rows.append(row)
        _log_surface_seam_audit(audit, prefix=f'Face scan target={target_faces}')
        if row['provisional_surface_acceptance_pass']:
            selected = int(target_faces)
            break

    scan_frame = pd.DataFrame(rows)
    scan_frame.to_csv(scan_dir / 'surface_face_scan_summary.csv', index=False)
    selection = {
        'scan_start_faces': int(start_faces),
        'scan_stop_faces': int(stop_faces),
        'scan_step_faces': int(step),
        'native_input_face_count': int(native_faces),
        'seam_screen_max_local_record_fraction': float(SURFACE_SEAM_MAX_FRACTION),
        'selected_target_faces': selected,
        'selection_rule': (
            'first descending target-face count with non-seam local capillary pass and seam-burden pass'),
        'seam_oversize_interpretation': (
            'reported local seam fragments only; not counted as unique real complete pores until cross-patch merge or final 3-D inspection'),
        'global_complete_pore_count_available': False,
    }
    (scan_dir / 'surface_face_scan_selection.json').write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
    if selected is None:
        log('  Surface face scan: no candidate passed the provisional surface screen. '
            'Inspect surface_face_scan_summary.csv; do not claim global pore compliance.')
    else:
        log(f'  Surface face scan selected target_faces={selected}.')
    return selected


def build_surface_structure(mesh) -> Tuple[SurfacePatchBuilder, pd.DataFrame, pd.DataFrame, int]:
    surface_cfg = _surface_build_config()
    surface_max_len = float(surface_cfg.max_edge_length_mm)
    os.environ.setdefault('PC_AEGG_CONTOUR_DIAGNOSTIC_DIR', str(OUT_DIR / 'contour_diagnostics'))
    surf_sys = SurfacePatchBuilder(mesh, surface_cfg).build()
    surf_sys.export_diagnostics(
        OUT_DIR / "surface_validation",
        model_name=IN_STL.stem,
        export_stl=False,
    )
    export_pore_and_pressure_analysis(
        OUT_DIR / "surface_validation",
        surface_cfg,
        layers=20,
        gamma_npm=GAMMA_NPM,
        liquid_density_kgm3=RHO_LIQ,
        gas_density_kgm3=RHO_GAS,
        gravity_ms2=G_GRAV,
        contact_angle_deg=CONTACT_ANGLE_DEG,
        additional_pressure_pa=SURFACE_EXTRA_PRESSURE_PA,
        safety_factor=SURFACE_CAPILLARY_SAFETY_FACTOR,
    )
    surf_sys.offset_outward(SURFACE_OUTWARD_OFFSET_MM)
    atoms_surf, edges_surf = surf_sys.export_data()
    next_id = int(atoms_surf["atom_id"].max()) + 1 if not atoms_surf.empty else 0
    log(
        f"  -> Curvature patches={len(surf_sys.charts)}, "
        f"surface nodes={len(atoms_surf)}, edge segments={len(edges_surf)}, "
        f"target faces={surface_cfg.target_faces}, max faces/patch={surface_cfg.max_faces_per_patch}, "
        f"max segment length={surface_max_len:.3f}mm"
    )
    export_stl(atoms_surf, edges_surf, OUT_DIR / "surface_only.stl")
    return surf_sys, atoms_surf, edges_surf, next_id

def build_internal_core_structure(
        mesh,
        start_id: int = 0,
        atoms_surf: Optional[pd.DataFrame] = None,
        edges_surf: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    atoms_int, edges_int = build_internal_structure_v2(
        mesh,
        int(start_id),
        1.0,
        atoms_surf=atoms_surf,
        edges_surf=edges_surf,
        output_dir=OUT_DIR,
    )
    max_ids = [int(start_id) - 1]
    if atoms_surf is not None and not atoms_surf.empty:
        max_ids.append(int(atoms_surf["atom_id"].max()))
    if atoms_int is not None and not atoms_int.empty:
        max_ids.append(int(atoms_int["atom_id"].max()))
    next_id = max(max_ids) + 1

    export_stl(atoms_int, edges_int, OUT_DIR / "internal_only.stl")
    return atoms_int, edges_int, int(next_id)


def run_global_physics_outputs(mesh, atoms_all: pd.DataFrame, edges_all: pd.DataFrame, z_bounds: Tuple[float, float]):
    nodes_np = _make_nodes_np_for_analysis(atoms_all, z_bounds)

    log("Starting full physics slice analysis...")
    zs = _pressure_feedback_heights(z_bounds)
    df_res = run_full_analysis(mesh, atoms_all, edges_all, nodes_np, zs)

    if not df_res.empty:
        df_sorted = df_res.sort_values("z_mm")
        z_arr = df_sorted["z_mm"].values * 1e-3
        A_liq_arr_real = (df_sorted["A_domain_mm2"] * df_sorted["porosity"]).values * 1e-6
        dz_arr_m = (df_sorted['slice_dz_mm'].to_numpy(dtype=float)
                    if 'slice_dz_mm' in df_sorted.columns
                    else np.full(len(df_sorted), float(SLICE_DZ_MM), dtype=float)) * 1e-3
        dE = RHO_LIQ * G_GRAV * z_arr * A_liq_arr_real * dz_arr_m
        total_energy = np.sum(dE)
        log(f"Total liquid gravitational potential energy: {total_energy:.6f} Joules")

        csv_path = OUT_DIR / "internal_slice_physics_analysis.csv"
        df_res.to_csv(csv_path, index=False)
        export_dynamic_analysis_outputs(mesh, atoms_all, edges_all, nodes_np, df_res, OUT_DIR)
        log("Physics charts generated.")
    return df_res


MODEL_CONSTRUCTION_STAGES = {
    "Surface network construction",
    "Adaptive internal construction",
    "Final internal pressure trim",
    "Surface-to-core stitching",
}
OUTPUT_ONLY_STAGES = {
    "Final internal export",
    "Final mesh export",
    "Analysis tables and figure export",
    "Physics analysis and export",
}
VALIDATION_ONLY_STAGES = {
    "Read-only assembled pressure verification",
    "Final graph and STL geometry audit",
    "Surface physics diagnostics",
    "Global slice physics analysis",
}


def _timing_category(stage: str) -> str:
    if stage in MODEL_CONSTRUCTION_STAGES:
        return "model_construction_or_repair"
    if stage in OUTPUT_ONLY_STAGES:
        return "output_only"
    if stage in VALIDATION_ONLY_STAGES:
        return "validation_only"
    return "setup_or_other"


def _timing_totals(stage_rows: List[Dict]) -> Dict[str, float]:
    totals = {
        "model_construction_and_repair_seconds": 0.0,
        "validation_seconds": 0.0,
        "output_seconds": 0.0,
        "setup_or_other_seconds": 0.0,
        "elapsed_stage_total_seconds": 0.0,
    }
    key_by_category = {
        "model_construction_or_repair": "model_construction_and_repair_seconds",
        "validation_only": "validation_seconds",
        "output_only": "output_seconds",
        "setup_or_other": "setup_or_other_seconds",
    }
    for row in stage_rows:
        seconds = float(row["duration_seconds"])
        totals["elapsed_stage_total_seconds"] += seconds
        totals[key_by_category[_timing_category(str(row["stage"]))]] += seconds
    return totals


def export_build_stage_timing(stage_rows: List[Dict], output_dir: Path) -> pd.DataFrame:
    """Report true construction/repair time separately from validation/output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(stage_rows)
    if df.empty:
        return df
    df = df.copy()
    totals = _timing_totals(stage_rows)
    total_s = float(totals["elapsed_stage_total_seconds"])
    model_s = float(totals["model_construction_and_repair_seconds"])
    df["timing_category"] = df["stage"].astype(str).map(_timing_category)
    df["included_in_model_time"] = df["timing_category"].eq(
        "model_construction_or_repair")
    df["sequence"] = np.arange(1, len(df) + 1)
    df["share_percent"] = np.where(total_s > 0, 100.0 * df["duration_seconds"] / total_s, 0.0)
    df["model_construction_and_repair_seconds"] = model_s
    df["elapsed_stage_total_seconds"] = total_s
    df.to_csv(output_dir / "build_stage_timing.csv", index=False)

    durations = df["duration_seconds"].to_numpy(dtype=float)
    d_min, d_max = float(durations.min()), float(durations.max())
    if d_max > d_min:
        sizes = 180.0 + 920.0 * (durations - d_min) / (d_max - d_min)
    else:
        sizes = np.full_like(durations, 500.0)
    y = np.arange(len(df), dtype=float)
    mid_minutes = 0.5 * (df["start_seconds"].to_numpy() + df["end_seconds"].to_numpy()) / 60.0

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 1.4,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }):
        fig, ax = plt.subplots(figsize=(7.2, max(5.2, 0.58 * len(df) + 1.5)))
        for yi, row in zip(y, df.itertuples(index=False)):
            ax.hlines(
                yi, float(row.start_seconds) / 60.0, float(row.end_seconds) / 60.0,
                color=FIGURE_GREY, linewidth=2.0, alpha=0.48, zorder=1,
            )
        scatter = ax.scatter(
            mid_minutes, y,
            c=df["share_percent"], s=sizes, cmap=FIGURE_CMAP,
            edgecolors="white", linewidth=0.8, alpha=0.95, zorder=2,
            vmin=0.0, vmax=max(float(df["share_percent"].max()), 1.0),
        )
        for x_mid, yi, seconds in zip(mid_minutes, y, durations):
            label = f"{seconds / 60.0:.1f} min" if seconds >= 60.0 else f"{seconds:.0f} s"
            place_right = bool(x_mid <= 0.85 * total_s / 60.0)
            ax.annotate(label, (x_mid, yi), xytext=(10 if place_right else -10, 0),
                        textcoords="offset points", ha="left" if place_right else "right",
                        va="center", fontsize=9, color="#333333")
        ax.set_yticks(y, df["stage"])
        ax.invert_yaxis()
        ax.set_xlabel("Cumulative build time (min)", fontweight="bold")
        ax.set_ylabel("Build stage", fontweight="bold")
        ax.set_title(f"Construction and repair timing ({model_s / 60.0:.1f} min model time)",
                     fontweight="bold", pad=12)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.65)
        ax.tick_params(direction="in", top=True, right=True)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Share of total time (%)", fontweight="bold")
        cbar.outline.set_linewidth(1.2)
        fig.tight_layout()
        _save_svg_only(fig, output_dir / "build_stage_timing.svg")
        plt.close(fig)
    log(
        f"Model construction/repair time: {model_s:.1f} s "
        f"({model_s / 60.0:.2f} min); validation={totals['validation_seconds']:.1f}s, "
        f"output={totals['output_seconds']:.1f}s, elapsed stages={total_s:.1f}s."
    )
    return df


def export_build_key_metrics(stage_rows: List[Dict], output_dir: Path,
                             final_edges: pd.DataFrame, pressure_profile: pd.DataFrame,
                             *, source: str) -> Dict:
    """Read-only report using the highest available fixed-thickness pressure slice."""
    kinds = final_edges['bond_type'].astype(str).value_counts().to_dict() if (
            final_edges is not None and not final_edges.empty and 'bond_type' in final_edges
    ) else {}
    timing_totals = _timing_totals(stage_rows)
    summary = {
        'source': source,
        'stage_seconds': {str(r['stage']): float(r['duration_seconds']) for r in stage_rows},
        **timing_totals,
        'model_time_definition': (
            'surface construction + adaptive internal construction/repair (including bottom supports) + '
            'final internal trim + surface-to-core stitching; '
            'validation, STL/CSV/SVG export, figures and setup are excluded'
        ),
        'final_graph_edge_count': int(len(final_edges)) if final_edges is not None else 0,
        'final_graph_edge_counts_by_type': {str(k): int(v) for k, v in kinds.items()},
        'pressure_slice_nominal_thickness_mm': float(SLICE_DZ_MM),
        'reference_flow_rate_m3_s': float(VISC_REF_Q_M3_S),
        'top_pressure_slice': None,
    }
    if pressure_profile is not None and not pressure_profile.empty:
        z_col = 'z_mid_mm' if 'z_mid_mm' in pressure_profile.columns else 'z_mm'
        heights = pd.to_numeric(pressure_profile[z_col], errors='coerce')
        valid = heights.notna()
        if bool(valid.any()):
            idx = heights[valid].idxmax()
            row = pressure_profile.loc[idx]
            z_actual = float(heights.loc[idx])
            p_cap = float(row['P_cap_Pa'])
            p_required = float(row['P_required_Pa'])
            p_required_base = float(row.get('P_required_base_Pa', p_required))
            p_hydro = float(row['P_hydro_Pa'])
            p_visc = float(row.get('P_viscous_Pa', row.get('P_viscous_capacity_proxy_Pa', 0.0)))
            lower_margin = float(_capillary_margin_values(
                pressure_profile.loc[[idx]], include_assembly_reserve=False)[0])
            summary['top_pressure_slice'] = {
                'actual_slice_z_mm': z_actual,
                'slice_dz_mm': float(row.get('slice_dz_mm', SLICE_DZ_MM)),
                'P_cap_Pa': p_cap,
                'P_hydro_Pa': p_hydro,
                'P_viscous_Pa': p_visc,
                'P_required_base_Pa': p_required_base,
                'P_required_Pa': p_required,
                'transport_margin_Pa': float(row.get('transport_margin_Pa', p_cap - p_required_base)),
                'capillary_lower_bound_margin_Pa': lower_margin,
                'lower_bound_pass': bool(lower_margin >= 0.0),
            }
    (output_dir / 'build_key_metrics.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    top = summary['top_pressure_slice']
    log('  Build metrics: retained edges={} (internal={}, surface={}, stitch={}), '
        'stage time={:.1f}s; top-slice transport margin={} Pa, lower-bound margin={} Pa.'.format(
        summary['final_graph_edge_count'], kinds.get('internal', 0),
        kinds.get('surface', 0), kinds.get('stitch', 0),
        summary['elapsed_stage_total_seconds'],
        f"{top['transport_margin_Pa']:.3f}" if top else 'not sampled',
        f"{top['capillary_lower_bound_margin_Pa']:.3f}" if top else 'not sampled'
    ))
    return summary


def summarize_final_pressure_screen(pressure_profile: pd.DataFrame,
                                    z_bounds: Optional[Tuple[float, float]] = None) -> Dict:
    """Build the final fixed-grid pressure gate for either internal-only or assembled runs."""
    if z_bounds is None:
        z_bounds = (0.0, TARGET_HEIGHT_MM)
    expected_z, expected_widths = _pressure_feedback_grid(z_bounds)
    expected_slice_count = len(expected_z)
    result = {
        'slice_count': int(len(pressure_profile)) if pressure_profile is not None else 0,
        'expected_slice_count': int(expected_slice_count),
        'lower_bound_fail_count': 0,
        'minimum_lower_bound_margin_pa': None,
        'ordinary_requirement_fail_count': 0,
        'minimum_ordinary_margin_pa': None,
        'pressure_slice_nominal_thickness_mm': float(SLICE_DZ_MM),
        'pass': False,
    }
    if pressure_profile is None or pressure_profile.empty:
        return result
    z_column = 'z_mid_mm' if 'z_mid_mm' in pressure_profile.columns else 'z_mm'
    z_values = pressure_profile[z_column].to_numpy(dtype=float)
    widths = (pressure_profile['slice_dz_mm'].to_numpy(dtype=float)
              if 'slice_dz_mm' in pressure_profile.columns else np.asarray([], dtype=float))
    grid_pass = bool(
        len(z_values) == len(expected_z)
        and len(widths) == len(expected_widths)
        and np.isfinite(z_values).all()
        and np.isfinite(widths).all()
        and np.allclose(z_values, expected_z, rtol=0.0, atol=1.0e-8)
        and np.allclose(widths, expected_widths, rtol=0.0, atol=1.0e-8)
    )
    lower = _capillary_margin_values(pressure_profile, include_assembly_reserve=False)
    ordinary = (
        pressure_profile['transport_margin_Pa'].to_numpy(dtype=float)
        if 'transport_margin_Pa' in pressure_profile.columns
        else pressure_profile['P_cap_Pa'].to_numpy(dtype=float)
             - pressure_profile.get(
            'P_required_base_Pa', pressure_profile['P_required_Pa']
        ).to_numpy(dtype=float)
    )
    result.update({
        'lower_bound_fail_count': int(np.sum(lower < 0.0)),
        'minimum_lower_bound_margin_pa': float(np.min(lower)),
        'ordinary_requirement_fail_count': int(np.sum(ordinary < 0.0)),
        'minimum_ordinary_margin_pa': float(np.min(ordinary)),
        'pressure_grid_pass': grid_pass,
        'pass': bool(
            len(pressure_profile) == expected_slice_count
            and grid_pass
            and len(lower) == expected_slice_count
            and len(ordinary) == expected_slice_count
            and np.all(np.isfinite(lower))
            and np.all(np.isfinite(ordinary))
            and np.all(lower >= 0.0)
            and np.all(ordinary >= 0.0)
        ),
    })
    return result


def evaluate_internal_only_acceptance(internal_support_final, abnormal_layer_audit,
                                      pressure_screen, final_stl_audit, trim_status):
    """Final acceptance gate used when surface design is intentionally disabled."""
    internal_pass = bool(internal_support_final.get('print_support_screen_pass', False))
    abnormal_pass = bool(
        abnormal_layer_audit.get('abnormal_layer_connectivity_pass', False))
    pressure_pass = bool(pressure_screen.get('pass', False))
    stl_pass = bool(
        final_stl_audit.get('binary_file_format_valid', False)
        and final_stl_audit.get('finite_vertices', False))
    graph_pass = bool(internal_pass and abnormal_pass and pressure_pass and stl_pass)
    return {
        'unified_acceptance_gate_version': 'PC-AEGG-ABNORMAL-LAYER-PRESSURE-20260929',
        'surface_design_enabled': False,
        'surface_gate_not_applicable': True,
        'internal_print_support_screen_pass': internal_pass,
        'abnormal_layer_exception_present': bool(
            abnormal_layer_audit.get('abnormal_layer_exception_present', False)),
        'abnormal_layer_connectivity_pass': abnormal_pass,
        'abnormal_layer_numbers_1_based': abnormal_layer_audit.get(
            'abnormal_layer_numbers_1_based', []),
        'pressure_profile_pass': pressure_pass,
        'fine_slice_pressure_fail_count': int(
            pressure_screen.get('lower_bound_fail_count', 0)),
        'stl_format_pass': stl_pass,
        'internal_pre_stitch_trim_status': trim_status.get('status', 'unknown'),
        'abnormal_layer_acceptance_rule': (
            'Abnormal layers are allowed and are not a failure by existence. Acceptance requires '
            'all retained abnormal nodes to have valid same-layer peer connections, the final '
            'fixed-grid pressure profile to pass, the internal print screen to pass, and a valid STL.'
        ),
        'graph_screen_pass': graph_pass,
        'status': ('graph_screen_pass_additional_validation_pending'
                   if graph_pass else 'graph_screen_fail'),
    }


def evaluate_unified_acceptance(dual_final_audit, internal_support_final,
                                final_surface_audit, final_global_surface_audit,
                                assembly_pressure_status,
                                fine_pressure_screen, final_stl_audit,
                                trim_status, abnormal_layer_audit=None):
    """Apply the one final gate used by every design domain.

    Construction feedback and final pressure acceptance share the same fixed
    nominal 1-mm pressure bins; the number of bins scales with design height.
    The surface decision uses the builder's positive-open-area local pore gate
    and the patch-independent global complete-pore gate. Fully occupied local
    records remain in raw diagnostics but do not count as failed open pores.
    """
    dual_pass = bool(dual_final_audit.get('dual_domain_pass', False))
    internal_pass = bool(
        internal_support_final.get('print_support_screen_pass', False))
    surface_local_pass = bool(
        final_surface_audit.get('local_chart_acceptance_pass', False))
    surface_seam_screen_pass = bool(
        final_surface_audit.get('seam_screen_pass', True))
    surface_global_pass = bool(
        final_global_surface_audit.get('global_complete_pore_acceptance_pass', False))
    # Final decision: local graph integrity + reconstructed global complete-pore physics.
    # Artificial seam-local fragments remain diagnostic and do not override a passing
    # cross-patch complete-pore reconstruction.
    surface_pass = bool(
        surface_local_pass and surface_global_pass
        and int(final_surface_audit.get('missing_edge_endpoints', 0)) == 0
        and int(final_global_surface_audit.get('missing_surface_edge_endpoint_count', 0)) == 0
    )
    coarse_pressure_pass = bool(
        assembly_pressure_status.get('pressure_profile_pass', False))
    fine_pressure_pass = bool(fine_pressure_screen.get('pass', False))
    pressure_pass = bool(coarse_pressure_pass and fine_pressure_pass)
    stl_pass = bool(
        final_stl_audit.get('binary_file_format_valid', False)
        and final_stl_audit.get('finite_vertices', False)
    )
    abnormal_layer_audit = abnormal_layer_audit or {
        'abnormal_layer_exception_present': False,
        'abnormal_layer_connectivity_pass': True,
        'abnormal_layer_numbers_1_based': [],
    }
    abnormal_pass = bool(
        abnormal_layer_audit.get('abnormal_layer_connectivity_pass', False))
    checks = dict(
        unified_acceptance_gate_version='PC-AEGG-OPEN-PORE-GATE-20261008',
        pressure_slice_count=int(fine_pressure_screen.get('slice_count', PRESSURE_SLICE_COUNT)),
        pressure_feedback_and_acceptance_share_grid=True,
        dual_graph_screen_pass=dual_pass,
        internal_print_support_screen_pass=internal_pass,
        abnormal_layer_exception_present=bool(
            abnormal_layer_audit.get('abnormal_layer_exception_present', False)),
        abnormal_layer_connectivity_pass=abnormal_pass,
        abnormal_layer_numbers_1_based=abnormal_layer_audit.get(
            'abnormal_layer_numbers_1_based', []),
        abnormal_layer_acceptance_rule=(
            'Abnormal layers are permitted; their existence is diagnostic, not a rejection. '
            'Every retained abnormal node must be connected by a valid same-layer peer edge, '
            'and all pressure, internal, surface, dual-graph and STL checks must still pass.'
        ),
        surface_open_pore_graph_screen_pass=surface_pass,
        surface_local_nonseam_capillary_pass=surface_local_pass,
        surface_seam_burden_screen_pass=surface_seam_screen_pass,
        surface_global_complete_pore_acceptance_pass=surface_global_pass,
        surface_global_radius_acceptance_pass=bool(
            final_global_surface_audit.get('global_complete_pore_radius_acceptance_pass', False)),
        surface_global_aspect_ratio_reference_pass=bool(
            final_global_surface_audit.get('global_complete_pore_aspect_ratio_acceptance_pass', False)),
        surface_global_aspect_ratio_diagnostic_only=True,
        surface_global_minimum_area_acceptance_pass=bool(
            final_global_surface_audit.get('global_complete_pore_minimum_area_acceptance_pass', False)),
        surface_global_complete_pore_count=int(
            final_global_surface_audit.get('global_complete_pore_count', 0)),
        surface_global_radius_exceeded_complete_pore_count=int(
            final_global_surface_audit.get('global_radius_exceeded_complete_pore_count', 0)),
        surface_global_AR_exceeded_complete_pore_count=int(
            final_global_surface_audit.get('global_aspect_ratio_exceeded_complete_pore_count', 0)),
        surface_acceptance_source=(
            'local retained-graph integrity AND global_surface_pore_audit.'
            'global_complete_pore_acceptance_pass'
        ),
        surface_missing_edge_endpoints_diagnostic=int(
            final_surface_audit.get('missing_edge_endpoints', 0)),
        surface_cross_patch_bridge_outside_chart_diagnostic=int(
            final_surface_audit.get('cross_patch_bridge_outside_chart', 0)),
        surface_seam_dependent_cells_excluded_from_local_gate=True,
        surface_undersized_open_fragments_diagnostic_only=True,
        surface_fully_occupied_zero_area_regions_treated_as_solid=True,
        pressure_profile_pass=pressure_pass,
        coarse_assembled_pressure_pass=coarse_pressure_pass,
        fine_slice_pressure_pass=fine_pressure_pass,
        fine_slice_pressure_fail_count=int(
            fine_pressure_screen.get('lower_bound_fail_count', 0)),
        internal_pre_stitch_trim_status=trim_status.get('status', 'unknown'),
        assembled_pressure_is_read_only=True,
        stl_format_pass=stl_pass,
        residual_blocked_pores=int(
            final_surface_audit.get('nonseam_positive_open_area_blocked_region_count', 0)),
        zero_open_area_solid_local_records=int(
            final_surface_audit.get('blocked_pore_count', 0)),
        nonseam_blocked_region_count=int(
            final_surface_audit.get('nonseam_positive_open_area_blocked_region_count', 0)),
        nonseam_undersized_open_pore_count=int(
            final_surface_audit.get('nonseam_undersized_open_pore_count', 0)),
        nonseam_radius_exceeded_pore_count=int(
            final_surface_audit.get('nonseam_radius_exceeded_pore_count', 0)),
        unresolved_surface_seam_local_records=int(
            final_surface_audit.get('seam_dependent_pore_count', 0)),
        surface_seam_local_record_fraction=float(
            final_surface_audit.get('seam_dependent_local_record_fraction', 0.0)),
        surface_seam_screen_max_fraction=float(
            final_surface_audit.get('seam_screen_max_fraction', SURFACE_SEAM_MAX_FRACTION)),
        surface_seam_radius_exceeded_local_record_count=int(
            final_surface_audit.get('seam_radius_exceeded_local_record_count', 0)),
        surface_global_complete_pore_count_available=bool(
            final_global_surface_audit.get('global_complete_pore_count_available', True)),
        surface_seam_pore_interpretation=(
            'Artificial inter-patch seam records are local pore fragments rather than unique physical pores. '
            'They are retained as diagnostics; the final pore-physics decision is made on reconstructed '
            'cross-patch global complete pores.'
        ),
        minimum_capillary_viscous_mobility_Pa_m3=(
            fine_pressure_screen.get('minimum_capillary_viscous_mobility_Pa_m3')),
        mobility_bottleneck_z_mm=(
            fine_pressure_screen.get('mobility_bottleneck_z_mm')),
        maximum_cumulative_geometry_resistance_m3=(
            fine_pressure_screen.get('maximum_cumulative_geometry_resistance_m-3')),
        capillary_stall_count=int(
            fine_pressure_screen.get('capillary_stall_count', 0)),
        resistance_calibration_factor=float(FLOW_RESISTANCE_CALIBRATION_FACTOR),
        reference_flow_rate_m3_s=float(VISC_REF_Q_M3_S),
        pressure_slice_nominal_thickness_mm=float(SLICE_DZ_MM),
        ordinary_fixed_flow_transport_reach_mm=(
            fine_pressure_screen.get('ordinary_fixed_flow_transport_reach_mm')),
        design_reserve_transport_reach_mm=(
            fine_pressure_screen.get('design_reserve_transport_reach_mm')),
        exact_3d_swept_cylinder_envelope_verified=False,
        short_bridge_am_span_experimentally_validated=False,
        top_any_angle_relaxation_used_edge_count=int(
            internal_support_final.get('edge_class_counts', {}).get(
                'top_relaxed', 0)),
        top_any_angle_relaxation_disclosed=True,
        strict_configured_angle_only_screen_pass=bool(
            internal_support_final.get(
                'strict_configured_angle_screen_pass', False)),
        graph_screen_pass=bool(
            dual_pass and internal_pass and abnormal_pass and surface_pass
            and pressure_pass and stl_pass),
    )
    checks['status'] = (
        'graph_screen_pass_additional_validation_pending'
        if checks['graph_screen_pass'] else 'graph_screen_fail')
    return checks


def main():
    global INITIAL_FACE_COUNT
    args = parse_cli_args()
    apply_cli_config(args)
    build_started = time.perf_counter()
    stage_started = build_started
    stage_rows: List[Dict] = []

    def finish_stage(stage: str):
        nonlocal stage_started
        stage_ended = time.perf_counter()
        stage_rows.append({
            "stage": stage,
            "start_seconds": stage_started - build_started,
            "end_seconds": stage_ended - build_started,
            "duration_seconds": stage_ended - stage_started,
        })
        stage_started = stage_ended

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for legacy_name in (
            "post_assembly_pressure_trim_profile.csv",
            "post_assembly_pressure_trim_status.json",
            "post_assembly_pressure_trim_remaining.csv",
            "assembly_nodes_pretrim.csv",
            "assembly_edges_pretrim.csv",
    ):
        (OUT_DIR / legacy_name).unlink(missing_ok=True)
    for legacy_name in (
            "dual_before_after_trim.json",
            "before_trim/dual_stitch_final_audit.json",
    ):
        (OUT_DIR / "dual_validation" / legacy_name).unlink(missing_ok=True)
    (OUT_DIR / "internal_validation" / "support_before_after_assembly_trim.json").unlink(missing_ok=True)
    export_run_config(OUT_DIR)
    if SURFACE_DESIGN_ENABLE and not args.internal_only:
        surface_design = _surface_capillary_design()
        log(
            "Surface capillary criterion: "
            f"P_hyd={surface_design.hydrostatic_pressure_pa:.3f} Pa, "
            f"P_required={surface_design.required_pressure_pa:.3f} Pa, "
            f"r_crit={surface_design.critical_equivalent_radius_mm:.6f} mm; "
            f"using {R_EFF_LIMIT_MM:.6f} mm [{SURFACE_RADIUS_SOURCE}]"
        )
    else:
        log("Surface design disabled: skipping surface graph construction and dual-domain stitching.")
    log(f"Input STL: {IN_STL}")
    log(f"Output dir: {OUT_DIR}")

    mesh, mm_per_unit = load_and_scale_mesh(IN_STL, TARGET_HEIGHT_MM)
    z_bounds = (mesh.bounds[0, 2], mesh.bounds[1, 2])
    finish_stage("Input loading and scaling")

    if args.surface_face_scan:
        if not SURFACE_DESIGN_ENABLE:
            raise ValueError("--surface-face-scan requires surface design to be enabled")
        selected_faces = run_surface_face_scan(mesh)
        finish_stage("Surface face-count seam scan")
        if selected_faces is None:
            export_build_stage_timing(stage_rows, OUT_DIR)
            log("Surface face scan did not find a provisional pass; stopping before internal construction.")
            return
        INITIAL_FACE_COUNT = int(selected_faces)
        # Re-export after automatic selection so the run contract records the
        # actual target face count used by the production surface build.
        export_run_config(OUT_DIR)

    if args.internal_only:
        atoms_int, edges_int, _ = build_internal_core_structure(mesh, start_id=0)
        finish_stage("Adaptive internal construction")
        internal_only_profile = run_global_physics_outputs(mesh, atoms_int, edges_int, z_bounds)
        finish_stage("Physics analysis and export")
        export_build_stage_timing(stage_rows, OUT_DIR)
        export_build_key_metrics(stage_rows, OUT_DIR, edges_int, internal_only_profile,
                                 source='internal_only_final_full_physics_profile')
        log("All Done!")
        return

    surf_sys = None
    atoms_surf = pd.DataFrame(columns=["atom_id", "x", "y", "z", "face_name"])
    edges_surf = pd.DataFrame(columns=["source", "target", "radius", "bond_type"])
    if SURFACE_DESIGN_ENABLE:
        surf_sys, atoms_surf, edges_surf, next_id = build_surface_structure(mesh)
        finish_stage("Surface network construction")
        if args.surface_only:
            surface_only_audit = surf_sys.audit_final_retained_graph(
                atoms_surf, edges_surf, OUT_DIR / "surface_validation",
                node_spheres_exported=False,
                include_stitch_anchors=False,
                apply_quality_gate=True,
            )
            _log_surface_seam_audit(surface_only_audit, prefix="Surface-only final audit")
            surface_only_global_audit = run_v11_global_surface_pore_audit()
            surface_only_status = {
                'surface_only': True,
                'internal_construction_skipped': True,
                'dual_domain_stitching_skipped': True,
                'provisional_surface_acceptance_pass': bool(
                    surface_only_audit.get('provisional_surface_acceptance_pass', False)),
                'local_chart_acceptance_pass': bool(
                    surface_only_audit.get('local_chart_acceptance_pass', False)),
                'seam_screen_pass': bool(surface_only_audit.get('seam_screen_pass', False)),
                'seam_dependent_local_record_count': int(
                    surface_only_audit.get('seam_dependent_pore_count', 0)),
                'seam_dependent_local_record_fraction': float(
                    surface_only_audit.get('seam_dependent_local_record_fraction', 0.0)),
                'seam_radius_exceeded_local_record_count': int(
                    surface_only_audit.get('seam_radius_exceeded_local_record_count', 0)),
                'nonseam_radius_exceeded_pore_count': int(
                    surface_only_audit.get('nonseam_radius_exceeded_pore_count', 0)),
                'global_complete_pore_count_available': True,
                'global_complete_pore_count': int(
                    surface_only_global_audit.get('global_complete_pore_count', 0)),
                'global_complete_pore_radius_acceptance_pass': bool(
                    surface_only_global_audit.get('global_complete_pore_radius_acceptance_pass', False)),
                'global_complete_pore_aspect_ratio_acceptance_pass': bool(
                    surface_only_global_audit.get('global_complete_pore_aspect_ratio_acceptance_pass', False)),
                'global_complete_pore_minimum_area_acceptance_pass': bool(
                    surface_only_global_audit.get('global_complete_pore_minimum_area_acceptance_pass', False)),
                'global_complete_pore_acceptance_pass': bool(
                    surface_only_global_audit.get('global_complete_pore_acceptance_pass', False)),
                'surface_only_contour_acceptance_pass': bool(
                    surface_only_audit.get('local_chart_acceptance_pass', False)
                    and surface_only_global_audit.get('global_complete_pore_acceptance_pass', False)
                    and int(surface_only_audit.get('missing_edge_endpoints', 0)) == 0),
                'interpretation': (
                    'Final surface acceptance combines local retained-graph integrity with a '
                    'cross-patch global complete-pore audit. Artificial seam-local pore fragments '
                    'remain diagnostic once the reconstructed global pore result is available.'),
            }
            (OUT_DIR / 'surface_only_acceptance.json').write_text(
                json.dumps(surface_only_status, ensure_ascii=False, indent=2), encoding='utf-8')
            finish_stage("Surface-only pore audit")
            export_build_stage_timing(stage_rows, OUT_DIR)
            log("Surface-only workflow finished; internal construction was skipped.")
            return
    else:
        next_id = 0
        log("  Surface construction bypassed by SURFACE_DESIGN_ENABLE=False; internal build starts at atom_id=0.")

    atoms_int, edges_int = build_internal_structure_v2(
        mesh,
        next_id,
        1.0,
        atoms_surf=(atoms_surf if SURFACE_DESIGN_ENABLE else None),
        edges_surf=(edges_surf if SURFACE_DESIGN_ENABLE else None),
        output_dir=OUT_DIR
    )
    finish_stage("Adaptive internal construction")
    next_id = int(max(
        int(atoms_surf["atom_id"].max()) if not atoms_surf.empty else -1,
        int(atoms_int["atom_id"].max()) if not atoms_int.empty else -1,
    )) + 1

    atoms_int.to_csv(OUT_DIR / "internal_nodes_before_pressure_trim.csv", index=False)
    _dedup_edges_local(edges_int).to_csv(
        OUT_DIR / "internal_edges_before_pressure_trim.csv", index=False)
    support_before_internal_trim = _internal_support_state(
        atoms_int, edges_int, z_bounds=z_bounds)
    atoms_int, edges_int, internal_trim_profile, trimmed_nodes, trimmed_edges = (
        trim_final_internal_pressure(mesh, atoms_int, edges_int, z_bounds)
    )
    internal_trim_profile.to_csv(
        OUT_DIR / "internal_pressure_trim_profile.csv", index=False)
    trim_status = dict(internal_trim_profile.attrs.get("internal_pressure_trim", {}))
    trim_status["stage"] = "internal_after_bottom_pillars_before_dual_stitch"
    (OUT_DIR / "internal_pressure_trim_status.json").write_text(
        json.dumps(trim_status, ensure_ascii=False, indent=2), encoding="utf-8")
    if internal_trim_profile.empty:
        remaining_trim_issues = internal_trim_profile.copy()
    else:
        trim_z_col = ("z_mid_mm" if "z_mid_mm" in internal_trim_profile.columns
                      else "z_mm")
        remaining_trim_issues = internal_trim_profile[
            (pd.to_numeric(internal_trim_profile[trim_z_col], errors="coerce")
             >= float(INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM))
            & (internal_trim_profile["P_cap_Pa"].astype(float)
               > float(INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO)
               * internal_trim_profile["P_required_Pa"].astype(float))
            ].copy()
        if not remaining_trim_issues.empty:
            remaining_trim_issues["capillary_ratio"] = (
                    remaining_trim_issues["P_cap_Pa"].astype(float)
                    / np.maximum(remaining_trim_issues["P_required_Pa"].astype(float), 1e-12)
            )
    remaining_trim_issues.to_csv(
        OUT_DIR / "internal_pressure_trim_remaining.csv", index=False)
    support_after_internal_trim = _internal_support_state(
        atoms_int, edges_int, z_bounds=z_bounds)
    internal_support_trim_comparison = {
        "before_internal_trim": {
            "node_count": len(support_before_internal_trim["nodes"]),
            "downward_supported_count": len(support_before_internal_trim["supported"]),
            "bottom_root_connected_count": len(support_before_internal_trim["root_connected"]),
        },
        "after_internal_trim": {
            "node_count": len(support_after_internal_trim["nodes"]),
            "downward_supported_count": len(support_after_internal_trim["supported"]),
            "bottom_root_connected_count": len(support_after_internal_trim["root_connected"]),
        },
        "previously_supported_retained_nodes_lost_support": len(
            (support_before_internal_trim["supported"] & support_after_internal_trim["nodes"])
            - support_after_internal_trim["supported"]),
        "previously_root_connected_retained_nodes_disconnected": len(
            (support_before_internal_trim["root_connected"] & support_after_internal_trim["nodes"])
            - support_after_internal_trim["root_connected"]),
    }
    (OUT_DIR / "internal_validation").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "internal_validation" / "support_before_after_internal_trim.json").write_text(
        json.dumps(internal_support_trim_comparison, indent=2, ensure_ascii=False),
        encoding="utf-8")
    log(f"  InternalPressureTrim (before stitch): status={trim_status.get('status')}, "
        f"removed_nodes={trimmed_nodes}, removed_edges={trimmed_edges}.")
    finish_stage("Final internal pressure trim")

    export_stl(atoms_int, edges_int, OUT_DIR / "internal_only.stl")
    finish_stage("Final internal export")

    if not SURFACE_DESIGN_ENABLE:

        internal_only_profile = run_global_physics_outputs(
            mesh, atoms_int, edges_int, z_bounds)
        finish_stage("Physics analysis and export")
        internal_support_final = export_internal_support_audit(
            atoms_int, edges_int, OUT_DIR / 'internal_validation', z_bounds=z_bounds)
        abnormal_layer_audit = audit_abnormal_layer_connectivity(
            atoms_int, edges_int, z_bounds=z_bounds,
            output_dir=OUT_DIR / 'internal_validation')
        internal_stl_path = OUT_DIR / 'internal_only.stl'
        final_stl_audit = audit_final_stl_file(
            internal_stl_path, mesh, atoms_int, edges_int)
        pressure_screen = summarize_final_pressure_screen(internal_only_profile, z_bounds)
        assembly_checks = evaluate_internal_only_acceptance(
            internal_support_final=internal_support_final,
            abnormal_layer_audit=abnormal_layer_audit,
            pressure_screen=pressure_screen,
            final_stl_audit=final_stl_audit,
            trim_status=trim_status,
        )
        (OUT_DIR / 'final_assembly_status.json').write_text(
            json.dumps(assembly_checks, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT_DIR / 'final_validation_summary.json').write_text(
            json.dumps({
                'surface_design_enabled': False,
                'surface_and_dual_domain_checks': 'not_applicable',
                'internal_print_support': internal_support_final,
                'abnormal_layer_connectivity': abnormal_layer_audit,
                'fine_slice_pressure_screen': pressure_screen,
                'internal_pressure_trim': trim_status,
                'exported_stl': final_stl_audit,
                'assembly_status': assembly_checks,
            }, ensure_ascii=False, indent=2), encoding='utf-8')
        finish_stage("Final graph and STL geometry audit")
        export_build_stage_timing(stage_rows, OUT_DIR)
        export_build_key_metrics(
            stage_rows, OUT_DIR, edges_int, internal_only_profile,
            source='surface_disabled_final_internal_full_physics_profile')
        if assembly_checks['graph_screen_pass']:
            log("Surface design is OFF; abnormal-layer connectivity and final pressure acceptance passed.")
        else:
            log("Surface design is OFF; completed with GRAPH_SCREEN_FAIL. See final_assembly_status.json.")
        log("All Done!")
        return

    surface_anchor_area_guard = surf_sys.make_stitch_anchor_area_guard(edges_surf)
    edges_stitch = stitch_surface_to_core_optimized(
        atoms_surf,
        atoms_int,
        r_stitch=STITCH_DISTANCE_MM,
        anchor_spacing=STITCH_ANCHOR_SPACING_MM,
        existing_edges=edges_int,
        surface_edges=edges_surf,
        mesh=mesh,
        z_bounds=z_bounds,
        max_internal_degree=INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW,
        max_surface_degree=8,
        relaxed_distance_mm=STITCH_RELAXED_DISTANCE_MM,
        surface_anchor_area_guard=surface_anchor_area_guard,
        audit_dir=OUT_DIR / 'dual_validation',
    )

    atoms_all = pd.concat([atoms_surf, atoms_int], ignore_index=True)
    edges_all = pd.concat([edges_surf, edges_int, edges_stitch], ignore_index=True)
    finish_stage("Surface-to-core stitching")

    atoms_all.to_csv(OUT_DIR / "assembly_nodes_final.csv", index=False)
    _dedup_edges_local(edges_all).to_csv(
        OUT_DIR / "assembly_edges_final.csv", index=False)

    assembled_pressure_profile = _pressure_profile_stl(
        mesh, atoms_all, _edge_records(edges_all), z_bounds,
        INTERNAL_DYNAMIC_PHYSICS_DZ_MM, cache={},
    )
    assembled_pressure_profile.to_csv(
        OUT_DIR / "post_assembly_pressure_profile.csv", index=False)
    assembled_margins = _capillary_margin_values(
        assembled_pressure_profile, include_assembly_reserve=False)
    assembled_preassembly_reserve_margins = _capillary_margin_values(
        assembled_pressure_profile, include_assembly_reserve=True)
    assembly_pressure_status = {
        "stage": "read_only_after_dual_stitch",
        "pressure_profile_pass": bool(_capillary_profile_passes(
            assembled_pressure_profile, include_assembly_reserve=False)),
        "minimum_margin_pa": (float(np.min(assembled_margins))
                              if len(assembled_margins) else None),
        "minimum_preassembly_reserve_margin_pa": (
            float(np.min(assembled_preassembly_reserve_margins))
            if len(assembled_preassembly_reserve_margins) else None),
        "final_acceptance_lower_ratio": float(INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO),
        "preassembly_lower_ratio": float(
            INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO
            + INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO),
        "graph_mutated_after_stitch": False,
        "internal_trim_status": str(trim_status.get("status", "unknown")),
    }
    (OUT_DIR / "post_assembly_pressure_status.json").write_text(
        json.dumps(assembly_pressure_status, ensure_ascii=False, indent=2),
        encoding="utf-8")
    log("  Assembled pressure (read-only): "
        f"pass={assembly_pressure_status['pressure_profile_pass']}, "
        f"min_margin={assembly_pressure_status['minimum_margin_pa']} Pa; "
        "no post-stitch deletion performed.")
    finish_stage("Read-only assembled pressure verification")
    internal_support_final = export_internal_support_audit(
        atoms_all, edges_all, OUT_DIR / 'internal_validation', z_bounds=z_bounds
    )
    abnormal_layer_audit = audit_abnormal_layer_connectivity(
        atoms_all, edges_all, z_bounds=z_bounds,
        output_dir=OUT_DIR / 'internal_validation')
    dual_final_audit = audit_final_dual_domain(
        atoms_all, edges_all, z_bounds=z_bounds,
        strict_length_mm=STITCH_DISTANCE_MM,
        relaxed_length_mm=STITCH_RELAXED_DISTANCE_MM,
        output_dir=OUT_DIR / 'dual_validation')
    log('  Final internal support audit: '
        f"unsupported={internal_support_final['downward_unsupported_node_count']}, "
        f"unanchored={internal_support_final['root_disconnected_node_count']}, "
        f"rejected-angle/long-span={internal_support_final['edge_class_counts']['rejected']}")
    final_stl_path = OUT_DIR / "final_stitched.stl"
    export_stl(atoms_all, edges_all, final_stl_path)
    finish_stage("Final mesh export")

    final_surface_audit = surf_sys.audit_final_retained_graph(
        atoms_all, edges_all, OUT_DIR / "surface_validation",
        node_spheres_exported=False,
        include_stitch_anchors=True,
    )
    final_global_surface_audit = run_v11_global_surface_pore_audit()
    final_stl_audit = audit_final_stl_file(
        final_stl_path, mesh, atoms_all, edges_all,
    )
    (OUT_DIR / "final_stl_geometry_audit.json").write_text(
        json.dumps(final_stl_audit, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (OUT_DIR / "final_validation_summary.json").write_text(
        json.dumps({"surface_graph": final_surface_audit,
                    "surface_global_complete_pores": final_global_surface_audit,
                    "internal_print_support": internal_support_final,
                    "abnormal_layer_connectivity": abnormal_layer_audit,
                    "dual_domain": dual_final_audit,
                    "internal_pressure_trim": trim_status,
                    "internal_support_before_after_trim": internal_support_trim_comparison,
                    "assembled_pressure_read_only": assembly_pressure_status,
                    "exported_stl": final_stl_audit},
                   indent=2, ensure_ascii=False), encoding="utf-8",
    )
    log("Final surface graph audit: "
        f"valid={final_surface_audit['valid_pore_count']}, "
        f"blocked={final_surface_audit['blocked_pore_count']}, "
        f"undersized={final_surface_audit['undersized_pore_count']}, "
        f"radius_exceeded={final_surface_audit['radius_exceeded_pore_count']}; "
        f"elapsed={final_surface_audit['elapsed_seconds']:.2f}s")
    _log_surface_seam_audit(final_surface_audit, prefix="Final assembled surface seam audit")
    finish_stage("Final graph and STL geometry audit")

    analyze_surface_physics_detailed(atoms_all, edges_all, OUT_DIR)
    finish_stage("Surface physics diagnostics")

    nodes_np = _make_nodes_np_for_analysis(atoms_all, z_bounds)

    log("Starting full physics slice analysis...")
    zs = _pressure_feedback_heights(z_bounds)
    df_res = run_full_analysis(mesh, atoms_all, edges_all, nodes_np, zs)
    fine_pressure_screen = {
        'slice_count': int(len(df_res)),
        'expected_slice_count': int(len(zs)),
        'ordinary_requirement_fail_count': 0,
        'lower_bound_fail_count': 0,
        'minimum_ordinary_margin_pa': None,
        'minimum_lower_bound_margin_pa': None,
        'preassembly_reserve_fail_count': 0,
        'minimum_preassembly_reserve_margin_pa': None,
        'minimum_capillary_viscous_mobility_Pa_m3': None,
        'mobility_bottleneck_z_mm': None,
        'maximum_cumulative_geometry_resistance_m-3': None,
        'capillary_stall_count': 0,
        'resistance_calibration_factor': float(FLOW_RESISTANCE_CALIBRATION_FACTOR),
        'reference_flow_rate_m3_s': float(VISC_REF_Q_M3_S),
        'pressure_slice_nominal_thickness_mm': float(SLICE_DZ_MM),
        'ordinary_fixed_flow_transport_reach_mm': None,
        'design_reserve_transport_reach_mm': None,
        'pass': False,
    }
    if not df_res.empty:
        ordinary_margin = (
            df_res['transport_margin_Pa'].to_numpy(dtype=float)
            if 'transport_margin_Pa' in df_res.columns
            else df_res['P_cap_Pa'].to_numpy(dtype=float)
                 - df_res.get('P_required_base_Pa', df_res['P_required_Pa']).to_numpy(dtype=float)
        )
        lower_margin = _capillary_margin_values(
            df_res, include_assembly_reserve=False)
        preassembly_reserve_margin = _capillary_margin_values(
            df_res, include_assembly_reserve=True)
        mobility = _viscous_mobility_values(df_res)
        mobility_idx = int(np.argmin(mobility))
        z_col = 'z_mid_mm' if 'z_mid_mm' in df_res.columns else 'z_mm'
        stall_count = int(np.count_nonzero(
            df_res.get('capillary_stall_proxy', pd.Series(False, index=df_res.index)).to_numpy(dtype=bool)))
        fine_pressure_screen.update({
            'ordinary_requirement_fail_count': int(np.sum(ordinary_margin < 0.0)),
            'lower_bound_fail_count': int(np.sum(lower_margin < 0.0)),
            'minimum_ordinary_margin_pa': float(np.min(ordinary_margin)),
            'minimum_lower_bound_margin_pa': float(np.min(lower_margin)),
            'preassembly_reserve_fail_count': int(np.sum(preassembly_reserve_margin < 0.0)),
            'minimum_preassembly_reserve_margin_pa': float(np.min(preassembly_reserve_margin)),
            'minimum_capillary_viscous_mobility_Pa_m3': float(mobility[mobility_idx]),
            'mobility_bottleneck_z_mm': float(df_res.iloc[mobility_idx][z_col]),
            'maximum_cumulative_geometry_resistance_m-3': float(
                df_res['R_geom_cumulative_m-3'].max()
            ),
            'capillary_stall_count': stall_count,
            'ordinary_fixed_flow_transport_reach_mm':
                _transport_reach_height_mm(df_res, use_design_reserve=False),
            'design_reserve_transport_reach_mm':
                _transport_reach_height_mm(df_res, use_design_reserve=True),
            'pass': bool(
                len(df_res) == len(zs)
                and np.allclose(np.sort(df_res[z_col].to_numpy(dtype=float)),
                                np.sort(zs), rtol=0.0, atol=1.0e-8)
                and np.all(np.isfinite(lower_margin))
                and np.all(np.isfinite(ordinary_margin))
                and np.all(np.isfinite(mobility))
                and np.all(lower_margin >= 0.0)
                and np.all(ordinary_margin >= 0.0)
            ),
        })
    finish_stage("Global slice physics analysis")

    if not df_res.empty:
        df_sorted = df_res.sort_values("z_mm")

        z_arr = df_sorted["z_mm"].values * 1e-3
        A_liq_arr_real = (df_sorted["A_domain_mm2"] * df_sorted["porosity"]).values * 1e-6
        dz_arr_m = (df_sorted['slice_dz_mm'].to_numpy(dtype=float)
                    if 'slice_dz_mm' in df_sorted.columns
                    else np.full(len(df_sorted), float(SLICE_DZ_MM), dtype=float)) * 1e-3
        dE = RHO_LIQ * G_GRAV * z_arr * A_liq_arr_real * dz_arr_m
        total_energy = np.sum(dE)
        log(f"Total liquid gravitational potential energy: {total_energy:.6f} Joules")

        csv_path = OUT_DIR / "internal_slice_physics_analysis.csv"
        df_res.to_csv(csv_path, index=False)
        export_dynamic_analysis_outputs(mesh, atoms_all, edges_all, nodes_np, df_res, OUT_DIR)
        log("Physics charts generated.")

    finish_stage("Analysis tables and figure export")
    export_build_stage_timing(stage_rows, OUT_DIR)
    export_build_key_metrics(stage_rows, OUT_DIR, edges_all, df_res,
                             source='final_fine_slice_pressure_profile')

    assembly_checks = evaluate_unified_acceptance(
        dual_final_audit=dual_final_audit,
        internal_support_final=internal_support_final,
        final_surface_audit=final_surface_audit,
        final_global_surface_audit=final_global_surface_audit,
        assembly_pressure_status=assembly_pressure_status,
        fine_pressure_screen=fine_pressure_screen,
        final_stl_audit=final_stl_audit,
        trim_status=trim_status,
        abnormal_layer_audit=abnormal_layer_audit,
    )
    (OUT_DIR / 'final_assembly_status.json').write_text(
        json.dumps(assembly_checks, ensure_ascii=False, indent=2), encoding='utf-8')
    existing_summary = json.loads((OUT_DIR / 'final_validation_summary.json').read_text(encoding='utf-8'))
    existing_summary['assembly_status'] = assembly_checks
    existing_summary['fine_slice_pressure_screen'] = fine_pressure_screen
    (OUT_DIR / 'final_validation_summary.json').write_text(
        json.dumps(existing_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if assembly_checks['graph_screen_pass']:
        log("Build completed: dual graph, capillary-viscous mobility, "
            "print-support, surface geometry and STL-format checks passed.")
    else:
        log("Completed with GRAPH_SCREEN_FAIL: final_assembly_status.json lists unmet checks; "
            "the STL is a diagnostic output, not a validated successful specimen.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        raise SystemExit(
            "Select an explicit case: python Code/Analysis/generate_case.py "
            "--case {water25,water35,organic22p5} --out NEW_DIRECTORY")
    try:
        main()
    except WorkflowCheckpointStop as exc:
        log(f"Checkpoint workflow completed: {exc}")
