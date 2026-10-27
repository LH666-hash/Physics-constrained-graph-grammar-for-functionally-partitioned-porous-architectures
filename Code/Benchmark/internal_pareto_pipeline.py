#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified 25-mm internal H_rise–porosity Pareto workflow.

Both periodic candidates and PC-AEGG references are evaluated with the same
state-updated viscous-profile function imported from the current PC-AEGG
implementation.

Revision: vertical sampling is deliberately finer than the periodic unit-cell
scale to avoid z-direction phase locking / aliasing. The true 25-mm top boundary
is explicitly evaluated. A finer z-grid is also used in the numerical
resolution-stability audit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist, directed_hausdorff
from scipy.stats import qmc


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

FAMILY_COLORS = {
    "Graded BCC": "#2F6FA3",
    "Graded Gyroid": "#7656A5",
    "Simple cubic": "#D98200",
    "Diamond": "#16856A",
    "Schwarz P": "#A45C2A",
}
UNIFORM_BENCHMARKS = {"Simple cubic", "Diamond", "Schwarz P"}
SHEET_TPMS_FAMILIES = {"Graded Gyroid", "Diamond", "Schwarz P"}
PC_COLOR = "#C44E52"
POOLED_COLOR = "#222222"


@dataclass(frozen=True)
class FluidConfig:
    surface_tension_n_m: float = 0.072
    contact_angle_deg: float = 15.0
    density_difference_kg_m3: float = 1000.0
    gravity_m_s2: float = 9.80665
    viscosity_pa_s: float = 1.0e-3
    reference_flow_m3_s: float = 1.0e-7


@dataclass(frozen=True)
class DomainConfig:
    width_mm: float = 10.0
    depth_mm: float = 10.0
    height_mm: float = 25.0

    # Main z-resolution requested for this run.
    # NOTE: 0.75 mm is substantially finer than the former 1.0-mm spacing,
    # but it can still alias with some periodic lengths. A 0.05-mm repeat
    # remains available for resolution verification (see CLI arguments below).
    slice_spacing_mm: float = 0.075

    xy_pitch_mm: float = 0.125
    min_manufacturable_feature_mm: float = 0.40


@dataclass(frozen=True)
class SearchBounds:
    label: str
    L_min_mm: float
    L_max_mm: float
    p_min: float
    p_max: float
    feature_to_L_max: float


INITIAL_BOUNDS = SearchBounds("initial", 1.0, 3.0, 0.5, 2.0, 1.0)
MAIN_BOUNDS = SearchBounds("expanded_main", 0.4, 10.0, 0.0625, 16.0, 2.0)


PROFILE_COLUMNS = [
    "family",
    "design_id",
    "search_stage",
    "z_mm",
    "A_pore_mm2",
    "C_wet_mm",
    "r_eq_mm",
    "porosity_z",
    "axial_radius4_sum_mm4",
    "n_flow_components",
    "P_cap_Pa",
    "P_hyd_Pa",
    "P_visc_Pa",
    "R_tr_Pa",
]


def load_module(path: Path, name: str):
    """Load an explicit source module without embedding a private source path."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProfileBatchWriter:
    def __init__(self, directory: Path, rows_per_part: int = 100_000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows_per_part = int(rows_per_part)
        self.buffer: list[pd.DataFrame] = []
        self.buffer_rows = 0
        self.part = 0

    def add(self, frame: pd.DataFrame) -> None:
        self.buffer.append(frame.loc[:, PROFILE_COLUMNS].copy())
        self.buffer_rows += len(frame)
        if self.buffer_rows >= self.rows_per_part:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.part += 1
        target = self.directory / f"internal_pareto_profiles_part_{self.part:03d}.csv.gz"
        pd.concat(self.buffer, ignore_index=True).to_csv(
            target, index=False, compression={"method": "gzip", "compresslevel": 6}
        )
        self.buffer.clear()
        self.buffer_rows = 0

    def close(self) -> None:
        self.flush()


def retain_profile_designs(directory: Path, allowed_design_ids: set[str]) -> int:
    """Rewrite profile batches so every profile has one exported candidate row."""
    directory = Path(directory)
    temporary = directory.with_name(directory.name + "_filtered")
    temporary.mkdir(parents=True, exist_ok=True)
    filtered_writer = ProfileBatchWriter(temporary)
    retained = 0
    for part in sorted(directory.glob("internal_pareto_profiles_part_*.csv.gz")):
        frame = pd.read_csv(part)
        frame = frame[frame["design_id"].isin(allowed_design_ids)]
        retained += len(frame)
        if len(frame):
            filtered_writer.add(frame)
    filtered_writer.close()
    for part in directory.glob("internal_pareto_profiles_part_*.csv.gz"):
        part.unlink()
    for part in sorted(temporary.glob("internal_pareto_profiles_part_*.csv.gz")):
        part.replace(directory / part.name)
    temporary.rmdir()
    return retained


class SliceGrid:
    def __init__(self, domain: DomainConfig):
        nx = int(round(domain.width_mm / domain.xy_pitch_mm))
        ny = int(round(domain.depth_mm / domain.xy_pitch_mm))
        if not math.isclose(nx * domain.xy_pitch_mm, domain.width_mm, abs_tol=1.0e-9):
            raise ValueError("width_mm must be divisible by xy_pitch_mm")
        if not math.isclose(ny * domain.xy_pitch_mm, domain.depth_mm, abs_tol=1.0e-9):
            raise ValueError("depth_mm must be divisible by xy_pitch_mm")
        self.domain = domain
        self.nx = nx
        self.ny = ny
        self.x = (np.arange(nx, dtype=float) + 0.5) * domain.xy_pitch_mm
        self.y = (np.arange(ny, dtype=float) + 0.5) * domain.xy_pitch_mm
        # IMPORTANT:
        # Use regular dz sampling through the full design height and always
        # append the true z = H top boundary. This allows spacings such as
        # 0.75 mm, for which H/dz is not an integer.
        dz = float(domain.slice_spacing_mm)
        if dz <= 0.0:
            raise ValueError("slice_spacing_mm must be positive")

        z_regular = np.arange(
            dz,
            domain.height_mm,
            dz,
            dtype=float,
        )

        # Round away accumulated floating-point noise.
        z_regular = np.round(z_regular, 12)

        # Explicitly include the true physical top boundary.
        if len(z_regular) == 0 or not math.isclose(
            float(z_regular[-1]),
            float(domain.height_mm),
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            self.z = np.concatenate(
                [z_regular, np.asarray([float(domain.height_mm)], dtype=float)]
            )
        else:
            self.z = z_regular
            self.z[-1] = float(domain.height_mm)

        # Guard against accidental duplicate points and enforce strict ordering.
        self.z = np.unique(np.round(self.z, 12))
        if len(self.z) == 0 or not math.isclose(
            float(self.z[-1]),
            float(domain.height_mm),
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise ValueError("The true top boundary z=height_mm was not included")
        if np.any(np.diff(self.z) <= 0.0):
            raise ValueError("z sampling points must be strictly increasing")

        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")


def gradient_feature(z_mm: np.ndarray, bottom: float, top: float, exponent: float, height: float) -> np.ndarray:
    fraction = np.clip(z_mm / height, 0.0, 1.0) ** exponent
    return bottom + (top - bottom) * fraction


def bcc_solid(grid: SliceGrid, L_mm: float, bottom_mm: float, top_mm: float, p: float) -> np.ndarray:
    """Rasterize the periodic BCC strut union over the full fine-z grid."""
    directions = np.asarray(
        [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    px = np.mod(grid.xx, L_mm) - 0.5 * L_mm
    py = np.mod(grid.yy, L_mm) - 0.5 * L_mm
    diameters = gradient_feature(grid.z, bottom_mm, top_mm, p, grid.domain.height_mm)
    slices = []
    for z_mm, diameter in zip(grid.z, diameters):
        pz = (z_mm % L_mm) - 0.5 * L_mm
        minimum_distance_sq = np.full_like(px, np.inf)
        for direction in directions:
            projection = px * direction[0] + py * direction[1] + pz * direction[2]
            distance_sq = px * px + py * py + pz * pz - projection * projection
            minimum_distance_sq = np.minimum(minimum_distance_sq, np.maximum(distance_sq, 0.0))
        slices.append(minimum_distance_sq <= (0.5 * diameter) ** 2)
    return np.asarray(slices, dtype=bool)


def simple_cubic_solid(
    grid: SliceGrid, L_mm: float, bottom_mm: float, top_mm: float, p: float
) -> np.ndarray:
    """Rasterize an orthogonal simple-cubic strut network."""
    px = np.abs(np.mod(grid.xx + 0.5 * L_mm, L_mm) - 0.5 * L_mm)
    py = np.abs(np.mod(grid.yy + 0.5 * L_mm, L_mm) - 0.5 * L_mm)
    diameters = gradient_feature(grid.z, bottom_mm, top_mm, p, grid.domain.height_mm)
    slices = []
    for z_mm, diameter in zip(grid.z, diameters):
        pz = abs((z_mm + 0.5 * L_mm) % L_mm - 0.5 * L_mm)
        distance = np.minimum.reduce((np.hypot(py, pz), np.hypot(px, pz), np.hypot(px, py)))
        slices.append(distance <= 0.5 * diameter)
    return np.asarray(slices, dtype=bool)


def tpms_sheet_solid(
    grid: SliceGrid,
    family: str,
    L_mm: float,
    bottom_mm: float,
    top_mm: float,
    p: float,
) -> np.ndarray:
    """Rasterize a physical-thickness TPMS sheet from its zero surface."""
    if family not in SHEET_TPMS_FAMILIES:
        raise ValueError(f"Unknown TPMS family: {family}")
    k = 2.0 * math.pi / L_mm
    x = k * grid.xx
    y = k * grid.yy
    sin_x, cos_x = np.sin(x), np.cos(x)
    sin_y, cos_y = np.sin(y), np.cos(y)
    thicknesses = gradient_feature(grid.z, bottom_mm, top_mm, p, grid.domain.height_mm)
    slices = []
    for z_mm, thickness in zip(grid.z, thicknesses):
        z = k * z_mm
        sin_z, cos_z = math.sin(z), math.cos(z)
        if family == "Graded Gyroid":
            field = sin_x * cos_y + sin_y * cos_z + sin_z * cos_x
            dfdx = k * (cos_x * cos_y - sin_z * sin_x)
            dfdy = k * (-sin_x * sin_y + cos_y * cos_z)
            dfdz = k * (-sin_y * sin_z + cos_z * cos_x)
        elif family == "Diamond":
            field = (
                sin_x * sin_y * sin_z
                + sin_x * cos_y * cos_z
                + cos_x * sin_y * cos_z
                + cos_x * cos_y * sin_z
            )
            dfdx = k * (
                cos_x * sin_y * sin_z
                + cos_x * cos_y * cos_z
                - sin_x * sin_y * cos_z
                - sin_x * cos_y * sin_z
            )
            dfdy = k * (
                sin_x * cos_y * sin_z
                - sin_x * sin_y * cos_z
                + cos_x * cos_y * cos_z
                - cos_x * sin_y * sin_z
            )
            dfdz = k * (
                sin_x * sin_y * cos_z
                - sin_x * cos_y * sin_z
                - cos_x * sin_y * sin_z
                + cos_x * cos_y * cos_z
            )
        else:  # Schwarz P
            field = cos_x + cos_y + cos_z
            dfdx = -k * sin_x
            dfdy = -k * sin_y
            dfdz = -k * sin_z
        signed_distance = np.abs(field) / np.maximum(np.sqrt(dfdx**2 + dfdy**2 + dfdz**2), 1.0e-9)
        slices.append(signed_distance <= 0.5 * thickness)
    return np.asarray(slices, dtype=bool)


def gyroid_solid(grid: SliceGrid, L_mm: float, bottom_mm: float, top_mm: float, p: float) -> np.ndarray:
    """Rasterize a graded sheet-Gyroid using a gradient-corrected distance field."""
    return tpms_sheet_solid(grid, "Graded Gyroid", L_mm, bottom_mm, top_mm, p)


def _component_geometry(liquid: np.ndarray, pitch_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Return per-slice sum(r_eq^4) and 2-D liquid component count."""
    plane_connectivity = np.zeros((3, 3, 3), dtype=np.uint8)
    plane_connectivity[1, 1, 1] = 1
    plane_connectivity[1, 0, 1] = 1
    plane_connectivity[1, 2, 1] = 1
    plane_connectivity[1, 1, 0] = 1
    plane_connectivity[1, 1, 2] = 1
    labels, count = ndimage.label(liquid, structure=plane_connectivity)
    if count == 0:
        return np.zeros(liquid.shape[0]), np.zeros(liquid.shape[0], dtype=int)

    flat = labels.ravel()
    pixels = np.bincount(flat, minlength=count + 1).astype(float)
    perimeter_edges = np.zeros(count + 1, dtype=float)

    for left, right in ((labels[:, 1:, :], labels[:, :-1, :]), (labels[:, :, 1:], labels[:, :, :-1])):
        difference = left != right
        a = left[difference]
        b = right[difference]
        perimeter_edges += np.bincount(a[a > 0], minlength=count + 1)
        perimeter_edges += np.bincount(b[b > 0], minlength=count + 1)

    for boundary in (labels[:, 0, :], labels[:, -1, :], labels[:, :, 0], labels[:, :, -1]):
        values = boundary[boundary > 0]
        perimeter_edges += np.bincount(values, minlength=count + 1)

    area_mm2 = pixels * pitch_mm**2
    perimeter_mm = perimeter_edges * pitch_mm
    radii = np.divide(
        2.0 * area_mm2,
        perimeter_mm,
        out=np.zeros_like(area_mm2),
        where=perimeter_mm > 0.0,
    )

    label_z = np.full(count + 1, -1, dtype=int)
    for z_index in range(labels.shape[0]):
        present = np.unique(labels[z_index])
        present = present[present > 0]
        label_z[present] = z_index
    radius4 = np.zeros(labels.shape[0], dtype=float)
    n_components = np.zeros(labels.shape[0], dtype=int)
    for label_id in range(1, count + 1):
        z_index = label_z[label_id]
        if z_index >= 0 and radii[label_id] > 0.0:
            radius4[z_index] += radii[label_id] ** 4
            n_components[z_index] += 1
    return radius4, n_components


def _spans_height(binary: np.ndarray) -> bool:
    six = ndimage.generate_binary_structure(3, 1)
    labels, _ = ndimage.label(binary, structure=six)
    bottom = set(np.unique(labels[0])) - {0}
    top = set(np.unique(labels[-1])) - {0}
    return bool(bottom.intersection(top))



def continuous_rise_height_from_margin(
    z_mm: np.ndarray,
    margin_pa: np.ndarray,
    domain_height_mm: float,
    *,
    interpolate_zero_crossing: bool = True,
) -> tuple[float, float]:
    """Return continuous rise height and the first failing sampled z.

    The liquid must satisfy the transport criterion continuously from the
    bottom. Once the first negative margin is encountered, later positive
    slices are ignored.

    If a positive-to-negative transition is bracketed by two sampled heights,
    a linear zero-crossing interpolation is used. With the default fine
    z-resolution this only refines the reported crossing within one dz and
    does not replace the underlying physics model.
    """
    z = np.asarray(z_mm, dtype=float)
    margin = np.asarray(margin_pa, dtype=float)

    if z.ndim != 1 or margin.ndim != 1 or len(z) != len(margin):
        raise ValueError("z_mm and margin_pa must be one-dimensional arrays of equal length")
    if len(z) == 0:
        raise ValueError("z_mm cannot be empty")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("z_mm must be strictly increasing")

    failures = np.flatnonzero(margin < 0.0)
    if not len(failures):
        return float(domain_height_mm), math.nan

    i = int(failures[0])
    first_failure_z = float(z[i])

    if i == 0:
        return 0.0, first_failure_z

    z0 = float(z[i - 1])
    z1 = float(z[i])
    m0 = float(margin[i - 1])
    m1 = float(margin[i])

    if (
        interpolate_zero_crossing
        and np.isfinite(m0)
        and np.isfinite(m1)
        and m0 >= 0.0
        and m1 < 0.0
        and not math.isclose(m0, m1)
    ):
        crossing = z0 + (0.0 - m0) * (z1 - z0) / (m1 - m0)
        crossing = float(np.clip(crossing, z0, z1))
        return crossing, first_failure_z

    return z0, first_failure_z


def profile_from_solid(
    solid: np.ndarray,
    grid: SliceGrid,
    fluid: FluidConfig,
    hydraulic_function: Callable,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    if solid.shape != (len(grid.z), grid.nx, grid.ny):
        raise ValueError(f"Unexpected solid shape: {solid.shape}")
    liquid = ~solid
    pitch = grid.domain.xy_pitch_mm
    area = liquid.sum(axis=(1, 2)).astype(float) * pitch**2
    porosity = area / (grid.domain.width_mm * grid.domain.depth_mm)
    interface_edges = (
        np.count_nonzero(solid[:, 1:, :] != solid[:, :-1, :], axis=(1, 2))
        + np.count_nonzero(solid[:, :, 1:] != solid[:, :, :-1], axis=(1, 2))
    )
    wetted_perimeter = interface_edges.astype(float) * pitch
    r_eq = np.divide(
        2.0 * area,
        wetted_perimeter,
        out=np.zeros_like(area),
        where=wetted_perimeter > 0.0,
    )
    radius4, component_count = _component_geometry(liquid, pitch)
    cos_theta = math.cos(math.radians(fluid.contact_angle_deg))
    p_cap = np.divide(
        2.0 * fluid.surface_tension_n_m * cos_theta,
        r_eq * 1.0e-3,
        out=np.zeros_like(r_eq),
        where=r_eq > 0.0,
    )
    p_hyd = fluid.density_difference_kg_m3 * fluid.gravity_m_s2 * grid.z * 1.0e-3
    raw = pd.DataFrame(
        {
            "z_mm": grid.z,
            "A_pore_mm2": area,
            "A_liq_mm2": area,
            "C_wet_mm": wetted_perimeter,
            "r_eq_mm": r_eq,
            "porosity_z": porosity,
            "axial_radius4_sum_mm4": radius4,
            "n_flow_components": component_count,
            "P_cap_Pa": p_cap,
            "P_hydro_Pa": p_hyd,
        }
    )
    evaluated = hydraulic_function(
        raw,
        viscosity_pa_s=fluid.viscosity_pa_s,
        reference_flow_m3_s=fluid.reference_flow_m3_s,
        default_dz_mm=grid.domain.slice_spacing_mm,
        target_fill_time_s=0.0,
        safety_column="__no_safety_factor__",
    )
    evaluated["P_hyd_Pa"] = evaluated["P_hydro_Pa"]
    evaluated["P_visc_Pa"] = evaluated["P_visc_dynamic_Pa"]
    evaluated["R_tr_Pa"] = evaluated["P_cap_Pa"] - evaluated["P_hyd_Pa"] - evaluated["P_visc_Pa"]

    predicted_height, first_failure_z_mm = continuous_rise_height_from_margin(
        grid.z,
        evaluated["R_tr_Pa"].to_numpy(float),
        grid.domain.height_mm,
        interpolate_zero_crossing=True,
    )

    geometry_valid = bool(
        np.all(np.isfinite(evaluated[["A_pore_mm2", "C_wet_mm", "r_eq_mm", "R_tr_Pa"]]))
        and np.all(area > 0.0)
        and np.all(wetted_perimeter > 0.0)
        and np.all(radius4 > 0.0)
        and np.any(solid)
    )
    liquid_connected = _spans_height(liquid) if geometry_valid else False
    adjacent_solid_overlap = bool(np.all(np.any(solid[:-1] & solid[1:], axis=(1, 2)))) if geometry_valid else False
    diagnostics: dict[str, float | bool] = {
        "predicted_rise_height_mm": predicted_height,
        "first_failure_z_mm": first_failure_z_mm,
        "porosity": float(np.mean(porosity)),
        "min_slice_porosity": float(np.min(porosity)),
        "max_slice_porosity": float(np.max(porosity)),
        "min_R_tr_Pa": float(np.min(evaluated["R_tr_Pa"])),
        "geometry_valid": geometry_valid,
        "liquid_connected": liquid_connected,
        "solid_layer_overlap_valid": adjacent_solid_overlap,
    }
    return evaluated, diagnostics


def decode_unit(unit: np.ndarray, bounds: SearchBounds, domain: DomainConfig) -> dict[str, float]:
    unit = np.clip(np.asarray(unit, dtype=float), 0.0, 1.0)
    L_mm = bounds.L_min_mm + (bounds.L_max_mm - bounds.L_min_mm) * unit[0]
    upper = bounds.feature_to_L_max * L_mm
    lower = domain.min_manufacturable_feature_mm
    bottom = lower + (upper - lower) * unit[1]
    top = lower + (upper - lower) * unit[2]
    exponent = math.exp(math.log(bounds.p_min) + unit[3] * math.log(bounds.p_max / bounds.p_min))
    return {"L_mm": L_mm, "feature_bottom_mm": bottom, "feature_top_mm": top, "p": exponent}


def encode_p(value: float, bounds: SearchBounds) -> float:
    return math.log(value / bounds.p_min) / math.log(bounds.p_max / bounds.p_min)


def canonical_unit(family: str, unit: np.ndarray, bounds: SearchBounds) -> np.ndarray:
    """Remove inactive gradient dimensions from uniform benchmark families."""
    canonical = np.clip(np.asarray(unit, dtype=float), 0.0, 1.0).copy()
    if family in UNIFORM_BENCHMARKS:
        canonical[2] = canonical[1]
        canonical[3] = np.clip(encode_p(1.0, bounds), 0.0, 1.0)
    return canonical


def evaluate_design(
    family: str,
    unit: np.ndarray,
    design_id: str,
    search_stage: str,
    bounds: SearchBounds,
    grid: SliceGrid,
    fluid: FluidConfig,
    hydraulic_function: Callable,
    writer: ProfileBatchWriter | None,
    sample_index: int | None = None,
    is_uniform: bool = False,
) -> dict[str, object]:
    unit = canonical_unit(family, unit, bounds)
    params = decode_unit(unit, bounds, grid.domain)
    if family == "Graded BCC":
        solid = bcc_solid(grid, params["L_mm"], params["feature_bottom_mm"], params["feature_top_mm"], params["p"])
    elif family == "Graded Gyroid":
        solid = gyroid_solid(grid, params["L_mm"], params["feature_bottom_mm"], params["feature_top_mm"], params["p"])
    elif family == "Simple cubic":
        solid = simple_cubic_solid(grid, params["L_mm"], params["feature_bottom_mm"], params["feature_top_mm"], params["p"])
    elif family in {"Diamond", "Schwarz P"}:
        solid = tpms_sheet_solid(
            grid, family, params["L_mm"], params["feature_bottom_mm"], params["feature_top_mm"], params["p"]
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    profile, metrics = profile_from_solid(solid, grid, fluid, hydraulic_function)
    manufacturing_valid = bool(
        min(params["feature_bottom_mm"], params["feature_top_mm"])
        >= grid.domain.min_manufacturable_feature_mm - 1.0e-12
    )
    feasible = bool(
        metrics["geometry_valid"]
        and metrics["liquid_connected"]
        and metrics["solid_layer_overlap_valid"]
        and manufacturing_valid
    )
    profile.insert(0, "search_stage", search_stage)
    profile.insert(0, "design_id", design_id)
    profile.insert(0, "family", family)
    if writer is not None:
        writer.add(profile)
    row: dict[str, object] = {
        "family": family,
        "design_id": design_id,
        "search_stage": search_stage,
        "bounds_label": bounds.label,
        "sample_index": sample_index,
        "is_uniform": bool(is_uniform or family in UNIFORM_BENCHMARKS),
        "u_L": float(unit[0]),
        "u_bottom": float(unit[1]),
        "u_top": float(unit[2]),
        "u_p": float(unit[3]),
        "L": params["L_mm"],
        "L_mm": params["L_mm"],
        "D_bottom_mm": params["feature_bottom_mm"] if family in {"Graded BCC", "Simple cubic"} else math.nan,
        "D_top_mm": params["feature_top_mm"] if family in {"Graded BCC", "Simple cubic"} else math.nan,
        "t_bottom_mm": params["feature_bottom_mm"] if family in SHEET_TPMS_FAMILIES else math.nan,
        "t_top_mm": params["feature_top_mm"] if family in SHEET_TPMS_FAMILIES else math.nan,
        "feature_bottom_mm": params["feature_bottom_mm"],
        "feature_top_mm": params["feature_top_mm"],
        "feature_bottom_to_L": params["feature_bottom_mm"] / params["L_mm"],
        "feature_top_to_L": params["feature_top_mm"] / params["L_mm"],
        "p": params["p"],
        "z_samples_per_period": params["L_mm"] / grid.domain.slice_spacing_mm,
        "z_resolution_warning_lt8_per_period": bool(
            params["L_mm"] / grid.domain.slice_spacing_mm < 8.0
        ),
        "epsilon": metrics["porosity"],
        "porosity": metrics["porosity"],
        "predicted_rise_height_mm": metrics["predicted_rise_height_mm"],
        "first_failure_z_mm": metrics["first_failure_z_mm"],
        "geometry_valid": bool(metrics["geometry_valid"]),
        "manufacturing_valid": manufacturing_valid,
        "liquid_connected": bool(metrics["liquid_connected"]),
        "solid_layer_overlap_valid": bool(metrics["solid_layer_overlap_valid"]),
        "feasible": feasible,
        "min_slice_porosity": metrics["min_slice_porosity"],
        "max_slice_porosity": metrics["max_slice_porosity"],
        "min_R_tr_Pa": metrics["min_R_tr_Pa"],
    }
    return row


def sobol_units(samples: int, seed: int, bounds: SearchBounds) -> tuple[np.ndarray, np.ndarray]:
    if samples <= 0 or samples & (samples - 1):
        raise ValueError("Sobol sample count must be a positive power of two")
    units = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(int(math.log2(samples)))
    uniform = np.zeros(samples, dtype=bool)
    uniform[::16] = True
    units[uniform, 2] = units[uniform, 1]
    units[uniform, 3] = np.clip(encode_p(1.0, bounds), 0.0, 1.0)
    return units, uniform


def run_sobol_search(
    bounds: SearchBounds,
    samples: int,
    seed: int,
    grid: SliceGrid,
    fluid: FluidConfig,
    hydraulic_function: Callable,
    writer: ProfileBatchWriter | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(FAMILY_COLORS):
        units, uniform = sobol_units(samples, seed + 100 * family_index, bounds)
        if family in UNIFORM_BENCHMARKS:
            units = np.asarray([canonical_unit(family, unit, bounds) for unit in units])
            uniform[:] = True
        for index, unit in enumerate(units):
            design_id = f"{bounds.label}-{family.replace(' ', '_')}-S{index + 1:05d}"
            rows.append(
                evaluate_design(
                    family,
                    unit,
                    design_id,
                    "sobol",
                    bounds,
                    grid,
                    fluid,
                    hydraulic_function,
                    writer,
                    sample_index=index + 1,
                    is_uniform=bool(uniform[index]),
                )
            )
    return pd.DataFrame(rows)


def strict_pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame[["epsilon", "predicted_rise_height_mm"]].to_numpy(float)
    order = np.lexsort((-values[:, 1], -values[:, 0]))
    keep = np.zeros(len(values), dtype=bool)
    best_h = -np.inf
    for index in order:
        h = values[index, 1]
        if h > best_h:
            keep[index] = True
            best_h = h
    return keep


def tolerance_pareto_mask(frame: pd.DataFrame, delta_epsilon: float, delta_h_mm: float) -> np.ndarray:
    eps = frame["epsilon"].to_numpy(float)
    height = frame["predicted_rise_height_mm"].to_numpy(float)
    order = np.argsort(eps)
    eps_sorted = eps[order]
    height_sorted = height[order]
    suffix_max = np.maximum.accumulate(height_sorted[::-1])[::-1]

    keep = np.ones(len(frame), dtype=bool)
    for i in range(len(frame)):
        threshold_h = np.searchsorted(eps_sorted, eps[i] - delta_epsilon, side="left")
        dominated_by_h = threshold_h < len(frame) and suffix_max[threshold_h] > height[i] + delta_h_mm
        threshold_e = np.searchsorted(eps_sorted, eps[i] + delta_epsilon, side="right")
        dominated_by_e = threshold_e < len(frame) and suffix_max[threshold_e] >= height[i] - delta_h_mm
        keep[i] = not (dominated_by_h or dominated_by_e)
    return keep


def front(frame: pd.DataFrame, tolerance: tuple[float, float] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = (
        strict_pareto_mask(frame)
        if tolerance is None
        else tolerance_pareto_mask(frame, tolerance[0], tolerance[1])
    )
    return frame.loc[mask].sort_values(["epsilon", "predicted_rise_height_mm"]).copy()


def normalized_front_points(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    return np.column_stack(
        [frame["epsilon"].to_numpy(float), frame["predicted_rise_height_mm"].to_numpy(float) / 25.0]
    )


def normalized_hausdorff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    pa, pb = normalized_front_points(a), normalized_front_points(b)
    if len(pa) == 0 or len(pb) == 0:
        return math.nan
    return float(max(directed_hausdorff(pa, pb)[0], directed_hausdorff(pb, pa)[0]))


def hypervolume_fraction(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    points = normalized_front_points(front(frame))
    points = points[np.argsort(points[:, 0])]
    area = 0.0
    previous_x = 0.0
    for x, y in points:
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        area += max(0.0, x - previous_x) * y
        previous_x = max(previous_x, x)
    return float(area)


def attainment_epsilon_shift(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float, float]:
    """Compare best attained porosity at every 1-mm rise-height threshold."""
    thresholds = np.arange(0.0, 25.0 + 1.0e-9, 1.0)

    def envelope(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                frame.loc[frame["predicted_rise_height_mm"] >= height, "epsilon"].max()
                for height in thresholds
            ],
            dtype=float,
        )

    differences = np.abs(envelope(a) - envelope(b))
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return math.nan, math.nan, math.nan
    return (
        float(np.mean(differences)),
        float(np.quantile(differences, 0.95)),
        float(np.max(differences)),
    )


def add_front_flags(candidates: pd.DataFrame, delta_epsilon: float, delta_h_mm: float) -> pd.DataFrame:
    result = candidates.copy()
    for name in ("family_strict_front", "family_tolerance_front", "pooled_strict_front", "pooled_tolerance_front"):
        result[name] = False
    feasible = result["feasible"].astype(bool)
    for family in FAMILY_COLORS:
        subset = result[feasible & (result["family"] == family)]
        if subset.empty:
            continue
        result.loc[subset.index[strict_pareto_mask(subset)], "family_strict_front"] = True
        result.loc[
            subset.index[tolerance_pareto_mask(subset, delta_epsilon, delta_h_mm)],
            "family_tolerance_front",
        ] = True
    pooled = result[feasible]
    if not pooled.empty:
        result.loc[pooled.index[strict_pareto_mask(pooled)], "pooled_strict_front"] = True
        result.loc[
            pooled.index[tolerance_pareto_mask(pooled, delta_epsilon, delta_h_mm)],
            "pooled_tolerance_front",
        ] = True
    return result


def convergence_table(candidates: pd.DataFrame, sample_levels: Iterable[int]) -> pd.DataFrame:
    rows = []
    previous: dict[str, pd.DataFrame] = {}
    for sample_count in sample_levels:
        subset = candidates[
            (candidates["search_stage"] == "sobol")
            & (candidates["sample_index"] <= sample_count)
            & candidates["feasible"]
        ]
        for scope in (*FAMILY_COLORS.keys(), "Pooled periodic"):
            scoped = subset if scope == "Pooled periodic" else subset[subset["family"] == scope]
            current = front(scoped)
            key = scope
            previous_front = previous.get(key)
            hv = hypervolume_fraction(current)
            attainment = (
                (math.nan, math.nan, math.nan)
                if previous_front is None
                else attainment_epsilon_shift(previous_front, current)
            )
            rows.append(
                {
                    "scope": scope,
                    "N_per_family": int(sample_count),
                    "feasible_candidates": int(len(scoped)),
                    "strict_front_points": int(len(current)),
                    "hypervolume_fraction": hv,
                    "hypervolume_relative_change_from_previous": (
                        math.nan
                        if previous_front is None
                        else (hv - hypervolume_fraction(previous_front)) / max(hv, 1.0e-12)
                    ),
                    "normalized_Hausdorff_from_previous": (
                        math.nan if previous_front is None else normalized_hausdorff(previous_front, current)
                    ),
                    "attainment_mean_abs_epsilon_shift_from_previous": attainment[0],
                    "attainment_p95_abs_epsilon_shift_from_previous": attainment[1],
                    "attainment_max_abs_epsilon_shift_from_previous": attainment[2],
                }
            )
            previous[key] = current
    return pd.DataFrame(rows)


def boundary_occupancy(frame: pd.DataFrame, bounds: SearchBounds, tolerance: tuple[float, float]) -> pd.DataFrame:
    tol_front = front(frame[frame["feasible"]], tolerance=tolerance)
    rows = []
    if tol_front.empty:
        return pd.DataFrame(rows)
    span_L = bounds.L_max_mm - bounds.L_min_mm
    for family in FAMILY_COLORS:
        subset = tol_front[tol_front["family"] == family]
        if subset.empty:
            continue
        tests = {
            "L_lower": subset["L_mm"] <= bounds.L_min_mm + 0.02 * span_L,
            "L_upper": subset["L_mm"] >= bounds.L_max_mm - 0.02 * span_L,
            "feature_bottom_manufacturing_lower": subset["feature_bottom_mm"] <= 0.40 + 0.02,
            "feature_bottom_search_upper": subset["feature_bottom_to_L"] >= bounds.feature_to_L_max - 0.02,
        }
        if family not in UNIFORM_BENCHMARKS:
            tests.update(
                {
                    "feature_top_manufacturing_lower": subset["feature_top_mm"] <= 0.40 + 0.02,
                    "feature_top_search_upper": subset["feature_top_to_L"] >= bounds.feature_to_L_max - 0.02,
                    "p_lower": subset["p"] <= bounds.p_min * 1.05,
                    "p_upper": subset["p"] >= bounds.p_max / 1.05,
                }
            )
        for boundary, hit in tests.items():
            if "manufacturing_lower" in boundary:
                boundary_class = "manufacturing_minimum"
            elif boundary == "L_upper" and bounds.L_max_mm >= 10.0 - 1.0e-12:
                boundary_class = "design_domain_period_limit"
            elif boundary == "L_lower" and bounds.L_min_mm <= 0.40 + 1.0e-12:
                boundary_class = "manufacturing_scale_limit"
            elif boundary in {"p_lower", "p_upper"} and bounds.p_min <= 0.0625 and bounds.p_max >= 16.0:
                boundary_class = "slice_resolution_saturation"
            else:
                boundary_class = "search_boundary"
            rows.append(
                {
                    "bounds_label": bounds.label,
                    "family": family,
                    "boundary": boundary,
                    "front_points": int(len(subset)),
                    "count_at_boundary": int(hit.sum()),
                    "fraction_at_boundary": float(hit.mean()),
                    "boundary_class": boundary_class,
                    "hard_or_saturated_boundary": boundary_class != "search_boundary",
                }
            )
    return pd.DataFrame(rows)


def load_pc_aegg_masks(
    run_dir: Path,
    grid: SliceGrid,
    geometry_module,
) -> np.ndarray:
    atoms = pd.read_csv(Path(run_dir) / "structure_nodes.csv")
    edges = pd.read_csv(Path(run_dir) / "structure_edges.csv")
    nodes_np = geometry_module._make_nodes_np_for_analysis(atoms, (0.0, grid.domain.height_mm))
    solid_slices = []
    try:
        from shapely import contains_xy
    except ImportError as exc:
        raise RuntimeError("Shapely >=2 is required for PC-AEGG rasterization") from exc
    for z_mm in grid.z:
        geometry = geometry_module._generate_true_solid_geometry_at_z_full(nodes_np, edges, atoms, float(z_mm))
        solid_slices.append(contains_xy(geometry, grid.xx, grid.yy))
    return np.asarray(solid_slices, dtype=bool)


def evaluate_pc_references(
    run_dirs: list[Path],
    grid: SliceGrid,
    fluid: FluidConfig,
    hydraulic_function: Callable,
    geometry_module,
    writer: ProfileBatchWriter,
) -> pd.DataFrame:
    rows = []
    for index, run_dir in enumerate(run_dirs, start=1):
        design_id = f"PC-AEGG-{index:02d}"
        solid = load_pc_aegg_masks(run_dir, grid, geometry_module)
        profile, metrics = profile_from_solid(solid, grid, fluid, hydraulic_function)
        profile.insert(0, "search_stage", "pc_reference")
        profile.insert(0, "design_id", design_id)
        profile.insert(0, "family", "PC-AEGG")
        writer.add(profile)
        rows.append(
            {
                "family": "PC-AEGG",
                "design_id": design_id,
                "replicate": index,
                "epsilon": metrics["porosity"],
                "porosity": metrics["porosity"],
                "predicted_rise_height_mm": metrics["predicted_rise_height_mm"],
                "first_failure_z_mm": metrics["first_failure_z_mm"],
                "min_R_tr_Pa": metrics["min_R_tr_Pa"],
                "geometry_valid": metrics["geometry_valid"],
                "liquid_connected": metrics["liquid_connected"],
                "solid_layer_overlap_valid": metrics["solid_layer_overlap_valid"],
            }
        )
    return pd.DataFrame(rows)


def stability_tolerances(
    candidates: pd.DataFrame,
    bounds: SearchBounds,
    reference_pitch_mm: float,
    reference_z_pitch_mm: float,
    domain: DomainConfig,
    fluid: FluidConfig,
    hydraulic_function: Callable,
    sample_size: int,
) -> tuple[float, float, pd.DataFrame]:
    feasible = candidates[candidates["feasible"]].copy()
    strict = front(feasible)
    remainder = feasible.drop(index=strict.index)
    selected = strict.copy()
    if len(selected) > sample_size // 2:
        selected = selected.sort_values(["family", "epsilon"]).iloc[
            np.linspace(0, len(selected) - 1, sample_size // 2, dtype=int)
        ]
    remaining_n = max(0, sample_size - len(selected))
    if remaining_n and len(remainder):
        distances = cdist(normalized_front_points(remainder), normalized_front_points(strict))
        nearest = np.min(distances, axis=1)
        near_index = remainder.index[np.argsort(nearest)[:remaining_n]]
        selected = pd.concat(
            [selected, remainder.loc[near_index]],
            ignore_index=False,
        )
    fine_domain = DomainConfig(
        **{
            **asdict(domain),
            "xy_pitch_mm": reference_pitch_mm,
            "slice_spacing_mm": reference_z_pitch_mm,
        }
    )
    fine_grid = SliceGrid(fine_domain)
    rows = []
    for row in selected.itertuples(index=False):
        unit = np.array([row.u_L, row.u_bottom, row.u_top, row.u_p], dtype=float)
        fine = evaluate_design(
            row.family,
            unit,
            f"stability-{row.design_id}",
            "resolution_stability",
            bounds,
            fine_grid,
            fluid,
            hydraulic_function,
            writer=None,
        )
        rows.append(
            {
                "family": row.family,
                "design_id": row.design_id,
                "coarse_xy_pitch_mm": domain.xy_pitch_mm,
                "fine_xy_pitch_mm": reference_pitch_mm,
                "coarse_z_pitch_mm": domain.slice_spacing_mm,
                "fine_z_pitch_mm": reference_z_pitch_mm,
                "coarse_epsilon": row.epsilon,
                "fine_epsilon": fine["epsilon"],
                "abs_delta_epsilon": abs(float(row.epsilon) - float(fine["epsilon"])),
                "coarse_height_mm": row.predicted_rise_height_mm,
                "fine_height_mm": fine["predicted_rise_height_mm"],
                "abs_delta_height_mm": abs(
                    float(row.predicted_rise_height_mm) - float(fine["predicted_rise_height_mm"])
                ),
            }
        )
    stability = pd.DataFrame(rows)
    delta_epsilon = max(0.005, float(stability["abs_delta_epsilon"].quantile(0.95)))
    delta_h_mm = max(domain.slice_spacing_mm, float(stability["abs_delta_height_mm"].quantile(0.95)))
    delta_epsilon = float(math.ceil(delta_epsilon * 1000.0) / 1000.0)
    delta_h_mm = float(math.ceil(delta_h_mm / domain.slice_spacing_mm) * domain.slice_spacing_mm)
    return delta_epsilon, delta_h_mm, stability


def run_refinement(
    sobol_candidates: pd.DataFrame,
    bounds: SearchBounds,
    grid: SliceGrid,
    fluid: FluidConfig,
    hydraulic_function: Callable,
    writer: ProfileBatchWriter,
    delta_epsilon: float,
    seed: int,
) -> tuple[pd.DataFrame, list[float]]:
    feasible = sobol_candidates[sobol_candidates["feasible"]].copy()
    sobol_front = front(feasible)
    lower = float(sobol_front["epsilon"].min())
    upper = float(sobol_front["epsilon"].max())
    targets = np.linspace(lower, upper, 5)
    eps_tol = max(0.01, delta_epsilon)
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(FAMILY_COLORS):
        family_data = feasible[feasible["family"] == family]
        seeds = front(family_data)
        for target_index, target in enumerate(targets):
            closest = seeds.iloc[(seeds["epsilon"] - target).abs().argsort()[: min(8, len(seeds))]]
            if closest.empty:
                closest = family_data.iloc[(family_data["epsilon"] - target).abs().argsort()[:8]]
            seed_units = closest[["u_L", "u_bottom", "u_top", "u_p"]].to_numpy(float)
            rng = np.random.default_rng(seed + family_index * 1000 + target_index)
            population = np.vstack([seed_units, np.clip(seed_units + rng.normal(0.0, 0.06, seed_units.shape), 0.0, 1.0)])
            while len(population) < 20:
                base = seed_units[rng.integers(0, len(seed_units))]
                population = np.vstack([population, np.clip(base + rng.normal(0.0, 0.08, 4), 0.0, 1.0)])
            population = population[: max(20, len(seed_units))]
            cache: dict[tuple[float, ...], dict[str, object]] = {}

            def objective(unit: np.ndarray) -> float:
                key = tuple(np.round(canonical_unit(family, unit, bounds), 7))
                if key not in cache:
                    design_id = (
                        f"refined-{family.replace(' ', '_')}-E{target:.3f}-"
                        f"{len(cache) + 1:04d}"
                    )
                    cache[key] = evaluate_design(
                        family,
                        np.asarray(key),
                        design_id,
                        "epsilon_constraint_refined",
                        bounds,
                        grid,
                        fluid,
                        hydraulic_function,
                        writer,
                    )
                    cache[key]["epsilon_target"] = float(target)
                    cache[key]["epsilon_constraint_tolerance"] = eps_tol
                result = cache[key]
                if not result["feasible"]:
                    return 1000.0
                deviation = abs(float(result["epsilon"]) - float(target))
                penalty = max(0.0, deviation - eps_tol) / eps_tol
                return -float(result["predicted_rise_height_mm"]) / grid.domain.height_mm + 10.0 * penalty + 0.01 * deviation / eps_tol

            differential_evolution(
                objective,
                bounds=[(0.0, 1.0)] * 4,
                seed=seed + family_index * 100 + target_index,
                maxiter=8,
                popsize=5,
                tol=0.01,
                polish=False,
                updating="immediate",
                workers=1,
                init=population,
            )
            rows.extend(cache.values())
    refined = pd.DataFrame(rows).drop_duplicates(subset=["family", "u_L", "u_bottom", "u_top", "u_p"])
    return refined, [float(x) for x in targets]


def build_front_export(candidates: pd.DataFrame, delta_epsilon: float, delta_h_mm: float) -> pd.DataFrame:
    records = []
    for stage_name, stage in (
        ("sobol_only", candidates[candidates["search_stage"] == "sobol"]),
        ("refined", candidates[candidates["search_stage"].isin(["sobol", "epsilon_constraint_refined"])]),
    ):
        feasible = stage[stage["feasible"]]
        for scope in (*FAMILY_COLORS.keys(), "Pooled periodic"):
            scoped = feasible if scope == "Pooled periodic" else feasible[feasible["family"] == scope]
            for kind, selected in (
                ("strict", front(scoped)),
                ("tolerance_aware", front(scoped, tolerance=(delta_epsilon, delta_h_mm))),
            ):
                if selected.empty:
                    continue
                block = selected.copy()
                block.insert(0, "front_type", kind)
                block.insert(0, "front_scope", scope)
                block.insert(0, "front_stage", stage_name)
                records.append(block)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def periodic_dominates_pc(
    periodic: pd.DataFrame, pc: pd.DataFrame, delta_epsilon: float, delta_h_mm: float
) -> pd.DataFrame:
    rows = []
    feasible = periodic[periodic["feasible"]]
    targets = pd.concat(
        [
            pc[["design_id", "epsilon", "predicted_rise_height_mm"]],
            pd.DataFrame(
                [
                    {
                        "design_id": "PC-AEGG-mean",
                        "epsilon": float(pc["epsilon"].mean()),
                        "predicted_rise_height_mm": float(pc["predicted_rise_height_mm"].mean()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    for pc_row in targets.itertuples(index=False):
        eps = feasible["epsilon"].to_numpy(float)
        height = feasible["predicted_rise_height_mm"].to_numpy(float)
        no_worse = (eps >= pc_row.epsilon - delta_epsilon) & (
            height >= pc_row.predicted_rise_height_mm - delta_h_mm
        )
        meaningful = (eps > pc_row.epsilon + delta_epsilon) | (
            height > pc_row.predicted_rise_height_mm + delta_h_mm
        )
        dominated = feasible[no_worse & meaningful]
        rows.append(
            {
                "pc_design_id": pc_row.design_id,
                "pc_epsilon": pc_row.epsilon,
                "pc_height_mm": pc_row.predicted_rise_height_mm,
                "periodic_dominator_count": int(len(dominated)),
                "is_tolerance_dominated": bool(len(dominated)),
                "best_dominator_design_id": (
                    dominated.sort_values(["predicted_rise_height_mm", "epsilon"], ascending=False).iloc[0]["design_id"]
                    if len(dominated)
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def export_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_main(candidates: pd.DataFrame, pc: pd.DataFrame, delta_epsilon: float, delta_h_mm: float, output: Path) -> None:
    feasible = candidates[candidates["feasible"]]
    fig, ax = plt.subplots(figsize=(3.50, 2.85), constrained_layout=True)
    ax.axvspan(0.58, 0.65, color="#D9D9D9", alpha=0.45, lw=0, zorder=0)
    for family, color in FAMILY_COLORS.items():
        subset = feasible[feasible["family"] == family]
        ax.scatter(
            subset["epsilon"], subset["predicted_rise_height_mm"], s=6, alpha=0.28,
            color=color, linewidths=0, rasterized=True, label=family,
        )
        family_front = front(subset)
        ax.plot(
            family_front["epsilon"], family_front["predicted_rise_height_mm"],
            color=color, lw=1.0, alpha=0.9, label=f"{family} front (guide)",
        )
    pooled_tolerance = front(feasible, tolerance=(delta_epsilon, delta_h_mm))
    pooled_strict = front(feasible)
    ax.scatter(
        pooled_tolerance["epsilon"], pooled_tolerance["predicted_rise_height_mm"],
        s=16, facecolor=POOLED_COLOR, edgecolor="white", linewidth=0.25, zorder=5,
        label="Periodic nondominated envelope",
    )
    ax.plot(
        pooled_strict["epsilon"], pooled_strict["predicted_rise_height_mm"],
        color=POOLED_COLOR, lw=0.8, zorder=4,
    )
    pc_mean = pc[["epsilon", "predicted_rise_height_mm"]].mean()
    pc_sd = pc[["epsilon", "predicted_rise_height_mm"]].std(ddof=1).fillna(0.0)
    ax.scatter(
        pc["epsilon"], pc["predicted_rise_height_mm"],
        s=12, marker="o", facecolor=PC_COLOR, edgecolor="white", linewidth=0.35,
        alpha=0.48, zorder=7, label="PC-AEGG replicates (n=5)",
    )
    ax.errorbar(
        pc_mean["epsilon"], pc_mean["predicted_rise_height_mm"],
        xerr=pc_sd["epsilon"], yerr=pc_sd["predicted_rise_height_mm"],
        fmt="D", ms=5.2, mfc=PC_COLOR, mec="white", mew=0.7, ecolor=PC_COLOR,
        elinewidth=0.8, capsize=2, zorder=8, label="PC-AEGG mean ± s.d.",
    )
    ax.axhline(25.0, color="#777777", lw=0.65, ls=(0, (2, 2)), zorder=1)
    ax.text(
        0.99, 0.965, "25-mm design-domain ceiling", transform=ax.transAxes,
        ha="right", va="top", fontsize=5.5, color="#666666",
    )
    ax.set_xlabel("Porosity, ε")
    ax.set_ylabel("Predicted rise height, Hrise (mm)")
    ax.set_ylim(-0.5, 25.7)
    x_values = pd.concat([feasible["epsilon"], pc["epsilon"]])
    ax.set_xlim(max(0.0, float(x_values.min()) - 0.03), min(1.0, float(x_values.max()) + 0.03))
    handles, labels = ax.get_legend_handles_labels()
    family_handle_indices = list(range(0, 2 * len(FAMILY_COLORS), 2))
    order = family_handle_indices + [2 * len(FAMILY_COLORS), 2 * len(FAMILY_COLORS) + 1, 2 * len(FAMILY_COLORS) + 2]
    ax.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="lower left", fontsize=5.2, handlelength=1.5, handletextpad=0.45,
        columnspacing=0.8, ncol=2,
    )
    export_figure(fig, output / "Figure_2e_internal_pareto")


def plot_convergence(candidates: pd.DataFrame, convergence: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    pooled = candidates[(candidates["search_stage"] == "sobol") & candidates["feasible"]]
    for n, color in zip((1024, 2048, 4096, 8192), ("#C9D7E5", "#91AEC9", "#5F86AA", "#234F78")):
        f = front(pooled[pooled["sample_index"] <= n])
        axes[0].plot(f["epsilon"], f["predicted_rise_height_mm"], color=color, lw=1.0, label=f"N={n}")
    axes[0].set(xlabel="Porosity, ε", ylabel="Hrise (mm)")
    axes[0].legend(fontsize=6)
    pooled_conv = convergence[convergence["scope"] == "Pooled periodic"]
    axes[1].plot(
        pooled_conv["N_per_family"], pooled_conv["hypervolume_fraction"],
        marker="o", ms=3.5, color="#234F78", label="Hypervolume",
    )
    axes[1].set(xlabel="Sobol samples per family", ylabel="Normalized hypervolume")
    twin = axes[1].twinx()
    twin.plot(
        pooled_conv["N_per_family"], pooled_conv["attainment_p95_abs_epsilon_shift_from_previous"],
        marker="s", ms=3.2, color="#C96B5A", label="Attainment shift (p95)",
    )
    twin.set_ylabel("Porosity shift at height thresholds")
    lines = axes[1].lines + twin.lines
    axes[1].legend(lines, [line.get_label() for line in lines], fontsize=6, loc="best")
    for label, ax in zip("ab", axes):
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    export_figure(fig, output / "SI_sampling_convergence")


def plot_boundary_sensitivity(
    initial: pd.DataFrame,
    expanded_main: pd.DataFrame,
    occupancy: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    for data, label, color, ls in (
        (initial[initial["feasible"]], "Initial domain", "#777777", "--"),
        (expanded_main[expanded_main["feasible"]], "Expanded main domain", "#C96B5A", "-"),
    ):
        f = front(data)
        axes[0].plot(f["epsilon"], f["predicted_rise_height_mm"], color=color, ls=ls, lw=1.2, label=label)
        axes[0].set(xlabel="Porosity, ε", ylabel="Hrise (mm)")
    axes[0].legend(fontsize=6)
    plot_data = occupancy.copy()
    pivot = plot_data.pivot_table(index="boundary", columns="bounds_label", values="fraction_at_boundary", aggfunc="mean").fillna(0.0)
    x = np.arange(len(pivot))
    width = 0.36
    for offset, label, color in (
        (-width / 2, "initial", "#777777"),
        (width / 2, "expanded_main", "#C96B5A"),
    ):
        values = pivot[label] if label in pivot else np.zeros(len(pivot))
        display = {"initial": "Initial", "expanded_main": "Expanded main"}[label]
        axes[1].bar(x + offset, values, width=width, color=color, label=display)
    short_labels = {
        "L_lower": "L\nmin.",
        "L_upper": "L\nmax.",
        "feature_bottom_manufacturing_lower": "Bottom\nmin.",
        "feature_bottom_search_upper": "Bottom\ncap",
        "feature_top_manufacturing_lower": "Top\nmin.",
        "feature_top_search_upper": "Top\ncap",
        "p_lower": "p\nmin.",
        "p_upper": "p\nmax.",
    }
    axes[1].set_xticks(x, [short_labels[value] for value in pivot.index], fontsize=5.5)
    axes[1].set_ylabel("Fraction of tolerance-front points")
    axes[1].legend(fontsize=6)
    for label, ax in zip("ab", axes):
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
    export_figure(fig, output / "SI_parameter_bound_sensitivity")


def plot_refinement(sobol: pd.DataFrame, combined: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    sobol_front = front(sobol[sobol["feasible"]])
    refined_front = front(combined[combined["feasible"]])
    ax.plot(
        sobol_front["epsilon"], sobol_front["predicted_rise_height_mm"],
        color="#777777", lw=1.0, marker="o", ms=2.5, label="Sobol only",
    )
    ax.plot(
        refined_front["epsilon"], refined_front["predicted_rise_height_mm"],
        color="#C96B5A", lw=1.1, marker="s", ms=2.5, label="After ε-constraint refinement",
    )
    ax.set(xlabel="Porosity, ε", ylabel="Hrise (mm)")
    ax.legend(fontsize=6)
    export_figure(fig, output / "SI_sobol_vs_refined_front")


def plot_front_parameters(candidates: pd.DataFrame, output: Path) -> None:
    data = candidates[candidates["pooled_tolerance_front"] & candidates["feasible"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    fields = (
        ("L_mm", "Unit-cell size, L (mm)"),
        ("feature_bottom_mm", "Bottom feature size (mm)"),
        ("feature_top_mm", "Top feature size (mm)"),
        ("p", "Gradient exponent, p"),
    )
    for ax, (field, label), panel in zip(axes.flat, fields, "abcd"):
        for family, color in FAMILY_COLORS.items():
            subset = data[data["family"] == family]
            if subset.empty:
                continue
            ax.scatter(subset["epsilon"], subset[field], s=14, alpha=0.75, color=color, label=family)
        ax.set(xlabel="Porosity, ε", ylabel=label)
        ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    export_figure(fig, output / "SI_pareto_parameter_distribution")


def write_manuscript_text(
    path: Path,
    summary: dict[str, object],
    pc: pd.DataFrame,
    dominance: pd.DataFrame,
) -> None:
    pc_eps_mean = float(pc["epsilon"].mean())
    pc_eps_sd = float(pc["epsilon"].std(ddof=1))
    pc_h_mean = float(pc["predicted_rise_height_mm"].mean())
    pc_h_sd = float(pc["predicted_rise_height_mm"].std(ddof=1))
    replicate_rows = dominance[dominance["pc_design_id"] != "PC-AEGG-mean"]
    replicate_dominated = int(replicate_rows["is_tolerance_dominated"].sum())
    mean_dominated = bool(
        dominance.loc[dominance["pc_design_id"] == "PC-AEGG-mean", "is_tolerance_dominated"].iloc[0]
    )
    if replicate_dominated == 0 and not mean_dominated:
        conclusion = "在所定义的数值容差下，没有周期候选同时支配 PC-AEGG 均值或任一独立结构。"
    else:
        conclusion = (
            f"在所定义的数值容差下，PC-AEGG 均值的支配状态为{'被支配' if mean_dominated else '未被支配'}，"
            f"且 {replicate_dominated}/5 个独立结构受到周期候选支配；因此不应使用‘无周期候选支配’的表述。"
        )
    front_status = (
        "因此，图中边界可表述为经采样密度、参数边界和局部精修检验的参数域内近似 Pareto front，而不应解释为覆盖所有可能周期结构的真实前沿。"
        if bool(summary["sampling_convergence_pass"])
        else "由于采样密度收敛门槛尚未通过，图中边界只能表述为每族 8192 点 Sobol 搜索并经局部精修得到的有限采样非支配包络，不能写成已收敛的近似 Pareto front。"
    )
    methods_front_status = (
        "所有前沿均指预定义可制造参数域内、经采样稳定性检验的近似 Pareto front。"
        if bool(summary["sampling_convergence_pass"])
        else "由于采样密度收敛门槛尚未通过，周期结构结果报告为当前参数域内的有限采样非支配包络，而不称为已收敛的 Pareto front。"
    )
    methods = f"""## Methods 草稿：周期参数空间与 Pareto 搜索

为评估周期结构在内部毛细输送与孔隙率之间的可达权衡，我们在统一的 10 × 10 × 25 mm 设计域内参数化梯度 BCC、梯度 Gyroid，以及均匀 Simple cubic、Diamond 和 Schwarz P 三类基准。BCC 采用 (L, D_bottom, D_top, p)，其中 D(z)=D_bottom+(D_top−D_bottom)(z/H)^p；Gyroid 采用 (L, t_bottom, t_top, p)，并使用相同的幂律梯度。Simple cubic 以 L 和均匀杆径 D 参数化；Diamond 与 Schwarz P 采用各自的标准隐式零曲面，并以梯度校正法将控制量换算为物理片层厚度 t。三类新增基准均令底部与顶部特征尺寸相等且 p=1；它们不作为梯度结构处理。基准参数域取 L=1.0–3.0 mm、p=0.5–2.0，最小杆径或壁厚为 0.40 mm；特征尺寸上界采用宽松的单元尺度筛选帽，最终可行性由实体连续性、液相贯通性及开放孔隙条件判定。另在扩展参数域内进行边界敏感性计算，以检查包络是否受搜索边界截断。

所有结构沿高度以 {summary['domain']['slice_spacing_mm']:.3g} mm 间距从底部向上取样，并显式检查 z=25.0 mm 顶部边界，从 10 × 10 mm 截面设计域中扣除实体得到液相区域。该细化避免了当周期尺度接近原 1 mm 切片间距时重复采样同一单元胞相位造成的相位锁定。每个高度计算液相面积 A_pore、润湿周长 C_wet 和等效毛细管半径 r_eq=2A_pore/C_wet；毛细压力、静水压力分别按 P_cap=2γcosθ/r_eq 和 P_hyd=Δρgz 计算。相邻切片的几何阻力按串联方式累计，并调用与 PC-AEGG 生成流程相同的状态更新黏性阻力函数得到 P_visc。输送余量定义为 R_tr=P_cap−P_hyd−P_visc。预测上升高度由自底部起第一次出现 R_tr<0 的位置确定，并在相邻采样点间对零交叉位置作线性插值；若直至真实顶部均满足，则记为 25.0 mm，不对结构作设计域外延拓。

全局搜索采用嵌套 Sobol 低差异序列，每个结构族依次取 N=1024、2048、4096 和 8192，并计算结构族前沿及合并周期非支配包络。使用归一化超体积、前沿间 Hausdorff 距离和各 1 mm 高度阈值下最优孔隙率的双向达到曲线偏移评估采样稳定性，并检查非支配点在 L、特征尺寸和 p 上的边界占用。随后以 Sobol 非支配点为种子，在若干孔隙率水平执行 ε-constraint 差分进化精修。考虑高度和横向几何离散误差，Pareto 判定同时报告严格支配和容差支配；本研究采用 δ_H={summary['delta_H_mm']:.3g} mm、δ_ε={summary['delta_epsilon']:.3g}。{methods_front_status}

PC-AEGG 参考点的孔隙率和预测上升高度按 5 次独立生成结构进行结构级汇总，并报告 mean ± s.d.。因此，ε_PC-AEGG={pc_eps_mean:.3f} ± {pc_eps_sd:.3f} 表示独立生成结构之间的变异。Fig. 4 中理论孔隙率为 64.4% 的样品是为后续详细表征选取的单一代表性结构；该数值描述个体结构，不是对上述五结构均值的再次估计，二者的统计对象不同。
"""
    results = f"""## Results 草稿

在统一的 25 mm 设计域和相同降阶水力代理下，Sobol 搜索获得了梯度 BCC、梯度 Gyroid、Simple cubic、Diamond 与 Schwarz P 的 H_rise–ε 可行分布及合并周期非支配包络。PC-AEGG 的结构级孔隙率为 {pc_eps_mean:.3f} ± {pc_eps_sd:.3f}，预测上升高度为 {pc_h_mean:.1f} ± {pc_h_sd:.1f} mm（mean ± s.d., n=5）；图中同时显示 5 个独立结构，以保留重复间差异及其支配状态。从 N=4096 增加至 8192 后，合并包络的归一化超体积相对变化为 {summary['pooled_hv_change_4096_to_8192']:.3%}，达到曲线的 95% 孔隙率偏移为 {summary['pooled_attainment_p95_4096_to_8192']:.4f}；最大 Hausdorff 距离作为端点敏感诊断另见补充数据。参数边界扩展与 ε-constraint 精修结果见补充图。{conclusion} {front_status}
"""
    if not bool(summary["sampling_convergence_pass"]):
        results += (
            f"\n> 写作限制：4096→8192 的超体积相对变化为 {summary['pooled_hv_change_4096_to_8192']:.3%}，"
            f"达到曲线的 95% 孔隙率偏移为 {summary['pooled_attainment_p95_4096_to_8192']:.4f}，综合收敛门槛未通过。"
            "正式正文不应写‘前沿已近似收敛’；当前结果应称为‘每族 8192 点 Sobol 搜索并经局部精修得到的有限采样非支配包络’，"
            "待增加采样后再升级为收敛的近似 Pareto front。\n"
        )
    legend_en = f"""## Fig. 2e legend draft (English)

**Fig. 2e | Internal capillary-rise–porosity design envelope.** Feasible graded BCC, graded Gyroid, uniform simple-cubic, Diamond and Schwarz P candidates are shown in their respective colours; coloured lines connect the family-wise non-dominated samples as visual guides. Black points and the thin black guide denote the tolerance-aware periodic non-dominated envelope obtained in the unified 10 × 10 × 25 mm design domain (δ_H={summary['delta_H_mm']:.3g} mm and δ_ε={summary['delta_epsilon']:.3g}). Small semi-transparent red circles show five independently generated PC-AEGG structures, and the red diamond and error bar show their structure-level mean ± s.d. The shaded region marks the ε=0.58–0.65 matched-comparison window and was not used to prefilter Pareto candidates. Predicted rise height is capped at 25 mm because no geometry was extrapolated beyond the common design domain. The lines are visual guides through discrete samples and do not represent an analytical or demonstrably converged Pareto front. The ε_PC-AEGG value of {pc_eps_mean:.3f} ± {pc_eps_sd:.3f} summarizes variation among the five independently generated structures; the 64.4% theoretical porosity reported for the representative specimen in Fig. 4 refers to one structure selected for subsequent detailed characterization rather than a second estimate of this five-structure mean.

## Fig. 2e 图注草稿（中文核对）

**图 2e | 内部毛细上升高度–孔隙率设计包络。** 图中显示全部可行的梯度 BCC、梯度 Gyroid，以及均匀 Simple cubic、Diamond 和 Schwarz P 候选；彩色线仅连接各结构族的离散非支配样本，作为视觉引导。黑色点及细线表示统一 10 × 10 × 25 mm 设计域内、在 δ_H={summary['delta_H_mm']:.3g} mm 和 δ_ε={summary['delta_epsilon']:.3g} 容差下得到的周期结构非支配包络。5 个半透明红色小圆点表示独立生成的 PC-AEGG 结构，红色菱形和误差线表示其结构级 mean ± s.d.。阴影区域仅标示 ε=0.58–0.65 的匹配比较窗口，未用于 Pareto 候选预筛选。由于所有结构均未在共同设计域之外延拓，预测上升高度上限为 25 mm。图中连线不表示解析连续或已经证明收敛的 Pareto front。ε_PC-AEGG={pc_eps_mean:.3f} ± {pc_eps_sd:.3f} 描述 5 次独立生成结构之间的变异；Fig. 4 中理论孔隙率为 64.4% 的代表性样品是后续详细表征所选的单一结构，并非对上述五结构均值的再次估计。
"""
    path.write_text(methods + "\n" + results + "\n" + legend_en, encoding="utf-8")


def write_readme(
    path: Path,
    fluid: FluidConfig,
    domain: DomainConfig,
    initial_bounds: SearchBounds,
    main_bounds: SearchBounds,
    summary: dict[str, object],
) -> None:
    text = f"""# Internal H_rise–ε Pareto workflow (方案 A)

## Scope

This package recomputes the internal periodic benchmark in one 10 × 10 × 25 mm design domain. No geometry is extended above 25 mm. The ordinate is the raw predicted rise height and the abscissa is raw porosity. The shaded ε=0.58–0.65 interval in the main figure is a matched-comparison window only; it is not a candidate filter.

## Figure contract

- Core conclusion: locate PC-AEGG relative to the tolerance-aware periodic design envelope in a unified 25-mm domain.
- Archetype: quantitative comparison panel; convergence and sensitivity are supporting SI panels.
- Backend: Python/matplotlib only.
- Main size: 89 mm wide; editable SVG/PDF plus 300-dpi PNG and 600-dpi TIFF.
- Reviewer risks addressed: common hydraulic function, no porosity prefilter, no normalized axes, explicit ceiling, convergence, active-bound checks, local refinement and numerical tolerance.
- Main-panel terminology: the black periodic boundary is labelled `Periodic nondominated envelope`, not a converged front. Five PC-AEGG replicates are shown individually in addition to their structure-level mean ± s.d.
- Porosity reporting: ε_PC-AEGG is the structure-level mean ± s.d. across five independent generations. The 64.4% theoretical porosity in Fig. 4 belongs to one representative structure selected for detailed characterization and is not a second estimate of that mean.

## Geometry and parameterization

- Families: graded BCC and graded sheet-Gyroid, plus uniform simple-cubic, sheet-Diamond and sheet-Schwarz-P benchmarks.
- The three added benchmark families use L and one uniform physical feature size; bottom=top and p=1 are enforced for every candidate.
- BCC: (L, D_bottom, D_top, p); Gyroid: (L, t_bottom, t_top, p).
- Gradient: s(z)=s_bottom+(s_top−s_bottom)(z/H)^p.
- Uniform designs are explicitly injected into the two graded families by setting s_bottom=s_top in every 16th nested Sobol point; all three added benchmark families are uniform throughout.
- Initial sensitivity domain: L={initial_bounds.L_min_mm:g}–{initial_bounds.L_max_mm:g} mm and p={initial_bounds.p_min:g}–{initial_bounds.p_max:g}.
- Expanded main domain after the active-bound check: L={main_bounds.L_min_mm:g}–{main_bounds.L_max_mm:g} mm and p={main_bounds.p_min:g}–{main_bounds.p_max:g}; minimum feature={domain.min_manufacturable_feature_mm:.2f} mm.
- No ε=0.58–0.65 prefilter is used. Geometry validity and top-to-bottom liquid connectivity are the only candidate exclusions beyond the manufacturing minimum.
- Procedural geometry is evaluated over the complete domain on a {domain.xy_pitch_mm:g}-mm lateral grid and a {domain.slice_spacing_mm:g}-mm vertical grid. The true z=25 mm top boundary is explicitly sampled even when 25 mm is not an integer multiple of the selected z spacing. A finer xy/z repeat supplies numerical tolerances.

## Unified hydraulic calculation

At z={domain.slice_spacing_mm:g}, 2×{domain.slice_spacing_mm:g}, …, 25.0 mm, the liquid complement provides A_pore and C_wet. The code evaluates r_eq=2A_pore/C_wet, P_cap=2γcosθ/r_eq, P_hyd=Δρgz, and cumulative slice resistance. The imported PC-AEGG function computes P_visc from the serial accumulated resistance using the same fluid state and reference flow for every candidate and PC-AEGG replicate. R_tr=P_cap−P_hyd−P_visc. The first negative transport margin terminates continuous rise; the zero crossing is linearly interpolated between adjacent z samples. A fully passing profile is capped at the explicitly evaluated 25.0-mm top boundary.

Fluid values: γ={fluid.surface_tension_n_m:g} N m⁻¹, θ={fluid.contact_angle_deg:g}°, Δρ={fluid.density_difference_kg_m3:g} kg m⁻³, g={fluid.gravity_m_s2:g} m s⁻², μ={fluid.viscosity_pa_s:g} Pa s, Q_ref={fluid.reference_flow_m3_s:g} m³ s⁻¹.

## Search, convergence and dominance

- Nested scrambled Sobol prefixes: 1024, 2048, 4096, 8192 per family.
- Strict fronts and tolerance-aware fronts are both exported.
- Tolerance rule: a is no worse than b within (δ_ε, δ_H) and is better than b by more than at least one tolerance. Here δ_ε={summary['delta_epsilon']:.3g} and δ_H={summary['delta_H_mm']:.3g} mm.
- Global coverage is followed by ε-constraint differential-evolution refinement seeded from Sobol non-dominated points.
- Hypervolume is normalized to the physical rectangle ε∈[0,1], H∈[0,25] mm. Hausdorff distances use the same normalized coordinates; a height-threshold attainment shift is also reported because sparse endpoint replacement can dominate the maximum Hausdorff distance.

## Files

- `internal_pareto_candidates.csv`: complete periodic candidate table and front flags.
- `profiles/*.csv.gz`: batched per-height profiles for all search stages and five PC-AEGG references.
- `internal_pareto_front.csv`: family and pooled, strict and tolerance-aware fronts, before/after refinement.
- `internal_pareto_convergence.csv/json`: sampling diagnostics and headline summary.
- `internal_pareto_bound_sensitivity.csv`: initial/expanded-main front comparison and boundary occupancy.
- `pc_aegg_reference.csv`, `pc_aegg_dominance_audit.csv`: independent reference points and tolerance-dominance audit.
- `Figure_2e_internal_pareto.*`: main panel.
- `SI_*`: convergence, boundary, refinement and Pareto-parameter figures.
- `manuscript_text_zh.md`: Methods and Results drafts using computed values.

The plotted lines connect discrete non-dominated samples only as visual guides; they are not analytic continuous Pareto boundaries.

## Current diagnostic status

- Sampling convergence pass: {bool(summary['sampling_convergence_pass'])}.
- 4096→8192 relative hypervolume change: {summary['pooled_hv_change_4096_to_8192']:.4%}.
- 4096→8192 attainment-curve p95 porosity shift: {summary['pooled_attainment_p95_4096_to_8192']:.4f}.
- Active non-hard boundary after refinement: {bool(summary['active_nonphysical_main_boundary'])}.

Because the sampling convergence gate is not passed when the value above is `False`, the current result must not be described as a converged Pareto front. It remains a finite-sample non-dominated envelope with explicit convergence diagnostics.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydraulic-module", type=Path, required=True)
    parser.add_argument("--geometry-module", type=Path, required=True)
    parser.add_argument("--pc-run", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--sensitivity-samples", type=int, default=2048)
    parser.add_argument("--xy-pitch-mm", type=float, default=0.125)

    # Main z sampling used for all Sobol/refined candidates (default 0.75 mm).
    parser.add_argument("--z-pitch-mm", type=float, default=0.75)

    # Finer grid used only in the numerical resolution-stability audit.
    parser.add_argument("--stability-pitch-mm", type=float, default=0.10)
    parser.add_argument("--stability-z-pitch-mm", type=float, default=0.05)
    parser.add_argument("--stability-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--skip-refinement", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.pc_run) != 5:
        raise ValueError("Exactly five --pc-run directories are required for structure-level mean ± s.d.")
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles_dir = output / "profiles"
    writer = ProfileBatchWriter(profiles_dir)
    fluid = FluidConfig()
    domain = DomainConfig(
        xy_pitch_mm=args.xy_pitch_mm,
        slice_spacing_mm=args.z_pitch_mm,
    )
    grid = SliceGrid(domain)
    hydraulic_module = load_module(args.hydraulic_module, "pc_aegg_hydraulic_model")
    geometry_module = load_module(args.geometry_module, "pc_aegg_geometry_model")
    hydraulic_function = hydraulic_module.apply_state_updated_viscous_profile

    start = time.time()
    main_candidates = run_sobol_search(
        MAIN_BOUNDS, args.samples, args.seed, grid, fluid, hydraulic_function, writer
    )
    delta_epsilon, delta_h_mm, stability = stability_tolerances(
        main_candidates,
        MAIN_BOUNDS,
        args.stability_pitch_mm,
        args.stability_z_pitch_mm,
        domain,
        fluid,
        hydraulic_function,
        args.stability_samples,
    )
    stability.to_csv(output / "internal_pareto_resolution_stability.csv", index=False)

    initial_sensitivity = run_sobol_search(
        INITIAL_BOUNDS,
        args.sensitivity_samples,
        args.seed + 5000,
        grid,
        fluid,
        hydraulic_function,
        writer,
    )

    if args.skip_refinement:
        refined = pd.DataFrame(columns=main_candidates.columns)
        refinement_targets: list[float] = []
    else:
        refined, refinement_targets = run_refinement(
            main_candidates,
            MAIN_BOUNDS,
            grid,
            fluid,
            hydraulic_function,
            writer,
            delta_epsilon,
            args.seed + 8000,
        )
    periodic = pd.concat([main_candidates, refined], ignore_index=True, sort=False)
    periodic = add_front_flags(periodic, delta_epsilon, delta_h_mm)
    pc = evaluate_pc_references(
        args.pc_run, grid, fluid, hydraulic_function, geometry_module, writer
    )
    writer.close()
    allowed_profile_ids = set(periodic["design_id"]) | set(initial_sensitivity["design_id"]) | set(pc["design_id"])
    retained_profile_rows = retain_profile_designs(profiles_dir, allowed_profile_ids)

    levels = [n for n in (1024, 2048, 4096, 8192) if n <= args.samples]
    convergence = convergence_table(main_candidates, levels)
    pooled_conv = (
        convergence[convergence["scope"] == "Pooled periodic"].set_index("N_per_family")
        if not convergence.empty
        else pd.DataFrame()
    )
    if 4096 in pooled_conv.index and 8192 in pooled_conv.index:
        hv_change = float(pooled_conv.loc[8192, "hypervolume_relative_change_from_previous"])
        hd_change = float(pooled_conv.loc[8192, "normalized_Hausdorff_from_previous"])
        attainment_p95 = float(
            pooled_conv.loc[8192, "attainment_p95_abs_epsilon_shift_from_previous"]
        )
    else:
        hv_change = math.nan
        hd_change = math.nan
        attainment_p95 = math.nan

    occupancy = pd.concat(
        [
            boundary_occupancy(initial_sensitivity, INITIAL_BOUNDS, (delta_epsilon, delta_h_mm)),
            boundary_occupancy(periodic, MAIN_BOUNDS, (delta_epsilon, delta_h_mm)),
        ],
        ignore_index=True,
    )
    initial_front = front(initial_sensitivity[initial_sensitivity["feasible"]])
    main_front = front(main_candidates[main_candidates["feasible"]])
    bound_comparison = pd.DataFrame(
        [
            {
                "record_type": "front_comparison",
                "bounds_label": "initial_vs_expanded_main",
                "initial_samples_per_family": args.sensitivity_samples,
                "expanded_main_samples_per_family": args.samples,
                "initial_hypervolume_fraction": hypervolume_fraction(initial_front),
                "expanded_main_hypervolume_fraction": hypervolume_fraction(main_front),
                "expanded_relative_hypervolume_gain": (
                    hypervolume_fraction(main_front) - hypervolume_fraction(initial_front)
                )
                / max(hypervolume_fraction(main_front), 1.0e-12),
                "normalized_Hausdorff_distance": normalized_hausdorff(initial_front, main_front),
            }
        ]
    )
    occupancy_export = occupancy.copy()
    occupancy_export.insert(0, "record_type", "boundary_occupancy")
    bound_export = pd.concat([bound_comparison, occupancy_export], ignore_index=True, sort=False)

    fronts = build_front_export(periodic, delta_epsilon, delta_h_mm)
    dominance = periodic_dominates_pc(periodic, pc, delta_epsilon, delta_h_mm)
    sobol_front = front(main_candidates[main_candidates["feasible"]])
    refined_front = front(periodic[periodic["feasible"]])
    refinement_hv_gain = (
        hypervolume_fraction(refined_front) - hypervolume_fraction(sobol_front)
    ) / max(hypervolume_fraction(refined_front), 1.0e-12)
    refinement_distance = normalized_hausdorff(sobol_front, refined_front)

    nonphysical_boundary = occupancy[
        (occupancy["bounds_label"] == MAIN_BOUNDS.label)
        & ~occupancy["hard_or_saturated_boundary"]
    ]
    active_nonphysical_boundary = bool(
        len(nonphysical_boundary) and nonphysical_boundary["fraction_at_boundary"].max() >= 0.20
    )
    convergence_pass = bool(
        np.isfinite(hv_change)
        and abs(hv_change) < 0.01
        and np.isfinite(attainment_p95)
        and attainment_p95 < max(0.01, 2.0 * delta_epsilon)
    )
    bound_gain = float(bound_comparison.iloc[0]["expanded_relative_hypervolume_gain"])
    bound_distance = float(bound_comparison.iloc[0]["normalized_Hausdorff_distance"])
    bound_sensitivity_pass = bool(not active_nonphysical_boundary)
    summary: dict[str, object] = {
        "scheme": "A: unified 25-mm design domain; no extension",
        "samples_per_family": args.samples,
        "initial_sensitivity_samples_per_family": args.sensitivity_samples,
        "feasible_main_candidates": int(main_candidates["feasible"].sum()),
        "total_main_candidates": int(len(main_candidates)),
        "refined_candidates": int(len(refined)),
        "delta_epsilon": delta_epsilon,
        "delta_H_mm": delta_h_mm,
        "pooled_hv_change_4096_to_8192": hv_change,
        "pooled_hausdorff_4096_to_8192": hd_change,
        "pooled_attainment_p95_4096_to_8192": attainment_p95,
        "sampling_convergence_pass": convergence_pass,
        "active_nonphysical_main_boundary": active_nonphysical_boundary,
        "expanded_relative_hypervolume_gain": bound_gain,
        "expanded_normalized_Hausdorff_distance": bound_distance,
        "bound_sensitivity_pass": bound_sensitivity_pass,
        "sobol_to_refined_relative_hypervolume_gain": refinement_hv_gain,
        "sobol_to_refined_normalized_Hausdorff_distance": refinement_distance,
        "refinement_epsilon_targets": refinement_targets,
        "pc_epsilon_mean": float(pc["epsilon"].mean()),
        "pc_epsilon_sd": float(pc["epsilon"].std(ddof=1)),
        "pc_height_mean_mm": float(pc["predicted_rise_height_mm"].mean()),
        "pc_height_sd_mm": float(pc["predicted_rise_height_mm"].std(ddof=1)),
        "pc_replicates_tolerance_dominated": int(
            dominance.loc[dominance["pc_design_id"] != "PC-AEGG-mean", "is_tolerance_dominated"].sum()
        ),
        "pc_mean_tolerance_dominated": bool(
            dominance.loc[dominance["pc_design_id"] == "PC-AEGG-mean", "is_tolerance_dominated"].iloc[0]
        ),
        "hydraulic_source_sha256": sha256_file(args.hydraulic_module),
        "geometry_source_sha256": sha256_file(args.geometry_module),
        "runtime_seconds": float(time.time() - start),
        "profile_rows": retained_profile_rows,
        "z_sampling": {
            "main_z_pitch_mm": float(domain.slice_spacing_mm),
            "stability_z_pitch_mm": float(args.stability_z_pitch_mm),
            "true_top_explicitly_sampled": True,
            "zero_crossing_interpolation": True,
        },
        "fluid": asdict(fluid),
        "domain": asdict(domain),
        "initial_bounds": asdict(INITIAL_BOUNDS),
        "main_bounds": asdict(MAIN_BOUNDS),
        "dominance_definition": (
            "a dominates b when a is no worse than b within delta_epsilon and delta_H, "
            "and a is better than b by more than at least one tolerance"
        ),
    }

    periodic.to_csv(output / "internal_pareto_candidates.csv", index=False)
    initial_sensitivity.to_csv(output / "internal_pareto_bound_sensitivity_candidates.csv", index=False)
    pc.to_csv(output / "pc_aegg_reference.csv", index=False)
    fronts.to_csv(output / "internal_pareto_front.csv", index=False)
    convergence.to_csv(output / "internal_pareto_convergence.csv", index=False)
    bound_export.to_csv(output / "internal_pareto_bound_sensitivity.csv", index=False)
    dominance.to_csv(output / "pc_aegg_dominance_audit.csv", index=False)
    (output / "internal_pareto_convergence.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_main(periodic, pc, delta_epsilon, delta_h_mm, output)
    if args.samples >= 1024:
        plot_convergence(main_candidates, convergence, output)
    plot_boundary_sensitivity(initial_sensitivity, periodic, occupancy, output)
    plot_refinement(main_candidates, periodic, output)
    plot_front_parameters(periodic, output)
    write_manuscript_text(output / "manuscript_text_zh.md", summary, pc, dominance)
    write_readme(output / "README.md", fluid, domain, INITIAL_BOUNDS, MAIN_BOUNDS, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
