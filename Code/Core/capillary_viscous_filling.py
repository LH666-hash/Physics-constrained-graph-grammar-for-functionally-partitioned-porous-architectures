# -*- coding: utf-8 -*-
"""Original-PC-AEGG-compatible slice-wise viscous-loss model.

This module keeps the design-loop pressure treatment used in the user's
original ``pore_network_viscous_compensation`` code:

    Rg_i       = 8 dz_i / (pi * sum_c r_eq,c^4)
    P_visc(z)  = mu * Q_ref * sum_{j<=i} Rg_j
    P_required = S(z) * [P_hydro(z) + P_visc(z)]
    R_tr       = P_cap(z) - P_required(z)

Components inside one horizontal slice are treated as parallel conduits and
successive slices as series resistances.  The only deliberate departures from
the old file are (i) compatibility aliases used by the newer main program and
(ii) accepting an explicit ``slice_dz_mm`` column so a nominal 1-mm grid can end
with one shorter top bin for non-integer design heights.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


def _polygons(geometry) -> List[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 0.0 else []
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        result: List[Polygon] = []
        for part in geometry.geoms:
            result.extend(_polygons(part))
        return result
    return []


def equivalent_radius_mm(area_mm2: float, perimeter_mm: float) -> float:
    """Circular-equivalent capillary radius, r_eq = 2A/P, in millimetres."""
    if area_mm2 <= 0.0 or perimeter_mm <= 0.0:
        return 0.0
    return 2.0 * float(area_mm2) / float(perimeter_mm)


def section_radius4_sum_mm4(
    liquid_geometry,
    min_area_mm2: float = 1.0e-4,
) -> Tuple[float, int]:
    """Return sum(r_eq**4) for parallel liquid components in one slice."""
    radii = []
    for poly in _polygons(liquid_geometry):
        if float(poly.area) < float(min_area_mm2):
            continue
        radius = equivalent_radius_mm(float(poly.area), float(poly.length))
        if radius > 0.0:
            radii.append(radius)
    return float(sum(radius ** 4 for radius in radii)), len(radii)


def apply_state_updated_viscous_profile(
    profile: pd.DataFrame,
    viscosity_pa_s: float,
    reference_flow_m3_s: float,
    default_dz_mm: float,
    target_fill_time_s: float = 0.0,
    safety_column: str = "safety_factor_z",
) -> pd.DataFrame:
    """Accumulate the original height-dependent viscous pressure budget.

    For slice ``i`` the geometric resistance is

        Rg_i = 8 dz_i / (pi sum_c r_eq,c^4).

    The cumulative viscous loss at front height ``z_i`` is

        P_visc,i = mu * Q_req,i * sum_(j<=i) Rg_j.

    With ``target_fill_time_s == 0`` (the design setting used here),
    ``Q_req`` is the fixed reference flow.  The original acceptance pressure is

        P_required = S(z) * (P_hydro + P_visc),

    and the corresponding transport/safety margin is

        R_tr = P_cap - P_required.

    ``target_fill_time_s`` is retained for compatibility with the original
    module but should remain zero when reproducing the manuscript's fixed-flow
    design rule.
    """
    if profile is None or profile.empty:
        return profile

    viscosity = float(viscosity_pa_s)
    q_ref = float(reference_flow_m3_s)
    if viscosity <= 0.0:
        raise ValueError("viscosity_pa_s must be > 0")
    if q_ref < 0.0:
        raise ValueError("reference_flow_m3_s must be >= 0")

    out = profile.sort_values(
        "z_mid_mm" if "z_mid_mm" in profile.columns else "z_mm"
    ).copy()
    z_col = "z_mid_mm" if "z_mid_mm" in out.columns else "z_mm"
    z_mm = out[z_col].to_numpy(dtype=float)

    # Original code derived dz from adjacent slice centres.  The newer main
    # program supplies the true bin width explicitly so a fixed 1-mm grid can
    # terminate exactly at a non-integer top surface without altering the
    # resistance formula itself.
    if "slice_dz_mm" in out.columns:
        dz_mm = np.maximum(out["slice_dz_mm"].to_numpy(dtype=float), 1.0e-9)
    elif len(z_mm) == 1:
        dz_mm = np.array([max(float(default_dz_mm), 1.0e-9)])
    else:
        gaps = np.diff(z_mm)
        dz_mm = np.r_[max(gaps[0], 1.0e-9), np.maximum(gaps, 1.0e-9)]

    if "axial_radius4_sum_mm4" in out.columns:
        radius4_mm4 = out["axial_radius4_sum_mm4"].to_numpy(dtype=float)
    else:
        radius4_mm4 = np.maximum(out["r_eff_mm"].to_numpy(dtype=float), 0.0) ** 4

    radius4_m4 = np.maximum(radius4_mm4 * 1.0e-12, 1.0e-30)
    segment_geom_m3_inv = 8.0 * dz_mm * 1.0e-3 / (math.pi * radius4_m4)
    cumulative_geom_m3_inv = np.cumsum(segment_geom_m3_inv)

    if target_fill_time_s > 0.0 and "A_liq_mm2" in out.columns:
        height_span_m = max(
            float(z_mm.max() - z_mm.min() + np.median(dz_mm)) * 1.0e-3,
            1.0e-12,
        )
        front_speed_m_s = height_span_m / float(target_fill_time_s)
        q_req = (
            np.maximum(out["A_liq_mm2"].to_numpy(dtype=float), 0.0)
            * 1.0e-6
            * front_speed_m_s
        )
        flow_source = "target_fill_time"
    else:
        q_req = np.full(len(out), max(q_ref, 0.0), dtype=float)
        front_speed_m_s = np.nan
        flow_source = "reference_flow"

    viscous_pa = viscosity * q_req * cumulative_geom_m3_inv
    safety = (
        out[safety_column].to_numpy(dtype=float)
        if safety_column in out.columns
        else np.ones(len(out))
    )
    hydro = out["P_hydro_Pa"].to_numpy(dtype=float)
    capillary = out["P_cap_Pa"].to_numpy(dtype=float)

    # This is the exact pressure construction used in the original code.
    required = safety * (hydro + viscous_pa)
    margin = capillary - required

    out["slice_dz_mm"] = dz_mm
    out["Q_required_m3_s"] = q_req
    out["Q_required_source"] = flow_source
    out["target_front_speed_m_s"] = front_speed_m_s
    out["R_geom_segment_m-3"] = segment_geom_m3_inv
    out["R_geom_cumulative_m-3"] = cumulative_geom_m3_inv
    out["R_hyd_cumulative_Pa_s_m-3"] = viscosity * cumulative_geom_m3_inv
    out["P_visc_dynamic_Pa"] = viscous_pa
    out["P_visc_ref_Pa"] = viscous_pa
    out["P_required_Pa"] = required
    out["safety_margin"] = margin
    out["Safety_Margin"] = margin
    out["deficit_Pa"] = np.maximum(0.0, -margin)
    out["rel_def"] = out["deficit_Pa"].to_numpy(dtype=float) / np.maximum(
        required, 1.0e-12
    )

    # Compatibility aliases for the newer main program.  They do not change
    # the original physics above.
    out["P_viscous_Pa"] = viscous_pa
    out["P_viscous_capacity_proxy_Pa"] = viscous_pa
    out["P_required_base_Pa"] = hydro + viscous_pa
    out["physical_transport_margin_Pa"] = capillary - (hydro + viscous_pa)
    out["transport_margin_Pa"] = margin
    out["R_tr_Pa"] = margin
    out["reference_flow_rate_m3_s"] = q_req
    out["ref_flow_m3_s"] = q_req

    # Newer code paths still use these as diagnostics/prioritisation only.
    # They are derived after the original pressure budget and do not alter it.
    safe = np.maximum(safety, 1.0e-12)
    available_head = capillary / safe - hydro
    hydraulic_resistance = viscosity * cumulative_geom_m3_inv
    residual_head = capillary / safe - hydro - viscous_pa
    q_capacity = np.maximum(available_head, 0.0) / np.maximum(hydraulic_resistance, 1.0e-300)
    mobility = np.maximum(available_head, 0.0) / np.maximum(cumulative_geom_m3_inv, 1.0e-300)
    out["available_pressure_head_Pa"] = available_head
    out["reference_flow_residual_head_Pa"] = residual_head
    out["Q_available_proxy_m3_s"] = q_capacity
    out["flow_capacity_ratio"] = q_capacity / np.maximum(q_req, 1.0e-300)
    out["capillary_viscous_mobility_Pa_m3"] = mobility
    out["capillary_stall_proxy"] = residual_head <= 0.0
    out["diagnostic_time_used_for_acceptance"] = False
    return out
