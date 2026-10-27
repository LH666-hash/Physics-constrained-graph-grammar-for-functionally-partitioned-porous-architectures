"""Restore explicit archived fluid, geometry and pressure settings for analysis."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_configuration(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    for section in ('fluid', 'surface', 'internal_dynamic', 'quasi_steady_filling'):
        if section not in data:
            raise ValueError(f'Missing archived configuration section: {section}')
    return data

def apply_saved_configuration(pc, path):
    """Set the quantities used by pressure reanalysis and same-seed stitching.

    This restores analysis conditions; it does not identify a missing historical
    generation source or guarantee identical graphs from the present source.
    """
    data=load_configuration(path)
    fluid=data['fluid']; surface=data['surface']; internal=data['internal_dynamic']
    mapping={
        'GAMMA_NPM':fluid['surface_tension_N_per_m'],
        'CONTACT_ANGLE_DEG':fluid['contact_angle_deg'],
        'RHO_LIQ':fluid['density_kg_per_m3'],
        'RHO_GAS':fluid['gas_density_kg_per_m3'],
        'G_GRAV':fluid['gravity_m_per_s2'],
        'SAFETY_FACTOR':fluid['safety_factor'],
        'SAFETY_FACTOR_BOTTOM':fluid['safety_factor_bottom'],
        'SAFETY_FACTOR_TOP':fluid['safety_factor_top'],
        'FLUID_VISCOSITY_PA_S':fluid['viscosity_Pa_s'],
        'VISC_REF_Q_M3_S':fluid['reference_flow_rate_m3_s'],
        'SURFACE_DIAM_MM':surface['diameter_mm'],
        'SURFACE_EXTRA_PRESSURE_PA':surface['additional_pressure_Pa'],
        'SURFACE_CAPILLARY_SAFETY_FACTOR':surface['capillary_safety_factor'],
        'R_EFF_LIMIT_MM':surface['r_eff_limit_mm'],
        'INTERNAL_DIAM_BOTTOM_MM':internal['internal_diam_bottom_mm'],
        'INTERNAL_DIAM_TOP_MM':internal['internal_diam_top_mm'],
        'TARGET_HEIGHT_MM':data['target_height_mm'],
        'SLICE_DZ_MM':data['quasi_steady_filling']['pressure_slice_nominal_thickness_mm'],
        'INTERNAL_DYNAMIC_PHYSICS_DZ_MM':data['quasi_steady_filling']['pressure_slice_nominal_thickness_mm'],
        'INTERNAL_DYNAMIC_GLOBAL_FEEDBACK_SPACING_MM':data['quasi_steady_filling']['pressure_slice_nominal_thickness_mm'],
        'INTERNAL_DYNAMIC_UPPER_FEEDBACK_SPACING_MM':data['quasi_steady_filling']['pressure_slice_nominal_thickness_mm'],
        'FLOW_RESISTANCE_CALIBRATION_FACTOR':data['quasi_steady_filling']['resistance_calibration_factor'],
        'INTERNAL_DYNAMIC_CAPILLARY_MIN_RATIO':internal['capillary_min_ratio'],
        'INTERNAL_DYNAMIC_CAPILLARY_TARGET_RATIO':internal['capillary_target_ratio'],
        'INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM':internal['capillary_band_start_mm'],
        'INTERNAL_DYNAMIC_CAPILLARY_ASSEMBLY_RESERVE_RATIO':internal['capillary_assembly_reserve_ratio'],
        'INTERNAL_DYNAMIC_MAX_EDGES_PER_NEW':internal['max_edges_per_new'],
    }
    stitch=data['dual_stitching']
    mapping.update(STITCH_DISTANCE_MM=stitch['standard_length_mm'],
                   STITCH_RELAXED_DISTANCE_MM=stitch['relaxed_search_length_mm'],
                   STITCH_ANCHOR_SPACING_MM=stitch['anchor_spacing_mm'])
    for key,value in mapping.items():
        if not hasattr(pc,key):
            raise ValueError(f'Current source does not implement setting {key}')
        setattr(pc,key,value)
    pc.PRESSURE_SLICE_COUNT = len(pc._pressure_feedback_heights((0.0, pc.TARGET_HEIGHT_MM)))
    return data

def pairing_fingerprint(paired, source):
    inputs={
        'surface_nodes':paired/'surface_validation/surface_nodes.csv',
        'surface_edges':paired/'surface_validation/surface_edges.csv',
        'surface_configuration':paired/'surface_validation/surface_build_config.json',
        'internal_nodes':source/'structure_nodes.csv',
        'internal_edges':source/'structure_edges.csv',
        'configuration':paired/'run_config.json',
        'input_geometry':PACKAGE/'Input_Geometries/长方体.stl',
    }
    inputs.update({f'core/{p.name}':p for p in (PACKAGE/'Code/Core').glob('*.py')})
    inputs.update({f'analysis/{n}':PACKAGE/'Code/Analysis'/n for n in
                   ('pair_ablation_controls.py','saved_configuration.py')})
    return {key:sha(path) for key,path in sorted(inputs.items())}
