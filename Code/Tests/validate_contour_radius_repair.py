"""Analytic square-pore repair check at two carrier orientations."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code' / 'Core'))
from global_surface_contour_repair import (  # noqa: E402
    ContourRadiusRepair, apply_global_contour_radius_repair,
)
from surface_patch_builder import SurfaceBuildConfig  # noqa: E402


def square(rotation_degrees: float, integrated: bool = False) -> dict:
    vertices = np.array([[-1.5, -1.5, 0], [1.5, -1.5, 0],
                         [1.5, 1.5, 0], [-1.5, 1.5, 0]], dtype=float)
    carrier = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    angle = np.deg2rad(rotation_degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    points = np.array([[-.75, -.75], [.75, -.75], [.75, .75], [-.75, .75]]) @ rotation.T
    nodes = pd.DataFrame([
        dict(atom_id=i, x=x, y=y, z=0, face_name='surface', patch_id=0,
             component_id=0, uv_x=x, uv_y=y, source_vertex_id=-1,
             node_role='analytic_square', patch_memberships='0')
        for i, (x, y) in enumerate(points)
    ])
    edges = pd.DataFrame([
        dict(source=i, target=(i+1) % 4, radius=.15, bond_type='surface',
             role='analytic_square', patch_id=0, patch_memberships='0',
             component_id=0, logical_edge_id=i)
        for i in range(4)
    ])
    metrics = pd.DataFrame([
        dict(logical_edge_id=i, patch_id=0, component_id=0, polyline_segments=1,
             max_chord_surface_error_mm=0, domain_violation=0, mapping_fallback=0,
             cross_component=0, anisotropy_wall=0, accepted=1)
        for i in range(4)
    ])
    config_path = (ROOT / 'Source_Data' / 'Extended_Geometries' /
                   'Gaussian_Curvature_Sign_Change_seed42' / 'surface_validation' /
                   'surface_build_config.json')
    config = SurfaceBuildConfig(**json.loads(config_path.read_text(encoding='utf-8')))
    builder = SimpleNamespace(original_mesh=carrier, base_mesh=carrier,
                              config=config, atoms=nodes, edges=edges,
                              edge_metrics=metrics, prefer_transverse=True)
    result = (apply_global_contour_radius_repair(builder) if integrated else
              ContourRadiusRepair(builder, max_repairs=3, candidates_per_pore=5).run())
    if not integrated:
        assert result['initial_maximum_radius_mm'] > result['radius_acceptance_mm']
    assert result['completed'] and result['final_radius_exceeded_count'] == 0
    assert result['final_maximum_radius_mm'] < result['radius_acceptance_mm']
    # The repair loop updates only affected components. Recompute the entire
    # carrier independently to check that its accepted result is identical.
    full, complete, bad = ContourRadiusRepair(builder).measure()
    assert not np.any(bad)
    assert np.isclose(np.max(full['radius'][complete]),
                      result['final_maximum_radius_mm'], atol=1e-8)
    return dict(rotation_degrees=rotation_degrees, integrated_workflow=integrated,
                initial_radius_mm=result['initial_maximum_radius_mm'],
                final_radius_mm=result['final_maximum_radius_mm'],
                accepted_wall_count=result['added_walls'])


if __name__ == '__main__':
    outcomes = [square(degrees, integrated) for integrated in (False, True)
                for degrees in (0, 22.5)]
    destination = (ROOT / 'Source_Data' / 'Validation_Standards' /
                   'square_pore_contour_repair_validation.json')
    destination.write_text(json.dumps(outcomes, indent=2) + '\n', encoding='utf-8')
    print(destination)
