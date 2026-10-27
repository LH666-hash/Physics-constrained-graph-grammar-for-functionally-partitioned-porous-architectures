"""Recheck each frozen surface-ablation graph with the current global pore audit.

The calculation uses the packaged code and writes temporary results outside
Source_Data. The archived summary is checked against the new calculation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code/Core'))
from global_surface_pore_audit import global_audit

GROUPS = ROOT / 'Source_Data/Ablation_Analysis/表面/Surface_Groups'
STL = ROOT / 'Input_Geometries/长方体.stl'


def verify(source: Path) -> None:
    prefix = source.name
    saved = json.loads((source / f'{prefix}_contour_summary.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='pc_aegg_ablation_') as temp:
        run_dir = Path(temp)
        validation = run_dir / 'surface_validation'
        validation.mkdir()
        for original, standard in (
            ('graph_nodes.csv', 'surface_nodes.csv'),
            ('graph_edges.csv', 'surface_edges.csv'),
            ('local_graph_summary.json', 'surface_final_graph_summary.json'),
        ):
            shutil.copy2(source / f'{prefix}_{original}', validation / standard)
        result = global_audit(SimpleNamespace(
            stl=STL, run_dir=run_dir, target_height_mm=25.0,
            global_max_edge_mm=float(saved['edge_mm']),
            solid_vote_fraction=0.5, aspect_ratio_max=5.0,
            aspect_ratio_min_area_mm2=0.5, minimum_pore_area_mm2=1e-6,
            radius_acceptance_mm=float(saved['radius_acceptance_mm']),
            search_padding_mm=0.02, out_dir_name='global_surface_pore_audit',
            export_face_classification=False,
        ))
    checks = (
        ('complete_pores', 'global_complete_pore_count', 0),
        ('max_radius_mm', 'global_max_complete_pore_equivalent_radius_mm', 1e-9),
        ('radius_exceeded_count', 'global_radius_exceeded_complete_pore_count', 0),
        ('open_area_fraction', 'global_liquid_open_area_fraction', 1e-9),
        ('mean_shape_factor', 'global_mean_complete_pore_shape_factor', 1e-9),
        ('mean_aspect_ratio', 'global_mean_complete_pore_aspect_ratio', 1e-9),
    )
    for saved_key, current_key, tolerance in checks:
        if abs(float(saved[saved_key]) - float(result[current_key])) > tolerance:
            raise AssertionError(f'{prefix}: {saved_key} differs from current code')
    print(f'{prefix}: verified')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', action='append', help='Group folder name; omit to audit all 15')
    args = parser.parse_args()
    sources = [GROUPS / name for name in args.group] if args.group else sorted(GROUPS.iterdir())
    for source in sources:
        if not source.is_dir():
            raise FileNotFoundError(source)
        verify(source)


if __name__ == '__main__':
    main()
