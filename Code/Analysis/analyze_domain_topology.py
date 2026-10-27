"""Use the production section-domain routine to audit input topology by height."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / 'Code' / 'Core'
sys.path.insert(0, str(CODE))
import PC_AEGG_主程序 as model  # noqa: E402

DOMAINS = [
    ('large_hole', '大孔洞体'),
    ('gaussian_sign_change', '高斯曲率变号'),
    ('local_negative_curvature', '局部高负曲率曲面'),
    ('smooth_variable_curvature', '连续光滑变曲率曲面'),
    ('knitted_elephant', 'knitted_elephant'),
]

rows = []
for slug, name in DOMAINS:
    mesh, _ = model.load_and_scale_mesh(ROOT / 'Input_Geometries' / (name + '.stl'), 25.0)
    for k in range(25):
        z = k + 0.5
        section = model._get_domain_poly_at_z(mesh, z)
        if section.is_empty:
            parts = []
        elif section.geom_type == 'Polygon':
            parts = [section]
        else:
            parts = [p for p in getattr(section, 'geoms', ()) if p.geom_type == 'Polygon']
        rows.append({
            'domain': slug,
            'height_mm': z,
            'cross_section_component_count': len(parts),
            'cross_section_hole_count': sum(len(p.interiors) for p in parts),
            'cross_section_euler_characteristic': len(parts) - sum(len(p.interiors) for p in parts),
            'cross_section_area_mm2': float(sum(p.area for p in parts)),
            'mesh_watertight': bool(mesh.is_watertight),
            'input_triangle_count': len(mesh.faces),
        })
    print(slug, 'done', flush=True)

out = ROOT / 'Source_Data' / 'Extended_Geometries' / 'domain_cross_section_topology.csv'
with out.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(out)
