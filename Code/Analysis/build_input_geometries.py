"""Build replacement box inputs from the recorded dimensions and carrier grid.

These are newly generated inputs, not recovered historical binary files.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import trimesh

PACKAGE = Path(__file__).resolve().parents[2]


def gridded_box(extents, divisions):
    vertices, faces, indices = [], [], {}

    def vertex(key):
        key = tuple(key)
        if key not in indices:
            indices[key] = len(vertices)
            vertices.append([extents[k] * key[k] / divisions[k] for k in range(3)])
        return indices[key]

    for axis in range(3):
        u, v = [k for k in range(3) if k != axis]
        for side in (0, divisions[axis]):
            for i in range(divisions[u]):
                for j in range(divisions[v]):
                    corners = []
                    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        key = [0, 0, 0]
                        key[axis], key[u], key[v] = side, i + du, j + dv
                        corners.append(vertex(key))
                    a, b, c, d = corners
                    faces.extend(((a, b, c), (a, c, d)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    return mesh


def main():
    folder = PACKAGE / 'Input_Geometries'
    specs = [('water35', 'box_10x10x35mm.stl', (10., 10., 35.), (1, 1, 1), 1000),
             ('organic22p5', 'box_10x10x22.5mm.stl', (10., 10., 22.5), (10, 20, 35), 5000)]
    sys.path.insert(0, str(PACKAGE / 'Code/Core'))
    import PC_AEGG_主程序 as pc
    from generate_case import command
    records = []
    for case, name, extents, divisions, target in specs:
        path = folder / name
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite an existing input: {path}')
        mesh = gridded_box(extents, divisions)
        payload = trimesh.exchange.stl.export_stl(mesh)
        path.write_bytes(payload)
        loaded = trimesh.load_mesh(path, process=True)
        assert len(loaded.faces) == 4 * sum(divisions[i] * divisions[j] for i, j in ((0, 1), (0, 2), (1, 2)))
        assert loaded.is_watertight and loaded.is_winding_consistent
        assert np.allclose(loaded.bounds, [[0, 0, 0], extents])
        assert np.isclose(loaded.volume, np.prod(extents))
        scaled, _ = pc.load_and_scale_mesh(path, extents[2])
        config = json.loads((PACKAGE / f'Code/Configurations/{case}_reference.json').read_text(encoding='utf-8'))
        pc_config_path = next((PACKAGE / ('Source_Data/35mm_extended' if case == 'water35' else 'Source_Data/Orgnic_structure')).rglob('surface_build_config.json'))
        surface_config = pc.SurfaceBuildConfig(**json.loads(pc_config_path.read_text(encoding='utf-8')))
        builder = pc.SurfacePatchBuilder(scaled, surface_config)
        carrier = builder._prepare_base_mesh()
        assert len(carrier.faces) == (12 if case == 'water35' else 5000)
        cmd, default_input = command(case, Path('NewRuns') / case)
        assert default_input.resolve() == path.resolve()
        assert float(cmd[cmd.index('--height-mm') + 1]) == config['target_height_mm']
        records.append(dict(case=case, file=f'Input_Geometries/{name}',
                            origin='newly generated from recorded box dimensions and grid settings',
                            extents_mm=list(extents), grid_divisions_xyz=list(divisions),
                            input_faces=len(loaded.faces), requested_target_faces=target,
                            actual_builder_carrier_faces=len(carrier.faces),
                            watertight=bool(loaded.is_watertight), winding_consistent=bool(loaded.is_winding_consistent),
                            volume_mm3=float(loaded.volume), surface_area_mm2=float(loaded.area),
                            sha256=hashlib.sha256(payload).hexdigest()))
    audit = {'generated_on': '2026-10-20',
             'historical_binary_identity_claimed': False,
             'organic_grid_basis': 'archived input_carrier_geometry_audit.json: divisions [10,20,35]',
             'water35_grid_basis': 'archived surface_patch_metrics.csv: six patches, two faces per patch',
             'files': records}
    (PACKAGE / 'Code/Configurations/generated_input_geometries.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.path.insert(0, str(PACKAGE / 'Code/Analysis'))
    main()
