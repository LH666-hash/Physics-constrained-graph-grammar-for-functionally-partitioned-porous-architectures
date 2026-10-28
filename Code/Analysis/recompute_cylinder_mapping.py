"""A versioned controlled cylinder-mapping check using the deposited surface code."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
from dataclasses import asdict
import numpy as np
import trimesh
from shapely.geometry import Point

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Code/Core'))
from surface_patch_builder import SurfacePatchBuilder,SurfaceBuildConfig

def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError('Choose a new output directory.')
    out.mkdir(parents=True)
    settings={'cylinder_radii_mm':[5.,10.,20.],'patch_arc_width_mm':2.5,'patch_height_mm':2.5,
              'grid_vertices_per_direction':6,'pores_per_radius':40,'polygon_vertices':128,
              'nominal_pore_radii_mm':[.1,.9],'centre_jitter_mm':.15,'random_seed':42,
              'comparison':'100 * (r_eq_3d - r_eq_2d) / r_eq_2d','threshold_percent':.5,
              'record_type':'New controlled recomputation; not recovered historical Figure3g scatter data.'}
    rows=[];summaries=[]
    for radius in settings['cylinder_radii_mm']:
        u=np.linspace(-1.25,1.25,6);v=np.linspace(-1.25,1.25,6)
        vertices=np.array([[radius*np.sin(x/radius),radius*(1-np.cos(x/radius)),z] for z in v for x in u])
        faces=[]
        for j in range(5):
            for i in range(5):
                a=j*6+i;faces.extend([[a,a+1,a+7],[a,a+7,a+6]])
        mesh=trimesh.Trimesh(vertices=vertices,faces=np.array(faces),process=False)
        config=SurfaceBuildConfig(mapping_surface='carrier',edge_sample_spacing_mm=.35)
        builder=SurfacePatchBuilder(mesh,config)
        builder.base_mesh=builder.original_mesh.copy()
        builder.base_vertex_gaussian_curvature=np.zeros(len(builder.base_mesh.vertices))
        chart=builder._make_chart(0,np.arange(len(builder.base_mesh.faces)))
        rng=np.random.default_rng(settings['random_seed'])
        scale=float(chart.scale_2d_to_3d)
        centres=rng.uniform(-.15,.15,(40,2))/scale
        for pore_id,(nominal,centre) in enumerate(zip(np.linspace(.1,.9,40),centres)):
            poly=Point(*centre).buffer(float(nominal)/scale,resolution=32)
            assert chart.polygon.covers(poly),'Test pore leaves its chart.'
            area2=float(poly.area)*scale**2;perimeter2=float(poly.length)*scale
            area3,perimeter3=builder._mapped_polygon_metrics(chart,poly)
            req2=2*area2/perimeter2;req3=2*area3/perimeter3
            error=100*(req3-req2)/req2
            rows.append({'cylinder_radius_mm':radius,'pore_id':pore_id,'nominal_pore_radius_mm':float(nominal),
                         'center_u':float(centre[0]),'center_v':float(centre[1]),'area_2d_mm2':area2,
                         'perimeter_2d_mm':perimeter2,'area_3d_mm2':area3,'perimeter_3d_mm':perimeter3,
                         'r_eq_2d_mm':req2,'r_eq_3d_mm':req3,'relative_error_percent':error,'absolute_error_percent':abs(error)})
        np.savez_compressed(out/f'cylinder_R{int(radius)}_chart.npz',uv=chart.uv,xyz=chart.xyz,faces=chart.local_faces,centres=centres,scale_2d_to_3d=scale)
        mesh.export(out/f'cylinder_R{int(radius)}_patch.stl')
        values=np.array([r['absolute_error_percent'] for r in rows if r['cylinder_radius_mm']==radius])
        summaries.append({'cylinder_radius_mm':radius,'pores':len(values),'maximum_absolute_error_percent':float(values.max()),
                          'p95_absolute_error_percent':float(np.percentile(values,95)),'all_below_0p5_percent':bool(np.all(values<.5)),
                          'parameterization_method':chart.parameterization_method,'scale_2d_to_3d':scale})
    with (out/'cylinder_mapping_results.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    source=ROOT/'Code/Core/surface_patch_builder.py'
    provenance={'settings':settings,'surface_config':asdict(config),'surface_builder_source':'Code/Core/surface_patch_builder.py',
                'surface_builder_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'calculation_script':'Code/Analysis/recompute_cylinder_mapping.py',
                'calculation_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'base_surface_source_commit':'206b527387ecc851bb60fcd1e25fb989cad134ac','python_version':sys.version.split()[0],
                'numpy_version':np.__version__,'trimesh_version':trimesh.__version__,'summary':summaries}
    (out/'cylinder_mapping_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (out/'README.md').write_text('''# Controlled cylinder-mapping recomputation

This calculation supplies a newly generated, traceable geometric verification for cylinder radii of 5, 10 and 20 mm. It uses the deposited surface builder's unmodified `_make_chart` and `_mapped_polygon_metrics` methods. It is separate from the historical scatter points plotted in Figure3g, whose original numerical source was not recovered.

The cylindrical patch spans 2.5 mm in arc length and 2.5 mm in height, with a 6-by-6 vertex grid (50 triangular faces). Its cylindrical geometry is analytic. Forty 128-sided test pores per radius span nominal radii of 0.1–0.9 mm, with fixed seed42 centre offsets of at most 0.15 mm. The builder constructs its principal-plane chart with the usual area-normalization factor and maps points through carrier-triangle barycentric coordinates. The mapped-area calculation clips each pore against carrier triangles. The mapped perimeter uses the builder's boundary sampling, capped at 0.05 mm. The output reports r_eq=2A/P and the signed percentage difference between the 3D and 2D values.

The CSV retains all 120 individual results. The NPZ files retain chart vertices, triangles and pore centres; the STL files retain the test surfaces. The provenance JSON records all settings, source checksums, dependency versions and summary values. These data verify the specified controlled test. They are not a new structure-generation run or a CT calculation.

From the extracted project directory, install the supplied tested dependencies and run:

```sh
python Code/Analysis/recompute_cylinder_mapping.py --out NewChecks/cylinder_mapping
```

Choose a new output directory. The historical full Figure3 is retained separately; this computation does not replace its plotted point coordinates.
''',encoding='utf-8')
    assert len(rows)==120 and all(np.isfinite(r['relative_error_percent']) for r in rows)
    print(json.dumps({'results':len(rows),'summary':summaries},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    run(parser.parse_args().out)
