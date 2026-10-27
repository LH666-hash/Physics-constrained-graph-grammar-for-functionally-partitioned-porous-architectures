from pathlib import Path
import sys,importlib.util,json
import pandas as pd
source=Path(__file__).resolve().parents[3]/'Code/Core'
sys.path.insert(0,str(source))
sys.path.insert(0,str(source.parent/'Analysis'))
from generate_case import initialize_output_directory
# The public entry point reserves an empty directory; direct calls use the
# same initializer. Existing results and all archived Source_Data stay protected.
output=initialize_output_directory(Path(sys.argv[sys.argv.index('--out-dir')+1]),
                                   allow_existing_empty=True)
spec=importlib.util.spec_from_file_location('fixed_diameter_core',source/'PC_AEGG_主程序.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
from joblib import Parallel as OriginalParallel
def serial_analysis_parallel(*args,**kwargs):
 kwargs['n_jobs']=1
 return OriginalParallel(*args,**kwargs)
m.Parallel=serial_analysis_parallel
from surface_patch_builder import SurfacePatchBuilder
original_prepare=SurfacePatchBuilder._prepare_base_mesh
def audited_prepare(self):
 mesh=original_prepare(self)
 out=Path(sys.argv[sys.argv.index('--out-dir')+1])
 info={'input_face_count':int(len(self.original_mesh.faces)),'requested_target_face_count':int(self.config.target_faces),'actual_carrier_face_count':int(len(mesh.faces)),'max_faces_per_patch':int(self.config.max_faces_per_patch),'bounds_mm':mesh.bounds.tolist(),'surface_target_interpretation':'triangular carrier faces; not the number of patches'}
 (out/'surface_carrier_resolution_audit.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
 if info['actual_carrier_face_count']!=5000:raise RuntimeError('Actual surface carrier did not retain5000 triangles')
 return mesh
SurfacePatchBuilder._prepare_base_mesh=audited_prepare
import global_surface_contour_repair as contour_repair
original_contour_init=contour_repair.ContourRadiusRepair.__init__
original_contour_run=contour_repair.ContourRadiusRepair.run
original_contour_measure=contour_repair.ContourRadiusRepair.measure
trace_counter=0
def extended_contour_init(self,builder,*args,**kwargs):
 kwargs['max_repairs']=256
 original_contour_init(self,builder,*args,**kwargs)
def measured_contours(self):
 import numpy as np
 result,complete,bad=original_contour_measure(self)
 radius=np.asarray(result['radius'],float);data=pd.DataFrame({'component_id':np.arange(len(radius)),'complete_pore':complete,'radius_mm':radius,'area_mm2':result['area'],'exceeded':bad,'acceptance_mm':self.acceptance_mm,'excess_um':np.maximum(radius-self.acceptance_mm,0)*1000})
 output=Path(sys.argv[sys.argv.index('--out-dir')+1]);tag=f'contour_edge{self.edge_mm:.5f}_ref{self.local_refine}_limit{self.acceptance_mm:.9f}'
 data.to_csv(output/(tag+'_pores.csv'),index=False)
 return result,complete,bad
def traced_contour_run(self,*args,**kwargs):
 global trace_counter
 trace_counter+=1;number=trace_counter
 try:return original_contour_run(self,*args,**kwargs)
 finally:
  info=dict(getattr(self.builder,'contour_radius_repair_summary',{}));info.update(configured_max_repairs=self.max_repairs,effective_iteration_limit=max(self.max_repairs,512,8*info['initial_radius_exceeded_count']) if 'initial_radius_exceeded_count' in info else None,candidates_per_pore=self.candidates_per_pore,carrier_edge_mm=self.edge_mm,refinement_level=self.local_refine)
  output=Path(sys.argv[sys.argv.index('--out-dir')+1]);(output/f'contour_pass{number}_repair_summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
contour_repair.ContourRadiusRepair.__init__=extended_contour_init
contour_repair.ContourRadiusRepair.run=traced_contour_run
contour_repair.ContourRadiusRepair.measure=measured_contours
(output/'contour_repair_budget.json').write_text(json.dumps({'coarse_configured_max_repairs':256,'fine_configured_max_repairs':256,'effective_iteration_limit_rule':'max(configured_max_repairs, 512, 8 * initial_radius_exceeded_count)','physical_acceptance_unchanged':True,'core_source_modified_by_launcher':False},indent=2),encoding='utf-8')
m.main()
out=Path(sys.argv[sys.argv.index('--out-dir')+1])
e=pd.read_csv(out/'assembly_edges_final.csv')
diameter=2*pd.to_numeric(e.radius,errors='coerce')
check={'nominal_rod_diameter_mm':0.4,'min_actual_rod_diameter_mm':float(diameter.min()),'max_actual_rod_diameter_mm':float(diameter.max()),'original_surface_and_internal_rewrite_rules_retained':True,'strict_uniform_diameter_required':False,'node_spheres_are_not_rods':True}
(out/'actual_diameter_range_audit.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
