"""Start a new run with explicit per-case settings; never overwrite archived results."""
import argparse,json,subprocess,sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[2]

def initialize_output_directory(out, *, allow_existing_empty=False):
    """Reserve a fresh run directory before a launcher writes any audit files."""
    out=Path(out).resolve()
    if out.is_relative_to((PACKAGE/'Source_Data').resolve()):
        raise ValueError('New generations must be outside archived Source_Data.')
    if allow_existing_empty and out.is_dir() and not any(out.iterdir()):
        return out
    out.mkdir(parents=True,exist_ok=False)
    return out

def command(case,out,input_override=None):
    path=PACKAGE/'Code/Configurations'/f'{case}_reference.json'
    c=json.loads(path.read_text(encoding='utf-8'))
    fluid=c['fluid'];surface=c['surface'];internal=c['internal_dynamic']
    stl=input_override or PACKAGE/'Input_Geometries'/dict(water25='长方体.stl',water35='box_10x10x35mm.stl',organic22p5='box_10x10x22.5mm.stl')[case]
    core=PACKAGE/'Code/Core/PC_AEGG_主程序.py'
    if case=='organic22p5':
        core=next((PACKAGE/'Source_Data/Orgnic_structure').rglob('nominal_diameter_launcher.py'))
    flags={
        'stl':stl,'height-mm':c['target_height_mm'],'out-dir':out,'build-mode':'dynamic',
        'surface-faces':surface['initial_face_count'],'surface-max-faces-per-patch':surface['max_faces_per_patch'],
        'surface-rounds':surface['requested_split_rounds'],'surface-diam':surface['diameter_mm'],
        'surface-extra-pressure-pa':surface['additional_pressure_Pa'],'surface-safety-factor':surface['capillary_safety_factor'],
        'surface-offset':surface['outward_offset_mm'],'layers':internal['layers'],'seed':internal['seed'],
        'initial-density-multiplier':internal['initial_skeleton_density_multiplier_effective'],
        'internal-diam-bottom':internal['internal_diam_bottom_mm'],'internal-diam-top':internal['internal_diam_top_mm'],
        'gamma':fluid['surface_tension_N_per_m'],'contact-angle':fluid['contact_angle_deg'],
        'rho':fluid['density_kg_per_m3'],'surface-gas-density':fluid['gas_density_kg_per_m3'],
        'gravity':fluid['gravity_m_per_s2'],'viscosity':fluid['viscosity_Pa_s'],
        'reference-flow-rate':fluid['reference_flow_rate_m3_s'],'safety-factor':fluid['safety_factor'],
        'safety-factor-bottom':fluid['safety_factor_bottom'],'safety-factor-top':fluid['safety_factor_top'],
        'slice-dz':c['quasi_steady_filling']['pressure_slice_nominal_thickness_mm'],
        'capillary-min-ratio':internal['capillary_min_ratio'],'capillary-target-ratio':internal['capillary_target_ratio'],
        'capillary-band-start-mm':internal['capillary_band_start_mm'],
        'capillary-assembly-reserve-ratio':internal['capillary_assembly_reserve_ratio'],
        'stitch-distance':c['dual_stitching']['standard_length_mm'],
        'stitch-relaxed-distance':c['dual_stitching']['relaxed_search_length_mm'],
        'stitch-anchor-spacing':c['dual_stitching']['anchor_spacing_mm'],
    }
    cmd=[sys.executable,str(core),'--surface-design']
    for key,value in flags.items():cmd.extend(['--'+key,str(value)])
    return cmd,Path(stl)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['water25','water35','organic22p5'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--input',type=Path,help='Explicit input geometry; caller is responsible for selecting the intended file.')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();cmd,stl=command(a.case,a.out,a.input)
    print(json.dumps(cmd,ensure_ascii=False,indent=2))
    if a.dry_run:return
    if not stl.is_file():raise FileNotFoundError(stl)
    initialize_output_directory(a.out)
    subprocess.run(cmd,check=True)
if __name__=='__main__':main()
