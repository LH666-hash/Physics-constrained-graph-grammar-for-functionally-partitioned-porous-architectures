"""Export the project analysis tables from the validated multibranch assembly."""
from pathlib import Path
import importlib.util
import sys
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Source_Data/Extended_Geometries/Multibranch_Free_Boundary_seed42'
sys.path.insert(0, str(ROOT / 'Code/Core'))
spec = importlib.util.spec_from_file_location('pc_aegg_main', ROOT / 'Code/Core/PC_AEGG_主程序.py')
pc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pc
spec.loader.exec_module(pc)
mpl.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
original_save = pc._save_svg_only
def save_all(fig, path):
    is_pressure = Path(path).stem == 'plot_forces'
    for axis in fig.axes:
        if axis.name == '3d' or axis.get_label() == '<colorbar>':
            continue
        if is_pressure and axis.get_xscale() == 'linear':
            axis.set_xlim(0, 25)
            axis.set_xticks([0, 5, 10, 15, 20, 25])
        elif axis.get_xscale() == 'linear':
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if is_pressure and axis.get_yscale() == 'linear':
            axis.set_ylim(0, 500)
            axis.set_yticks([0, 100, 200, 300, 400, 500])
        elif axis.get_yscale() == 'linear':
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        legend = axis.get_legend()
        if legend:
            for label in legend.get_texts():
                label.set_text(label.get_text().replace(r'\mathrm{hydro}',
                                                       r'\mathrm{hyd}'))
    result = original_save(fig, path)
    target = Path(path)
    fig.savefig(target.with_suffix('.png'), dpi=400, bbox_inches='tight', facecolor='white')
    fig.savefig(target.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    return result
pc._save_svg_only = save_all
mesh, _ = pc.load_and_scale_mesh(ROOT / 'Input_Geometries/knitted_elephant.stl', 25.0)
nodes = pd.read_csv(OUT / 'assembly_nodes_final.csv')
edges = pd.read_csv(OUT / 'assembly_edges_final.csv')
profile = pd.read_csv(OUT / 'post_assembly_pressure_profile.csv')
pc.export_dynamic_analysis_outputs(mesh, nodes, edges,
    pc._make_nodes_np_for_analysis(nodes, (0.0, 25.0)), profile, OUT)
print('Project analysis tables and figures exported')
