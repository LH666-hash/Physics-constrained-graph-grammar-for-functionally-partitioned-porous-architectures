from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

root=Path(__file__).resolve().parents[2]/'Source_Data'/'Extended_Geometries'/'Multibranch_Free_Boundary_seed42'
out=root/'Surface_Analysis'
out.mkdir(exist_ok=True)
pores=pd.read_csv(root/'global_surface_pore_audit'/'global_complete_pores.csv')
profile=pd.read_csv(root/'post_assembly_pressure_profile.csv')
summary=json.loads((root/'global_surface_pore_audit'/'global_surface_pore_summary.json').read_text(encoding='utf-8'))
nodes=pd.read_csv(root/'assembly_nodes_final.csv').set_index('atom_id')
edges=pd.read_csv(root/'assembly_edges_final.csv')
surface_edges=edges[edges.bond_type=='surface'].copy()
surface_nodes=nodes.loc[nodes.face_name=='surface']
profile['pressure_required_lower_bound_Pa']=np.where(profile.z_mm<5,1.0,1.15)*profile.P_required_Pa
profile['pressure_lower_bound_margin_Pa']=profile.P_cap_Pa-profile.pressure_required_lower_bound_Pa

metrics={
 'global_complete_pore_count':len(pores),
 'surface_node_count':len(surface_nodes),
 'surface_edge_count':len(surface_edges),
 'mean_pore_equivalent_radius_mm':pores.equivalent_radius_mm.mean(),
 'median_pore_equivalent_radius_mm':pores.equivalent_radius_mm.median(),
 'p95_pore_equivalent_radius_mm':pores.equivalent_radius_mm.quantile(.95),
 'max_pore_equivalent_radius_mm':pores.equivalent_radius_mm.max(),
 'radius_acceptance_mm':float(summary['radius_acceptance_mm']),
 'radius_exceeded_count':int(pores.radius_exceeded.sum()),
 'global_surface_open_area_fraction':float(summary['global_liquid_open_area_fraction']),
 'min_pressure_lower_bound_margin_Pa':float(profile.pressure_lower_bound_margin_Pa.min()),
 'min_physical_transport_margin_Pa':float(profile.physical_transport_margin_Pa.min()),
}
pd.DataFrame([metrics]).to_csv(out/'surface_analysis_summary.csv',index=False)
profile[['z_mm','P_cap_Pa','P_hydro_Pa','P_visc_dynamic_Pa','P_required_Pa','pressure_required_lower_bound_Pa','pressure_lower_bound_margin_Pa','safety_margin','r_eff_mm','P_wet_mm']].to_csv(out/'surface_pressure_25_slices.csv',index=False)
surface_nodes.reset_index()[['atom_id','x','y','z']].to_csv(out/'surface_graph_nodes.csv',index=False)
surface_edges[['source','target','radius','bond_type']].to_csv(out/'surface_graph_edges.csv',index=False)

mpl.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':10.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42})
def finish(fig,stem):
    for ext in ('png','svg','pdf'):
        fig.savefig(out/(stem+'.'+ext),dpi=400,facecolor='white',bbox_inches='tight')
    plt.close(fig)
def style(ax):
    ax.tick_params(direction='in',top=True,right=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis='y',color='#E8E8EC',lw=.5)

# Complete-pore equivalent radius, using the same global reconstructed pore set as the audit.
bins=np.linspace(0,.6,25)
count,edges_bin=np.histogram(pores.equivalent_radius_mm,bins=bins)
hist=pd.DataFrame({'bin_left_mm':edges_bin[:-1], 'bin_right_mm':edges_bin[1:], 'count':count,'frequency_percent':count/len(pores)*100})
hist.to_csv(out/'surface_pore_radius_histogram.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(6.89,3.0),constrained_layout=True)
axs[0].bar(hist.bin_left_mm,hist.frequency_percent,width=np.diff(edges_bin),align='edge',color='#6DAFD6',edgecolor='white',lw=.3)
axs[0].axvline(metrics['radius_acceptance_mm'],color='#B84F63',ls='--',lw=1.3,label=r'$r_{eq}$ limit')
axs[0].set(xlabel=r'$r_{eq}$ (mm)',ylabel='Frequency per bin (%)',xlim=(0,.6))
axs[0].set_xticks([0,.15,.3,.45,.6]);axs[0].legend(frameon=False,loc='upper right',fontsize=9)
sorted_r=np.sort(pores.equivalent_radius_mm.to_numpy())
axs[1].plot(sorted_r,100*np.arange(1,len(sorted_r)+1)/len(sorted_r),color='#44749A',lw=1.8)
axs[1].axvline(metrics['radius_acceptance_mm'],color='#B84F63',ls='--',lw=1.3)
axs[1].set(xlabel=r'$r_{eq}$ (mm)',ylabel='Cumulative fraction (%)',xlim=(0,.6),ylim=(0,100))
axs[1].set_xticks([0,.15,.3,.45,.6]);axs[1].set_yticks([0,25,50,75,100])
for a,l in zip(axs,'ab'): style(a);a.text(-.12,1.04,l,transform=a.transAxes,fontweight='bold',fontsize=11)
finish(fig,'surface_pore_equivalent_radius')

# Pressure balance of the final merged structure at 25 one-millimetre slices.
fig,axs=plt.subplots(1,2,figsize=(6.89,3.0),constrained_layout=True)
z=np.r_[profile.z_mm.to_numpy(),25.0]
def to_upper(values):
    values=np.asarray(values,dtype=float)
    return np.r_[values,values[-1]]
axs[0].plot(z,to_upper(profile.P_cap_Pa),color='#44749A',lw=1.8,label=r'$P_{cap}$')
axs[0].plot(z,to_upper(profile.P_hydro_Pa+profile.P_visc_dynamic_Pa),color='#B26979',lw=1.6,label=r'$P_{hyd}+P_{visc}$')
axs[0].plot(z,to_upper(profile.pressure_required_lower_bound_Pa),color='#8C77AE',lw=1.4,ls='--',label=r'$\lambda(z)P_{req}$')
axs[0].set_ylabel('Pressure (Pa)');axs[0].legend(frameon=False,fontsize=8.6,loc='upper left')
axs[0].set_ylim(0,500);axs[0].set_yticks([0,100,200,300,400,500])
axs[1].plot(z,to_upper(profile.pressure_lower_bound_margin_Pa),color='#277D85',lw=1.8)
axs[1].axhline(0,color='#555555',lw=.7)
axs[1].set_ylabel('Lower-bound margin (Pa)')
axs[1].set_ylim(0,250);axs[1].set_yticks([0,50,100,150,200,250])
for a,l in zip(axs,'ab'):
    a.set(xlabel='Height (mm)',xlim=(0,25));a.set_xticks([0,5,10,15,20,25]);style(a)
    a.text(-.12,1.04,l,transform=a.transAxes,fontweight='bold',fontsize=11)
axs[0].set_yticks([0,100,200,300,400,500])
axs[1].set_yticks([0,50,100,150,200,250])
finish(fig,'surface_pressure_balance')

# Surface graph projection; no internal or stitch edges are included.
ids=nodes.index
xyz=nodes[['x','z']]
valid=surface_edges.source.isin(ids)&surface_edges.target.isin(ids)
e=surface_edges.loc[valid]
xy0=xyz.loc[e.source].to_numpy();xy1=xyz.loc[e.target].to_numpy()
fig,ax=plt.subplots(figsize=(6.89,3.35),constrained_layout=True)
ax.add_collection(LineCollection(np.stack([xy0,xy1],axis=1),colors='#477DA5',linewidths=.23,alpha=.42,rasterized=True))
ax.autoscale();ax.set(xlabel='x (mm)',ylabel='Height (mm)');ax.set_xticks(np.linspace(0,30,5));ax.set_yticks(np.linspace(0,25,5));style(ax)
finish(fig,'surface_graph_xz_projection')

(out/'README.txt').write_text('Final multibranch free-boundary seed-42 surface analysis. All values are computed from the final merged dual-domain graph, 25-slice pressure profile and global complete-pore audit in the parent folder. surface_analysis_summary.csv reports the final metrics. surface_graph_nodes.csv and surface_graph_edges.csv contain surface-only geometry; surface_graph_xz_projection visualizes this graph. surface_pore_radius_histogram.csv and surface_pore_equivalent_radius show the complete-pore equivalent-radius distribution. surface_pressure_25_slices.csv and surface_pressure_balance show capillary pressure, hydrostatic plus viscous pressure, the pressure requirement and the margin across height. The plotted 25 mm endpoint extends the final 24.5 mm slice value to its upper boundary; the table retains 25 computed slices. PNG files are for document placement; SVG/PDF files are editable vector figures.\n',encoding='utf-8')
print(metrics)
