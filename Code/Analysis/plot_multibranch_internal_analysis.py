from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

root=Path(__file__).resolve().parents[2]/'Source_Data'/'Extended_Geometries'/'Multibranch_Free_Boundary_seed42'
out=root/'Internal_Analysis'
out.mkdir(exist_ok=True)
nodes=pd.read_csv(root/'assembly_nodes_final.csv').set_index('atom_id')
edges=pd.read_csv(root/'assembly_edges_final.csv')
support_nodes=pd.read_csv(root/'internal_validation'/'internal_print_support_nodes.csv')
support_edges=pd.read_csv(root/'internal_validation'/'internal_print_support_edges.csv')
support=json.loads((root/'internal_validation'/'internal_print_support_summary.json').read_text(encoding='utf-8'))
abnormal=json.loads((root/'internal_validation'/'abnormal_layer_connectivity.json').read_text(encoding='utf-8'))
profile=pd.read_csv(root/'post_assembly_pressure_profile.csv')
internal=nodes[nodes.face_name!='surface'].copy()
int_edges=edges[edges.bond_type=='internal'].copy()
assert len(internal)==support['internal_node_count']==len(support_nodes)
assert len(int_edges)==support['internal_edge_count']==len(support_edges)

node_table=internal.reset_index()[['atom_id','x','y','z','node_role']].merge(support_nodes,on='atom_id',how='left',validate='one_to_one')
node_table.to_csv(out/'internal_nodes_with_support.csv',index=False)
edge_table=int_edges[['source','target','radius','role']].merge(support_edges,on=['source','target'],how='left',validate='one_to_one',suffixes=('_assembly','_support'))
assert edge_table.length_mm.notna().all()
edge_table.to_csv(out/'internal_edges_with_geometry.csv',index=False)
layers=np.arange(0,26)
node_table['height_layer_mm']=np.clip(np.floor(node_table.z).astype(int),0,24)
by_height=node_table.groupby('height_layer_mm').agg(node_count=('atom_id','size'),mean_internal_degree=('internal_degree','mean'),unsupported_count=('downward_supported',lambda x:int((~x.astype(bool)).sum()))).reindex(range(25),fill_value=0).reset_index()
by_height.to_csv(out/'internal_nodes_by_height.csv',index=False)
classes=edge_table.manufacturing_screen_class.value_counts().rename_axis('edge_class').reset_index(name='count')
classes.to_csv(out/'internal_edge_class_counts.csv',index=False)
profile[['z_mm','porosity','r_eff_mm','P_wet_mm','A_domain_mm2','A_solid_mm2','A_liq_mm2','n_flow_components']].to_csv(out/'assembled_internal_section_25_slices.csv',index=False)
summary={
 'internal_node_count':len(node_table),
 'internal_edge_count':len(edge_table),
 'mean_internal_degree':node_table.internal_degree.mean(),
 'median_internal_degree':node_table.internal_degree.median(),
 'median_internal_edge_length_mm':edge_table.length_mm.median(),
 'p95_internal_edge_length_mm':edge_table.length_mm.quantile(.95),
 'isolated_internal_node_count':int(support['isolated_internal_node_count']),
 'downward_unsupported_node_count':int(support['downward_unsupported_node_count']),
 'root_disconnected_node_count':int(support['root_disconnected_node_count']),
 'internal_support_check_pass':bool(support['print_support_screen_pass']),
 'abnormal_layer_connectivity_pass':bool(abnormal['abnormal_layer_connectivity_pass']),
 'minimum_final_section_porosity':profile.porosity.min(),
 'maximum_final_section_porosity':profile.porosity.max(),
 'minimum_final_effective_radius_mm':profile.r_eff_mm.min(),
 'maximum_final_effective_radius_mm':profile.r_eff_mm.max(),
}
pd.DataFrame([summary]).to_csv(out/'internal_analysis_summary.csv',index=False)

mpl.rcParams.update({'font.family':'Arial','font.size':10,'axes.labelsize':10.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42})
def style(ax):
    ax.tick_params(direction='in',top=True,right=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis='y',color='#E8E8EC',lw=.5)
def save(fig,stem):
    for ext in ('png','svg','pdf'):
        fig.savefig(out/(stem+'.'+ext),dpi=400,facecolor='white',bbox_inches='tight')
    plt.close(fig)

# Final internal graph, excluding surface and stitch bonds.
loc=nodes[['x','z']]
seg=np.stack([loc.loc[int_edges.source].to_numpy(),loc.loc[int_edges.target].to_numpy()],axis=1)
fig,ax=plt.subplots(figsize=(6.89,3.3),constrained_layout=True)
ax.add_collection(LineCollection(seg,colors='#547FA6',linewidths=.55,alpha=.35,rasterized=True))
ax.scatter(internal.x,internal.z,s=1.5,c='#365E81',alpha=.5,rasterized=True)
ax.autoscale();ax.set(xlabel='x (mm)',ylabel='Height (mm)',xlim=(0,30),ylim=(0,25))
ax.set_xticks([0,7.5,15,22.5,30]);ax.set_yticks([0,6,12,18,25]);style(ax)
save(fig,'internal_graph_xz_projection')

# Node occupancy by height and final edge-length distribution.
fig,axs=plt.subplots(1,2,figsize=(6.89,3.0),constrained_layout=True)
axs[0].bar(by_height.height_layer_mm+.5,by_height.node_count,width=.85,color='#6DAFD6')
axs[0].set(xlabel='Height (mm)',ylabel='Internal nodes per 1 mm',xlim=(0,25))
axs[0].set_xticks([0,6,12,18,25])
lengths=edge_table.length_mm.to_numpy()
edges_bin=np.linspace(0,np.ceil(lengths.max()*2)/2,21)
counts,bins=np.histogram(lengths,bins=edges_bin)
hist=pd.DataFrame({'bin_left_mm':bins[:-1],'bin_right_mm':bins[1:],'count':counts,'frequency_percent':counts/len(lengths)*100})
hist.to_csv(out/'internal_edge_length_histogram.csv',index=False)
axs[1].bar(hist.bin_left_mm,hist.frequency_percent,width=np.diff(bins),align='edge',color='#B3A7CF',edgecolor='white',lw=.3)
axs[1].set(xlabel='Internal edge length (mm)',ylabel='Frequency per bin (%)',xlim=(0,bins[-1]))
axs[1].xaxis.set_major_locator(MaxNLocator(nbins=4))
for ax,label in zip(axs,'ab'):
    style(ax);ax.text(-.12,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=11)
save(fig,'internal_nodes_and_edge_lengths')

# Cross-sectional liquid space and capillary length scale after final assembly.
fig,axs=plt.subplots(1,2,figsize=(6.89,3.0),constrained_layout=True)
z=profile.z_mm.to_numpy()
axs[0].plot(z,profile.porosity*100,color='#447FA7',lw=1.8)
axs[0].set_ylabel('Section porosity (%)')
axs[1].plot(z,profile.r_eff_mm,color='#AD6D8F',lw=1.8)
axs[1].set_ylabel(r'$r_{eff}$ (mm)')
for ax,label in zip(axs,'ab'):
    ax.set(xlabel='Height (mm)',xlim=(0,25));ax.set_xticks([0,6,12,18,25]);style(ax)
    ax.text(-.12,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=11)
save(fig,'internal_section_porosity_effective_radius')

(out/'README.txt').write_text('Final multibranch free-boundary seed-42 internal analysis. internal_nodes_with_support.csv and internal_edges_with_geometry.csv are extracted from the final merged graph and its support check. internal_analysis_summary.csv, internal_nodes_by_height.csv, internal_edge_class_counts.csv and internal_edge_length_histogram.csv contain the plotted statistics. internal_graph_xz_projection and internal_nodes_and_edge_lengths visualize the final internal solid graph. assembled_internal_section_25_slices.csv and internal_section_porosity_effective_radius show cross-sectional liquid-space state after surface and internal domains are assembled; these section values describe the complete assembled structure. PNG files support document placement; SVG and PDF files are vector figures.\n',encoding='utf-8')
print(summary)
