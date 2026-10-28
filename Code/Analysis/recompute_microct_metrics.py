"""Check archived micro-CT array arithmetic and component-table totals."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np

def table(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def equal(actual,expected,name):
    if not math.isclose(float(actual),float(expected),rel_tol=1e-10,abs_tol=1e-9):
        raise ValueError(f'{name}: calculated {actual}, archived {expected}')
def yes(value):return str(value).lower()=='true'
def array(path):
    value=np.load(path,mmap_mode='r',allow_pickle=False)
    if value.ndim!=3 or value.dtype!=np.dtype(bool):
        raise ValueError(f'Expected a three-dimensional Boolean array: {path}')
    return value
def run(root):
    root=root.resolve();data=root/'03_源数据表'
    results=table(data/'微CT主结果.csv')
    pores=table(data/'all_pore_components_corrected.csv')
    closed=table(data/'all_closed_components_corrected.csv')
    mapping={int(r['Seed']):r for r in table(root/'样品与文件对应表.csv')}
    if {int(r['Seed']) for r in results}!={42,101,202,303,404}:
        raise ValueError('The five expected sample identities are required.')
    output=[]
    for row in results:
        seed=int(row['Seed']);model=root/'04_模型与配准'/f'Seed{seed}'
        grid=json.loads((model/'比较网格说明.json').read_text(encoding='utf-8-sig'))
        ct=array(root/grid['CT_mask']);design=array(root/grid['design_mask'])
        if ct.shape!=design.shape or list(ct.shape)!=grid['shape_zyx']:
            raise ValueError(f'Seed{seed}: comparison grid shape mismatch.')
        nct=ndesign=overlap=0
        for z in range(0,ct.shape[0],16):
            a=ct[z:z+16];b=design[z:z+16]
            nct+=int(np.count_nonzero(a));ndesign+=int(np.count_nonzero(b))
            overlap+=int(np.count_nonzero(a&b))
        if nct+ndesign==0:raise ValueError('Empty comparison masks.')
        dice=2*overlap/(nct+ndesign)
        equal(dice,row['Rigid_physical_Dice'],f'Seed{seed} solid Dice')
        equal(dice,grid['Rigid_physical_Dice'],f'Seed{seed} grid Dice')
        native=root/'06_原生体数据'/f'Seed{seed}'
        metadata=json.loads((native/'metadata.json').read_text(encoding='utf-8-sig'))
        solid=array(root/metadata['native_solid']);roi=array(root/metadata['native_ROI'])
        if solid.shape!=roi.shape:raise ValueError(f'Seed{seed}: native array shape mismatch.')
        invalid=set(metadata.get('invalid_source_slices_1based',[]))
        origin_z=int(metadata['crop_origin_voxel_xyz'][2])
        roi_vox=void_vox=valid_roi=valid_void=0
        for z in range(0,solid.shape[0],16):
            region=roi[z:z+16];void=region&~solid[z:z+16]
            per_roi=np.count_nonzero(region,axis=(1,2));per_void=np.count_nonzero(void,axis=(1,2))
            keep=np.array([origin_z+local+1 not in invalid for local in range(z,z+len(region))])
            roi_vox+=int(per_roi.sum());void_vox+=int(per_void.sum())
            valid_roi+=int(per_roi[keep].sum());valid_void+=int(per_void[keep].sum())
        if not valid_roi or not roi_vox:raise ValueError(f'Seed{seed}: empty analysis region.')
        voxel_volume=float(metadata['voxel_mm'])**3
        porosity=100*void_vox/roi_vox;valid_porosity=100*valid_void/valid_roi
        equal(porosity,row['Porosity_full_continuous_volume_percent'],f'Seed{seed} continuous porosity')
        equal(valid_porosity,row['Porosity_valid_original_slices_percent'],f'Seed{seed} valid-slice porosity')
        equal(roi_vox*voxel_volume,row['ROI_volume_mm3'],f'Seed{seed} ROI volume')
        equal(void_vox*voxel_volume,row['Void_volume_mm3'],f'Seed{seed} void volume')
        components=[r for r in pores if int(r['Seed'])==seed]
        enclosed=[r for r in closed if int(r['Seed'])==seed]
        if sum(int(r['voxels']) for r in components)!=void_vox:
            raise ValueError(f'Seed{seed}: component voxels do not cover the native void volume.')
        component_closed=[r for r in components if yes(r['closed'])]
        if {r['label'] for r in component_closed}!={r['label'] for r in enclosed}:
            raise ValueError(f'Seed{seed}: closed-component labels differ between tables.')
        if len(enclosed)!=int(row['All_closed_count']):raise ValueError(f'Seed{seed}: closed-component count mismatch.')
        if any(int(r['voxels'])<1 or not yes(r['primary_included']) for r in enclosed):
            raise ValueError(f'Seed{seed}: closed-component inclusion mismatch.')
        closed_voxels=sum(int(r['voxels']) for r in enclosed)
        if closed_voxels!=int(row['All_closed_voxels']):raise ValueError(f'Seed{seed}: closed voxel sum mismatch.')
        equal(closed_voxels*voxel_volume,row['All_closed_volume_mm3'],f'Seed{seed} closed volume')
        equal(100*closed_voxels/roi_vox,row['All_closed_percent_ROI'],f'Seed{seed} closed/ROI percent')
        equal(100*closed_voxels/void_vox,row['All_closed_percent_all_void'],f'Seed{seed} closed/void percent')
        for component in components+enclosed:
            equal(int(component['voxels'])*voxel_volume,component['volume_mm3'],f'Seed{seed} component volume')
        spanning=[r for r in components if yes(r['double_end_spanning'])]
        if len(spanning)!=int(row['Upward_spanning_components']):raise ValueError(f'Seed{seed}: spanning count mismatch.')
        largest=max(int(r['voxels']) for r in spanning)
        equal(100*largest/void_vox,row['Largest_upward_spanning_percent_all_void'],f'Seed{seed} largest spanning percentage')
        equal(largest*voxel_volume,row['Largest_upward_spanning_volume_mm3'],f'Seed{seed} largest spanning volume')
        boundary=sum(int(r['voxels']) for r in components if yes(r['boundary_open']))
        equal(100*boundary/void_vox,row['Boundary_open_percent_all_void'],f'Seed{seed} boundary-open percentage')
        equal(100*(boundary-largest)/void_vox,row['Other_boundary_open_percent_all_void'],f'Seed{seed} other open percentage')
        if metadata['source']!=Path(mapping[seed]['原始TXM']).name:raise ValueError(f'Seed{seed}: scan identity mismatch.')
        for key in ('原生分割','分析区'):
            if not (root/mapping[seed][key]).is_file():raise FileNotFoundError(root/mapping[seed][key])
        output.append(dict(seed=seed,solid_Dice=dice,continuous_porosity_percent=porosity,
                           valid_original_slice_porosity_percent=valid_porosity,closed_components=len(enclosed),
                           closed_voxels=closed_voxels,roi_voxels=roi_vox,void_voxels=void_vox,pass_all=True))
    return {'samples':output,'all_passed':True,
            'inputs':'Saved segmentations, aligned comparison arrays and archived component tables.',
            'scope':'Array arithmetic and component-table consistency; scan reconstruction, registration and component labelling are earlier saved processing stages.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ct-root',type=Path,default=Path('Micro_CT'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.ct_root)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'microct_recomputed_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified saved-array Dice, porosity and component-table totals for five CT samples.')
if __name__=='__main__':main()
