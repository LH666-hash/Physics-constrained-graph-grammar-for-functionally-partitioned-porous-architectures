"""Recompute archived pressure and ICP arithmetic without generating graphs or figures."""
import argparse,csv,json,hashlib,statistics
from pathlib import Path
import openpyxl

PACKAGE=Path(__file__).resolve().parents[2]
WEIGHTS={'U':238.02891,'Th':232.0377,'Yb':173.045,'La':138.90547}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csv_rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def icp():
    path=PACKAGE/'Source_Data/ICP/Multi-ion sequential experiment.xlsx'
    ws=openpyxl.load_workbook(path,data_only=True).worksheets[0]
    groups={};sample=None
    for row in list(ws.values)[28:]:
        if row[0] in ('Aq','N1923','TOA','C272'):sample=row[0]
        element=row[3]
        if sample and element in WEIGHTS:
            # Sheet column E is instrument mg/L, F is the dilution factor.
            try:value=float(row[4])*float(row[5])
            except (TypeError,ValueError):value=None
            groups.setdefault((sample,element),[]).append(value)
    result=[]
    for (sample,element),values in groups.items():
        assert len(values)==3,(sample,element,len(values))
        valid=[x for x in values if x is not None]
        mean=statistics.mean(valid) if valid else None
        result.append(dict(sample=sample,element=element,technical_reads=3,
                           corrected_reads_mg_L=values,mean_mg_L=mean,
                           mean_mol_L=mean/(1000*WEIGHTS[element]) if mean is not None else None,
                           status='quantified' if len(valid)==3 else 'not_detected'))
    thpath=PACKAGE/'Source_Data/ICP/Thorium ion extraction.xlsx'
    ws=openpyxl.load_workbook(thpath,data_only=True).worksheets[0]
    values=[]
    for row in list(ws.values)[28:]:
        if row[3]=='Th':
            try:values.append(float(row[4]))
            except (ValueError,TypeError):pass
    assert len(values)==3,values
    concentration=statistics.mean(values)*10/3
    single=dict(instrument_reads_mg_L=values,dilution_factor=10/3,
                corrected_mean_mg_L=concentration,
                corrected_mean_mol_L=concentration/(1000*WEIGHTS['Th']),
                concentration_reduction_percent=(1-concentration/(1000*WEIGHTS['Th'])/5e-4)*100,
                replicate_unit='three instrument reads of one sample',source_sha256=digest(thpath))
    la=next(x for x in result if x['sample']=='Aq' and x['element']=='La')
    return dict(multi_ion=result,multi_ion_source_sha256=digest(path),
                la_final_to_initial_concentration_percent=100*la['mean_mol_L']/5e-4,
                th_single_endpoint=single)

def pressure():
    base=PACKAGE/'Source_Data/Ablation_Analysis'
    records=[]
    for seed in (101,202,303,404,505):
        full=base/f'Paired_Full_Dual_Domain/paired_full_seed{seed}'
        paths=[('Full PC-AEGG',full)]+[(g,base/f'Paired_Control_Dual_Domain/{folder}/seed_{seed}') for g,folder in
                [('Fully random','fully_random'),('Initial gradient skeleton','initial_gradient_skeleton'),('Matched random position','random_position_fixed_budget')]]
        config_path=full/'run_config.json';cfg=json.loads(config_path.read_text(encoding='utf-8'))
        for group,path in paths:
            p=path/'post_assembly_pressure_profile.csv';rows=csv_rows(p);assert len(rows)==25
            margins=[];reserves=[]
            for row in rows:
                cap=float(row['P_cap_Pa']);hyd=float(row['P_hydro_Pa']);vis=float(row['P_visc_ref_Pa']);sf=float(row['safety_factor_z'])
                required=sf*(hyd+vis);z=float(row['z_mid_mm'])
                margin=cap-required;margins.append(margin)
                ratio=1.0 if z<cfg['internal_dynamic']['capillary_band_start_mm'] else cfg['internal_dynamic']['capillary_min_ratio']
                reserves.append(cap-ratio*required)
                assert abs(margin-float(row['physical_transport_margin_Pa']))<1e-7
            records.append(dict(group=group,seed=seed,minimum_transport_margin_Pa=min(margins),minimum_final_reserve_Pa=min(reserves),
                                profile_sha256=digest(p),configuration_sha256=digest(config_path)))
    summaries=[]
    for group in sorted(set(x['group'] for x in records)):
        vals=[x['minimum_transport_margin_Pa'] for x in records if x['group']==group]
        summaries.append(dict(group=group,n=5,mean_Pa=statistics.mean(vals),sample_sd_Pa=statistics.stdev(vals)))
    return dict(runs=records,group_summary=summaries)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='New analysis output directory; archived inputs are read only.')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=dict(pressure=pressure(),icp=icp())
    (args.out/'recomputed_results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified 20 archived pressure profiles and ICP arithmetic; no figure files changed.')
if __name__=='__main__':main()
