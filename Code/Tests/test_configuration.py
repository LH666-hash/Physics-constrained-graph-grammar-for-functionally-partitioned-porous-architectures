import json,sys,tempfile,unittest
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(PACKAGE/'Code/Core'),str(PACKAGE/'Code/Analysis')]
import PC_AEGG_主程序 as pc
from saved_configuration import apply_saved_configuration,pairing_fingerprint
from generate_case import command

class ConfigurationTests(unittest.TestCase):
    def test_fluid_switch_and_restore(self):
        for name,gamma,viscosity,height,diameter,count in [
            ('water25',.072,.001,25,.3,25),
            ('organic22p5',.0195,.000266,22.5,.4,23),
            ('water35',.072,.001,35,.3,35),
            ('water25',.072,.001,25,.3,25)]:
            cfg=apply_saved_configuration(pc,PACKAGE/f'Code/Configurations/{name}_reference.json')
            self.assertEqual(pc.GAMMA_NPM,gamma)
            self.assertEqual(pc.FLUID_VISCOSITY_PA_S,viscosity)
            self.assertEqual(pc.TARGET_HEIGHT_MM,height)
            self.assertEqual(pc.SURFACE_DIAM_MM,diameter)
            self.assertEqual(pc.INTERNAL_DYNAMIC_CAPILLARY_BAND_START_MM,cfg['internal_dynamic']['capillary_band_start_mm'])
            self.assertEqual(pc.PRESSURE_SLICE_COUNT,count)
            self.assertEqual(len(pc._pressure_feedback_heights((0.0,height))),count)
            for setting in ('INTERNAL_DYNAMIC_PHYSICS_DZ_MM',
                            'INTERNAL_DYNAMIC_GLOBAL_FEEDBACK_SPACING_MM',
                            'INTERNAL_DYNAMIC_UPPER_FEEDBACK_SPACING_MM'):
                self.assertEqual(getattr(pc,setting),pc.SLICE_DZ_MM)

    def test_restored_slice_thickness_updates_derived_settings(self):
        reference=PACKAGE/'Code/Configurations/water25_reference.json'
        cfg=json.loads(reference.read_text(encoding='utf-8'))
        cfg['quasi_steady_filling']['pressure_slice_nominal_thickness_mm']=2.0
        try:
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'configuration.json'
                path.write_text(json.dumps(cfg),encoding='utf-8')
                apply_saved_configuration(pc,path)
            self.assertEqual(pc.PRESSURE_SLICE_COUNT,13)
            for setting in ('INTERNAL_DYNAMIC_PHYSICS_DZ_MM',
                            'INTERNAL_DYNAMIC_GLOBAL_FEEDBACK_SPACING_MM',
                            'INTERNAL_DYNAMIC_UPPER_FEEDBACK_SPACING_MM'):
                self.assertEqual(getattr(pc,setting),2.0)
            _,widths=pc._pressure_feedback_grid((0.0,25.0))
            self.assertEqual(widths[-1],1.0)
        finally:
            apply_saved_configuration(pc,reference)

    def test_saved_seed101_profile_uses_actual_pressure_grid(self):
        apply_saved_configuration(pc,PACKAGE/'Code/Configurations/water25_reference.json')
        path=PACKAGE/'Source_Data/Ablation_Analysis/Paired_Full_Dual_Domain/paired_full_seed101/post_assembly_pressure_profile.csv'
        profile=pc.pd.read_csv(path)
        self.assertEqual(len(profile),25)
        try:
            pc.PRESSURE_SLICE_COUNT=23
            for bounds in (None,(0.0,25.0)):
                screen=pc.summarize_final_pressure_screen(profile,bounds)
                self.assertEqual(screen['expected_slice_count'],25)
                self.assertTrue(screen['pressure_grid_pass'])
                self.assertEqual(screen['lower_bound_fail_count'],0)
                self.assertEqual(screen['ordinary_requirement_fail_count'],0)
                self.assertTrue(screen['pass'])
            self.assertFalse(pc.summarize_final_pressure_screen(profile.iloc[:-1])['pass'])
            self.assertFalse(pc.summarize_final_pressure_screen(profile,(0.0,22.5))['pass'])
            self.assertEqual(pc.summarize_final_pressure_screen(None)['expected_slice_count'],25)
        finally:
            apply_saved_configuration(pc,PACKAGE/'Code/Configurations/water25_reference.json')

    def test_generation_commands_have_explicit_conditions(self):
        for case in ['water25','water35','organic22p5']:
            cmd,_=command(case,Path('new_test_output'))
            old=sys.argv
            try:
                sys.argv=[cmd[1],*cmd[2:]]
                parsed=pc.parse_cli_args()
            finally:sys.argv=old
            cfg=__import__('json').loads((PACKAGE/f'Code/Configurations/{case}_reference.json').read_text(encoding='utf-8'))
            self.assertEqual(parsed.viscosity,cfg['fluid']['viscosity_Pa_s'])
            self.assertEqual(parsed.surface_diam,cfg['surface']['diameter_mm'])
            self.assertEqual(parsed.capillary_band_start_mm,cfg['internal_dynamic']['capillary_band_start_mm'])
            self.assertEqual(parsed.height_mm,cfg['target_height_mm'])

    def test_pair_cache_includes_configuration_edges_and_source(self):
        a=PACKAGE/'Source_Data/Ablation_Analysis'
        fp=pairing_fingerprint(a/'Paired_Full_Dual_Domain/paired_full_seed101',
                              a/'内部/Internal_Fully_Random/seed_101')
        for key in ['surface_nodes','surface_edges','internal_nodes','internal_edges','configuration',
                    'surface_configuration','input_geometry','core/PC_AEGG_主程序.py','analysis/saved_configuration.py']:
            self.assertIn(key,fp)
            self.assertEqual(len(fp[key]),64)

if __name__=='__main__':unittest.main()
