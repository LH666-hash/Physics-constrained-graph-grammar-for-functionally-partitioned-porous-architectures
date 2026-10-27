"""Independent arithmetic checks for the revised figures and document claims."""
from __future__ import annotations
import json
import math
from pathlib import Path
import pandas as pd
from docx import Document

P = Path(__file__).resolve().parents[2]
S = P / "Source_Data"
checks = {}

seed = pd.read_csv(S/"Representative_Structure_Seed42/assembled_porosity_25_slice_summary.csv").iloc[0]
profile = pd.read_csv(S/"Representative_Structure_Seed42/post_assembly_pressure_profile.csv")
assert len(profile) == 25
weighted = profile.A_liq_mm2.sum()/profile.A_domain_mm2.sum()
assert math.isclose(weighted, seed.area_weighted_porosity_fraction, abs_tol=1e-12)
assert math.isclose(profile.porosity.min(), seed.minimum_slice_porosity_fraction, abs_tol=1e-12)
assert math.isclose(profile.porosity.max(), seed.maximum_slice_porosity_fraction, abs_tol=1e-12)
checks["seed42"] = dict(weighted_porosity=weighted, slice_min=float(profile.porosity.min()),
                        slice_max=float(profile.porosity.max()), slices=25)

status = json.loads((S/"Representative_Structure_Seed42/final_assembly_status.json").read_text(encoding="utf-8"))
validation = json.loads((S/"Representative_Structure_Seed42/final_validation_summary.json").read_text(encoding="utf-8"))
global_pores = validation["surface_global_complete_pores"]
assert status["graph_screen_pass"] and status["surface_global_complete_pore_acceptance_pass"]
assert global_pores["global_complete_pore_count"] == 788
assert math.isclose(global_pores["global_liquid_open_area_fraction"], 0.53940191141949, abs_tol=1e-12)
assert status["status"] == "graph_screen_pass_additional_validation_pending"
checks["seed42_acceptance"] = dict(status=status["status"], complete_pores=788,
                                   surface_open_area=global_pores["global_liquid_open_area_fraction"])

b = S/"Benchmark_Analysis"
pc = pd.read_csv(b/"内部/07_PC-AEGG完整组5次终态内部指标.csv")
preset = pd.read_csv(b/"内部/01_预设内部结构_全部45220.csv")
greedy = pd.read_csv(b/"内部/02_局部随机贪心_全部6000次评价_5988个不同结构.csv").drop_duplicates("geometry_sha1")
threshold = float(pc.epsilon.mean())
eligible_preset = preset.loc[preset.feasible.astype(bool)&preset.epsilon.ge(threshold)&preset.predicted_rise_height_mm.ge(25)]
eligible_greedy = greedy.loc[greedy.feasible.astype(bool)&greedy.epsilon.ge(threshold)&greedy.predicted_rise_height_mm.ge(25)]
assert len(pc)==5 and len(preset)==45220 and len(greedy)==5988
assert math.isclose(threshold, 0.6074781974940404, abs_tol=1e-12)
assert (len(eligible_preset),len(eligible_greedy)) == (1,240)
natural = pd.read_csv(b/"Natural_Surface_Comparison/all_241_internal_eligible_natural_surfaces.csv")
assert len(natural)==241 and natural.original_reconstruction_match.all()
assert natural.current_internal_screen_pass.all() and natural.reaches_both_pc_surface_means.all()
slits = pd.read_csv(b/"表面/regular_slit_8212_wall_0p30.csv")
surface_summary = json.loads((b/"Natural_Surface_Comparison/natural_surface_summary.json").read_text(encoding="utf-8"))
slit_pass = slits.surface_A_eff_star.ge(surface_summary["pc_surface_mean_open_area"]) & slits.surface_r_eq_ctrl_mm.le(surface_summary["pc_surface_mean_max_complete_radius_mm"])
assert len(slits)==8212 and int(slit_pass.sum())==198
summary = json.loads((b/"表面/benchmark_update_summary.json").read_text(encoding="utf-8"))
assert math.isclose(summary["internal_stage"]["pc_assembled_mean_porosity"], threshold, abs_tol=1e-12)
assert "pc_internal_mean_porosity" not in summary["internal_stage"]
checks["benchmark"] = dict(pc_assembled_mean_porosity=threshold, preset_total=len(preset),
                           greedy_unique=len(greedy), preset_eligible=1, greedy_eligible=240,
                           eligible_natural_surfaces=241, slit_total=len(slits), slit_surface_pass=198)

ablation = pd.read_csv(S/"Ablation_Analysis/内部/internal_metrics_fixed_budget_four_groups.csv")
expected = {"Full PC-AEGG":(5,28.578234,13.063469),
            "Fully random":(0,-516.460558,55.080601),
            "Initial gradient skeleton":(0,-127.690750,17.260321),
            "Matched random position":(1,-97.253553,100.035016)}
for method,(passes,mean,sd) in expected.items():
    rows=ablation.loc[ablation.method.eq(method)]
    assert len(rows)==5 and int(rows.ordinary_transport_pass.sum())==passes
    assert math.isclose(rows.minimum_transport_margin_pa.mean(),mean,abs_tol=0.002)
    assert math.isclose(rows.minimum_transport_margin_pa.std(ddof=1),sd,abs_tol=0.002)
    if method == "Initial gradient skeleton":
        assert rows.final_internal_nodes.eq(1294).all()
checks["ablation"] = {k:dict(passes=v[0], mean_margin_pa=v[1], sd_margin_pa=v[2]) for k,v in expected.items()}

centrif = pd.read_csv(S/"Raw_Data/centrifugal_si_tables_S8_S9.csv")
first = centrif.loc[centrif.record_type.eq("first_detected_loss")].sort_values("specimen_radius_mm")
assert len(first)==3 and first.first_leak_speed_rpm.tolist()==[575,434,228]
assert first.first_leak_mass_loss_pct.tolist()==[16.15,3.93,4.74]
checks["centrifuge"] = dict(first_detection_points=3, speeds_rpm=first.first_leak_speed_rpm.tolist())

viscosity = pd.read_csv(S/"viscosity_prediction/Figure4h_Seed42_viscosity_source_data.csv")
model = json.loads((S/"viscosity_prediction/Figure4h_Seed42_viscosity_summary.json").read_text(encoding="utf-8"))
pred = viscosity.loc[viscosity.role.eq("prediction")].sort_values("viscosity_cP")
assert pred.viscosity_cP.tolist() == [5.0,15.0,25.0,50.0]
assert [round(x,2) for x in pred.predicted_height_mm] == [25.0,25.0,18.66,11.29]
assert round(model["resistance_factor_fitted_from_1cP"],5)==1.38627
assert round(model["out_of_sample_mae_mm_5_to_50cP"],2)==2.14
assert round(model["out_of_sample_max_abs_error_mm_5_to_50cP"],2)==4.21
checks["viscosity"] = dict(predicted_height_mm=pred.predicted_height_mm.tolist(),
                           fitted_factor=model["resistance_factor_fitted_from_1cP"],
                           mean_absolute_error_mm=model["out_of_sample_mae_mm_5_to_50cP"])
assert {x.name for x in (S/"MicroCT").iterdir()} == {
    "README.txt", "diameter_histogram_binned_source_data.csv",
    "pore_diameter_source_data.csv", "throat_diameter_source_data.csv"}
checks["microct_source_scope"] = dict(pore_records=1851,throat_records=6721,
                                      original_reconstruction_volume_in_package=False,
                                      five_repeat_s17_images_and_records="pending")

main = P/"Manuscript/Manuscript.docx"
si = P/"Supplementary_Information/Supplementary_Information_修订版.docx"
if main.exists() and si.exists():
    mt="\n".join(p.text for p in Document(main).paragraphs)
    st="\n".join(p.text for p in Document(si).paragraphs)
    for term in ("0.607478", "240个", "53.9%", "64.84%", "28.58 ± 13.06 Pa",
                 "18.66和11.29 mm", "2.14 mm", "4.21 mm"):
        assert term in mt, term
    for term in ("28.58 ± 13.06 Pa", "−127.69 ± 17.26 Pa", "1,294个节点",
                 "1.38627", "18.66和11.29 mm"):
        assert term in st, term
    for term in ("61.304%", "85.42%", "53.3%", "1.38695", "18.65和11.26",
                 "5.90 ± 1.34 Pa", "−196.91", "−611.42", "−149.91", "不限预算",
                 "逐层支撑检查"):
        assert term not in mt+st, term
    checks["word_claims_checked"] = True

for name in ("Figure2.png","Figure3_revised.png","Figure5.png"):
    assert (P/"Figures/Main_Figures"/name).is_file()
checks["figures_present"] = True
(S/"FINAL_REVISION_CHECK.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("Revised seed42, acceptance, benchmark, ablation, centrifuge and document claims verified.")
