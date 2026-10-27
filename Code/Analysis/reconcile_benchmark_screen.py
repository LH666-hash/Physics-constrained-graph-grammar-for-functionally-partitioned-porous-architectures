"""Reconcile the archived internal search against the assembled PC-AEGG mean."""
from pathlib import Path
import hashlib
import json

import pandas as pd

PKG = Path(__file__).resolve().parents[2]
ROOT = PKG / "Source_Data/Benchmark_Analysis"
INT = ROOT / "内部"
SUR = ROOT / "表面"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pre_path = INT / "01_预设内部结构_全部45220.csv"
    gre_path = INT / "02_局部随机贪心_全部6000次评价_5988个不同结构.csv"
    pc_path = INT / "07_PC-AEGG完整组5次终态内部指标.csv"
    preset = pd.read_csv(pre_path)
    greedy_raw = pd.read_csv(gre_path)
    greedy = greedy_raw.drop_duplicates("geometry_sha1", keep="first")
    pc = pd.read_csv(pc_path)
    assert (len(preset), len(greedy)) == (45220, 5988)
    mean = float(pc.epsilon.mean())
    goal = float(pc.predicted_rise_height_mm.mean())
    eligible = []
    for group, table in (("preset", preset), ("local_greedy", greedy)):
        subset = table.loc[table.feasible.astype(bool) & table.epsilon.ge(mean) & table.predicted_rise_height_mm.ge(goal)].copy()
        subset.insert(0, "search_group", group)
        eligible.append(subset)
    combined = pd.concat(eligible, ignore_index=True)
    combined.to_csv(INT / "internally_eligible_baseline_structures.csv", index=False)
    summary_path = SUR / "benchmark_update_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stage = summary["internal_stage"]
    stage.pop("pc_internal_mean_porosity", None)  # stale preassembly reference, not the assembled threshold
    stage.pop("pc_internal_mean_rise_height_mm", None)
    stage.update(pc_assembled_mean_porosity=mean, pc_mean_rise_height_mm=goal,
                 unique_preset=len(preset), unique_local_greedy=len(greedy),
                 eligible_preset=len(eligible[0]), eligible_local_greedy=len(eligible[1]),
                 criterion="Archived internal-search feasible and epsilon >= five-run assembled PC-AEGG mean and predicted rise >= 25 mm; preassembly search scope only.",
                 source_sha256={p.name: sha(p) for p in (pre_path, gre_path, pc_path)})
    natural = pd.read_csv(ROOT / "Natural_Surface_Comparison/all_241_internal_eligible_natural_surfaces.csv")
    summary["natural_surface_stage"] = {
        "method": "Natural outer boundary of each reconstructed original Diamond structure; 0.125-mm isotropic surface raster; six faces connected across cube edges; equivalent radius 2A/P",
        "original_search_reconstruction_matches": int(natural.original_reconstruction_match.sum()),
        "current_physics_internal_screen_pass": int(natural.current_internal_screen_pass.sum()),
        "eligible_preset_natural_surface_reaching_both_pc_means": int(natural.query("search_group == 'preset'").reaches_both_pc_surface_means.sum()),
        "eligible_greedy_natural_surface_reaching_both_pc_means": int(natural.query("search_group == 'local_greedy'").reaches_both_pc_surface_means.sum()),
        "source": "Source_Data/Benchmark_Analysis/Natural_Surface_Comparison/all_241_internal_eligible_natural_surfaces.csv"}
    summary["comparison_scope"] = {
        "preset": "Its own natural external boundary, no replacement skin",
        "local_greedy": "Each internally eligible structure's own natural external boundary, no slit assembly",
        "regular_slit": "Independent surface-only reference; no internal pairing",
        "pc_aegg": "Five same-seed full-group paired assemblies for internal reference; their full-group surface series assessed separately for the surface reference"}
    summary["interpretation"] = "At the corrected 0.6074781975 internal-search threshold, 1 preset and 240 geometrically distinct local-greedy Diamond designs meet the internal feasibility, porosity and 25-mm height screen. Their natural outer surfaces all reach both PC-AEGG surface means under the stated 0.125-mm raster assessment. Separately, 198 of 8212 regular-slit surfaces reach both surface means. These are separate-domain comparisons and do not establish a baseline dual-domain acceptance pass."
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PC mean {mean:.10f}; internal-search eligible preset={len(eligible[0])}, greedy={len(eligible[1])}")


if __name__ == "__main__":
    main()
