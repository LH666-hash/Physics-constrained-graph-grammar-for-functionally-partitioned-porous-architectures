"""Index final structure outputs and verify their minimum deliverables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parents[2]
SRC = PKG / "Source_Data"
ABL = SRC / "Ablation_Analysis"
SEEDS = (101, 202, 303, 404, 505)


def readj(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def item(path: Path, purpose: str, group: str, seed: int, min_stl: int, min_svg: int,
         require_pass: bool = False) -> dict:
    assert path.is_dir(), path
    stls = list(path.rglob("*.stl"))
    svgs = list(path.rglob("*.svg"))
    assert len(stls) >= min_stl and len(svgs) >= min_svg, (path, len(stls), len(svgs))
    status = readj(path / "final_assembly_status.json")
    paired_control = readj(path / "final_pairing_status.json")
    if require_pass:
        assert status.get("graph_screen_pass") is True, path
    provenance = readj(path / "generation_provenance.json")
    hashes = provenance.get("core_code_sha256", {})
    generation_surface_hash = hashes.get("surface_patch_builder.py", "") if isinstance(hashes, dict) else ""
    if paired_control:
        generation_surface_hash = paired_control.get("surface_builder_code_sha256", "")
    return dict(purpose=purpose, group=group, seed=seed, directory=path.relative_to(PKG).as_posix(),
                structure_stl_count=len(stls), analysis_svg_count=len(svgs),
                final_graph_screen_pass=status.get("graph_screen_pass", "not_a_paired_structure"),
                paired_control_pressure_screen_pass=(paired_control.get("pressure_screen", {}).get("pass", "")
                                                     if paired_control else ""),
                paired_control_dual_domain_pass=(paired_control.get("dual_domain_audit", {}).get("dual_domain_pass", "")
                                                 if paired_control else ""),
                generation_surface_code_sha256=generation_surface_hash)


def main() -> None:
    rows = []
    for seed in SEEDS:
        rows.append(item(ABL / "Paired_Full_Dual_Domain" / f"paired_full_seed{seed}",
                         "paired full dual-domain output", "paired_full", seed, 3, 1, True))
        for group in ("full", "random_pore", "none"):
            rows.append(item(ABL / "表面/Surface_Groups" / f"surface_{group}_seed{seed}",
                             "standalone surface ablation output", f"surface_{group}", seed, 1, 1))
        for group, folder, name in (
            ("full", "Internal_Full_Runs", f"seed_{seed}"),
            ("fully_random", "Internal_Fully_Random", f"seed_{seed}"),
            ("initial_gradient_skeleton", "Internal_Initial_Gradient_Skeleton", f"seed_{seed}"),
            ("random_position_fixed_budget", "Internal_Random_Position_Fixed_Budget",
             f"internal_random_position_fixed_budget_seed{seed}"),
        ):
            rows.append(item(ABL / "内部" / folder / name,
                             "standalone internal ablation output", f"internal_{group}", seed, 1, 1))
    assert len(rows) == 40
    for group in ("random_position_fixed_budget", "initial_gradient_skeleton", "fully_random"):
        for seed in SEEDS:
            rows.append(item(ABL / "Paired_Control_Dual_Domain" / group / f"seed_{seed}",
                             "same-seed full surface paired with internal ablation control",
                             f"paired_control_{group}", seed, 3, 1))
    rows.append(item(SRC / "Representative_Structure_Seed42", "representative dual-domain output",
                     "representative", 42, 3, 1, True))
    for path in sorted((SRC / "Extended_Geometries").iterdir()):
        if path.is_dir():
            rows.append(item(path, "extended dual-domain output", path.name, 42, 3, 1, True))
    assert len(rows) == 61
    for folder, seed in (("35mm_extended", 101), ("Orgnic_structure", 42)):
        for config in (SRC/folder).rglob("run_config.json"):
            rows.append(item(config.parent, "height or fluid extension", folder, seed, 3, 1, True))
    assert len(rows) == 63
    current_hash = digest(PKG / "Code/Core/surface_patch_builder.py")
    table = pd.DataFrame(rows)
    table.to_csv(SRC / "FINAL_DATA_INDEX.csv", index=False)
    audit = {
        "indexed_structure_outputs": len(rows), "paired_full_outputs": 5,
        "paired_control_outputs": 15,
        "surface_ablation_outputs": 15, "internal_ablation_outputs": 20,
        "representative_outputs": 1, "extended_outputs": 5,
        "all_required_stl_and_svg_present": True,
        "all_paired_representative_extended_graph_screens_pass": True,
        "current_surface_code_sha256": current_hash,
        "generation_surface_code_sha256_values": sorted(set(table.generation_surface_code_sha256) - {""}),
        "generation_code_relation": "Recorded generation hashes are retained alongside the separately calculated current-code checksums.",
        "ablation_comparison_tables": [
            "Source_Data/Ablation_Analysis/表面/surface_ablation_group_seed_final_graph_metrics.csv",
            "Source_Data/Ablation_Analysis/内部/internal_metrics_fixed_budget_four_groups.csv",
            "Source_Data/Benchmark_Analysis/内部/07_PC-AEGG完整组5次终态内部指标.csv",
        ],
    }
    (SRC / "FINAL_DATA_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {len(rows)} structure outputs; all required STL/SVG present.")


if __name__ == "__main__":
    main()
