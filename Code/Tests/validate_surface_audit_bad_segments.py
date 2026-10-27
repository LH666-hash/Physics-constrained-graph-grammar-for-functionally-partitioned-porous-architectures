"""A malformed retained surface edge must never pass the global pore gate."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Core"))
import global_surface_pore_audit as audit  # noqa: E402


def main() -> None:
    mesh = trimesh.Trimesh(
        vertices=[[-1.5, -1.5, 0], [1.5, -1.5, 0],
                  [1.5, 1.5, 0], [-1.5, 1.5, 0]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False,
    )
    original_loader = audit._load_scale_align
    audit._load_scale_align = lambda *_: mesh
    try:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            val = run / "surface_validation"
            val.mkdir()
            nodes = pd.DataFrame({
                "atom_id": [0, 1, 2, 3],
                "x": [-.75, .75, .75, -.75],
                "y": [-.75, -.75, .75, .75],
                "z": [0., 0., 0., 0.],
            })
            edges = pd.DataFrame({
                "source": [0, 1, 2, 3], "target": [1, 2, 3, 0],
                "radius": [.15] * 4, "bond_type": ["surface"] * 4,
            })
            nodes.to_csv(val / "surface_nodes.csv", index=False)
            (val / "surface_final_graph_summary.json").write_text(
                json.dumps({"radius_acceptance_mm": 1.0}), encoding="utf-8")
            args = SimpleNamespace(
                stl=run / "unused.stl", run_dir=run, target_height_mm=3.0,
                global_max_edge_mm=.25, radius_acceptance_mm=1.0,
                minimum_pore_area_mm2=1e-6, aspect_ratio_max=5.0,
                aspect_ratio_min_area_mm2=.5, out_dir_name="audit",
                export_face_classification=False,
            )
            edges.to_csv(val / "surface_edges.csv", index=False)
            good = audit.global_audit(args)
            assert good["global_complete_pore_acceptance_pass"]
            for label, extra in (
                ("nonfinite_radius", {"source": 0, "target": 2, "radius": np.nan,
                                      "bond_type": "surface"}),
                ("zero_length", {"source": 0, "target": 0, "radius": .15,
                                  "bond_type": "surface"}),
            ):
                pd.concat([edges, pd.DataFrame([extra])], ignore_index=True).to_csv(
                    val / "surface_edges.csv", index=False)
                result = audit.global_audit(args)
                assert not result["global_complete_pore_acceptance_pass"], label
                assert not result["global_complete_pore_radius_acceptance_pass"], label
                assert result["invalid_surface_segment_count"] == 1, label
            fractional = edges.copy()
            fractional["source"] = fractional["source"].astype(float)
            fractional.loc[0, "source"] = .5
            fractional.to_csv(val / "surface_edges.csv", index=False)
            try:
                audit.global_audit(args)
            except ValueError as exc:
                assert "noninteger source" in str(exc)
            else:
                raise AssertionError("Fractional endpoint ID was silently truncated")
            print("Malformed surface segments correctly fail the global pore gate.")
    finally:
        audit._load_scale_align = original_loader


if __name__ == "__main__":
    main()
