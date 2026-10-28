# PC-AEGG code guide

We provide the generation implementation, case configurations and saved-result analyses used with our study. The code repository and Zenodo draft remain private during peer review. Our [project README](../README.md) and Zenodo `README_DATA.md` describe access and the data layout. Run the commands below from the project root.

## Environment and saved inputs

Use Python 3.11. `requirements-tested.txt` records the dependency versions checked with Python 3.11.5. `requirements.txt` lists the packages used by the implementation.

```sh
python -m pip install -r Code/requirements-tested.txt
python Code/Analysis/prepare_review_inputs.py
```

For input preparation, keep `PC_AEGG_Review_Inputs_20261028.zip` in the project root or its parent directory. The command verifies the archive and each selected input, then extracts only `Source_Data/` and `Micro_CT/`. It preserves the current code. The case configurations use project-relative geometry and output paths, so no author-directory edits are needed.

## Statistics from archived results

```sh
python Code/Analysis/recompute_archived_results.py --out ReviewChecks/archived_statistics
```

The command reads 20 post-assembly pressure profiles, representing four groups of five seeds, and two workbooks in `Source_Data/ICP/`. It calculates per-structure minimum pressure margins, group means, sample standard deviations, dilution-corrected concentrations and concentration ratios. It writes `recomputed_results.json` in the selected output directory and updates that JSON if it already exists. The three instrument readings for each solution are technical repeats. The La endpoint is a final-to-initial concentration ratio.

These calculations use saved profiles and measured readings. Geometry generation and pressure evaluation from geometry are separate operations.

## Case entry points

```sh
python Code/Analysis/generate_case.py --case water25 --out NewRuns/water25 --dry-run
python Code/Analysis/generate_case.py --case water35 --out NewRuns/water35 --dry-run
python Code/Analysis/generate_case.py --case organic22p5 --out NewRuns/organic22p5 --dry-run
```

`--dry-run` displays the command without starting generation. Removing that flag starts a new calculation.

| Case | Fluid | Height (mm) | Seed | Pressure slices |
| --- | --- | ---: | ---: | ---: |
| `water25` | Water | 25 | 42 | 25 |
| `water35` | Water | 35 | 101 | 35 |
| `organic22p5` | Petroleum ether | 22.5 | 42 | 23 |

Pressure slices have a nominal thickness of 1 mm; the final slice is 0.5 mm thick for the 22.5 mm case. All three reference configurations use 20 initial skeleton layers. Skeleton layers and pressure slices are separate parameters.

The entry point sets the fluid properties, nominal diameters, skeleton layers, seed, pressure grid and safety reserves explicitly. After checking the input geometry, it creates a fresh output directory outside archived `Source_Data/`, then starts the calculation. Existing output directories are rejected. Running the core program without arguments directs the reader to the case entry point.

### Geometry inputs

| Case | Default geometry |
| --- | --- |
| `water25` | [Input_Geometries/长方体.stl](../Input_Geometries/长方体.stl) |
| `water35` | [Input_Geometries/box_10x10x35mm.stl](../Input_Geometries/box_10x10x35mm.stl) |
| `organic22p5` | [Input_Geometries/box_10x10x22.5mm.stl](../Input_Geometries/box_10x10x22.5mm.stl) |

Use `--input` to select another geometry and `--out` to choose its output directory. `input_stl` and `output_dir` in the reference JSON files are project-relative. The entry point resolves the default input within the supplied project.

The 35 mm and petroleum-ether boxes were reconstructed from saved dimensions and carrier-mesh records. [generated_input_geometries.json](Configurations/generated_input_geometries.json) records their provenance and checksums. The petroleum-ether input is 10 × 10 × 22.5 mm with 5,000 triangles; the 35 mm input is 10 × 10 × 35 mm with 12 triangles. Both files are supplied.

### Organic-phase contour-repair budget

The launcher checks that the carrier has 5,000 triangles and sets the requested global-contour repair parameter to 256. The effective outer-iteration budget for each repair call is `max(256, 512, 8 * N0)`, where `N0` is the initial number of oversized complete pores in that call. The call stops when no oversized pore remains, candidates are exhausted, or the effective budget is reached. Outer iterations, candidate attempts and accepted repairs are recorded separately.

## Versions and production rules

Generation uses the current supplied implementation. Historical generated structures retain their own configurations and source fingerprints; these identify a different source version. Current source checksums are recorded in the root `CODE_MANIFEST.csv`.

[production_code_map.csv](Configurations/production_code_map.csv) links Supplementary Tables S1–S3 to implementation functions. [recorded_execution_evidence.json](Configurations/recorded_execution_evidence.json) retains historical execution events and their source checksums. Absolute paths inside historical records describe their original provenance; active entry points resolve inputs within the current project.

The five sensitivity CSV files preserve the archived carrier/patch parameter study. Their `target_faces` and `actual_faces` columns distinguish requested and achieved mesh counts. `max_faces_per_patch` controls initial patch growth; later merging of small fragments can exceed that value. These records describe geometric preprocessing, rather than 108 complete functional-generation runs. A dedicated driver that recreates the complete archived 108-combination study is not included.

The original Figure 3g workbook and CSV are in `PC_AEGG_Figure3g_Source_Data_20261028.zip`. The accompanying data README defines the recorded Error field and the relative percentage difference calculated from the original radii.

## Paired analyses and caching

`Analysis/saved_configuration.py` restores fluid properties, height, slice thickness and derived settings together. Terminal pressure acceptance derives its expected slice count from the actual height grid.

The pairing cache checks nodes, edges, surface/run configurations, input geometry and related source files. It also checks output STL, graph and pressure-table hashes. A mismatch triggers recalculation of the affected paired results.

## Maintenance scripts

| Script | Operation and output |
| --- | --- |
| `Analysis/Rebuild_All_Current.ps1` | Generates water cases and updates corresponding archived results after acceptance; staging and backups are placed above the project directory. `-SingleRepresentative` selects its representative-case workflow. |
| `Analysis/pair_ablation_controls.py` | Rebuilds paired analyses under `Source_Data/Ablation_Analysis/`, retaining backups when existing results are replaced. |
| `Analysis/audit_final_package.py` | Updates `Source_Data/FINAL_DATA_INDEX.csv` and `Source_Data/FINAL_DATA_AUDIT.json`. |
| `Analysis/build_input_geometries.py` | Reconstructs the two box inputs and their provenance record; stops if those inputs already exist. |
| `Analysis/plot_*.py`, `compose_figure2.py`, `export_*.py` | Export analysis figures or combined panels according to each script's settings. |

These maintenance operations write results. The documented saved-result checks above use archived inputs. Mesh union, watertightness and manufacturing validation have their own geometry checks or experiments.

## Configuration and entry-point checks

```sh
python -m unittest discover -s Code/Tests -p "test_*.py"
```

The checks cover sequential switching through 25 mm water, 22.5 mm petroleum ether, 35 mm water and back to 25 mm water, terminal acceptance of saved pressure grids, explicit case arguments, cache inputs and output-directory protection.

To run only the entry-point checks:

```sh
python -m unittest discover -s Code/Tests -p "test_generation_entrypoint.py"
```

## Saved micro-CT calculations

```sh
python Code/Analysis/recompute_microct_metrics.py --ct-root Micro_CT --out ReviewChecks/microct
```

The command reads five samples' saved native segmentations, ROI masks, aligned comparison arrays and component tables. It calculates solid Dice and both porosity definitions, and checks voxel/volume totals from the component tables. Its output is `microct_recomputed_metrics.json`. The input archive supplies the selected inputs; the complete CT archives retain the models and scans. TXM files are not required by this command.

## Benchmark inputs

`Benchmark/run_strong_baseline_experiment.py` reads archived search tables and PC-AEGG reference indicators from `Source_Data/Benchmark_Analysis/内部/`. It uses the supplied `internal_pareto_pipeline.py` and `Core/capillary_viscous_filling.py`; `--out-dir` selects the location for a new search. `internal_pareto_pipeline.py` accepts the geometry, hydraulic module and five reference-run directories through explicit command-line arguments. The archived search tables and natural-surface comparison can be inspected directly.
