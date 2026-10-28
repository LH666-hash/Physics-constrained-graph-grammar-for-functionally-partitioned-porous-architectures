# PC-AEGG

Physics-constrained graph grammar for functionally partitioned porous architectures

We provide the PC-AEGG implementation, case configurations and analysis scripts accompanying our study. The data are deposited in Zenodo record 23696832. During peer review, both repositories remain private; we provide the Zenodo read-only review link with the submission.

Start with `README_DATA.md` in the Zenodo deposit to locate the data. `FILE_CATALOG.csv` lists all archived files, and `FIGURE_DATA_MAP.csv` connects article figures and tables to their supporting materials. The five extended geometries and their individual outputs are in `Source_Data/Extended_Geometries/` in the structure-data archive.

## Environment and project layout

Use Python 3.11. The dependency checks were performed with Python 3.11.5.

```sh
python -m pip install -r Code/requirements-tested.txt
```

Run commands from the project root. `Code/`, `Input_Geometries/`, `Source_Data/` and, for CT checks, `Micro_CT/` use the relative paths supplied in the archives.

| Directory | Contents |
| --- | --- |
| `Code/Core/` | Graph generation, geometry and physical evaluation |
| `Code/Analysis/` | Case entry points and saved-result calculations |
| `Code/Benchmark/` | Benchmark geometry and internal-performance tools |
| `Code/Configurations/` | Case definitions, production mapping and input provenance |
| `Code/Tests/` | Configuration and entry-point regressions |

## Prepare the Zenodo inputs automatically

Download `PC_AEGG_Code_20261028.zip` and `PC_AEGG_Review_Inputs_20261028.zip`. Extract the code archive into a project directory. Keep the input archive in that directory or its parent, then run:

```sh
python Code/Analysis/prepare_review_inputs.py
```

The script verifies the archive and extracts only its `Source_Data/` and `Micro_CT/` inputs. It preserves the current code and verifies every extracted input. Case configurations use package-relative geometry and output paths. No source-path edits are required.

## Review package and archived calculations

`PC_AEGG_Review_Inputs_20261028.zip` contains the saved inputs needed by these commands. The current code is supplied separately in `PC_AEGG_Code_20261028.zip`:

```sh
python -m unittest discover -s Code/Tests -p "test_*.py"
python Code/Analysis/recompute_archived_results.py --out ReviewChecks/archived_statistics
python Code/Analysis/recompute_microct_metrics.py --ct-root Micro_CT --out ReviewChecks/microct
```

The first analysis reads 20 saved pressure profiles and two ICP workbooks. The second reads five CT samples' segmentations, ROI masks, aligned comparison arrays and component tables. They calculate statistics from saved outputs. The data guide describes their inputs, units, statistical definitions and output files.

The pressure/ICP output is `recomputed_results.json`; the CT output is `microct_recomputed_metrics.json`, within the selected output directories.

## Figure 3g source data

The original Figure 3g workbook is supplied in `PC_AEGG_Figure3g_Source_Data_20261028.zip` in Zenodo. `Source_Data/Cylinder_Surface_Mapping/Cylinder_Surface_Mapping.xlsx` uses English column headings and retains the original 150 rows for each of the three cylinder radii (5, 10 and 20 mm). A CSV export preserves the same row and column order. These are the original figure-source values.

## Current structure generation

Inspect a case without starting generation:

```sh
python Code/Analysis/generate_case.py --case water25 --out NewRuns/water25 --dry-run
python Code/Analysis/generate_case.py --case water35 --out NewRuns/water35 --dry-run
python Code/Analysis/generate_case.py --case organic22p5 --out NewRuns/organic22p5 --dry-run
```

The cases use 25 mm water, 35 mm water and 22.5 mm petroleum-ether settings. Remove `--dry-run` to calculate a new structure. Each run requires a new output directory, created by the entry point.

Historical structures retain the configurations and source fingerprints of their generation runs. Current generation uses the current implementation; saved-result calculations use the deposited archived outputs. Additional configuration and implementation details are in `Code/README.md`.

## File and rule records

`CODE_MANIFEST.csv` records source-file SHA-256 values. `Code/Configurations/production_code_map.csv` links 74 production-rule entries to their implementation functions. The same mapping is deposited separately in Zenodo as `PRODUCTION_CODE_MAP.csv`.

Data-file versions and scope are documented in the Zenodo data guide. Numerical UV-vis spectral exports are outside the present deposit; the corresponding plotted figure files are supplied. All five CT samples' processed analysis arrays and models are supplied. The saved-result commands do not require TXM scan volumes.
