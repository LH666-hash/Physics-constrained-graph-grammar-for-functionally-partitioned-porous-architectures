# PC-AEGG 代码使用说明

本目录提供 PC-AEGG 的核心实现、案例配置、统计复算和分析程序。代码通过 GitHub 私有仓库管理，配套研究数据保存在 Zenodo 私有草稿中，并通过作者提供的只读链接供审稿核查。审稿核查包记录其对应的代码提交号。项目概览见[项目说明](../README.md)。以下命令在项目根目录执行。

## 环境与依赖

`requirements-tested.txt` 记录 Python 3.11.5 下用于配置和入口回归检查的依赖版本；`requirements.txt` 列出当前实现使用的依赖包。

```sh
python -m pip install -r Code/requirements-tested.txt
```

## 归档统计复算

```sh
python Code/Analysis/recompute_archived_results.py --out ReproductionResults
```

该程序读取四组、每组五个种子的共 20 份装配后压力剖面，以及 `Source_Data/ICP/` 中的工作簿，计算逐结构最小压力余量、组均值、样本标准差、稀释校正后的浓度和相应浓度比例。输入为归档剖面及实验读数，结构生成和从几何计算压力由生成程序执行。

结果写入 `--out` 指定目录中的 `recomputed_results.json`；已有同名文件时更新该 JSON。三个 ICP 读数为同一样品的技术重复。La 指标采用终浓度与初浓度之比，浓度比例与物质量平衡分别解释。

## 案例配置与生成入口

```sh
python Code/Analysis/generate_case.py --case water25 --out NewRuns/water25 --dry-run
```

`--dry-run` 显示实际命令，移除后开始计算。三个案例读取相应的 `Code/Configurations/*_reference.json`：

| case | 流体 | 目标高度（mm） | 随机种子 | 压力切片数 |
| --- | --- | --- | --- | --- |
| `water25` | 水相 | 25 | 42 | 25 |
| `water35` | 水相 | 35 | 101 | 35 |
| `organic22p5` | 石油醚 | 22.5 | 42 | 23 |

压力切片标称厚度为 1 mm，22.5 mm 案例的最后一个切片厚 0.5 mm。三个参考配置的内部初始骨架均为 20 层；骨架层数与压力切片数是不同参数。

入口显式设置流体物性、名义杆径、内部骨架层数、随机种子、压力切片和安全储备。输入有效且输出路径位于归档 `Source_Data/` 之外时，入口创建新目录，再启动计算；已有输出目录会被拒绝。直接无参数运行核心主程序时，程序提示使用案例入口。

### 输入几何

| case | 默认输入 |
| --- | --- |
| `water25` | [Input_Geometries/长方体.stl](../Input_Geometries/长方体.stl) |
| `water35` | [Input_Geometries/box_10x10x35mm.stl](../Input_Geometries/box_10x10x35mm.stl) |
| `organic22p5` | [Input_Geometries/box_10x10x22.5mm.stl](../Input_Geometries/box_10x10x22.5mm.stl) |

`--input` 可指定其他输入。参考 JSON 中的 `input_stl` 和 `output_dir` 保留原运行位置；生成入口采用上表中的包内输入，并以 `--out` 设置实际输出目录。

35 mm 和石油醚盒体输入由保存的尺寸与载体网格记录重建，来源和校验值见 [generated_input_geometries.json](Configurations/generated_input_geometries.json)。石油醚输入为 10 × 10 × 22.5 mm、5000 个三角面；35 mm 输入为 10 × 10 × 35 mm、12 个三角面。两份输入均已随包提供。

### 有机相轮廓修复预算

有机相启动器检查实际载体为 5000 个三角面，并将全局轮廓修复的请求参数设为 256。每次修复调用的有效外层轮次预算为 `max(256, 512, 8 * N0)`，其中 `N0` 是该次调用开始时完整轮廓中的半径超限孔数。无超限孔、候选耗尽或有效预算用尽时停止。外层轮次、候选尝试数和成功修复数分别记录。

## 版本、规则与归档记录

生成入口使用随包提供的当前实现，其源码版本与归档生成记录中的版本不同。归档记录中的 `core_code_sha256` 标识生成时的源码，当前仓库文件的校验值见根目录 `CODE_MANIFEST.csv`。

[production_code_map.csv](Configurations/production_code_map.csv) 列出补充表 S1–S3 的规则与当前实现函数的对应关系。[recorded_execution_evidence.json](Configurations/recorded_execution_evidence.json) 保留归档运行事件、来源路径及校验信息。历史记录中的绝对路径用于来源追溯，实际入口按当前项目目录解析输入。

## 配对分析与缓存

`Analysis/saved_configuration.py` 在配置恢复时同步流体物性、高度、层厚及派生量。终态验收依据实际高度网格确定预期压力切片数。

配对缓存核对节点、连边、表面与运行配置、输入几何和相关源码，同时核对输出 STL、节点、连边及压力表的校验值。校验不匹配时重新计算相应配对结果。配对与重建脚本的输出范围见下表。

## 维护脚本的输出范围

| 脚本 | 用途与输出 |
| --- | --- |
| `Analysis/Rebuild_All_Current.ps1` | 批量生成水相结构、验收后更新 `Source_Data/` 中的对应结果；暂存与备份目录位于项目上一级。`-SingleRepresentative` 选择代表结构流程。 |
| `Analysis/pair_ablation_controls.py` | 重建配对分析，在 `Source_Data/Ablation_Analysis/` 中写入配对结果；替换已有结果时保留备份。 |
| `Analysis/audit_final_package.py` | 生成结构索引，更新 `Source_Data/FINAL_DATA_INDEX.csv` 和 `Source_Data/FINAL_DATA_AUDIT.json`。 |
| `Analysis/build_input_geometries.py` | 重建两份盒体输入及输入来源记录；同名输入已存在时停止。 |
| `Analysis/plot_*.py`、`compose_figure2.py`、`export_*.py` | 根据各脚本设置导出分析图件或组合图，供图件维护使用。 |

结构索引记录 STL、SVG、图结构筛选和压力评价等项目。几何融合、水密性与制造验证分别由相应几何检查或实验评价。

## 回归检查

只检查入口与输出目录保护：

```sh
python -m unittest discover -s Code/Tests -p "test_generation_entrypoint.py"
```

审稿核查包已包含配置与入口检查所需的数据，解压后可运行：

```sh
python -m unittest discover -s Code/Tests -p "test_*.py"
```

检查包括 25 mm 水相、22.5 mm 有机相、35 mm 水相再返回 25 mm 水相的连续切换，以及保存剖面的压力网格验收。

## 微 CT 保存结果核对

```sh
python Code/Analysis/recompute_microct_metrics.py --ct-root Micro_CT --out ReviewChecks/microct
```

该入口读取五组保存的原生分割、分析区掩膜、已配准比较数组和组分表，复算实体 Dice、两种孔隙率口径并核对体素及体积汇总。输出为 `microct_recomputed_metrics.json`。核查所用文件随审稿核查包提供；完整扫描和原生 STL 保存在完整 CT 压缩包中。

## 基准程序的包内输入

`Benchmark/run_strong_baseline_experiment.py` 从包内 `Source_Data/Benchmark_Analysis/内部/` 读取预设结构搜索表及 PC-AEGG 参考指标，使用同目录中的 `internal_pareto_pipeline.py` 和 `Core/capillary_viscous_filling.py`。其 `--out-dir` 是新计算结果的位置。`internal_pareto_pipeline.py` 的几何、液压模块和五个参考运行目录通过命令行显式指定。归档搜索表和自然表面比较结果可直接查看。
