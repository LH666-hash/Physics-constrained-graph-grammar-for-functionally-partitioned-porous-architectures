# Physics-constrained graph grammar for functionally partitioned porous architectures

本项目提供 PC-AEGG 的实现代码、案例配置、统计复算脚本和辅助分析程序。代码通过 GitHub 私有仓库管理，研究数据保存在 Zenodo 私有草稿中，包括生成结构、实验读数和更新后的微 CT 材料。作者通过只读审稿链接提供访问；代码和数据的公开发布由作者另行确定。

Zenodo 中的 `PC_AEGG_Reviewer_Package_20261028.zip` 包含对应 GitHub 提交的代码及复算所需输入。解压后可直接按照下面的命令核对保存结果；代码提交号与文件校验值记录在包内。完整扫描、原生模型、图件和视频由其余分组压缩包提供。

代码仓库：<https://github.com/LH666-hash/Physics-constrained-graph-grammar-for-functionally-partitioned-porous-architectures>

## 仓库内容

| 路径 | 内容 |
| --- | --- |
| [Code/Core](Code/Core/) | 图结构生成、几何处理和压力评价的核心实现 |
| [Code/Analysis](Code/Analysis/) | 归档统计复算、案例入口和分析脚本 |
| [Code/Benchmark](Code/Benchmark/) | 内部结构基准程序 |
| [Code/Configurations](Code/Configurations/) | 案例参数、规则与函数对应表、输入来源记录 |
| [Code/Tests](Code/Tests/) | 配置恢复、压力网格和输出目录保护的回归检查 |
| [Input_Geometries](Input_Geometries/) | 三个案例入口使用的小型几何输入 |
| [Source_Data](Source_Data/) | 按原相对路径保留的有机相启动器及黏度模型脚本 |

原始扫描、归档结构输出、工作簿、论文图件和视频由配套数据归档提供。详细命令见 [Code/README.md](Code/README.md)。

## 环境

以下命令在项目根目录执行。依赖版本记录对应 Python 3.11.5 下的配置与入口检查环境。

```sh
python -m pip install -r Code/requirements-tested.txt
```

## 复算论文所报统计

解压审稿核查包，或将完整数据包的数据目录合并到项目根目录，保持 `Code/` 与 `Source_Data/` 并列。统计复算使用 `Source_Data/Ablation_Analysis/` 中的归档压力剖面和 `Source_Data/ICP/` 中的工作簿。

```sh
python Code/Analysis/recompute_archived_results.py --out ReproductionResults
```

程序从保存的压力剖面与 ICP 原始读数计算压力指标、汇总统计和浓度换算结果，输出 `ReproductionResults/recomputed_results.json`。此入口采用归档数值作为输入，结构生成使用下面的独立入口。

## 按配置运行生成程序

```sh
python Code/Analysis/generate_case.py --case water25 --out NewRuns/water25 --dry-run
```

`--dry-run` 显示展开后的命令；移除后开始计算。其他案例为 `water35` 和 `organic22p5`。生成入口采用随包提供的当前实现，参数和输入来源见[代码使用说明](Code/README.md)。新输出目录由入口创建。

## 版本与文件校验

论文归档结构的生成版本由对应生成记录中的源码校验值标识。当前生成源码与这些归档生成版本不同，仓库文件的 SHA-256 校验值列于 `CODE_MANIFEST.csv`。Zenodo 审稿核查包记录所含代码的 GitHub 提交号；数据文件的校验值列于对应数据清单。

## 回归检查

入口与目录保护检查可在代码仓库中运行：

```sh
python -m unittest discover -s Code/Tests -p "test_generation_entrypoint.py"
```

完整配置检查所需的压力剖面和配对缓存输入已包含在审稿核查包中；解压后可运行：

```sh
python -m unittest discover -s Code/Tests -p "test_*.py"
```

## 微 CT 保存结果核对

审稿核查包中的 `Micro_CT/` 提供保存的分割、配准数组和定量表；完整 CT 压缩包另提供扫描、原生 STL、图件和视频。

```sh
python Code/Analysis/recompute_microct_metrics.py --ct-root Micro_CT --out ReviewChecks/microct
```

程序从保存的数组复算五个样品的实体 Dice、连续体积孔隙率和有效原始层孔隙率，并核对孔隙组分表的体素与体积总和。分割、配准和孔隙标记属于已保存的上游处理阶段。完整原生 STL 约为每个样品 1.2–1.4 GB，查看几何时可先使用同目录中的预览 STL。
