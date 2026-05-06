# SionnaMeasurementSim 新系统文档群索引

本目录是新仿真系统 `SionnaMeasurementSim` 的开发文档群。文档目录约定为：

```text
SionnaMeasurementSim/docs/
```

参考旧项目 `SimpleSionna` 作为历史参考放在：

```text
SionnaMeasurementSim/old/SimpleSionna/
```

本文档群中的路径均以仓库根目录 `SionnaMeasurementSim/` 为基准。

## 阅读顺序

0. [00_global_constraints_and_official_references.md](00_global_constraints_and_official_references.md)  
   定义全局必须/禁止约束，以及 Sionna、uv 等官方参考链接。所有后续文档和实现都必须遵循它。

1. [01_system_goals_and_scope.md](01_system_goals_and_scope.md)  
   定义新系统目标、非目标、核心原则，以及为什么必须区分 `H_true` 和 `H_obs`。

2. [02_architecture.md](02_architecture.md)  
   定义推荐目录结构、模块职责和核心 pipeline。实现时应优先遵循此文档的模块边界。

3. [03_data_contract_hdf5.md](03_data_contract_hdf5.md)  
   定义 HDF5 数据契约、维度顺序、路径级数据、Doppler、速度、朝向、极化、观测诊断等字段。所有存储实现必须遵循此文档。

4. [04_sionna_rt_adapter_and_path_data.md](04_sionna_rt_adapter_and_path_data.md)  
   定义 Sionna 2.x RT adapter 的职责，说明如何提取 `vertices`、`objects`、`interactions`、`primitives`、`doppler` 等路径级信息。

5. [05_phy_observation_and_impairments.md](05_phy_observation_and_impairments.md)  
   定义 PHY 观测链、硬件损伤、同步、信道估计、`H_obs` 生成逻辑。

6. [06_config_and_experiment_schema.md](06_config_and_experiment_schema.md)  
   定义配置分组、实验配置 schema、单位、随机种子和 profile 组织。

7. [07_project_layout_uv_git_workflow.md](07_project_layout_uv_git_workflow.md)  
   定义新仓库目录、`uv` 环境管理、git 分支/提交规范、数据和输出目录策略。

8. [08_roadmap_milestones_acceptance.md](08_roadmap_milestones_acceptance.md)  
   定义阶段、里程碑、每阶段验收标准，以及何时测试和提交代码。

9. [09_testing_and_quality_gates.md](09_testing_and_quality_gates.md)  
   定义单元测试、adapter 测试、集成测试、统计测试、数据契约测试和质量门。

10. [10_old_system_reuse_and_migration.md](10_old_system_reuse_and_migration.md)  
    定义旧系统哪些代码和经验可复用，哪些必须重写，以及旧系统放入 `old/` 后的使用方式。

11. [11_calibration_and_diagnostics.md](11_calibration_and_diagnostics.md)  
    定义实测标定、诊断指标、分布对齐和 profile 管理。

## 全局约束

- 新系统使用 Sionna 2.x，不在新核心链路中继续依赖 TensorFlow。
- 新系统使用 `uv` 管理 Python 环境和依赖。
- 新仓库 git 根目录是 `SionnaMeasurementSim/`，不是 Python 包目录。
- 测试场景放在 `SionnaMeasurementSim/data/scenes/test/`。
- 大型输入数据、运行输出和 HDF5 默认不提交 git。
- 所有阶段都必须有测试和验收标准；通过后及时提交 git。
- 数据契约优先于实现便利。后续需要较少字段时，从完整采集数据中取子集，而不是一开始丢失关键物理信息。
- 任何 schema、adapter、配置、验收标准变更都必须同步更新文档和测试。
