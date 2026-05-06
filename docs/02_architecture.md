# 02. 系统架构

本文定义新系统推荐架构。数据落盘必须遵循 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)，Sionna RT 相关实现必须隔离到 [04_sionna_rt_adapter_and_path_data.md](04_sionna_rt_adapter_and_path_data.md) 所定义的 adapter 层。

## 1. 仓库根目录

仓库根目录：

```text
SionnaMeasurementSim/
  .git/
  README.md
  pyproject.toml
  uv.lock
  .gitignore

  sionna_measurement_sim/
  config/
  docs/
  tests/
  scripts/
  data/
  outputs/
  artifacts/
  old/
```

其中：

- `sionna_measurement_sim/` 是 Python 包。
- `docs/` 放本文档群。
- `data/` 放小样例、场景、标注、实测标定数据。
- `outputs/` 放运行输出，默认不提交 git。
- `artifacts/` 放标定 profile、报告、中间产物。
- `old/` 放旧 `SimpleSionna` 项目作为历史参考。

工程规范详见 [07_project_layout_uv_git_workflow.md](07_project_layout_uv_git_workflow.md)。

## 2. Python 包结构

推荐包结构：

```text
sionna_measurement_sim/
  app/
    cli.py
    pipeline.py
    run_context.py

  config/
    schema.py
    loader.py

  domain/
    topology.py
    devices.py
    antenna.py
    scene.py
    path.py
    channel.py
    observation.py
    results.py
    units.py

  io/
    label_parser.py
    scene_assets.py
    output_layout.py
    hdf5_writer.py
    hdf5_reader.py
    manifest.py

  adapters/
    sionna_rt/
      scene_adapter.py
      rt_solver.py
      path_adapter.py
      material_adapter.py
      render_adapter.py
    sionna_phy/
      ofdm_adapter.py
      channel_adapter.py

  rt/
    truth_pipeline.py
    batch_runner.py
    path_summary.py

  phy/
    waveform.py
    pilots.py
    tx_chain.py
    channel_application.py
    rx_frontend.py
    synchronization.py
    channel_estimation.py
    observation_pipeline.py

  impairments/
    awgn.py
    cfo_sfo.py
    phase_noise.py
    iq_imbalance.py
    agc_adc.py
    nonlinearities.py
    interference.py

  calibration/
    measurement_loader.py
    parameter_fitting.py
    profiles.py
    validation.py

  analysis/
    truth_stats.py
    observation_stats.py
    error_metrics.py
    quality_flags.py

  visualization/
    topology_plots.py
    path_plots.py
    channel_plots.py
    observation_plots.py
    render.py

  preflight/
    system.py
    backend.py
    input_output.py
```

## 3. 分层原则

### 3.1 App Layer

职责：

- CLI 参数解析。
- pipeline 编排。
- 日志、run id、错误处理。
- 阶段状态记录。

禁止：

- 直接访问 Sionna `Scene`、`Paths`。
- 直接写 HDF5 dataset。
- 混入信道估计算法。

### 3.2 Domain Layer

职责：

- 定义系统内部数据模型。
- 约定单位、维度和字段语义。

典型对象：

- `Topology`
- `RadioDeviceState`
- `AntennaArraySpec`
- `PathTable`
- `RTTruthResult`
- `ObservationResult`
- `MeasurementSimulationResult`

禁止：

- 依赖 Sionna 对象。
- 在 dataclass 中直接存 Sionna `Paths` 作为主数据。

### 3.3 Adapter Layer

职责：

- 封装 Sionna RT/PHY API。
- 把 Sionna 对象转换为内部 domain 对象。
- 隔离 Sionna 2.x API 变化。

原则：

- 所有 `from sionna...` 应尽量集中在 `adapters/`。
- 业务层不直接读取 `paths.vertices` 等属性，而是通过 `PathAdapter` 输出标准 `PathTable`。

### 3.4 RT Layer

职责：

- 传播真值仿真。
- 批处理。
- 路径摘要。
- `H_true` 生成。

输出必须符合 [03_data_contract_hdf5.md](03_data_contract_hdf5.md) 中 `/channel/truth` 与 `/paths` 的约定。

### 3.5 PHY Layer

职责：

- 资源栅格、导频、OFDM 波形。
- 接收机观测链。
- 同步和信道估计。
- `H_obs` 生成。

输出必须符合 [05_phy_observation_and_impairments.md](05_phy_observation_and_impairments.md) 与 [03_data_contract_hdf5.md](03_data_contract_hdf5.md) 中 `/observation` 的约定。

### 3.6 IO Layer

职责：

- 输入解析。
- 输出目录。
- HDF5 writer/reader。
- manifest。

禁止：

- 计算路径。
- 修改 `H_true` 或 `H_obs`。
- 对 Sionna 对象做深层解析。

## 4. 主 Pipeline

推荐主流程：

```text
1. load config
2. validate config schema
3. preflight
4. parse label/topology
5. create run directory
6. build scene adapter
7. run RT truth pipeline
8. extract path table and path summary
9. run PHY observation pipeline, if enabled
10. compute evaluation metrics
11. write HDF5
12. write manifest
13. generate visualizations
14. write stats summary
```

## 5. 数据流

```text
Label + Config
  -> Topology + ExperimentSpec
  -> SionnaRTSceneAdapter
  -> RTTruthResult
     -> PathTable
     -> H_true
  -> PHYObservationPipeline
     -> H_obs + diagnostics
  -> EvaluationMetrics
  -> HDF5 + Manifest + Plots
```

## 6. 扩展点

必须预留：

- 不同 RT backend。
- 不同 PHY standard：custom OFDM、WiFi-like、NR-like。
- 不同 channel estimator：LS、LMMSE、自定义估计器。
- 不同 impairment profile。
- 不同 calibration profile。
- 不同 HDF5 schema version 的读兼容。

## 7. 最重要的边界

新系统成败取决于两个边界：

1. `PathAdapter` 边界  
   把 Sionna `Paths` 中的 `vertices`、`objects`、`interactions`、`doppler` 等转换为稳定内部格式。

2. `ObservationPipeline` 边界  
   把理想信道真值转换为真实接收机观测值，并记录所有非理想因素。

这两个边界必须有单元测试、集成测试和数据契约测试。
