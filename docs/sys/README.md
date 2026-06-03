# System Docs Index

`docs/sys/` 是当前系统设计与接口说明的主参考。新对话或新 agent 应先读
`docs/agent_handoff.md` 获取当前状态，再按任务需要阅读本目录。

## 阅读顺序

| 顺序 | 文档 | 作用 |
|---:|---|---|
| 0 | `docs/agent_handoff.md` | 5 分钟上手：当前系统事实、推荐配置、最近 baseline、常见坑 |
| 1 | `00_project_overview.md` | 项目定位、架构、数据流、核心维度 |
| 2 | `01_app_and_config.md` | CLI、配置 schema、模板和 visualization 入口 |
| 3 | `04_rt_pipeline.md` | pipeline 编排、BS/UE 到 TX/RX 的 role 解析、shard 执行 |
| 4 | `05_phy_observation.md` | custom OFDM、NR PUSCH、NR SRS 三条 PHY 链路 |
| 5 | `07_config_and_h5_format.md` | 配置字段和 HDF5 输出契约 |
| 6 | `06_io_and_testing.md` | HDF5 writer/validator/manifest 和测试质量门 |

其他文档：

- `02_domain_models.md`：domain dataclass 结构和 shape 约束。
- `03_adapters.md`：Sionna RT adapter。
- `phy_module_development.md`：新增 PHY module 的最小开发规范。
- `../todo/README.md`：当前 active TODO 总入口。
- `../todo/feature.md`：NR SRS 标准完整性、ranging 和研究功能增强。
- `../todo/structure.md`：reader、数据契约、benchmark 入口和 legacy 模块整理。
- `../todo/performance.md`：写盘、RT、空间谱、GPU 调度和大规模运行优化。
- `../performance/benchmark_harness.md`：当前 `benchmark rt/write/spectrum` 入口和输出格式。
- `indoor_fr1_100mhz_validation.md`：室内 FR1 100 MHz SRS/PUSCH 验证和成本结论。
- `../performance/README.md`：历史性能实验记录索引和 legacy 审查状态。

## 当前事实

- 配置和 label 层使用 `BS/UE`；仿真和 HDF5 层使用 resolved `TX/RX`。
- `link.phy_link_direction="uplink"` 时，`TX=UE`、`RX=BS`。
- 当前 SRS 生产基线使用 direct uplink、`rt.synthetic_array=false`。
- 本地数据在 ignored `data/` 下；场景目录应提供 `scene.xml`、标准 label `0.1.0`
  JSON 和可选 floorplan 资源。
- 标准 label 顶层 `bs_points`/`ue_points` 是全场景默认点集，`groups` 只是房间/区域等
  子集元数据；当前 pipeline 不按 group 过滤。
- 标准 floorplan 命名使用截断高度，例如 `floorplan_1p60.png` 表示 `z=1.60 m`。
- 当前 SRS 生产模板推荐 `output.sharding.shard_size=20`。
- 大规模输出采用 `results/result_xxx.h5` 多文件 shard + `manifest/manifest.json`，不建议合成单个巨大 HDF5。
- 当前 schema 版本是 `1.6.0`；NR PUSCH/SRS 统一写 `/waveform/tx_grid`、
  `/waveform/rx_grid`、`/waveform/noise_variance` 和通用 power/RSSI metadata；array 旧别名
  `/array/spatial_spectrum_label` 和 `/array/spatial_spectrum_srs` 已移除。
- `phy.tx_power_dbm` 已接入 SRS/PUSCH 发射 grid；默认 `relative_snr` 保持配置 SNR，
  `absolute_thermal` 使用 kTB + NF 固定噪声。
- 当前 CLI 提供 `benchmark rt/write/spectrum`，分别隔离 RT solve、HDF5 writer/schema
  validate 和 Bartlett 空间谱成本；输出为 ignored `outputs/` 下的 JSON/CSV/log artifact。
- `/derived` 保留 truth 语义，`first_path_propagation_range_m` 表示最早路径传播距离；
  估计型 ToA/range 写在 `/ranging`。
- NR PUSCH 与 NR SRS 已共享 `common_link.py` 的 clean channel →
  impairment/AWGN 链路；`custom_ofdm` 仍是 legacy 路径。
- NR SRS 当前是 standards-shaped v2 subset：full-slot resource grid、comb/BWP/hopping
  mask、`zc_like`/`nr_zc` pilot、group/sequence hopping、cyclic-shift port multiplexing、
  power scaling、resource LS/despread 和 full-band interpolation；仍不声称完整 3GPP SRS。

## 注意

`docs/performance/` 是实验记录，里面的参数反映当时实验事实，不一定是当前默认配置。
例如历史报告中的 `shard_size=25` 不是当前 SRS 生产建议；当前建议见本目录和
`docs/agent_handoff.md`。

`docs/todo/` 是 active TODO 的唯一入口；`docs/legacy/` 用于放置过时但暂时保留给人工
复核的文档。
