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
| 4 | `05_phy_observation.md` | custom OFDM、NR PUSCH、NR SRS-like 三条 PHY 链路 |
| 5 | `07_config_and_h5_format.md` | 配置字段和 HDF5 输出契约 |
| 6 | `06_io_and_testing.md` | HDF5 writer/validator/manifest 和测试质量门 |

其他文档：

- `02_domain_models.md`：domain dataclass 结构和 shape 约束。
- `03_adapters.md`：Sionna RT adapter。
- `phy_module_development.md`：新增 PHY module 的最小开发规范。
- `nr_srs_standard_todo.md`：当前 SRS-like 与完整 3GPP NR SRS 的差距。
- `indoor_fr1_100mhz_validation.md`：室内 FR1 100 MHz SRS/PUSCH 验证和成本结论。

## 当前事实

- 配置和 label 层使用 `BS/UE`；仿真和 HDF5 层使用 resolved `TX/RX`。
- `link.phy_link_direction="uplink"` 时，`TX=UE`、`RX=BS`。
- 当前 SRS 生产基线使用 direct uplink、`rt.synthetic_array=false`。
- 本地数据在 ignored `data/` 下，当前主要是 `dense/`、`medium/`、`sparse/`。
- 每个场景有 `label0p1.json`、`label0p2.json`、`label0p4.json` 三种 UE 采样粒度。
- 当前推荐 baseline 粒度是 `label0p2.json`。
- 当前 SRS 生产模板推荐 `output.sharding.shard_size=20`。
- 大规模输出采用 `result_xxx.h5` 多文件 shard + 根目录 `manifest.json`，不建议合成单个巨大 HDF5。

## 注意

`docs/performance/` 是实验记录，里面的参数反映当时实验事实，不一定是当前默认配置。
例如历史报告中的 `shard_size=25` 不是当前 SRS 生产建议；当前建议见本目录和
`docs/agent_handoff.md`。
