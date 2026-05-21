# Structure TODO

本页记录数据契约、reader/API、benchmark 入口、legacy 模块和输出结构整理类 TODO。
顶部列表按当前重要程度排序；每次修改都要重新检查顺序。

## Priority List

| 顺位 | ID | TODO | 简述 |
|---:|---|---|---|
| 1 | STR-001 | shard-aware reader / dataset loader | 为多 `result_xxx.h5` 输出提供统一训练/分析入口，支持 manifest 和全局 UE/BS 索引。 |
| 2 | STR-002 | 通用 benchmark 入口 | 建立 RT-only、PHY-only、write-only 等稳定 CLI/API，避免端到端耗时掩盖模块瓶颈。 |
| 3 | STR-003 | custom OFDM legacy 处理 | 决定 custom OFDM 是迁移到通用链路、导出真实 waveform grid，还是作为 legacy 移除。 |
| 4 | STR-004 | array label alias 精简 | 处理 `aoa_heatmap_label` / `spatial_spectrum_label` 的兼容 alias 和字段冗余。 |

## Details

### STR-001: shard-aware reader / dataset loader

目的：多文件 shard 已经是生产输出方式，下游训练和分析需要稳定 reader，而不是手写遍历
`results/result_xxx.h5`。

涉及模块：新增或扩展 IO/reader 包、manifest schema、训练/分析脚本、文档。

验收标准：reader 能按 manifest 聚合 shard，支持全局 UE/BS 索引定位、按 shard/UE range
迭代、读取 config snapshot 和 schema 状态；至少有 fixture 单元测试和一个真实 manifest smoke。

重点提醒：不要假设 shard 文件名连续；fallback 可能生成 `result_089_00.h5` 这类子 shard。

### STR-002: 通用 benchmark 入口

目的：把性能诊断从临时脚本提升为稳定入口，分别测 RT、PHY、array、write 等模块成本。

涉及模块：CLI、debug profiling、performance docs、可能的 `scripts/` benchmark helpers。

验收标准：提供 `rt-only`、`phy-only`、`write-only` 或等价入口；输出统一 JSON/CSV summary；
能复用现有配置和 fixture，不要求真实大数据才能跑通。

重点提醒：benchmark 是工程接口，不应绑定 NR PUSCH/SRS 内部对象；未来 WiFi-like 或 6G-like
链路也应能复用。

### STR-003: custom OFDM legacy 处理

目的：当前 custom OFDM 是 legacy 路径，HDF5 writer 里也保留了真实 `tx_grid/rx_grid`
导出的 TODO。需要明确它的后续命运。

涉及模块：`custom_ofdm` PHY module、`common_link.py`、HDF5 writer/schema/docs/tests。

验收标准：二选一完成：迁移到通用 clean channel/impairment 链路并导出真实 waveform grid；
或正式标记/移除 legacy path，清理 schema 和文档中的模糊承诺。

重点提醒：不要写 fake waveform grid；如果没有真实频域 waveform tensor，就不要导出统一字段。

### STR-004: array label alias 精简

目的：`aoa_heatmap_label` 和 `spatial_spectrum_label` 当前语义上是兼容 alias，后续需要减少
字段冗余和 HDF5 体积。

涉及模块：array output builder、HDF5 writer/schema validator、visualization、docs、旧数据兼容说明。

验收标准：明确保留哪个物理 dataset；另一个如需兼容，应通过 attrs/manifest 或 reader alias 处理；
新 schema 和 visualization 测试通过。

重点提醒：这会影响已有分析脚本，执行前需要先扫引用并提供迁移说明。
