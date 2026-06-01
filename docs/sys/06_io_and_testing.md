# 06. I/O 层与测试体系

## I/O 层

`io/` 目录负责 HDF5 读写、schema 校验、manifest 输出和标签解析。

### `hdf5_writer.py` — HDF5 写入

```python
def write_measurement_result(path: str | Path, result: MeasurementSimulationResult) -> Path
```

将 `MeasurementSimulationResult` 中的所有 domain 对象序列化为 HDF5。内部按 group 分拆为 `_write_meta()`、`_write_topology()`、`_write_truth()`、`_write_observation()` 等独立函数。

**关键约束：**
- Writer 不导入、不检查 Sionna 对象
- 只接受 domain 层的纯 Python/numpy 数据
- 所有 dataset 标注 `unit` 和 `index_order` attribute
- 大数组使用 gzip compression + shuffle filter

**写入的顶层 group：**
```
/meta /input /topology /devices /antenna /scene /frequency
/channel/truth /paths/samples /paths/full /link
/waveform /observation /ranging /impairments /receiver /evaluation
/derived /array /calibration /motion /runtime
```

### `schema_validator.py` — HDF5 契约校验

```python
def validate_hdf5_contract(path: str | Path) -> None
```

在 HDF5 写入后自动调用（也在测试中独立使用）。检查：

1. **必填 group**：`meta`、`channel/truth`、`paths/samples`、`runtime`、`link` 等
2. **必填 dataset**（truth）：
   - `meta/schema_version`、`meta/index_order`、`meta/unit_convention`
   - `/channel/truth/cfr`、`/topology/tx_positions_m`、`/frequency/frequencies_hz`
   - `/paths/samples/vertices_m`、`doppler_hz`、`tau_s`
3. **必填 dataset**（observation）：
   - `/observation/cfr_est`、`/evaluation/nmse_db`、`/evaluation/ber`、`/evaluation/bler`
   - `/evaluation/num_bit_errors`、`/evaluation/num_block_errors`、`/evaluation/num_blocks`
4. **Shape 关系**：
   - `truth_cfr.ndim == 5`，`cfr_est.ndim == 6`
   - `cfr_est.shape[1:] == truth_cfr.shape`
   - `frequencies_hz.shape[-1] == truth_cfr.shape[-1]`
5. **数值有效性**：CFR 至少一个有限值、频率严格递增
6. **禁止路径**：`/channel/cfr`、`/derived/rtt_like_m`、`/derived/rtt_like_s` 必须不存在
7. **NR PUSCH 专有字段**（当 `waveform/standard == "nr_pusch"`）：
   - `num_prb`、`num_layers`、`num_antenna_ports`、`mimo_detector` 等
   - `num_layers >= 1`、`num_antenna_ports >= num_layers`
8. **NR SRS 专有字段**（当 `waveform/standard == "nr_srs"`）：
   - 统一 waveform 字段 `tx_grid`、`rx_grid`、`noise_variance`
   - SRS 专属 `srs_resource_mask`、`srs_pilot_symbols`、`srs_re_symbol_indices`、
     `srs_re_subcarrier_indices`、`srs_port_tx_ant_map`、PRB/cyclic-shift/power metadata
   - schema `1.5.0` 后不再写 `/waveform/pilot_code`、`/waveform/srs_port_index`、`/observation/srs_cfr_est`、`/array/spatial_spectrum_label` 或 `/array/spatial_spectrum_srs`
9. **BLER 契约**（NR PUSCH）：
   - `num_blocks > 0`
   - `0 <= num_block_errors <= num_blocks`
   - `bler == num_block_errors / num_blocks`
10. **Ranging 契约**（当 `/ranging` 存在）：
    - 必须已有 `/observation/cfr_est`
    - `pdp_peak` / `phase_slope` 输出 shape 为 `[snapshot, tx, rx]`
    - 成功位置为 finite，失败位置为 NaN，`selected_delay_bin` 失败为 -1

### `hdf5_reader.py`

提供便捷的 HDF5 读回函数，自动调用 `validate_hdf5_contract`。

### `manifest.py`

```python
def write_manifest(path: str | Path, data: dict) -> Path
```

输出 JSON manifest，记录运行参数、CFR shape、路径数、耗时、诊断摘要。启用
`output.sharding.enabled=true` 时，`manifest/manifest.json` 是 aggregate 入口，记录每个
`results/result_xxx.h5` 的全局 BS/UE 索引、resolved TX/RX 索引、schema/debug 信息和性能摘要。
`manifest/` 同目录还会保存 `config_snapshot.json`；CLI 运行时，输出根目录会保存最终 YAML
`run_config.yaml`；若发生自动 shard fallback，会额外写 `shard_attempts.jsonl`。
队列、验收、可视化后处理等 wrapper 产生的运行级 artifact 必须继续写在同一个 run
目录内：`logs/run.log`、`logs/heatmap.log` 和 `summary.json` 是推荐位置。不要把
`<run_name>.run.log`、`<run_name>.heatmap.log` 或 `<run_name>_summary.json` 写到
`outputs/` 根目录。

### `perf.py` 与 benchmark artifact

`PerfTracer` 是 opt-in 性能追踪工具。开启 debug 或 benchmark 时，它会写：

- `logs/perf_events*.jsonl`
- `logs/hardware_samples*.csv`（可关闭）
- `logs/perf_summary*.json`

summary 在成功和失败运行中都会尽量写出，包含 `status`、`exception`、
`stage_totals_s`、`hardware_summary` 和 `dataset_write_summary`。HDF5 writer 会对每个
dataset 写入发出 `hdf5.dataset_write` event，因此 `benchmark write` 可以直接复用这套
统计来比较 raw bytes、storage bytes、compression ratio 和最慢/最大 dataset。

`benchmark rt/write/spectrum` 输出 `benchmark_summary.json`、`benchmark_rows.csv` 和
`config_snapshot.json`；这些是性能 artifact，不属于正式 HDF5 schema。

### `label_parser.py`

```python
def load_role_topology_from_label(label_file: Path, max_bs: int, max_ue: int) -> RoleTopology
```

解析标准 label `0.1.0` JSON，提取顶层 `bs_points`/`ue_points` 作为全场景
BS/UE 位置；`groups` 仅作为子集元数据，不参与默认 topology 选择。pipeline 随后根据
`link.phy_link_direction` 解析为 TX/RX link-view topology。

## 测试体系

测试位于 `tests/`，按本文件的质量门和各测试目录职责组织。

### 测试分类

| 目录 | 内容 | 示例 |
|------|------|------|
| `tests/unit/` | 单元测试 | domain 模型、config 加载、impairments、MIMO channel bridge、MIMO config/detector builder、PerfTracer |
| `tests/schema/` | HDF5 schema 测试 | truth schema、CIR schema、NR PUSCH/SRS schema |
| `tests/adapter/` | Sionna adapter 测试 | RT shape contracts、CIR adapter、truth adapter |
| `tests/integration/` | 集成测试 | RT truth pipeline、4x4 SU-MIMO、MU-MIMO、batch、calibration、benchmark CLI |
| `tests/statistical/` | 统计测试 | AWGN observation、impairments、motion、NR PUSCH link metrics、MIMO metrics |

### 关键测试文件

```
tests/
├── unit/
│   ├── test_nr_mimo_channel_bridge.py    # CIR→CFR shape + backend comparison (21 tests)
│   ├── test_nr_pusch_mimo_config.py      # multi-user configs + detector builder (13 tests)
│   ├── test_nr_pusch_config.py           # PUSCHConfig + run_nr_pusch_observation (14 tests)
│   ├── test_common_link.py               # 通用 impairment/AWGN 链路
│   ├── test_observation_pipeline.py      # AWGN + LS 估计
│   ├── test_impairments.py               # CFO/SFO/相偏/定时偏/削波
│   ├── test_reciprocity.py               # TDD 互易性 transpose
│   ├── test_domain_models.py             # domain dataclass 校验
│   └── test_config_loader.py             # YAML → pydantic 加载
├── schema/
│   ├── test_hdf5_schema.py               # 基础 HDF5 契约
│   ├── test_rt_cir_schema.py             # CIR shape/dtype 校验
│   ├── test_nr_pusch_schema.py           # NR PUSCH 字段 + BLER 契约
│   └── test_nr_srs_schema.py             # NR SRS resource 字段 + schema 契约
├── integration/
│   ├── test_rt_truth_pipeline.py         # RT 最小闭环
│   ├── test_rt_mimo_4x4_pipeline.py      # RT 4x4 MIMO CFR shape
│   ├── test_nr_pusch_observation.py      # NR PUSCH 自生成 HDF5
│   ├── test_nr_pusch_mimo_observation.py # 4x4 SU-MIMO (8 tests)
│   ├── test_nr_pusch_mu_mimo_observation.py  # MU-MIMO (5 tests)
│   └── test_batch.py                     # 批量实验
└── statistical/
    ├── test_awgn_observation.py          # SNR-NMSE 单调性
    ├── test_impairments_statistical.py   # CFO/削波 统计
    ├── test_nr_pusch_link_metrics.py     # NR PUSCH BER/BLER 随 Eb/N0
    └── test_nr_pusch_mimo_metrics.py     # 4x4 MIMO perfect vs estimated CSI (7 tests)
```

### 质量门

```bash
uv run ruff check .      # lint
uv run pytest            # 全量测试，以当前输出为准；最近结果 223 passed, 19 skipped
```

涉及 adapter 变更还需：
```bash
uv run pytest tests/adapter tests/integration
```

涉及 schema 变更还需：
```bash
uv run pytest tests/schema
```

## 禁止事项

- 不允许 skip 失败测试后声称通过
- 不允许 schema 改了但文档不改
- 不允许大输出文件入 git
- 不允许业务层绕过 adapter 直接读取 Sionna Paths
- 不允许测试只检查"文件存在"而不检查关键 dataset/shape/dtype
