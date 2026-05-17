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
/waveform /observation /impairments /receiver /evaluation
/calibration /motion /runtime
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
6. **禁止路径**：`/channel/cfr` 必须不存在
7. **NR PUSCH 专有字段**（当 `waveform/standard == "nr_pusch"`）：
   - `num_prb`、`num_layers`、`num_antenna_ports`、`mimo_detector` 等
   - `num_layers >= 1`、`num_antenna_ports >= num_layers`
8. **NR SRS-like 专有字段**（当 `waveform/standard == "nr_srs"`）：
   - `srs_tx_grid`、`srs_rx_grid`、`srs_noise_variance`、`srs_pilot_code`
   - `/observation/srs_cfr_est` 与 `/observation/cfr_est` shape 一致
9. **BLER 契约**（NR PUSCH）：
   - `num_blocks > 0`
   - `0 <= num_block_errors <= num_blocks`
   - `bler == num_block_errors / num_blocks`

### `hdf5_reader.py`

提供便捷的 HDF5 读回函数，自动调用 `validate_hdf5_contract`。

### `manifest.py`

```python
def write_manifest(path: str | Path, data: dict) -> Path
```

输出 JSON manifest，记录运行参数、CFR shape、路径数、耗时、诊断摘要。

### `label_parser.py`

```python
def load_topology_from_label(label_file: Path, max_tx: int, max_rx: int) -> Topology
```

解析测试标签 JSON（`tests/fixtures/scenes/test/test5.json`），提取 BS/UE 位置。

## 测试体系

测试位于 `tests/`，按本文件的质量门和各测试目录职责组织。

### 测试分类

| 目录 | 内容 | 示例 |
|------|------|------|
| `tests/unit/` | 单元测试 | domain 模型、config 加载、impairments、MIMO channel bridge、MIMO config/detector builder |
| `tests/schema/` | HDF5 schema 测试 | truth schema、CIR schema、NR PUSCH/SRS schema |
| `tests/adapter/` | Sionna adapter 测试 | RT shape contracts、CIR adapter、truth adapter |
| `tests/integration/` | 集成测试 | RT truth pipeline、4x4 SU-MIMO、MU-MIMO、batch、calibration |
| `tests/statistical/` | 统计测试 | AWGN observation、impairments、motion、NR PUSCH link metrics、MIMO metrics |

### 关键测试文件

```
tests/
├── unit/
│   ├── test_nr_mimo_channel_bridge.py    # CIR→CFR shape + backend comparison (21 tests)
│   ├── test_nr_pusch_mimo_config.py      # multi-user configs + detector builder (13 tests)
│   ├── test_nr_pusch_config.py           # PUSCHConfig + run_nr_pusch_observation (14 tests)
│   ├── test_observation_pipeline.py      # AWGN + LS 估计
│   ├── test_impairments.py               # CFO/SFO/相偏/定时偏/削波
│   ├── test_reciprocity.py               # TDD 互易性 transpose
│   ├── test_domain_models.py             # domain dataclass 校验
│   └── test_config_loader.py             # YAML → pydantic 加载
├── schema/
│   ├── test_hdf5_schema.py               # 基础 HDF5 契约
│   ├── test_rt_cir_schema.py             # CIR shape/dtype 校验
│   ├── test_nr_pusch_schema.py           # NR PUSCH 字段 + BLER 契约
│   └── test_nr_srs_schema.py             # NR SRS-like 字段 + schema 契约
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
uv run pytest            # 190 collected / 188 passed / 2 skipped
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
