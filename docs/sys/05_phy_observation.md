# 05. PHY 观测与 NR PUSCH MIMO

`phy/` 目录实现 PHY 层观测链路。当前通过 PHY module registry 接入
`custom_ofdm`、`nr_pusch`、`nr_srs` 三条路径。

官方参考：
- [Sionna 5G NR PUSCH Tutorial](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html)
- [Sionna RT Link-Level Tutorial](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Link_Level_Simulations_with_RT.html)
- [Sionna OFDM Channel API](https://nvlabs.github.io/sionna/phy/api/channel.html)

## 文件结构

```
phy/
├── modules.py                # PHYContext / PHYModuleResult / registry / built-in adapters
├── observation_pipeline.py   # custom OFDM: AWGN + LS 估计
├── impairments.py            # 基带损伤模型
├── reciprocity.py            # TDD 互易性 transpose
├── nr_mimo_channel.py        # CIR → CFR 桥接 + 维度转换
├── nr_channel_backend.py     # 可插拔信道后端
├── nr_pusch_observation.py   # NR PUSCH 主链路 (SU/MU-MIMO)
└── nr_srs_observation.py     # NR SRS-like full-band sounding
```

## 零、PHY Module Registry (`modules.py`)

pipeline 不直接分支调用某个 PHY 文件，而是通过 registry 查找模块：

```python
PHY_REGISTRY = {
    "custom_ofdm": CustomOFDMModule(),
    "nr_pusch": NRPUSCHModule(),
    "nr_srs": NRSRSModule(),
}
```

模块接收 `PHYContext(config, adapter_result)`，返回 `PHYModuleResult`。其中
`waveform`、`observation`、`receiver`、`evaluation` 等字段最终写入 HDF5；
`waveform_extras` 用于写 NR PUSCH/SRS-like 的频域 grid 和专属元数据。

## 一、Custom OFDM 观测 (`observation_pipeline.py`)

简化链路，用于快速验证和 impairment 测试：

```python
def run_awgn_ls_observation(
    h_true: np.ndarray,           # [tx, rx, rx_ant, tx_ant, subcarrier]
    config: AWGNObservationConfig,
    has_signal: np.ndarray,        # [tx, rx] bool
    cfr_snapshots: np.ndarray | None = None,  # 6-D snapshots
) -> PHYObservationBundle
```

**处理流程：**
1. 从 `h_true` 构造导频符号（全子载波激活）
2. 施加基带损伤（`apply_base_impairments`）
3. 添加 AWGN
4. LS 信道估计
5. 计算 NMSE / 幅度误差 / 相位误差 / 相关系数
6. 返回 `PHYObservationBundle`（包含 `WaveformSpec`、`ObservationResult`、`EvaluationResult`）

## 二、Impairments (`impairments.py`)

```python
@dataclass(frozen=True)
class ImpairmentConfig:
    random_seed: int = 142
    cfo_hz: float | None = None       # 载波频偏
    sfo_ppm: float | None = None      # 采样频偏
    phase_offset_rad: float | None = None
    timing_offset_samples: float | None = None
    agc_gain_db: float = 0.0
    clipping_threshold: float | None = None

def apply_base_impairments(
    cfr: torch.Tensor,
    fft_size: int,
    sample_rate_hz: float,
    config: ImpairmentConfig,
) -> tuple[torch.Tensor, ImpairmentSample]
```

施加顺序：IFFT → CFO（时域）→ FFT → SFO → 相偏 → 定时偏 → AGC/ADC 削波。

## 三、NR MIMO Channel Bridge (`nr_mimo_channel.py`)

将项目 CIR（`[snap, tx, rx, rx_ant, tx_ant, path]`）转换为 Sionna PUSCH 兼容的 CFR 格式。

### `PUSCHMIMOChannel`

```python
@dataclass(frozen=True)
class PUSCHMIMOChannel:
    cfr: np.ndarray                # [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, subcarrier]
    num_snap: int; num_ul_tx: int; num_ul_rx: int
    num_ul_tx_ant: int; num_ul_rx_ant: int
    num_subcarriers: int
    reciprocity_applied: bool
```

> **Link-view 约定**：CFR 在 channel bridge 内部已经是 resolved TX/RX 视角。
> `phy_link_direction="uplink"` 时 TX=UE、RX=BS；`downlink` 时 TX=BS、RX=UE。
> 旧的 reciprocity transpose 仅作为低层 legacy fallback，不是当前用户配置口径。

### 核心函数

```python
def build_mimo_cfr_from_cir(cir_coeff, cir_delays, link_config,
                             sc_spacing_hz, num_subcarriers) -> PUSCHMIMOChannel
```
- 读取 resolved link-view CIR
- 必要时只对 legacy fallback 应用 TDD 互易性 transpose
- 调用 Sionna `cir_to_ofdm_channel` 将 CIR 转为 CFR

```python
def cfr_to_pusch_perfect_h(channel, snap_idx, ul_tx_idx, ul_rx_idx,
                            num_ofdm_symbols) -> torch.Tensor
```
- 提取单 link 的 MIMO CFR
- 输出 shape：`[1, 1, rx_ant, 1, tx_ant, num_ofdm_symbols, subcarrier]`
- 用于 SU-MIMO per-link perfect CSI

```python
def cfr_to_full_mimo_h(channel, snap_idx, num_ofdm_symbols) -> torch.Tensor
```
- 提取全 multi-TX/RX MIMO CFR
- 输出 shape：`[1, num_ul_rx, rx_ant, num_ul_tx, tx_ant, sym, subcarrier]`
- 用于 MU-MIMO joint processing

```python
def pusch_h_to_cfr_est(h: torch.Tensor) -> np.ndarray
```
- 将 PUSCH perfect-CSI `h` 张量转回 3-D CFR slice
- 取第一个 OFDM 符号（静态信道假设）

```python
def reverse_reciprocity_cfr(cfr: np.ndarray) -> np.ndarray
```
- 将 UL 视角 CFR 反转为 DL 视角（TX/RX 角色互换 + 天线维度互换）

## 四、Channel Backend (`nr_channel_backend.py`)

可插拔信道后端，封装了信道构建、perfect-CSI 提取和信道施加。通过 `channel_backend` 配置字段切换。

### 接口约定

两个 backend 实现相同的 duck-typed 接口：

| 方法 | 用途 |
|------|------|
| `build(cir_coeff, cir_delays, link, sc_hz, nsub)` | 工厂方法 |
| `perfect_h(snap, ul_tx, ul_rx, sym)` | 单 link perfect-CSI tensor |
| `apply(x, no, snap, ul_tx, ul_rx, sym, rg)` | 施加信道，返回 y |
| `apply_with_h(...)` | 返回 `ChannelApplyResult(y, h)` |
| `apply_full_with_h(x, no, snap, sym, rg)` | 全 multi-TX/RX 施加（MU-MIMO） |
| `cfr` (property) | 预计算 CFR（用于 NMSE/一致性） |
| `num_snap / num_ul_tx / num_ul_rx / num_ul_tx_ant / num_ul_rx_ant` | 维度属性 |

### `ApplyOFDMChannelBackend`（稳定后端）

- 在 `build()` 时预计算全量 CFR（`build_mimo_cfr_from_cir` → `cir_to_ofdm_channel`）
- `perfect_h()` 从预计算 CFR 切片
- `apply()` / `apply_with_h()` 使用 Sionna `ApplyOFDMChannel` 施加信道
- `apply_full_with_h()` 使用 `cfr_to_full_mimo_h` + `ApplyOFDMChannel`
- 支持 SU-MIMO 和 MU-MIMO

### `CIRDatasetOFDMChannelBackend`（官方 API 后端）

- 在 `build()` 时也预计算 CFR（用于 `cfr` 属性和 NMSE）
- 但 `apply_with_h()` 使用官方的 `CIRDataset` + `OFDMChannel(return_channel=True)` 路径
- 每个 link 创建新的 `CIRDataset` generator
- **delay 处理**：`CIRDataset` 要求 link-level `tau: [num_rx, num_tx, path]`（非 per-antenna-pair）。使用 per-link **median over (rx_ant, tx_ant)** 作为 shared delay
- `shared_tau_for_link(snap, ul_tx, ul_rx)` 公开方法返回使用的 median delay
- `apply_full_with_h()` 抛 `NotImplementedError`（不支持 MU-MIMO）

### `ChannelApplyResult`

```python
class ChannelApplyResult:
    __slots__ = ("y", "h")
    def __init__(self, y: torch.Tensor, h: torch.Tensor): ...
```

### Backend 工厂

```python
def create_channel_backend(..., *, backend_name: str = "apply_ofdm"):
```
- `"apply_ofdm"` → `ApplyOFDMChannelBackend`
- `"cir_dataset_ofdm"` → `CIRDatasetOFDMChannelBackend`

### Backend 支持矩阵

| mimo_mode | apply_ofdm | cir_dataset_ofdm |
|-----------|-----------|-----------------|
| su_mimo | ✅ | ✅ (per-link) |
| mu_mimo | ✅ | ❌ (入口拒绝) |

## 五、NR PUSCH 主链路 (`nr_pusch_observation.py`)

### 入口：`run_nr_pusch_observation()`

```python
def run_nr_pusch_observation(
    cir_coefficients: np.ndarray,   # [snap, tx, rx, rx_ant, tx_ant, path]
    cir_delays: np.ndarray,
    link_config: LinkConfig,
    phy_config,                     # PHYConfig 或 RTTruthRunConfig
    carrier_config,
) -> dict
```

**完整流程：**

```
1. 提取配置 (perfect_csi, mimo_mode, channel_backend, ...)
2. 校验 mimo_mode ∈ {su_mimo, mu_mimo}
3. create_channel_backend(...)  → backend
4. 校验 mu_mimo + cir_dataset_ofdm → NotImplementedError
5. build_multiuser_pusch_configs(phy_config, carrier_config, num_pusch_tx=...)
6. PUSCHTransmitter(configs, output_domain="freq", return_bits=True)
7. build_stream_management(num_rx, num_tx, num_layers)
8. build_mimo_detector(resource_grid, stream_management, ...)
9. PUSCHReceiver(tx, channel_estimator, mimo_detector, stream_management)
10. 计算 noise (ebnodb2no 或 SNR)
11. if su_mimo: _process_su_mimo_per_link(...)
    if mu_mimo: _process_mu_mimo(...)
12. reverse_reciprocity_cfr() → 恢复 DL 视角
13. 组装 ObservationResult + EvaluationResult
14. 返回 dict
```

## 六、NR SRS-like Sounding (`nr_srs_observation.py`)

`nr_srs` 当前是 full-band SRS-like uplink sounding，不是完整 3GPP NR SRS：

1. 将 RT truth CFR 投影到 uplink 接收端视角 `[snap, ue, bs, bs_ant, ue_ant, subcarrier]`
2. 所有 active subcarrier 发送已知 pilot
3. UE 多天线通过 OFDM symbol 维度的正交 DFT code 分离
4. 通过信道和 AWGN 得到 `srs_rx_grid`
5. LS 估计 `H_hat = Y X^H / N_symbol`
6. 转回项目 HDF5 的 DL 视角，写入 `/observation/cfr_est` 和 `/observation/srs_cfr_est`

SRS-like 不输出 BER/BLER，`evaluation` 只包含 NMSE、幅度/相位误差、correlation
和 estimation success。若 `array.spectrum.sources` 包含 `srs_cfr_est`，会从
SRS LS 估计信道生成 `/array/spatial_spectrum_srs`。

标准 NR SRS 的 comb、sequence、cyclic shift、hopping 等尚未实现，详见
`docs/sys/nr_srs_standard_todo.md`。

### SU-MIMO per-link 处理：`_process_su_mimo_per_link()`

对每个 `(snap, ul_tx, ul_rx)` 独立调用 `_process_one_pusch_link()`：

```python
def _process_one_pusch_link(snap, ul_tx, ul_rx, backend, tx, rx, no,
                             perfect_csi, num_ofdm_symbols, failure_policy):
    # 1. tx(1) → tx_signal, tx_bits
    # 2. backend.apply_with_h(tx_signal, no, ...) → ChannelApplyResult(y, h)
    # 3. if perfect_csi: rx(y, no, h_perfect) → rx_bits, tb_crc_status
    #    else:           PUSCHLSChannelEstimator + rx(y, no)
    # 4. pusch_h_to_cfr_est(h_perfect) → cfr_est_slice
    # 5. BER/BLER from bit errors + TB CRC status
    # 6. NMSE: cfr_est_slice vs truth_slice
```

SU-MIMO 支持两条路径：`su_mimo_link_batch_size <= 1` 时使用逐 link 稳定路径；大于 1 时使用 batched path，把多个独立 `(snapshot, ul_tx, ul_rx)` link 合成一个 PUSCHTransmitter/ApplyOFDMChannel/PUSCHReceiver batch。batch 失败时会递归降级到更小 batch，最终可回退到单 link，并在 manifest 的 `nr_pusch_batching` 记录 fallback 统计。

GPU 大规模生产运行建议同时开启 UE shard：多个进程分别绑定 GPU、分别写 `result_xxx.h5`，避免多个进程竞争同一个 HDF5 文件。

### MU-MIMO per-snapshot 处理：`_process_mu_mimo()`

每个 snapshot 一次 joint PUSCH 调用：

```python
def _process_mu_mimo(backend, tx, rx, no, ...):
    for snap_idx:
        # 1. cfr_to_full_mimo_h() → 全 TX/RX h_full
        # 2. tx(1) → 所有 UE 的联合 tx_signal
        # 3. backend.apply_full_with_h() → ChannelApplyResult(y, h_full)
        # 4. rx(y, no, h_full) → rx_bits, tb_crc_status
        # 5. 从 h_full 提取 per-link CFR slice
        # 6. BER/BLER (joint, 不重复累计)
        # 7. per-link NMSE
```

### 辅助函数

```python
def build_multiuser_pusch_configs(phy_config, carrier_config, *,
    num_pusch_tx: int | None = None) -> list[PUSCHConfig]
```
- SU-MIMO：返回 1 个 `PUSCHConfig`
- MU-MIMO：返回 N 个 `PUSCHConfig`，每个 UE 的 `dmrs_port_set` 不重叠
- `num_layers < num_antenna_ports` 时设置 `precoding="codebook"`

```python
def build_stream_management(num_rx, num_tx, num_layers) -> StreamManagement
```
- 创建 `rx_tx_association = ones([num_rx, num_tx])` 的全连接 MIMO

```python
def build_mimo_detector(resource_grid, stream_management,
    detector_type="lmmse", num_bits_per_symbol=4
) -> LinearDetector | KBestDetector
```

### 关键约束

- **estimated CSI**：`num_layers == num_antenna_ports` 时支持；不等时抛 `NotImplementedError`（DMRS LS 估计返回 effective stream channel，不能直接写成 physical antenna-pair CFR）
- **MU-MIMO bit counter**：`total_bit_errors` / `total_bits` 按 snapshot 累计一次（joint），不在 per-link 循环中重复累计
- **`perfect_csi`**：`h_perfect` 来自 `backend.apply_with_h()` 返回的 `ChannelApplyResult.h`，确保 PUSCHReceiver 使用的 CSI 与实际施加的信道一致
