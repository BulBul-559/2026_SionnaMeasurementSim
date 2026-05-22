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
├── common_link.py            # 通用 clean grid → impairment → AWGN 链路
├── observation_pipeline.py   # custom OFDM: AWGN + LS 估计
├── impairments.py            # 基带损伤模型
├── reciprocity.py            # TDD 互易性 transpose
├── nr_mimo_channel.py        # CIR → CFR 桥接 + 维度转换
├── nr_channel_backend.py     # 可插拔信道后端
├── nr_pusch_observation.py   # NR PUSCH 主链路 (SU/MU-MIMO)
└── nr_srs_observation.py     # NR SRS standards-shaped v2 subset
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
`waveform_extras` 用于写 NR PUSCH/SRS 的频域 grid 和专属元数据。

## 一、通用 PHY link (`common_link.py`)

`ObservationImpairmentChain` 是当前 SRS/PUSCH 共享的底层观测链：

```
rx_grid_clean -> apply_base_impairments -> AWGN -> rx_grid
```

输入 shape 统一为 `[snapshot, tx, rx, ..., subcarrier]`。前 3 维是 resolved
link-view，后续维度由具体标准决定，例如 SRS/PUSCH 都使用
`[antenna, ofdm_symbol, subcarrier]`。该链输出：

- `rx_grid`：施加基带损伤和 AWGN 后的频域接收 grid。
- `noise_variance`：`[snapshot, tx, rx]`，由 common chain 统一写入 waveform。
- `ImpairmentSample`：CFO/SFO/timing/phase/AGC/clipping per-link 观测值。
- `ImpairmentSpec`：写入 `/impairments` 的配置快照。

SRS、PUSCH 只负责各自的 waveform builder 和 receiver/estimator。后续新增
WiFi-like 或 6G waveform 时，也应复用这个 clean-grid impairment 链，而不是在
标准模块内重写 AWGN 或损伤 metadata。

waveform-level ranging 不属于 SRS/PUSCH 私有逻辑。pipeline 会在 PHY observation
完成后读取统一的 `/observation/cfr_est`，由 `sionna_measurement_sim.ranging`
中的 estimator 生成 `/ranging`。因此后续 WiFi-like 只要能产出同口径 `cfr_est`
和频率网格，就可以复用同一套 PDP peak / phase-slope ToA 估计。

`custom_ofdm` 仍保留在旧 `observation_pipeline.py` 中作为 legacy 路径，用于历史
测试和快速 sanity check；它暂不迁移到通用链路，后续单独移除或重写。

## 二、Custom OFDM 观测 (`observation_pipeline.py`)

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

## 三、Impairments (`impairments.py`)

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

## 四、NR MIMO Channel Bridge (`nr_mimo_channel.py`)

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
- 当前用户配置口径已经使用 direct link mapping；这个函数只应视为 legacy/fallback
  工具，不是 SRS/PUSCH direct uplink 的默认流程。

## 五、Channel Backend (`nr_channel_backend.py`)

可插拔信道后端，封装了信道构建、perfect-CSI 提取和信道施加。通过 `channel_backend` 配置字段切换。

### 接口约定

两个 backend 实现相同的 duck-typed 接口：

| 方法 | 用途 |
|------|------|
| `build(cir_coeff, cir_delays, link, sc_hz, nsub)` | 工厂方法 |
| `perfect_h(snap, ul_tx, ul_rx, sym)` | 单 link perfect-CSI tensor |
| `apply_clean_with_h(...)` | clean channel apply，返回 `ChannelApplyResult(y_clean, h)` |
| `apply_clean_batch_with_h(...)` | SU batched clean channel apply |
| `apply_clean_full_with_h(...)` | MU-MIMO clean channel apply |
| `apply(x, no, snap, ul_tx, ul_rx, sym, rg)` | legacy/noisy apply，保留兼容 |
| `apply_with_h(...)` | legacy/noisy apply，返回 `ChannelApplyResult(y, h)` |
| `apply_full_with_h(x, no, snap, sym, rg)` | legacy/noisy 全 multi-TX/RX 施加 |
| `cfr` (property) | 预计算 CFR（用于 NMSE/一致性） |
| `num_snap / num_ul_tx / num_ul_rx / num_ul_tx_ant / num_ul_rx_ant` | 维度属性 |

### `ApplyOFDMChannelBackend`（稳定后端）

- 在 `build()` 时预计算全量 CFR（`build_mimo_cfr_from_cir` → `cir_to_ofdm_channel`）
- `perfect_h()` 从预计算 CFR 切片
- `apply_clean_*()` 使用 Sionna `ApplyOFDMChannel` 只施加 clean channel
- `apply()` / `apply_with_h()` 仍保留旧 noisy apply 兼容入口
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

## 六、NR PUSCH 主链路 (`nr_pusch_observation.py`)

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
10. 计算目标 noise 口径 (ebnodb2no 或 SNR)
11. backend clean apply 得到 `y_clean, h`
12. `ObservationImpairmentChain` 对 `y_clean` 统一施加 impairment/AWGN
13. PUSCHReceiver 使用 impaired `rx_grid` 和 common chain 返回的 receiver noise
14. if su_mimo: _process_su_mimo_per_link(...)
    if mu_mimo: _process_mu_mimo(...)
15. 组装 resolved TX/RX link-view 的 ObservationResult + EvaluationResult
16. 返回 dict
```

## 七、NR SRS Subset (`nr_srs_observation.py`)

`nr_srs` 当前是 standards-shaped NR SRS v2 subset，不是完整 3GPP NR SRS：

1. 读取 resolved link-view truth CFR；uplink 下 shape 为 `[tx=ue, rx=bs, bs_ant, ue_ant, subcarrier]`
2. 按 `phy.srs` 生成 full-slot 14-symbol `tx_grid`，非 SRS symbol 为 0
3. comb/BWP/hopping resource 写入 `srs_resource_mask`、`srs_re_symbol_indices` 和 `srs_re_subcarrier_indices`
4. 生成可复现 `zc_like` 或 deterministic `nr_zc` pilot，可启用 group/sequence hopping
5. 按 `ports` 生成 `srs_port_tx_ant_map`，支持 one-to-one 与简化 antenna switching
6. 可选按 RT `path_power_db` 做 open-loop SRS power scaling，写 `srs_tx_power_dbm` 和 `srs_power_scale_linear`
7. 通过 clean channel 得到 `y_clean`
8. `ObservationImpairmentChain` 统一施加 impairment/AWGN 得到 `rx_grid`
9. receiver 从 flattened SRS RE 做 time-code LS 或 cyclic-shift delay-window despreading，写 `/observation/cfr_est_resource`
10. 频域 linear interpolation 得到 full-band `/observation/cfr_est`

SRS 不输出 BER/BLER，`evaluation` 包含 full-band NMSE、幅度/相位误差、correlation、
estimation success，以及 SRS resource NMSE / interpolation NMSE / resource SNR 等
quality 指标。若 `array.spectrum.sources` 包含 `cfr_est`，会从统一
`/observation/cfr_est` 生成 `/array/spatial_spectrum_cfr_est`；schema 1.5.0 后不再接受
历史 `srs_cfr_est` source，也不再写 `/array/spatial_spectrum_srs`。

`/array/spatial_spectrum_*` 使用 scene/global 角度网格。实现时先按 Sionna
`PlanarArray` 的本地 y-z 平面、top-left 起、column-first 编号、第一行 z 为正生成
接收阵列元素位置，再用每个 RX 的 `/devices/rx_orientation_rad` 旋转到 scene
坐标后构造 Bartlett steering vector。可视化阶段按 `/link/tx_role` 和
`/link/rx_role` 将 link-view 轴映射回 BS/UE 标签，direct uplink 中
`tx=UE, rx=BS` 不再按历史 `BS->UE` 假设取图。

schema `1.5.0` 后，NR SRS HDF5 使用统一 `/waveform/tx_grid`、`rx_grid`、
`noise_variance`，并写 SRS 专属 `/waveform/srs_resource_mask`、
`/waveform/srs_pilot_symbols`、`/waveform/srs_re_symbol_indices`、
`/waveform/srs_re_subcarrier_indices`、`/waveform/srs_port_tx_ant_map`、
per-symbol PRB、cyclic shift、sequence 和 power-control metadata。不再写
`/waveform/pilot_code`、`/waveform/srs_port_index`、`/observation/srs_cfr_est`、
`/array/spatial_spectrum_srs` 或 `/array/spatial_spectrum_label`。

当前 v2 已覆盖 group/sequence hopping、同 symbol cyclic-shift port multiplexing、
frequency/bandwidth hopping、port/antenna switching 口径和简化 power scaling；
仍未做 38.211/38.213 reference 对齐、完整 antenna switching procedure、闭环功控或
3GPP-compliant 声明，详见 `docs/todo/feature.md`。

### SU-MIMO per-link 处理：`_process_su_mimo_per_link()`

对每个 `(snap, ul_tx, ul_rx)` 独立调用 `_process_one_pusch_link()`：

```python
def _process_one_pusch_link(snap, ul_tx, ul_rx, backend, tx, rx, no,
                             perfect_csi, num_ofdm_symbols, failure_policy):
    # 1. tx(1) → tx_signal, tx_bits
    # 2. backend.apply_clean_with_h(tx_signal, ...) → ChannelApplyResult(y_clean, h)
    # 3. ObservationImpairmentChain(y_clean) → y, noise_variance, impairment metadata
    # 4. if perfect_csi: rx(y, no, h_perfect) → rx_bits, tb_crc_status
    #    else:           rx(y, no) with internal PUSCH LS estimator
    # 5. export cfr_est_slice:
    #    perfect_csi=True  → pusch_h_to_cfr_est(h_perfect)
    #    perfect_csi=False → external PUSCHLSChannelEstimator(y, no)
    # 6. BER/BLER from bit errors + TB CRC status
    # 7. NMSE: cfr_est_slice vs truth_slice
```

SU-MIMO 支持两条路径：`su_mimo_link_batch_size <= 1` 时使用逐 link 稳定路径；大于 1 时使用 batched path，把多个独立 `(snapshot, ul_tx, ul_rx)` link 合成一个 PUSCHTransmitter/ApplyOFDMChannel/PUSCHReceiver batch。batch 失败时会递归降级到更小 batch，最终可回退到单 link，并在 manifest 的 `nr_pusch_batching` 记录 fallback 统计。

GPU 大规模生产运行建议同时开启 UE shard：多个进程分别绑定 GPU、分别写 `results/result_xxx.h5`，避免多个进程竞争同一个 HDF5 文件。训练和分析侧应以 `manifest/manifest.json` 为入口，因为 fallback 可能把单个计划 shard 拆成多个子 result 文件。

### MU-MIMO per-snapshot 处理：`_process_mu_mimo()`

每个 snapshot 一次 joint PUSCH 调用：

```python
def _process_mu_mimo(backend, tx, rx, no, ...):
    for snap_idx:
        # 1. cfr_to_full_mimo_h() → 全 TX/RX h_full
        # 2. tx(1) → 所有 UE 的联合 tx_signal
        # 3. backend.apply_clean_full_with_h() → ChannelApplyResult(y_clean, h_full)
        # 4. ObservationImpairmentChain(y_clean) → y, noise_variance, metadata
        # 5. rx(y, no, h_full) → rx_bits, tb_crc_status
        # 6. 从 h_full 提取 per-link CFR slice
        # 7. BER/BLER (joint, 不重复累计)
        # 8. per-link NMSE
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
- **`perfect_csi`**：`h_perfect` 来自 clean backend 返回的 `ChannelApplyResult.h`。
  `perfect_csi=true` 时 receiver 使用 oracle CSI，但输入仍是 impaired `rx_grid`；
  导出的 `cfr_est` 是 oracle channel CSI。`perfect_csi=false` 时 receiver 不接收
  真值 `h`，而是使用 PUSCHReceiver 内部 DMRS LS；导出的 `cfr_est` 来自外部
  `PUSCHLSChannelEstimator(y, no)`，与 receiver 使用同一个 impaired `rx_grid` 和
  common-chain noise 口径。
- **SNR/noise 口径**：普通 `snr_db` 路径下 common chain 按 clean `rx_grid` 的每
  link 平均功率计算 `/waveform/noise_variance`；只有 `ebno_db` 非空时，PUSCH 使用
  Sionna `ebnodb2no` 结果作为 override。

## 八、Waveform Ranging Observation (`ranging/`)

`ranging.enabled=true` 时，pipeline 在 derived truth 和 PHY observation 都生成后运行
独立 ranging runner。v1 只支持 `source: "cfr_est"`；如果开启 ranging 但没有
`/observation/cfr_est`，pipeline 会 fail-fast。

输出语义：

- `/derived/first_path_delay_s` 和 `/derived/first_path_propagation_range_m` 是 RT/path truth。
- `/ranging/pdp_peak/*` 是从 `cfr_est` 的 IFFT PDP 峰值估计得到的 ToA/range。
- `/ranging/phase_slope/*` 是 phase vs subcarrier frequency 斜率诊断 estimator。
- `/ranging/*/rtt_equiv_s = 2 * toa_est_s` 只是 two-way equivalent，不是 MAC/协议 RTT。

PDP peak estimator 会对每条 `[snapshot, tx, rx]` link 的天线 pair PDP 做非相干平均，
选择相对最强峰超过阈值的最早可检测峰，并用 log-power parabola 做亚 bin 插值。
相近多径低于带宽分辨率时可能合并成一个主瓣，这是 estimator 的物理限制，不应把
它解释为 RT truth 错误。
