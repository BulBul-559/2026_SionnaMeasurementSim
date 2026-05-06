# 05. PHY 观测链与硬件损伤

本文定义如何从传播真值 `H_true` 生成接近真实测量的观测信道 `H_obs`。输出字段必须遵循 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)，配置字段必须遵循 [06_config_and_experiment_schema.md](06_config_and_experiment_schema.md)。

官方参考：

- Sionna PHY documentation: https://nvlabs.github.io/sionna/phy/index.html
- Link-level simulations with Sionna RT: https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Link_Level_Simulations_with_RT.html
- Sionna installation and PyTorch requirement: https://nvlabs.github.io/sionna/installation.html

## 1. 核心目标

PHY 观测链回答的问题不是“物理信道是什么”，而是：

```text
在给定波形、导频、噪声、硬件损伤和接收机算法后，设备估计出了什么信道？
```

因此，PHY pipeline 输入是 `H_true` 或 CIR，输出是：

- `H_obs` / `cfr_est`
- 接收诊断量
- 失败标志
- 质量指标

## 2. 最小观测链

第一版必须实现：

```text
H_true
  -> OFDM resource grid
  -> pilot symbols
  -> channel application
  -> AWGN
  -> LS channel estimation
  -> H_obs
  -> NMSE(H_obs, H_true)
```

此阶段不要求完整协议，但必须记录所有波形和估计器配置。

## 3. 标准 pipeline

推荐完整 pipeline：

```text
1. Build waveform/resource grid
2. Generate pilots/training symbols
3. Apply TX chain
4. Apply channel truth or CIR
5. Add interference/noise
6. Apply receiver frontend
7. Run packet detection/synchronization
8. Estimate channel
9. Interpolate/smooth
10. Produce H_obs and diagnostics
11. Compute evaluation metrics
```

## 4. Waveform

必须支持：

- `custom_ofdm`
- `wifi_like` 作为后续 profile
- `nr_like` 作为后续 profile

必选参数：

```text
sample_rate_hz
fft_size
cp_length
num_ofdm_symbols
pilot_indices
pilot_symbols
active_subcarrier_mask
tx_power_dbm
```

输出写入：

```text
/waveform
```

详见 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)。

## 5. Impairments

第一版：

- AWGN
- 可选 CFO

第二版起逐步加入：

- SFO
- phase noise
- IQ imbalance
- AGC
- ADC quantization
- clipping
- PA/LNA nonlinearities
- external interference

所有 impairment 必须同时保存：

- 配置分布。
- 本次运行实际采样值。
- 随机种子。
- 是否启用。

落盘位置：

```text
/impairments
/observation
```

## 6. 接收机诊断

每个 `[snapshot, tx, rx]` 至少保存：

```text
valid_mask
detection_success
estimation_success
snr_db
rssi_dbm
noise_power_dbm
cfo_hz
sfo_ppm
timing_offset_samples
phase_offset_rad
```

推荐保存：

```text
agc_gain_db
clipping_flag
failure_reason
quality_score
estimator_noise_var
```

## 7. 信道估计器

第一版：

- LS estimator。

后续：

- LMMSE estimator。
- pilot interpolation。
- time/frequency smoothing。
- learned estimator, optional。

估计器必须输出：

- `cfr_est`
- success flag
- noise variance or confidence
- failure reason

## 8. 动态与 Doppler

如果 RT 分支使用 `paths.cfr(..., num_time_steps > 1)` 或 `paths.cir(..., num_time_steps > 1)`，PHY pipeline 应支持多 snapshot/time-step 输入。

要求：

- `H_true` 和 `H_obs` 的时间轴对齐。
- `/motion/timestamp_s` 与 `/observation/timestamp_s` 一致或可映射。
- Doppler 相关字段来自 [04_sionna_rt_adapter_and_path_data.md](04_sionna_rt_adapter_and_path_data.md)。

## 9. 质量验证

第一版必须满足：

- AWGN-only 时，SNR 提高，`NMSE(H_obs, H_true)` 单调下降。
- 无 impairment 且高 SNR 时，`H_obs` 接近 `H_true`。
- 加 CFO 后，相位随 OFDM symbol 出现可解释漂移。
- failure flags 可被测试构造触发。
- `/observation/cfr_est.shape[1:] == /channel/truth/cfr.shape`。
- `/evaluation/nmse_db.shape == [snapshot, tx, rx]`。
- 所有 observation 诊断字段都能被 HDF5 reader 读回。

测试要求见 [09_testing_and_quality_gates.md](09_testing_and_quality_gates.md)。
