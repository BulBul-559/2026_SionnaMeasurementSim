# 11. 标定与诊断

本文定义如何让仿真观测值逐步接近真实测量数据。PHY 观测链见 [05_phy_observation_and_impairments.md](05_phy_observation_and_impairments.md)，数据契约见 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)。

## 1. 目标

标定模块的目标：

- 从真实测量数据中提取统计特征。
- 拟合 impairment 参数分布。
- 生成 calibration profile。
- 对比 `H_obs` 与真实测量分布。
- 输出诊断报告。

## 2. Calibration Profile

推荐字段：

```text
profile_id
created_at
measurement_dataset_id
device_info
frequency_band
bandwidth
snr_distribution
cfo_distribution
sfo_distribution
phase_noise_profile
iq_imbalance_distribution
agc_adc_profile
detection_failure_model
estimation_failure_model
validation_metrics
```

profile 应能被 [06_config_and_experiment_schema.md](06_config_and_experiment_schema.md) 中 `calibration.profile_id` 引用。

## 3. 诊断指标

必须输出：

- NMSE 分布。
- SNR 分布。
- RSSI 分布。
- detection success rate。
- estimation failure rate。
- phase drift。
- amplitude error。

建议输出：

- delay spread。
- Doppler spread。
- coherence bandwidth。
- subcarrier correlation。
- time correlation。
- weak-link miss rate。

## 4. 与实测对齐

对齐方式：

```text
real measurement stats
  -> fit impairment distributions
  -> run simulation with profile
  -> compare simulated H_obs stats
  -> update profile
```

可用距离指标：

- 分位数误差。
- KL divergence。
- Wasserstein/EMD。
- KS statistic。
- 均值/方差差异。

## 5. HDF5 落盘

写入：

```text
/calibration/profile_id
/calibration/measurement_dataset_id
/calibration/fitted_parameters
/calibration/validation_metrics
/evaluation/*
```

## 6. 第一版要求

第一版不要求真实数据标定完整闭环，但必须：

- 定义 profile 文件格式。
- HDF5 中预留 `/calibration`。
- 输出 truth vs observation 的基础诊断。
- 允许未来接入实测数据而不改主 schema。

