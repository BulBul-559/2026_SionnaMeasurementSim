# 01. 系统目标与边界

本文定义 `SionnaMeasurementSim` 的目标、非目标和核心原则。架构实现应遵循 [02_architecture.md](02_architecture.md)，数据落盘应遵循 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)。

## 1. 核心目标

`SionnaMeasurementSim` 的目标是构建一个基于 Sionna 2.x 的真实测量值导向仿真系统。

系统应同时产出：

- `H_true`：由 Sionna RT 计算得到的传播真值信道。
- `H_obs`：经过 PHY 波形、硬件损伤、噪声、同步和信道估计后得到的观测信道。
- 路径级几何数据：LoS/NLoS、多径中间交互点、交互对象、交互类型、路径延迟、路径 Doppler、AoA/AoD 等。
- 设备和场景元数据：TX/RX 位置、速度、朝向、天线阵列、极化、材质、动态对象状态等。
- 观测诊断：SNR、RSSI、CFO、SFO、相位噪声、AGC、ADC、同步/估计成功标志等。

一句话目标：

```text
不是只算理想 CFR，而是模拟真实设备如何采集、估计并记录信道。
```

## 2. 为什么要区分 H_true 与 H_obs

`H_true` 是传播物理真值。它回答：

- 几何和材质允许哪些传播路径？
- 每条路径的延迟、增益、角度、Doppler 是什么？
- 理想频率响应是什么？

`H_obs` 是接收机观测值。它回答：

- 给定导频和接收机算法后，设备能估计出什么？
- 弱链路是否被检测到？
- 同步误差、噪声和硬件损伤会怎样污染 CFR/CSI？
- 真实采集数据中常见的相位漂移、估计失败和异常样本如何出现？

真实设备采集到的通常更接近 `H_obs`，不是 `H_true`。因此，新系统的数据契约必须同时保存两者，避免后续训练或分析时把真值误当观测值。

## 3. 第一阶段范围

第一阶段不是完整复刻 WiFi、5G NR 或任意商用协议，而是建立可追溯、可扩展、可验证的仿真底座。

必须完成：

- Sionna 2.x RT 最小闭环。
- 路径级数据提取，包括 `vertices`、`objects`、`interactions`、`primitives`、`doppler`。
- 完整 HDF5 schema v1。
- TX/RX 位置、速度、朝向、阵列、极化配置落盘。
- `H_true` 保存。
- 最小 PHY 观测链：OFDM pilot + AWGN + LS channel estimation。
- `H_obs` 保存。
- NMSE、SNR、估计成功率等基础诊断。
- 使用 `uv` 管理环境。
- 每个里程碑通过测试后提交 git。

## 4. 后续阶段范围

后续逐步扩展：

- CFO/SFO、相位噪声、IQ imbalance、AGC、ADC 量化、clipping。
- 多快照动态仿真。
- Doppler-based time evolution。
- 场景对象运动和重新追踪。
- 实测数据标定。
- WiFi-like 或 NR-like 具体帧结构。
- 批处理和大规模数据集生成。

## 5. 非目标

短期内不追求：

- 完整 MAC 层。
- 完整商用芯片行为复刻。
- 编码译码闭环。
- 分布式作业系统。
- 一开始就全量保存所有 raw waveform。
- 手写替代 Sionna 已经成熟的 RT/PHY 基础能力。

## 6. 基线数据

验收使用一个小型 3D 场景和标注数据。约定位置为：

```text
SionnaMeasurementSim/data/scenes/test/
```

该测试集用于：

- RT adapter 最小集成测试。
- HDF5 schema 写入和读回测试。
- 可视化 smoke test。
- `H_true`/`H_obs` 对比测试。

如果 `old/` 中存在同等测试数据，只能人工参考或一次性复制到上述新位置；新系统运行时不得依赖 `old/` 内部路径。
