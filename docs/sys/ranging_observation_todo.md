# Ranging Observation TODO

日期：2026-05-19

当前 `ranging.enabled=true` 实现的是 waveform-level ToA / one-way range
observation：从受损后的 `/observation/cfr_est` 估计 `/ranging/*/toa_est_s` 和
`/ranging/*/one_way_range_est_m`。`/derived/first_path_propagation_range_m`
仍是 truth label，`/ranging/*/rtt_equiv_s` 只是 `2 * toa_est_s` 的 two-way
equivalent，不是完整协议 RTT。

本文只记录本轮 `schema 1.2.0` 之外、后续可以作为研究增强推进的 active TODO。
计数口径：下表每行算 1 个顶层 TODO，本表当前共 4 个。

## TODO

| 优先级 | 项目 | 当前状态 | 下一步 |
|---|---|---|---|
| P1 | MUSIC / ESPRIT / SAGE 等超分辨估计器 | v1 只有 PDP peak 和 phase-slope 两个可解释 baseline | 在 `sionna_measurement_sim/ranging/` 下新增 estimator 插件接口实现，优先做合成多径单元测试，再做真实场景误差和失败率对比 |
| P1 | 完整协议 RTT observation model | v1 只输出 uplink waveform ToA 和 `rtt_equiv_s` | 建模双向 packet exchange、MAC turnaround、timestamp 语义和协议侧可观测 RTT，明确与 one-way propagation truth 的差异 |
| P1 | device clock / turnaround / bias 模型 | v1 没有设备 clock offset/drift、timestamp quantization、chip/group-delay bias | 新增独立 ranging observation impairment，不改 `/derived` truth；支持可配置设备 profile、校准前后两套输出和可复现实验 seed |
| P2 | NLoS bias detection / correction | v1 只报告 estimator 输出和 range error，不做 NLoS 修正 | 结合 path truth、PDP shape、SNR、AoA/空间谱或 learned feature 做 NLoS 判别、bias correction 和 confidence/validity 指标 |

## 验收建议

- 每个 estimator 先用单径、双径、弱首径、多天线合成 CFR 做严格单元测试。
- 真实实验不要只看 mean error，应同时报告 finite rate、P50/P80/P95、NLoS/LoS 分组和失败样本。
- 协议 RTT、clock/bias、NLoS correction 都应作为独立模块挂到 `/ranging` 或后续 observation group，
  不覆盖 `/derived` truth label。
