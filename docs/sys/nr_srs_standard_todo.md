# NR SRS Standard TODO

当前 `phy.standard: "nr_srs"` 是 **standards-shaped NR SRS v2 subset**：
已支持 full-slot time allocation、comb/BWP resource、flattened SRS RE、
`zc_like`/`nr_zc` sequence、group/sequence hopping、同 symbol cyclic-shift port
multiplexing、frequency/bandwidth hopping、port/antenna switching 口径、简化
uplink power scaling、resource LS/despread 和 full-band interpolation。它适合做室内
定位和 PUSCH-DMRS CSI proxy 的基线对比，但还不能声称是完整 3GPP NR SRS。

## 当前能力与剩余差距

| 项目 | 当前状态 | 后续要求 |
|---|---|---|
| comb / BWP | 已有 comb=1/2/4、BWP PRB、flattened RE 和 64 PRB + `K_TC=2` 验证 | 对齐 38.211 的完整 SRS bandwidth/resource set 表 |
| sequence | 已有 deterministic `nr_zc`、legacy `zc_like`、group/sequence hopping metadata | 与 38.211 low-PAPR/ZC 序列 reference 做逐项一致性验证 |
| cyclic shift | 已支持同 symbol cyclic-shift 复用和 delay-window despreading | 增加真实 cyclic shift / delay spread 约束建模和 reference validation |
| time allocation | 已有 single-slot start/length/trigger/period/offset 校验 | 扩展到多 slot 时间轴和真实 periodic/aperiodic/semipersistent 触发流程 |
| frequency hopping | 已支持 per-symbol PRB offset 和 bandwidth list | 对齐 bandwidth hopping / frequency hopping 的完整标准规则 |
| ports/layers | 已有 one-to-one port map、简化 antenna switching map、codebook/non-codebook metadata | 对齐完整 SRS port、antenna switching procedure、codebook/non-codebook uplink |
| power control | 已有 pathloss-based open-loop SRS power scaling 与 metadata | 实现 NR uplink power control 的闭环/绝对噪声/服务小区流程 |
| receiver | 已有 flattened resource extraction、time-code LS、cyclic-shift despread、linear interpolation、quality 指标 | 增加更标准的 interpolation、delay spread 检测和 reference validation |

## 推荐实施顺序

真实 **UE→BS uplink** 与 BS/UE → TX/RX 语义解耦已经完成并进入生产口径：
`phy_link_direction="uplink"` 会解析为 TX=UE、RX=BS，SRS direct uplink 已完成
`median_0000 label0p2` 的 `7 BS × 2583 UE` shard baseline。当前剩余风险主要在
更复杂场景的 RT 规模、非合成阵列、大规模 shard 成本和 3GPP reference validation，
不是 HDF5 数据契约。

| 阶段 | 目标 | 产物 |
|---|---|---|
| P0 | direct UE→BS uplink 验证 | 已完成 direct uplink / AoA role / schema / manifest 口径；继续保留小规模 probe 作为回归 |
| P1 | direct uplink 生产化 shard | 已有 `median_0000 label0p2` SRS baseline；继续做不同密度/场景、PUSCH 对照和必要的二维 BS/UE block shard |
| P2 | SRS standards-shaped v2 subset | 已完成 comb/BWP/hopping、sequence hopping、cyclic-shift multiplexing、ports map、power scaling、resource LS/despread |
| P3 | 3GPP NR SRS 完整化 | 38.211/38.213 reference 对齐、完整 antenna switching、闭环功控、多 slot 触发流程 |

## 论文写法建议

- 当前结果称为 “NR SRS standards-shaped v2 subset”。
- 不要写 “3GPP-compliant NR SRS”，除非资源映射、序列、hopping、端口和功控都已通过标准 reference 对齐。
- 如果和 PUSCH-DMRS CSI proxy 对比，说明两者输出都落到统一 `/observation/cfr_est`
  契约，区别在 pilot/resource pattern、receiver 指标和 SRS-specific resource metadata。
