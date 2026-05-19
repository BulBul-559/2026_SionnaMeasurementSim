# NR SRS Standard TODO

当前 `phy.standard: "nr_srs"` 是 **SRS-like full-band uplink sounding**：
所有 active subcarrier 都放已知 pilot，UE 多天线通过 OFDM symbol 维度上的正交 DFT
code 分离，然后用 LS 得到 CSI。它适合做室内定位和 PUSCH-DMRS CSI proxy 的基线对比，
但还不能声称是完整 3GPP NR SRS。

后续要升级为标准 NR SRS，至少需要补齐：

| 项目 | 当前状态 | 后续要求 |
|---|---|---|
| comb structure | 全子载波 sounding | 支持 NR SRS comb、频域梳状映射和带宽部分配置 |
| sequence | 简单 DFT 正交码 | 支持 Zadoff-Chu / low-PAPR SRS sequence、group/sequence hopping |
| cyclic shift | 未实现 | 支持 cyclic shift 复用和端口分离 |
| time allocation | 用 `num_ofdm_symbols` 做正交 sounding | 支持 slot 内 SRS symbol 位置、周期、aperiodic/semipersistent 触发 |
| frequency hopping | 未实现 | 支持 SRS bandwidth hopping / frequency hopping |
| ports/layers | 以 UE antenna 数构造正交 pilot | 对齐 NR SRS port、antenna switching、codebook/non-codebook uplink |
| power control | 简化 AWGN SNR | 补 NR uplink power control、pathloss compensation、power scaling |
| receiver | 简单 LS | 增加标准 SRS resource extraction、interpolation、noise/quality 指标 |

## 推荐实施顺序

真实 **UE→BS uplink** 与 BS/UE → TX/RX 语义解耦已经完成并进入生产口径：
`phy_link_direction="uplink"` 会解析为 TX=UE、RX=BS，SRS-like direct uplink 已完成
`median_0000 label0p2` 的 `7 BS × 2583 UE` shard baseline。当前剩余风险主要在
更复杂场景的 RT 规模、非合成阵列、大规模 shard 成本和标准 NR SRS 资源映射，而不是
HDF5 数据契约。

建议顺序：

| 阶段 | 目标 | 产物 |
|---|---|---|
| P0 | direct UE→BS uplink 验证 | 已完成当前 direct uplink / AoA role / schema / manifest 口径；继续保留小规模 probe 作为回归 |
| P1 | direct uplink 生产化 shard | 已有 `median_0000 label0p2` SRS baseline；继续做不同密度/场景、PUSCH 对照和必要的二维 BS/UE block shard |
| P2 | SRS v1 标准子集 | comb、resource mask、cyclic shift、多端口正交、sparse LS 和 full-band interpolation |
| P3 | 3GPP NR SRS 完整化 | group/sequence hopping、bandwidth hopping、periodic/aperiodic/semipersistent 触发、标准 reference 对齐 |

## SRS v1 标准子集计划

第一版不要直接追求认证级 3GPP-compliant SRS，而是实现 “standards-shaped
NR SRS subset”。推荐范围：

1. 增加 `NRSRSConfig` 或等价配置段，包含 `comb_size`、`num_srs_ports`、
   `start_symbol`、`num_symbols`、`freq_position`、`bandwidth_config`、
   `cyclic_shift`、`sequence_id`、`group_hopping`、`sequence_hopping`。
2. 新增 SRS resource grid builder，输出：
   - `/waveform/srs_resource_mask`
   - `/waveform/srs_pilot_symbols`
   - `/waveform/srs_port_index`
   - `/waveform/tx_grid`
3. 从全带宽 pilot 改为按 comb/resource mask 提取 SRS RE：
   `H_hat[k] = Y[k] / X[k]`，并额外保存 sparse 与插值结果：
   - `/observation/srs_cfr_est_sparse`
   - `/observation/cfr_est`
4. 支持空间谱来源选择：只用 SRS RE、用插值后的 full-band CFR，或二者都写入
   供对比。
5. 增加验证：
   - 无噪声下 LS 接近 truth CFR。
   - comb mask 与端口映射正确。
   - cyclic shift 多端口可分离。
   - 与当前 full-band SRS-like 的 NMSE、有效链路率和空间谱峰值做对比。

复杂度评估：**中高**。Pipeline、HDF5、schema 和 PHY registry 已经具备扩展点；
主要复杂度集中在 3GPP SRS resource/sequence 规则和 reference validation。
因此当前可以在保持 P0/P1 回归的同时，进入 P2 的 standards-shaped SRS 子集设计。

论文写法建议：

- 当前结果称为 “SRS-like full-band uplink sounding”。
- 不要写 “3GPP-compliant NR SRS”，除非上述资源映射和序列机制已经实现并验证。
- 如果和 PUSCH-DMRS CSI proxy 对比，说明两者输出都落到统一 `/observation/cfr_est`
  契约，区别在 pilot/resource pattern 和 receiver 指标。
