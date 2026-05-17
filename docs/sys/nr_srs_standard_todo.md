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

论文写法建议：

- 当前结果称为 “SRS-like full-band uplink sounding”。
- 不要写 “3GPP-compliant NR SRS”，除非上述资源映射和序列机制已经实现并验证。
- 如果和 PUSCH-DMRS CSI proxy 对比，说明两者输出都落到统一 `/observation/cfr_est`
  契约，区别在 pilot/resource pattern 和 receiver 指标。
