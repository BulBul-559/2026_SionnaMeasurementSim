# Feature TODO

本页记录会扩展系统能力、算法能力或标准完整性的 TODO。顶部列表按当前重要程度排序；
每次新增、完成或拆分 TODO 时，都要重新检查排序。

## Priority List

| 顺位 | ID | TODO | 简述 |
|---:|---|---|---|
| 1 | FEAT-SRS-001 | NR SRS reference validation | 用 38.211/38.213 reference 校验当前 v2 subset 的资源映射、序列、hopping、端口和功控口径。 |
| 2 | FEAT-RNG-001 | 完整协议 RTT observation | 在 waveform ToA 之外建模双向 packet exchange、turnaround 和 timestamp 语义。 |
| 3 | FEAT-RNG-002 | 设备 clock / bias 模型 | 给 ranging observation 增加 clock offset/drift、timestamp quantization 和 group-delay bias。 |
| 4 | FEAT-RNG-003 | 超分辨 ToA estimator | 增加 MUSIC / ESPRIT / SAGE 等 estimator，与 PDP peak 和 phase-slope 对照。 |
| 5 | FEAT-SRS-002 | 完整 SRS sequence 对齐 | 将 deterministic `nr_zc` 与 38.211 low-PAPR/ZC 序列逐项对齐。 |
| 6 | FEAT-SRS-003 | cyclic shift 约束建模 | 为 cyclic-shift 复用加入 delay-spread 冲突约束、失败检测和 reference validation。 |
| 7 | FEAT-SRS-004 | 多 slot time allocation | 扩展到多 slot 观测轴，支持真实 periodic / aperiodic / semipersistent 触发流程。 |
| 8 | FEAT-SRS-005 | 完整 frequency / bandwidth hopping | 对齐标准 bandwidth hopping / frequency hopping 规则，而不仅是 per-symbol offset/list。 |
| 9 | FEAT-SRS-006 | 完整 ports / layers / antenna switching | 对齐 NR SRS port、antenna switching procedure 和 codebook/non-codebook uplink。 |
| 10 | FEAT-SRS-008 | 标准化 SRS receiver quality | 强化 interpolation、delay-spread 检测、quality 指标和 reference validation。 |
| 11 | FEAT-RNG-004 | NLoS bias detection / correction | 基于 path truth、PDP、SNR、AoA/空间谱或 learned feature 做 NLoS 判别和偏差修正。 |

## Details

### FEAT-SRS-001: NR SRS reference validation

目的：把当前 `nr_srs` 从 “standards-shaped v2 subset” 推进到可审计的标准对齐实现，
明确哪些字段与 3GPP 38.211/38.213 一致，哪些仍是项目内简化。

涉及模块：`sionna_measurement_sim/phy/nr_srs_resources.py`、
`sionna_measurement_sim/phy/nr_srs_observation.py`、`sionna_measurement_sim/config/schema.py`、
HDF5 writer/schema validator、SRS 相关测试和 `docs/sys/05_phy_observation.md`。

验收标准：新增一组 reference case，覆盖 resource set、comb、BWP、sequence、hopping、
cyclic shift、port mapping 和 power metadata；文档能清楚说明通过项和仍不能声称
3GPP-compliant 的项。

重点提醒：先做 reference validation，不要先重写运行路径；避免破坏现有
`/observation/cfr_est`、array spectrum 和 ranging 的消费口径。

### FEAT-RNG-001: 完整协议 RTT observation

目的：在现有 waveform-level one-way ToA 基础上，新增完整协议侧 RTT observation model，
用于区分 propagation truth、two-way equivalent 和真实设备协议观测。

涉及模块：`sionna_measurement_sim/ranging/`、config `ranging` 子树、HDF5 `/ranging`
group、schema validator、ranging docs 和 smoke summary scripts。

验收标准：支持可配置 packet exchange、MAC turnaround、timestamp 语义和 RTT output；
`/derived/first_path_propagation_range_m` 仍保持 truth，不被观测模型覆盖；小实验能同时输出
truth range、waveform ToA range 和 protocol RTT-derived range。

重点提醒：协议 RTT 应作为独立 observation，不要把 bias 或 turnaround 写进 derived truth。

### FEAT-RNG-002: 设备 clock / bias 模型

目的：给 ranging observation 加入真实设备常见误差源，包括 clock offset/drift、timestamp
quantization、chip/group-delay bias 和校准前后差异。

涉及模块：`sionna_measurement_sim/ranging/`、config `ranging.device_profile` 或等价配置、
HDF5 `/ranging/*` metadata、schema/docs/tests。

验收标准：配置固定 seed 时输出可复现；校准前后输出可区分；单元测试覆盖零偏置、
固定偏置、随机漂移和 timestamp 量化；真实 smoke 报告 bias 前后误差分布。

重点提醒：不要把 PHY impairment 链路和 ranging device profile 混成一个模块；二者都可以影响
observation，但职责不同。

### FEAT-RNG-003: 超分辨 ToA estimator

目的：在 PDP peak 和 phase-slope baseline 之外，增加 MUSIC / ESPRIT / SAGE 等
super-resolution estimator，改善多径分辨和弱首径场景的分析能力。

涉及模块：`sionna_measurement_sim/ranging/estimators`、ranging runner、config estimator list、
unit tests、summary/plot scripts。

验收标准：合成单径、双径、弱首径和多天线 CFR 单元测试通过；真实场景报告 finite rate、
P50/P80/P95、LoS/NLoS 分组和失败样本；输出命名不与 truth 混淆。

重点提醒：先做插件接口和小型 reference case，再接真实数据；不要让 estimator 直接读写 HDF5。

### FEAT-SRS-002: 完整 SRS sequence 对齐

目的：把当前 deterministic `nr_zc` 和 legacy `zc_like` 的边界说清楚，并让 `nr_zc`
通过 38.211 low-PAPR/ZC sequence reference case。

涉及模块：`nr_srs_resources.py` sequence 生成、group/sequence hopping metadata、SRS unit tests。

验收标准：不同 `sequence_id`、slot、symbol、group hopping、sequence hopping 的 root/group/sequence
metadata 与 reference 一致；unit magnitude、低 PAPR 和复现性测试覆盖关键边界。

重点提醒：`zc_like` 可以保留为 legacy-compatible 简化序列，但文档必须明确它不用于标准声明。

### FEAT-SRS-003: cyclic shift 约束建模

目的：让同 symbol cyclic-shift port multiplexing 不只“可分离”，还要具备 delay spread
约束、冲突检测和失败标记。

涉及模块：SRS resource/receiver despreading、quality metrics、HDF5 SRS metadata、schema tests。

验收标准：重复 cyclic shift fail-fast；delay-window 冲突或过大 delay spread 能标记
`estimation_success=false` 或写出明确质量指标；无噪声 flat channel 多 port case 仍可分离。

重点提醒：不要用 time-symbol orthogonality 掩盖 cyclic-shift 问题；测试要覆盖同 symbol 多 port。

### FEAT-SRS-004: 多 slot time allocation

目的：把单 slot scheduling 语义扩展为真实多 slot 时间轴，支持 periodic、aperiodic 和
semipersistent 触发流程。

涉及模块：config `phy.srs`、waveform builder、domain shape、HDF5 waveform/observation shape、
pipeline snapshot/time 语义。

验收标准：多 slot 配置能生成正确 active slot/symbol mask；未调度 slot 不再只能 fail-fast，
而是可显式输出空观测或跳过；文档说明 snapshot、slot 和 motion 的关系。

重点提醒：这会触碰 shape 契约，必须同步更新 schema、reader 和下游脚本。

### FEAT-SRS-005: 完整 frequency / bandwidth hopping

目的：把当前 per-symbol PRB offset 和 bandwidth list 扩展为标准 bandwidth hopping /
frequency hopping 规则。

涉及模块：SRS resource plan、config schema、HDF5 per-symbol PRB metadata、resource tests。

验收标准：标准 hopping case 的 PRB start/count/resource mask 与 reference 一致；越界配置
fail-fast；full-band interpolation 在 hopping 后仍保持 shape 和 finite policy。

重点提醒：normal smoke 中 64 PRB 占用不应被误读为跨 hop 累计带宽；实验配置要写清楚。

### FEAT-SRS-006: 完整 ports / layers / antenna switching

目的：补齐 NR SRS port、antenna switching procedure、codebook/non-codebook uplink 语义，
并让 metadata 能支撑后续 PUSCH codebook 或定位模型使用。

涉及模块：SRS port mapping、antenna arrays、HDF5 `/waveform/srs_port_tx_ant_map`、
docs/tests。

验收标准：one-to-one、explicit map、antenna switching procedure 都有 reference tests；
每个 TX antenna 的 sounding 覆盖可验证；metadata 清楚区分 port、antenna、layer 和 usage。

重点提醒：`ports.usage` 当前只是 metadata，不要在没有 receiver/selection 逻辑前声称实现了
PUSCH codebook selection。

### FEAT-SRS-008: 标准化 SRS receiver quality

目的：强化 SRS resource extraction、interpolation、delay-spread 检测和 quality metrics，
让 `/observation/cfr_est` 的质量能被训练/分析侧稳定筛选。

涉及模块：SRS receiver、`evaluation/*` SRS 指标、schema/docs、summary scripts。

验收标准：常量/线性/多径信道下 interpolation 误差有单元测试；quality 指标覆盖 resource NMSE、
full-band NMSE、resource SNR、failure reason；真实 smoke 输出 summary CSV/JSON。

重点提醒：`cfr_est_resource` 是 SRS RE 上的直接估计，`cfr_est` 是插值后的 full-band 输入；
文档和下游脚本必须保持这个区分。

### FEAT-RNG-004: NLoS bias detection / correction

目的：为 ranging observation 增加 NLoS 判别、bias correction 和 confidence/validity 指标，
减少多径和遮挡导致的 range 偏差。

涉及模块：ranging estimators、path truth、PDP feature、AoA/空间谱输出、summary scripts。

验收标准：真实实验按 LoS/NLoS 分组报告 correction 前后误差；输出 detection confidence；
失败或低置信度样本不被静默当成正常估计。

重点提醒：可以先做基于 path truth 的 oracle analysis，再逐步替换成 observation-only feature。
