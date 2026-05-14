# 性能优化设计指南：NR 专属路径与通用模块分层

日期：2026-05-14

来源：性能优化实验。本文是可带回 `main` 的设计指南，描述推荐边界、接口语义和验收口径；不要求保留或检出实验分支代码。

## 适用范围

本文面向后续性能优化和新 PHY 模块设计，基于当前 3x3000、5x5000 性能报告和系统文档：

- `docs/performance/nr_pusch_3x3000_light_baseline.md`
- `docs/performance/nr_pusch_3x3000_split_benchmarks.md`
- `docs/performance/nr_pusch_5x5000_profile_zh.md`
- `docs/sys/04_rt_pipeline.md`
- `docs/sys/05_phy_observation.md`
- `docs/sys/07_config_and_h5_format.md`

核心原则是：把 Sionna RT、CIR/CFR 数据、HDF5/result 分片、benchmark harness 做成通用底座；把 NR PUSCH receiver、DMRS/LS 估计、resource grid 和 MIMO detector 留在 NR 专属层。未来 UE 直接 uplink、WiFi-like/custom OFDM 或其他制式应复用通用底座，而不是复制一条新的大规模链路。

## 当前瓶颈结论

3x3000 拆分 benchmark 已经把端到端流程拆成三类成本：

| 阶段 | 3x3000 结果 | 主要含义 |
|---|---:|---|
| `RT-only` | 42.8 s | Sionna RT 求路径/CIR/CFR，主要由 `rt_solve` 决定 |
| `PUSCH-only` | 297.3 s | 9000 条 SU-MIMO link 串行 receiver 调用，是当前最大热点 |
| `Write-only` | 53.1 s | 空间谱、HDF5 clone/schema、可视化长尾 |

5x5000 full-output 报告进一步说明：

| 阶段 | 5x5000 结果 | 判断 |
|---|---:|---|
| NR PUSCH observation | 839.3 s | GPU 显存常驻但利用率低，受逐 link 小调用限制 |
| HDF5 写入 | 206.5 s | 大数组和压缩/I/O 长尾明显 |
| Sionna RT 求解 | 114.8 s | GPU compute-bound，和 PUSCH 是不同瓶颈 |
| Array/可视化 | 108.7 s | CPU 多核空间谱 + Matplotlib/读 HDF5 |

因此不能把“GPU 优化”当成一个单点问题。RT、NR PUSCH、输出写入、可视化的瓶颈类型不同，必须分别 benchmark、分别归因。

## 分层边界

### 通用层

通用层应只表达无线仿真系统的共性，不包含 NR PUSCH 语义：

| 模块 | 责任 | 不应包含 |
|---|---|---|
| RT solver/cache | 场景、TX/RX 拓扑、路径、CIR、delay、truth CFR | PUSCH resource grid、DMRS、receiver 细节 |
| CIR/CFR bridge | 统一 shape、UL/DL 视角转换、`cir_to_ofdm_channel` 结果复用 | 某个制式的 pilot/decoder 假设 |
| CFR batch 数据结构 | 按 `(snap, ul_tx, ul_rx)` 或 shard 批量提供 CFR/CIR slice | 只服务 NR 的 `PUSCHConfig` 对象 |
| result shard | `result_xxx.h5`、manifest、global index、schema 验证 | 单进程单文件写死 |
| benchmark harness | `RT-only`、PHY-only、write-only、端到端回归 | 只叫 `PUSCH-only` 的硬编码路径 |

通用层的接口目标是：给任意 PHY 标准一个稳定的输入批次和输出落盘方式。只要一个新模块需要同一批 BS/UE、同一个场景、同一个频率网格，它就应该能读取同一个 `rt_cache.h5` 或同一组 `result_xxx.h5` shard。

### NR 专属层

NR 专属层可以大胆优化，但优化对象应限制在 NR 语义内：

| 模块 | NR 专属优化方向 |
|---|---|
| `nr_pusch_observation.py` | SU-MIMO link batching、receiver 对象复用、减少 Python 逐 link 调度 |
| `PUSCHReceiver` | batch size 8/16/32/64 sweep，对比 BER/BLER/NMSE、吞吐、显存和 GPU 利用率 |
| `PUSCHLSChannelEstimator` | 按 resource grid / DMRS config 缓存或外提构造 |
| PUSCH transmitter | 对相同配置的重复 TX generation 做批量调用或缓存策略 |
| MIMO detector | 仅在 resource grid、stream management、层数语义一致时复用 |

NR 层可以消费通用 CFR batch，但不应该把通用 batch 设计成“PUSCHReceiver 的参数打包”。否则 WiFi-like/custom OFDM 后续会被迫绕开这套结构。

## 通用 CFR Batch 数据结构建议

建议把面向 PHY 的批次抽象成通用 `CFRBatch` 或等价结构，字段保持制式无关：

| 字段 | 建议 shape | 说明 |
|---|---|---|
| `snap_indices` | `[batch]` | 对应 snapshot |
| `ul_tx_indices` | `[batch]` | uplink 发送端，当前通常是 UE |
| `ul_rx_indices` | `[batch]` | uplink 接收端，当前通常是 BS |
| `cfr_ul` | `[batch, ul_rx_ant, ul_tx_ant, subcarrier]` 或 `[batch, ul_rx_ant, ul_tx_ant, sym, subcarrier]` | UL 视角 CFR |
| `cir_coefficients` | `[batch, ul_rx_ant, ul_tx_ant, path]` | 可选，给需要时域/路径级处理的 PHY |
| `cir_delays_s` | `[batch, ul_rx_ant, ul_tx_ant, path]` 或 shared delay | 可选，保留 delay 口径 |
| `link_valid_mask` | `[batch]` | 与 `/derived/link_valid_mask` 对齐 |
| `path_count` | `[batch]` | 与 `/derived/path_count` 或 truth path count 对齐 |
| `los_flag` / `nlos_flag` | `[batch]` | 用于 LoS/NLoS 分层指标 |

注意事项：

1. batch 维度只表示一批 link，不表达 NR 的 `num_pusch_tx` 或 OFDM stream 数。
2. UL/DL 视角必须显式命名。内部给 PHY 用 `cfr_ul`，写回 HDF5 仍遵循现有 truth CFR 契约。
3. 不要默认丢掉 path 维度。即使当前 NR PUSCH 主要用 CFR，RT 调参和未来 ToA/AoA/路径级模型仍需要 CIR/delay。
4. batch 结构要能从 `rt_cache.h5`、完整 `results.h5`、per-shard `result_xxx.h5` 三种来源构造。

## RT Cache 复用策略

当前 `RT-only` benchmark 已经生成 `rt_cache.h5`，包含：

| Dataset | Shape |
|---|---|
| `/channel/truth/cfr` | `(3, 3000, 4, 4, 48)` |
| `/channel/truth/cir_coefficients` | `(1, 3, 3000, 4, 4, 17)` |
| `/channel/truth/cir_delays_s` | `(1, 3, 3000, 4, 4, 17)` |

后续应把这个缓存从“PUSCH-only 的输入”提升为“PHY-only 通用输入”。建议：

1. `RT-only` manifest 记录完整 RT 参数、场景文件、label 文件、频率网格、天线配置、TX/RX 截断或 shard 范围。
2. PHY-only benchmark 只接受 cache + PHY config，不重新运行 RT。
3. cache key 至少包含 scene、label、TX/RX 范围、carrier、antenna、`max_depth`、LoS/specular/diffuse/refraction/diffraction、synthetic array、normalize 策略。
4. 只改 NR receiver、custom OFDM 或 WiFi-like 算法时，必须复用同一个 RT cache，避免把 RT 随机性或路径变化混进 PHY 性能结论。
5. 修改 RT 参数后必须刷新 cache，并保留旧 cache 的指标对照。

## Future UE Direct Uplink

当前系统文档把 NR PUSCH 的内部 CFR 约定为 UL 视角：UE 是 `ul_tx`，BS 是 `ul_rx`；写回 HDF5 前恢复到既有 truth/observation 契约。未来做 UE 直接 uplink 时，建议沿用这个约定：

1. RT 层仍按项目拓扑生成 truth CFR/CIR，不把“谁发谁收”的 PHY 语义写死进 RT。
2. Channel bridge 提供明确的 UL 视角 batch，BS 的阵列维度是 `ul_rx_ant`，UE 的阵列维度是 `ul_tx_ant`。
3. PHY 层只消费 `CFRBatch.cfr_ul`，不直接读取 `/channel/truth/cfr` 后自行猜测转置。
4. 写回结果时保留 `/observation/cfr_est` 与 truth CFR 的现有 shape 对齐，避免下游训练和 schema 破裂。
5. 多 UE uplink 与 SU-MIMO/MU-MIMO 是 PHY 调度问题，不应该改变 RT cache 的基本 layout。

这样未来无论是 NR PUSCH uplink、WiFi uplink sounding，还是自定义 OFDM uplink，都能共用同一套 RT cache 和 batch 构造逻辑。

## WiFi-like / Custom OFDM 新模块复用方式

WiFi-like 或 custom OFDM 不应复制 NR PUSCH 的 `run_nr_pusch_observation()` 调度模型。推荐路径是：

1. 从 `rt_cache.h5` 或 `CFRBatch` 读取 `cfr_ul` / CIR / delay。
2. 使用自己的 pilot、preamble、LS/MMSE 估计、同步或 impairment 模型。
3. 输出统一的 `ObservationResult`、`EvaluationResult`、waveform/grid 摘要。
4. 复用通用 HDF5 writer 和 schema validator；只有确有必要时新增制式专属 group。
5. 复用 PHY-only benchmark 框架，命名为 `phy-only --standard custom_ofdm` 或类似形式，而不是新增孤立脚本。

关键坑位：

- custom OFDM 当前是快速验证路径，不能假设它的 LS 估计 shape 能代表 NR DMRS LS 估计。
- WiFi-like 的 preamble 和子载波 mask 可能不同，但输出的 link 维度仍应对齐 `(snap, tx, rx)` 或 UL batch 的 global index。
- 如果某个标准只需要 CFR，不需要 full CIR，也不要在通用 cache 中删除 CIR/delay。
- 如果某个标准需要时域 waveform，重型 waveform 输出应受配置控制，并且 shard-aware。

## 多文件 `result_xxx.h5` 和 Shard 设计

5x5000 报告已经说明，当前 `results.h5` 单文件写入会成为明显瓶颈。大规模生产建议先支持多文件 shard，再考虑多 GPU 调度。

推荐命名：

| 文件 | 内容 |
|---|---|
| `result_0000.h5` | 第 0 个 UE/BS shard 的完整局部结果 |
| `result_0001.h5` | 第 1 个 shard |
| `results_manifest.json` | shard 列表、global index 范围、配置摘要、schema 版本 |
| `rt_cache_0000.h5` | 可选，对应 shard 的 RT cache |

优先按 UE 维度分 shard，因为当前大规模 case 是少量 BS、大量 UE：

| Shard 策略 | 适合场景 | 注意 |
|---|---|---|
| UE range shard | 5x5000、6x8884 这类 UE 多、BS 少 | 需要记录 `rx_start/rx_count` 或显式 `rx_indices` |
| BS range shard | BS 很多或 BS 间独立实验 | 下游合并空间谱和多 BS 特征时要保留 global BS index |
| Link list shard | 稀疏 link 或按 LoS/NLoS 分层采样 | manifest 必须记录每条 link 的 global `(tx, rx)` |

不要直接让多个进程并发写同一个 HDF5。更稳的流程是：每进程绑定一张 GPU，写一个 `result_xxxx.h5`，最后由 manifest 或离线 merge 对外呈现统一数据集。manifest 至少应记录全局 UE/BS 范围、scene_id、配置摘要、schema 版本、每个文件的 shape 和校验状态。

## Benchmark Harness 设计

现有三类 benchmark 应保留，但命名和能力应逐步通用化：

| 当前 benchmark | 通用化后职责 |
|---|---|
| `RT-only` | 只测 topology load + Sionna RT + cache write |
| `PUSCH-only` | NR PUSCH 的 `PHY-only --standard nr_pusch` |
| `Write-only` | 从已有结果或内存数组测 array outputs、HDF5、schema、visualization |
| 端到端 3x3000 | 每次优化后的真实流程回归 |
| 5x5000 标准基线 | 大规模效果确认 |

每次优化的判断顺序：

1. 只改 NR receiver、LS estimator、link batching：先跑 NR PHY-only。
2. 只改 RT 参数、scene load、path/CIR/CFR：先跑 RT-only。
3. 只改 HDF5、空间谱、可视化：先跑 write-only。
4. 拆分 benchmark 有收益后，跑 3x3000 端到端确认 schema、输出、可视化。
5. 3x3000 稳定后，再跑 5x5000；准备生产全量前再跑多 shard 试验。

benchmark 输出必须包含：

- wall time 和阶段耗时；
- link 数、batch size、失败数；
- GPU/CPU/RSS 采样；
- 关键 dataset shape 和 dtype；
- BER/BLER/NMSE 或对应标准的观测指标；
- `path_count`、LoS/NLoS 分布、CFR 差异摘要，尤其是 RT 调参时。

## RT-only 后续调参建议

RT-only 调参要以“语义受控”为前提。`max_depth`、diffuse/refraction/diffraction、TX/BS 分组策略都会改变物理路径集合，不能只看耗时。

### `max_depth`

建议实验矩阵：

| 实验 | 目的 |
|---|---|
| `max_depth=0` | LoS-only 下界，用于确认最小路径成本和 LoS 覆盖 |
| `max_depth=1` | 当前常用轻量反射基准 |
| `max_depth=2` | 观察二阶交互带来的 path_count、CFR 和耗时增量 |
| `max_depth=3` | 只在 3x3000 先试，确认显存、路径数和收益后再扩大 |

每个设置必须记录：

- `/derived/path_count` 或等价 path count 的均值、P50、P90、P99、max；
- `los_flag` / `nlos_flag` 占比；
- truth CFR 幅度/相位差异摘要，例如相对 `max_depth=1` 的 NMSE 或每 link CFR delta；
- RT-only 耗时、GPU 利用率、显存峰值；
- PHY 指标变化，至少 NR PUSCH 的 BER/BLER/NMSE 或 custom OFDM 的估计误差。

### diffuse / refraction / diffraction

这些开关不是单纯性能开关，而是传播语义开关。建议一次只改一个变量：

| 开关 | 记录重点 |
|---|---|
| `diffuse_reflection` | path_count 增量、弱路径功率分布、CFR 纹理变化 |
| `refraction` | 穿透路径是否改变 LoS/NLoS 判定和强路径 delay |
| `diffraction` | 阴影区覆盖改善、NLoS path 数量和 ToA/AoA 标签变化 |

禁止默认打开这些开关作为“更真实”的无记录改动。只有当报告证明它改善目标任务，且下游知道语义变化时，才能进入默认配置讨论。

### TX/BS 分组策略

当前 5x5000 形态是少量 BS、大量 UE。后续可评估：

| 策略 | 用途 | 风险 |
|---|---|---|
| 所有 BS 一次性 RT | 保持完整多 BS 结果，适合当前小 BS 数 | UE 很多时内存和输出大 |
| 按 UE shard、全 BS | 推荐生产主路径，方便训练样本按 UE 合并多 BS 特征 | 每个 shard 都要记录 global UE index |
| 按 BS shard、全 UE | BS 数变大时可用 | 多 BS 特征合并更复杂 |
| 按 link list | 精细采样 LoS/NLoS 或困难样本 | manifest 复杂，容易丢 global index |

TX/BS 分组会影响 path cache、输出文件和下游样本组织，但不应改变物理语义。同一组全局 `(tx, rx)` 在 shard 内外的 CFR、path_count、LoS/NLoS 应保持一致，允许的差异只能来自浮点或 Sionna 非确定性，并需要报告。

## 不要踩的坑

1. 不要把 `max_tx/max_rx` 当作 shard。它们当前是截断，不是带 global offset 的分片。
2. 不要把 `run-batch` 当作 UE/BS 分块。它主要是多 seed / 多实验批次。
3. 不要让 NR PUSCH batch 结构污染通用 RT cache。RT cache 应服务所有 PHY。
4. 不要为了省空间删除 CIR/delay/path metadata。RT 调参和定位标签需要这些信息。
5. 不要在没有兼容策略时复制大型等价 dataset；若两个字段语义相同，优先保留一个物理 dataset，并用 attrs/manifest 说明兼容名称和语义。
6. 不要用端到端总耗时判断某个局部优化是否有效。先用拆分 benchmark 归因。
7. 不要在 RT 调参中只报告“更快”。必须同时报告 path_count、LoS/NLoS 和 CFR/PHY 指标影响。
8. 不要默认改变传播语义。任何 RT 开关或 `max_depth` 默认值变化都应有显式实验记录和迁移说明。

## 建议落地顺序

1. 先把 `rt_cache.h5` 的 manifest 和 cache key 口径补齐，让它成为通用 PHY-only 输入。
2. 定义通用 `CFRBatch` 数据结构和从 cache/result/shard 读取 batch 的工具。
3. 在 NR PUSCH 内部实现 link batching，batch size 从 8、16、32、64 逐步试；正式配置中保留可调 batch size 和失败降级记录。
4. 增加 UE range shard：`rx_start/rx_count` 或 `rx_indices`，每个 shard 写独立 `result_xxxx.h5`，manifest 负责把局部 index 映射回全局 UE/BS。
5. 把 `PUSCH-only` 泛化为 PHY-only harness，保留 NR 专属指标。
6. 给 custom OFDM / WiFi-like 新模块接入同一个 cache、batch、result shard、benchmark harness。
7. 单独开展 RT-only 调参矩阵，先 3x3000，再 5x5000；所有语义变化都进入报告。

这套顺序可以让 NR PUSCH 先拿到最直接的性能收益，同时不把后续制式扩展锁死在 NR 的内部对象模型里。
