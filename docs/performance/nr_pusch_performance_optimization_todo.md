# NR PUSCH 性能优化 TODO

日期：2026-05-14

来源：性能优化实验与生产化验收。本文只保留仍需推进的事项、实验结论和建议实施方式；已经完成并通过验收的能力不再列入 TODO 表。

已完成能力以 `docs/performance/nr_pusch_sharded_productionization.md` 为准，包括：

- CLI `--max-tx` / `--max-rx` override 修复和测试。
- 配置驱动 debug profiling。
- `run-full` 原生 UE/RX shard 输出，直接生成 `result_xxx.h5`。
- 4 GPU UE shard 运行与 `6x8884` 全量验收。
- NR PUSCH SU-MIMO batch 配置与 batch64 验证。
- 空间谱 link chunk 生成。
- HDF5 compression 配置传递。

相关背景文档：

- `docs/performance/nr_pusch_5x5000_profile_zh.md`
- `docs/performance/nr_pusch_optimization_iteration_1.md`
- `docs/performance/performance_optimization_design_guide.md`
- `docs/performance/nr_pusch_sharded_productionization.md`

## Tag 说明

| Tag | 含义 |
|---|---|
| 🟡 `PARTIAL` | 已有初步能力或实验结果，但还需要继续产品化或扩大验证 |
| 🟠 `TODO` | 最初计划中提到，但仍未完成 |
| 🔴 `NEW-TODO` | 最初没有明确提出，但根据最新验收结果应该补齐 |

## 高优先级 TODO

| 优先级 | Tag | 项目 | 当前状态 | 下一步 |
|---|---|---|---|---|
| P0 | 🟠 `TODO` | HDF5 写入深度优化 | `6x8884` shard 验收中，HDF5 write 仍是最大阶段，总计约 `439.6 s` | 系统测试 dataset chunk shape、压缩算法、压缩等级、flush 策略、并行写文件数和 schema 校验成本 |
| P0 | 🔴 `NEW-TODO` | shard 后的 reader / dataset loader | 当前写盘已变为多个 `result_xxx.h5`，训练或分析侧需要稳定读取入口 | 增加统一 reader，按 manifest 聚合多个 result 文件，并支持全局 UE/BS 索引定位 |
| P1 | 🟡 `PARTIAL` | 大规模空间谱继续优化 | 已支持 link chunk；`6x8884` 中 `array_outputs` 约 `75.1 s`，仍有优化空间 | 复用 steering matrix、减少重复归一化和中间数组分配；评估不同 `link_chunk_size` 对 RSS/耗时的影响 |
| P1 | 🟠 `TODO` | RT 参数调优实验 | RT 是独立 GPU compute-bound 阶段；`6x8884` 中 `rt_solve` 约 `287.2 s` | 建立 RT-only 实验矩阵，同时记录 path_count、LoS/NLoS、CFR 差异和耗时 |
| P1 | 🟡 `PARTIAL` | visualization 开销优化 | shard 模式默认只画 first shard；`6x8884` 中 visualization 仍约 `43.7 s` | 减少重复 HDF5 打开、重复 Matplotlib 初始化和重复坐标计算；必要时提供 `minimal` plot bundle |

## 仍需推进的设计项

| Tag | 项目 | 已有基础 | 剩余缺口 | 建议验收 |
|---|---|---|---|---|
| 🟡 `PARTIAL` | 通用模块性能接口 | 已有 debug profiling、shard manifest、NR PUSCH batch 和性能设计指南 | RT cache、PHY-only benchmark、write-only benchmark 还没有形成稳定 CLI/API | 新增 `benchmark rt-only`、`benchmark phy-only`、`benchmark write-only` 或等价入口，避免端到端耗时掩盖模块瓶颈 |
| 🟡 `PARTIAL` | batch size 自适应 | 已有正式配置项和 batch fallback；RTX 4090、4x4 SU-MIMO 下 batch64 已通过验收 | 不同 GPU、UE/BS 规模、频域配置下的稳定 batch 边界未知 | 增加 warmup sweep 或自动降级策略报告，记录 requested/effective batch、OOM 和 fallback |
| 🟡 `PARTIAL` | 多 GPU 调度策略 | 4 GPU shard 已通过 `6x8884` 验收 | 还未测试 6/8 GPU、动态负载均衡、尾 shard 调度和 GPU 忙碌场景 | 对 2/4/6/8 GPU 做扩展性曲线，记录每 shard 耗时、GPU util、尾 shard 空闲损失 |
| 🟠 `TODO` | 关闭 spectrum / visualization 的隔离对比 | debug 日志已经能拆阶段耗时 | 缺少配置开关矩阵下的系统对比 | 对比 spectrum off/on、visualization off/on、三类 spectrum source 的增量成本 |
| 🟠 `TODO` | 真实 write-only 从数组写盘 benchmark | 当前 HDF5 write 已被定位为最大瓶颈 | 还没有从已有内存数组或轻量中间文件直接写盘的独立 benchmark | 构造固定 shape 和 dtype 的数组，只测 writer、compression、schema validate 和 manifest |
| 🟠 `TODO` | `aoa_heatmap_label` / `spatial_spectrum_label` 冗余处理 | 两者当前语义上是兼容 alias | 本轮明确不处理字段冗余 | 输出字段精简阶段再决定保留一个物理 dataset，并用 attrs/manifest 说明兼容名称 |

## 建议执行顺序

1. 做 HDF5 写入实验：chunk/compression/flush/schema validate 分别测，优先降低最大阶段耗时。
2. 增加 shard-aware reader / dataset loader，保证多文件输出能被训练和分析稳定消费。
3. 建立 RT-only、PHY-only、write-only benchmark，避免后续优化只能看端到端总耗时。
4. 做空间谱 chunk 参数 sweep，找出 `link_chunk_size` 与内存峰值、耗时之间的平衡点。
5. 做 spectrum / visualization 开关矩阵，明确重型输出对生产运行的增量成本。
6. 做 batch size 自适应和多 GPU 扩展性测试，覆盖不同卡数和不同场景规模。
7. 单独处理输出字段冗余，包括 `aoa_heatmap_label` 和 `spatial_spectrum_label`。

## 回 main 前建议门槛

- `uv run ruff check .` 通过。
- `uv run pytest` 通过。
- 3x3000 端到端 full-output 通过，schema、figures、manifest 正常。
- 6x8884 shard full-output 通过，所有 `result_xxx.h5` schema 正常。
- 对会影响输出结构的改动，必须更新 `docs/sys/07_config_and_h5_format.md`、配置模板和 schema 测试。
- 对仅属实验性的性能代码，必须明确是否保留在性能分支，不能无说明带入 `main`。
