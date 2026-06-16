# Performance TODO

本页记录运行成本、吞吐、内存、GPU 调度、写盘和可视化开销相关 TODO。顶部列表按当前重要程度排序；
每次修改都要重新检查顺序。

## Priority List

| 顺位 | ID | TODO | 简述 |
|---:|---|---|---|
| 1 | PERF-002 | RT 参数调优实验 | 建立 RT-only 实验矩阵，量化 max_depth、反射/折射/绕射等配置对耗时和 CFR 的影响。 |
| 2 | PERF-003 | 大规模空间谱优化 | 复用 steering matrix、减少中间数组，并 sweep `link_chunk_size` 的耗时/RSS 平衡。 |
| 3 | PERF-001 | HDF5 写入深度优化 | 在 `mixed`/`gzip_level` 已落地后，继续测试 buffered/append writer、chunk、flush 和 schema validate 成本。 |
| 4 | PERF-004 | visualization 开销优化 | 降低采样报告中的 HDF5 打开、Matplotlib 初始化和重复坐标计算成本。 |
| 5 | PERF-006 | spectrum / visualization 开关矩阵 | 隔离评估 spectrum off/on、visualization off/on 和不同 spectrum source 的增量成本。 |
| 6 | PERF-007 | batch size 自适应 | 记录并自动选择不同 GPU、UE/BS 规模和频域配置下稳定 batch 边界。 |
| 7 | PERF-008 | 多 GPU 调度扩展性 | 测 2/4/6/8 GPU shard 扩展性、尾 shard 调度和 GPU 忙碌场景。 |

## Details

### PERF-001: HDF5 写入深度优化

目的：大规模 full-output 中 HDF5 write 仍是主要瓶颈之一，需要系统性调优。

涉及模块：`sionna_measurement_sim/io/hdf5_writer.py`、schema validator、output compression config、
manifest、`benchmark write`。

当前状态：已增加 `output.compression: "mixed"` 与 `output.gzip_level`。正式 full 仿真中
高熵复数观测数组跳过 gzip，路径表/稀疏数组继续 gzip；`gzip_level: 1` 是当前 64 PRB
任务推荐值。`front3d_0002 0p5` full 对照中 `hdf5_write` 从约 365 s 降到约 114 s，
详见 `docs/performance/single_scene_full_perf_2026-06-16.md`。

下一步方向：用户建议的 “compute chunk 小批量 + write batch 缓冲/append” 值得作为二阶段
实验，但当前探针显示单纯 one-file 或 extendable 大数组不一定更快；收益主要可能来自减少重复
metadata/group/attrs/schema 成本，而不是大数组写入本身。

验收标准：继续比较 buffered writer、bundle HDF5 contract、chunk shape、flush 策略、并行写文件数和
schema validate 开关；输出推荐配置和风险说明；真实 shard 或 `benchmark write` 有可复现实验结果。

重点提醒：优化写盘不能破坏 HDF5 schema、自包含 shard 和 manifest 契约。不要让多个 GPU worker
同时写同一个 HDF5；若做 bundle/append 模式，需要新增 reader/schema/manifest 适配和 checkpoint。

### PERF-002: RT 参数调优实验

目的：RT solve 是独立 GPU compute-bound 阶段，需要知道不同场景和参数下的成本/质量边界。

涉及模块：RT config、truth pipeline、debug profiling、path truth/CFR summary scripts。

验收标准：基于 `benchmark rt` 建立 RT-only 矩阵，记录耗时、显存、path_count、LoS/NLoS、CFR 差异和失败边界；
给出 SRS/PUSCH 生产模板的推荐参数。

重点提醒：性能文档是实验记录；最终默认配置要回写到 `config/README.md` 和 sys docs。

### PERF-003: 大规模空间谱优化

目的：空间谱在大规模输出中仍有明显 CPU/RSS 成本，需要减少重复计算和中间数组分配。

涉及模块：`sionna_measurement_sim/phy/spatial_spectrum.py`、array output builder、HDF5 writer、
visualization。

当前状态：已新增 covariance-based Bartlett 入口，并让 observation spectrum 复用
`rx_snapshot_matrix` covariance；收益较小。batched matmul 投影在单进程微基准更快，但正式多 worker
运行中因 CPU/BLAS 线程竞争变慢，已放弃。

验收标准：基于 `benchmark spectrum` 评估 steering matrix cache、按 link chunk 复用、归一化优化和不同 `link_chunk_size`；
输出 RSS/耗时曲线；确认 scene/global angle 和 array orientation 语义不回退。

重点提醒：空间谱已统一到 scene frame，优化时不要重新引入本地坐标或 PlanarArray 编号错误。

### PERF-004: visualization 开销优化

目的：采样可视化应是轻量诊断，不应成为 shard 完成后的长尾阶段。

涉及模块：`sionna_measurement_sim/visualization/report.py`、visualization config、CLI visualize。

验收标准：减少重复 HDF5 打开、重复 Matplotlib 初始化和重复坐标计算；必要时新增 `minimal`
plot bundle；小规模和真实 first-shard visualization 都能正常输出。

重点提醒：`path_samples` 已限制为单条 UE-BS link；不要恢复多链路默认混画。

### PERF-006: spectrum / visualization 开关矩阵

目的：量化重型输出对生产运行的增量成本，指导默认模板和实验配置。

涉及模块：config templates、debug profiling、array spectrum、visualization。

验收标准：结合 `benchmark spectrum` 与真实小规模 run，比较 spectrum off/on、visualization off/on、`truth_cfr`/`cfr_est`/`rx_grid` 等 source
组合；输出推荐默认值和 “需要诊断时再打开” 的说明。

重点提醒：不要只看总耗时，要拆 RT、PHY、array、write、visualization 阶段。

### PERF-007: batch size 自适应

目的：当前已有 PUSCH batch 和 fallback，但不同 GPU、规模和频域配置下的稳定边界未知。

涉及模块：PUSCH/SRS shard runner、batch config、debug profiling、fallback 策略。

验收标准：warmup sweep 或自动降级策略能报告 requested/effective batch、OOM/fallback 原因和最终 batch；
覆盖至少两种 GPU 或两种规模配置。

重点提醒：自动调参必须可复现，不能让同一配置在不同机器上静默改变输出语义。

### PERF-008: 多 GPU 调度扩展性

目的：4 GPU shard 已验证，但还需要 2/6/8 GPU、动态负载均衡、尾 shard 和 GPU 忙碌场景的扩展性数据。

涉及模块：shard runner、GPU binding、manifest、debug profiling。

验收标准：输出不同 GPU 数的 wall time、per-shard time、GPU util、失败/fallback 记录和尾 shard 空闲损失；
给出生产运行建议。

重点提醒：不要并发写同一个 HDF5；继续保持每进程独立 shard 文件。
