# NR PUSCH 性能优化 TODO

日期：2026-05-14

来源：性能优化实验。本文只保留可带回 `main` 的待办事项、实验结论和建议实施方式；不依赖实验分支中的具体代码、提交或输出目录。

本文从最初的 5x5000 性能报告和本轮优化计划出发，只记录非 `DONE` 项：已经部分完成但还不能作为生产能力使用的工作、计划中提到但尚未完成的工作、本轮额外发现的问题，以及最初没明确提出但现在应该补齐的工作。

更新：`run-full` 原生 UE/RX shard、CLI override、debug profiling、NR PUSCH SU-MIMO batching、空间谱 chunk 和 HDF5 compression 配置已经完成生产化验收，详见 `docs/performance/nr_pusch_sharded_productionization.md`。下方早期条目中涉及这些能力的内容以新报告为准，后续 TODO 重点转向 HDF5 写盘、RT-only 参数矩阵、空间谱继续优化和 visualization 开销。

相关背景文档：

- `docs/performance/nr_pusch_5x5000_profile_zh.md`
- `docs/performance/nr_pusch_optimization_iteration_1.md`
- `docs/performance/performance_optimization_design_guide.md`

## Tag 说明

| Tag | 含义 |
|---|---|
| 🟡 `PARTIAL` | 最初提到，本轮做了一部分，但还不能作为生产级能力 |
| 🟠 `TODO` | 最初提到，但本轮没有完成 |
| 🔵 `EXTRA` | 最初没有明确提到，本轮额外发现或额外完成，需要决定是否纳入主线 |
| 🔴 `NEW-TODO` | 最初没有明确提到，本轮也没有完成，但现在看应该补齐 |

## 高优先级 TODO

| 优先级 | Tag | 项目 | 当前状态 | 下一步 |
|---|---|---|---|---|
| P0 | 🔴 `NEW-TODO` | 修复 CLI `--max-tx` / `--max-rx` override 语义 | 6x8884 验收时已发现：当 CLI 参数等于 argparse 默认值时，可能无法覆盖 YAML | 修改 CLI 参数默认值或引入“是否显式传参”判断；补单测覆盖 YAML `max_tx=3` + CLI `--max-tx 6` |
| P0 | 🟡 `PARTIAL` | run-full 原生 shard 写盘 | 实验证明 `result_xxx.h5` + manifest 语义可行；真实端到端仍写单个 `results.h5` | 在 `run-full` 中按 UE shard 直接生成 `result_000.h5`、`result_001.h5`，避免先写完整 HDF5 再拆 |
| P0 | 🟠 `TODO` | HDF5 写入深度优化 | 6x8884 中 HDF5 write 约 `431.4 s`，已经是最大阶段 | 系统测试 HDF5 chunk shape、压缩策略、flush 策略、按 shard 写入和 schema 校验成本 |
| P1 | 🟠 `TODO` | 2 GPU / 4 GPU UE shard 试验实现 | 尚未实现正式多 GPU shard 仿真 | 基于 run-full 原生 shard，每进程绑定一张 GPU，不并发写同一个 HDF5 |
| P1 | 🔴 `NEW-TODO` | 大规模 UE-chunk vectorized spectrum | 当前 Bartlett 向量化只对中小规模收益明显；5x5000/6x8884 回退保守路径 | 以 UE chunk 为单位复用 steering matrix，控制内存峰值，同时避免逐 link Python 循环 |

## 部分完成但仍需推进

| Tag | 项目 | 已完成 | 未完成 | 建议验收 |
|---|---|---|---|---|
| 🟡 `PARTIAL` | 多文件输出 / shard | 已验证 `result_000.h5` 风格、manifest 字段和 shard 范围语义 | 未接入生产级 `run-full`；从完整 HDF5 再拆分不是提速方案 | 3x3000 / 5x5000 端到端直接产出多个 `result_xxx.h5`，每个文件 schema 通过，manifest 可定位全局 UE/BS |
| 🟡 `PARTIAL` | 空间谱优化 | 3x3000 write-only 中 `array_outputs` 从 `18.0 s` 降到 `5.1 s` | 大规模下受内存阈值影响，5x5000/6x8884 仍走保守路径 | 5x5000 和 6x8884 下 `array_outputs` 相对本轮结果继续下降，且 RSS/显存不过高 |
| 🟡 `PARTIAL` | visualization 优化 | 已做选中 link 的 HDF5 slice 复用和部分预计算 | 5x5000 visualization 仍约 `43.7 s`，收益有限 | 采样图不变的前提下减少重复 HDF5 打开、重复 Matplotlib 初始化和重复坐标计算 |
| 🟡 `PARTIAL` | 多 GPU 利用 | 多 GPU 已用于并行实验和子任务调度 | 没有正式多 GPU UE shard 仿真 | 2 GPU 至少接近 `1.6x`，4 GPU 至少接近 `2.8x`，输出不出现写锁/竞态 |
| 🟡 `PARTIAL` | 通用模块设计指南 | 已形成 RT cache、PHY-only benchmark、result shard、CFR batch 的分层建议 | 尚未全部落到代码接口 | 将这些概念抽成稳定的通用接口，避免新 PHY 模块复制 NR PUSCH 的专属调度路径 |

## 最初提到但本轮未完成

| Tag | 项目 | 原始动机 | 当前缺口 | 下一步 |
|---|---|---|---|---|
| 🟠 `TODO` | RT 参数调优实验 | RT 是独立 GPU compute-bound 阶段，需要单独优化 | 尚未系统测试 `max_depth`、diffuse、refraction、diffraction 等参数 | 建立 RT-only 实验矩阵，同时记录 path_count、LoS/NLoS、CFR 差异和耗时 |
| 🟠 `TODO` | 关闭 spectrum / visualization 的隔离对比 | 判断重型输出对端到端耗时的影响 | 目前有拆分 benchmark，但缺少开关矩阵 | 对比 spectrum off/on、visualization off/on、三类 spectrum source 的增量成本 |
| 🟠 `TODO` | `aoa_heatmap_label` / `spatial_spectrum_label` 冗余处理 | 两者语义上是 alias，可能重复写大数组 | 用户已明确本轮先不处理 | 输出字段精简阶段再决定保留一个物理 dataset，并用 attrs/manifest 明确另一个名称的兼容语义；不要依赖实验分支实现或特殊链接机制 |
| 🟠 `TODO` | 真实 write-only 从数组写盘 benchmark | 更准确衡量 HDF5 writer 成本 | 当前 write-only 仍更接近 clone/split/重算 array 的组合 | 增加从已有内存数组或轻量中间文件直接写 HDF5 的 benchmark |
| 🟠 `TODO` | batch size 设备/场景自适应 | batch size 受 GPU、UE/BS 规模和 Sionna 内部实现影响 | 当前经验值是 RTX 4090、4x4 SU-MIMO 下 batch=64 | 增加正式配置项和 sweep/warmup 探测；失败时自动降级并记录 |

## 额外发现或额外完成后需要决策

| Tag | 项目 | 状态 | 是否建议带回 main | 备注 |
|---|---|---|---|---|
| 🔵 `EXTRA` | Bartlett 空间谱向量化 | 已验证中小规模向量化能显著降低 `array_outputs` 耗时 | 建议，但需要补大规模 chunk 版本 | 中小规模收益明显，生产合入前需要避免大规模回退 |
| 🔵 `EXTRA` | CLI override bug | 已发现，未修复 | 必须修 | 直接影响 benchmark 规模可信度 |
| 🔵 `EXTRA` | 6x8884 真实全量输出验证 | 已完成 | 报告可带回 main | 可作为 batch64 稳定性证据 |
| 🔵 `EXTRA` | 性能设计指南 | 已形成初版 | 建议带回 main | 有助于后续 NR、WiFi-like、custom OFDM 共用性能底座 |
| 🔵 `EXTRA` | `result_xxx.h5` reader / manifest 语义 | 已验证基本语义 | 建议保留设计，具体代码是否合入需看 run-full shard 方案 | 当前实验说明多文件命名和 manifest 可行，但不能代表最终写盘性能 |

## 建议执行顺序

1. 修复 CLI override 语义，保证所有后续 benchmark 的规模可信。
2. 将 NR PUSCH SU-MIMO batch size 沉淀为正式配置项，环境变量最多作为临时 override。
3. 做 run-full 原生 UE shard 写盘，直接产出 `result_xxx.h5`。
4. 基于原生 shard 做 2 GPU、4 GPU UE shard 试验实现。
5. 做 HDF5 chunk/compression/flush/schema 校验成本实验。
6. 做 UE-chunk vectorized spectrum，使 5x5000、6x8884 不再回退保守路径。
7. 做 RT-only 参数矩阵实验，分清性能收益和传播语义变化。
8. 单独处理输出字段冗余，包括 `aoa_heatmap_label` 和 `spatial_spectrum_label`。

## 回 main 前建议门槛

- `uv run ruff check .` 通过。
- `uv run pytest` 通过。
- 3x3000 端到端 full-output 通过，schema、figures、manifest 正常。
- 5x5000 标准 full-output 通过。
- 对会影响输出结构的改动，必须更新 `docs/sys/07_config_and_h5_format.md`、配置模板和 schema 测试。
- 对仅属实验性的性能代码，必须明确是否保留在性能分支，不能无说明带入 `main`。
