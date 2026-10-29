# DiffeRT 交叉验证子项目

用 **DiffeRT**（JAX 可微射线追踪，0.12.0）作为**独立第二引擎**，在与 Sionna RT 完全相同的
场景、材质、阵列和频率契约下生成理想 CFR，用来量化"换一个传播引擎"时下游模型的鲁棒性。

本子项目**不接入** `sionna_measurement_sim` 的仿真系统：不导入其任何模块，只读取它的产物
（`scene.xml`、prepared label、339 CFR 数据集）做对比。

## 为什么是 DiffeRT

| 维度 | DiffeRT | 说明 |
|---|---|---|
| 场景导入 | `Scene.load_xml()` 直接读 Sionna/Mitsuba `scene.xml` | 339 场景零转换，实测 46k 三角面 0.1–0.3 s |
| 材质 | ITU-R P.2040-4 全表，别名 `itu_concrete/itu_wood/...` | 与 339 场景四种材质逐字对应 |
| 求解器 | `sbr` / `hybrid` / `exhaustive` | exhaustive 在 46k 面上内存爆炸（实测 70 GB 被 kill），SBR 2–4 s |
| 极化/方向图 | `tx/rx_polarization="V"` = 单位幅度、θ̂ 方向 | 等价于 Sionna `pattern="iso", polarization="V"`，LoS 幅度对 Friis 比值 0.9993 |
| 输出 | 每径 `(τ, 复增益)` | CFR 自己合成 `Σ a_k e^{-j2πf τ_k}`，与 Sionna 契约一致 |

## 目录

```
subprojects/differt_crosscheck/
├── differt_crosscheck/        # 库代码
│   ├── config.py              # 冻结契约（频率网格、阵列、求解器）
│   ├── antenna.py             # iso+V 说明与显式实现
│   ├── scene.py               # 场景加载 + 阵列按 Sionna 公式布放
│   ├── solver.py              # SBR 追迹 + 分块 vmap + CFR 合成
│   ├── io.py                  # HDF5 写出（镜像 Sionna 数据集布局）
│   ├── metrics.py             # 相似度指标
│   └── select.py              # 大/中/小场景 × 1 BS × 10 UE 选取
├── scripts/
│   ├── bootstrap_env.sh       # 建隔离 venv（不碰仓库根 .venv）
│   ├── select_links.py        # 生成 links/selection.json
│   ├── run_links.py           # 跑 DiffeRT，写 cfr/<scene>.h5
│   └── compare_with_sionna.py # 与 339 结果逐链路对比
└── tests/                     # 阵列几何、天线等价性
```

数据落在数据盘，与 Sionna 输出同级、目录名自带区分：

```
/data/sunmeiyuan/projects/sionna/outputs/
├── front3d_339_dichasus_like_cfr_v1/     # Sionna 参考（只读）
└── differt_rt_crosscheck_v1/             # 本子项目产出
    ├── links/selection.json
    ├── cfr/<scene>.h5, run_summary.json
    └── compare/metrics_per_link.csv, summary.json, figures/
```

## 快速开始

```bash
cd subprojects/differt_crosscheck

./scripts/bootstrap_env.sh                 # 一次性：建 .venv-differt
PYTHONPATH=. .venv-differt/bin/python tests/test_geometry.py
PYTHONPATH=. .venv-differt/bin/python tests/test_antenna.py

PYTHONPATH=. .venv-differt/bin/python scripts/select_links.py
PYTHONPATH=. .venv-differt/bin/python scripts/run_links.py            # 3 场景 / 30 链路
PYTHONPATH=. .venv-differt/bin/python scripts/compare_with_sionna.py
```

常用参数：

```bash
# 只跑一个场景（先验证链路）
... run_links.py --scene front3d_3682

# 打开绕射 / 透射（更贴近 Sionna 的 max_depth=4 + diffraction + refraction）
... run_links.py --diffraction --transmission

# 换后端
... run_links.py --tracer hybrid --order 3 --num-rays 200000
```

## 契约对齐（与 `docs/proposals/dichasus_like_cfr_v1.md`）

| 项 | 值 |
|---|---|
| 中心频率 | 3.438 GHz |
| 子载波 | 768 × 30 kHz，3,426,495,000 – 3,449,505,000 Hz |
| BS 面板 | 4×8 平面阵，0.04364 m（0.5005λ），iso + V，逐 BS yaw |
| UE | 1×1，iso + V |
| CFR | `[link, 32, 1, 768] complex64`，不加功率、不做归一化 |
| 时延 | 绝对传播时延，不消除公共时延 |

阵列布放严格按 Sionna `PlanarArray` 公式：`y = d·j − (C−1)d/2`、`z = −d·i + (R−1)d/2`、
`x = 0`，扁平序号 `k = i·C + j`，再按 yaw 绕 +z 旋转。`tests/test_geometry.py` 钉住了这一点。

## 已知差异（这是"对照组"的一部分，不是 bug）

1. **SBR 非穷举**：候选由有限射线发现（默认 40 万条），不保证覆盖小立体角路径。
2. **绕射默认关闭**：Front3D 网格普遍非流形，DiffeRT 会把非流形边排除在绕射之外；开启后
   与 Sionna 的绕射集仍不会完全一致。
3. **透射模型**：DiffeRT 走 Fresnel/slab 转移矩阵，与 Sionna 的实现细节不同。
4. 因此不追求逐比特一致，只比较统计意义上的相似度（见 `metrics.py`）。

## 可行性结论（2026-10-29 实测）

目标配置是"对齐 Sionna：`max_depth=4` + 绕射 + 透射"。实测结果：

| 配置 | 结果 | 证据 |
|---|---|---|
| 全网格 + 绕射候选 | **不可行** | front3d_3682（4.7 万面）候选生成 RSS 冲到 **279 GB** 后被 kill |
| 仅建筑网格（1,030 面）+ 反射/绕射/透射 | **不可行** | hybrid、order=2 即 **43 GB** 且迟迟不返回 |
| 全网格 + order=4 反射（无绕射/透射） | **可行** | 3682：824 s（2.48 万条路径）；2508：547 s（0.55 万条路径） |
| SBR 找跨房间 NLoS | **找不到** | 单链路 order=4，射线 40 万 / 200 万 / 800 万 → 有效路径均为 **0** |

原因不是性能调优，而是机制缺失：室内跨房间 NLoS 主要靠**穿墙透射**（Sionna `refraction=true`）
和绕射，而 DiffeRT 的 SBR 只能发现反射；要补上这两类相互作用必须走图式候选生成
（`hybrid`/`exhaustive`），其开销随面数幂增，在 Front3D 网格上不可用。

### order=4 与 Sionna 的相似度（20 条链路）

| 场景 | cos_sim | 功率偏差 | PDP 相关 | 结论 |
|---|---:|---:|---:|---|
| front3d_3682（小，LoS 为主） | 0.103 | +0.9 dB | 1.000 | 能量与 PDP 对得上，细粒度 CFR 不相关 |
| front3d_2508（中，多房间 NLoS） | 0.079 | **-41.0 dB** | 0.787 | 缺透射/绕射，能量直接掉 41 dB |

对比 order≤2 的 3682（cos_sim 0.102、+0.9 dB）：**把阶数从 2 提到 4 几乎没有改善**，
说明差距来自"机制集合"而不是"阶数"。

### 因此

* DiffeRT 在本项目里只能作为**"LoS + 镜面反射"子集**的独立对照，不能作为 Sionna 全机制 CFR 的
  等价第二引擎。
* 若后续仍需要逐链路可比的全机制对照，必须先做几何简化（建筑-only、合并共面、单房间裁剪）
  并把 `max_diffractions` / `max_transmissions` 限制为 1，否则图式候选生成不可用。
* 大场景（front3d_5704，约 50 万面）在 order=4 下未跑完（order=1 已 196 s），需要 GPU 或简化几何。

## 2 跳 + 反射 + 透射：实测（2026-10-29）

配置：`tracer=exhaustive`、`order=2`、`reflection + transmission`、材料厚度 0.1 m
（对齐 Sionna `DEFAULT_THICKNESS`）、建筑-only 几何、BS 4×8、768×30 kHz。

| 场景 | 几何 | 链路 | 耗时 | 峰值内存 | 输出 |
|---|---|---:|---:|---:|---:|
| front3d_3682（小，18.8 m²） | 建筑-only 1,030 面 | 10 | 233 s | 26 GB | 1.91 MiB |
| front3d_2508（中，73.3 m²） | 建筑-only 1,934 面 | 10 | 进行中 | ~48 GB | — |

关键结论：

1. **小房间可用**。相对 Sionna：`cos_sim=0.103`、功率偏差 `+1.0 dB`、PDP 相关 `0.999`，
   与纯反射 4 跳（0.103 / +0.9 dB）一致——该场景 10 条链路都是 LoS 主导，透射没有增量。
2. **order=2 的候选浪费极大**：3682 单场景生成 15,774,804 个候选，最终有效 7,867（2005:1），
   时间和内存基本都花在候选生成上。
3. **多房间场景下 ≤2 跳覆盖不到深 NLoS 链路**。front3d_2508 里 4 条 LoS 链路在 order=0 就
   有 32 条有效路径，另 6 条（Sionna 功率 < -80 dB）在 order 0/1/2 的有效路径都是 **0** ——
   它们至少需要 3 次相互作用（两次穿墙等），而 order=3 在当前实现下不可行。
4. **`hybrid` 在 2508 上会触发 DiffeRT 内部 Warp 切片 bug**（`Invalid indexing in slice:
   8214:8192:1`，固定 8192 缓冲越界），改用 `exhaustive` 并把 32 个端口分块
   （`--rx-chunk 8`）后内存可控（约 43 GB/组）。

### 因此这套配置的适用范围

能覆盖 LoS + 一次穿墙 + 两次反射这一档，对"同房间 / 隔一道墙"的链路与 Sionna 基本对得上；
跨多个房间的深 NLoS 链路对不上，且不是调参能解决的（需要更高阶，而高阶不可行）。

## 实现坑位（都已踩过并写进代码注释）

| 坑 | 现象 | 处理 |
|---|---|---|
| 传播相位重复计入 | 逐频 `fields` 已含 `exp(-j2πf·d/c)`，再乘 `exp(-j2πfτ)` 会让不同长度路径的相对相位错乱（|H| 和 PDP 仍然对，但复相关≈0） | 直接对逐径复数场求和，不再乘延迟相位（`solver.py`） |
| 候选未被裁剪 | order 2 时 4.3k 有效但保留 32k 槽位，order 4 更多 | 求解前按 `mask.any()` 裁剪候选轴 |
| 频繁小批调用 | 768 频点 / chunk=8 × 10 tx ≈ 960 次 dispatch，单场景卡数分钟 | `freq_chunk=128`、`tx_chunk=2` |
| 一次请求多阶 | `order=(0,1,2,3,4)` 会按 `max_num_candidates` 补齐，曾触发 1 TB 分配 | 逐阶追迹后按 CFR 线性叠加 |
| `max_num_candidates` 偏小 | 触发内部 `Invalid indexing in slice` FFI 错误 | 保持 ≥2e5 |
| JAX 常量捕获 | 把 paths 闭包进 jit 会捕获 7.2 GB 常量 | paths 作为实参、mesh 保持静态（绕射几何需要具体值） |
| 图样 API 以位置为方向 | `polarization_vectors` 收到的是 `T + r̂`，隐含天线在原点 | 多发射机场景使用 `"V"` 快捷方式；显式天线类仅用于单发射机/测试 |

## 与 PyLayers 路线的关系

`subprojects/pylayers_crosscheck/` 是另一条（2.5D `.lay`）对照路线，两者的场景表达方式不同，
不要混用产物。
