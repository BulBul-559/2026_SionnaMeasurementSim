# DICHASUS-like 理想 CFR 测试配置 v1：冻结方案与逐项审查

冻结日期：2026-10-26。

模板：[`config/tasks/nr_srs_64prb_dichasus_like_cfr_v1.yaml`](../../config/tasks/nr_srs_64prb_dichasus_like_cfr_v1.yaml)。

状态：**参数已冻结，逐设备朝向、339场景输入准备和真实小规模RT已验收**。
逐点朝向已从label接通RT，旧label沿用公共默认值；相关功能FEAT-ANT-001已归档到
[已完成记录](../todo/history.md#feature)。339场景全量仿真尚未启动；已按用户要求完成
[大中小三个完整场景试跑](../performance/dichasus_like_cfr_v1_three_scene_pilot_20261026.md)，供模型测试。
HDF5继续写既有设备朝向字段，schema版本不变。

## 实验目的和已接受的差异

在原 Front3D 测试集的相同地图、材质和 BS/UE 位置上，生成新阵列配置的理想 CFR，
供现有 VisionRadioLoc CSI 编码器评估。旧配置为每 BS `16 RX × 2 TX`，新配置为
每 BS `32 RX × 1 TX`；每条链路都可展平为 32 个天线对。

| 项目 | 原2988 | 冻结的v1 |
|---|---|---|
| 载频 | 3.5 GHz | 3.438 GHz |
| 接收阵列 | 4×4 | 4×8 |
| 发射阵列 | 1×2 | 1×1 |
| BS阵元间距 | 0.5λ | 0.04364 m，约0.500460622λ |
| BS朝向 | 公共 `[0,0,0]` | 逐个竖直朝向整张地图中心；已接通并验收 |
| UE朝向 | 固定 `[0,0,0]` | 相同 |
| 频率采样 | 768点、30 kHz、23.04 MHz | 完全保留，不裁剪、不删中心点 |
| 传播模型 | depth=4；LoS、镜面反射、折射、绕射 | 相同 |
| 数据产品 | 理想RT CFR | 相同 |

v1 **明确保留**与 DICHASUS 原始50 MHz/1024点及其中心769点去中心频点预处理的差异。
不增加 DICHASUS 波形、接收机、参考发射机同步、独立 RF 通道损伤或58/66 dB增益模型。
阵元方向图/极化继续使用 `iso`/V，这是理想化假设，并非已恢复实采天线型号。
不强制只保留两个 BS，不修改位置和安装高度，不增加轨迹，不重新划分 train/val/test。

## 已确认的339场景测试集

用户于2026-10-26指定并完成只读核对：

```text
dataset_root = /data/sunmeiyuan/projects/sionna/outputs/datasets/Front3d_2988_CFR_Only
split_root   = <dataset_root>/splits/m117_seed2026_scene_level
scene_list   = <split_root>/test_scenes.txt
scene_table  = <split_root>/scene_split.jsonl  # 仅取split=test
link_plan    = <split_root>/test_links.pt
```

测试名单为339个唯一场景，52个small_room、287个normal_room，与train/val不重叠。
名单文件SHA256：`fe2a270f7f60be7dada7793fe6fb08228e9836d84b749ae67db3b323d8ff25ab`，
与split_manifest中的记录一致，按原文件顺序保留，不重新抽样或排序。
模板示例改用测试集首个 `front3d_0015`；此前比较实验的 `front3d_4484` 属于train，
不能将该场景作为339测试场景之一。

| 核对项 | 结果 |
|---|---:|
| 原始UE–BS组合 | 934,128 |
| 原valid链接 | 817,938 |
| m117筛选后的测试链接 | 783,687 |
| 测试集最大BS数/场景 | 23 |
| 测试集最大UE数/场景 | 664 |
| 找到scene H5 | 339/339 |
| 找到源scene.xml、label及原run_config | 各339/339 |
| 找到打包的floorplan PNG和meta.json | 各339/339 |

`max_bs=64`、`max_ue=20000` 不会截断这339个场景。只做显式名单对应的文件检查，
没有递归扫描数据根目录。原始CFR shape由scene表核对为16×2×768，抽查两个H5确认一致。
目录名仍保留2988，但当前清理后的active总数是2987：train=2326、val=322、test=339；
复刻只使用这份明确的339测试名单。

339定义场景范围，783,687定义旧模型实际使用的m117链路子集
（`valid_mask=true`且`csi_power_db>=-117`）。生产时可保留934,128个原UE–BS组合；
逐链接对照应根据旧 `(scene_id, ue_id, bs_id)` 身份映射到新结果，不能直接复用旧H5行号，
也不能默认在新CFR上重新按功率阈值筛选仍得到相同样本集合。

`test_links.pt` 的783,687个global/local link id、UE/BS id、功率和validity已逐行与
原 `link_index.pt` 对照一致；`record_index` 是test内局部0..338序号，不能按整个
dataset manifest的场景行号解释。test场景名单和test链接计划的SHA256也已核对通过。
顺带发现 `split_manifest.json` 中train/all名单的内嵌hash未同步，而独立
`checksums.sha256` 与实际文件一致；test名单的两处hash均一致，本轮未修改源数据。

## 朝向和输出约定

地图中心使用原平面图 `meta.json` 的物理坐标：

```text
center_xy = origin_xy_m + extent_xy_m / 2
yaw_bs = atan2(center_y - bs_y, center_x - bs_x)
orientation_bs = [yaw_bs, 0, 0]   # 水平转向；pitch=roll=0；面板保持竖直
```

目标规则为显式逐点朝向优先于公共朝向，公共朝向缺省为 `[0,0,0]`。BS恰好位于
中心时使用公共值。UE继续固定 `[0,0,0]`。朝向按完整地图预处理，在 BS/UE
筛选和分片之前固定；不按分片内UE均值、不按所属子房间、不按遮挡情况重新选方向。
原RT中的遮挡/反射等传播仍保持启用。

`io/label_parser.py` 读取已选点的可选朝向，`domain/topology.py` 校验每行的有限3向量，
并按BS/UE角色映射到TX/RX；`adapters/sionna_rt/rt_solver.py::_build_scene()` 逐设备应用。
缺省或null保持公共配置，显式 `[0,0,0]` 也是有效覆盖；TX原有look-at行为只用于没有
显式覆盖的点。RX公共模式保持fixed。模板引用已准备好的
`outputs/front3d_339_dichasus_like_cfr_v1/prepared/labels/front3d_0015/label_panel_0p5_bs_center_yaw.json`。
不引入新的RX look-at模式，也不按局部UE重新估计方向。

输出和模型输入必须区分：

```text
原始HDF5 /channel/truth/cfr: [UE, BS, 32, 1, 768]
scene-level /cfr:           [link, 32, 1, 768]
现有编码器预处理后:           [B, 2, 32, 768]
```

RT保留原始复数CFR及功率/传播时延；能量归一化和实虚部分离继续由下游编码器完成。
通道的32×1物理语义不伪装成16×2。后续验收需记录 Sionna 的列优先阵元顺序与
DICHASUS适配后的顺序映射，不通过复制或插值凑天线数量。

## 逐项审查

按模板从上到下列出全部176个显式叶子选项。下表字段相对于对应节名；列表整体算一个
选项。未显式列出的Pydantic字段使用当前项目默认值；它们位于关闭的模块中，不能仅
通过开启 `phy.enabled` 就把本模板改称已经验收过的完整SRS仿真。

审查结论中的“参考”表示被上级开关或 `products=[cfr_truth]` 截断，不参与本轮CFR数值。
参数加载成功只证明schema有效，不代表朝向、硬件资源或整批场景已验收。

### `runtime`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `seed` | 42 | 传入RT solver；沿用原生产seed。改变阵列后不承诺与旧CFR逐元素相同。 |
| `device` | cuda | 沿用GPU执行/调度配置；不等于独占GPU资源。 |
| `require_gpu` | false | 保留默认；当前生产映射不使用它，不能当作GPU准入检查。 |
| `precision` | single | 保留默认；当前映射未以它切换RT精度，结果以实际输出dtype为准。 |
| `torch_deterministic` | false | 保留默认；当前映射不以此启用确定性算法。 |

### `debug`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `enabled` | false | 关闭额外profiling；保留正常run/manifest输出。 |
| `hardware_interval_s` | 2.0 | 参考：debug关闭。 |
| `link_log_interval` | 250 | 参考：debug关闭。 |
| `torch_synchronize` | false | 不增加逐阶段强制同步。 |
| `write_hardware_samples` | false | 不写额外硬件采样。 |

### `input`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `label_file` | prepared/labels/front3d_0015的center-yaw label | 已生成且朝向生效；339场景index分别指向对应派生label，原点集/顺序/ID保持不变。 |
| `scene_file` | front3d_0015/scene.xml | 测试集首个场景文件存在；每场景替换，必须与原测试集同源。 |
| `scene_id` | front3d_0015 | 339场景测试名单中的稳定场景ID，仅为模板示例。 |
| `map_id` | 空字符串 | 沿用原空值；不能用它替代原测试split清单。 |
| `label_schema` | simplesionna_v1 | 既有信息字段；不负责转换坐标/读取逐点orientation。 |
| `coordinate_system` | scene_local_xyz_m | 声明scene本地米制XYZ；当前不是坐标变换开关。 |
| `max_bs` | 64 | 是截断上限，不是生成64个BS；生产前确认每场景点数不超限。 |
| `max_ue` | 20000 | 是截断上限；检查实际点数，不能静默丢弃测试样本。 |

### `output`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `profile` | full | 使用现有full contract，受products裁剪。 |
| `products` | [cfr_truth] | 关键门控：只计算/写理想CFR及必要元数据，不跑PHY。 |
| `root_dir` | outputs/front3d_0015_dichasus_like_cfr_v1 | 新命名空间；逐场景覆盖，不覆盖原2988结果。 |
| `run_id_format` | {label_stem}_{timestamp} | 兼容字段；当前实际目录由root_dir/CLI决定，不能指望自动防覆盖。 |
| `hdf5_filename` | results.h5 | 未分片路径文件名；分片时由filename_pattern命名。 |
| `compression` | mixed | 沿用原CFR-only配置；不改变数值。 |
| `gzip_level` | 1 | 沿用原压缩等级。 |
| `save_full_paths` | false | 不写完整path表。 |
| `save_sampled_paths` | false | 明确关闭；真正裁剪由products保证。 |
| `save_raw_waveform` | false | 明确关闭；本轮也没有waveform计算。 |
| `sharding.enabled` | true | 沿用UE分片路径。 |
| `sharding.axis` | ue | 在uplink中分的是TX/UE轴，BS点集保持完整。 |
| `sharding.shard_size` | 20 | 沿用原CFR-only值；新32阵元接收侧尚未实测成本，OOM可fallback。 |
| `sharding.filename_pattern` | result_{shard_index:03d}.h5 | 符合shard命名约束；读者使用manifest，不能猜文件连续。 |
| `sharding.results_dir` | results | 合法相对子目录。 |
| `sharding.manifest_dir` | manifest | 合法相对子目录。 |
| `sharding.parallel_workers` | 8 | 原执行参数；必须按实际分配GPU数量覆盖。 |
| `sharding.recycle_workers` | true | 沿用成功后回收worker策略。 |
| `sharding.gpu_ids` | [0,1,2,3,4,5,6,7] | 原机器示例，不是资源授权；运行前替换成实际分配的设备。 |
| `sharding.visualization_mode` | none | 不生成逐shard图。 |
| `sharding.gpu_scheduler.enabled` | true | 使用项目内部显存调度；不要求外部集群调度器。 |
| `sharding.gpu_scheduler.free_memory_threshold` | 0.6 | 空闲比例阈值，不是预留/限额显存。 |
| `sharding.gpu_scheduler.scan_interval_s` | 0.2 | 沿用原CFR-only模板。 |
| `sharding.gpu_scheduler.cross_scene_pipeline` | true | run-scene-index才跨场景；单场景不产生其他场景。 |
| `sharding.fallback.enabled` | true | retryable错误允许拆分。 |
| `sharding.fallback.min_shard_size` | 1 | 可拆到单UE，不改变实验阵列。 |
| `sharding.fallback.split_factor` | 2 | 每次二分。 |
| `sharding.fallback.retry_errors` | [cuda_oom,drjit_array_limit] | 只重试已知资源错误，朝向/输入错误不能靠fallback解决。 |
| `sharding.fallback.failure_policy` | fail_run | 不静默跳过失败shard。 |
| `sharding.fallback.isolation_mode` | on_failure | 沿用原隔离策略。 |
| `sharding.postprocess.async_write` | false | 沿用同步写盘。 |
| `sharding.postprocess.max_pending_writes` | 16 | 参考：异步写盘关闭。 |
| `sharding.bundle.enabled` | false | 保持一计算shard一HDF5。 |
| `sharding.bundle.max_planned_shards_per_bundle` | 10 | 参考：bundle关闭。 |
| `sharding.bundle.bundles_dir` | bundles | 参考：合法相对目录。 |
| `sharding.bundle.filename_pattern` | bundle_worker{worker_index:03d}_{bundle_index:03d}.h5 | 参考：含两个必需占位符。 |
| `sharding.bundle.validate_schema` | true | 参考：bundle关闭；不等于本次已验证真实HDF5。 |

### `carrier`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `center_frequency_hz` | 3438000000.0 | 改为DCXX载频；同时用于波长/材质频率评估。 |
| `bandwidth_hz` | 23040000.0 | 用户明确保留原NR网格；不能写成DICHASUS的50MHz。 |
| `num_subcarriers` | 768 | 保持原网络长度，与64×12一致。 |
| `subcarrier_spacing_hz` | 30000.0 | 与23.04MHz/768严格一致；不删中心频点、不引入中心空缺。 |

频率由现有 `FrequencyGrid.from_center_bandwidth()` 生成：3,426,495,000 Hz至
3,449,505,000 Hz，间隔30,000 Hz；偶数点网格围绕中心对称，本来就不含中心频率。
不应再套用DICHASUS的去中心点操作。

### `antenna`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `bs_array.type` | planar | 匹配现有PlanarArray后端。 |
| `bs_array.num_rows` | 4 | 竖向4行。 |
| `bs_array.num_cols` | 8 | 横向8列，共32个单极化RX通道。 |
| `bs_array.vertical_spacing_lambda` | 0.5004606219947 | 在3.438GHz下对应0.04364m。 |
| `bs_array.horizontal_spacing_lambda` | 0.5004606219947 | 同上，不混用米和波长。 |
| `bs_array.polarization` | V | 保留原理想极化；没有实采型号校准声明。 |
| `bs_array.pattern` | iso | 保留原方向图；朝向仍改变阵元几何/空间相位，不代表定向面板增益。 |
| `bs_array.orientation_mode` | fixed | RX公共模式保持fixed；逐点覆盖不依赖新的look-at模式。 |
| `bs_array.orientation_rad` | [0,0,0] | 公共回退；有显式label朝向的BS使用预处理的中心yaw，已验收。 |
| `ue_array.type` | planar | 1×1退化为单天线，后端兼容。 |
| `ue_array.num_rows` | 1 | 与单TX一致。 |
| `ue_array.num_cols` | 1 | 从旧2列改为1列，不复制成2个TX。 |
| `ue_array.vertical_spacing_lambda` | 0.5 | 单阵元无间距效应，保留默认。 |
| `ue_array.horizontal_spacing_lambda` | 0.5 | 单阵元无间距效应。 |
| `ue_array.polarization` | V | 与旧配置相同。 |
| `ue_array.pattern` | iso | 与旧配置相同。 |
| `ue_array.orientation_mode` | fixed | 不增加UE轨迹/朝向估计。 |
| `ue_array.orientation_rad` | [0,0,0] | 保持UE方向不变。 |

### `array`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `spectrum.enabled` | false | 不计算Bartlett空间谱。 |
| `spectrum.sources` | [truth_cfr] | 参考；即使查看配置也不误引入观测CSI源。 |

### `visualization`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `enabled` | false | 不计算额外图，保留原floorplan资产。 |
| `plots` | [] | 不启用任何图列表。 |

### `rt`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `engine` | sionna_rt | 当前实际后端；不是切换其他solver的新增接口。 |
| `max_depth` | 4 | 与原2988一致。 |
| `los` | true | 保留直射路径；不保证每条链路都有LoS。 |
| `specular_reflection` | true | 保留镜面反射。 |
| `diffuse_reflection` | false | 保持旧实验未启用漫反射。 |
| `refraction` | true | 保持折射/穿透设置。 |
| `diffraction` | true | 保留绕射。 |
| `synthetic_array` | false | 逐阵元真实几何，不能为省时改为合成阵列。 |
| `normalize_cfr` | false | RT不做能量归一化，保留原始幅度。 |
| `normalize_delays` | false | 不消除传播绝对时延，避免改变频率相位。 |
| `merge_shapes` | false | 保留原几何加载方式。 |

### `link`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `duplex_mode` | tdd | 兼容项目契约；静态单向CFR不额外建模双工时序。 |
| `phy_link_direction` | uplink | UE=TX、BS=RX；决定32×1维度的角色语义。 |

### `phy`

下列各项均被 `enabled=false` 和产品计划截断；单独校验SRS resource builder仅用于
确认端口、符号、BWP设置内部自洽，不代表本轮运行了SRS接收机。

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `enabled` | false | 核心门控；不生成估计CSI。 |
| `standard` | nr_srs | 保留原模板类型，作为参考记录。 |
| `fft_size` | 768 | 参考；纠正旧模板遗留64，不参与本轮CFR；SRS按carrier确定网格。 |
| `cp_length` | 0 | 参考；不声称这是实采CP，也不参与RT CFR。 |
| `num_ofdm_symbols` | 14 | 参考；单slot沿用原设置。 |
| `subcarrier_spacing_khz` | 30 | 参考，与carrier自洽。 |
| `num_prb` | 64 | 参考，64×12=768。 |
| `num_layers` | 1 | 参考，与单TX一致。 |
| `num_antenna_ports` | 1 | 参考；随UE单天线由2改为1，避免遗留冲突。 |
| `perfect_csi` | false | PUSCH接收机参考字段；不表示本轮输出不是理想CFR。 |
| `channel_estimator` | srs_ls | 参考；本轮不做LS或插值。 |
| `receiver_failure_policy` | mark_invalid | 参考；不影响RT有效链路判定。 |
| `snr_db` | 30 | 参考；不向理想CFR注入30dB噪声。 |
| `tx_power_dbm` | 23 | 参考；不对RT传递函数乘发射功率，不是DCXX发射功率声明。 |
| `power.reference_tx_power_dbm` | 0 | 参考：PHY关闭。 |
| `power.apply_tx_power_to_grid` | true | 参考：没有TX grid，不改变RT CFR。 |
| `power.noise_mode` | absolute_thermal | 参考：无噪声链执行。 |
| `power.thermal_noise.temperature_k` | 290 | 参考：不参与CFR。 |
| `power.thermal_noise.noise_figure_db` | 7 | 参考：不是已确认的实采NF。 |
| `power.thermal_noise.bandwidth_hz` | null | 参考：无热噪声计算。 |
| `power.uplink_control.enabled` | false | 禁用功控。 |
| `iq.enabled` | false | 不写IQ。 |
| `iq.clean_output` | null | 无clean IQ产品。 |
| `iq.save_frequency_observed` | false | 无observed频域IQ。 |
| `iq.save_time_observed` | false | 无observed时域IQ。 |
| `iq.cp_length` | null | 参考：IQ关闭。 |
| `srs.slot_length_symbols` | 14 | 参考：沿用slot长度。 |
| `srs.start_symbol` | 12 | 参考：末两个符号。 |
| `srs.num_srs_symbols` | 2 | 参考：12+2=14，未越界。 |
| `srs.comb_size` | 2 | 参考：合法comb。 |
| `srs.comb_offset` | 0 | 参考：小于comb_size。 |
| `srs.bwp_start_prb` | 0 | 参考：从第一个PRB开始。 |
| `srs.bwp_num_prb` | 64 | 参考：覆盖全部64PRB，不越界。 |
| `srs.trigger_mode` | aperiodic | 参考：沿用原触发字段。 |
| `srs.periodicity_slots` | 1 | 参考：不生成时间序列。 |
| `srs.slot_offset` | 0 | 参考：合法offset。 |
| `srs.slot_number` | 0 | 参考：固定sequence上下文。 |
| `srs.sequence_type` | nr_zc | 参考：沿用当前standards-shaped subset，不声明完整3GPP。 |
| `srs.sequence_id` | 0 | 参考：固定序列ID。 |
| `srs.group_hopping` | enabled | 参考：沿用原设置，不是RT时变。 |
| `srs.sequence_hopping` | enabled | 参考：同上。 |
| `srs.cyclic_shift_multiplexing` | cyclic_shift | 参考：单端口无端口间分离问题。 |
| `srs.cyclic_shift_indices` | null | 参考：由当前builder推导。 |
| `srs.hopping.enabled` | false | 不增加频率/BWP hopping。 |
| `srs.hopping.frequency_offsets_prb` | [] | 参考：hopping关闭。 |
| `srs.hopping.bandwidth_num_prb` | [] | 参考：hopping关闭。 |
| `srs.ports.num_srs_ports` | 1 | 显式锁定单SRS端口，不依赖默认推导。 |
| `srs.ports.mapping` | one_to_one | 单端口映射到唯一TX阵元。 |
| `srs.ports.port_tx_ant_map` | null | 参考：按one_to_one推导。 |
| `srs.ports.usage` | non_codebook | 参考metadata，沿用原设置。 |
| `srs.power_control.enabled` | false | 禁用SRS专用功控。 |
| `srs.multiuser.enabled` | false | 各UE独立理想链路，不叠加同时发射干扰。 |

### `ranging`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `enabled` | false | 不计算接收机ToA/range估计。 |

### `noncooperative`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `enabled` | false | 不生成非合作shared IQ。 |

### `impairments`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `awgn.enabled` | false | 明确不添加噪声；PHY关闭也保证噪声链不执行。 |
| `cfo.enabled` | false | CFO关闭。 |
| `cfo.cfo_hz` | null | 无残留100Hz参数。 |
| `sfo.enabled` | false | SFO关闭。 |
| `sfo.sfo_ppm` | null | 无残留5ppm参数。 |
| `phase_noise.enabled` | false | 相偏关闭。 |
| `phase_noise.phase_offset_rad` | null | 无残留0.5rad参数。 |
| `timing_offset.enabled` | false | 定时偏移关闭。 |
| `timing_offset.timing_offset_samples` | null | 无残留2sample参数。 |
| `agc_adc.enabled` | false | 不模拟增益/削波/量化。 |
| `agc_adc.agc_gain_db` | 0 | 无额外增益；不设置实采58/66dB。 |
| `agc_adc.clipping_threshold` | null | 无削波阈值。 |
| `impairment_seed` | 142 | 参考：无损伤采样执行。 |

### `receiver`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `estimator_type` | srs_ls | 参考：不调用接收机。 |
| `channel_estimator` | srs_ls | 参考：不调用估计器。 |
| `mimo_detector` | none | 无检测器。 |
| `sync_method` | ideal | 参考metadata，不运行实际同步。 |
| `interpolation_method` | none | 理想CFR不经SRS插值。 |
| `failure_policy` | mark_invalid | 参考：RT错误由run/shard策略处理。 |
| `calibration_profile_id` | synthetic_default | 参考：不加载实采校准profile。 |

### `motion`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `enabled` | false | 每个label点是独立静态位置。 |
| `mobility_mode` | static | 不增加机器人运动序列。 |
| `num_time_steps` | 1 | 单快照，不额外增加时间轴。 |
| `sampling_frequency_hz` | 0 | 不采样Doppler时间过程；不是RF的采样率。 |
| `bs_velocity_mps` | [0,0,0] | BS静止。 |
| `ue_velocity_mps` | [0,0,0] | 各点静态计算。 |

### `calibration`

| 字段 | 固定值 | 审查结论 |
|---|---|---|
| `enabled` | false | 不运行校准，不读取实采offset。 |
| `profile_id` | synthetic_default | 参考：关闭状态。 |

## 审查发现与当前状态

1. **原朝向阻塞已解决。** 真实上行、BS顺序反转的UE子shard和下行结果均核对了HDF5
   中的实际设备朝向；零向量显式覆盖、缺省公共值、中心重合、UE子集和fallback子shard
   也有单元测试。不能用仅比较shape代替这些验收。
2. **339场景输入已物化。** 派生label及新仿真index已生成，并按原test名单顺序保留。
   run-scene-index的339场景dry-run用于确认生成配置全部指向派生label，原index的label
   路径覆盖问题已通过新index解决。正式运行时按实际分配覆盖GPU部署参数。
3. **已纠正无效遗留值。** UE变为1TX后，将通用端口数和SRS端口数设为1；参考
   fft_size从64设为768。它们在理想CFR模式不改变数值，但避免模板内部误导。
4. **只读元数据开关不是运行保证。** require_gpu/precision/torch_deterministic和
   run_id_format目前没有对应生产映射，部署要看实际环境和输出；本轮不扩展这些接口。
5. **接受的实验差异有明确记录。** 频点网格、iso/V、不同真实场景/材质和实采残差
   仍可能影响性能；shape相同不等于物理统计完全相同。这里不据此增加额外仿真变量。

## 准备和运行入口

```bash
# 已执行：只准备label/index，不进行RT，也不修改原数据
uv run python scripts/prepare_dichasus_like_test.py \
  --dataset-root /data/sunmeiyuan/projects/sionna/outputs/datasets/Front3d_2988_CFR_Only \
  --output-root outputs/front3d_339_dichasus_like_cfr_v1/prepared

# 已执行：339场景生成配置检查；没有提交GPU仿真任务
uv run python -m sionna_measurement_sim.app.cli \
  --config config/tasks/nr_srs_64prb_dichasus_like_cfr_v1.yaml \
  run-scene-index \
  --index-file outputs/front3d_339_dichasus_like_cfr_v1/prepared/test_scene_index.jsonl \
  --output-root outputs/front3d_339_dichasus_like_cfr_v1/dry_run \
  --dry-run
```

准备工具要求一个不存在的输出目录，避免重写已有输入。它读取归档的label/floorplan，
派生label只改顶层点的orientation并增加provenance；位置、ID、groups保持不变。
`prepared/summary.json` 记录名单、label/index校验和与统计；339场景共3,283个BS、
85,124个UE，原始934,128个UE–BS组合全部保留，逐点yaw检查最大误差为0。
新index的SHA256为 `332da55901356097b69def71e185db15e84312051c32c532595d9dc443332b0b`。

真实RT验收目录为 `outputs/front3d_339_dichasus_like_cfr_v1/smoke/`，使用物理GPU2、
`front3d_0015`、原depth=4及所有传播开关：

| Case | CFR shape | 非零链路 | 朝向/finite/schema |
|---|---|---:|---|
| uplink_full | [2,2,32,1,768] | 4/4 | 全通过 |
| uplink_reordered_shard | [1,2,32,1,768] | 2/2 | 全通过，BS顺序[1,0] |
| downlink | [2,1,1,32,768] | 2/2 | 全通过，BS朝向正确映射到TX |

三个case中BS yaw分别约为-0.919921和1.035801rad，pitch/roll均为0；UE角度为0。
结果无waveform或observation组。实际读回角度与预处理一致，finite比例均为100%。
验收只证明这组小场景的行为和现有contract，不代表339场景耗时/显存都已测量。

## 验证记录

- YAML经当前 `load_config()`/Pydantic加载，176个显式选项均为已声明字段，无被忽略的拼写错误。
- 逐项表覆盖全部显式选项，检查取值、门控和部署条件。
- 产品计划：compute/write CFR开启；PHY、CIR、path samples和其他数据产品关闭。
- 频率网格：768点、30kHz，端点为3,426,495,000/3,449,505,000Hz。
- 阵元间距反算为0.04364m；每链路shape为32×1×768，预期编码器输入为2×32×768。
- 单TX/单SRS端口资源映射做独立builder一致性检查；不执行接收机或生成受损CSI。
- 初次模板审查的配置/资源/产品测试为52 passed；朝向实现后另运行全量测试和ruff。
- 最终 `uv run ruff check sionna_measurement_sim tests scripts` 通过；
  `CUDA_VISIBLE_DEVICES=3 uv run pytest -q` 为 **378 passed、19 skipped**，76.40s。
  跳过项均为既有外部输出artifact缺失；另有既有KBestDetector参数裁剪warning。
- 339场景dry-run为 **339 planned、0 failed**，没有提交正式批量RT任务。
- 339派生label逐一核对原文件hash、点位/ID/groups、BS yaw/竖直姿态和固定UE方向，全部通过。
- 已生成上述三个真实RT HDF5并通过schema/shape/finite/实际朝向验收，数值摘要见smoke/summary.json。
- 正式339场景重跑尚未启动；没有把小规模结果记作完整测试集生产完成。
