# PHY Module Development

本项目的 PHY 观测链路通过 registry 接入，目标是让 PUSCH、SRS-like、后续
WiFi-like 或 6G waveform 都复用同一条 RT、HDF5、schema、visualization pipeline。

## 接口

新增模块需要实现 `PHYModule.run(context)`：

```python
from sionna_measurement_sim.phy.modules import PHYContext, PHYModuleResult

class MyPHYModule:
    standard = "my_phy"

    def run(self, context: PHYContext) -> PHYModuleResult:
        ...
```

`context.config` 是 `RTTruthRunConfig`，`context.adapter_result` 包含 RT truth CFR、
CIR、path table 和 runtime versions。模块应返回领域模型，而不是 HDF5 writer 细节：

- `waveform`: `WaveformSpec`
- `observation`: `ObservationResult`
- `impairments`: `ImpairmentSpec`
- `receiver`: `ReceiverSpec`
- `evaluation`: `EvaluationResult`
- `waveform_extras`: 频域 grid、标准专属元数据
- `array_outputs`: 如模块能直接生成阵列输出，可放这里；否则由 pipeline 根据 grid/CFR 统一补齐

## 注册

在 `sionna_measurement_sim/phy/modules.py` 中加入实例：

```python
PHY_REGISTRY["my_phy"] = MyPHYModule()
```

同时更新：

- `sionna_measurement_sim/app/cli.py` 的 `--phy-standard` choices
- `sionna_measurement_sim/config/schema.py` 中相关配置校验
- `sionna_measurement_sim/io/hdf5_writer.py` 和 `schema_validator.py` 的新增输出字段
- `config/defaults/*.yaml` 模板和 `config/README.md`

## 输出约定

通用 CSI 必须写入 `/observation/cfr_est`，shape 为
`[snapshot, tx, rx, rx_ant, tx_ant, subcarrier]`。如果模块内部使用 uplink 视角，
写盘前必须转回项目约定的 DL view；现有 PUSCH/SRS-like 都遵守这一点。

时域 waveform 默认不保存。需要保存频域观测时，优先写入 `/waveform/*_grid`，
并为每个 dataset 设置 `unit` 和 `index_order`。

reverse/uplink 场景下，`/derived/*aoa*` 和 `/array/aoa_label_rad` 表示 PHY 接收侧
到达方向，因此使用原 RT 的 AoD。`/paths/nlos_truth` 中的 AoA/AoD 原始字段不改语义。

## 测试清单

新增模块至少需要：

1. registry 单测：能创建模块，未知 standard 报清晰错误。
2. 纯算法单测：无噪声或高 SNR 下估计结果接近 truth CFR。
3. schema 测试：新增 HDF5 字段存在、shape 和 attrs 正确。
4. 小规模 pipeline 集成：旧 custom OFDM/PUSCH 不受影响，新模块能写 manifest 和 schema。
5. 文档：说明模块是不是标准实现，哪些物理细节仍是 TODO。
