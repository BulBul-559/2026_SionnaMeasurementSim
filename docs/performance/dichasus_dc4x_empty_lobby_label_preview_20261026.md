# DICHASUS DC4x 空大厅 0.2m label 预览

日期：2026-10-26。状态：label和预览已生成，未启动CFR仿真。

用户要求使用 `empty_lobby` 和[已冻结DICHASUS-like v1配置](../proposals/dichasus_like_cfr_v1.md)，
先准备大厅0.2m采样label并展示平面图叠加效果。

## 输入与区域

场景目录：`/data/sunmeiyuan/projects/DICHASUS/dcxx/floorplan/blend_0000(dc4x)`。
源XML、OBJ、平面图及meta只读；坐标保持原导出后的scene-local XY，未再次平移。
原模型中还包含三条长走廊和厨房，本轮只取中央大厅加宽区，以建筑转角和走廊开口为边界。
精确ROI多边形保存在label的 `sampling.roi_polygon_xy_m`，面积约40.251923㎡。
这是一版几何预览范围，不宣称与某一条实采轨迹的覆盖域完全一致。

UE在全局XY轴上按0.2m格点采样，z=1.6m。先要求投影落在真实floor三角形内，
再要求离ROI边界至少0.2m，且距离RT网格与z=1.6m平面的相交墙/柱线段至少0.2m。
不按平面图灰度猜测通行区域。最终880个唯一UE点，XY范围为[6.8,22.4]到[13.6,29.8]m；
最小墙体距离0.200329m，最小ROI边界距离0.201169m。

BS取模型天线patch对象包围盒中心，未改成随机位置或2.4m固定高度：

| BS | 模型对象 | scene XYZ（米） |
|---|---|---|
| BS0 | antenna-array-north-patches | [15.68559, 2.2991, 1.39335] |
| BS1 | antenna-array-south-patches | [1.8145, 47.7216, 1.4197] |

BS保持竖直并朝整张平面图中心[18.0157051,25.0826511]m，UE固定零朝向。
UE高度沿用此前仿真假设，BS来自模型网格，不等同于硬件校准参考点。

## 输出和验证

输出目录：

```text
/data/sunmeiyuan/projects/sionna/outputs/dichasus_dc4x_empty_lobby_panel0p2_v1/
```

- `label/label_panel_0p2.json`：标准顶层bs_points/ue_points、分组、采样区域与朝向。
- `label/ue_points.csv`：880点XYZ与编号。
- `floorplan/`：原平面图和meta副本。
- `figures/lobby_sampling_overlay.png`：全场景BS/ROI位置和大厅点位局部并排图。
- `figures/lobby_sampling_detail.png`：大厅局部大图。
- `simulation_config.yaml`：冻结模板仅覆盖scene_id、label/scene路径及新输出路径；
  3.438GHz、4×8 RX/单TX、768点/30kHz、理想CFR及全部传播开关均保持不变。
- `summary.json`/`validation.json`：区域、间距、点数、配置与源文件hash检查。
- `prepare_label.py`：可复现脚本，不覆盖已有label；`--render-only`只重画预览。

880个UE和2个BS均通过项目原生label parser，配置caps不会截断；坐标有限、点位唯一，
格点间隔、地面支持、ROI和墙体净空均检查通过。原始平面图/meta副本hash一致，
六个源输入（含冻结模板）hash保持不变；脚本ruff通过。未修改生产代码、schema或默认模板，
未为880×2=1,760条候选链路运行RT。README/config/sys接口及TODO无需变更。
