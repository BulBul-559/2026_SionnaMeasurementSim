# TODO Index

`docs/todo/` 是当前项目 active TODO 的唯一入口。这里的文档只按任务类型分类，
不再按实验批次或历史分支拆分；历史实验记录仍留在 `docs/performance/`，过时文档可移动到
`docs/legacy/` 供人工定期复核。

## 维护规则

- 使用 Codex 维护 TODO 时，优先加载项目级 skill：
  `.codex/skills/sionna-todo-docs/SKILL.md`。
- 新增 TODO 时先选择分类文档：`feature`、`structure`、`performance`、`bug`。
- 每个分类文档顶部都有按重要程度排序的简表；修改 TODO 时必须重新检查排序。
- 每个 TODO 至少说明目的、涉及模块、期望效果、粗粒度验收标准和重点提醒。
- 已完成 TODO 不留在 active 文档里，移动到 `history.md`，只保留完成日期和一句话描述。
- `docs/sys/` 继续描述当前系统事实；TODO 只链接到这里，不在 sys 文档中重复维护清单。

## 分类

| 文档 | 用途 |
|---|---|
| [feature](feature.md) | 新功能、标准完整性、算法增强和研究能力 |
| [structure](structure.md) | 数据契约、reader、benchmark 入口、legacy 模块和输出结构整理 |
| [performance](performance.md) | 大规模运行、写盘、RT、空间谱、GPU 调度和可视化开销 |
| [bug](bug.md) | 已确认缺陷、回归和需要修复的错误行为 |
| [history](history.md) | 已完成 TODO 的简洁归档 |

## 当前数量

| 分类 | Active TODO 数 |
|---|---:|
| feature | 12 |
| structure | 4 |
| performance | 8 |
| bug | 0 |

合计：24 个 active TODO。
