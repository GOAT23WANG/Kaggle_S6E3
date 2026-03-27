# Logging Only Prompt

你当前的唯一任务是把已有实验结果整理为统一日志，不新增实验。

要求：

1. 按 experiment_log_schema.md 的字段补齐信息
2. 对于历史实验，无法精确恢复的字段必须显式标注 unknown，而不是猜测
3. 至少区分 completed、failed、stopped 三种状态
4. 如某历史方向已在 ideas.md 中被否决，应在 notes 中补充该上下文

目标：

为后续自动实验建立可靠的 parent baseline 与历史对照集