# Plan Folder Usage

这个目录用于存放给执行代理使用的计划材料，而不是实验产物本身。

## 文件结构

- [PlanV2.md](../PlanV2.md)
  完整版执行计划，适合作为主 prompt 或审核基线。
- [schemas/experiment_log_schema.md](schemas/experiment_log_schema.md)
  定义实验日志字段、类型、填写规则与示例。
- [templates/batch_summary_template.md](templates/batch_summary_template.md)
  定义每批实验结束后的统一汇报格式。
- [prompts/baseline_consolidation.prompt.md](prompts/baseline_consolidation.prompt.md)
  用于强基线重建阶段。
- [prompts/experiment_batch.prompt.md](prompts/experiment_batch.prompt.md)
  用于执行一批自动实验。
- [prompts/review_and_pause.prompt.md](prompts/review_and_pause.prompt.md)
  用于达到停止规则后的总结与暂停。
- [prompts/agent_quickstart.prompt.md](prompts/agent_quickstart.prompt.md)
  适合直接贴给代理的短版总入口。

## 使用顺序

1. 先阅读 [PlanV2.md](../PlanV2.md) 了解全局约束。
2. 再按阶段选择对应 prompt。
3. 运行中产生的实验结果应写入仓库的 experiments 目录，而不是写回 plan 目录。

## 设计目标

1. 给代理提供统一约束，减少自由发挥导致的跑偏。
2. 保证实验记录字段一致，便于跨批次比较。
3. 让“何时继续、何时暂停”有明确规则，而不是靠临时判断。