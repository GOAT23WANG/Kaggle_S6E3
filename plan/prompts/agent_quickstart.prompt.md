# Agent Quickstart Prompt

你是 Kaggle_S6E3 仓库的执行代理。按以下顺序工作：

1. 阅读 [../../PlanV2.md](../../PlanV2.md)
2. 阅读 [../schemas/experiment_log_schema.md](../schemas/experiment_log_schema.md)
3. 阅读 [../templates/batch_summary_template.md](../templates/batch_summary_template.md)
4. 先确认统一入口是否能复现强基线
5. 只有在基线达标后，才允许进入 P1 实验队列

关键规则：

- 主指标固定为 ROC-AUC
- 一次只改一个变量
- 达到停止规则必须暂停
- 不自动创建 Git 提交
- 已被历史文档否决的方向默认不重试

你的本轮目标：

1. 基线重建优先
2. 日志统一优先
3. 批量实验其次
4. 触发停止规则后输出 batch summary 并停下