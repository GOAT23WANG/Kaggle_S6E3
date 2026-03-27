# Experiment Log Schema

该 schema 用于定义单个实验在日志中的最小记录格式。所有实验都必须能完整回答：改了什么、相对谁比较、结果如何、是否保留。

## 推荐存储形式

推荐同时保留两层：

1. 一份聚合表
- 建议文件：experiments/log.csv
- 用途：快速排序、筛选、比较历史实验

2. 一份单实验明细
- 建议文件：experiments/runs/EXP-XXX.json 或 experiments/runs/EXP-XXX.md
- 用途：记录更详细的配置、输出路径、异常信息

## 字段定义

| Field | Type | Required | Description |
|---|---|---|---|
| experiment_id | string | yes | 唯一实验编号，例如 EXP-P1-001 |
| parent_baseline | string | yes | 本次对比的父基线编号，例如 BASE-V16B-UNIFIED |
| stage | string | yes | 当前阶段，例如 baseline_consolidation / feature_batch / param_batch |
| priority | string | yes | 队列优先级，取值 P0 / P1 / P2 / P3 / P4 |
| change_type | string | yes | 改动类型，取值 feature_toggle / params / cv / threshold |
| change_detail | string | yes | 单变量改动说明，必须可归因 |
| hypothesis | string | yes | 本实验为何值得做的简短假设 |
| metric_name | string | yes | 主指标名称，固定为 roc_auc |
| cv_mean | float | yes | 当前实验的 OOF 或 CV 均值 |
| cv_std | float | yes | 当前实验的 fold 标准差 |
| best_before | float | yes | 实验开始前当前 best 分数 |
| delta_vs_best | float | yes | 相对 best_before 的差值 |
| runtime_minutes | float | yes | 实验总耗时，单位分钟 |
| status | string | yes | completed / failed / stopped |
| decision | string | yes | keep / drop / unstable / blocked |
| stop_rule_triggered | string | no | 若触发停止规则，填规则名称 |
| artifact_oof | string | no | OOF 文件路径 |
| artifact_submission | string | no | submission 文件路径 |
| artifact_model_dir | string | no | 模型文件目录 |
| artifact_analysis | string | no | 分析日志路径 |
| notes | string | no | 补充说明，包括异常与观察 |

## 填写规则

1. change_detail 必须具体，不能写“尝试优化参数”这类空泛描述。
2. hypothesis 必须先于实验存在，不能在结果出来后倒填。
3. delta_vs_best 统一按 `cv_mean - best_before` 计算。
4. decision 规则建议如下：
- keep：达到有效提升，且稳定性可接受
- unstable：分数略升但波动明显恶化
- drop：无效增益或退化
- blocked：因环境、数据、资源或代码错误未完成
5. failed 和 stopped 也必须记录，不能只保留成功实验。

## CSV 示例

```csv
experiment_id,parent_baseline,stage,priority,change_type,change_detail,hypothesis,metric_name,cv_mean,cv_std,best_before,delta_vs_best,runtime_minutes,status,decision,stop_rule_triggered,artifact_oof,artifact_submission,artifact_model_dir,artifact_analysis,notes
EXP-P1-001,BASE-V16B-UNIFIED,feature_batch,P1,feature_toggle,disable_distribution_features,distribution features may be redundant on unified baseline,roc_auc,0.91908,0.00181,0.91917,-0.00009,41.6,completed,drop,,,experiments/oof/EXP-P1-001.csv,experiments/submissions/EXP-P1-001.csv,models/EXP-P1-001,experiments/analysis/EXP-P1-001.txt,no material gain
EXP-P1-002,BASE-V16B-UNIFIED,feature_batch,P1,feature_toggle,disable_ngram_te,ngram TE may be carrying most categorical interaction signal,roc_auc,0.91841,0.00175,0.91917,-0.00076,39.4,completed,drop,hard_drop,,,models/EXP-P1-002,experiments/analysis/EXP-P1-002.txt,significant regression; direction downgraded
```

## 代理执行要求

1. 每完成一个实验立即追加日志。
2. 若实验异常中断，仍要写一条 status=failed 的记录。
3. batch summary 里的结论必须能追溯到这里的字段。