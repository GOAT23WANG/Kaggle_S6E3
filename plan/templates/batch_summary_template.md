# Batch Summary Template

每一批自动实验结束后，执行代理必须按以下结构输出总结。该模板的目标是让用户快速判断：这批实验有没有价值、为什么暂停、下一步该不该继续。

```md
## Batch Summary: {batch_id}

### 1. Batch Context
- stage: {stage}
- priority_range: {priority_range}
- parent_baseline: {parent_baseline}
- batch_start_time: {batch_start_time}
- batch_end_time: {batch_end_time}
- total_runtime_hours: {total_runtime_hours}
- experiments_completed: {experiments_completed}
- experiments_failed: {experiments_failed}

### 2. Best Result
- current_best_experiment: {current_best_experiment}
- best_cv_auc: {best_cv_auc}
- delta_vs_parent_baseline: {delta_vs_parent_baseline}
- stability_note: {stability_note}

### 3. Experiment Table
| experiment_id | change_detail | cv_auc | delta_vs_best_before | runtime_min | decision | note |
|---|---|---:|---:|---:|---|---|
| {exp_1} | {change_1} | {auc_1} | {delta_1} | {runtime_1} | {decision_1} | {note_1} |
| {exp_2} | {change_2} | {auc_2} | {delta_2} | {runtime_2} | {decision_2} | {note_2} |

### 4. Retained Directions
- {retained_direction_1}
- {retained_direction_2}

### 5. Dropped Directions
- {dropped_direction_1}
- {dropped_direction_2}

### 6. Stop Rules
- triggered_rule: {triggered_rule}
- trigger_reason: {trigger_reason}
- was_pause_expected: {was_pause_expected}

### 7. Interpretation
- signal_summary: {signal_summary}
- risk_summary: {risk_summary}
- compute_summary: {compute_summary}

### 8. Recommended Next Queue
1. {next_experiment_1}
2. {next_experiment_2}
3. {next_experiment_3}

### 9. User Decision Needed
- continue_current_track: {yes_or_no}
- open_high_cost_track: {yes_or_no}
- require_code_cleanup_before_next_batch: {yes_or_no}
```

## 填写要求

1. Stop Rules 部分必须明确指出触发了哪一条规则，不能写成“效果一般，因此暂停”。
2. Interpretation 必须区分“有信号但不稳定”和“完全无信号”。
3. Recommended Next Queue 只能列未被黑名单否决、且仍满足单变量约束的候选。
4. 如果本批没有任何有效提升，也必须明确写出“当前 best 未改变”。