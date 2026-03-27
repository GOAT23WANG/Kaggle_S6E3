import csv
import json
import os
from datetime import datetime


EXPERIMENT_COLUMNS = [
    "experiment_id",
    "parent_baseline",
    "stage",
    "priority",
    "change_type",
    "change_detail",
    "hypothesis",
    "metric_name",
    "cv_mean",
    "cv_std",
    "best_before",
    "delta_vs_best",
    "runtime_minutes",
    "status",
    "decision",
    "stop_rule_triggered",
    "artifact_oof",
    "artifact_submission",
    "artifact_model_dir",
    "artifact_analysis",
    "notes",
]


def ensure_experiment_layout(experiments_dir):
    subdirs = [
        experiments_dir,
        os.path.join(experiments_dir, "configs"),
        os.path.join(experiments_dir, "runs"),
        os.path.join(experiments_dir, "oof"),
        os.path.join(experiments_dir, "submissions"),
        os.path.join(experiments_dir, "analysis"),
        os.path.join(experiments_dir, "batch_summaries"),
    ]
    for path in subdirs:
        os.makedirs(path, exist_ok=True)

    log_path = os.path.join(experiments_dir, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_COLUMNS)
            writer.writeheader()

    return log_path


def append_experiment_log(log_path, record):
    row = {column: record.get(column, "") for column in EXPERIMENT_COLUMNS}
    with open(log_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_COLUMNS)
        writer.writerow(row)


def write_run_record(runs_dir, record):
    os.makedirs(runs_dir, exist_ok=True)
    run_path = os.path.join(runs_dir, f"{record['experiment_id']}.json")
    payload = dict(record)
    payload["written_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(run_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return run_path


def write_batch_summary(summary_path, summary):
    retained = summary.get("retained_directions") or ["none"]
    dropped = summary.get("dropped_directions") or ["none"]
    next_queue = summary.get("recommended_next_queue") or ["none"]

    lines = [
        f"## Batch Summary: {summary['batch_id']}",
        "",
        "### 1. Batch Context",
        f"- stage: {summary['stage']}",
        f"- priority_range: {summary['priority_range']}",
        f"- parent_baseline: {summary['parent_baseline']}",
        f"- batch_start_time: {summary['batch_start_time']}",
        f"- batch_end_time: {summary['batch_end_time']}",
        f"- total_runtime_hours: {summary['total_runtime_hours']}",
        f"- experiments_completed: {summary['experiments_completed']}",
        f"- experiments_failed: {summary['experiments_failed']}",
        "",
        "### 2. Best Result",
        f"- current_best_experiment: {summary['current_best_experiment']}",
        f"- best_cv_auc: {summary['best_cv_auc']}",
        f"- delta_vs_parent_baseline: {summary['delta_vs_parent_baseline']}",
        f"- stability_note: {summary['stability_note']}",
        "",
        "### 3. Experiment Table",
        "| experiment_id | change_detail | cv_auc | delta_vs_best_before | runtime_min | decision | note |",
        "|---|---|---:|---:|---:|---|---|",
        (
            f"| {summary['current_best_experiment']} | {summary['change_detail']} | {summary['best_cv_auc']} | "
            f"{summary['delta_vs_parent_baseline']} | {summary['runtime_minutes']} | {summary['decision']} | {summary['table_note']} |"
        ),
        "",
        "### 4. Retained Directions",
    ]

    lines.extend([f"- {item}" for item in retained])
    lines.extend([
        "",
        "### 5. Dropped Directions",
    ])
    lines.extend([f"- {item}" for item in dropped])
    lines.extend([
        "",
        "### 6. Stop Rules",
        f"- triggered_rule: {summary['triggered_rule']}",
        f"- trigger_reason: {summary['trigger_reason']}",
        f"- was_pause_expected: {summary['was_pause_expected']}",
        "",
        "### 7. Interpretation",
        f"- signal_summary: {summary['signal_summary']}",
        f"- risk_summary: {summary['risk_summary']}",
        f"- compute_summary: {summary['compute_summary']}",
        "",
        "### 8. Recommended Next Queue",
    ])
    lines.extend([f"1. {item}" for item in next_queue])
    lines.extend([
        "",
        "### 9. User Decision Needed",
        f"- continue_current_track: {summary['continue_current_track']}",
        f"- open_high_cost_track: {summary['open_high_cost_track']}",
        f"- require_code_cleanup_before_next_batch: {summary['require_code_cleanup_before_next_batch']}",
    ])

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
