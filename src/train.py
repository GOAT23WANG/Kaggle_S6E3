import json
import os
import re
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder as SKTargetEncoder

from experiment_tracking import append_experiment_log, ensure_experiment_layout, write_batch_summary, write_run_record
from features import ID_COL, TARGET_COL, bi_tri_target_encoding, build_unified_v16_features, build_ridge_predictions


def get_next_run_number(folder, prefix, suffix):
    os.makedirs(folder, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(suffix)}$")
    max_num = 0
    for file_name in os.listdir(folder):
        match = pattern.match(file_name)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def parse_bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int_env(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def parse_float_env(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def get_env_value(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return value
    return default


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def infer_stage(priority, explicit_stage=None):
    if explicit_stage:
        return explicit_stage
    if priority == "P2":
        return "parameter_search"
    if priority == "P1":
        return "feature_batch"
    return "baseline_consolidation"


def resolve_best_before(parent_baseline):
    explicit_best = get_env_value("BEST_BEFORE")
    if explicit_best is not None:
        return float(explicit_best)

    run_record_path = os.path.join(EXPERIMENT_RUNS_DIR, f"{parent_baseline}.json")
    if os.path.exists(run_record_path):
        record = load_json(run_record_path)
        if record and "cv_mean" in record:
            return float(record["cv_mean"])

    if os.path.exists(EXPERIMENT_LOG_PATH):
        log_df = pd.read_csv(EXPERIMENT_LOG_PATH)
        matched = log_df.loc[log_df["experiment_id"] == parent_baseline, "cv_mean"]
        if not matched.empty:
            return float(matched.iloc[-1])

    registry = load_json(BASELINE_REGISTRY_PATH)
    active_baseline_id = registry.get("active_baseline_id") if registry else None
    if active_baseline_id == parent_baseline:
        matched = pd.read_csv(EXPERIMENT_LOG_PATH)
        active_row = matched.loc[matched["experiment_id"] == active_baseline_id, "cv_mean"]
        if not active_row.empty:
            return float(active_row.iloc[-1])

    return 0.0


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BASE_SAVE_DIR = os.path.join(PROJECT_ROOT, "Previously Trained Files")
OOF_DIR = os.path.join(BASE_SAVE_DIR, "oof_self")
SUB_DIR = os.path.join(BASE_SAVE_DIR, "sub_self")
ANALYSIS_DIR = os.path.join(BASE_SAVE_DIR, "whole analysis")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
CONFIGS_DIR = os.path.join(EXPERIMENTS_DIR, "configs")
BASELINE_REGISTRY_PATH = os.path.join(CONFIGS_DIR, "baseline_registry.json")
P1_QUEUE_PATH = os.path.join(CONFIGS_DIR, "p1_queue.json")
ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

EXPERIMENT_LOG_PATH = ensure_experiment_layout(EXPERIMENTS_DIR)
EXPERIMENT_OOF_DIR = os.path.join(EXPERIMENTS_DIR, "oof")
EXPERIMENT_SUB_DIR = os.path.join(EXPERIMENTS_DIR, "submissions")
EXPERIMENT_ANALYSIS_DIR = os.path.join(EXPERIMENTS_DIR, "analysis")
EXPERIMENT_RUNS_DIR = os.path.join(EXPERIMENTS_DIR, "runs")
EXPERIMENT_BATCH_DIR = os.path.join(EXPERIMENTS_DIR, "batch_summaries")

os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RUN_PROFILE = get_env_value("RUN_PROFILE", default="simple")
RUN_PRIORITY = get_env_value("RUN_PRIORITY", "EXPERIMENT_PRIORITY", default="P0")
RUN_STAGE = infer_stage(RUN_PRIORITY, get_env_value("RUN_STAGE", "EXPERIMENT_STAGE"))
RUN_CHANGE_TYPE = get_env_value("RUN_CHANGE_TYPE", "CHANGE_TYPE", default="baseline")
RUN_CHANGE_DETAIL = get_env_value("RUN_CHANGE_DETAIL", "CHANGE_DETAIL", default=f"{RUN_PROFILE}_run")
RUN_HYPOTHESIS = get_env_value("RUN_HYPOTHESIS", "HYPOTHESIS", default="Baseline migration run.")
PARENT_BASELINE = os.environ.get("PARENT_BASELINE", "BASE-SIMPLE-5FOLD-TE-XGB")
RUN_START_TIME = time.time()

run_num = get_next_run_number(SUB_DIR, "submission", ".csv")
experiment_id = os.environ.get("EXPERIMENT_ID", f"EXP-{RUN_PRIORITY}-{run_num:03d}")
batch_id = os.environ.get("BATCH_ID", f"BATCH-{RUN_PRIORITY}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
BEST_BEFORE = resolve_best_before(PARENT_BASELINE)


def load_train_test():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    train[TARGET_COL] = train[TARGET_COL].astype(str).str.strip().map({"No": 0, "Yes": 1}).astype(int)
    return train, test


def build_analysis_header(train_df, test_df):
    return [
        f"Run Number: {run_num}",
        f"Experiment ID: {experiment_id}",
        f"Run Profile: {RUN_PROFILE}",
        f"Parent Baseline: {PARENT_BASELINE}",
        f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"train shape: {train_df.shape}",
        f"test shape: {test_df.shape}",
        f"train columns: {train_df.columns.tolist()}",
        f"test columns: {test_df.columns.tolist()}",
        "",
    ]


def run_simple_baseline(train_df, test_df, analysis_lines):
    feature_cols = [column for column in train_df.columns if column not in [TARGET_COL, ID_COL]]
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "seed": 42,
    }
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_aucs = []

    analysis_lines.append("Model Params:")
    for key, value in params.items():
        analysis_lines.append(f"{key}: {value}")
    analysis_lines.append("")

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df[TARGET_COL]), start=1):
        print(f"\n========== Fold {fold} ==========")
        analysis_lines.append(f"========== Fold {fold} ==========")
        x_train = train_df.iloc[train_idx].copy()
        x_val = train_df.iloc[val_idx].copy()
        x_test = test_df.copy()
        y_train = x_train[TARGET_COL]
        y_val = x_val[TARGET_COL]

        x_train_enc, x_val_enc, x_test_enc = bi_tri_target_encoding(x_train, x_val, x_test, feature_cols, target_col=TARGET_COL)
        x_train_enc = x_train_enc.fillna(-1)
        x_val_enc = x_val_enc.fillna(-1)
        x_test_enc = x_test_enc.fillna(-1)

        train_features = [column for column in x_train_enc.columns if column not in [TARGET_COL, ID_COL]]
        dtrain = xgb.DMatrix(x_train_enc[train_features], label=y_train)
        dval = xgb.DMatrix(x_val_enc[train_features], label=y_val)
        dtest = xgb.DMatrix(x_test_enc[train_features])

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=100,
            verbose_eval=100,
        )

        model_path = os.path.join(MODELS_DIR, f"xgb_fold{fold}_run{run_num}.json")
        model.save_model(model_path)
        val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

        oof_preds[val_idx] = val_pred
        test_preds += test_pred / kf.n_splits
        fold_auc = float(roc_auc_score(y_val, val_pred))
        fold_aucs.append(fold_auc)

        print(f"Fold {fold} AUC: {fold_auc:.6f}")
        analysis_lines.append(f"Fold {fold} AUC: {fold_auc:.6f}")
        analysis_lines.append(f"Best iteration: {model.best_iteration}")
        analysis_lines.append(f"Saved model: {model_path}")
        analysis_lines.append("")

    return oof_preds, test_preds, fold_aucs, {"profile": "simple", "n_splits": 5}


def run_unified_v16(train_df, test_df, analysis_lines):
    if not os.path.exists(ORIGINAL_DATA_PATH):
        raise FileNotFoundError(f"Missing original IBM dataset: {ORIGINAL_DATA_PATH}")

    orig = pd.read_csv(ORIGINAL_DATA_PATH)
    feature_flags = {
        "orig_proba": not parse_bool_env("DISABLE_ORIG_PROBA", False),
        "distribution": not parse_bool_env("DISABLE_DISTRIBUTION", False),
        "digit": not parse_bool_env("DISABLE_DIGIT", False),
        "ngram": not parse_bool_env("DISABLE_NGRAM", False),
    }
    outer_folds = parse_int_env("N_FOLDS", parse_int_env("OUTER_FOLDS", 10))
    inner_folds = parse_int_env("INNER_FOLDS", 5)

    train_features, test_features, metadata = build_unified_v16_features(train_df, test_df, orig, feature_flags=feature_flags)
    features = metadata["features"]
    te_columns = metadata["te_columns"]
    te_ngram_columns = metadata["te_ngram_columns"]
    drop_columns = metadata["drop_columns"]

    params = {
        "n_estimators": parse_int_env("XGB_N_ESTIMATORS", 50000),
        "learning_rate": parse_float_env("XGB_LEARNING_RATE", 0.0063),
        "max_depth": parse_int_env("XGB_MAX_DEPTH", 5),
        "subsample": parse_float_env("XGB_SUBSAMPLE", 0.81),
        "colsample_bytree": parse_float_env("XGB_COLSAMPLE_BYTREE", 0.32),
        "min_child_weight": parse_float_env("XGB_MIN_CHILD_WEIGHT", 6),
        "reg_alpha": parse_float_env("XGB_REG_ALPHA", 3.5017),
        "reg_lambda": parse_float_env("XGB_REG_LAMBDA", 1.2925),
        "gamma": parse_float_env("XGB_GAMMA", 0.790),
        "random_state": 42,
        "early_stopping_rounds": parse_int_env("XGB_EARLY_STOPPING_ROUNDS", 500),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "enable_categorical": False,
        "device": os.environ.get("XGB_DEVICE", "cpu"),
        "tree_method": os.environ.get("XGB_TREE_METHOD", "hist"),
        "verbosity": 0,
    }

    enable_ridge = parse_bool_env("ENABLE_RIDGE_FEATURE", False)
    ridge_inner_folds = parse_int_env("RIDGE_INNER_FOLDS", 5)
    multi_seed = parse_bool_env("MULTI_SEED", False)
    n_seeds = parse_int_env("N_SEEDS", 5)
    seed_list = [42, 123, 456, 789, 2026, 3141, 5926, 5358, 9793, 2384][:n_seeds]

    analysis_lines.append(f"Feature flags: {feature_flags}")
    analysis_lines.append(f"Ridge feature: {enable_ridge}")
    analysis_lines.append(f"Multi-seed: {multi_seed} (n_seeds={n_seeds})")
    analysis_lines.append(f"Unified feature count before encoding: {len(features)}")
    analysis_lines.append("Model Params:")
    for key, value in params.items():
        analysis_lines.append(f"{key}: {value}")
    analysis_lines.append("")

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_features))
    test_preds = np.zeros(len(test_features))
    fold_aucs = []
    te_stats = ["std", "min", "max"]

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(train_features, train_features[TARGET_COL]), start=1):
        print(f"\n--- Fold {fold}/{outer_folds} ---")
        analysis_lines.append(f"--- Fold {fold}/{outer_folds} ---")

        x_tr = train_features.loc[train_idx, features + [TARGET_COL]].reset_index(drop=True).copy()
        y_tr = train_features.loc[train_idx, TARGET_COL].values
        x_val = train_features.loc[val_idx, features].reset_index(drop=True).copy()
        y_val = train_features.loc[val_idx, TARGET_COL].values
        x_te = test_features[features].reset_index(drop=True).copy()

        for in_tr, in_va in inner_cv.split(x_tr, y_tr):
            x_tr_inner = x_tr.loc[in_tr, features + [TARGET_COL]].copy()
            x_va_inner = x_tr.loc[in_va, features].copy()
            for column in te_columns:
                stats_df = x_tr_inner.groupby(column, observed=False)[TARGET_COL].agg(te_stats)
                stats_df.columns = [f"TE1_{column}_{stat}" for stat in te_stats]
                x_va_inner = x_va_inner.merge(stats_df, on=column, how="left")
                for encoded_col in stats_df.columns:
                    x_tr.loc[in_va, encoded_col] = x_va_inner[encoded_col].values.astype("float32")

        for column in te_columns:
            stats_df = x_tr.groupby(column, observed=False)[TARGET_COL].agg(te_stats)
            stats_df.columns = [f"TE1_{column}_{stat}" for stat in te_stats]
            stats_df = stats_df.astype("float32")
            x_val = x_val.merge(stats_df, on=column, how="left")
            x_te = x_te.merge(stats_df, on=column, how="left")
            for encoded_col in stats_df.columns:
                for frame in [x_tr, x_val, x_te]:
                    frame[encoded_col] = frame[encoded_col].fillna(0)

        for in_tr, in_va in inner_cv.split(x_tr, y_tr):
            x_tr_inner = x_tr.loc[in_tr].copy()
            x_va_inner = x_tr.loc[in_va].copy()
            for column in te_ngram_columns:
                ngram_target = x_tr_inner.groupby(column, observed=False)[TARGET_COL].mean()
                feature_name = f"TE_ng_{column}"
                mapped = x_va_inner[column].astype(str).map(ngram_target)
                x_tr.loc[in_va, feature_name] = pd.to_numeric(mapped, errors="coerce").fillna(0.5).astype("float32").values

        for column in te_ngram_columns:
            ngram_target = x_tr.groupby(column, observed=False)[TARGET_COL].mean()
            feature_name = f"TE_ng_{column}"
            x_val[feature_name] = pd.to_numeric(x_val[column].astype(str).map(ngram_target), errors="coerce").fillna(0.5).astype("float32")
            x_te[feature_name] = pd.to_numeric(x_te[column].astype(str).map(ngram_target), errors="coerce").fillna(0.5).astype("float32")
            if feature_name in x_tr.columns:
                x_tr[feature_name] = pd.to_numeric(x_tr[feature_name], errors="coerce").fillna(0.5).astype("float32")
            else:
                x_tr[feature_name] = 0.5

        te_mean_cols = [f"TE_{column}" for column in te_columns]
        mean_encoder = SKTargetEncoder(cv=inner_folds, shuffle=True, smooth="auto", target_type="binary", random_state=42)
        x_tr[te_mean_cols] = mean_encoder.fit_transform(x_tr[te_columns], y_tr)
        x_val[te_mean_cols] = mean_encoder.transform(x_val[te_columns])
        x_te[te_mean_cols] = mean_encoder.transform(x_te[te_columns])

        for frame in [x_tr, x_val, x_te]:
            frame.drop(columns=[column for column in drop_columns if column in frame.columns], inplace=True, errors="ignore")

        x_tr.drop(columns=[TARGET_COL], inplace=True, errors="ignore")
        x_tr = x_tr.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
        x_val = x_val.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
        x_te = x_te.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")

        if enable_ridge:
            ridge_tr, ridge_va, ridge_te = build_ridge_predictions(
                x_tr, y_tr, x_val, x_te, inner_folds=ridge_inner_folds
            )
            x_tr["ridge_pred"] = ridge_tr
            x_val["ridge_pred"] = ridge_va
            x_te["ridge_pred"] = ridge_te

        if fold == 1:
            analysis_lines.append(f"Unified feature count for XGBoost: {len(x_tr.columns)}")

        if multi_seed:
            val_pred_accum = np.zeros(len(x_val))
            test_pred_accum = np.zeros(len(x_te))
            for seed_idx, seed_val in enumerate(seed_list):
                seed_params = {**params, "random_state": seed_val}
                model = xgb.XGBClassifier(**seed_params)
                model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=0)
                val_pred_accum += model.predict_proba(x_val)[:, 1]
                test_pred_accum += model.predict_proba(x_te)[:, 1]
                if fold == 1 and seed_idx == 0:
                    model_path = os.path.join(MODELS_DIR, f"xgb_unified_v16_fold{fold}_run{run_num}.json")
                    model.save_model(model_path)
            val_pred = val_pred_accum / n_seeds
            test_pred = test_pred_accum / n_seeds
            analysis_lines.append(f"Multi-seed: averaged {n_seeds} seeds")
        else:
            model = xgb.XGBClassifier(**params)
            model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=1000)
            model_path = os.path.join(MODELS_DIR, f"xgb_unified_v16_fold{fold}_run{run_num}.json")
            model.save_model(model_path)
            val_pred = model.predict_proba(x_val)[:, 1]
            test_pred = model.predict_proba(x_te)[:, 1]

        oof_preds[val_idx] = val_pred
        test_preds += test_pred / outer_folds

        fold_auc = float(roc_auc_score(y_val, val_pred))
        fold_aucs.append(fold_auc)
        print(f"Fold {fold} AUC: {fold_auc:.6f}")
        analysis_lines.append(f"Fold {fold} AUC: {fold_auc:.6f}")
        analysis_lines.append("")

    profile_label = "unified_v16"
    if enable_ridge:
        profile_label += "+ridge"
    if multi_seed:
        profile_label += f"+{n_seeds}seed"
    return oof_preds, test_preds, fold_aucs, {"profile": profile_label, "n_splits": outer_folds}


def update_baseline_registry_if_needed(record):
    registry = load_json(BASELINE_REGISTRY_PATH)
    if not registry:
        return
    if RUN_PROFILE == "unified_v16" and record["decision"] == "keep" and (
        record["best_before"] == 0 or record["delta_vs_best"] >= 0.0002
    ):
        registry["active_baseline_id"] = experiment_id
        registry["active_baseline_description"] = "Unified V16-style strong baseline executed through src/train.py."
        registry["status"] = "unified-baseline-ready"
        write_json(BASELINE_REGISTRY_PATH, registry)

        queue = load_json(P1_QUEUE_PATH)
        if queue:
            queue["depends_on_baseline_id"] = experiment_id
            queue["status"] = "ready"
            write_json(P1_QUEUE_PATH, queue)


def main():
    train_df, test_df = load_train_test()
    analysis_lines = build_analysis_header(train_df, test_df)

    if RUN_PROFILE == "unified_v16":
        oof_preds, test_preds, fold_aucs, run_meta = run_unified_v16(train_df, test_df, analysis_lines)
    else:
        oof_preds, test_preds, fold_aucs, run_meta = run_simple_baseline(train_df, test_df, analysis_lines)

    overall_auc = float(roc_auc_score(train_df[TARGET_COL], oof_preds))
    print(f"\nOverall CV AUC: {overall_auc:.6f}")
    analysis_lines.append(f"Overall CV AUC: {overall_auc:.6f}")
    analysis_lines.append("")

    oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET_COL: train_df[TARGET_COL], "oof_pred": oof_preds})
    sub_df = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_COL: test_preds})

    oof_path = os.path.join(OOF_DIR, f"oof_predictions_{run_num}.csv")
    sub_path = os.path.join(SUB_DIR, f"submission_{run_num}.csv")
    analysis_path = os.path.join(ANALYSIS_DIR, f"analysis_{run_num}.txt")
    exp_oof_path = os.path.join(EXPERIMENT_OOF_DIR, f"{experiment_id}.csv")
    exp_sub_path = os.path.join(EXPERIMENT_SUB_DIR, f"{experiment_id}.csv")
    exp_analysis_path = os.path.join(EXPERIMENT_ANALYSIS_DIR, f"{experiment_id}.txt")
    batch_summary_path = os.path.join(EXPERIMENT_BATCH_DIR, f"{batch_id}.md")

    oof_df.to_csv(oof_path, index=False)
    sub_df.to_csv(sub_path, index=False)
    oof_df.to_csv(exp_oof_path, index=False)
    sub_df.to_csv(exp_sub_path, index=False)

    analysis_lines.append(f"OOF file: {oof_path}")
    analysis_lines.append(f"Submission file: {sub_path}")
    analysis_lines.append(f"Analysis file: {analysis_path}")
    analysis_lines.append(f"Experiment OOF file: {exp_oof_path}")
    analysis_lines.append(f"Experiment Submission file: {exp_sub_path}")
    analysis_lines.append(f"Experiment Analysis file: {exp_analysis_path}")

    with open(analysis_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(map(str, analysis_lines)))
    shutil.copyfile(analysis_path, exp_analysis_path)

    runtime_minutes = round((time.time() - RUN_START_TIME) / 60, 2)
    cv_std = float(np.std(fold_aucs)) if fold_aucs else 0.0
    delta_vs_best = float(overall_auc - BEST_BEFORE)

    if BEST_BEFORE == 0:
        decision = "keep"
        stop_rule_triggered = ""
        table_note = "bootstrap baseline created"
    elif delta_vs_best >= 0.0002:
        decision = "keep"
        stop_rule_triggered = ""
        table_note = "effective gain"
    elif delta_vs_best <= -0.0010:
        decision = "drop"
        stop_rule_triggered = "hard_drop"
        table_note = "significant regression"
    else:
        decision = "drop"
        stop_rule_triggered = "min_gain_not_met"
        table_note = "no material gain"

    record = {
        "experiment_id": experiment_id,
        "parent_baseline": PARENT_BASELINE,
        "stage": RUN_STAGE,
        "priority": RUN_PRIORITY,
        "change_type": RUN_CHANGE_TYPE,
        "change_detail": RUN_CHANGE_DETAIL,
        "hypothesis": RUN_HYPOTHESIS,
        "metric_name": "roc_auc",
        "cv_mean": round(overall_auc, 6),
        "cv_std": round(cv_std, 6),
        "best_before": round(BEST_BEFORE, 6),
        "delta_vs_best": round(delta_vs_best, 6),
        "runtime_minutes": runtime_minutes,
        "status": "completed",
        "decision": decision,
        "stop_rule_triggered": stop_rule_triggered,
        "artifact_oof": exp_oof_path,
        "artifact_submission": exp_sub_path,
        "artifact_model_dir": MODELS_DIR,
        "artifact_analysis": exp_analysis_path,
        "notes": f"profile={run_meta['profile']};fold_aucs={','.join(f'{score:.6f}' for score in fold_aucs)}",
    }

    append_experiment_log(EXPERIMENT_LOG_PATH, record)
    write_run_record(EXPERIMENT_RUNS_DIR, record)
    update_baseline_registry_if_needed(record)

    recommended_next_queue = [
        "Run unified_v16 baseline consolidation",
        "Validate BASE-V16B-UNIFIED against historical 0.919xx OOF",
        "Unlock P1 queue after unified baseline is stable",
    ]
    if RUN_PRIORITY == "P2" or RUN_CHANGE_TYPE == "params":
        recommended_next_queue = [
            "Run the next local parameter variant from p2_queue.json",
            "Stop the current P2 lane after three consecutive no-gain results",
            "Keep EXP-P0-003 active unless a P2 run beats it by at least 0.0002 ROC-AUC",
        ]
    elif RUN_PROFILE == "unified_v16":
        recommended_next_queue = [
            "Toggle digit features off to measure marginal contribution",
            "Toggle n-gram target encoding off to measure categorical interaction value",
            "Compare 5-fold vs 10-fold runtime and AUC under unified baseline",
        ]

    summary = {
        "batch_id": batch_id,
        "stage": RUN_STAGE,
        "priority_range": RUN_PRIORITY,
        "parent_baseline": PARENT_BASELINE,
        "batch_start_time": datetime.fromtimestamp(RUN_START_TIME).strftime("%Y-%m-%d %H:%M:%S"),
        "batch_end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_hours": round(runtime_minutes / 60, 2),
        "experiments_completed": 1,
        "experiments_failed": 0,
        "current_best_experiment": experiment_id,
        "best_cv_auc": round(overall_auc, 6),
        "delta_vs_parent_baseline": round(delta_vs_best, 6),
        "stability_note": f"fold_std={cv_std:.6f}",
        "change_detail": RUN_CHANGE_DETAIL,
        "runtime_minutes": runtime_minutes,
        "decision": decision,
        "table_note": table_note,
        "retained_directions": [RUN_CHANGE_DETAIL] if decision == "keep" else [],
        "dropped_directions": [RUN_CHANGE_DETAIL] if decision != "keep" else [],
        "triggered_rule": stop_rule_triggered or "none",
        "trigger_reason": table_note,
        "was_pause_expected": "yes",
        "signal_summary": f"Current run achieved CV AUC {overall_auc:.6f} using profile {RUN_PROFILE}.",
        "risk_summary": f"Decision={decision}; stop_rule={stop_rule_triggered or 'none'}.",
        "compute_summary": f"Runtime {runtime_minutes} minutes over {run_meta['n_splits']} folds.",
        "recommended_next_queue": recommended_next_queue,
        "continue_current_track": "yes" if decision == "keep" else "review first",
        "open_high_cost_track": "no",
        "require_code_cleanup_before_next_batch": "no",
    }
    write_batch_summary(batch_summary_path, summary)

    print("\nGenerated files:")
    print(oof_path)
    print(sub_path)
    print(analysis_path)
    print(exp_oof_path)
    print(exp_sub_path)
    print(exp_analysis_path)
    print(batch_summary_path)


if __name__ == "__main__":
    main()