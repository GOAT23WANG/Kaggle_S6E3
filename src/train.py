import argparse
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import build_feature_matrices, encode_fold_features


TARGET_COL = "Churn"
ID_COL = "id"

BASE_SAVE_DIR = "./Previously Trained Files"
OOF_DIR = os.path.join(BASE_SAVE_DIR, "oof_self")
SUB_DIR = os.path.join(BASE_SAVE_DIR, "sub_self")
ANALYSIS_DIR = os.path.join(BASE_SAVE_DIR, "whole analysis")
MODELS_DIR = "./models"

DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "eta": 0.0063,
    "subsample": 0.81,
    "colsample_bytree": 0.32,
    "min_child_weight": 6,
    "reg_alpha": 3.5017,
    "reg_lambda": 1.2925,
    "gamma": 0.79,
    "seed": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle S6E3 training pipeline")
    parser.add_argument("--train-path", type=str, default="./data/train.csv")
    parser.add_argument("--test-path", type=str, default="./data/test.csv")
    parser.add_argument("--orig-path", type=str, default="./data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--n-folds", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=12000)
    parser.add_argument("--early-stopping-rounds", type=int, default=500)
    parser.add_argument("--verbose-eval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_next_run_number(folder: str, prefix: str, suffix: str) -> int:
    os.makedirs(folder, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(suffix)}$")
    max_num = 0

    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num + 1


def main() -> None:
    args = parse_args()

    os.makedirs(OOF_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    run_num = get_next_run_number(SUB_DIR, "submission", ".csv")

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    orig = pd.read_csv(args.orig_path)

    train[TARGET_COL] = train[TARGET_COL].astype(str).str.strip().map({"No": 0, "Yes": 1})
    orig[TARGET_COL] = orig[TARGET_COL].astype(str).str.strip().map({"No": 0, "Yes": 1})

    if train[TARGET_COL].isna().any():
        raise ValueError("train Churn contains NaN after label mapping")
    if TARGET_COL not in train.columns:
        raise ValueError(f"missing target column: {TARGET_COL}")
    if ID_COL not in test.columns:
        raise ValueError(f"missing id column: {ID_COL}")

    analysis_lines = [
        f"Run Number: {run_num}",
        f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"train shape: {train.shape}",
        f"test shape: {test.shape}",
        f"orig shape: {orig.shape}",
        "",
        "Config:",
        f"n_folds={args.n_folds}",
        f"inner_folds={args.inner_folds}",
        f"num_boost_round={args.num_boost_round}",
        f"early_stopping_rounds={args.early_stopping_rounds}",
        "",
    ]

    print("Building feature matrices...")
    train_feat, test_feat, meta = build_feature_matrices(
        train_df=train,
        test_df=test,
        orig_df=orig,
        target_col=TARGET_COL,
    )
    print("Feature engineering done.")
    print("train features shape:", train_feat.shape)
    print("test features shape:", test_feat.shape)
    analysis_lines.append(f"train feature shape: {train_feat.shape}")
    analysis_lines.append(f"test feature shape: {test_feat.shape}")
    analysis_lines.append("")

    kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof_preds = np.zeros(len(train_feat), dtype=np.float32)
    test_preds = np.zeros(len(test_feat), dtype=np.float32)

    xgb_params = DEFAULT_XGB_PARAMS.copy()
    xgb_params["seed"] = args.seed

    analysis_lines.append("Model Params:")
    for k, v in xgb_params.items():
        analysis_lines.append(f"{k}: {v}")
    analysis_lines.append("")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_feat, train_feat[TARGET_COL])):
        print(f"\n========== Fold {fold + 1}/{args.n_folds} ==========")

        tr_df = train_feat.iloc[tr_idx].copy()
        va_df = train_feat.iloc[va_idx].copy()
        y_tr = tr_df[TARGET_COL].values
        y_va = va_df[TARGET_COL].values

        X_tr_enc, X_va_enc, X_te_enc, feature_cols = encode_fold_features(
            train_fold_df=tr_df,
            val_fold_df=va_df,
            test_df=test_feat,
            target_col=TARGET_COL,
            id_col=ID_COL,
            te_base_cols=meta["te_base_cols"],
            ngram_cols=meta["ngram_cols"],
            inner_folds=args.inner_folds,
            seed=args.seed,
        )

        dtrain = xgb.DMatrix(X_tr_enc[feature_cols], label=y_tr)
        dvalid = xgb.DMatrix(X_va_enc[feature_cols], label=y_va)
        dtest = xgb.DMatrix(X_te_enc[feature_cols])

        model = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=args.num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=args.verbose_eval,
        )

        model_path = os.path.join(MODELS_DIR, f"xgb_fold{fold + 1}_run{run_num}.json")
        model.save_model(model_path)

        val_pred = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

        oof_preds[va_idx] = val_pred
        test_preds += test_pred / args.n_folds

        fold_auc = roc_auc_score(y_va, val_pred)
        print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")

        analysis_lines.append(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
        analysis_lines.append(f"Best iteration: {model.best_iteration}")
        analysis_lines.append(f"Saved model: {model_path}")
        analysis_lines.append("")

    overall_auc = roc_auc_score(train_feat[TARGET_COL], oof_preds)
    print(f"\nOverall CV AUC: {overall_auc:.6f}")

    oof_df = pd.DataFrame({ID_COL: train_feat[ID_COL], TARGET_COL: train_feat[TARGET_COL], "oof_pred": oof_preds})
    sub_df = pd.DataFrame({ID_COL: test_feat[ID_COL], TARGET_COL: test_preds})

    oof_path = os.path.join(OOF_DIR, f"oof_predictions_{run_num}.csv")
    sub_path = os.path.join(SUB_DIR, f"submission_{run_num}.csv")
    analysis_path = os.path.join(ANALYSIS_DIR, f"analysis_{run_num}.txt")

    oof_df.to_csv(oof_path, index=False)
    sub_df.to_csv(sub_path, index=False)

    analysis_lines.append(f"Overall CV AUC: {overall_auc:.6f}")
    analysis_lines.append("")
    analysis_lines.append(f"OOF file: {oof_path}")
    analysis_lines.append(f"Submission file: {sub_path}")
    analysis_lines.append(f"Analysis file: {analysis_path}")

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(analysis_lines))

    print("\nSaved:")
    print(oof_path)
    print(sub_path)
    print(analysis_path)


if __name__ == "__main__":
    main()
