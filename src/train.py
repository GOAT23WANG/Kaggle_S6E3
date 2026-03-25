from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import (
    ANALYSIS_DIR,
    MODELS_DIR,
    OOF_DIR,
    OUTPUT_DIRS,
    SUB_DIR,
    TEST_PATH,
    TRAIN_PATH,
    VISUALS_DIR,
    TrainingConfig,
)
from features import bi_tri_target_encoding
from plotting import (
    plot_feature_importance,
    plot_fold_auc,
    plot_prediction_distribution,
    plot_summary,
)
from utils import (
    ExperimentLogger,
    Timer,
    dump_json,
    ensure_directories,
    format_seconds,
    get_next_run_number,
    save_dataframe,
)


def load_data(config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if config.target_col not in train.columns:
        raise ValueError(f"Training data missing target column: {config.target_col}")
    if config.id_col not in train.columns or config.id_col not in test.columns:
        raise ValueError(f"Train/test data missing id column: {config.id_col}")

    train[config.target_col] = (
        train[config.target_col]
        .astype(str)
        .str.strip()
        .map({"No": 0, "Yes": 1})
        .astype("Int64")
    )

    if train[config.target_col].isna().any():
        raise ValueError("Target mapping produced missing values. Expected labels are 'No' and 'Yes'.")

    train[config.target_col] = train[config.target_col].astype("int8")
    return train, test


def reduce_memory_usage(df: pd.DataFrame, exclude_columns: set[str]) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        if column in exclude_columns:
            continue

        series = df[column]
        if pd.api.types.is_integer_dtype(series):
            df[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            df[column] = pd.to_numeric(series, downcast="float")

    return df


def build_analysis_paths(run_number: int) -> dict[str, Path]:
    return {
        "oof": OOF_DIR / f"oof_predictions_{run_number}.csv",
        "submission": SUB_DIR / f"submission_{run_number}.csv",
        "analysis": ANALYSIS_DIR / f"analysis_{run_number}.txt",
        "fold_auc": VISUALS_DIR / f"fold_auc_{run_number}.png",
        "summary": VISUALS_DIR / f"summary_{run_number}.png",
        "feature_importance": VISUALS_DIR / f"feature_importance_{run_number}.png",
        "oof_distribution": VISUALS_DIR / f"oof_distribution_{run_number}.png",
        "eval_history": ANALYSIS_DIR / f"eval_history_{run_number}.json",
    }


def train_single_fold(
    fold_number: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_cols: list[str],
    config: TrainingConfig,
    run_number: int,
):
    fold_train = train_df.iloc[train_idx].copy()
    fold_val = train_df.iloc[val_idx].copy()
    fold_test = test_df.copy()

    y_train = fold_train[config.target_col].to_numpy(dtype=np.float32, copy=False)
    y_val = fold_val[config.target_col].to_numpy(dtype=np.float32, copy=False)

    fold_train, fold_val, fold_test, metadata = bi_tri_target_encoding(
        fold_train,
        fold_val,
        fold_test,
        feature_cols,
        target_col=config.target_col,
        return_metadata=True,
    )

    train_matrix = fold_train[feature_cols].fillna(config.missing_value_fill)
    val_matrix = fold_val[feature_cols].fillna(config.missing_value_fill)
    test_matrix = fold_test[feature_cols].fillna(config.missing_value_fill)

    dtrain = xgb.DMatrix(train_matrix, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(val_matrix, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(test_matrix, feature_names=feature_cols)

    evals_result: dict = {}
    model = xgb.train(
        params=config.xgb_params,
        dtrain=dtrain,
        num_boost_round=config.num_boost_round,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=config.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=100,
    )

    best_iteration = model.best_iteration if model.best_iteration is not None else config.num_boost_round - 1
    val_pred = model.predict(dval, iteration_range=(0, best_iteration + 1))
    test_pred = model.predict(dtest, iteration_range=(0, best_iteration + 1))
    fold_auc = roc_auc_score(y_val, val_pred)

    model_path = MODELS_DIR / f"xgb_fold{fold_number}_run{run_number}.json"
    model.save_model(model_path)

    importance = model.get_score(importance_type="gain")

    return {
        "fold_number": fold_number,
        "val_idx": val_idx,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "fold_auc": fold_auc,
        "best_iteration": best_iteration,
        "model_path": model_path,
        "evals_result": evals_result,
        "importance": importance,
        "categorical_columns": metadata.categorical_columns,
    }


def main() -> None:
    config = TrainingConfig()
    timer = Timer()
    ensure_directories(OUTPUT_DIRS)

    run_number = get_next_run_number(SUB_DIR, "submission", ".csv")
    logger = ExperimentLogger(run_number=run_number)
    output_paths = build_analysis_paths(run_number)

    train_df, test_df = load_data(config)
    train_df = reduce_memory_usage(train_df, exclude_columns={config.target_col, config.id_col})
    test_df = reduce_memory_usage(test_df, exclude_columns={config.id_col})

    feature_cols = [column for column in train_df.columns if column not in {config.target_col, config.id_col}]

    logger.add(f"Train Shape: {train_df.shape}")
    logger.add(f"Test Shape: {test_df.shape}")
    logger.add(f"Feature Count: {len(feature_cols)}")
    logger.add(f"Feature Columns: {feature_cols}")
    logger.add("")
    logger.add(f"Target Distribution: {train_df[config.target_col].value_counts(normalize=False).to_dict()}")
    logger.add(
        f"Target Distribution Ratio: {train_df[config.target_col].value_counts(normalize=True).round(6).to_dict()}"
    )
    logger.add("")
    logger.add_mapping("XGBoost Parameters:", config.xgb_params)

    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    oof_predictions = np.zeros(len(train_df), dtype=np.float32)
    test_predictions = np.zeros(len(test_df), dtype=np.float32)
    fold_aucs: list[float] = []
    fold_best_iterations: list[int] = []
    model_paths: list[str] = []
    eval_history: dict[str, dict] = {}
    categorical_columns_snapshot: list[str] = []
    aggregated_importance: dict[str, float] = {}

    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(train_df, train_df[config.target_col]),
        start=1,
    ):
        print(f"\n========== Fold {fold_number} ==========")
        logger.add_section(f"========== Fold {fold_number} ==========")

        fold_result = train_single_fold(
            fold_number=fold_number,
            train_df=train_df,
            test_df=test_df,
            train_idx=train_idx,
            val_idx=val_idx,
            feature_cols=feature_cols,
            config=config,
            run_number=run_number,
        )

        oof_predictions[fold_result["val_idx"]] = fold_result["val_pred"]
        test_predictions += fold_result["test_pred"] / config.n_splits
        fold_aucs.append(fold_result["fold_auc"])
        fold_best_iterations.append(fold_result["best_iteration"])
        model_paths.append(str(fold_result["model_path"]))
        eval_history[f"fold_{fold_number}"] = fold_result["evals_result"]
        best_valid_auc = fold_result["evals_result"]["valid"]["auc"][fold_result["best_iteration"]]
        best_train_auc = fold_result["evals_result"]["train"]["auc"][fold_result["best_iteration"]]

        if not categorical_columns_snapshot:
            categorical_columns_snapshot = fold_result["categorical_columns"]

        for feature_name, importance_value in fold_result["importance"].items():
            aggregated_importance[feature_name] = aggregated_importance.get(feature_name, 0.0) + importance_value

        logger.add(f"Fold {fold_number} AUC: {fold_result['fold_auc']:.6f}")
        logger.add(f"Best Iteration: {fold_result['best_iteration']}")
        logger.add(f"Best Train AUC: {best_train_auc:.6f}")
        logger.add(f"Best Valid AUC: {best_valid_auc:.6f}")
        logger.add(f"Model Path: {fold_result['model_path']}")
        logger.add("")
        print(f"Fold {fold_number} AUC: {fold_result['fold_auc']:.6f}")

    overall_auc = roc_auc_score(train_df[config.target_col], oof_predictions)
    print(f"\nOverall CV AUC: {overall_auc:.6f}")

    oof_df = pd.DataFrame(
        {
            config.id_col: train_df[config.id_col],
            config.target_col: train_df[config.target_col],
            "oof_pred": oof_predictions,
        }
    )
    submission_df = pd.DataFrame(
        {
            config.id_col: test_df[config.id_col],
            config.target_col: test_predictions,
        }
    )

    save_dataframe(oof_df, output_paths["oof"])
    save_dataframe(submission_df, output_paths["submission"])
    dump_json(eval_history, output_paths["eval_history"])

    mean_importance = {
        feature_name: score / config.n_splits for feature_name, score in aggregated_importance.items()
    }
    plot_fold_auc(fold_aucs, output_paths["fold_auc"])
    plot_summary(fold_aucs, overall_auc, output_paths["summary"])
    plot_feature_importance(mean_importance, output_paths["feature_importance"])
    plot_prediction_distribution(oof_predictions, output_paths["oof_distribution"])

    logger.add_section("Run Summary")
    logger.add(f"Detected Categorical Columns: {categorical_columns_snapshot}")
    logger.add(f"Fold AUCs: {[round(score, 6) for score in fold_aucs]}")
    logger.add(f"Overall CV AUC: {overall_auc:.6f}")
    logger.add(f"Best Iterations: {fold_best_iterations}")
    logger.add(f"Model Paths: {model_paths}")
    logger.add(f"OOF Path: {output_paths['oof']}")
    logger.add(f"Submission Path: {output_paths['submission']}")
    logger.add(f"Fold AUC Plot: {output_paths['fold_auc']}")
    logger.add(f"Summary Plot: {output_paths['summary']}")
    logger.add(f"Feature Importance Plot: {output_paths['feature_importance']}")
    logger.add(f"OOF Distribution Plot: {output_paths['oof_distribution']}")
    logger.add(f"Visuals Directory: {VISUALS_DIR}")
    logger.add(f"Eval History Path: {output_paths['eval_history']}")
    logger.add(f"Training Time: {format_seconds(timer.elapsed_seconds)}")
    logger.add("")
    logger.add("Experiment Notes:")
    logger.add("This run keeps the XGBoost pipeline intact, centralizes output management,")
    logger.add("uses fold-safe target encoding for categorical features, and writes visual summaries.")

    logger.write(output_paths["analysis"])

    print("\nGenerated files:")
    print(output_paths["oof"])
    print(output_paths["submission"])
    print(output_paths["analysis"])
    print(output_paths["fold_auc"])
    print(output_paths["summary"])
    print(output_paths["feature_importance"])
    print(output_paths["oof_distribution"])


if __name__ == "__main__":
    main()
