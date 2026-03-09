import os
import re
from datetime import datetime

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import bi_tri_target_encoding


def get_next_run_number(folder, prefix, suffix):
    """
    获取下一个递增编号，例如：
    submission_1.csv
    submission_2.csv
    submission_3.csv
    """
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


TARGET_COL = "Churn"
ID_COL = "id"

# ===== 保存目录 =====
BASE_SAVE_DIR = "./Previously Trained Files"
OOF_DIR = os.path.join(BASE_SAVE_DIR, "oof_self")
SUB_DIR = os.path.join(BASE_SAVE_DIR, "sub_self")
ANALYSIS_DIR = os.path.join(BASE_SAVE_DIR, "whole analysis")
MODELS_DIR = "./models"

os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== 本次运行编号 =====
run_num = get_next_run_number(SUB_DIR, "submission", ".csv")

# ===== 日志记录 =====
analysis_lines = []
analysis_lines.append(f"Run Number: {run_num}")
analysis_lines.append(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
analysis_lines.append("")

# ===== 读取数据 =====
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

train[TARGET_COL] = train[TARGET_COL].astype(str).str.strip().map({"No": 0, "Yes": 1})

print("train shape:", train.shape)
print("test shape:", test.shape)
print("train columns:", train.columns.tolist())
print("test columns:", test.columns.tolist())
print("target unique values after mapping:", train[TARGET_COL].unique())
print("target dtype after mapping:", train[TARGET_COL].dtype)

analysis_lines.append(f"train shape: {train.shape}")
analysis_lines.append(f"test shape: {test.shape}")
analysis_lines.append(f"train columns: {train.columns.tolist()}")
analysis_lines.append(f"test columns: {test.columns.tolist()}")
analysis_lines.append(f"target unique values after mapping: {train[TARGET_COL].unique()}")
analysis_lines.append(f"target dtype after mapping: {train[TARGET_COL].dtype}")
analysis_lines.append("")

if train[TARGET_COL].isna().any():
    raise ValueError("Churn 映射后出现了空值，请检查原始标签是不是除了 No/Yes 还有别的写法")

# 2. 检查关键列
if TARGET_COL not in train.columns:
    raise ValueError(f"训练集里找不到目标列: {TARGET_COL}，请先确认真实标签列名")

if ID_COL not in test.columns:
    raise ValueError(f"测试集里找不到ID列: {ID_COL}，请先确认真实ID列名")

# 3. 原始特征列
feature_cols = [col for col in train.columns if col not in [TARGET_COL, ID_COL]]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

# ===== 模型参数记录 =====
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "seed": 42
}

analysis_lines.append("Model Params:")
for k, v in params.items():
    analysis_lines.append(f"{k}: {v}")
analysis_lines.append("")

for fold, (train_idx, val_idx) in enumerate(kf.split(train, train[TARGET_COL])):
    print(f"\n========== Fold {fold + 1} ==========")
    analysis_lines.append(f"========== Fold {fold + 1} ==========")

    X_train = train.iloc[train_idx].copy()
    X_val = train.iloc[val_idx].copy()
    X_test = test.copy()

    y_train = X_train[TARGET_COL]
    y_val = X_val[TARGET_COL]

    X_train_enc, X_val_enc, X_test_enc = bi_tri_target_encoding(
        X_train,
        X_val,
        X_test,
        feature_cols,
        target_col=TARGET_COL
    )

    X_train_enc = X_train_enc.fillna(-1)
    X_val_enc = X_val_enc.fillna(-1)
    X_test_enc = X_test_enc.fillna(-1)

    train_features = [col for col in X_train_enc.columns if col not in [TARGET_COL, ID_COL]]

    dtrain = xgb.DMatrix(X_train_enc[train_features], label=y_train)
    dval = xgb.DMatrix(X_val_enc[train_features], label=y_val)
    dtest = xgb.DMatrix(X_test_enc[train_features])

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=100,
        verbose_eval=100
    )

    # 保存每一折模型，带运行编号
    model.save_model(f"./models/xgb_fold{fold + 1}_run{run_num}.json")

    val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    oof_preds[val_idx] = val_pred
    test_preds += test_pred / kf.n_splits

    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")

    analysis_lines.append(f"Fold {fold + 1} AUC: {fold_auc:.6f}")
    analysis_lines.append(f"Best iteration: {model.best_iteration}")
    analysis_lines.append(f"Saved model: ./models/xgb_fold{fold + 1}_run{run_num}.json")
    analysis_lines.append("")

overall_auc = roc_auc_score(train[TARGET_COL], oof_preds)
print(f"\nOverall CV AUC: {overall_auc:.6f}")

analysis_lines.append(f"Overall CV AUC: {overall_auc:.6f}")
analysis_lines.append("")

# 5. 保存 OOF
oof_df = pd.DataFrame({
    ID_COL: train[ID_COL],
    TARGET_COL: train[TARGET_COL],
    "oof_pred": oof_preds
})

# 6. 保存 submission
sub = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET_COL: test_preds
})

oof_path = os.path.join(OOF_DIR, f"oof_predictions_{run_num}.csv")
sub_path = os.path.join(SUB_DIR, f"submission_{run_num}.csv")
analysis_path = os.path.join(ANALYSIS_DIR, f"analysis_{run_num}.txt")

oof_df.to_csv(oof_path, index=False)
sub.to_csv(sub_path, index=False)

analysis_lines.append(f"OOF file: {oof_path}")
analysis_lines.append(f"Submission file: {sub_path}")
analysis_lines.append(f"Analysis file: {analysis_path}")

with open(analysis_path, "w", encoding="utf-8") as f:
    f.write("\n".join(map(str, analysis_lines)))

print("\n已生成文件:")
print(oof_path)
print(sub_path)
print(analysis_path)