# Kaggle S6E3 | Customer Churn Prediction

![Task](https://img.shields.io/badge/Task-Binary%20Classification-2ea44f)
![Metric](https://img.shields.io/badge/Metric-ROC--AUC-ff9800)
![Model](https://img.shields.io/badge/Model-XGBoost-1f6feb)
![CV](https://img.shields.io/badge/CV-5--Fold%20Stratified-8250df)
![Status](https://img.shields.io/badge/Status-Active-00bcd4)
![Python](https://img.shields.io/badge/Python-3.13-3776ab)

English | [中文](#中文说明)

版权声明：`Copyright (c) 2024 shixuan Wang and contributors  All rights reserved.`

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Training Pipeline](#training-pipeline)
- [Outputs](#outputs)
- [Screenshots](#screenshots)
- [Notes](#notes)
- [中文说明](#中文说明)

## Overview

This repository contains my complete workflow for Kaggle Playground Series S6E3 (customer churn prediction), including data, feature engineering, training scripts, model artifacts, and submission history.

### Quick Start

```bash
python -m pip install -r requirements.txt
python src/train.py
```

## Repository Structure

```text
S6E3/
|- data/
|  |- train.csv
|  |- test.csv
|  |- sample_submission.csv
|  `- WA_Fn-UseC_-Telco-Customer-Churn.csv
|- src/
|  |- features.py
|  |- config.py
|  |- plotting.py
|  |- train.py
|  `- utils.py
|- models/
|  |- xgb_fold1_run1.json
|  |- xgb_fold2_run1.json
|  |- xgb_fold3_run1.json
|  |- xgb_fold4_run1.json
|  `- xgb_fold5_run1.json
|- notebooks/
|  `- s6e3_notebook.ipynb
|- Memory/
|  |- daily_log.md
|  |- feature_engineering.md
|  |- ideas.md
|  |- public_scores.md
|  |- training_logs.md
|  `- trials_and_errors.md
|- Previously Trained Files/
|  |- Archieve/
|  |- oof_formal/
|  |- oof_self/
|  |  `- oof_predictions_1.csv
|  |- sub_formal/
|  |- sub_self/
|  |  `- submission_1.csv
|  |- visuals/
|  `- whole analysis/
|- requirements.txt
|- LICENSE
|- S6E3 Readme.md
`- README.md
```

## Training Pipeline

1. Load `data/train.csv` and `data/test.csv`
2. Convert labels `No/Yes` to `0/1`
3. Build 5-fold `StratifiedKFold`
4. Apply target encoding in `src/features.py`
5. Train XGBoost with early stopping
6. Save fold models, OOF predictions, submission file, analysis logs, and plots

## Outputs

- `models/xgb_fold*_run1.json`
- `Previously Trained Files/oof_self/oof_predictions_1.csv`
- `Previously Trained Files/sub_self/submission_1.csv`
- `Previously Trained Files/whole analysis/analysis_*.txt`
- `Previously Trained Files/whole analysis/eval_history_*.json`
- `Previously Trained Files/visuals/fold_auc_*.png`
- `Previously Trained Files/visuals/summary_*.png`
- `Previously Trained Files/visuals/feature_importance_*.png`
- `Previously Trained Files/visuals/oof_distribution_*.png`

## Screenshots

> Add your screenshots here after each important run.

![Training Log Placeholder](./assets/training-log-placeholder.png)
![Feature Importance Placeholder](./assets/feature-importance-placeholder.png)
![Kaggle Score Placeholder](./assets/kaggle-score-placeholder.png)

## Notes

- This repo keeps data and model artifacts for reproducibility.
- `data/train.csv` is large and may trigger GitHub large-file warnings (still below 100MB hard limit).

---

## 中文说明

[English](#kaggle-s6e3--customer-churn-prediction) | 中文

版权声明：`Copyright (c) 2024 yuxuan zhou @mornscience All rights reserved.`

### 项目简介

本仓库记录了我在 Kaggle Playground Series S6E3（客户流失预测）中的完整工作流，包含数据、特征工程、训练脚本、模型产物与提交历史。

### 快速开始

```bash
python -m pip install -r requirements.txt
python src/train.py
```

### 核心信息

- 任务：预测 `Churn` 概率（二分类）
- 指标：ROC-AUC
- 主模型：XGBoost
- 验证方式：5 折分层交叉验证
- 编码方式：Target Encoding (`category_encoders`)

### 训练流程

1. 读取 `data/train.csv` 与 `data/test.csv`
2. 标签映射：`No -> 0`，`Yes -> 1`
3. 构建 5 折分层交叉验证
4. 在 `src/features.py` 中执行目标编码
5. 使用早停策略训练 XGBoost
6. 保存各折模型、OOF 结果、submission、analysis 日志与可视化图表

### 结果产物

- `models/xgb_fold*_run1.json`
- `Previously Trained Files/oof_self/oof_predictions_1.csv`
- `Previously Trained Files/sub_self/submission_1.csv`
- `Previously Trained Files/whole analysis/analysis_*.txt`
- `Previously Trained Files/whole analysis/eval_history_*.json`
- `Previously Trained Files/visuals/fold_auc_*.png`
- `Previously Trained Files/visuals/summary_*.png`
- `Previously Trained Files/visuals/feature_importance_*.png`
- `Previously Trained Files/visuals/oof_distribution_*.png`

### 截图占位

> 你可以把训练日志、特征重要性、Kaggle 分数截图放到 `assets/` 目录，然后替换上面的占位图。

### 许可证

本项目遵循 [LICENSE](LICENSE) 中的许可条款。
