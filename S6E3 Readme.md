# S6E3 Customer Churn Project

This folder is my working repository for Kaggle Playground Series S6E3 (customer churn prediction), including training code, feature engineering, experiment history, and saved model outputs.

## Project Highlights

- Task: Binary classification (`Churn`)
- Metric: ROC-AUC
- Main model: XGBoost with 5-fold Stratified CV
- Encoding: Target Encoding for categorical features
- Outputs: OOF predictions, submission files, and fold models

## Folder Structure

```text
S6E3/
|- data/
|  |- train.csv
|  |- test.csv
|  |- sample_submission.csv
|  `- WA_Fn-UseC_-Telco-Customer-Churn.csv
|- src/
|  |- train.py
|  `- features.py
|- notebooks/
|  `- s6e3_notebook.ipynb
|- models/
|  `- xgb_fold1.json (and other fold models after training)
|- Previously Trained Files/
|  |- Archieve/
|  |- oof_formal/
|  |- oof_self/
|  |- sub_formal/
|  `- sub_self/
|- Memory/
|  |- daily_log.md
|  |- feature_engineering.md
|  |- ideas.md
|  |- public_scores.md
|  |- training_logs.md
|  `- trials_and_errors.md
|- requirements.txt
`- LICENSE
```

## Training Pipeline

`src/train.py` currently does:

1. Read `data/train.csv` and `data/test.csv`
2. Map target labels (`No -> 0`, `Yes -> 1`)
3. Run 5-fold `StratifiedKFold`
4. Apply Target Encoding from `src/features.py`
5. Train XGBoost with early stopping
6. Save artifacts:
   - `models/xgb_fold*.json`
   - `Previously Trained Files/oof_self/oof_predictions.csv`
   - `Previously Trained Files/sub_self/submission.csv`

## Quick Start

```bash
# From S6E3 folder
python -m pip install -r requirements.txt
python src/train.py
```

## Dependencies

Core dependencies from `requirements.txt`:

- pandas
- numpy
- scikit-learn
- xgboost
- category_encoders
- tqdm
- joblib

## Notes

- `Previously Trained Files` stores historical experiments and past submissions.
- `Memory` stores process notes (ideas, logs, feature records) to track iteration decisions.
