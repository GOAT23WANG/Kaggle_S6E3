from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TARGET_COL = "Churn"
ID_COL = "id"
N_SPLITS = 5
RANDOM_STATE = 42
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
MISSING_VALUE_FILL = -1.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

BASE_SAVE_DIR = PROJECT_ROOT / "Previously Trained Files"
OOF_DIR = BASE_SAVE_DIR / "oof_self"
SUB_DIR = BASE_SAVE_DIR / "sub_self"
ANALYSIS_DIR = BASE_SAVE_DIR / "whole analysis"
VISUALS_DIR = BASE_SAVE_DIR / "visuals"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class TrainingConfig:
    target_col: str = TARGET_COL
    id_col: str = ID_COL
    n_splits: int = N_SPLITS
    random_state: int = RANDOM_STATE
    num_boost_round: int = NUM_BOOST_ROUND
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS
    missing_value_fill: float = MISSING_VALUE_FILL

    @property
    def xgb_params(self) -> dict:
        return {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "max_bin": 256,
            "tree_method": "hist",
            "nthread": -1,
            "seed": self.random_state,
        }


OUTPUT_DIRS = {
    "oof": OOF_DIR,
    "submission": SUB_DIR,
    "analysis": ANALYSIS_DIR,
    "visuals": VISUALS_DIR,
    "models": MODELS_DIR,
}
