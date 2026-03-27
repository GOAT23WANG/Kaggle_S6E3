#!/usr/bin/env bash
set -e

export EXPERIMENT_ID=EXP-P3-003
export RUN_PROFILE=unified_v16
export RUN_PRIORITY=P3
export RUN_STAGE=architecture_innovation
export RUN_CHANGE_TYPE=ensemble
export RUN_CHANGE_DETAIL=multi_seed_5
export HYPOTHESIS="Averaging predictions from 5 different random seeds reduces variance and improves generalization."
export PARENT_BASELINE=EXP-P0-003
export N_FOLDS=10
export ENABLE_RIDGE_FEATURE=0
export MULTI_SEED=1
export N_SEEDS=5
export XGB_DEVICE=cuda
export XGB_TREE_METHOD=hist

cd /root/aicloud-data/Kaggle_S6E3/src
exec /root/aicloud-data/Kaggle_S6E3/.conda/bin/python train.py
