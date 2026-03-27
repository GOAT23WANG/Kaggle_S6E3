#!/usr/bin/env bash
set -e

export EXPERIMENT_ID=EXP-P3-002
export RUN_PROFILE=unified_v16
export RUN_PRIORITY=P3
export RUN_STAGE=architecture_innovation
export RUN_CHANGE_TYPE=cv_strategy
export RUN_CHANGE_DETAIL=20fold_cv
export HYPOTHESIS="20-fold CV reduces variance in OOF estimates and produces more stable predictions for ensembling."
export PARENT_BASELINE=EXP-P0-003
export N_FOLDS=20
export ENABLE_RIDGE_FEATURE=0
export MULTI_SEED=0
export XGB_DEVICE=cuda
export XGB_TREE_METHOD=hist

cd /root/aicloud-data/Kaggle_S6E3/src
exec /root/aicloud-data/Kaggle_S6E3/.conda/bin/python train.py
