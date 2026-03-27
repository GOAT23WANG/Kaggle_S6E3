#!/usr/bin/env bash
set -e

export EXPERIMENT_ID=EXP-P3-001
export RUN_PROFILE=unified_v16
export RUN_PRIORITY=P3
export RUN_STAGE=architecture_innovation
export RUN_CHANGE_TYPE=architecture
export RUN_CHANGE_DETAIL=two_stage_ridge_xgb
export HYPOTHESIS="Ridge linear predictions as additional feature provide orthogonal linear signal that XGB trees cannot efficiently reconstruct."
export PARENT_BASELINE=EXP-P0-003
export ENABLE_RIDGE_FEATURE=1
export MULTI_SEED=0
export XGB_DEVICE=cuda
export XGB_TREE_METHOD=hist

cd /root/aicloud-data/Kaggle_S6E3/src
exec /root/aicloud-data/Kaggle_S6E3/.conda/bin/python train.py
