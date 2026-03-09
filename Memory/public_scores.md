# S6E3 Public Leaderboard Scores

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed from Kaggle
> 2. **DO NOT EDIT/REMOVE** previous score entries
> 3. **PREPEND** new scores (latest first) within category
> 4. **ORDER** by LB Score (Highest on Top)
> 5. **Include:** OOF, LB, Gap, Training Time
> 6. **CATEGORIZE:** TabM, XGBoost, LightGBM, FTT, Ensemble
> 7. **Status:** 🏆 Best | ✅ Good | ❌ Failed/Overfit

---

## 📝 Score Logging Format

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V# | MM-DD | X.XXXXX | X.XXXXX | -0.XXX | XX min | `file.py` | `oof.csv` | `sub.csv` | Notes |

---

### Leaderboard Scores Top 5

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V16b | 03-07 | **0.91680** | 0.91925 | -0.00245 | 80.0 min | `S6E3_V16_XGB_DigitFeatures.py` | `oof_v16b.csv` | `sub_v16b.csv` | V16 with 20-Fold CV 🏆 OVERALL BEST |
| V16 | 03-06 | **0.91679** | 0.91917 | -0.00238 | 38.0 min | `S6E3_V16_XGB_DigitFeatures.py` | `oof_v16.csv` | `sub_v16.csv` | V14 + 46 Digit Features (Modulo/Rounding) 🏆 2nd Best Base |
| V15 | 03-06 | **0.91657** | 0.91897 | +0.00068 | 69.2 min | `S6E3_V15_20Fold.py` | `oof_v15.csv` | `sub_v15.csv` | V14 Bi-gram TE with 20-Fold CV ✅ |
| V14 | 03-04 | **0.91656** | 0.91889 | -0.00233 | 31.6 min | `S6E3_V14_BigramTE.py` | `oof_v14.csv` | `sub_v14.csv` | Bi-gram/Tri-gram TE (Composite Categoricals) ✅ |
| V12 | 03-04 | **0.91652** | 0.91892 | -0.00240 | 47.2 min | `S6E3_V12_XGBoost_Optuna.py` | `oof_v12.csv` | `sub_v12.csv` | Optuna HPO (93 trials) + V7 Features ✅ |
| V13 | 03-04 | **0.91652** | 0.91890 | -0.00238 | 89.0 min | `S6E3_V13_LightGBM_Optuna.py` | `oof_v13.csv` | `sub_v13.csv` | Optuna LGBM HPO (50 trials) 🏆 Tied |

### XGBoost Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V16b | 03-07 | **0.91680** | 0.91925 | -0.00245 | 80.0 min | `S6E3_V16_XGB_DigitFeatures.py` | `oof_v16b.csv` | `sub_v16b.csv` | V16 with 20-Fold CV 🏆 OVERALL BEST |
| V16 | 03-06 | **0.91679** | 0.91917 | -0.00238 | 38.0 min | `S6E3_V16_XGB_DigitFeatures.py` | `oof_v16.csv` | `sub_v16.csv` | V14 + 46 Digit Features (Modulo/Rounding) 🏆 2nd Best Base |
| V15 | 03-06 | **0.91657** | 0.91897 | +0.00240 | 69.2 min | `S6E3_V15_20Fold.py` | `oof_v15.csv` | `sub_v15.csv` | V14 Bi-gram TE with 20-Fold CV 🏆 OVERALL BEST |
| V14 | 03-04 | **0.91656** | 0.91889 | -0.00233 | 31.6 min | `S6E3_V14_BigramTE.py` | `oof_v14.csv` | `sub_v14.csv` | Bi-gram/Tri-gram TE (Composite Categoricals) ✅ |
| V12 | 03-04 | **0.91652** | 0.91892 | -0.00240 | 47.2 min | `S6E3_V12_XGBoost_Optuna.py` | `oof_v12.csv` | `sub_v12.csv` | Optuna HPO (93 trials, lr↓8x, col↓60%) + V7 Features ✅ |
| V8 | 03-02 | **0.91645** | 0.91857 | -0.00212 | 10.8 min | `S6E3_V8_XGBoost_AllDistFeatures.py` | `oof_v8.csv` | `sub_v8.csv` | V3 XGB + V7 Features (EXP3+EXP5) ✅ |
| V17 | 03-07 | **0.91621** | 0.93770 | -0.02149 | 38.7 min | `S6E3_V17_NoisePruning.py` | `oof_v17.csv` | `sub_v17.csv` | Two-Stage Confident Learning Pruning ❌ Failed (Lost generalization) |
| V3 | 03-01 | **0.91607** | 0.91774 | -0.00167 | 15.2 min | `S6E3_V3_InnerKFoldTE.py` | `oof_v3.csv` | `sub_v3.csv` | Inner K-Fold TE, Freq Enc, Arithmetic Interactions, 10 Folds ✅ |
| V1 | 03-01 | **0.91411** | 0.91659 | -0.00248 | 4.1 min | `S6E3_V1_Baseline.py` | `oof_v1.csv` | `sub_v1.csv` | First XGB+cuDF Pseudo-Label Baseline ✅ |

### LightGBM Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V20 | 03-08 | 0.91661 | 0.91908 | -0.00253 | 151.9 min | `S6E3_V20_LightGBM.py` | `oof_v20.csv` | `sub_v20.csv` | LightGBM Optuna HPO + V16 Features. ⚠️ Worse than XGBoost V16b |
| V13 | 03-04 | **0.91652** | 0.91890 | -0.00238 | 89.0 min | `S6E3_V13_LightGBM_Optuna.py` | `oof_v13.csv` | `sub_v13.csv` | Optuna HPO (50 trials, col=0.30, path_smooth=8.89) 🏆 |
| V7 | 03-02 | **0.91637** | 0.91851 | -0.00214 | 29.7 min | `S6E3_V7_LightGBM_QuantileDistFeatures.py` | `oof_v7.csv` | `sub_v7.csv` | V6 + 8 EXP5 Quantile Distance Features ✅ |
| V6 | 03-02 | **0.91630** | 0.91842 | -0.00212 | 29.2 min | `S6E3_V6_LightGBM_DistFeatures.py` | `oof_v6.csv` | `sub_v6.csv` | V4 + 9 EXP3 Distribution Features ✅ |
| V4 | 03-01 | **0.91609** | 0.91827 | -0.00218 | 28.2 min | `S6E3_V4_LightGBM_InnerKFoldTE.py` | `oof_v4.csv` | `sub_v4.csv` | 1:1 LightGBM architecture swap of V3 ✅ |

### Neural Network Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V9 | 03-03 | **0.91625** | 0.91845 | -0.00220 | 232.7 min | `S6E3_V9_TabM_AllDistFeatures.py` | `oof_v9.csv` | `sub_v9.csv` | TabM (ICLR 2025) + V7 Features + Inner TE 🏆 Best NN |
| V10 | 03-03 | 0.91491 | 0.91633 | -0.00142 | 263.4 min | `S6E3_V10_RealMLP_AllDistFeatures.py` | `oof_v10.csv` | `sub_v10.csv` | RealMLP_TD (S6E2 V48 params) + V7 Features ✅ |
| V5 | 03-01 | 0.91377 | 0.91396 | -0.00019 | 48.0 min | `S6E3_V5_RealMLP_DualRep.py` | `oof_v5.csv` | `sub_v5.csv` | pytabkit RealMLP, DualRep, Orig Stats ✅ |

### CatBoost Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V19 | 03-08 | 0.91648 | 0.91900 | -0.00252 | 49.1 min | `S6E3_V19_CatBoost.py` | `oof_v19.csv` | `sub_v19.csv` | CatBoost Optuna HPO + V16 Features. ⚠️ Still worse than XGBoost V16b |
| V18 | 03-07 | 0.91640 | 0.91892 | -0.00052 | 29.8 min | `S6E3_V18_CatBoost_DigitFeatures.py` | `oof_v18.csv` | `sub_v18.csv` | CatBoost + V16 Digit Features. ❌ Underperforms XGBoost V16b |
| V11 | 03-03 | 0.91494 | 0.91736 | -0.00242 | 17.7 min | `S6E3_V11_CatBoost_AllDistFeatures.py` | `oof_v11.csv` | `sub_v11.csv` | CatBoost Depthwise + V7 Features. ❌ Underperforms XGB/LGBM |

### Failed Experiments (No LB Submission or Major Drop)
| Version | Date | Status | Fold 1 AUC | Time | Script | Notes |
|---------|------|--------|-----------|------|--------|-------|
| V14b | 03-04 | ❌ OVERFIT | 0.91923 (OOF: 0.91891) | 28.3 min | `S6E3_V14b_PolyFeatures.py` | Polynomials (x², x³) on numericals. Overfit: LB 0.91627 (-0.00025 vs V12). |
| V14 | 03-04 | ❌ KILLED | 0.91846 | 350 min/fold | `S6E3_V14_XGBoost_DART.py` | DART booster: 74x slower, -0.00078 vs V12. **NEVER USE** |
| V15 | 03-04 | ❌ NO GAIN | — | 178.6 min | `S6E3_V15_MultiExperiment.py` | Focal Loss / scale_pos_weight / colsample grid / feature pruning all ≤+0.00004 |
| EXP-V15 | 03-05 | ❌ NO GAIN | — | 22.1 min | `S6E3_EXP_V15_MultiFeature.py` | 5-tech screen: Binning+TE (=), Churn Flags (-0.00007), Quantile TF (=), DAE (-0.00027), SHAP RFE (-0.00005). FE ceiling reached. |
| V15-TabR | 03-05 | ❌ KILLED | 0.79934 (Best epoch 4) | ~30 min | `S6E3_V15_TabR.py` | TabR ICLR 2024: ~6 min/epoch on 534K rows. 20hr ETA for 10 folds. PERMANENTLY DEAD at this scale. |
| V15f | 03-05 | ❌ NO GAIN | 0.91922 (OOF: 0.91883) | 29.0 min | `S6E3_V15_AllCat_CatBoost.py` | AllCat Mega-String TE: Concatenating 16 cats created 44k unique profiles. Too sparse, smoothed away (-0.00006 vs V14). |
| V15g | 03-05 | ❌ NO GAIN | 0.91618 (OOF: 0.91639) | 19.8 min | `S6E3_V15_AllCat_CatBoost.py` | CatBoost LIGHT (Raw+Newton): Native ordered TE much weaker than manual Inner K-Fold TE (-0.00250 vs V14). |
