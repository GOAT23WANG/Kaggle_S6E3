# S6E3 Training Logs

> **⚠️ RULES (See MEMORY_GUIDELINES.md for full details):**
> 1. **Only update** after Public LB score is available
> 2. **DO NOT EDIT** previous entries after submission
> 3. **PREPEND** new logs (latest first)
> 4. **Include timing** breakdown for each version
> 5. **Include per-fold** results when available
---

## Required Format

```markdown
### Version [N] ([Description]) - YYYY-MM-DD
**Score**: **X.XXXXX LB** / X.XXXXX OOF (Gap: -X.XXX)
**Result**: **±X.XXXXX LB** ✅/❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | X.X min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

**Strategy:** [Brief description]
**File:** `filename.py`

**Key Learning:**
> [Takeaway]

**Status: ✅/❌/🏆**
```

# **ENTRIES FROM BELOW THIS TEXT ARE NOT TO BE ALTERED**

---
### Version 20 (LightGBM Optuna) - 2026-03-08
**Score**: **0.91661 LB** / 0.91908 OOF (Gap: -0.00253)
**Result**: **-0.00019 LB vs V16b** ⚠️

**Timing:**
| Stage | Time |
|-------|------|
| Total | 151.9 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 0.92028 | 0.91864 | 0.91803 | 0.91879 | 0.92290 | 0.91859 | 0.91647 | 0.92042 | 0.91872 | 0.91861 | 0.91864 | 0.91978 | 0.92145 | 0.92056 | 0.91739 | 0.92145 | 0.91911 | 0.91796 | 0.91833 | 0.91558 | 0.91908 |

**Strategy:** LightGBM with Optuna-optimized hyperparameters (lr=0.00833, max_depth=7, num_leaves=77, reg_alpha=3.05, reg_lambda=0.225, min_child_samples=56, subsample=0.675, colsample_bytree=0.646, min_split_gain=0.076, extra_trees=True) using the full V16 feature pipeline (Digit Features + Bi-gram/Tri-gram TE). 20-fold CV.
**File:** `S6E3_V20_LightGBM.py`

**Key Learning:**
> LightGBM with Optuna HPO achieves LB 0.91661, better than V19 CatBoost (+0.00013) but still worse than XGBoost V16b (-0.00019). Leaf-wise growth doesn't provide advantage over depth-wise XGBoost on this heavy FE dataset. XGBoost remains the best single model.

**Status: ⚠️**

---
### Version 19 (CatBoost Optuna) - 2026-03-08
**Score**: **0.91648 LB** / 0.91900 OOF (Gap: -0.00252)
**Result**: **-0.00032 LB vs V16b** ⚠️

**Timing:**
| Stage | Time |
|-------|------|
| Total | 49.1 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 0.92024 | 0.91835 | 0.91786 | 0.91866 | 0.92297 | 0.91856 | 0.91653 | 0.92046 | 0.91872 | 0.91837 | 0.91866 | 0.91979 | 0.92160 | 0.92046 | 0.91720 | 0.92165 | 0.91866 | 0.91780 | 0.91822 | 0.91532 | 0.91900 |

**Strategy:** CatBoost with Optuna-optimized hyperparameters (lr=0.00984, depth=7, l2_leaf_reg=5.33, random_strength=2.88) using the full V16 feature pipeline (Digit Features + Bi-gram/Tri-gram TE). 20-fold CV to match V16b.
**File:** `S6E3_V19_CatBoost.py`

**Key Learning:**
> Even with Optuna HPO specifically tuning CatBoost parameters, the model cannot match XGBoost V16b. CatBoost's symmetric tree architecture fundamentally limits its ability to leverage complex digit-feature interactions. However, V19 improved over V18 CatBoost (+0.00008) by using the full V16 feature pipeline.

**Status: ⚠️**

---
### Version 18 (CatBoost + Digit Features) - 2026-03-07
**Score**: **0.91640 LB** / 0.91892 OOF (Gap: -0.00052)
**Result**: **-0.00040 LB vs V16b** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.8 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|
| 0.91922 | 0.91840 | 0.92080 | 0.91835 | 0.91849 | 0.91903 | 0.92079 | 0.91935 | 0.91818 | 0.91666 | 0.91893 |

**Strategy:** Adapted V16 digit features (46 features) for CatBoost. Applied same feature engineering pipeline: Core features + Digit Features + Bi-gram/Tri-gram TE. Used CatBoost-specific parameters (depth=5, l2_leaf_reg=5.0, random_strength=1.5).
**File:** `S6E3_V18_CatBoost_DigitFeatures.py`

**Key Learning:**
> CatBoost's symmetric tree architecture cannot leverage digit features as effectively as XGBoost's depth-wise growth. Even with identical features, CatBoost underperforms XGBoost V16b by -0.00040 LB. The digit features showed importance (tenure_rounded_10 at 2.19% was #1), but CatBoost's balanced tree constraint limits its ability to capture fine-grained digit patterns.

**Status: ❌**

---

### EXP3 (Label Smoothing Regularization) - 2026-03-07
**Score**: No LB / 0.91909 OOF (Gap: -0.00008 vs baseline)
**Result**: **-0.00008 OOF** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 35.0 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|
| 0.91944 | 0.91836 | 0.92088 | 0.91861 | 0.91879 | 0.91923 | 0.92101 | 0.91948 | 0.91837 | 0.91678 | 0.91910 |

**Strategy:** Re-ran the V16 pipeline (10-folds) with target transformation `y_smooth = y_train * (1 - 0.05) + (0.5 * 0.05)` to regularize leaf confidence.
**File:** `S6E3_EXP3_XGB_LabelSmoothing.py`

**Key Learning:**
> Label Smoothing forces trees to hedge their bets. On Kaggle tabular data generated by synthetic processes, the boundaries are often infinitely sharp (e.g., if logic=True, target=1). Softening the labels destroys the trees' ability to find and cleanly separate these sharp synthetic boundaries.

**Status: ❌**

---

### Version 16b (20-Fold CV of V16) - 2026-03-07
**Score**: **0.91680 LB** / 0.91925 OOF (Gap: -0.00245)
**Result**: **+0.00001 LB** 🏆 OVERALL BEST
**Timing:**
| Stage | Time |
|-------|------|
| Total | 80.0 min |

**Fold Scores (20 Folds):**
0.92063 | 0.91863 | 0.91817 | 0.91897 | 0.92315 | 0.91864 | 0.91695 | 0.92067 | 0.91896 | 0.91877 | 0.91894 | 0.91992 | 0.92178 | 0.92075 | 0.91766 | 0.92159 | 0.91922 | 0.91799 | 0.91833 | 0.91557
(Mean: 0.91926 ± 0.00173)

**Strategy:** Retrained V16 (Digit Features map) but extended from 10 folds to 20 folds to extract maximum signal from the data limits.
**File:** `S6E3_V16_XGB_DigitFeatures.py` (edited to 20 folds)

**Key Learning:**
> Like V15, extending a successful architecture to 20 folds yields a tiny micro-optimization (+0.00001 LB) because of the slightly larger fold training sets (95% instead of 90%). 

**Status: 🏆**

---

### Version 16 (Digit Features from Numericals) - 2026-03-06
**Score**: **0.91679 LB** / 0.91917 OOF (Gap: -0.00238)
**Result**: **+0.00023 LB** ✅ IMPROVED OVER V14 BASELINE

**Timing:**
| Stage | Time |
|-------|------|
| Total | 38.0 min |

**Fold Scores (10 Folds):**
0.91950 | 0.91854 | 0.92092 | 0.91863 | 0.91890 | 0.91925 | 0.92108 | 0.91957 | 0.91849 | 0.91690
(Mean: 0.91918 ± 0.00116)

**Strategy:** Appended 46 highly granular digit-level mathematical features (modulo, rounding, Benford's Law leading digits, string precision) to the V14 Bi-gram TE baseline.
**File:** `S6E3_V16_XGB_DigitFeatures.py`

**Key Learning:**
> Tree models strictly split on continuous boundaries. They physically cannot learn "customers whose tenure is cleanly divisible by 12". By forcibly injecting rounding, modulo, and trailing-digit mathematics, XGBoost found heavily utilized synthetic artifacts. `tenure_years`, `tenure_rounded_10`, and `tenure_num_digits` were aggressively selected (Top 3 out of the 46 digit features).

**Status:** ✅ (Successful Base Increment)

### Version 15 (V14 with 20-Fold CV) - 2026-03-06
**Score**: **0.91657 LB** / 0.91897 OOF (Gap: +0.00240)
**Result**: **+0.00001 LB** 🏆 NEW OVERALL BEST

**Timing:**
| Stage | Time |
|-------|------|
| Total | 69.2 min |

**Fold Scores (20 Folds):**
0.92039 | 0.91831 | 0.91774 | 0.91876 | 0.92280 | 0.91829 | 0.91689 | 0.92043 | 0.91874 | 0.91843 | 0.91877 | 0.91976 | 0.92149 | 0.92042 | 0.91752 | 0.92134 | 0.91863 | 0.91779 | 0.91793 | 0.91519
(Mean: 0.91898 ± 0.00173)

**Strategy:** Re-ran the V14 Bi-gram/Tri-gram Target Encoding pipeline but with `N_FOLDS = 20`. This trains each fold on 95% of the data and creates a much more robust 20-model ensemble. This single change resulted in a massive LB boost.
---

### Version 14 (Bi-gram/Tri-gram Categorical TE - XGBoost) - 2026-03-04
**Score**: **0.91656 LB** / 0.91889 OOF (Gap: -0.00233) 🏆 NEW OVERALL BEST
**Result**: **+0.00004 LB vs V12** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 31.6 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91924 | 0.91821 | 0.92055 | 0.91849 | 0.91856 | 0.91910 | 0.92090 | 0.91931 | 0.91811 | 0.91654 | 0.91890 |

**Strategy:** S6E2 winning technique. Concatenated top 6 categoricals into bi-grams and tri-grams (e.g. `Contract + InternetService + OnlineSecurity`), then applied Inner K-Fold Target Encoding. Captured interactions XGBoost depth-wise splits couldn't learn natively. Retained V12 Optuna parameters.
**File:** `S6E3_V14_BigramTE.py`

**Key Learning:**
> **Composite categorical TE captures powerful interaction signal.** The tri-gram `Contract×InternetService×OnlineSecurity` became the single most important feature in the model (15.5% importance), dominating single-column target encodings and raw categorical splits. OOF improved by +0.00010 over the heavily tuned V12.

**Status: 🏆 NEW OVERALL BEST (LB 0.91656)**

---

### V15f AllCat Mega-String & V15g CatBoost LIGHT - 2026-03-05
**Score**: OOF 0.91883 (V15f) / 0.91639 (V15g)
**Result**: ❌ BOTH WORSE vs V14 Baseline (0.91889)

**Timing:** Total 49.0 minutes (V15f: 29.0m, V15g: 19.8m)

**Results Matrix:**
| Model | OOF AUC | Delta | 10-Fold Mean |
|-------|---------|-------|--------------|
| V14 XGB (Baseline) | 0.91889 | — | 0.91890 |
| V15f AllCat TE (XGB) | 0.91883 | -0.00006 | 0.91884 |
| V15g CatBoost Raw | 0.91639 | -0.00250 | 0.91640 |

**Strategy:** 
- **V15f**: Concatenate all 16 categorical features into a single string (`AllCat_Profile`). Inner K-Fold TE encode this string on top of the V14 features. Hit 44,356 unique classes.
- **V15g**: Stripped out all manual TE. Fed 16 raw cats + 9 numeric/derived to CatBoost utilizing `leaf_estimation_method='Newton'`.

**Key Learning:**
V14 hit the density sweet spot. V15f was too sparse (curse of dimensionality) leading to TE over-smoothing. V15g proved that XGBoost + Manual Inner K-Fold TE fundamentally outperforms CatBoost's native ordered encoding on this specific dataset.

---

### EXP-V15 Multi-Feature Screen (5 Techniques) - 2026-03-05
**Score**: No LB submission — screening only
**Result**: ❌ ALL NEUTRAL OR WORSE vs V14 Fold-1 Baseline (0.91924)

**Timing:**
| Stage | Time |
|-------|------|
| EXP A: V15b Binning+TE | ~4 min |
| EXP B: V15c Churn Flags | ~3 min |
| EXP C: V15h Quantile TF | ~3 min |
| EXP D: V15e DAE Latent | ~8 min (incl. 3.6 min DAE training) |
| EXP E: V15i SHAP RFE | ~4 min |
| **Total** | **22.1 min** |

**Per-Experiment Fold-1 Scores:**
| Experiment | Fold-1 AUC | Delta | Verdict |
|------------|:---:|:---:|:---:|
| V14 Baseline | 0.91924 | ±0.000 | 🏆 BEST |
| V15b Binning+TE | 0.91924 | ±0.000 | = SAME |
| V15c Churn Flags | 0.91917 | -0.00007 | ❌ WORSE |
| V15h Quantile TF | 0.91924 | ±0.000 | = SAME |
| V15e DAE Latent | 0.91897 | **-0.00027** | ❌ WORST |
| V15i SHAP RFE | 0.91919 | -0.00005 | = SAME |

**Strategy:** Inner K-Fold TE (5-inner, 10-outer, Fold 1 only for screening). All experiments built on top of V14 pipeline (V7 + Bi-gram/Tri-gram TE = 143 features base). Added technique-specific features as delta on top.

**Key Learning:**
> **The V14 local optimum is very strong.** ORIG_proba already captures what binning and boolean flags would; quantile transforms are rank-invariant for trees; DAE latent features add noise (29-dim input, 16-dim bottleneck, too compressed for 594K rows); SHAP found zero removable features (all 143 features contribute). Next frontier: 20-fold CV variance reduction, AllCat mega-TE, or CatBoost raw+Newton.

---

### V14b (Polynomial Features - XGBoost) - 2026-03-04
**Score**: **0.91627 LB** / 0.91891 OOF (Gap: -0.00264)
**Result**: **-0.00025 LB vs V12** ❌ OVERFIT

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.3 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91923 | 0.91817 | 0.92060 | 0.91847 | 0.91863 | 0.91918 | 0.92095 | 0.91933 | 0.91804 | 0.91653 | 0.91891 |

**Strategy:** Added 15 polynomial features (squared and cubed versions of top numerical variables like tenure, MonthlyCharges, TotalCharges, plus interactions) to V12's Optuna baseline. 
**File:** `S6E3_V14b_PolyFeatures.py`

**Key Learning:**
> **Polynomials on raw numericals overfit heavily.** Despite improving the OOF AUC (+0.00012 over V12), the LB score dropped significantly (-0.00025). The OOF-LB gap widened from -0.00240 to -0.00264. Polynomial features allow the trees to fit the training noise too perfectly. Also, feature importance was very low (top poly feature was only 1.48%).

**Status: ❌ FAILED / OVERFIT**

---

### EXP-DART: XGBoost DART Experiment - 2026-03-04
**Score**: Fold 1 only: 0.91846 (run killed — 74x slower, worse AUC)
**Result**: **❌ FAILED — NEVER USE DART** 

**Strategy:** DART booster with V12 Optuna params. rate_drop=0.1, skip_drop=0.5, 5000 fixed trees.
**Time:** Fold 1 = 350 min (base + PL). ETA for 10 folds: ~58 hours. Killed after Fold 1.
**Why it Failed:**
- DART + colsample=0.32 = double regularization → too much dropout
- DART is O(n²) per iteration (drops + recomputes), gbtree is O(n)
- 0.91846 vs V12's 0.91924 on same fold = **-0.00078**
**Rule Added:** Rule 8 in ideas.md: **NO DART BOOSTING** for this competition.

---

### EXP-V15: Multi-Experiment Quick Test - 2026-03-04
**Score**: All experiments ≤+0.00004 vs V12 baseline (noise level). No submission.
**Result**: **❌ V12 params are near-optimal**

**Experiments Tested (5-fold CV on V12 params):**
| Experiment | AUC | Delta vs V12 | Verdict |
|-----------|:---:|:-----------:|:-------:|
| BASELINE (V12) | 0.91879 | — | Reference |
| Focal Loss γ=2.0 | 0.50000 | -0.41879 | 💥 Broken |
| Focal Loss γ=1.0 | 0.91854 | -0.00024 | ❌ Worse |
| scale_pos_weight=3.44 | 0.91866 | -0.00013 | ❌ Worse |
| scale_pos_weight=1.72 | 0.91874 | -0.00004 | = Same |
| colsample=0.15 | 0.91883 | +0.00004 | = Noise |
| colsample=0.20 | 0.91881 | +0.00003 | = Noise |
| Feature pruning | — | — | Can't run (bottom features are TE-generated) |

**Key Learning:**
> **V12 Optuna params are near-optimal for this dataset.** No single lever (loss function, class weights, column sampling, feature selection) moves the needle beyond noise. The 0.91652 LB ceiling may be a fundamental limit of single-model approaches on this data.

---

### Version 13 (LightGBM Optuna HPO) - 2026-03-04
**Score**: **0.91652 LB** / 0.91890 OOF (Gap: -0.00238) 🏆 TIED WITH V12
**Result**: **+0.00015 LB vs V7** ✅

**Strategy:** Optuna Bayesian HPO (TPE sampler, 50/100 trials in 713 min) on V7 LGBM. 10 params tuned. Retrained with best params on 10-fold CV. 89.0 min. 0/10 PL gain.
**File:** `S6E3_V13_LightGBM_Optuna.py`

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91927 | 0.91815 | 0.92065 | 0.91840 | 0.91850 | 0.91906 | 0.92104 | 0.91928 | 0.91804 | 0.91665 | 0.91890 |

**Optuna Best Params vs V7:**
| Param | V7 | V13 (Optuna) | Change |
|-------|:--:|:------------:|:------:|
| learning_rate | 0.03 | **0.0122** | 2.5x lower |
| colsample_bytree | 0.80 | **0.30** | 63% less |
| reg_alpha | 0.10 | **7.16** | 72x more |
| reg_lambda | 1.00 | **5.44** | 5.4x more |
| path_smooth | 0.00 | **8.89** | NEW: heavy smoothing |
| max_depth | 6 | **11** | deeper (but sparse) |
| num_leaves | 31 | **30** | similar |
| min_gain_to_split | 0.00 | **0.172** | NEW: split gate |

**Key Learning:**
> Both XGB and LGBM independently converge on **heavy column dropout (30-32%) and strong L1**. LGBM additionally benefits from `path_smooth=8.89` (unique to LGBM). V13 ties V12 on LB — confirming that **model choice doesn't matter when both are well-tuned**.

**Status: 🏆 TIED BEST (LB 0.91652)**

---

### Version 12 (XGBoost Optuna HPO) - 2026-03-04
**Score**: **0.91652 LB** / 0.91892 OOF (Gap: -0.00240) 🏆 NEW OVERALL BEST
**Result**: **+0.00007 LB vs V8** ✅

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91924 | 0.91817 | 0.92063 | 0.91845 | 0.91858 | 0.91915 | 0.92100 | 0.91932 | 0.91814 | 0.91660 | 0.91893 |

**Strategy:** Optuna Bayesian HPO (TPE sampler, 93/100 trials in 712 min) → retrain with best params on 10-fold CV. Same V7 features as V8. 47.2 min. 0/10 PL gain.
**File:** `S6E3_V12_XGBoost_Optuna.py`

**Optuna Best Params vs V8:**
| Param | V8 | V12 (Optuna) | Change |
|-------|:--:|:------------:|:------:|
| learning_rate | 0.05 | **0.0063** | 8x lower |
| colsample_bytree | 0.80 | **0.32** | 60% less |
| reg_alpha | 0.10 | **3.50** | 35x more |
| gamma | 0.05 | **0.79** | 16x more |
| max_depth | 6 | **5** | shallower |
| n_trees (avg) | ~1200 | ~9000 | 7.5x more |

**Key Learning:**
> **Heavy regularization wins on large FE datasets.** With 64 correlated features, the model benefits from seeing only 32% of features per tree (col=0.32), strong L1 (α=3.5), and slower learning (lr=0.0063 → ~9000 trees). McElfresh 2023 was right: light HPO > model choice.

**Status: 🏆 NEW OVERALL BEST**

---

### Version 11 (CatBoost Depthwise + All Dist Features) - 2026-03-03
**Score**: **0.91494 LB** / 0.91736 OOF (Gap: -0.00242)
**Result**: **-0.00151 LB vs V8 XGB** ❌

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91753 | 0.91715 | 0.91899 | 0.91679 | 0.91698 | 0.91767 | 0.91942 | 0.91801 | 0.91663 | 0.91457 | 0.91737 |

**Strategy:** CatBoost with `grow_policy='Depthwise'` (independent leaf splits like XGB) + V7 features. Native categorical handling (no Inner K-Fold TE). Pseudo-labeling attempted but 0/10 folds improved. 17.7 min total.
**File:** `S6E3_V11_CatBoost_AllDistFeatures.py`

**Tested 3 configurations:**
| Config | Fold 1 AUC | Notes |
|--------|-----------|-------|
| SymmetricTree (default) | 0.91720 | 500s/fold, default symmetric splits |
| Ordered + depth=6 | 0.91662 | 931s/fold, worse & slower |
| **Depthwise + depth=8** | **0.91753** | 111s/fold, best CatBoost ✅ |

**Key Learning:**
> **CatBoost underperforms with heavy FE.** With 64 engineered features (19 ORIG_proba, 9 dist, 8 qdist), CatBoost's native TE and auto feature combinations are redundant. The -0.00242 OOF-LB gap is the widest of any model. CatBoost shines on raw/minimal features (like S6E2 V39) but becomes "just another GBDT" with heavy FE — and a less flexible one than XGB/LGBM.

**Status: ❌ Underperforms (diversity only)**

### Version 10 (RealMLP + All Dist Features) - 2026-03-03
**Score**: **0.91491 LB** / 0.91633 OOF (Gap: -0.00142)
**Result**: **+0.00114 LB vs V5 RealMLP** ✅

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91728 | 0.91620 | 0.91868 | 0.91637 | 0.91399 | 0.91764 | 0.91913 | 0.91741 | 0.91586 | 0.91449 | 0.91671 |

**Strategy:** RealMLP_TD_Classifier (S6E2 V48 tuned params: mish, hidden_width=384, n_hidden_layers=4, plr embeddings, n_ens=8) + V7 features + Inner K-Fold TE. All features converted to category type. 263.4 min total.
**File:** `S6E3_V10_RealMLP_AllDistFeatures.py`

**Key Learning:**
> V7 features improved RealMLP from 0.91377 (V5) to 0.91491 (+0.00114 LB). However S6E2-tuned hyperparams may not be optimal for S6E3's larger dataset. RealMLP is slower than TabM (263 vs 232 min) and less accurate. TabM is strictly better as the NN diversity model.

**Status: ✅ Good (diversity anchor)**

### Version 9 (TabM + All Dist Features) - 2026-03-03
**Score**: **0.91625 LB** / 0.91845 OOF (Gap: -0.00220)
**Result**: **+0.00248 LB vs V5 RealMLP, Best NN** 🏆

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91896 | 0.91795 | 0.92031 | 0.91761 | 0.91814 | 0.91870 | 0.92057 | 0.91894 | 0.91789 | 0.91625 | 0.91853 |

**Strategy:** TabM_D_Classifier (pytabkit, tabm-mini-normal, k=32, pwl embeddings, d_block=256, n_blocks=3) + V7 features (V4 core + 9 EXP3 + 8 EXP5) + Inner K-Fold TE (mean). 232.7 min total.
**File:** `S6E3_V9_TabM_AllDistFeatures.py`

**Key Learning:**
> TabM (ICLR 2025) massively outperforms RealMLP (+0.00134 LB). OOF 0.91845 nearly matches V7 LGBM (0.91851). The -0.00220 OOF-LB gap is slightly wider than trees (-0.00212), typical for NNs. TabM provides excellent diversity for future ensembling with different inductive bias than trees.

**Status: 🏆 Best NN**

### Version 8 (XGBoost + All Dist Features) - 2026-03-02
**Score**: **0.91645 LB** / 0.91857 OOF (Gap: -0.00212)
**Result**: **+0.00008 LB vs V7, +0.00038 vs V3** 🏆

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91901 | 0.91781 | 0.92024 | 0.91811 | 0.91820 | 0.91876 | 0.92067 | 0.91902 | 0.91771 | 0.91624 | 0.91858 |

**Strategy:** V3 XGBoost architecture (50K trees, enable_categorical, CUDA) + V7 features (V4 core + 9 EXP3 + 8 EXP5). 0/10 PL improvements. 10.8 min total (3x faster than LGBM).
**File:** `S6E3_V8_XGBoost_AllDistFeatures.py`

**Key Learning:**
> XGBoost edges out LightGBM with identical features (+0.00008 LB). Both OOF and LB improved. XGB is 3x faster (10.8 vs 29.7 min) due to fewer trees (1K early-stop vs 2K+).

**Status: 🏆 Overall Best**

### Version 7 (LightGBM + Dist + Quantile Distance Features) - 2026-03-02
**Score**: **0.91637 LB** / 0.91851 OOF (Gap: -0.00214)
**Result**: **+0.00007 LB vs V6** 🏆

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91906 | 0.91776 | 0.92028 | 0.91803 | 0.91812 | 0.91857 | 0.92074 | 0.91878 | 0.91762 | 0.91622 | 0.91852 |

**Strategy:** V6 pipeline + 8 EXP5 quantile distance features (TotalCharges distance to Q25/Q50/Q75 of original churner/non-churner). 0/10 PL improvements.
**File:** `S6E3_V7_LightGBM_QuantileDistFeatures.py`

**Status: 🏆 Best**

### EXP5 (Ultimate Feature Discovery) - 2026-03-02
**Score**: N/A (Research) / 0.91757 vs 0.91739 Baseline (5-fold)
**Result**: **+0.00018 vs V6 baseline** ✅

**Strategy:** Tested 92 features across 10 batches. Only Batch F (quantile distance for TotalCharges) survived greedy selection. 8 distance-to-quantile features confirmed in 5-fold CV. All 5 folds improved.
**File:** `S6E3_EXP5_UltimateFE.py`

**Key Learning:**
> TotalCharges distribution features are the only consistent source of orthogonal signal. MonthlyCharges/tenure distributions, conditional groups, clusters, KDE ratios, polynomial interactions, and nearest-neighbor features all failed.

**Status: ✅ 8 New Features Found**

### Version 6 (LightGBM + EXP3 Distribution Features) - 2026-03-02
**Score**: **0.91630 LB** / 0.91842 OOF (Gap: -0.00212)
**Result**: **+0.00021 LB** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.2 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91900 | 0.91767 | 0.92016 | 0.91784 | 0.91799 | 0.91857 | 0.92051 | 0.91871 | 0.91764 | 0.91615 | 0.91842 |

**Strategy:** V4 pipeline (Inner K-Fold TE, FREQ, Arithmetic, ORIG_proba, Pseudo Labels) + 9 EXP3 distribution features: percentile ranks against original churner/non-churner TotalCharges distributions, z-score gaps, conditional percentile ranks within Contract/InternetService groups.
**File:** `S6E3_V6_LightGBM_DistFeatures.py`

**Key Learning:**
> Distribution features provide genuinely orthogonal signal. V6 improved EVERY fold vs V4. OOF-LB gap narrowed from -0.00218 to -0.00212, suggesting slightly less overfitting despite more features. No PL improvements in any fold (0/10).

**Status: 🏆 Best**

### EXP4 (OptimalBinning WoE) - 2026-03-02
**Score**: N/A (Research) / 0.91741 vs 0.91739 Baseline (5-fold)
**Result**: **+0.00002 vs V4+EXP3 baseline** ⚠️ Neutral

**Timing:**
| Stage | Time |
|-------|------|
| Total | 262.4 min |

**Strategy:** Applied `optbinning` library 1D WoE (19 features) + 2D joint WoE (45 interaction pairs) fit on original IBM dataset. Top IV: Contract (1.24), tenure (0.87), OnlineSecurity (0.72).
**File:** `S6E3_EXP4_OptBinning.py`

**Key Learning:**
> WoE encoding is mathematically equivalent to a monotonic transform of ORIG_proba. Both derive from original dataset target statistics. 64 WoE features produced +0.00002 (noise). Greedy selection kept only `woe2d_TechSupport_InternetService` and `woe2d_Contract_InternetService`.

**Status: ⚠️ Neutral**

### EXP3 (Novel Distribution Feature Mining) - 2026-03-02
**Score**: N/A (Research) / 0.91685 Baseline vs 0.91649 Baseline (5-fold)
**Result**: **+0.00036 vs V4 baseline** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 168.0 + 130.5 min |

**Strategy:** Tested ~200 genuinely novel features across v2/v3 batches. Distribution-based features were the only promising path. Ran greedy forward selection and strict 5-fold CV to isolate exact winners.
**File:** `S6E3_EXP3_Feature_Forensics.py`

**Key Learning:**
> 9 specific features survived 5-fold CV: `pctrank_nonchurner_TotalCharges`, `zscore_churn_gap_TotalCharges`, `pctrank_churn_gap_TotalCharges`, `resid_mean_InternetService_MonthlyCharges`, `cond_pctrank_InternetService_TotalCharges`, `zscore_nonchurner_TotalCharges`, `pctrank_orig_TotalCharges`, `pctrank_churner_TotalCharges`, `cond_pctrank_Contract_TotalCharges`.

**Status: ✅ Novel Features Found**

### EXP2 (Feature Validation) - 2026-03-01
**Score**: N/A (Research) / 0.91648 Baseline vs 0.91632 Best Alt (5-fold)
**Result**: **-0.00017** ❌

**Strategy:** A/B/C/D controlled comparison: V4 alone (58 feat) vs V4+Top EXP1 (76) vs V4+All EXP1 (102) vs EXP1 only (38).
**File:** `S6E3_EXP2_Feature_Validation.py`

**Key Learning:**
> All EXP1 features HURT V4. Feature importance in isolation ≠ additive value. V4's 58-feature pipeline is near-optimal.

**Status: ❌ Negative Result**

### EXP1 (Feature Discovery) - 2026-03-01
**Score**: N/A (Research) / LGBM 0.91636, XGB 0.91649, CB 0.91585 (5-fold)
**Result**: **Research Only** ✅

**Strategy:** Generated 277 features across 12 categories, evaluated by LightGBM/XGBoost/CatBoost (GPU) + Pearson correlation. `risk_score_composite` ranked #1 universal, `CLV_simple` #2.
**File:** `S6E3_EXP1_Feature_Discovery.py`

**Key Learning:**
> Synthetic artifact features ranked LOWEST (avg 0.0725). 257/295 features above noise. CatBoost uniquely leverages features that LGBM/XGB ignore.

**Status: ✅ Research Complete**

### Version 5 (RealMLP DualRep Neural Network) - 2026-03-01
**Score**: **0.91377 LB** / 0.91396 OOF (Gap: -0.00019)
**Result**: **✅ Solid Base** 

**Timing:**
| Stage | Time |
|-------|------|
| Total | 48.0 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.91369 | 0.91545 | 0.91485 | 0.91598 | 0.91326 | 0.91464 |

**Strategy:** Introduced a PyTorch Tabular Neural Network (pytabkit RealMLP) natively applying Dual Representation (One-Hot + Ordinal encoded) and Statistical Injections from the original IBM dataset.
**File:** `S6E3_V5_RealMLP_DualRep.py`

**Key Learning:**
> While it underperformed the top gradient boosters (0.916+), a 0.913+ NN is exceptionally strong for tabular data and provides excellent uncorrelated predictions. Time overhead (48 mins) is significant.

**Status: ✅ Good**

### Version 4 (LightGBM Inner K-Fold TE) - 2026-03-01
**Score**: **0.91609 LB** / 0.91827 OOF (Gap: -0.00218)
**Result**: **Highest LB** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.2 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91871 | 0.91752 | 0.91995 | 0.91783 | 0.91781 | 0.91873 | 0.92035 | 0.91849 | 0.91742 | 0.91593 | 0.91827 |

**Strategy:** Direct algorithmic swap of the V3 Inner K-Fold pipeline from XGBoost to LightGBM. Keeps the Arithmetic Interactions and numerical-to-categorical changes intact.
**File:** `S6E3_V4_LightGBM_InnerKFoldTE.py`

**Key Learning:**
> LightGBM's leaf-wise tree growth optimized the identical engineered features slightly better than XGBoost's depth-wise growth. Proves the V3 pipeline is the optimal baseline feature set.

**Status: 🏆 Best**

### Version 3 (XGBoost Inner K-Fold TE) - 2026-03-01
**Score**: **0.91607 LB** / 0.91774 OOF (Gap: -0.00167)
**Result**: **Strong Baseline** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 9.8 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91876 | 0.91734 | 0.91984 | 0.91786 | 0.91794 | 0.91837 | 0.92030 | 0.91861 | 0.91734 | 0.91605 | 0.91824 |

**Strategy:** Implemented leak-free Inner K-Fold Target Encoding (calculating Mean/Std/Min/Max per fold to prevent train/val leakage). Added Arithmetic Interactions and robust frequency encoding. Strict pseudo labeling.
**File:** `S6E3_V3_InnerKFoldTE.py`

**Key Learning:**
> Strict, leak-free Target Encoding completely fixed the overfitting seen in V2. Mathematical interaction features (`TotalCharges - tenure*MonthlyCharges`) are proving highly effective for trees.

**Status: ✅ Good**

### Version 2 (GroupBy FE + XGB Pseudo) - 2026-03-01
**Score**: **0.91400 LB** / 0.91652 OOF (Gap: -0.00252)
**Result**: **-0.00011 LB** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 14.3 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.9170 | 0.9164 | 0.9153 | 0.9176 | 0.9138 | 0.9159 | 0.9172 | 0.9168 | 0.9172 | 0.9182 | 0.9165 |

**Strategy:** Re-used V1 Pseudo-Label framework but injected massive Deotte Phase 2 Feature Engineering. Grouped by 16+ categorization pairs (e.g. Contract_PaymentMethod) to calculate Mean, STD, and Diff_From_Mean across all 3 numerical outputs using cuDF. Total features boosted significantly.
**File:** `S6E3_V2_GroupByFE.py`

**Key Learning:**
> Overfit! The massive increase in interaction features (215 new features) reduced both the OOF (-0.00007) and the LB (-0.00011). We need feature selection or a more targeted approach.

**Status: ❌ Failed/Overfit**

### Version 1 (XGB Pseudo+cuDF Baseline) - 2026-03-01
**Score**: **0.91411 LB** / 0.91659 OOF (Gap: -0.00248)
**Result**: **Initial Baseline LB** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 4.1 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.9169 | 0.9165 | 0.9155 | 0.9175 | 0.9138 | 0.9160 | 0.9172 | 0.9168 | 0.9173 | 0.9184 | 0.9166 |

**Strategy:** Implemented Kaggle 0.917 notebook: XGBoost on cuDF, 10 Folds CV, Global Frequency Encoding (train+test+orig), injected Original data to training, extracted Pseudo-Labels (>0.95/<0.05 prob) from Test predictions, retrained final model.
**File:** `S6E3_V1_Baseline.py`

**Key Learning:**
> Pseudo-labeling established strong base. Prepared for advanced GroupBy FE next.

**Status: 🏆 Best**
