# S6E3 Daily Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed OR experiment OOF available
> 2. **DO NOT EDIT** previous day's entries
> 3. **PREPEND** new days (latest first)
> 4. **Include:** Experiments run, Timing, Key learnings
> 5. **Status icons:** 🏆 Best | ✅ Success | ⚠️ Partial | ❌ Failed

---

### March 08, 2026

**28. S6E3 V20 LightGBM Optuna — ⚠️ WORSE (LB 0.91661)**
   - **Goal:** Apply Optuna-optimized hyperparameters to LightGBM with V16 feature set (Digit Features + Bi-gram/Tri-gram TE).
   - **Outcome:** OOF **0.91908** (±0.00170 std) / LB **0.91661** (-0.00019 vs V16b). 151.9 min. 20-Fold CV.
   - **Insight:** LightGBM with Optuna HPO improves over V19 CatBoost (+0.00013) but still cannot match XGBoost V16b. The leaf-wise growth doesn't provide an advantage over depth-wise XGBoost on this heavy FE dataset. XGBoost remains the best single model.

**27. S6E3 V19 CatBoost Optuna — ⚠️ WORSE (LB 0.91648)**
   - **Goal:** Apply Optuna-optimized hyperparameters to CatBoost with V16 feature set (Digit Features + Bi-gram/Tri-gram TE).
   - **Outcome:** OOF **0.91900** (±0.00178 std) / LB **0.91648** (-0.00032 vs V16b). 49.1 min. 20-Fold CV.
   - **Insight:** Even with Optuna HPO, CatBoost cannot match XGBoost V16b on heavy FE datasets. The symmetric tree architecture limits its ability to leverage complex digit-feature interactions. However, V19 improved over V18 by +0.00008 using the full V16 feature pipeline.

---

### March 07, 2026 (Continued)

**26. S6E3 V18 CatBoost + Digit Features — ❌ WORSE (LB 0.91640)**
   - **Goal:** Test if CatBoost can leverage V16's 46 digit features (modulo, rounding) with same pipeline as XGBoost.
   - **Outcome:** OOF **0.91892** / LB **0.91640** (-0.00040 vs V16b). 29.8 min.
   - **Insight:** CatBoost's symmetric tree architecture cannot leverage digit features as effectively as XGBoost. tenure_rounded_10 was #1 feature (2.19%), but structural limitation prevents optimal utilization. Confirms CatBoost is not suitable for heavy FE datasets.

**24. S6E3 V18 CatBoost Residual Learning — ❌ NEUTRAL (OOF 0.91925)**
   - **Goal:** Use CatBoostClassifier with baseline parameter to sequentially boost on V16b XGBoost margins (logits).
   - **Outcome:** OOF **0.91925** (±0.00000 vs V16b baseline). 14.6 min.
   - **Insight:** CatBoost early-stopped at iteration 0 on ALL 10 folds. Could not find any orthogonal splits. Predictions are 100% correlated with V16b (correlation = 1.00000). Sequential boosting requires weak spots that V16b no longer has.

**25. S6E3 V19 RGF (Regularized Greedy Forest) — ❌ FAILED (Killed)**
   - **Goal:** Test RGFClassifier as diversity model on V16 feature set.
   - **Outcome:** Fold 1 AUC 0.91864 (-0.00199 vs V16b), Fold 2 AUC 0.91778. Killed after Fold 2 due to time (130+ min per fold, 10-fold ETA: 21+ hours).
   - **Insight:** RGF is catastrophically slow and worse in AUC than XGBoost. Not viable for this competition at this scale.

**6. S6E3 EXP4 (CatBoost Sequential Baseline Boosting) — ❌ NEUTRAL (OOF 0.91925)**
   - **Goal:** Inject V16b XGBoost predictions as a log-odds `baseline` into CatBoostClassifier. Aimed to let CatBoost's native categorical handling find orthogonal splits from XGBoost's local minimum.
   - **Outcome:** OOF **0.91925** (±0.00000 vs V16b baseline).
   - **Insight:** CatBoost immediately early-stopped at Iteration 0 on every fold. It could not find a single split that improved the XGBoost Logloss. V16b has officially saturated 100% of the available feature signal. Sequential boosting requires orthogonal weak spots, which V16b no longer has.

**5. S6E3 EXP3 (Label Smoothing Regularization) — ❌ WORSE (OOF 0.91909)**
   - **Goal:** Soften Kaggle's synthetic binary targets (1 -> 0.975, 0 -> 0.025) to prevent tree models from overfitting on edge-case noisy boundaries.
   - **Outcome:** OOF **0.91909** (-0.00008 vs V16 baseline). 35.0 min.
   - **Insight:** Forcing XGBoost to build fuzzy leaf structures prevented it from capturing the exact micro-signals required by the Kaggle synthetic generation process. Hard targets are necessary.

---

# **ENTRIES FROM BELOW THIS TEXT ARE NOT TO BE ALTERED**

### March 06, 2026

**23. S6E3 V16 (Digit Features from Numericals) — 🏆 NEW SINGLE MODEL BEST (LB 0.91679)**
   - **Goal:** Inject arithmetic extraction of the string structure of numericals (`tenure % 10`, `TotalCharges string length`, Benford's law leading digits).
   - **Outcome:** OOF **0.91917** / LB **0.91679** (+0.00023 vs V14 base). 38.0 min.
   - **Insight:** Trees cannot split cleanly on geometric concepts like "divisible by 12". Providing these manually (`tenure_years`, `tenure_rounded_10`, `tenure_num_digits`) exposes synthetic artifacts the model heavily relies on.

**22. S6E3 V15 (V14 20-Fold CV) — 🏆 NEW OVERALL BEST (LB 0.91657)**
   - **Goal:** Run the best V14 Bi-gram/Tri-gram TE pipeline with 20-fold CV instead of 10-fold to reduce variance and improve the ensemble.
   - **Outcome:** OOF **0.91897** (+0.00008) / LB **0.91657** (+0.00001 vs V14). 69.2 min.
   - **Insight:** 20-fold CV provides a tiny edge by bleeding less training data away from the model. 

---

### March 05, 2026

**19. EXP-V15 Multi-Feature Screening (5 techniques, 1-fold each) — ❌ ALL NEUTRAL/WORSE**
   - **Goal:** Screen 5 Phase-11 techniques (Binning+TE, Churn Flags, Quantile TF, DAE, SHAP RFE) against V14 Fold-1 baseline (0.91924).
   - **Outcome:** No improvement. V15b Binning and V15h Quantile TF were SAME (±0.00000). V15c Churn Flags -0.00007. V15e DAE -0.00027 (worst). V15i SHAP RFE -0.00005. No LB submission. Total: 22.1 min.
   - **Insight:** V14 with Bi-gram/Tri-gram TE has reached a local FE optimum. Remaining standard tricks are redundant with existing ORIG_proba + categorical TE encodings. Trees are also rank-invariant so quantile transforms add nothing. DAE latent features are harmful on this dataset.

**20. V15 TabR (ICLR 2024) — ❌ KILLED (Not Viable at 594K rows)**
   - **Goal:** Official TabR implementation (FAISS top-k retrieval + label encoder + T-transform) on V14's 143 TE-encoded features.
   - **Outcome:** Killed at Fold 1 Epoch 5. Best AUC 0.79934 (vs V14's 0.91924). ~6 min/epoch → estimated **20 hours** for full 10-fold. Kaggle limit is 9 hours.
   - **Insight:** TabR requires the entire training set (534K rows) as FAISS candidates every batch → O(N) per step. Designed for sub-100K datasets. PERMANENTLY DEAD for this competition at this scale.

**21. V15f AllCat TE & V15g CatBoost LIGHT — ❌ BOTH WORSE**
   - **Goal:** Test opposite extremes: V15f created one massive 16-category profile string for Inner K-Fold TE (XGB). V15g used 0 manual TE, relying solely on CatBoost native ordered TE + Newton Step.
   - **Outcome:** V15f AUC 0.91883 (-0.00006 vs V14). V15g AUC 0.91639 (-0.00250 vs V14). No new LB submit.
   - **Insight:** V14's Bi/Tri-grams hit the "Goldilocks zone". V15f's 16-way string created 44,356 unique profiles (too sparse, smoothed away). V15g proved CatBoost's internal TE is far weaker than our manual cross-fold `std`/`min`/`max` TE on XGBoost.

### March 07, 2026

**6. S6E3 EXP4 (CatBoost Sequential Baseline Boosting) — ❌ NEUTRAL (OOF 0.91925)**
   - **Goal:** Inject V16b XGBoost predictions as a log-odds `baseline` into CatBoostClassifier. Aimed to let CatBoost's native categorical handling find orthogonal splits from XGBoost's local minimum.
   - **Outcome:** OOF **0.91925** (±0.00000 vs V16b baseline).
   - **Insight:** CatBoost immediately early-stopped at Iteration 0 on every fold. It could not find a single split that improved the XGBoost Logloss. V16b has officially saturated 100% of the available feature signal. Sequential boosting requires orthogonal weak spots, which V16b no longer has.

**5. S6E3 EXP3 (Label Smoothing Regularization) — ❌ WORSE (OOF 0.91909)**
   - **Goal:** Soften Kaggle's synthetic binary targets (1 -> 0.975, 0 -> 0.025) to prevent tree models from overfitting on edge-case noisy boundaries.
   - **Outcome:** OOF **0.91909** (-0.00008 vs V16 baseline). 35.0 min.
   - **Insight:** Forcing XGBoost to build fuzzy leaf structures prevented it from capturing the exact micro-signals required by the Kaggle synthetic generation process. Hard targets are necessary.


**4. S6E3 EXP-V17c (Monotonic Constraints) — ⚠️ SKIPPED (OOF 0.91915)**
   - **Goal:** Hardcode `-1` monotonic constraints on `tenure` and `TotalCharges` inside XGBoost to force domain logic and prevent noisy splits.
   - **Outcome:** OOF **0.91915** (-0.00002 vs V16 baseline).
   - **Insight:** The base V12 XGBoost parameters are already extremely heavily tuned to combat overfit (Gamma 0.79, reg_alpha 3.5). Adding physical hard constraints on top prevents the tree from capturing genuine micro-signals.

**3. S6E3 EXP-V17b (Multi-Target TE) — ⚠️ SKIPPED (OOF 0.91918)**
   - **Goal:** Encode standard categoricals against 5 demographic sub-targets (e.g., Dependents) from the original dataset instead of encoding against Churn.
   - **Outcome:** OOF **0.91918** (+0.00001 vs V16 baseline).
   - **Insight:** Predicting other demographic variables per group just creates another highly correlated proxy for predicting Churn. No orthogonal signal was extracted.

**2. S6E3 EXP-V17 (Round/Binning + TE) — ⚠️ SKIPPED (OOF 0.91916)**
   - **Goal:** Discretize continuous columns (`tenure`, `MonthlyCharges`) into granular bins ($10 blocks, 3-mo blocks) and apply targeting encoding to extract time/price correlations.
   - **Outcome:** OOF **0.91916** (-0.00001 vs V16).
   - **Insight:** Trees inherently discretize numeric data via splits. Creating hard manual bins (even with interaction dimensions) proved redundant to the existing `ORIG_proba` probability mappings.

**1. S6E3 V16b (20-Fold Re-run of V16) — 🏆 NEW OVERALL BEST (LB 0.91680)**
   - **Goal:** Squeeze the final micro-percentile of efficiency out of our best baseline (V16) by running 20 Folds instead of 10.
   - **Outcome:** OOF **0.91925** (+0.00008 vs V16) / LB **0.91680** (+0.00001 vs V16). 80.0 min total training time.
   - **Insight:** 20 Folds provides a consistently tiny (~0.00001) but real lift because models get 95% of data per fold instead of 90%.

---

### March 06, 2026

**17. S6E3 V14 Submission (Bi-gram/Tri-gram TE) — 🏆 NEW OVERALL BEST (LB 0.91656)**
   - **Goal:** Apply S6E2 winning technique: inner K-Fold Target Encoding on concatenated composite categorical columns (bi-grams & tri-grams).
   - **Outcome:** OOF **0.91889** (+0.00010 vs V12) / LB **0.91656** (+0.00004 vs V12). 31.6 min.
   - **Insight:** Tri-grams dominated feature importance (`Contract×InternetService×OnlineSecurity` was #1). Composite categorical TE captures interactions trees struggle to learn cleanly through sequential splits alone.

**18. S6E3 V14b Polynomial Features (x², x³) — ❌ OVERFIT (LB 0.91627)**
   - **Goal:** Add 15 polynomial features (squares, cubes, interactions of top numericals) based on S5E12 winning solutions.
   - **Outcome:** OOF **0.91891** (+0.00012 vs V12) / LB **0.91627** (-0.00025 vs V12). Gap widened significantly to -0.00264.
   - **Insight:** Polynomials allow trees to fit training noise too perfectly on this dataset, artificially inflating OOF while tanking real generalization (LB). Top poly feature only had 1.48% importance.

---

### March 02, 2026

**1. S6E3 EXP3 v3 Deep Distribution Mining — ✅ SUCCESS (+0.00036)**
   - **Goal:** Aggressively mine distribution-based features (the only proven direction from EXP3 v2).
   - **Outcome:** 9 features survived strict 4-stage evaluation. V4+EXP3 = **0.91685** (5-fold) vs V4 alone 0.91649.
   - **Winners:** `pctrank_nonchurner_TotalCharges`, `zscore_churn_gap_TotalCharges`, `pctrank_churn_gap_TotalCharges`, `resid_mean_InternetService_MonthlyCharges`, `cond_pctrank_InternetService_TotalCharges`, + 4 more.

**2. S6E3 EXP4 OptimalBinning WoE — ⚠️ NEUTRAL (+0.00002)**
   - **Goal:** Test if `optbinning` library's 1D/2D WoE encoding adds signal on top of V4+EXP3.
   - **Outcome:** 64 WoE features tested (19 1D + 45 2D pairs). +0.00002 in 5-fold = noise.
   - **Insight:** WoE ≈ ORIG_proba (both encode target statistics from original). 2D interactions redundant because trees learn them natively.

**3. S6E3 V6 Submission — 🏆 NEW BEST LB (+0.00021)**
   - **Goal:** Submit V4 pipeline + 9 EXP3 distribution features.
   - **Outcome:** OOF **0.91842** (+0.00015 vs V4) / LB **0.91630** (+0.00021 vs V4). Gap narrowed -0.00218 → -0.00212.
   - **Insight:** Distribution features genuinely help on both OOF AND LB. Every fold improved. 0/10 PL improvements.

**4. S6E3 EXP5 Ultimate Feature Discovery — ✅ SUCCESS (+0.00018)**
   - **Goal:** Exhaustive search of 92 features across 10 new directions before moving to model diversity.
   - **Outcome:** Only Batch F (TotalCharges quantile distance) survived. 8 features confirmed +0.00018 in 5-fold.
   - **Dead ends confirmed:** MonthlyCharges/tenure distributions, conditional groups, 3-way conditionals, KDE density ratios, KMeans clusters, polynomial feature interactions, nearest-neighbor distance — all neutral or hurt.

**5. S6E3 V7 Submission — 🏆 NEW BEST LB (+0.00007 vs V6)**
   - **Goal:** Submit V6 pipeline + 8 EXP5 quantile distance features.
   - **Outcome:** OOF **0.91851** (+0.00009 vs V6) / LB **0.91637** (+0.00007 vs V6). 0/10 PL improvements.
   - **Running total:** V4 (0.91609) → V6 (+0.00021) → V7 (+0.00007) = **+0.00028 total LB gain from FE.**

**6. S6E3 V8 XGBoost Submission — 🏆 NEW OVERALL BEST (+0.00008 vs V7)**
   - **Goal:** XGBoost (V3 architecture) + V7 full feature set (17 dist features).
   - **Outcome:** OOF **0.91857** / LB **0.91645** (+0.00008 vs V7 LGBM, +0.00038 vs V3 XGB). 3x faster (10.8 min). 0/10 PL.
   - **Insight:** XGBoost slightly outperforms LGBM with identical features. Both algorithms benefit equally from distribution FE.

**7. Deep NN Research — TabM Selected as Best NN**
   - **Research:** 7 web searches, ICLR 2025 paper, TabM GitHub API, winning solutions from S4E1/S5E11/S5E12/S6E2.
   - **Finding:** TabM (Yandex, ICLR 2025) = parameter-efficient MLP ensemble using BatchEnsemble. Used by S5E11 5th, S5E12 4th.
   - **Also researched:** FT-Transformer, TabPFN v2 (too small for 594K rows), CatBoost (native ordered TE + auto feature combinations).
   - **Hidden tricks:** Multi-seed TabM, PiecewiseLinearEmbeddings, train k members independently, average probabilities not logits.

**8. S6E3 V9 TabM Submission — 🏆 BEST NN (LB 0.91625)**
   - **Goal:** TabM (ICLR 2025, BatchEnsemble MLP k=32) + V7 features. Different inductive bias for ensemble diversity.
   - **Outcome:** OOF **0.91845** / LB **0.91625** (-0.00020 vs V8 XGB). 232.7 min. Best NN model by far.
   - **Insight:** TabM OOF (0.91845) nearly matches LGBM V7 (0.91851). Massive +0.00248 LB over V5 RealMLP.

**9. S6E3 V10 RealMLP Submission — ✅ (LB 0.91491)**
   - **Goal:** RealMLP_TD (S6E2 V48 tuned params) + V7 features. Test if V7 features improve V5.
   - **Outcome:** OOF **0.91633** / LB **0.91491** (+0.00114 vs V5). 263.4 min. Slower and weaker than TabM.
   - **Insight:** V7 features helped RealMLP (+0.00114 LB) but TabM strictly dominates (+0.00134 LB faster).

**10. S6E3 V11 CatBoost Submission — ❌ Underperforms (LB 0.91494)**
   - **Goal:** CatBoost (Depthwise grow_policy) + V7 features. Test 3 configs: SymmetricTree, Ordered, Depthwise.
   - **Outcome:** OOF **0.91736** / LB **0.91494** (-0.00151 vs V8 XGB). 17.7 min. 0/10 PL gain.
   - **Insight:** CatBoost's native TE is redundant with our 64 engineered features. Heavy FE saturates CatBoost's advantage. CatBoost shines on raw features (S6E2 V39 was top-2), not on heavy FE datasets.

**11. S6E3 V12 Optuna HPO Search — Phase 1 (93/100 trials, 712 min)**
   - **Goal:** Bayesian hyperparameter optimization (TPE sampler) on V8 XGBoost. 100 trials × 5-fold CV.
   - **Outcome:** Best 5-fold AUC **0.91879** vs V8 baseline 0.91844 (+0.00035). Timed out at trial 93/100 (12h limit).
   - **Insight:** Optimal params: lr=0.0063, depth=5, col=0.32, α=3.5, γ=0.79. Heavy regularization needed for 64 correlated features.

**12. S6E3 V12 Optuna Submission — 🏆 NEW OVERALL BEST (LB 0.91652)**
   - **Goal:** Retrain with Optuna best params (hardcoded) on full 10-fold CV + Pseudo Labels.
   - **Outcome:** OOF **0.91892** / LB **0.91652** (+0.00007 vs V8). 47.2 min. 0/10 PL gain.
   - **Insight:** McElfresh 2023 confirmed: light HPO > model choice. +0.00035 OOF, +0.00007 LB from pure param tuning.

**13. S6E3 V13 LGBM Optuna Search — Phase 1 (50/100 trials, 713 min)**
   - **Goal:** Optuna HPO on V7 LGBM. 100 trials × 5-fold CV. 10 params (incl. path_smooth, min_gain_to_split).
   - **Outcome:** Best 5-fold AUC **0.91869** vs V7 baseline 0.91835 (+0.00034). Timed out at trial 50/100.
   - **Insight:** Same patterns as V12 XGB: col=0.30, heavy L1 (α=7.16), path_smooth=8.89 (LGBM-unique win).

**14. S6E3 V14 DART XGBoost — ❌ FAILED (too slow, worse AUC)**
   - **Goal:** DART booster with V12 Optuna params. rate_drop=0.1, skip_drop=0.5, 5000 trees.
   - **Outcome:** Fold 1 AUC **0.91846** (-0.00078 vs V12) in **350 min** (74x slower). 10-fold ETA: 58 hours.
   - **Insight:** DART + colsample=0.32 = double regularization → too much. DART also O(n²) per iteration.

**15. S6E3 V13 LGBM Optuna Retrain — 🏆 TIED WITH V12 (LB 0.91652)**
   - **Goal:** Retrain with Optuna best LGBM params on full 10-fold CV + PL.
   - **Outcome:** OOF **0.91890** / LB **0.91652** (+0.00015 vs V7, tied with V12 XGB). 89.0 min. 0/10 PL.
   - **Insight:** LGBM matches XGB when both are Optuna-tuned. Both converge on col=0.30-0.32, heavy L1.

**16. S6E3 V15 Multi-Experiment — ❌ ALL FAILED (V12 is near-optimal)**
   - **Goal:** Test 4 ideas on V12 params via 5-fold CV: Focal Loss, scale_pos_weight, colsample grid, feature pruning.
   - **Outcome:** Max gain: +0.00004 (noise). Focal Loss γ=2.0: AUC 0.50 (broken). γ=1.0: -0.00024. All SPW: worse. Colsample 0.15-0.50: all within ±0.00005 of 0.32.
   - **Insight:** V12 params are near-optimal. No single parameter lever improves beyond noise. Feature pruning couldn't run (bottom features are TE-generated, which are handled externally).

### March 01, 2026

**1. S6E3 EXP1 Feature Discovery (277 features, 12 categories)**
   - **Goal:** Generate every conceivable feature and rank by LightGBM/XGBoost/CatBoost gain + Pearson correlation.
   - **Outcome:** `risk_score_composite` (#1 universal), `CLV_simple` (#2), cross-interactions dominate trees.
   - **Insight:** Synthetic artifact features rank LOWEST (avg 0.0725). 257/295 features above random noise. Time: 7.9 min.

**2. S6E3 EXP2 Feature Validation — ❌ NEGATIVE RESULT**
   - **Goal:** Test if EXP1's top features actually improve V4 LightGBM baseline (0.91648 OOF).
   - **Outcome:** V4 alone (58 feats) = 0.91648 > V4+Top (76 feats) = 0.91632 > V4+All (102 feats) = 0.91624.
   - **Insight:** Feature importance in isolation ≠ additive value. V4's Inner K-Fold TE pipeline is already near-optimal. More features = more overfitting.

### March 01, 2026 (Earlier)

**1. S6E3 V4 LightGBM Inner K-Fold TE Model**
   - **Goal:** Perform a direct algorithmic swap of the proven V3 Inner K-Fold Leak-Free pipeline from XGBoost to LightGBM.
   - **Outcome:** Successfully implemented `S6E3_V4_LightGBM_InnerKFoldTE.py`. 
   - **Validation:** 0.91827 OOF AUC, 0.91609 LB AUC (New Best).
   - **Insight:** LightGBM's leaf-wise tree growth optimized the identical engineered features slightly better than XGBoost's depth-wise growth. Proves the V3 pipeline is the optimal baseline feature set.

**2. S6E3 V5 RealMLP Neural Network**
   - **Goal:** Introduce a PyTorch Neural Network using `pytabkit` to diversify our modeling approaches.
   - **Outcome:** Successfully implemented `S6E3_V5_RealMLP_DualRep.py` with 5 folds.
   - **Validation:** 0.91396 OOF AUC, 0.91377 LB AUC.
   - **Insight:** While it underperformed the top gradient boosters (0.916+), a 0.913+ NN is exceptionally strong for tabular data and provides excellent uncorrelated predictions. Time overhead (48 mins) confirms we should stick to LightGBM/XGBoost for rapid feature iterations.

**3. S6E3 V3 Inner K-Fold TE Model**
   - **Goal:** Replicate 0.91610 LB XGBoost baseline with leak-free target encoding and restricted pseudo labels.
   - **Outcome:** Successfully implemented `S6E3_V3_InnerKFoldTE.py`. 
   - **Validation:** 0.91774 OOF AUC, 0.91607 LB AUC. 
   - **Insight:** The inner K-fold target encoding cleanly prevented catastrophic overfitting seen in V2. The strict pseudo-label condition (must improve validation score) was critical, only firing on one fold. This is our new strong baseline.

**2. S6E3 Tracking Setup**
   - **Goal:** S6E3 environment setup and template creation.
   - **Outcome:** Adapted V1 baseline script from S6E2. Implemented LightGBM with simple pseudo labels.
   - **Validation:** V1 scored 0.91659 OOF and 0.91411 LB. A solid start.

**3. S6E3 V2 GroupBy FE Analysis**
   - **Goal:** Replicate Chris Deotte's 1st place massive FE strategy (GroupBy mean/std) for S6E3.
   - **Outcome:** Generated 215 features via cuDF, pushing local OOF to 0.91652.
   - **Insight:** Waiting on final LB to see if the massive interaction features overfit on this specific dataset compared to the baseline V1. Note: V3 has massively outperformed this conceptually.

**4. S6E2 Final Readme Creation**gs:**
    *   The pseudo-labeling pipeline provides a very strong V1 anchor point.
    *   The massive GroupBy interaction features (215+ new features) caused slight overfitting, dropping the LB score. This indicates we need feature selection or a more careful approach to categorical interactions.
*   **Next Steps:** Implement Phase 3 strategies (Target Encoding or Feature Selection on the V2 dataset) to improve the LB score.

## 2026-03-01
*   **Experiments Run:**
    *   `S6E3_V1_Baseline.py`: Ran the first XGBoost baseline utilizing cuDF, Global Frequency Encoding, and pseudo-labeling logic scraped from a top Kaggle notebook.
    *   `S6E3_V2_GroupByFE.py`: Implemented massive GroupBy aggregation feature engineering (Chris Deotte style) using cuDF.
*   **Result:** 
    *   V1: **0.91411 LB / 0.91659 OOF** 🏆
    *   V2: 0.91400 LB / 0.91652 OOF ❌
*   **Key Learnings:**
    *   The pseudo-labeling pipeline provides a very strong V1 anchor point.
    *   The massive GroupBy interaction features (215+ new features) caused slight overfitting, dropping the LB score. This indicates we need feature selection or a more careful approach to categorical interactions.
*   **Next Steps:** Implement Phase 3 strategies (Target Encoding or Feature Selection on the V2 dataset) to improve the LB score.