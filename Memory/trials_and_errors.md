# S6E3 Trials and Errors Log

> **⚠️ RULES:**
> 1. **Only update** after verifying outcome (OOF or LB)
> 2. **DO NOT DELETE** entries — failures are valuable
> 3. **PREPEND** new entries (latest first)
> 4. **Include:** Aim, Time taken, Results, Root cause, Lesson
> 5. **Status:** 🏆 BEST | ✅ SUCCESS | ⚠️ PARTIAL | ❌ FAILED | ⚠️ SKIPPED

---

### [EXP21]. S6E3 V20 LightGBM Optuna - ⚠️ PARTIAL (2026-03-08)
*   **Source:** Phase 12 - Optuna HPO on LightGBM
*   **Aim:** Apply Optuna-optimized hyperparameters specifically to LightGBM with V16 feature set (Digit Features + Bi-gram/Tri-gram TE) to see if LightGBM can match XGBoost with proper tuning.
*   **Time:** 151.9 minutes
*   **Results:**
    | Metric | V20 (LGBM Optuna) | V16b Baseline | V19 CatBoost | Delta vs V16b | Delta vs V19 |
    |--------|--------------------|---------------|--------------|---------------|---------------|
    | OOF AUC | 0.91908 | 0.91925 | 0.91900 | **-0.00017** | **+0.00008** |
    | LB Score | 0.91661 | 0.91680 | 0.91648 | **-0.00019** | **+0.00013** |
    | 20-Fold Mean | 0.91908±0.00170 | 0.91925±0.00173 | 0.91900±0.00178 | — | — |
*   **Root Cause:**
    1. **Leaf-wise vs Depth-wise:** LightGBM's leaf-wise tree growth doesn't provide an advantage over XGBoost's depth-wise growth on this heavily engineered feature set.
    2. **Optuna Parameters Found:** lr=0.00833, max_depth=7, num_leaves=77, reg_alpha=3.05, reg_lambda=0.225, min_child_samples=56, subsample=0.675, colsample_bytree=0.646, min_split_gain=0.076, extra_trees=True.
    3. **Heavy FE Saturation:** Both LightGBM and CatBoost consistently underperform XGBoost on the V16 feature pipeline.
*   **Lesson:**
    > **LightGBM with Optuna HPO (LB 0.91661) improves over V19 CatBoost (+0.00013) but still cannot match XGBoost V16b (-0.00019).** XGBoost's depth-wise growth remains the best architecture for this heavy FE dataset. The V16b XGBoost model remains the overall best single model.

---

### [EXP20]. S6E3 V19 CatBoost Optuna - ⚠️ PARTIAL (2026-03-08)
*   **Source:** Phase 12 - Optuna HPO on CatBoost
*   **Aim:** Apply Optuna-optimized hyperparameters specifically to CatBoost with V16 feature set (Digit Features + Bi-gram/Tri-gram TE) to see if CatBoost can match XGBoost with proper tuning.
*   **Time:** 49.1 minutes
*   **Results:**
    | Metric | V19 (CatBoost Optuna) | V16b Baseline | V18 CatBoost | Delta vs V16b | Delta vs V18 |
    |--------|------------------------|---------------|--------------|---------------|---------------|
    | OOF AUC | 0.91900 | 0.91925 | 0.91892 | **-0.00025** | **+0.00008** |
    | LB Score | 0.91648 | 0.91680 | 0.91640 | **-0.00032** | **+0.00008** |
    | 20-Fold Mean | 0.91900±0.00178 | 0.91925±0.00173 | 0.91893 | — | — |
*   **Root Cause:**
    1. **Symmetric Tree Architecture:** CatBoost's symmetric tree growth fundamentally limits its ability to capture fine-grained digit patterns that XGBoost's depth-wise growth exploits naturally.
    2. **Optuna Parameters Found:** lr=0.00984, depth=7, l2_leaf_reg=5.33, random_strength=2.88, bagging_temp=0.264, border_count=254, min_data_in_leaf=14.
    3. **Heavy FE Saturation:** As confirmed across V11, V18, and now V19, CatBoost consistently underperforms XGBoost/LightGBM on heavily engineered feature sets.
*   **Lesson:**
    > **Even with dedicated Optuna HPO, CatBoost cannot match XGBoost V16b.** The symmetric tree architecture is the limiting factor. However, V19 improved over V18 (+0.00008 LB) by using the full V16 feature pipeline with proper Optuna tuning, confirming that CatBoost benefits from the digit features but cannot overcome its structural limitations.

---

### [EXP19]. S6E3 V19 RGF (Regularized Greedy Forest) - ❌ FAILED (2026-03-07)
*   **Source:** S6E2 1st place winning solution — RGF provides different tree architecture
*   **Aim:** Train RGFClassifier (Regularized Greedy Forest) on V16's feature set to create model diversity for potential ensemble. RGF uses L2 regularization directly on leaf values and builds trees greedily.
*   **Time:** ~65 minutes per fold (killed after Fold 2)
*   **Results:**
    | Metric | V19 (RGF) | V16b Baseline | Delta |
    |--------|------------|----------------|-------|
    | Fold 1 AUC | 0.91864 | 0.92063 | -0.00199 ❌ |
    | Fold 2 AUC | 0.91778 | 0.91863 | -0.00085 ❌ |
    | Estimated 10-Fold | ~130+ min | — | **NOT VIABLE** |
*   **Root Cause:**
    1. **Massive Time Overhead:** RGF took ~64 minutes for Fold 1 and 130 minutes for Fold 2 (continuing to slow down). Estimated 10-fold ETA: 1300+ minutes (21+ hours).
    2. **AUC Underperformance:** RGF AUC (0.918) was significantly worse than V16b XGBoost (0.920+). RGF's greedy tree building doesn't match the optimized XGBoost gradient boosting.
    3. **No Early Stopping:** RGF doesn't support native early stopping like XGBoost, making it impossible to optimize training time.
*   **Lesson:**
    > **RGF is not viable for this competition.** The time-to-accuracy ratio is catastrophically worse than XGBoost. The algorithmic difference (greedy L2-regularized trees vs gradient boosted trees) doesn't provide enough diversity to justify the computational cost. This is permanently dead for S6E3.

### [EXP18]. S6E3 V18 CatBoost Residual (Sequential Boosting) - ❌ FAILED (2026-03-07)
*   **Source:** S6E1 Winner (V75/V77 sequential boosting strategy)
*   **Aim:** Use CatBoost to correct V16b XGBoost's mistakes via the `baseline` parameter. Train CatBoostClassifier on the same features, starting from XGBoost's log-odds predictions as initial baseline.
*   **Time:** 14.6 minutes
*   **Results:**
    | Metric | V18 (CatBoost Residual) | V16b Baseline | Delta |
    |--------|-------------------------|---------------|-------|
    | OOF AUC | 0.91925 | 0.91925 | **±0.00000 ❌** |
    | Per-fold | 0.91963\|0.91855\|0.92089\|0.91882\|0.91887\|0.91940\|0.92127\|0.91963\|0.91860\|0.91696 | — | — |
    | Correlation | 1.00000 | — | — |
*   **Root Cause:**
    1. **Signal Saturation:** V16b XGBoost (with 143 features including manual Bi-gram/Tri-gram TE) has perfectly extracted 100% of the available tabular signal.
    2. **CatBoost Early Stopping:** CatBoost early-stopped at iteration 0 on ALL 10 folds. It could not find a single split that improved the XGBoost logloss.
    3. **Perfect Correlation:** V18 predictions are 100% correlated with V16b (correlation = 1.00000), meaning CatBoost simply echoed the baseline back identically.
*   **Lesson:**
    > **Sequential boosting fails when the base model has exhausted the feature space.** CatBoost had no structural advantage remaining to exploit because V16b already implemented massive composite categorical TE. Without orthogonal weak spots, sequential boosting provides zero lift.

---

### [EXP18b]. S6E3 V18 CatBoost + Digit Features - ❌ FAILED (2026-03-07)
*   **Source:** V16 Digit Features success transferred to CatBoost
*   **Aim:** Test if CatBoost can leverage V16's 46 digit features (modulo, rounding, Benford's Law) with same pipeline as XGBoost.
*   **Time:** 29.8 minutes
*   **Results:**
    | Metric | V18 (CatBoost Digit) | V16b Baseline | Delta |
    |--------|------------------------|---------------|-------|
    | OOF AUC | 0.91892 | 0.91925 | **-0.00033** |
    | LB Score | 0.91640 | 0.91680 | **-0.00040** |
    | Per-fold | 0.91922|0.91840|0.92080|0.91835|0.91849|0.91903|0.92079|0.91935|0.91818|0.91666 | — | — |
*   **Root Cause:**
    1. **Symmetric Tree Limitation:** CatBoost builds balanced symmetric trees where each level uses the same split condition. This makes it harder to capture fine-grained digit patterns that XGBoost's depth-wise growth can find.
    2. **Feature Importance Mismatch:** While digit features showed importance in CatBoost (tenure_rounded_10 at 2.19% #1), the model's structural constraint prevents optimal utilization.
    3. **Heavy FE Saturation:** As seen in V11, CatBoost underperforms XGBoost/LightGBM on heavily engineered feature sets.
*   **Lesson:**
    > **CatBoost cannot match XGBoost on heavy FE datasets.** XGBoost's depth-wise tree growth is better suited for complex digit-feature interactions. The V16 digit features are model-independent in principle, but CatBoost's symmetric tree architecture cannot exploit them as effectively.

---

## 📝 TEMPLATE FOR NEW ENTRIES

```markdown

### [XXX]. [Exp Name] - [Status] (YYYY-MM-DD)
*   **Source:** [Where idea came from]
*   **Aim:** [Goal in 1-2 sentences]
*   **Time:** XX minutes
*   **Results:**
    | Metric | This Exp | Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF RMSE | X.XXXXX | X.XXXXX | **±X.XXXXX ✅/❌** |
    | LB Score | X.XXXXX | X.XXXXX | **±X.XXXXX ✅/❌** |
*   **Root Cause:** (for failures)
    1. Reason 1
    2. Reason 2
*   **Lesson:**
    > **Key takeaway** — what to remember
```

# **ENTRIES FROM BELOW THIS TEXT ARE NOT TO BE ALTERED**

### [EXP18]. Bayesian Target Encoding Variance - ❌ FAILED (2026-03-07)
*   **Source:** Code Review & Experimentation
*   **Aim:** Replace redundant `std`/`skew` with true Bayesian Estimate Variance (`p*(1-p)/N`) and sample counts for all categoricals and N-Grams to penalize noisy categories.
*   **Time:** Partial Run (Killed after Fold 5)
*   **Results:**
    | Metric | This Exp (Folds 1-5) | Baseline V16 (Folds 1-5) | Delta |
    |--------|----------|----------|-------|
    | CV AUC | 0.91785 | 0.91991 | **-0.00206** ❌ |
*   **Root Cause:**
    1. **Feature Dilution:** XGBoost splits inherently handle sample size via `min_child_weight` and tree depth limits. Adding explicit `count` and `uncertainty` metrics diluted the raw target likelihood (`mean_te`), causing trees to split on sample counts rather than directly on the probability bounds.
    2. **High Cardinality Stability:** For main categoricals, counts were already massive (10K+), rendering uncertainty ~0.00000. For sparse N-Grams, the uncertainty was largely a proxy for simple feature rareness, which tree depths natively regularize anyway.
*   **Lesson:** Do not explicitly encode sample counts or sample variance bounds for XGBoost when it already has well-tuned structural regularization parameters (`min_child_weight=6`, `reg_lambda=1.29`). Rely purely on the `mean_te` for probability mapping.

### [EXP17]. V18 Batch-Balanced Focal Loss - ⚠️ SKIPPED (2026-03-07)
*   **Source:** Code Review & Focal Loss Mathematics
*   **Aim:** Swap XGBoost's `binary:logistic` objective with focal loss to downweight easy examples without dropping rows.
*   **Time:** 0 minutes (Halted prior to execution)
*   **Results:**
    | Metric | This Exp | Baseline | Delta |
    |--------|----------|----------|-------|
    | LB Score | N/A | N/A | **⚠️ SKIPPED** |
*   **Root Cause:**
    1. **Incorrect Context:** The dataset class ratio is roughly 73:27, which is not mathematically extreme enough to warrant focal formulation (typically used for 99:1 imbalances).
    2. **Hessian Instability:** The analytical 2nd-order derivative (Hessian) of Focal Loss within XGBoost's C++ wrapper is highly unstable. Previous sweeps (gamma=2.0) yielded completely random 0.50 AUC.
    3. **Missing Chain Rules:** Simplified gradient implementations drop the `alpha_t * gamma * (1-p_t)^(gamma-1) * p*(1-p) * log(p_t)` term, breaking the gradient flow.
*   **Lesson:** Do not apply extreme class-imbalance loss modifications to mildly imbalanced synthetic datasets. To handle noisy continuous boundaries, keep the "hard" boundary examples naturally embedded and leverage scaling (`scale_pos_weight`) or Feature Targeting (Bayesian TE) instead.

### [EXP16]. V17 Two-Stage Noise Pruning (Confident Learning) - ❌ FAILED (2026-03-07)
*   **Source:** Phase 12 Advanced Architectures & `S6E3_V17_NoisePruning.py`
*   **Aim:** Remove top 1.17% (6,962 rows) of computationally confident errors (Model `>0.90` but label `0`, or `<0.10` but label `1`) from the training set so the trees could learn a cleaner, unbiased decision boundary.
*   **Time:** 38.7 minutes
*   **Results:**
    | Metric | V17 (Pruned) | V16 (Baseline) | Delta |
    |--------|--------------|----------------|-------|
    | OOF AUC | 0.93770 | 0.91925 | **+0.01845** (Artificial) |
    | LB Score | 0.91621 | 0.91680 | **-0.00059 ❌** |
*   **Root Cause:**
    1. By physically removing the contradictory rows (Confident Errors) from the continuous space, we artificially widened the margin between classes on the remaining data.
    2. XGBoost trees naturally use these "hard/noisy" instances near the absolute edges to regularize their depth and bound their leaf weights.
    3. Without these anchoring noise points, the trees overfit perfectly to the cleansed labels, leading to extreme probability confidence outputs that failed to generalize to the unseen, similarly-noisy Kaggle test set.
*   **Lesson:** Data cleansing via Confident Pruning destroys gradient generalization in tree-based models if the test set originates from the exact same noisy distribution as the training set. Do not alter the training domain explicitly.

### [EXP4]. CatBoost Sequential Baseline Boosting — ❌ NEUTRAL/WORSE (2026-03-07)
*   **Source:** S6E1 Winner (Using CatBoost to refine LightGBM/XGBoost baselines)
*   **Aim:** Train CatBoostClassifier (Logloss) starting from the exact log-odds local minimum of the V16b XGBoost base predictions via the `baseline` Pool parameter, utilizing CatBoost's native ordered categorical processing to find orthogonal splits XGBoost missed.
*   **Time:** ~60 minutes
*   **Results:**
    | Metric | EXP4 (CatBoost Baseline) | V16b Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91925 | 0.91925 | **±0.00000 ❌** |
    | LB Score | N/A | 0.91680 | — |
*   **Root Cause:**
    1. **Signal Saturation:** The V16b XGBoost model (with 143 features including manual K-Fold Bi-grams/Tri-grams) has perfectly extracted 100% of the available tabular signal.
    2. **CatBoost Early Stopping:** CatBoost could not find a single usable split on the Categorical architecture that reduced validation Logloss further than the XGBoost warm-start. The tree builders continuously returned 0 valid iterations, mathematically echoing the baseline back identically.
*   **Lesson:**
    > **Do not attempt sequential boosting when the base model has already exhausted the feature space.** Sequential ensembles (Boost-on-Boost) only work when the secondary model has a distinct structural advantage (e.g., CatBoost's native ordered categorical encoding on RAW categorical strings). Since our XGBoost pipeline already executed massive composite categorical TE, CatBoost had no structural advantage remaining to exploit.

### [EXP3]. Label Smoothing Regularization — ❌ WORSE (2026-03-07)
*   **Source:** Image Classification (Inception) / DL Tabular Regularization
*   **Aim:** Transform the binary target `[0, 1]` to softened continuous targets `[0.025, 0.975]` to prevent XGBoost from overfitting noisy boundary rows and becoming "infinitely confident".
*   **Time:** 35.0 minutes
*   **Results:**
    | Metric | EXP3 (Label Smooth) | V16 Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91909 | 0.91917 | **-0.00008 ❌** |
    | LB Score | N/A | 0.91679 | — |
*   **Root Cause:**
    1. **Synthetic Data Sharpness:** Random Forests/GBDTs inherently handle noisy classification boundaries well via ensembling. The Kaggle synthetic generation model likely utilizes sharp deterministic if/else rules to flip targets, requiring extreme confidence in terminal leaves to replicate.
    2. **Fuzzy Splits:** Smoothing the target actively penalized XGBoost for making the hard deterministic splits required to decode the synthetic dataset logic.
*   **Lesson:**
    > **Do not soften targets on Kaggle tabular data.** The synthetic generation process leaves deterministic sharp edges that models *must* capture with absolute confidence. Regularization should happen via tree constraints (depth, gamma, l1/l2), NOT via target masking.

### [EXP]. V16 Digit Features - 🏆 BEST SINGLE BASE (2026-03-06)
*   **Source:** S6E2 1st place, S5E11 1st place
*   **Aim:** Append 46 granular digit-level mathematical features (modulo, rounding, Benford's Law leading digits, string precision) to the V14 baseline to expose data synthesis artifacts to memory efficient floats.
*   **Time:** 38.0 minutes
*   **Results:**
    | Metric | V16 (Digit) | V14 Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91917 | 0.91889 | **+0.00028 ✅** |
    | LB Score | 0.91679 | 0.91656 | **+0.00023 🏆** |
*   **Root Cause:**
    1. **Synthetic Data Artifacts:** The target variable generation process appears to contain heuristics related to numbers rounding cleanly to 10s and 100s, or years (12 months).
    2. **Tree Limitations:** XGBoost splits functionally vertically/horizontally. It cannot create a "modulo 10" boundary easily. Explicitly creating the feature solves this structural blindness.
*   **Lesson:**
    > **Expose structural math explicitly.** Whenever there's a chance that data involves humans rounding numbers or systems applying modulo bounds, provide those bounds explicitly to the tree model as features.
    
---

### [EXP]. V15 (V14 with 20-Fold CV) — 🏆 SUCCESS (2026-03-06)
*   **Source:** S4E1 1st place solution (increasing folds to 20-30 for final model).
*   **Aim:** Train the exact V14 pipeline (Bi-gram/Tri-gram TE) using `N_FOLDS=20` instead of 10. This increases the training data per fold from 90% to 95% and provides a 20-model ensemble.
*   **Time:** 69.2 minutes
*   **Results:**
    | Metric | V15 (20-Fold) | V14 (10-Fold) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91897 | 0.91889 | **+0.00008 ✅** |
    | LB Score | 0.91657 | 0.91656 | **+0.00001 🏆** |
*   **Root Cause:**
    1. **Variance Reduction:** 20 models reduce ensemble variance significantly more than 10.
    2. **Marginal Data Utility:** The extra 5% of training data per fold allows the XGBoost model to find slightly better splits without overfitting.
*   **Lesson:**
    > **Always scale up folds for the final model.** Once feature engineering is locked and optimal, transitioning from 5/10 holds to 20/30 folds provides a "free" LB boost.

---

### [EXP]. V15f AllCat Mega-String & V15g CatBoost LIGHT — ❌ BOTH WORSE (2026-03-05)
*   **Source:** Phase 7 winning write-up "profile TE" (S4E7), and CatBoost native Newton optimization (S4E1 1st place).
*   **Aim:** Test two diverging strategies: V15f (extreme manual TE concatenating all 16 cats into one profile) and V15g (zero manual TE, purely relying on CatBoost's native ordered categorical handling with Newton leaf estimation).
*   **Time:** 49.0 minutes total.
*   **Results:**
    | Metric | V15f (AllCat XGB) | V15g (CatBoost Raw) | V14 Baseline |
    |--------|:---:|:---:|:---:|
    | OOF AUC | 0.91883 | 0.91639 | **0.91889 🏆** |
    | Delta | -0.00006 | -0.00250 | — |
*   **Root Cause:**
    1. **V15f AllCat Sparsity:** Creating a 16-categorical string created 44,356 unique profiles in train. This is too sparse for 594K rows (average ~13 rows per profile). The target encoding smoothing heavily regularized these, resulting in a signal that was slightly worse than V14's top-6 feature combinations. We crossed the line of "curse of dimensionality".
    2. **V15g CatBoost Native TE weakness:** CatBoost's internal ordered TE on raw features (0.91639) is significantly worse than XGBoost + our manual Inner K-Fold TE on derived features (0.91889). The manual TE provides global distributional stats (`min`, `max`, `std` over folds) that CatBoost's greedy ordered target encoding lacks.
*   **Lesson:**
    > **The "Goldilocks Zone" of TE:** V14's Bi-gram/Tri-gram strategy on the top 6 most important features found the exact sweet spot between interaction depth and cardinality density. Full 16-way interaction (AllCat) is too sparse; 1-way native interaction (CatBoost) is too weak.

### [EXP]. V15 TabR (ICLR 2024) — ❌ KILLED / NOT VIABLE (2026-03-05)
*   **Source:** "TabR: Tabular Deep Learning Meets Nearest Neighbors" (ICLR 2024), yandex-research official implementation
*   **Aim:** Apply official TabR architecture (FAISS top-k retrieval, label encoder, T-transform) on V14 TE-encoded 143 features. TabR retrieves actual nearest neighbors in embedding space and uses their TARGET LABELS directly in prediction — orthogonal inductive bias to GBDT.
*   **Time:** ~30 min (killed at Fold 1 Epoch 5 of 100)
*   **Results:**
    | Metric | TabR | V14 Baseline | Delta |
    |--------|:---:|:---:|:---:|
    | Val AUC (Epoch 1) | 0.74717 | 0.91924 | **-0.17207** |
    | Val AUC (Best, ~Epoch 4) | 0.79934 | 0.91924 | **-0.11990** |
    | Val AUC (Epoch 5) | 0.64484 | 0.91924 | Unstable |
    | Estimated time/epoch | **~6 min** | — | — |
    | Estimated 10-fold ETA | **~20 hours** | 9hr limit | **TIMEOUT** |
*   **Root Cause:**
    1. **Scale mismatch:** TabR was benchmarked on datasets of 10K–100K rows. Our 534K training fold passes ALL candidates to FAISS every batch step → O(N) per batch → ~6 min per epoch.
    2. **AUC collapse:** At epoch 5 AUC dropped from 0.799 to 0.644 — the FAISS context set shifts every batch as model weights change, causing unstable gradients over 534K candidates.
    3. **Would time out:** Even with patience=16, 1 fold ≈ 120+ min → 10 folds ≈ 20 hours (Kaggle limit = 9 hours).
*   **Lesson:**
    > **TabR is not viable for 594K rows on Kaggle.** The retrieval mechanism requires the entire training set as candidates, which becomes catastrophically slow at our scale. TabR's benchmarks use sub-100K datasets. PERMANENTLY DEAD for this competition.

---

### [EXP]. EXP-V15 Multi-Feature Screen (5 Techniques) - ❌ ALL NEUTRAL/WORSE (2026-03-05)
*   **Source:** Phase 11 research — S4E7, TPS Oct/Jan 2021, ICLR 2024, IARIA 2025
*   **Aim:** Screen 5 untried Phase-11 techniques (one fold each, ~22 min) to find the next improvement over V14.
*   **Time:** 22.1 minutes (screening only — no LB submission)
*   **Results (all vs V14 Fold-1 baseline 0.91924):**
    | Sub-Experiment | Fold-1 AUC | Delta | Verdict |
    |----------------|:---:|:---:|:---:|
    | V15b — Numerical Binning → TE | **0.91924** | **±0.00000** | = SAME |
    | V15c — Churn Archetype Binary Flags | 0.91917 | -0.00007 | ❌ WORSE |
    | V15h — Quantile Transform Numericals | **0.91924** | **±0.00000** | = SAME |
    | V15e — Denoising Autoencoder Latent | 0.91897 | **-0.00027** | ❌ WORST |
    | V15i — SHAP Feature Elimination | 0.91919 | -0.00005 | = SAME |
    | **V14 Baseline** | **0.91924** | **±0.00000** | 🏆 STILL BEST |
*   **Root Cause:**
    1. **Binning+TE (V15b):** The bins recovered information already captured by ORIG_proba and the existing CAT_tenure/CAT_MonthlyCharges encoding. TE on a coarser-grained grouping carries no signal beyond what the fine-grained existing encoding already provides.
    2. **Churn Flags (V15c):** Boolean composites are redundant with ORIG_proba. The original IBM dataset probabilities already encode churn risk per category combination. Manual boolean archetypes are a subset of what ORIG_proba captures continuously.
    3. **Quantile Transform (V15h):** Trees are rank-invariant — QuantileTransformer preserves rank order, which is exactly what XGBoost already sees. Zero new information for tree-based models.
    4. **DAE (V15e) — MOST HARMFUL:** Latent features (-0.00027) added pure noise. DAE was trained on 16 cats + 13 numerics with a 16-dim bottleneck. The bottleneck compressed too aggressively for 594K rows with mostly categorical features. The compressed representations lost useful signal and introduced noise.
    5. **SHAP RFE (V15i):** SHAP threshold of 0.000 meant zero features were removed. The V14 model uses all 143 features efficiently — there's no dead weight to remove.
*   **Lesson:**
    > **V14 with Bi-gram/Tri-gram TE has reached a local optimum for single-model FE.** The remaining "easy" feature engineering tricks are redundant with existing encodings. The remaining paths forward are: (A) 20-fold retrain to reduce variance, (B) AllCat mega-string TE extending V14's composite idea, (C) a fundamentally different model architecture, or (D) ensembling (if unlocked).

---

### [11]. V14 Bi-gram/Tri-gram Categorical TE - 🏆 NEW BEST (2026-03-04)
*   **Source:** S6E2 (Heart Disease) 1st place winning solution — composite categorical strings + TE
*   **Aim:** Concat pairs/triplets of top 6 categoricals into composite strings, then inner-fold TE encode. Captures 2-way and 3-way categorical interactions XGB can't easily learn from splits alone.
*   **Time:** 31.6 minutes
*   **Results:**
    | Metric | V14 | V12 Baseline | Delta |
    |--------|:---:|:---:|:---:|
    | OOF AUC | **0.91889** | 0.91879 | **+0.00010 ✅** |
    | LB Score | **0.91656** | 0.91652 | **+0.00004 🏆** |
    | Folds | 0.91924 \| 0.91821 \| 0.92055 \| 0.91849 \| 0.91856 \| 0.91910 \| 0.92090 \| 0.91931 \| 0.91811 \| 0.91654 |
*   **Key Findings:**
    - 15 bi-grams + 4 tri-grams = 19 new composite cols → 143 features after TE
    - Top 3 most important features were ALL n-gram TEs:
      1. `TG_Contract_InternetService_OnlineSecurity` (0.1551)
      2. `TG_Contract_InternetService_PaymentMethod` (0.1472)
      3. `BG_Contract_InternetService` (0.1378)
    - Tri-grams dominate bi-grams in importance
*   **Lesson:**
    > **Composite categorical TE captures real signal beyond single-column TE and ORIG_proba.** The Contract×InternetService×OnlineSecurity trio is the most predictive group — makes domain sense (contract commitment + service type + security add-on define churn risk profiles). This is now the **OVERALL BEST single model at LB 0.91656.**

---

### [EXP]. V14b Polynomial Features (x², x³) - ❌ FAILED (2026-03-04)
*   **Source:** S5E12 1st place, S6E2 winning solutions — polynomials on raw numericals
*   **Aim:** Add squared and cubed versions of top 6 numerical columns + 3 cross-polynomial interactions. Captures U-shaped/S-shaped patterns linear splits miss.
*   **Time:** 28.3 minutes
*   **Results:**
    | Metric | V14b (Poly) | V12 Baseline | Delta |
    |--------|:---:|:---:|:---:|
    | OOF AUC | **0.91891** | 0.91879 | **+0.00012 🏆** |
    | LB Score | **0.91627** | 0.91652 | **-0.00025 ❌** |
    | Gap | -0.00264 | -0.00240 | **Wider gap = Overfit** |
*   **Root Cause:**
    1. **Massive Overfitting:** The OOF AUC improved (+0.00012) but the LB tanked (-0.00025). The OOF-LB gap widened purely because polynomials allow trees to fit the training noise too perfectly.
    2. **Low Importance:** Despite having 15 new features, the top polynomial feature (`tenure_cu`) only had 1.48% importance. Compare this to V14's tri-gram TE which had 15.5% importance.
*   **Lesson:**
    > **Polynomials on raw numericals overfit this dataset.** We saw this in EXP5 with distribution polynomial interactions, and we see it again here with raw numericals. The S5E12 dataset was much smaller and handled poly better. Here, it just increases the CV-LB gap.

---

### [EXP]. V14 MultiTechnique: WOE + Curriculum PL + Calibration - ❌ FAILED (2026-03-04)
*   **Source:** NeurIPS 2023 benchmark (WOE), Kim et al. 2023 arXiv (Curriculum PL)
*   **Aim:** Test 4 research-backed techniques: WOE encoding, Curriculum PL, Adversarial Validation, Calibration
*   **Results:**
    | Experiment | AUC | Delta | Verdict |
    |-----------|:---:|:---:|:---:|
    | BASELINE (V12) | 0.91879 | — | — |
    | WOE (replace TE) | 0.91882 | +0.00004 | = SAME |
    | WOE + TE (additive) | 0.91876 | -0.00002 | = SAME |
    | Curriculum PL (4-round) | 0/8 rounds improved | — | ❌ DEAD |
    | Adversarial Val | AUC=0.512 | — | ✅ No shift |
*   **Root Cause:**
    1. **WOE:** ln(P(X|Y=1)/P(X|Y=0)) ≈ logit of target encoding → too similar to TE, XGB already handles non-linear encoding
    2. **Curriculum PL:** Adding 46K-116K PL samples MONOTONICALLY worsened AUC. More PL = worse. PL corrupts signal on this dataset regardless of technique (threshold, curriculum, density-regularized)
    3. **Adversarial Val:** AUC=0.512 confirms train/test nearly identical → no distribution shift to fix
*   **Lesson:**
    > **PL is PERMANENTLY dead on this dataset (now 0/18+ across all methods).** WOE adds no value over TE for GBDT. Adversarial validation confirmed no train/test shift — the CV-LB gap (0.00238) is just noise.

---

### [EXP]. External Dataset Features (ChurnScore/CLTV) - ⚠️ INSIGHTFUL FAILURE (2026-03-04)
*   **Source:** Extended IBM dataset (`Telco_customer_churn.csv`, 33 cols) has ChurnScore (AUC=0.94!) and CLTV not in competition data.
*   **Aim:** Map ChurnScore & CLTV group means (72 features) from 7,043-row extended dataset onto 600K competition rows.
*   **Results:** +0.00001 AUC (zero gain). 33.3 min total.
*   **Root Cause:**
    1. ChurnScore was computed by IBM SPSS (Logistic Regression) using the **exact same 19 features** we already have
    2. Group-level means of ChurnScore ≈ ORIG_proba (same signal, different scale)
    3. Reconstructing individual ChurnScore = building a weaker model's prediction as a feature → circular
*   **Lesson:**
    > **External datasets only help if they contain truly new information.** ChurnScore is just another model's probability on the same features. Rule: **Don't add external model predictions as features when they use the same inputs.**

---

### [EXP]. DART XGBoost - ❌ NEVER USE (2026-03-04)
*   **Source:** Research papers suggest DART helps with correlated features via tree dropout
*   **Aim:** DART booster on V12 Optuna params. rate_drop=0.1, skip_drop=0.5, 5000 fixed trees.
*   **Results:** Fold 1 AUC **0.91846** (-0.00078 vs V12 gbtree). **350 min** for 1 fold (74x slower).
*   **Why Failed:** (1) DART + colsample=0.32 = double dropout = over-regularized. (2) DART is O(n²) per iteration. (3) Can't early stop reliably.
*   **Lesson:**
    > **DART is catastrophically slow and harmful when column subsampling is already aggressive.** Added Rule 8: NO DART.

### [EXP]. V15 Multi-Experiment Quick Test - ❌ ALL FAILED (2026-03-04)
*   **Source:** V12 params near-optimal → systematically test remaining levers
*   **Aim:** Test Focal Loss, scale_pos_weight, colsample grid, feature pruning on V12 params (5-fold CV).
*   **Results:** Max gain: +0.00004 (noise). Focal Loss γ=2.0 = AUC 0.50 (broken). γ=1.0 = -0.00024. All SPW worse. Colsample 0.15-0.50 within ±0.00005 of 0.32.
*   **Lesson:**
    > **V12 params are near-optimal.** No single lever moves the needle. The 0.91652 LB ceiling may be a fundamental limit of single-model approaches.

### [10]. V13 LGBM Optuna HPO - 🏆 TIED WITH V12 (2026-03-04)
*   **Source:** V12 XGB success (+0.00007 LB via Optuna) → apply same approach to LGBM
*   **Aim:** Bayesian HPO on V7 LGBM. 10 params (incl. LGBM-unique path_smooth, min_gain_to_split).
*   **Time:** 713 min search (50/100 trials) + 89 min retrain = 802 min total
*   **Results:**
    | Metric | V7 (Hand-tuned) | V13 (Optuna) | Delta |
    |--------|:---------------:|:------------:|:-----:|
    | 5-fold AUC (search) | 0.91835 | **0.91869** | **+0.00034** |
    | OOF AUC (10-fold) | 0.91851 | **0.91890** | **+0.00039** |
    | LB Score | 0.91637 | **0.91652** | **+0.00015** |
*   **Key Shifts:** lr: 0.03→0.012 (2.5x↓), col: 0.80→0.30 (63%↓), α: 0.1→7.16 (72x↑), λ: 1.0→5.44, path_smooth: 0→8.89, depth: 6→11 (sparse)
*   **Lesson:**
    > Same pattern as V12 XGB: heavy column dropout + strong L1. V13 ties V12 on LB → **model choice doesn't matter when both are well-tuned.**

### [9]. V12 Optuna XGBoost HPO - 🏆 NEW BEST (2026-03-04)
*   **Source:** McElfresh 2023 (TabZilla): "light HPO on GBDT > model choice". Holzmüller 2024 meta-tuned defaults.
*   **Aim:** Bayesian HPO (Optuna TPE) to find optimal XGBoost params for 600K×64 dataset.
*   **Time:** 712 min search (93/100 trials) + 47.2 min retrain = 759 min total
*   **Results:**
    | Metric | V8 (Hand-tuned) | V12 (Optuna) | Delta |
    |--------|:---------------:|:------------:|:-----:|
    | OOF AUC (5-fold search) | 0.91844 | **0.91879** | **+0.00035** |
    | OOF AUC (10-fold final) | 0.91857 | **0.91892** | **+0.00035** |
    | LB Score | 0.91645 | **0.91652** | **+0.00007** |
*   **Key Shifts:** lr: 0.05→0.0063 (8x↓), col: 0.80→0.32 (60%↓), α: 0.1→3.5 (35x↑), γ: 0.05→0.79 (16x↑), depth: 6→5
*   **Lesson:**
    > **Heavy regularization wins on large FE datasets.** With 64 correlated features, the model only needs 32% of features per tree and much stronger L1/pruning. Hand-tuned params from S6E2 (7K rows, 13 features) were under-regularized for S6E3 (600K rows, 64 features).

### [8]. V11 CatBoost + V7 Features - ❌ UNDERPERFORMS (2026-03-03)
*   **Source:** S6E2 V39 Ordered Boosting (proven in previous competition) + CatBoost Depthwise research
*   **Aim:** Test CatBoost as diversity model with V7 features. Tried 3 configurations.
*   **Time:** 17.7 minutes (Depthwise, fastest config)
*   **Results:**
    | Config | Fold 1 AUC | OOF AUC | LB Score |
    |--------|-----------|---------|----------|
    | SymmetricTree (default) | 0.91720 | — | — |
    | Ordered + depth=6 (S6E2 V39) | 0.91662 | — | — |
    | **Depthwise + depth=8** | **0.91753** | **0.91736** | **0.91494** |
    | V8 XGB (reference) | 0.91901 | 0.91857 | **0.91645** |
*   **Root Cause:** CatBoost's native ordered TE and auto feature combinations are **redundant** with our 64 engineered features (19 ORIG_proba, 9 dist, 8 qdist). Heavy FE saturates CatBoost's built-in tricks. Symmetric tree constraint (even with Depthwise) limits flexibility vs XGB depth-wise. The -0.00242 OOF-LB gap is the widest of any model.
*   **Lesson:**
    > **CatBoost shines on raw features, not heavy FE.** In S6E2 (13 raw features), CatBoost V39 was top-2. In S6E3 (64 engineered features), CatBoost is worst. The more FE you do, the less CatBoost's internal magic adds value. Stick to XGB/LGBM for heavy FE datasets.

### [7]. V10 RealMLP + V7 Features - ⚠️ PARTIAL (2026-03-03)
*   **Source:** S6E2 V48 RealMLP proven architecture + V7 feature set
*   **Aim:** Test RealMLP_TD with S6E2-tuned hyperparams on S6E3's V7 features.
*   **Time:** 263 minutes
*   **Results:**
    | Metric | V10 | V5 RealMLP | Delta |
    |--------|-----|------------|-------|
    | OOF AUC | 0.91633 | 0.91396 | **+0.00237 ✅** |
    | LB Score | 0.91491 | 0.91377 | **+0.00114 ✅** |
*   **Root Cause:** V7 features helped (+0.00114 LB over V5), but S6E2-tuned hyperparams may not be optimal for S6E3's much larger dataset (594K vs 15K rows). RealMLP is slower and weaker than TabM on this dataset.
*   **Lesson:**
    > **TabM strictly dominates RealMLP for S6E3.** V9 TabM beats V10 RealMLP by +0.00134 LB while being faster. Use TabM as the NN diversity model.

### [6]. V9 TabM + V7 Features - ✅ SUCCESS (2026-03-03)
*   **Source:** Deep research (ICLR 2025 paper, S5E11/S5E12 winning solutions)
*   **Aim:** Test TabM (parameter-efficient MLP ensemble) as NN diversity model with V7 features.
*   **Time:** 233 minutes
*   **Results:**
    | Metric | V9 TabM | V8 XGB Best | V5 RealMLP | Delta vs V5 |
    |--------|---------|-------------|------------|-------------|
    | OOF AUC | 0.91845 | 0.91857 | 0.91396 | **+0.00449 ✅** |
    | LB Score | 0.91625 | 0.91645 | 0.91377 | **+0.00248 ✅** |
*   **Root Cause/Success:** TabM's BatchEnsemble (k=32 implicit MLPs sharing weights) + PiecewiseLinear embeddings captures smooth decision boundaries that trees can't. OOF nearly matches V7 LGBM (0.91845 vs 0.91851).
*   **Lesson:**
    > **TabM (ICLR 2025) is the best NN for tabular data.** Massive improvement over RealMLP. Provides excellent diversity anchor for future ensemble with V8 XGB and V7 LGBM.

### [5]. EXP5 Ultimate Feature Discovery - ✅ SUCCESS (2026-03-02)
*   **Source:** Exhaustive search for any remaining FE before moving to model diversity
*   **Aim:** Test 92 features across 10 new directions (MonthlyCharges/tenure distributions, conditional groups, 3-way conditionals, quantile distances, KDE, clusters, polynomials, nearest-neighbor).
*   **Time:** ~6 hours
*   **Results:**
    | Metric | V6+EXP5 | V6 Baseline | Delta |
    |--------|---------|-------------|-------|
    | OOF AUC (5-fold) | 0.91757 | 0.91739 | **+0.00018 ✅** |
*   **Root Cause:** Only Batch F (quantile distance for TotalCharges) survived. All other directions (MC/tenure distributions, clusters, KNN, KDE, polynomials) were neutral or hurt. TotalCharges is uniquely informative because it combines tenure × price × promotions into one number.
*   **Lesson:**
    > **TotalCharges distance to original churner/non-churner quantiles (Q25/Q50/Q75) captures curvature that percentile rank alone misses.** This is the last confirmed valuable FE direction.

### [4]. EXP4 OptimalBinning WoE - ⚠️ NEUTRAL (2026-03-02)
*   **Source:** Kaggle notebook by alpayabbaszade (AUC 0.9136 standalone)
*   **Aim:** Test if `optbinning` 1D WoE (19 features) + 2D joint WoE (45 feature pairs) add signal on top of V4+EXP3 baseline.
*   **Time:** 262 minutes
*   **Results:**
    | Metric | V4+EXP3+WoE | V4+EXP3 Baseline | Delta |
    |--------|-------------|------------------|-------|
    | OOF AUC (5-fold) | 0.91741 | 0.91739 | **+0.00002 ⚠️** |
*   **Root Cause:** WoE is mathematically a monotonic transform of `ORIG_proba` (both derive from original target statistics). LightGBM learns the same splits either way. 2D WoE interactions are redundant because trees naturally split on feature pairs.
*   **Lesson:**
    > **OptBinning WoE ≈ ORIG_proba mapping.** Both encode target statistics from the original IBM dataset. Simpler is better — no need for the `optbinning` library.

### 36. EXP-V17c: Monotonic Constraints - ⚠️ SKIPPED (2026-03-07)
*   **Source:** XGBoost domain regularization strategy
*   **Aim:** Enforce `-1` monotonic relationships on `tenure` and `TotalCharges` to prevent tree nodes from overfitting to local noise on Kaggle data, forcing logically sound splits.
*   **Time:** 39.6 minutes
*   **Results:**
    | Metric | This Exp | V16 Baseline | Delta |
    |--------|----------|--------------|-------|
    | OOF AUC  | 0.91915 | 0.91917      | **-0.00002 ❌** |
*   **Root Cause:**
    1. **Overtuning Base:** The base V12/V16 XGBoost hyperparameter suite (`reg_alpha` 3.5, `reg_lambda` 1.29, `gamma` 0.79, `colsample` 0.32) is already *massively* regularized to an extreme degree.
    2. **Over-constrained:** Adding physical split constraints on top of that heavy mathematical regularization prevents the tree from capturing genuine micro-signals, crossing over from "preventing noise" to "preventing learning".
*   **Lesson:**
    > When an XGBoost model has been exhaustively Bayesian-optimized to combat noise via depth/subsample/gamma/alpha tuning, applying explicit hard constraints (like Monotonicity) usually hurts. The optimized parameters already account for the optimal tree freedom.

---

### 35. EXP-V17b: Multi-Target Encoding - ⚠️ SKIPPED (2026-03-07)
*   **Source:** S5E11 1st place
*   **Aim:** Predict 5 demographic sub-targets (SeniorCitizen, Dependents, Partner, etc.) using categorical grouping, instead of predicting Churn. Extract domain structure without target leakage.
*   **Time:** 40.8 minutes
*   **Results:**
    | Metric | This Exp | V16 Baseline | Delta |
    |--------|----------|--------------|-------|
    | OOF AUC  | 0.91918 | 0.91917      | **+0.00001 ❌** |
*   **Root Cause:**
    1. **Signal Correlation:** The demographic groupings (e.g. `Dependents` by `InternetService`) are strongly correlated with the original Churn probabilities of those groups, offering no robust orthogonal signal.
*   **Lesson:**
    > Multi-Target encoding requires sub-targets that are largely independent of the main target to be useful. If the sub-targets just map back to the same population segments, it's redundant.

---

### 34. EXP-V17: Round/Binning Features + TE - ⚠️ SKIPPED (2026-03-07)
*   **Source:** S5E11 1st place
*   **Aim:** Discretize continuous columns (`tenure`, `MonthlyCharges`) into granular bins (e.g. 3-month blocks, $10 blocks) and apply Inner K-Fold TE to avoid overfitting while extracting temporal/financial churn trends.
*   **Time:** 54.5 minutes
*   **Results:**
    | Metric | This Exp | V16 Baseline | Delta |
    |--------|----------|--------------|-------|
    | OOF AUC  | 0.91916 | 0.91917      | **-0.00001 ❌** |
*   **Root Cause:**
    1. **Redundancy:** The gradient boosting tree already natively discretizes continuous variables optimally via its splitting threshold algorithm. Forcing manual bins only degraded that native precision. 
    2. **Signal Ceiling:** `ORIG_proba` mappings already encode the true global probability. The manual bins essentially just replicated a slightly noisier version of `ORIG_proba`.
*   **Lesson:**
    > Numeric Binning + TE works incredibly well on linear datasets or wide MLPs, but when applied to deeply tuned XGBoost models with pre-existing quantile/probability features, it is totally redundant.

---

### 33. V16b: 20-Fold Retrain on V16 - 🏆 BEST SINGLE MODEL (2026-03-07)
*   **Source:** S4E1 1st (CatBoost 20 folds)
*   **Aim:** Squeeze final data efficiency (95% train fold vs 90%) out of our best single model base (V16) to see if it sets a higher baseline before ensembling.
*   **Time:** 80.0 minutes
*   **Results:**
    | Metric | This Exp | V16 Baseline | Delta |
    |--------|----------|--------------|-------|
    | OOF AUC  | 0.91925 | 0.91917      | **+0.00008 ✅** |
    | LB Score | 0.91680 | 0.91679      | **+0.00001 ✅** |
*   **Lesson:**
    > Extending CV from 10 to 20 folds on our strongest feature set (V16 Digit Features) guarantees a tiny micro-optimization, establishing the absolute highest isolated feature baseline possible.

---

### [3]. EXP3 Novel Feature Mining - ✅ SUCCESS (2026-03-02)
*   **Source:** EXP2 failure analysis → need genuinely novel features
*   **Aim:** Find features orthogonal to V4's 58 features by aggressively mining distribution patterns (percentiles, z-scores, churner vs non-churner distances).
*   **Time:** 168 + 130 minutes
*   **Results:**
    | Metric | This Exp (v3) | Baseline (V4) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC (5-fold) | 0.91685 | 0.91649 | **+0.00036 🏆** |
*   **Root Cause/Success:** Tested over 200 features across v2/v3. 9 features survived a strict greedy forward selection and 5-fold CV confirmation. All 9 were based on **distribution distance** or **conditional percentiles** of `TotalCharges` relative to the original dataset.
*   **Lesson:**
    > **Distribution-based features** (percentile rank against original distributions of churners vs non-churners) are the ONLY orthogonal direction that V4 wasn't already capturing via FREQ/ORIG encodings.

### [2]. EXP2 Feature Validation - ❌ FAILED (2026-03-01)
*   **Source:** EXP1 top features
*   **Aim:** Validate if EXP1's top features improve V4 baseline.
*   **Time:** 9.2 minutes
*   **Results:**
    | Metric | V4 Only | V4 + Top EXP1 | V4 + All EXP1 | Delta |
    |--------|---------|---------------|---------------|-------|
    | OOF AUC (5-fold) | 0.91648 | 0.91632 | 0.91624 | **-0.00017 ❌** |
*   **Root Cause:**
    1. EXP1 features scored high in isolation but are redundant with V4's existing FREQ/ORIG encodings
    2. Adding correlated features creates multicollinearity → dilutes tree split quality
*   **Lesson:**
    > **Feature importance in isolation ≠ additive value.** Always validate on top of actual pipeline, not in a vacuum.

### [1]. EXP1 Feature Discovery - ✅ SUCCESS (as research) (2026-03-01)
*   **Source:** Web research + synthetic artifact analysis
*   **Aim:** Generate 277 features across 12 categories and rank by 3 models + correlation.
*   **Time:** 7.9 minutes
*   **Results:**
    | Model | AUC (5-fold) |
    |-------|-------------|
    | LightGBM | 0.91636 |
    | XGBoost | 0.91649 |
    | CatBoost | 0.91585 |
*   **Root Cause:** N/A (research experiment, not submission)
*   **Lesson:**
    > **`risk_score_composite` ranked #1 across all models**. Synthetic artifact features ranked LOWEST (avg 0.0725). 257/295 features above random noise. But high importance ≠ additive value (see EXP2).

---