# S6E3 Ideas Tracker — Master Plan

## ⚠️ RULES

1. **Try ideas in ORDER** (top to bottom within phase)
2. **Mark `[x]` when tried** and record result
3. **Check "What Doesn't Work"** before starting — SKIP if overlap
4. **Include timing estimates** for pending experiments
5. **Record BOTH OOF and LB** for every submission
6. **NO ENSEMBLING / BLENDING / STACKING/ MULTISEED** until explicitly requested by user. Do not even suggest it.
7. **Status icons:** 🏆 Best | ✅ Done | ⚠️ Partial | ❌ Failed | [ ] Pending
8. **NO DART BOOSTING.** Tested in V14: 74x slower than gbtree, -0.00078 AUC. Never suggest or use DART for this competition.

### 📝 Version Table Format
| Version | Base | Source Files | Changes | Expected | Time Est | Status |
|---------|------|--------------|---------|----------|----------|--------|

---

# 🔍 PRE-RUN CHECKLIST
1. [ ] **Not in "Already Tried"** section
2. [ ] **Runnable** — no gated models, auth, or blocked libraries
3. [ ] **Time estimate** fits your session
4. [ ] **Expected gain** justifies effort

---

# **ENTRIES FROM BELOW THIS TEXT ARE NOT TO BE ALTERED**

## 🎯 Phase 1-6 Master Learnings (COMPLETED)
*   ✅ **cuDF Speed FE**: Required for massive groupby interactions.
*   ✅ **Global Frequency Encoding**: Best done across Train+Test+Original.
*   ✅ **ORIG Probability Mapping**: Pulling target means from the original IBM dataset explicitly boosts score.
*   ✅ **Leak-Free Inner K-Fold TE**: Solved the XGBoost categorical target leakage problem.
*   ✅ **Restricted Pseudo-Labeling**: Only inserting test-set probabilities into training if validation AUC strictly increases in that fold.

---

## 🚀 NEW: Phase 7 Advanced Algorithms & Scaling (S6E3 V4+)

The current limitation is that we have solely relied on XGBoost. For telecom churn (heavily categorical with 16 categorical variables), switching algorithms while keeping the proven V3 feature pipeline is the best next step (as proven in S6E1/S6E2).

*   ✅ **LightGBM Algorithm (V4 Primary Goal):** LGBM's leaf-wise tree growth often finds deeper non-linear categorical interactions than XGBoost's depth-wise growth. Combine this EXACTLY with V3's Inner K-Fold TE features. **(Result: OOF 0.91827 / LB 0.91609 - Slight improvement over V3 XGBoost!)**
*   [ ] **CatBoost Algorithm:** CatBoost natively handles categoricals perfectly using ordered target encoding. This might completely eliminate the need for our complex Inner K-Fold TE loop while mathematically preventing target leakage better than XGBoost.
*   [ ] **Optuna Hyperparameter Tuning:** Running 50-100 trials of Optuna specifically on max_depth, learning_rate, colsample, and l2_leaf_reg to squeeze out the final +0.002 AUC.
*   [ ] **Feature Concatenation (The "AllCat" trick):** Combine all 16 categoricals into one single string vector (`gender_contract_internet_...`) and let model hash it.
*   [ ] **UMAP / PCA Dimensionality Reduction:** Embed the numerical + mapped categorical features into 2D continuous space to give tree algorithms diagonal splits.

## 🧠 Phase 8 Neural Network Expansion

Based on deep research (7 searches, ICLR 2025 paper, TabM GitHub API, winning solutions from S4E1/S5E11/S5E12/S6E2):

### NN Architecture Analysis for S6E3 (594K rows, 16 cats, 67 features)
| Model | Paper | Strengths | Kaggle Wins | Our Fit |
|-------|-------|-----------|-------------|---------|
| **TabM** | ICLR 2025 | BatchEnsemble MLP, k members, native cats, `pip install tabm` | S5E11 5th, S5E12 4th | ⭐⭐⭐⭐⭐ |
| **RealMLP** | pytabkit | Simple, tested as V5 (0.91377 LB) | Several comps | ⭐⭐⭐ |
| **FT-Transformer** | NeurIPS 2021 | Attention over features | Research-only | ⭐⭐⭐ |
| **TabPFN v2** | 2025 | Foundation model, zero-shot | ❌ (max 10K rows) | ❌ |

*   ✅ **RealMLP Dual Representation (V5):** `pytabkit` NN. **(Result: OOF 0.91396 / LB 0.91377 — diversity anchor)**
*   ✅ **TabM (V9):** ICLR 2025. BatchEnsemble MLP (k=32) + PiecewiseLinear embeddings. **(Result: OOF 0.91845 / LB 0.91625 — 🏆 Best NN)**
*   ✅ **RealMLP + V7 (V10):** S6E2 V48 tuned params + V7 features. **(Result: OOF 0.91633 / LB 0.91491 — TabM strictly better)**

### Hidden Techniques from Winning Solutions
1. **Multi-seed TabM** — S5E11 5th: "xgb+lgbm+tabm5seeds" (5 seeds averaged)
2. **PiecewiseLinearEmbeddings** — TabM GitHub: "most popular choice" for embeddings
3. **Train k members independently** — Mean loss, NOT loss of mean prediction
4. **Average probabilities, not logits** — For binary classification inference
5. **CatBoost auto feature combinations** — Discovers pairwise categorical interactions automatically during tree construction

## 🔬 Phase 9: Feature Discovery Research (EXP1/EXP2/EXP3)

**Key Finding: V4's 58-feature pipeline is near-optimal for LightGBM. Adding features monotonically hurts.**

*   ✅ **EXP1 Feature Discovery:** Generated 277 features across 12 categories. `risk_score_composite` #1 universal, `CLV_simple` #2. Synthetic artifacts ranked LOWEST.
*   ❌ **EXP2 Validation:** All EXP1 features HURT V4 baseline (-0.00017 top, -0.00024 all). Feature importance ≠ additive value.
*   ❌ **EXP3 v1 Forensics:** 11/20 EXP1 features >0.8 correlated with V4. 0/20 help individually. Cross-interactions 0.96+ corr with raw Contract.
*   ⚠️ **EXP3 v2 Novel Features:** 111 features across 6 novel batches. Only 2 helped: `pctrank_orig_TotalCharges` (+0.00010), `zscore_orig_TotalCharges` (+0.00005). Combined: +0.00012.
*   ✅ **EXP3 v3 Deep Distribution Mining:** 9 validated features, +0.00036 (5-fold confirmed).
*   ⚠️ **EXP4 OptBinning WoE:** 64 features (1D+2D WoE from `optbinning`), +0.00002 (noise). WoE ≈ ORIG_proba.
*   ✅ **EXP5 Ultimate FE:** 92 features across 10 directions. Only 8 quantile distance features survived (+0.00018).

### What DOESN'T Work (Confirmed Dead Ends)
*   ❌ Risk flags / composites → redundant with FREQ + ORIG_prob
*   ❌ Cross-interactions → redundant with raw categoricals (0.96+ correlation)
*   ❌ CLV / RFM → redundant with tenure × charges
*   ❌ WoE encoding (simple + OptBinning + V14 WOE) → redundant with ORIG_prob/TE (EXP3 v2 + EXP4 + V14)
*   ❌ Multi-stat TE (std/count/sum) → neutral effect
*   ❌ Conditional churn stats → neutral or hurts
*   ❌ Massive GroupBy features → overfitting (confirmed V2)
*   ❌ 2D joint WoE interactions → trees learn these splits natively (EXP4)
*   ❌ MonthlyCharges/tenure distributions → neutral (EXP5)
*   ❌ 3-way conditional groups, KDE ratio, KMeans clusters, nearest-neighbor → hurt (EXP5)
*   ❌ Polynomial interactions on dist features AND raw numericals → massive overfitting / neutral (EXP5, V14b)
*   ❌ CatBoost with heavy FE → native TE/feature combos redundant, underperforms XGB/LGBM (V11, V18 LB 0.91640, V19 LB 0.91648)
*   ❌ LightGBM with heavy FE → underperforms XGBoost on V16 features (V20 LB 0.91661 vs V16b LB 0.91680)
*   ❌ DART booster → 74x slower, -0.00078 AUC. NEVER USE (V14-DART)
*   ❌ Focal Loss / scale_pos_weight tuning → ≤+0.00004 (V15)
*   ❌ Pseudo-Labeling (ALL variants) → 0/18+ rounds across threshold, curriculum, density-reg. PL corrupts signal. DEAD.
*   ❌ External dataset features (ChurnScore/CLTV) → ChurnScore = IBM's model on same inputs. Group means ≈ ORIG_proba. Zero gain. (EXP-EXT)
*   ❌ Adversarial Validation → AUC=0.512, train/test nearly identical. No actionable shift. (V14-EXP-C)
*   ❌ Isotonic Calibration → AUC is rank-invariant, calibration can't improve it. (V14-EXP-D)

### What DEFINITELY Works (The 0.916+ Playbook)
*   ✅ **Inner K-Fold Leak-Free TE** — absolutely required for trees
*   ✅ **Arithmetic Interactions** (`charges_deviation`) — essential
*   ✅ **Global Frequency Encoding** — trains well on Train+Test+Orig
*   ✅ **Original Probabilities** — target mean from original dataset
*   ✅ **TotalCharges Distribution Features** — pctrank / zscore vs original churner / non-churner (EXP3)
*   ✅ **TotalCharges Quantile Distance** — distance to Q25/Q50/Q75 of original churner/non-churner (EXP5)
*   ✅ **Bi-gram/Tri-gram Categorical TE** — composite string TE on top 6 cats. Contract×IS×OnlineSecurity = top feature (V14)

### Strategic Pivot: Version History
*   ✅ **V6 = V4 + EXP3 Integration** — LB 0.91630. 
*   ✅ **V7 = V6 + EXP5 Quantile Distance** — LB **0.91637** 🏆 Best LGBM. FE is DONE.
*   ✅ **V8 = XGBoost + V7 Features** — LB **0.91645**. XGB edges LGBM by +0.00008.
*   ✅ **V9 = TabM NN + V7 Features** — LB **0.91625** 🏆 Best NN. OOF 0.91845 nearly matches LGBM.
*   ✅ **V10 = RealMLP + V7 Features** — LB **0.91491**. TabM strictly dominates (+0.00134 LB, faster).
*   ❌ **V11 = CatBoost + V7 Features** — LB **0.91494**. Underperforms. Heavy FE saturates CatBoost's advantage.
*   ❌ **V19 = CatBoost Optuna + V16 Features** — LB **0.91648**. Even with Optuna HPO, CatBoost cannot match XGBoost V16b due to symmetric tree architecture.
*   ❌ **V20 = LightGBM Optuna + V16 Features** — LB **0.91661**. Better than V19 CatBoost (+0.00013) but still worse than XGBoost V16b (-0.00019). Leaf-wise growth doesn't beat depth-wise on heavy FE.
*   ✅ **V12 = Optuna XGBoost HPO** — LB **0.91652**. +0.00007 vs V8. Heavy regularization wins.
*   ✅ **V13 = Optuna LGBM HPO** — LB **0.91652**. Tied with V12! OOF 0.91890.
*   🏆 **V14 = Bi-gram/Tri-gram TE** — LB **0.91656** 🏆 **OVERALL BEST**. OOF 0.91889 (+0.00010). 19 composite cat TE features.
*   ❌ **V14b = Polynomial Features** — LB 0.91627. -0.00025 vs V12. Massive overfitting (gap: -0.00264).
*   ❌ **V14-DART = DART XGBoost** — Fold 1: 0.91846 (-0.00078 vs V12), 74x slower.
*   ❌ **V15 = Multi-Experiment** — Focal Loss/scale_pos_weight/colsample grid/feature pruning all ≤+0.00004.
*   ❌ **EXP-V14-MT = WOE + Curriculum PL** — WOE +0.00004 (same), CPL 0/8 rounds, calibration no effect.
*   ❌ **EXP-EXT = External ChurnScore/CLTV** — +0.00001 (dead). Group means ≈ ORIG_proba.

---

## 🆕 Phase 10: Competition-Research-Driven Ideas (2026-03-04)

**Source:** Deep analysis of 15+ Binary Classification + AUC Kaggle Playground competitions. Filtered against all dead ends above.

### Relevance Map — Competitions Studied
| Competition | Problem | Metric | Match | Key Technique Found |
|-------------|---------|:---:|:---:|---------------------|
| **S6E2** | Heart disease | AUC | ⭐⭐⭐⭐⭐ | Bi-gram/tri-gram TE, DVAE latent feats, 20-seed retrain |
| **S5E12** | Diabetes | AUC | ⭐⭐⭐⭐ | Polynomial x²/x³, ratio features, Hill Climbing ensemble |
| **S5E11** | Loan payback | AUC | ⭐⭐⭐⭐ | CatBoost single=0.924, "FE is everything" |
| **S5E8** | Bank binary | AUC | ⭐⭐⭐⭐ | NODE ensemble, genetic programming FE, 108-OOF blend, multi-encoding |
| **S5E3** | Rainfall | AUC | ⭐⭐⭐ | Standard GBDT |
| **S4E10** | Loan approval | AUC | ⭐⭐⭐ | Binary + AUC, similar techniques |
| **S4E7** | Insurance cross-sell | AUC | ⭐⭐⭐ | Categorical-heavy binary |
| **S4E1** | Bank churn | AUC | ⭐⭐⭐⭐⭐ | CatBoost 20-fold, high-cardinality encoding |
| **S3E7** | Reservation cancel | AUC | ⭐⭐⭐ | External data integration, interaction FE |
| **S3E4** | Credit fraud | AUC | ⭐⭐ | Imbalanced data (PCA features, different domain) |
| **S3E3** | Employee attrition | AUC | ⭐⭐⭐⭐ | Risk threshold indicators, XGB+LGBM+CB ensemble |
| **May 2022** | Binary classification | AUC | ⭐⭐⭐ | Ternary interaction features from feature-pair projections |
| **Apr 2022** | Binary classification | AUC | ⭐⭐⭐ | Standard GBDT ensemble |
| **Nov 2021** | Binary classification | AUC | ⭐⭐⭐ | Standard techniques |
| **Mar 2021** | Binary classification | AUC | ⭐⭐⭐ | Standard GBDT |

### 🔥 Untried Techniques (Prioritized by Expected Impact)

#### Tier 1: High Impact — Try First
*   [x] **Bi-gram / Tri-gram Categorical TE** (~15 min) — *Source: S6E2 1st place* → 🏆 **V14 = LB 0.91656 (+0.00004). OOF 0.91889 (+0.00010). NEW BEST!** Top features: TG_Contract×IS×OnlineSecurity (0.155), TG_Contract×IS×PM (0.147), BG_Contract×IS (0.138).

*   ❌ **Polynomial Features (x², x³)** (~5 min) — *Source: S5E12 1st place, S6E2*
    Result: ❌ **FAILED (V14b)**. OOF improved to 0.91891 but LB tanked to 0.91627 (-0.00025 vs V12). Massive overfitting (gap -0.00264). Polynomials fit the training noise too perfectly.

*   [ ] **Higher Fold Count (15-fold, 20-fold)** (~25 min) — *Source: S4E1 1st (20 folds), S5E10 5th ("100 folds")*
    Our current 10-fold CV has std=±0.00099. More folds → less variance in OOF → potentially tighter LB. S4E1 winner used 20 folds on CatBoost.

#### Tier 2: Medium Impact — Try Next
*   [ ] **CatBoost LIGHT (raw features only)** (~15 min) — *Source: S4E1 1st, S5E11*
    V11 failed because we used heavy FE (67 features) which saturated CatBoost's advantage. Try CatBoost with ONLY the raw 19 columns (16 cats + 3 nums) + Optuna HPO. Let CatBoost's native ordered TE and automatic feature combinations do the work.

*   [ ] **Feature Concatenation ("AllCat" trick)** (~10 min) — *Source: ideas.md Phase 7 (listed but never tried)*
    Combine all 16 categoricals into one mega-string `gender_contract_internet_...` and let model hash it. This creates a single high-cardinality categorical that captures the full customer profile.

*   [ ] **Risk Threshold Binary Indicators** (~10 min) — *Source: S3E3 1st place*
    Create binary risk flags WITH SPECIFIC THRESHOLDS derived from EDA:
    - `tenure_risk`: 1 if tenure < threshold (short tenure = high churn risk)
    - `charges_risk`: 1 if MonthlyCharges > threshold (expensive plans churn more)
    - `contract_risk`: 1 if Month-to-month
    - `aggregate_risk_score`: sum of all risk flags
    ⚠️ NOTE: We tried risk flags in EXP1/EXP2 and they were "redundant with FREQ + ORIG_prob". BUT those were generic composites, not threshold-based indicators. The S3E3 approach uses carefully chosen threshold cutoffs.

#### Tier 3: Advanced — Explore If Tier 1-2 Show Promise
*   [ ] **Denoising VAE Latent Features** (~30 min) — *Source: S6E2 1st place*
    Train a denoising variational autoencoder on all features, use latent dimensions as new features. Creates nonlinear compressed representations that increase model diversity.

*   [ ] **Genetic Programming Features** (~20 min) — *Source: S5E8 4th/15th place, S5E10 1st ("genetic programming")*
    Use `gplearn` or `featuretools` to discover novel mathematical combinations of features automatically. S5E10 1st: "I think it was genetic programming."

*   [ ] **NODE (Neural Oblivious Decision Ensembles)** (~30 min) — *Source: S5E8 10th place*
    Use NODE as a meta-model or diversity generator. Creates oblivious decision trees as neural network layers. Good for diversity with GBDT.

*   [ ] **kNN Pseudo-Features** (~15 min) — *Source: S5E8 4th place*
    For each sample, find k nearest neighbors and compute: mean target of neighbors, distance to nearest churner/non-churner, density ratio. Different from our distribution features because it's instance-level, not group-level.

*   [ ] **Multi-Encoding Diversity (TE Variants)** (~15 min) — *Source: S5E8 19th place*
    Currently we use TE mean + TE std/min/max. Try adding: TE median, TE variance, TE count, TE sum as separate feature sets. The S5E8 19th used "wide variety of encoding strategies" to create diverse model inputs.
    ⚠️ NOTE: Our V12 already has TE1_col_std/min/max (inner-fold stats). This would add median/variance/count on top. May be redundant but worth a quick test.

### ⚠️ RULES REMINDER
*   **NO ENSEMBLING / BLENDING / STACKING / MULTISEED** until explicitly requested by user.
*   **NO DART BOOSTING** — permanently dead.
*   **NO PSEUDO-LABELING** — permanently dead (0/18+ across all methods).
*   **NO EXTERNAL DATASETS** — `Telco_customer_churn.csv` confirmed dead. Delete it.

---

## 🆕 Phase 11: Deep Competition Writeup Research (2026-03-04)

**Source:** Deep dive into 10+ AUC Binary Classification Kaggle Playground competitions: S4E7 (Insurance), S4E1 (Bank Churn), S3E7 (Reservation Cancel), S3E3 (Employee Attrition), S3E24 (Smoker Biosignals), S5E8 (Bank Binary), TPS Oct/Sep/Nov 2021, TPS May/Apr 2022. Browser-fetched actual solution content.

### 🔥 New Techniques Discovered (Filtered Against Dead-End List)

#### Tier 1: High Potential — Try Next (Proven in Similar AUC competitions)

*   [x] **V15: Higher Fold Count — 20-Fold Retrain on V14** (~35 min) — *Source: S4E1 1st (20 folds on CatBoost), S5E10 5th*
    Our 10-fold OOF std=±0.00118. Retrain V14 (our best) with 20 folds. Each fold uses ~5% more training data, tighter OOF estimate, potentially smaller generalization gap. Same feature set, same params — zero extra risk.
    - **Expected gain:** ±0.00001 to ±0.00005 LB
    - **Implementation:** Change `n_splits=10` → `n_splits=20` in `S6E3_V14_BigramTE.py`
    - **Result:** LB 0.91657 (+0.00001 over V14) 🏆

*   ❌ **V15b: Numerical Binning + TE** — *Source: S4E7 2nd place, S5E11*
    **RESULT: = SAME (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91924 (delta ±0.00000). Bins recovered no signal beyond what ORIG_proba + CAT_tenure already captures. Coarser grouping = subset of existing fine-grained encoding.

*   ❌ **V15c: Composite Binary Interaction Features** — *Source: S4E7 2nd place, S3E3 1st place*
    **RESULT: ❌ WORSE (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91917 (delta -0.00007). Domain-specific boolean archetypes are a subset of what ORIG_proba captures continuously — confirmed same root cause as EXP1/EXP2 generic composites.

#### Tier 2: Medium Impact — Try After Tier 1

*   [ ] **V15d: Ordinal-Encoded Service Count Feature** (~15 min) — *Source: S5E11 loan feature engineering, S4E7 feature store diversity*
    Create a true ordinal risk score from internet/phone services:
    - `service_count` = number of active add-ons (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
    - `risky_services_ratio` = risky_services / total_services (risky = Fiber, no security add-ons)
    - ⚠️ We have `service_count` already in V3. But we NEVER did `risky_services_ratio` or a weighted risk score per service.

*   ❌ **V15e: Denoising Autoencoder Latent Features** — *Source: TPS Jan 2021 1st, TPS Oct 2021, S6E2 1st place (DVAE)*
    **RESULT: ❌ WORST (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91897 (delta **-0.00027**). DAE (29-dim input → 16-dim bottleneck, 50 epochs) added pure noise. Bottleneck too aggressive for 594K rows of mostly categorical + mixed features. Compressed representations lost signal, introduced noise to XGB.

#### Tier 3: Advanced / Speculative

*   ❌ **V15f: Full AllCat Mega-String Profile TE** — *Source: S4E7 composite string interactions*
    **RESULT: ❌ WORSE (V15f, 2026-03-05).** OOF AUC 0.91883 (delta -0.00006). Concatenating all 16 cats into one string created 44,356 unique profiles in train. This was far too sparse (avg ~13 rows per profile), so TE smoothing essentially destroyed the signal. The "curse of dimensionality" broke the TE. V14's 2-way and 3-way interactions are the true "Goldilocks zone".

*   ❌ **V15g: CatBoost LIGHT (raw features, Newton optimization)** — *Source: S4E7 2nd place*
    **RESULT: ❌ WORSE (V15g, 2026-03-05).** OOF AUC 0.91639 (delta -0.00250). Proved definitively that CatBoost's native ordered target encoding on raw features cannot compete with XGBoost using our manual Inner K-Fold TE on derived features (std/min/max spread global stats). Even with `Newton` leaf estimation, it falls short.

*   ❌ **V15h: Quantile Transform on Numericals** — *Source: OpenReview 2025*
    **RESULT: = SAME (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91924 (delta ±0.00000). XGBoost is rank-invariant — QuantileTransformer preserves rank order, which trees already see. Zero new information for tree-based models.

*   ❌ **V15i: SHAP-based Feature Elimination** — *Source: IARIA 2025*
    **RESULT: ❌ SAME/WORSE (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91919 (delta -0.00005). Not a single feature had importance below threshold (0.000). V14s 143 features all contribute meaningfully — no dead weight to remove.

*   ❌ **V15j: TabR Neural Network** — *Source: ICLR 2024, yandex-research*
    **RESULT: ❌ KILLED (2026-03-05).** Best AUC: 0.79934 at Epoch 4 (vs V14's 0.91924). Each epoch ~6 min for our 534K training fold (FAISS over all candidates every step). Estimated 20 hours for 10-fold — Kaggle limit is 9 hours. **PERMANENTLY NOT VIABLE** at our dataset scale. TabR was benchmarked on sub-100K datasets only.

*   ❌ **TabPFN v2 / TabICL** — *NOT for this dataset*
    TabPFN is a foundation model for in-context learning on tabular data. Hard limit: designed for <10K rows. Our dataset has 594K. TabICL (AutoGluon variant) might scale but expected to underperform GBDTs on this size.

### 🔬 Research Notes Added to Relevance Map
| Competition / Source | Problem | Metric | Match | New Key Technique Found |
|-------------|---------|:---:|:---:|-------------------------|
| **S4E7** | Insurance cross-sell | AUC | ⭐⭐⭐⭐ | Composite string interactions, Age/Premium binning + TE, CatBoost Newton optimization |
| **TPS Jan 2021** | Binary (synthetic) | AUC | ⭐⭐⭐ | Denoising Autoencoder (DAE) latent features fed to GBDT |
| **TPS Oct 2021** | Binary (synthetic) | AUC | ⭐⭐⭐ | DAE + KMeans cluster features + MLP ensemble |
| **S5E11** | Loan payback | AUC | ⭐⭐⭐⭐ | Grade/subgrade ordinal codes, target encoding on `loan_purpose` |
| **S3E24** | Smoker biosignals | AUC | ⭐⭐⭐ | Gender feature inference from latent dataset (our equiv = none available) |
| **ICLR 2024** | TabR paper | AUC | ⭐⭐⭐ | k-NN retrieval-augmented neural net for tabular data |
| **IARIA/preprints 2025** | SHAP stability paper | — | ⭐⭐⭐ | SHAP-based RFE outperforms gain/permutation for feature selection |
| **OpenReview 2025** | Quantile normalization | — | ⭐⭐⭐ | Structural quantile transforms cleanse skewed/artifact distributions |

---

## 🆕 Phase 12: 2024/2025 Advanced Architectures & Pipelines (2026-03-06)

**Source:** Web search for "tabular data deep learning state of the art 2024 2025" and "multi-phase tabular classification", heavily filtered against past S6E3 failures.

### 🔥 Active Implementation Roadmap

*   [ ] **Multi-Seed TabM + Stochastic Weight Averaging (SWA)** (~90 min)
    *   **Source:** ICLR 2025 TabM (Used by S5E11 5th place)
    *   **Strategy:** Our current V9 TabM uses a default single-seed. Train 5 separate seeds of TabM to ensure true randomness diversity. Use SWA (averaging weights over the last 30% of epochs) to find flatter minima in the loss landscape, vastly improving generalization. Average the 5 seed probability predictions.
    *   **Expected Gain:** +0.0008 to +0.002 LB.

*   [ ] **GBDT → NN Knowledge Distillation** (~45 min)
    *   **Source:** "Augmented Distillation for Tabular Data" - NeurIPS 2020
    *   **Strategy:** This is NOT pseudo-labeling (which failed). Extract the "soft probabilities" from our powerful V15 XGBoost teacher. Train a PyTorch Neural Network (TabM/RealMLP) to mimic these soft probabilities. Loss = `α*CE(true) + (1-α)*KL_div(teacher_probs)`. The NN captures the XGBoost decision boundary but applies its own inductive bias.
    *   **Expected Gain:** +0.0003 to +0.0007 LB.

*   [ ] **CatBoost Residual Learning (Sequential Boosting)** (~20 min)
    *   **Source:** S6E1 V75/V77 (1st place strategy)
    *   **Strategy:** Instead of standard ensembling, use CatBoost to predict the *errors* of our best XGBoost model. Convert V15 XGBoost probabilities to raw margins (log-odds). Feed these margins directly into the `baseline` parameter of a CatBoost `Pool`. Train CatBoost on the exact same feature set to correct V15's mistakes.
    *   **Expected Gain:** +0.0005 to +0.001 LB.

*   [ ] **Fast Geometric Ensembling (FGE) Snapshots** (~15 min)
    *   **Source:** NeurIPS 2018 (FGE/SWA)
    *   **Strategy:** Implement a custom callback in XGBoost to cycle the learning rate (e.g., 0.1 → 0.001 → 0.1) during a single training run. Save model snapshots at the bottom of each cycle, capturing different local minima. Average the snapshots. Provides "free" ensemble diversity without breaking the single-model rule.
    *   **Expected Gain:** +0.0005 to +0.001 LB.

*   [ ] **RankXGBoost (AUC-Maximizing Metalearning)** (~15 min)
    *   **Source:** "AUC-Maximizing Ensembles through Metalearning" - PMC 2016
    *   **Strategy:** Standard Cross-Entropy loss (Logloss) does not directly optimize Rank/AUC. Change the `objective` in `XGB_PARAMS` to `rank:pairwise`. By optimizing the pairwise ordering of samples directly, it aligns the loss function perfectly with the Kaggle AUC metric.
    *   **Expected Gain:** +0.0002 to +0.0005 LB.

*   [ ] **Two-Stage Noise Pruning (Data Cleansing)** (~20 min)
    *   **Source:** General multi-stage classification concepts for mislabeled data.
    *   **Strategy:** Train V14 (10-fold CV). For each row, calculate the log-loss of its OOF prediction vs true target. Identify the ~2% of training rows with the worst error (highly confident but wrong). Drop these impossible-to-predict outliers (synthetic noise) and retrain the entire 20-fold pipeline on the cleansed 98% dataset.

*   ❌ **Label Smoothing Regularization** (~10 min)
    *   **Source:** Original Inception Paper (Szegedy et al.)
    *   **Strategy:** Synthetic data often has noisy boundaries where the target is randomly flipped. Tree models fitting hard labels (0, 1) become overconfident. Transform `y_train` directly: `y_smooth = y_train * (1 - 0.05) + 0.5 * 0.05` (softening to 0.05 and 0.95).
    *   **Result:** ⚠️ **WORSE (OOF 0.91909 / -0.00008).** Forcing the trees to build fuzzy leaf structures prevented them from capturing the exact micro-signals required by the Kaggle synthetic generation process.

*   ❌ **Monotonic Constraints** (~5 min)
    *   **Source:** XGBoost Docs / Domain Logic (Hidden Gem)
    *   **Strategy:** Enforce monotonic relationships based on pure domain logic via `monotone_constraints` in `XGB_PARAMS`. E.g., higher `tenure` strictly decreases churn (-1). Acts as powerful "free" regularization against noisy boundaries.
    *   **Result:** ⚠️ **NEUTRAL (OOF 0.91915 / -0.00002).** XGB parameters are already highly regularized; hard constraints over-penalized micro-signals.

*   [x] **V16: Digit Features from Numericals** (~10 min)
    *   **Source:** S6E2 1st place, S5E11 1st place
    *   **Strategy:** Extract individual digit positions from numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`). Models often fail to learn exact digit-level heuristics (e.g. `tenure % 10` or rounding artifacts).
    *   **Result:** 🏆 **LB 0.91680 (V16b 20-Fold)** / **OOF 0.91925.** Massive +0.00028 OOF gain over V14! Extensive 46-feature digit set highly utilized by XGBoost.

*   ❌ **V17: Round/Binning Features + Target Encoding** (~10 min)
    *   **Source:** S5E11 1st place
    *   **Strategy:** Round continuous features into discrete buckets (e.g., `tenure` to nearest 6 months, `MonthlyCharges` to $10 buckets), then apply Inner K-Fold Target Encoding to these new discrete buckets.
    *   **Result:** ⚠️ **NEUTRAL (OOF 0.91916 / -0.00001).** Redundant with `ORIG_proba`. GBDTs already split continuous features optimally.

*   [ ] **Entropy Features from Original Dataset** (~10 min)
    *   **Source:** S6E2 1st place
    *   **Strategy:** For each categorical level, compute the *entropy* of the target distribution (`-sum(p*log(p))`) in the Original IBM dataset. Maps how "chaotic" or mixed a category is regarding churn.
    *   **Expected Gain:** +0.00005 to +0.00010 LB.

*   ❌ **Multi-Target Encoding from Original Dataset** (~15 min)
    *   **Source:** S5E11 1st place
    *   **Strategy:** The original IBM dataset contains columns beyond `Churn` that act as targets. Create Target Encoding cross-features predicting `gender` or `SeniorCitizen` based on current categoricals to extract hidden structure, then use those TE values as features for the main model.
    *   **Result:** ⚠️ **NEUTRAL (OOF 0.91918 / +0.00001).** Highly correlated with traditional TE, providing zero orthogonal lift.

*   ❌ **RGF (Regularized Greedy Forest) Model** — ❌ FAILED (2026-03-07)
    *   **Source:** S6E2 1st place
    *   **Strategy:** RGF builds trees greedily but applies L2 regularization directly on the leaf values. Very rarely used, provides excellent uncorrelated architectural diversity compared to GBDT.
    *   **Result:** ❌ FAILED — Catastrophically slow (~130 min/fold, 21+ hour ETA), AUC worse than XGBoost (0.918 vs 0.920+). Permanently dead for this competition.

*   [ ] **Bayesian Target Encoding Variance** (~15 min)
    *   **Source:** Bayesian TE Research (Hidden Gem)
## 🧪 Phase 12 Top Priority Ideas (CURRENT FOCUS)

*   [ ] **Algorithmic:** 📉 **Monotonic Constraints** — Force XGBoost to learn mathematically rigid rules (e.g., Higher Tenure = Lower Churn) to prevent trees from making noisy, overfit splits on synthetic outliers. (Next logical step since FE is saturated).
*   [ ] **Algorithmic:** 🔄 **Fast Geometric Ensembling (FGE)** — Cycle the learning rate during a single training run to capture and average multiple local loss-minima. Simulates an ensemble within a single model.
*   [ ] **Algorithmic:** 🧠 **Multi-Seed TabM + SWA** — Run PyTorch TabM across 5 seeds and apply Stochastic Weight Averaging to smooth out the Deep Learning loss landscape.
*   ✅ **Feature Selection/Targeting:** ❌ **Bayesian TE Variance (Failed in EXP18)** — Explicitly calculating TE uncertainty `(p(1-p)/N)` and giving XGBoost the sample sizes caused trees to overfit the cardinality (count) rather than the probability.
*   ✅ **Objective Function:** ❌ **Batch-Balanced Focal Loss (Skipped/Failed)** — Attempted to dynamically down-weight easy samples. `binary:logistic` and standard weighting (`min_child_weight`) are mathematically more stable for the 73:27 class ratio.

*   [ ] **Feature Interaction Constraints** (~10 min)
    *   **Source:** XGBoost Docs (Hidden Gem)
    *   **Strategy:** Explicitly control which features are allowed to interact in tree splits via `interaction_constraints`, forcing the model to focus on known predictive combinations (like `Contract`×`InternetService`×`OnlineSecurity`) and preventing spurious interactions.

*   [ ] **Diamond Feature Interaction Discovery** (~45 min)
    *   **Source:** Nature Machine Intelligence 2025
    *   **Strategy:** Statistical FDR-controlled method to find the exact n-gram combinations that matter, potentially replacing our heuristic Top-6 combination approach.

*   [ ] **Mambular (State-Space Models)** (~60 min)
    *   **Strategy:** Mambular adapts "Mamba" (SSM) architecture to tabular data.
    *   **Risk:** DL models have historically underperformed tuned XGBoost here, but SSMs are a fundamentally new paradigm.

*   [ ] **Self-Supervised Tabular Pre-training (T-JEPA/Contrastive)** (~120 min)
    *   **Source:** ICLR 2025
    *   **Strategy:** Predict representations in latent space rather than reconstructing raw features (unlike failed DAE). Much more robust to tabular noise.

*   [ ] **GBDT-NN Hybrid Architectures (GATE-Fusion)** (~90 min)
    *   **Source:** ACM 2025 / MDPI 2025
    *   **Strategy:** Extract tree structures (one-hot leaf indices or continuous leaf probabilities) from XGBoost/LightGBM ensembles, and feed them into a Neural Network or Attention layer to learn smooth non-linear combinations.

*   [ ] **LLM Embeddings for Categoricals** (~45 min)
    *   **Source:** OpenReview
    *   **Strategy:** Use a sentence transformer to convert categorical values ('Two year', 'Month-to-month') into dense semantic embeddings before feeding to the model.

#### ❌ CANCELLED / HIGH RISK (Based on Past S6E3 Failures)
*   ❌ **Density Ratio Weighting** — *CANCELLED* 
    We already ran Adversarial Validation (V14-EXP-C) which returned an AUC of 0.512, proving train and test distributions are functionally identical. Reweighting `p(test)/p(train)` will just yield 1.0 weights everywhere.
*   ❌ **Deep Feature Embedding Autoencoder** — *CANCELLED*
    Standard Autoencoders compress features. We tested this in V15e (Denoising Autoencoder) aiming for a 16-dim bottleneck, and it lowered AUC by -0.00027. S6E3's 594K categorical-heavy dataset loses too much signal in compression paradigms.
*   ❌ **FT-Transformer** — *HIGH LIKELIHOOD OF TIMEOUT*
    Like TabR (which we had to kill after 1 epoch took 6 mins), FT-Transformer tokenizes every feature into an embedding. For 594K rows × 67 features, the attention mechanism `O(N^2)` memory/time complexity will almost certainly breach the Kaggle 9-hour limit.