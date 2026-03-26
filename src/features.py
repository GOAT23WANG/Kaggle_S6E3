import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TOP_CATS_FOR_NGRAM = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
]


def _safe_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(out["TotalCharges"].median())
    return out


def _pctrank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_sorted = np.sort(reference)
    if len(ref_sorted) == 0:
        return np.zeros(len(values), dtype=np.float32)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype(np.float32)


def _zscore_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mu = np.mean(reference)
    sigma = np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype=np.float32)
    return ((values - mu) / sigma).astype(np.float32)


def _add_digit_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    tenure = pd.to_numeric(out["tenure"], errors="coerce").fillna(0)
    monthly = pd.to_numeric(out["MonthlyCharges"], errors="coerce").fillna(0)
    total = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0)

    t_int = np.floor(tenure).astype(int)
    m_int = np.floor(monthly).astype(int)
    c_int = np.floor(total).astype(int)

    out["tenure_mod_12"] = (t_int % 12).astype(np.float32)
    out["tenure_years"] = (t_int // 12).astype(np.float32)
    out["tenure_last_digit"] = (t_int % 10).astype(np.float32)
    out["tenure_tens_digit"] = ((t_int // 10) % 10).astype(np.float32)
    out["tenure_is_multiple_10"] = (t_int % 10 == 0).astype(np.float32)
    out["tenure_rounded_10"] = (np.round(t_int / 10) * 10).astype(np.float32)
    out["tenure_dev_from_round10"] = np.abs(t_int - out["tenure_rounded_10"]).astype(np.float32)
    out["tenure_digit_count"] = pd.Series(t_int).astype(str).str.len().astype(np.float32)

    out["monthly_mod_10"] = (m_int % 10).astype(np.float32)
    out["monthly_mod_100"] = (m_int % 100).astype(np.float32)
    out["monthly_last_digit"] = (m_int % 10).astype(np.float32)
    out["monthly_tens_digit"] = ((m_int // 10) % 10).astype(np.float32)
    out["monthly_is_multiple_10"] = (m_int % 10 == 0).astype(np.float32)
    out["monthly_is_multiple_50"] = (m_int % 50 == 0).astype(np.float32)
    out["monthly_rounded_10"] = (np.round(monthly / 10) * 10).astype(np.float32)
    out["monthly_dev_from_round10"] = np.abs(monthly - out["monthly_rounded_10"]).astype(np.float32)
    out["monthly_fractional"] = (monthly - np.floor(monthly)).astype(np.float32)
    out["monthly_digit_count"] = pd.Series(m_int).astype(str).str.len().astype(np.float32)

    out["total_mod_10"] = (c_int % 10).astype(np.float32)
    out["total_mod_100"] = (c_int % 100).astype(np.float32)
    out["total_last_digit"] = (c_int % 10).astype(np.float32)
    out["total_tens_digit"] = ((c_int // 10) % 10).astype(np.float32)
    out["total_is_multiple_10"] = (c_int % 10 == 0).astype(np.float32)
    out["total_is_multiple_100"] = (c_int % 100 == 0).astype(np.float32)
    out["total_rounded_100"] = (np.round(total / 100) * 100).astype(np.float32)
    out["total_dev_from_round100"] = np.abs(total - out["total_rounded_100"]).astype(np.float32)
    out["total_fractional"] = (total - np.floor(total)).astype(np.float32)
    out["total_digit_count"] = pd.Series(c_int).astype(str).str.len().astype(np.float32)

    out["monthly_per_tenure"] = (monthly / (tenure + 1)).astype(np.float32)
    out["total_per_tenure"] = (total / (tenure + 1)).astype(np.float32)

    return out


def _add_common_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    orig: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    for col in NUMERICAL_COLS:
        freq = pd.concat([train[col], test[col], orig[col]], axis=0).value_counts(normalize=True)
        train[f"FREQ_{col}"] = train[col].map(freq).fillna(0).astype(np.float32)
        test[f"FREQ_{col}"] = test[col].map(freq).fillna(0).astype(np.float32)

    for df in [train, test]:
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype(np.float32)
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype(np.float32)
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype(np.float32)

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for df in [train, test]:
        df["service_count"] = (df[service_cols] == "Yes").sum(axis=1).astype(np.float32)
        df["has_internet"] = (df["InternetService"] != "No").astype(np.float32)
        df["has_phone"] = (df["PhoneService"] == "Yes").astype(np.float32)

    for col in CATEGORICAL_COLS + NUMERICAL_COLS:
        mapped = orig.groupby(col, observed=False)[target_col].mean()
        name = f"ORIG_proba_{col}"
        train[name] = train[col].map(mapped).fillna(0.5).astype(np.float32)
        test[name] = test[col].map(mapped).fillna(0.5).astype(np.float32)

    orig_churner_tc = orig.loc[orig[target_col] == 1, "TotalCharges"].values
    orig_nonchurner_tc = orig.loc[orig[target_col] == 0, "TotalCharges"].values
    orig_tc = orig["TotalCharges"].values
    orig_is_mc_mean = orig.groupby("InternetService", observed=False)["MonthlyCharges"].mean()
    orig_contract_values = orig["Contract"].dropna().unique()
    orig_is_values = orig["InternetService"].dropna().unique()

    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_nonchurner_TC"] = _pctrank_against(tc, orig_nonchurner_tc)
        df["pctrank_churner_TC"] = _pctrank_against(tc, orig_churner_tc)
        df["pctrank_orig_TC"] = _pctrank_against(tc, orig_tc)
        df["zscore_churn_gap_TC"] = (
            np.abs(_zscore_against(tc, orig_churner_tc)) - np.abs(_zscore_against(tc, orig_nonchurner_tc))
        ).astype(np.float32)
        df["zscore_nonchurner_TC"] = _zscore_against(tc, orig_nonchurner_tc)
        df["pctrank_churn_gap_TC"] = (
            _pctrank_against(tc, orig_churner_tc) - _pctrank_against(tc, orig_nonchurner_tc)
        ).astype(np.float32)
        df["resid_IS_MC"] = (df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc_mean).fillna(0)).astype(
            np.float32
        )

        vals = np.zeros(len(df), dtype=np.float32)
        for cat_val in orig_is_values:
            mask = df["InternetService"] == cat_val
            ref = orig.loc[orig["InternetService"] == cat_val, "TotalCharges"].values
            if len(ref) > 0 and mask.sum() > 0:
                vals[mask] = _pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
        df["cond_pctrank_IS_TC"] = vals

        vals = np.zeros(len(df), dtype=np.float32)
        for cat_val in orig_contract_values:
            mask = df["Contract"] == cat_val
            ref = orig.loc[orig["Contract"] == cat_val, "TotalCharges"].values
            if len(ref) > 0 and mask.sum() > 0:
                vals[mask] = _pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
        df["cond_pctrank_C_TC"] = vals

    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for df in [train, test]:
            df[f"dist_To_ch_{q_label}"] = np.abs(df["TotalCharges"] - ch_q).astype(np.float32)
            df[f"dist_To_nc_{q_label}"] = np.abs(df["TotalCharges"] - nc_q).astype(np.float32)
            df[f"qdist_gap_To_{q_label}"] = (df[f"dist_To_nc_{q_label}"] - df[f"dist_To_ch_{q_label}"]).astype(
                np.float32
            )

    return train, test


def _add_ngram_cols(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    test = test.copy()
    ngram_cols: list[str] = []

    top4 = TOP_CATS_FOR_NGRAM[:4]
    for i in range(len(TOP_CATS_FOR_NGRAM)):
        for j in range(i + 1, len(TOP_CATS_FOR_NGRAM)):
            c1 = TOP_CATS_FOR_NGRAM[i]
            c2 = TOP_CATS_FOR_NGRAM[j]
            col = f"BG_{c1}_{c2}"
            train[col] = (train[c1].astype(str) + "_" + train[c2].astype(str)).astype("category")
            test[col] = (test[c1].astype(str) + "_" + test[c2].astype(str)).astype("category")
            ngram_cols.append(col)

    for i in range(len(top4)):
        for j in range(i + 1, len(top4)):
            for k in range(j + 1, len(top4)):
                c1, c2, c3 = top4[i], top4[j], top4[k]
                col = f"TG_{c1}_{c2}_{c3}"
                train[col] = (
                    train[c1].astype(str) + "_" + train[c2].astype(str) + "_" + train[c3].astype(str)
                ).astype("category")
                test[col] = (
                    test[c1].astype(str) + "_" + test[c2].astype(str) + "_" + test[c3].astype(str)
                ).astype("category")
                ngram_cols.append(col)

    return train, test, ngram_cols


def build_feature_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame,
    target_col: str = "Churn",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = _safe_total_charges(train_df)
    test = _safe_total_charges(test_df)
    orig = _safe_total_charges(orig_df)

    train, test = _add_common_features(train, test, orig, target_col)
    train = _add_digit_features(train)
    test = _add_digit_features(test)

    for col in NUMERICAL_COLS:
        cat_col = f"CAT_{col}"
        train[cat_col] = train[col].astype(str).astype("category")
        test[cat_col] = test[col].astype(str).astype("category")

    train, test, ngram_cols = _add_ngram_cols(train, test)

    te_base_cols = [f"CAT_{col}" for col in NUMERICAL_COLS] + CATEGORICAL_COLS
    metadata = {
        "te_base_cols": te_base_cols,
        "ngram_cols": ngram_cols,
        "drop_raw_cols": te_base_cols + ngram_cols,
    }
    return train, test, metadata


def _inner_kfold_target_stats(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
    inner_folds: int,
    seed: int,
    stats: tuple[str, ...] = ("mean", "std", "min", "max"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    X_test = X_test.copy()

    y_train = pd.Series(y_train).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    if not isinstance(X_train.index, pd.RangeIndex):
        X_train = X_train.reset_index(drop=True)
    if not isinstance(X_valid.index, pd.RangeIndex):
        X_valid = X_valid.reset_index(drop=True)
    if not isinstance(X_test.index, pd.RangeIndex):
        X_test = X_test.reset_index(drop=True)

    for col in columns:
        for stat in stats:
            feat = f"TE_{col}_{stat}"
            X_train[feat] = np.nan

        for inner_tr_idx, inner_va_idx in skf.split(X_train, y_train):
            tr_slice = X_train.loc[inner_tr_idx, [col]].copy()
            tr_slice["_target_"] = y_train.loc[inner_tr_idx].values
            agg = tr_slice.groupby(col, observed=False)["_target_"].agg(list(stats))

            va_col = X_train.loc[inner_va_idx, col]
            for stat in stats:
                feat = f"TE_{col}_{stat}"
                mapped = pd.to_numeric(va_col.map(agg[stat]), errors="coerce")
                X_train.loc[inner_va_idx, feat] = mapped.astype(np.float32).values

        full_slice = X_train[[col]].copy()
        full_slice["_target_"] = y_train.values
        full_agg = full_slice.groupby(col, observed=False)["_target_"].agg(list(stats))

        for stat in stats:
            feat = f"TE_{col}_{stat}"
            global_fill = float(y_train.mean()) if stat == "mean" else 0.0
            X_train[feat] = pd.to_numeric(X_train[feat], errors="coerce").fillna(global_fill).astype(np.float32)
            X_valid[feat] = pd.to_numeric(X_valid[col].map(full_agg[stat]), errors="coerce").fillna(global_fill).astype(
                np.float32
            )
            X_test[feat] = pd.to_numeric(X_test[col].map(full_agg[stat]), errors="coerce").fillna(global_fill).astype(
                np.float32
            )

    return X_train, X_valid, X_test


def encode_fold_features(
    train_fold_df: pd.DataFrame,
    val_fold_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_col: str,
    te_base_cols: list[str],
    ngram_cols: list[str],
    inner_folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    X_train = train_fold_df.copy().reset_index(drop=True)
    X_valid = val_fold_df.copy().reset_index(drop=True)
    X_test = test_df.copy().reset_index(drop=True)

    y_train = X_train[target_col].copy()

    X_train, X_valid, X_test = _inner_kfold_target_stats(
        X_train,
        y_train,
        X_valid,
        X_test,
        columns=te_base_cols,
        inner_folds=inner_folds,
        seed=seed,
        stats=("mean", "std", "min", "max"),
    )

    if ngram_cols:
        X_train, X_valid, X_test = _inner_kfold_target_stats(
            X_train,
            y_train,
            X_valid,
            X_test,
            columns=ngram_cols,
            inner_folds=inner_folds,
            seed=seed,
            stats=("mean",),
        )

    drop_cols = [c for c in te_base_cols + ngram_cols if c in X_train.columns]
    for df in [X_train, X_valid, X_test]:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

    feature_cols = [c for c in X_train.columns if c not in [target_col, id_col]]
    for df in [X_train, X_valid, X_test]:
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[feature_cols] = df[feature_cols].fillna(-1).astype(np.float32)

    return X_train, X_valid, X_test, feature_cols
