from itertools import combinations

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder as CETargetEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


TARGET_COL = "Churn"
ID_COL = "id"

CATS = [
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

NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]

TOP_CATS_FOR_NGRAM = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
]


def detect_categorical_columns(df, cols):
    cat_cols = []
    for col in cols:
        dtype_str = str(df[col].dtype).lower()
        if any(token in dtype_str for token in ["object", "string", "category", "str"]):
            cat_cols.append(col)
    return cat_cols


def bi_tri_target_encoding(train_df, val_df, test_df, cols, target_col=TARGET_COL):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    cat_cols = detect_categorical_columns(train_df, cols)
    print("categorical columns:", cat_cols)

    if cat_cols:
        encoder = CETargetEncoder(cols=cat_cols)
        train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols], train_df[target_col])
        val_df[cat_cols] = encoder.transform(val_df[cat_cols])
        test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    for df in [train_df, val_df, test_df]:
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return train_df, val_df, test_df


def prepare_original_dataframe(orig_df, target_col=TARGET_COL):
    orig = orig_df.copy()
    if target_col in orig.columns:
        orig[target_col] = orig[target_col].map({"No": 0, "Yes": 1}).astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())
    if "customerID" in orig.columns:
        orig = orig.drop(columns=["customerID"])
    return orig


def _pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def _zscore_against(values, reference):
    mean_value = np.mean(reference)
    std_value = np.std(reference)
    if std_value == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mean_value) / std_value).astype("float32")


def add_core_features(train_df, test_df, orig_df, feature_flags):
    train = train_df.copy()
    test = test_df.copy()
    orig = orig_df.copy()

    new_nums = []
    num_as_cat = []

    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for frame in [train, test, orig]:
            frame[f"FREQ_{col}"] = frame[col].map(freq).fillna(0).astype("float32")
        new_nums.append(f"FREQ_{col}")

    for frame in [train, test, orig]:
        frame["charges_deviation"] = (frame["TotalCharges"] - frame["tenure"] * frame["MonthlyCharges"]).astype("float32")
        frame["monthly_to_total_ratio"] = (frame["MonthlyCharges"] / (frame["TotalCharges"] + 1)).astype("float32")
        frame["avg_monthly_charges"] = (frame["TotalCharges"] / (frame["tenure"] + 1)).astype("float32")
    new_nums.extend(["charges_deviation", "monthly_to_total_ratio", "avg_monthly_charges"])

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
    for frame in [train, test, orig]:
        frame["service_count"] = (frame[service_cols] == "Yes").sum(axis=1).astype("float32")
        frame["has_internet"] = (frame["InternetService"] != "No").astype("float32")
        frame["has_phone"] = (frame["PhoneService"] == "Yes").astype("float32")
    new_nums.extend(["service_count", "has_internet", "has_phone"])

    if feature_flags.get("orig_proba", True):
        for col in CATS + NUMS:
            mapped = orig.groupby(col)[TARGET_COL].mean()
            feature_name = f"ORIG_proba_{col}"
            train = train.merge(mapped.rename(feature_name), on=col, how="left")
            test = test.merge(mapped.rename(feature_name), on=col, how="left")
            for frame in [train, test]:
                frame[feature_name] = frame[feature_name].fillna(0.5).astype("float32")
            new_nums.append(feature_name)

    if feature_flags.get("distribution", True):
        orig_churner_tc = orig.loc[orig[TARGET_COL] == 1, "TotalCharges"].values
        orig_nonchurner_tc = orig.loc[orig[TARGET_COL] == 0, "TotalCharges"].values
        orig_tc = orig["TotalCharges"].values
        orig_is_mc_mean = orig.groupby("InternetService")["MonthlyCharges"].mean()

        for frame in [train, test]:
            tc = frame["TotalCharges"].values
            frame["pctrank_nonchurner_TC"] = _pctrank_against(tc, orig_nonchurner_tc)
            frame["pctrank_churner_TC"] = _pctrank_against(tc, orig_churner_tc)
            frame["pctrank_orig_TC"] = _pctrank_against(tc, orig_tc)
            frame["zscore_churn_gap_TC"] = (
                np.abs(_zscore_against(tc, orig_churner_tc)) - np.abs(_zscore_against(tc, orig_nonchurner_tc))
            ).astype("float32")
            frame["zscore_nonchurner_TC"] = _zscore_against(tc, orig_nonchurner_tc)
            frame["pctrank_churn_gap_TC"] = (
                _pctrank_against(tc, orig_churner_tc) - _pctrank_against(tc, orig_nonchurner_tc)
            ).astype("float32")
            frame["resid_IS_MC"] = (
                frame["MonthlyCharges"] - frame["InternetService"].map(orig_is_mc_mean).fillna(0)
            ).astype("float32")

            values = np.zeros(len(frame), dtype="float32")
            for cat_value in orig["InternetService"].unique():
                mask = frame["InternetService"] == cat_value
                reference = orig.loc[orig["InternetService"] == cat_value, "TotalCharges"].values
                if len(reference) > 0 and mask.sum() > 0:
                    values[mask] = _pctrank_against(frame.loc[mask, "TotalCharges"].values, reference)
            frame["cond_pctrank_IS_TC"] = values

            values = np.zeros(len(frame), dtype="float32")
            for cat_value in orig["Contract"].unique():
                mask = frame["Contract"] == cat_value
                reference = orig.loc[orig["Contract"] == cat_value, "TotalCharges"].values
                if len(reference) > 0 and mask.sum() > 0:
                    values[mask] = _pctrank_against(frame.loc[mask, "TotalCharges"].values, reference)
            frame["cond_pctrank_C_TC"] = values

        new_nums.extend([
            "pctrank_nonchurner_TC",
            "zscore_churn_gap_TC",
            "pctrank_churn_gap_TC",
            "resid_IS_MC",
            "cond_pctrank_IS_TC",
            "zscore_nonchurner_TC",
            "pctrank_orig_TC",
            "pctrank_churner_TC",
            "cond_pctrank_C_TC",
        ])

        for q_label, quantile in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
            churner_quantile = np.quantile(orig_churner_tc, quantile)
            nonchurner_quantile = np.quantile(orig_nonchurner_tc, quantile)
            for frame in [train, test]:
                frame[f"dist_To_ch_{q_label}"] = np.abs(frame["TotalCharges"] - churner_quantile).astype("float32")
                frame[f"dist_To_nc_{q_label}"] = np.abs(frame["TotalCharges"] - nonchurner_quantile).astype("float32")
                frame[f"qdist_gap_To_{q_label}"] = (
                    frame[f"dist_To_nc_{q_label}"] - frame[f"dist_To_ch_{q_label}"]
                ).astype("float32")
        new_nums.extend([
            "qdist_gap_To_q50",
            "dist_To_ch_q50",
            "dist_To_nc_q50",
            "dist_To_nc_q25",
            "qdist_gap_To_q25",
            "dist_To_nc_q75",
            "dist_To_ch_q75",
            "qdist_gap_To_q75",
        ])

    for col in NUMS:
        cat_name = f"CAT_{col}"
        num_as_cat.append(cat_name)
        for frame in [train, test]:
            frame[cat_name] = frame[col].astype(str).astype("category")

    return train, test, new_nums, num_as_cat


def add_digit_features(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    digit_features = [
        "tenure_first_digit",
        "tenure_last_digit",
        "tenure_second_digit",
        "tenure_mod10",
        "tenure_mod12",
        "tenure_num_digits",
        "tenure_is_multiple_10",
        "tenure_rounded_10",
        "tenure_dev_from_round10",
        "mc_first_digit",
        "mc_last_digit",
        "mc_second_digit",
        "mc_mod10",
        "mc_mod100",
        "mc_num_digits",
        "mc_is_multiple_10",
        "mc_is_multiple_50",
        "mc_rounded_10",
        "mc_fractional",
        "mc_dev_from_round10",
        "tc_first_digit",
        "tc_last_digit",
        "tc_second_digit",
        "tc_mod10",
        "tc_mod100",
        "tc_num_digits",
        "tc_is_multiple_10",
        "tc_is_multiple_100",
        "tc_rounded_100",
        "tc_fractional",
        "tc_dev_from_round100",
        "tenure_years",
        "tenure_months_in_year",
        "mc_per_digit",
        "tc_per_digit",
    ]

    for frame in [train, test]:
        tenure_str = frame["tenure"].astype(int).astype(str)
        frame["tenure_first_digit"] = tenure_str.str[0].astype(int)
        frame["tenure_last_digit"] = tenure_str.str[-1].astype(int)
        frame["tenure_second_digit"] = tenure_str.apply(lambda value: int(value[1]) if len(value) > 1 else 0)
        frame["tenure_mod10"] = frame["tenure"] % 10
        frame["tenure_mod12"] = frame["tenure"] % 12
        frame["tenure_num_digits"] = tenure_str.str.len()
        frame["tenure_is_multiple_10"] = (frame["tenure"] % 10 == 0).astype("float32")
        frame["tenure_rounded_10"] = np.round(frame["tenure"] / 10) * 10
        frame["tenure_dev_from_round10"] = np.abs(frame["tenure"] - frame["tenure_rounded_10"])

        monthly_str = frame["MonthlyCharges"].astype(str).str.replace(".", "", regex=False)
        frame["mc_first_digit"] = monthly_str.str[0].astype(int)
        frame["mc_last_digit"] = monthly_str.str[-1].astype(int)
        frame["mc_second_digit"] = monthly_str.apply(lambda value: int(value[1]) if len(value) > 1 else 0)
        frame["mc_mod10"] = np.floor(frame["MonthlyCharges"]) % 10
        frame["mc_mod100"] = np.floor(frame["MonthlyCharges"]) % 100
        frame["mc_num_digits"] = np.floor(frame["MonthlyCharges"]).astype(int).astype(str).str.len()
        frame["mc_is_multiple_10"] = (np.floor(frame["MonthlyCharges"]) % 10 == 0).astype("float32")
        frame["mc_is_multiple_50"] = (np.floor(frame["MonthlyCharges"]) % 50 == 0).astype("float32")
        frame["mc_rounded_10"] = np.round(frame["MonthlyCharges"] / 10) * 10
        frame["mc_fractional"] = frame["MonthlyCharges"] - np.floor(frame["MonthlyCharges"])
        frame["mc_dev_from_round10"] = np.abs(frame["MonthlyCharges"] - frame["mc_rounded_10"])

        total_str = frame["TotalCharges"].astype(str).str.replace(".", "", regex=False)
        frame["tc_first_digit"] = total_str.str[0].astype(int)
        frame["tc_last_digit"] = total_str.str[-1].astype(int)
        frame["tc_second_digit"] = total_str.apply(lambda value: int(value[1]) if len(value) > 1 else 0)
        frame["tc_mod10"] = np.floor(frame["TotalCharges"]) % 10
        frame["tc_mod100"] = np.floor(frame["TotalCharges"]) % 100
        frame["tc_num_digits"] = np.floor(frame["TotalCharges"]).astype(int).astype(str).str.len()
        frame["tc_is_multiple_10"] = (np.floor(frame["TotalCharges"]) % 10 == 0).astype("float32")
        frame["tc_is_multiple_100"] = (np.floor(frame["TotalCharges"]) % 100 == 0).astype("float32")
        frame["tc_rounded_100"] = np.round(frame["TotalCharges"] / 100) * 100
        frame["tc_fractional"] = frame["TotalCharges"] - np.floor(frame["TotalCharges"])
        frame["tc_dev_from_round100"] = np.abs(frame["TotalCharges"] - frame["tc_rounded_100"])

        frame["tenure_years"] = frame["tenure"] // 12
        frame["tenure_months_in_year"] = frame["tenure"] % 12
        frame["mc_per_digit"] = frame["MonthlyCharges"] / (frame["mc_num_digits"] + 0.001)
        frame["tc_per_digit"] = frame["TotalCharges"] / (frame["tc_num_digits"] + 0.001)

        for feature in digit_features:
            frame[feature] = frame[feature].astype("float32")

    return train, test, digit_features


def add_ngram_features(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    bigram_cols = []
    trigram_cols = []

    for col1, col2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        feature_name = f"BG_{col1}_{col2}"
        for frame in [train, test]:
            frame[feature_name] = (frame[col1].astype(str) + "_" + frame[col2].astype(str)).astype("category")
        bigram_cols.append(feature_name)

    for col1, col2, col3 in combinations(TOP_CATS_FOR_NGRAM[:4], 3):
        feature_name = f"TG_{col1}_{col2}_{col3}"
        for frame in [train, test]:
            frame[feature_name] = (
                frame[col1].astype(str) + "_" + frame[col2].astype(str) + "_" + frame[col3].astype(str)
            ).astype("category")
        trigram_cols.append(feature_name)

    return train, test, bigram_cols + trigram_cols


def build_unified_v16_features(train_df, test_df, orig_df, feature_flags=None):
    feature_flags = feature_flags or {}
    flags = {
        "orig_proba": feature_flags.get("orig_proba", True),
        "distribution": feature_flags.get("distribution", True),
        "digit": feature_flags.get("digit", True),
        "ngram": feature_flags.get("ngram", True),
    }

    train = train_df.copy()
    test = test_df.copy()
    orig = prepare_original_dataframe(orig_df)

    train, test, new_nums, num_as_cat = add_core_features(train, test, orig, flags)

    digit_features = []
    if flags["digit"]:
        train, test, digit_features = add_digit_features(train, test)
        new_nums.extend(digit_features)

    ngram_cols = []
    if flags["ngram"]:
        train, test, ngram_cols = add_ngram_features(train, test)

    features = NUMS + CATS + new_nums + num_as_cat + ngram_cols
    metadata = {
        "cats": CATS,
        "nums": NUMS,
        "new_nums": new_nums,
        "num_as_cat": num_as_cat,
        "ngram_cols": ngram_cols,
        "features": features,
        "te_columns": num_as_cat + CATS,
        "te_ngram_columns": ngram_cols,
        "drop_columns": num_as_cat + CATS + ngram_cols,
        "digit_features": digit_features,
        "feature_flags": flags,
    }
    return train, test, metadata


def build_ridge_predictions(x_tr, y_tr, x_val, x_te, inner_folds=5, random_state=42):
    """Generate nested-CV Ridge predictions as an additional feature for XGB.

    Uses inner K-fold on training data to produce leak-free OOF ridge predictions,
    then fits on full training fold to predict val and test.
    All numeric columns in x_tr are used; they are StandardScaled inside.
    """
    numeric_cols = x_tr.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr[numeric_cols].fillna(0).values)
    x_val_scaled = scaler.transform(x_val[numeric_cols].fillna(0).values)
    x_te_scaled = scaler.transform(x_te[numeric_cols].fillna(0).values)

    ridge_oof = np.zeros(len(x_tr_scaled), dtype="float64")
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    for in_tr, in_va in inner_cv.split(x_tr_scaled, y_tr):
        ridge = Ridge(alpha=1.0)
        ridge.fit(x_tr_scaled[in_tr], y_tr[in_tr])
        ridge_oof[in_va] = ridge.predict(x_tr_scaled[in_va])

    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(x_tr_scaled, y_tr)
    ridge_val = ridge_full.predict(x_val_scaled)
    ridge_te = ridge_full.predict(x_te_scaled)

    return ridge_oof.astype("float32"), ridge_val.astype("float32"), ridge_te.astype("float32")