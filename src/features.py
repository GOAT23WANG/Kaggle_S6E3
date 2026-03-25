from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from category_encoders import TargetEncoder
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_object_dtype,
    is_string_dtype,
)


@dataclass
class EncodingMetadata:
    categorical_columns: list[str]
    numeric_columns: list[str]


def get_categorical_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    categorical_columns: list[str] = []

    for column in feature_cols:
        series = df[column]
        if (
            is_object_dtype(series)
            or is_string_dtype(series)
            or is_categorical_dtype(series)
            or is_bool_dtype(series)
        ):
            categorical_columns.append(column)

    return categorical_columns


def _prepare_categorical_values(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_columns: list[str],
) -> None:
    for frame in (train_df, val_df, test_df):
        for column in categorical_columns:
            frame[column] = frame[column].fillna("__MISSING__").astype("string")


def _convert_feature_columns_to_numeric(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    for frame in (train_df, val_df, test_df):
        for column in feature_cols:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")


def bi_tri_target_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    target_col: str = "Churn",
    return_metadata: bool = False,
):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    categorical_columns = get_categorical_columns(train_df, cols)
    numeric_columns = [column for column in cols if column not in categorical_columns]

    if categorical_columns:
        _prepare_categorical_values(train_df, val_df, test_df, categorical_columns)
        encoder = TargetEncoder(cols=categorical_columns, smoothing=20.0)
        train_df[categorical_columns] = encoder.fit_transform(
            train_df[categorical_columns],
            train_df[target_col],
        )
        val_df[categorical_columns] = encoder.transform(val_df[categorical_columns])
        test_df[categorical_columns] = encoder.transform(test_df[categorical_columns])

    _convert_feature_columns_to_numeric(train_df, val_df, test_df, cols)

    metadata = EncodingMetadata(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )

    if return_metadata:
        return train_df, val_df, test_df, metadata

    return train_df, val_df, test_df
