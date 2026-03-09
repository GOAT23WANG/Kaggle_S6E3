import pandas as pd
from category_encoders import TargetEncoder

def bi_tri_target_encoding(train_df, val_df, test_df, cols, target_col="Churn"):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    cat_cols = []
    for col in cols:
        dtype_str = str(train_df[col].dtype).lower()
        if "object" in dtype_str or "string" in dtype_str or "category" in dtype_str:
            cat_cols.append(col)

    print("categorical columns:", cat_cols)

    if cat_cols:
        encoder = TargetEncoder(cols=cat_cols)
        train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols], train_df[target_col])
        val_df[cat_cols] = encoder.transform(val_df[cat_cols])
        test_df[cat_cols] = encoder.transform(test_df[cat_cols])

    # 确保所有特征列都转成数值
    for df in [train_df, val_df, test_df]:
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return train_df, val_df, test_df