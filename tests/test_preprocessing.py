import pandas as pd
import numpy as np
from src.preprocessing import Preprocessor, add_engineered_features

SAMPLE = pd.DataFrame({
    'fasting_glucose': [100, 110, np.nan, 95],
    'fasting_insulin': [15, np.nan, 22, 11],
    'bmi': [28.1, 31.2, 26.5, 29.0],
    'age': [45, 52, 37, 61],
    'triglycerides': [150, 180, 140, 200],
    'hdl_cholesterol': [45, 40, 50, 47],
    'waist_circumference': [90, 100, 85, 95],
    'hip_circumference': [100, 110, 98, 102],
    'ir_label': [1, 1, 0, 0]
})


def test_selective_imputation():
    df = add_engineered_features(SAMPLE.drop(columns=['ir_label']))
    X = df.copy()
    y = SAMPLE['ir_label']
    pre = Preprocessor()
    X_t, cols = pre.fit_transform(X, y)
    assert 'fasting_insulin' in pre.numeric_cols
    # Fasting insulin should have no NaNs after KNN imputation
    assert not pd.isna(X_t[:, pre.numeric_cols.index('fasting_insulin')]).any()


def test_feature_selection_threshold():
    df = add_engineered_features(SAMPLE.drop(columns=['ir_label']))
    pre = Preprocessor()
    pre.fit(df, SAMPLE['ir_label'])
    # All selected features must meet threshold logic (non-empty)
    assert isinstance(pre.selected_features, list)
    assert len(pre.selected_features) > 0


def test_engineered_features_presence():
    df = add_engineered_features(SAMPLE.drop(columns=['ir_label']))
    for col in ['homa_ir', 'homa_ir_log', 'quicki', 'tg_hdl_ratio', 'waist_hip_ratio', 'age_bmi']:
        assert col in df.columns
