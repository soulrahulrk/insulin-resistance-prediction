import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.ensemble import get_oof_predictions, train_meta_learner, build_stacking_classifier

# Synthetic dataset
np.random.seed(42)
N = 120
X = pd.DataFrame({
    'f1': np.random.randn(N),
    'f2': np.random.randn(N),
    'f3': np.random.randn(N),
})
y = (X['f1'] + X['f2'] * 0.5 + np.random.randn(N) * 0.2 > 0).astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Minimal base models (two logistic regressions with different random states)
base_models = {
    'lr_a': LogisticRegression(max_iter=200, random_state=1),
    'lr_b': LogisticRegression(max_iter=200, random_state=2)
}

for name, model in base_models.items():
    model.fit(X_train, y_train)

train_meta, val_meta = get_oof_predictions(base_models, X_train, y_train, X_val, n_folds=3)
assert train_meta.shape[0] == X_train.shape[0]
assert train_meta.shape[1] == len(base_models)
assert val_meta.shape[0] == X_val.shape[0]

(meta_scaler, meta_model), best_C = train_meta_learner(train_meta, y_train.values)
stacker, val_proba = build_stacking_classifier(base_models, (meta_scaler, meta_model), train_meta, val_meta, y_val.values)

assert 'meta_feature_names' in stacker
assert len(stacker['meta_feature_names']) == len(base_models)
assert val_proba.shape[0] == X_val.shape[0]
