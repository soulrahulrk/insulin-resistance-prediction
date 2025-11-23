"""
Custom scikit-learn wrapper for the stacking ensemble model.
"""
from sklearn.base import BaseEstimator, ClassifierMixin

class StackerWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper to make the stacking ensemble compatible with scikit-learn's API,
    particularly for use with meta-estimators like CalibratedClassifierCV.
    """
    def __init__(self, base_models=None, meta_learner=None, scaler=None):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.scaler = scaler

    def fit(self, X, y=None):
        """
        The fit method is a no-op because the stacker is assumed to be pre-trained.
        However, we must set the `classes_` attribute for scikit-learn compatibility.
        """
        if self.meta_learner is not None:
            # The meta-learner holds the class information.
            self.classes_ = self.meta_learner.classes_
        return self

    def predict_proba(self, X):
        """
        Generate class probabilities. The input X is expected to be the original
        feature set, which will be transformed into meta-features internally.
        """
        # 1. Generate base model predictions (meta-features)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models.values()):
            meta_features[:, i] = model.predict_proba(X)[:, 1]

        # 2. Scale meta-features
        scaled_meta_features = self.scaler.transform(meta_features)

        # 3. Predict with meta-learner
        return self.meta_learner.predict_proba(scaled_meta_features)

    def predict(self, X):
        """
        Generate class predictions.
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return {
            "base_models": self.base_models,
            "meta_learner": self.meta_learner,
            "scaler": self.scaler
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
