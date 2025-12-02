from sklearn.base import BaseEstimator, TransformerMixin

class InteractionFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["bed_bath_interactions"] = X["bed"] * X["bath"]
        X["size_per_bed"] = X["house_size"] / (X["bed"] + 1)
        X["bath_per_bed"] = X["bath"] / (X["bed"] + 1)
        X["size_bath_interaction"] = X["house_size"] * X["bath"]
        return X
