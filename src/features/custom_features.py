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

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.freq_maps = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.freq_maps[col] = X[col].value_counts().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col + "_freq"] = X[col].map(self.freq_maps[col]).fillna(0)
        return X
