from abc import abstractmethod, ABC
from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from kindel.utils.data import featurize

Dataset = namedtuple("Dataset", ["train", "valid", "test"])
Example = namedtuple("Example", ["x", "y", "w"])


class Model(ABC):
    def __init__(self, **hyperparameters):
        self.model = self._create_model(**hyperparameters)

    @abstractmethod
    def prepare_dataset(self, df_train, df_valid, df_test): ...

    @abstractmethod
    def train(self): ...

    @abstractmethod
    def predict(self, x): ...

    @abstractmethod
    def featurize(self, df): ...

    @abstractmethod
    def _create_model(self, hyperparameters): ...


class ScikitLearnModel(Model):
    def prepare_dataset(self, df_train, df_valid, df_test):
        X_train, y_train, w_train = self.featurize(df_train)
        X_valid, y_valid, w_valid = self.featurize(df_valid)
        X_test,  y_test,  w_test  = self.featurize(df_test)
        self.data = Dataset(
            train=Example(x=X_train, y=y_train, w=w_train),
            valid=Example(x=X_valid, y=y_valid, w=w_valid),
            test=Example(x=X_test,  y=y_test,  w=w_test),
        )
        return self.data

    def train(self):
        # use sample_weight if the estimator supports it
        try:
            self.model.fit(self.data.train.x, self.data.train.y,
                           sample_weight=self.data.train.w)
        except TypeError:
            print("Warning: this model does not support sample_weight (ignored)")
            self.model.fit(self.data.train.x, self.data.train.y)

    @staticmethod
    def _add_noise_and_weight(df):
        """
        From replicate counts seq_target_*, compute mean, std, CV and
        produce both new columns + a sample weight vector (inverse CV).
        """
        df2 = df.copy()
        rep_cols = [c for c in df2.columns if c.startswith("seq_target_")]
        if not rep_cols:
            df2["_weight"] = 1.0
            return df2, np.ones(len(df2))

        means = df2[rep_cols].mean(axis=1)
        stds  = df2[rep_cols].std(axis=1)
        cv = stds / means.replace(0, np.nan)

        df2["noise_replicate_mean"] = means
        df2["noise_replicate_std"]  = stds
        df2["noise_replicate_cv"]   = cv
        df2["noise_num_zero"]       = (df2[rep_cols] == 0).sum(axis=1)

        weights = 1.0 / (1.0 + cv.fillna(np.inf))
        df2["_weight"] = weights
        return df2, weights.values

    def featurize(self, df):
        """
        Append replicate-based noise features + return sample weights.
        """
        df2, w = self._add_noise_and_weight(df)
        X, y = featurize(df2, smiles_col="smiles", label_col="y")
        return X, y, w

    def predict(self, x):
        return self.model.predict(x)


class RandomForestNoiseAware(ScikitLearnModel):
    def _create_model(self, **hyperparameters):
        return RandomForestRegressor(**hyperparameters, n_jobs=-1)


class XGBoostNoiseAware(ScikitLearnModel):
    def _create_model(self, **hyperparameters):
        return XGBRegressor(**hyperparameters)


class KNearestNeighborsNoiseAware(ScikitLearnModel):
    def _create_model(self, **hyperparameters):
        return KNeighborsRegressor(**hyperparameters)
