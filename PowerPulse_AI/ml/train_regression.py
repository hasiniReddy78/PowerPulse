"""SoH regression candidates and battery-grouped cross-validation."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml.config import RANDOM_SEED, TREE_PARAMS, XGB_PARAMS
from ml.evaluate import regression_metrics, summarise_fold_metrics
from ml.feature_engineering import DischargeFeatureBuilder


def regression_estimators() -> dict:
    return {
        "linear_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": RandomForestRegressor(**TREE_PARAMS),
        "extra_trees": ExtraTreesRegressor(**TREE_PARAMS),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=TREE_PARAMS["n_estimators"],
            min_samples_leaf=TREE_PARAMS["min_samples_leaf"],
            random_state=RANDOM_SEED,
        ),
        "xgboost": XGBRegressor(**XGB_PARAMS),
    }


def build_regressor(name: str, feature_columns: list[str]) -> Pipeline:
    estimator = regression_estimators()[name]
    return Pipeline(
        [
            ("features", DischargeFeatureBuilder(columns=list(feature_columns))),
            ("model", clone(estimator)),
        ]
    )


def cross_validate_regressor(pipeline: Pipeline, frame, y, groups, n_splits: int = 4) -> dict:
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_index, test_index in splitter.split(frame, y, groups):
        model = clone(pipeline)
        model.fit(frame.iloc[train_index], y.iloc[train_index])
        predicted = model.predict(frame.iloc[test_index])
        folds.append(regression_metrics(y.iloc[test_index], predicted))
    return summarise_fold_metrics(folds, ["mae", "rmse", "r2"])


def cross_validate_matrix(model, matrix: np.ndarray, y, groups, n_splits: int = 4) -> dict:
    """Ablation helper. The matrix is already built without target columns."""
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_index, test_index in splitter.split(matrix, y, groups):
        fitted = clone(model)
        fitted.fit(matrix[train_index], y.iloc[train_index])
        predicted = fitted.predict(matrix[test_index])
        folds.append(regression_metrics(y.iloc[test_index], predicted))
    return summarise_fold_metrics(folds, ["mae", "rmse", "r2"])


def ablation_model() -> RandomForestRegressor:
    return RandomForestRegressor(**TREE_PARAMS)


def refit_regressor(name: str, feature_columns: list[str], frame, y) -> Pipeline:
    model = build_regressor(name, feature_columns)
    model.fit(frame, y)
    return model
