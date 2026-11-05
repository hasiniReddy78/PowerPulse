"""Condition classification with class-balance weighting."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ml.config import CLASS_LABELS, RANDOM_SEED, TREE_PARAMS, XGB_PARAMS
from ml.evaluate import classification_metrics, summarise_fold_metrics
from ml.feature_engineering import DischargeFeatureBuilder


class WrappedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBClassifier wrapper that encodes and decodes string class labels."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        random_state: int = RANDOM_SEED,
        n_jobs: int = 1,
        verbosity: int = 0,
        tree_method: str = "hist",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.tree_method = tree_method
        self.classes_ = np.array(CLASS_LABELS)

    def _init_estimator(self) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
            tree_method=self.tree_method,
        )

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array(CLASS_LABELS)
        label_to_int = {label: i for i, label in enumerate(self.classes_)}
        y_int = np.array([label_to_int[val] for val in y], dtype=int)
        self.estimator_ = self._init_estimator()
        self.estimator_.fit(X, y_int, sample_weight=sample_weight)
        return self

    def predict(self, X):
        preds = self.estimator_.predict(X)
        return np.array([self.classes_[int(i)] for i in preds])

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


def classification_estimators() -> dict:
    tree = {**TREE_PARAMS, "class_weight": "balanced_subsample"}
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(**tree),
        "extra_trees": ExtraTreesClassifier(**tree),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=TREE_PARAMS["n_estimators"],
            min_samples_leaf=TREE_PARAMS["min_samples_leaf"],
            random_state=RANDOM_SEED,
        ),
        "xgboost": WrappedXGBClassifier(**XGB_PARAMS),
    }


def build_classifier(name: str, feature_columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("features", DischargeFeatureBuilder(columns=list(feature_columns))),
            ("model", clone(classification_estimators()[name])),
        ]
    )


def _fit_classifier(pipeline: Pipeline, frame, y):
    model = clone(pipeline)
    final_estimator = model.named_steps["model"]
    # Logistic regression and the forests already carry class_weight.
    # Gradient boosting and XGBoost receive balanced sample weights instead.
    if isinstance(final_estimator, (GradientBoostingClassifier, WrappedXGBClassifier)) or final_estimator.__class__.__name__.startswith("XGB"):
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(frame, y, model__sample_weight=weights)
    else:
        model.fit(frame, y)
    return model


def cross_validate_classifier(pipeline: Pipeline, frame, y, groups, n_splits: int = 4) -> dict:
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_index, test_index in splitter.split(frame, y, groups):
        model = _fit_classifier(pipeline, frame.iloc[train_index], y.iloc[train_index])
        predicted = model.predict(frame.iloc[test_index])
        scored = classification_metrics(y.iloc[test_index], predicted)
        folds.append(
            {
                "accuracy": scored["accuracy"],
                "precision_macro": scored["precision_macro"],
                "recall_macro": scored["recall_macro"],
                "f1_macro": scored["f1_macro"],
                "f1_weighted": scored["f1_weighted"],
                "n": scored["n"],
            }
        )
    return summarise_fold_metrics(
        folds,
        ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"],
    )


def refit_classifier(name: str, feature_columns: list[str], frame, y) -> Pipeline:
    model = build_classifier(name, feature_columns)
    return _fit_classifier(model, frame, y)


def label_order() -> list[str]:
    return list(CLASS_LABELS)
