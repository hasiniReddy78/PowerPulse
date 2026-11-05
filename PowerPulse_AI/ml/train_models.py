"""Train, evaluate, and save the lithium-ion models.

Run from the PowerPulse_AI directory:

    python -m ml.train_models

The held-out batteries are scored once, after model selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from ml.config import (
    ALGORITHM_TIE_F1,
    ALGORITHM_TIE_MAE,
    BASE_FEATURES,
    CYCLE_FEATURE,
    CYCLE_MAE_IMPROVEMENT,
    DATA_CSV,
    ENGINEERED_FEATURES,
    ENGINEERED_MAE_TOLERANCE,
    GROUP_COLUMN,
    LEAKAGE_COLUMNS,
    MODELS_DIR,
    MODEL_VERSION,
    REFERENCE_CAPACITY_AH,
    REPORTS_DIR,
    SIMPLICITY_RANK,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
)
from ml.data_analysis import file_sha256
from ml.dataset import load_clean_dataset
from ml.evaluate import classification_metrics, regression_metrics
from ml.feature_engineering import DischargeFeatureBuilder, assert_no_leakage, engineer_discharge_frame
from ml.preprocessing import raw_columns_for, training_ranges
from ml.splitting import ambient_split_summary, partition_frame, split_batteries
from ml.train_classification import (
    _fit_classifier,
    build_classifier,
    classification_estimators,
    cross_validate_classifier,
    refit_classifier,
)
from ml.train_regression import (
    ablation_model,
    build_regressor,
    cross_validate_matrix,
    cross_validate_regressor,
    refit_regressor,
    regression_estimators,
)


def _select_features(train: pd.DataFrame) -> tuple[list[str], dict]:
    y = train[TARGET_REGRESSION]
    groups = train[GROUP_COLUMN]
    raw = train[BASE_FEATURES + [CYCLE_FEATURE]]
    engineered = engineer_discharge_frame(raw)
    candidates = {
        "base": list(BASE_FEATURES),
        "base_plus_engineered": list(BASE_FEATURES) + list(ENGINEERED_FEATURES),
        "base_engineered_cycle": list(BASE_FEATURES) + list(ENGINEERED_FEATURES) + [CYCLE_FEATURE],
    }
    results = {}
    model = ablation_model()
    for name, columns in candidates.items():
        assert_no_leakage(columns)
        matrix = engineered.loc[:, columns].to_numpy(dtype=float)
        scored = cross_validate_matrix(model, matrix, y, groups)
        results[name] = {
            "columns": columns,
            "mae_mean": scored["mae"]["mean"],
            "rmse_mean": scored["rmse"]["mean"],
            "r2_mean": scored["r2"]["mean"],
            "cv": scored,
        }
        print(f"ablation {name}: MAE {scored['mae']['mean']:.3f}")

    base_mae = results["base"]["mae_mean"]
    engineered_mae = results["base_plus_engineered"]["mae_mean"]
    if engineered_mae <= base_mae + ENGINEERED_MAE_TOLERANCE:
        chosen_name = "base_plus_engineered"
    else:
        chosen_name = "base"
    chosen_mae = results[chosen_name]["mae_mean"]
    cycle_mae = results["base_engineered_cycle"]["mae_mean"]
    include_cycle = chosen_name != "base" and (chosen_mae - cycle_mae) >= CYCLE_MAE_IMPROVEMENT
    # If engineered features were rejected, still test cycle against the base set.
    if chosen_name == "base" and (base_mae - cycle_mae) >= CYCLE_MAE_IMPROVEMENT:
        include_cycle = True
        columns = list(BASE_FEATURES) + [CYCLE_FEATURE]
        rationale = (
            "Engineered features increased grouped cross-validation MAE by more than "
            f"{ENGINEERED_MAE_TOLERANCE} points, so they were dropped. Discharge cycle "
            f"reduced MAE by at least {CYCLE_MAE_IMPROVEMENT} points and was kept."
        )
    elif include_cycle:
        columns = list(candidates["base_engineered_cycle"])
        rationale = (
            "Voltage range and temperature-rise features stayed within "
            f"{ENGINEERED_MAE_TOLERANCE} MAE of the base measurements. "
            f"Discharge cycle reduced grouped MAE by at least {CYCLE_MAE_IMPROVEMENT} "
            "SoH points, so it is included. The user must supply the cycle index."
        )
    else:
        columns = list(candidates[chosen_name])
        rationale = (
            "Discharge cycle was not included. Grouped cross-validation did not show "
            f"an MAE improvement of at least {CYCLE_MAE_IMPROVEMENT} SoH points, and "
            "cycle numbers are not aligned across the NASA experiments "
            "(several batteries do not start at cycle 1). "
            "Engineered features were "
            + (
                "kept because they did not make grouped MAE meaningfully worse."
                if chosen_name == "base_plus_engineered"
                else "dropped because they made grouped MAE worse."
            )
        )
    assert_no_leakage(columns)
    return columns, {"candidates": results, "selected": columns, "rationale": rationale}


def _choose(scores: dict[str, dict], metric: str, higher_is_better: bool, tie: float) -> str:
    ranked = []
    for name, payload in scores.items():
        value = payload[metric]["mean"]
        ranked.append((value, SIMPLICITY_RANK[name], name))
    if higher_is_better:
        best_value = max(item[0] for item in ranked)
        tied = [item for item in ranked if best_value - item[0] <= tie]
    else:
        best_value = min(item[0] for item in ranked)
        tied = [item for item in ranked if item[0] - best_value <= tie]
    tied.sort(key=lambda item: item[1])
    return tied[0][2]


def _importance(pipeline, feature_names: list[str]) -> dict:
    model = pipeline.named_steps["model"]
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "impurity_importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coefficients) if coefficients.ndim == 1 else np.mean(np.abs(coefficients), axis=0)
        kind = "mean_absolute_coefficient"
    else:
        return {"kind": "unavailable", "values": []}
    order = np.argsort(values)[::-1]
    return {
        "kind": kind,
        "values": [
            {"feature": feature_names[int(index)], "value": float(values[int(index)])}
            for index in order
        ],
    }


def _leave_one_battery_out(pipeline, frame, y, groups) -> dict:
    splitter = LeaveOneGroupOut()
    is_classification = isinstance(y.iloc[0], str) or pd.api.types.is_string_dtype(y)
    folds = []
    for train_index, test_index in splitter.split(frame, y, groups):
        battery_id = str(groups.iloc[test_index].iloc[0])
        if is_classification:
            model = _fit_classifier(pipeline, frame.iloc[train_index], y.iloc[train_index])
        else:
            model = clone(pipeline)
            model.fit(frame.iloc[train_index], y.iloc[train_index])
        predicted = model.predict(frame.iloc[test_index])
        if is_classification:
            present = sorted(set(y.iloc[test_index].astype(str)))
            folds.append(
                {
                    "battery_id": battery_id,
                    "accuracy": float(accuracy_score(y.iloc[test_index], predicted)),
                    "f1_macro": float(
                        f1_score(
                            y.iloc[test_index],
                            predicted,
                            labels=present,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "n": int(len(test_index)),
                }
            )
        else:
            scored = regression_metrics(y.iloc[test_index], predicted)
            folds.append(
                {
                    "battery_id": battery_id,
                    "mae": scored["mae"],
                    "rmse": scored["rmse"],
                    "r2": scored["r2"],
                    "n": scored["n"],
                }
            )
    return {"batteries": folds}


def _hyperparameters() -> dict:
    payload = {}
    for name, estimator in regression_estimators().items():
        payload[f"regression:{name}"] = _params(estimator)
    for name, estimator in classification_estimators().items():
        payload[f"classification:{name}"] = _params(estimator)
    return payload


def _params(estimator) -> dict:
    if hasattr(estimator, "get_params"):
        raw = estimator.get_params(deep=True)
        return {key: _jsonable(value) for key, value in raw.items() if not key.endswith("dtype")}
    return {}


def _convert(value):
    if isinstance(value, dict):
        return {str(key): _convert(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert(item) for item in value]
    if isinstance(value, np.ndarray):
        return _convert(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_report(path: Path, metadata: dict, evaluation: dict) -> None:
    regression = metadata["regression"]
    classification = metadata["classification"]
    lines = [
        "# Lithium-ion model evaluation",
        "",
        f"Model version: {metadata['model_version']}",
        f"Trained at (UTC): {metadata['trained_at_utc']}",
        "",
        "Selection used GroupKFold on the training batteries only. "
        "The test batteries were scored after the algorithms were fixed.",
        "",
        "## Split",
        "",
        f"- Training batteries ({len(metadata['split']['train_batteries'])}): "
        + ", ".join(metadata["split"]["train_batteries"]),
        f"- Test batteries ({len(metadata['split']['test_batteries'])}): "
        + ", ".join(metadata["split"]["test_batteries"]),
        "",
        "| Ambient °C | Train batteries | Test batteries | Train rows | Test rows |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for row in metadata["split"]["ambient"]:
        lines.append(
            f"| {row['ambient_temperature']:.0f} | {', '.join(row['train_batteries'])} | "
            f"{', '.join(row['test_batteries'])} | {row['train_records']} | {row['test_records']} |"
        )
    lines.extend(
        [
            "",
            "## Features",
            "",
            metadata["discharge_cycle_rationale"],
            "",
            "Model features: " + ", ".join(f"`{name}`" for name in metadata["feature_columns"]),
            "",
            "Excluded because they leak the target or identify the cell: "
            + ", ".join(f"`{name}`" for name in metadata["leakage_columns_excluded"] + ["battery_id"]),
            "",
            "## Regression cross-validation (training batteries)",
            "",
            "| Algorithm | MAE mean | MAE std | RMSE mean | R² mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, payload in evaluation["regression_cv"].items():
        lines.append(
            f"| {name} | {payload['mae']['mean']:.3f} | {payload['mae']['std']:.3f} | "
            f"{payload['rmse']['mean']:.3f} | {payload['r2']['mean']:.3f} |"
        )
    test = regression["test_metrics"]
    lines.extend(
        [
            "",
            f"Selected regressor: `{regression['algorithm']}`.",
            "",
            "## Regression test batteries (scored once)",
            "",
            f"- MAE: {test['mae']:.3f} percentage points",
            f"- RMSE: {test['rmse']:.3f} percentage points",
            f"- R²: {test['r2']:.3f}",
            f"- Test records: {test['n']}",
            "",
            "Leave-one-battery-out MAE on the training batteries "
            f"(diagnostic, not used for selection): "
            f"mean {evaluation['regression_lobo']['mae_mean']:.3f}, "
            f"std {evaluation['regression_lobo']['mae_std']:.3f}, "
            f"min {evaluation['regression_lobo']['mae_min']:.3f}, "
            f"max {evaluation['regression_lobo']['mae_max']:.3f}.",
            "",
            "## Classification cross-validation (training batteries)",
            "",
            "Poor is the largest class, so macro F1 is the selection metric.",
            "",
            "| Algorithm | Accuracy | Macro precision | Macro recall | Macro F1 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, payload in evaluation["classification_cv"].items():
        lines.append(
            f"| {name} | {payload['accuracy']['mean']:.3f} | {payload['precision_macro']['mean']:.3f} | "
            f"{payload['recall_macro']['mean']:.3f} | {payload['f1_macro']['mean']:.3f} |"
        )
    class_test = classification["test_metrics"]
    lines.extend(
        [
            "",
            f"Selected classifier: `{classification['algorithm']}`.",
            "",
            "## Classification test batteries (scored once)",
            "",
            f"- Accuracy: {class_test['accuracy']:.3f}",
            f"- Macro precision: {class_test['precision_macro']:.3f}",
            f"- Macro recall: {class_test['recall_macro']:.3f}",
            f"- Macro F1: {class_test['f1_macro']:.3f}",
            f"- Weighted F1: {class_test['f1_weighted']:.3f}",
            "",
            "Confusion matrix rows and columns are Good, Fair, Poor:",
            "",
            "```",
            json.dumps(class_test["confusion_matrix"]["values"]),
            "```",
            "",
            "```",
            json.dumps(class_test["classification_report"], indent=2),
            "```",
            "",
            "## Limitations",
            "",
        ]
    )
    for item in metadata["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def train(data_path: Path, output_dir: Path, version: str) -> dict:
    frame = load_clean_dataset(data_path)
    train_ids, test_ids = split_batteries(frame)
    train, test = partition_frame(frame, train_ids, test_ids)
    print(f"train batteries={len(train_ids)} rows={len(train)}")
    print(f"test batteries={len(test_ids)} rows={len(test)}")
    print("train", ", ".join(train_ids))
    print("test", ", ".join(test_ids))

    feature_columns, ablation = _select_features(train)
    raw_columns = raw_columns_for(feature_columns)
    train_raw = train[raw_columns]
    test_raw = test[raw_columns]

    regression_cv = {}
    for name in regression_estimators():
        pipeline = build_regressor(name, feature_columns)
        scored = cross_validate_regressor(
            pipeline, train_raw, train[TARGET_REGRESSION], train[GROUP_COLUMN]
        )
        regression_cv[name] = scored
        print(f"regression {name}: MAE {scored['mae']['mean']:.3f} R2 {scored['r2']['mean']:.3f}")
    regression_name = _choose(regression_cv, "mae", higher_is_better=False, tie=ALGORITHM_TIE_MAE)
    print(f"selected regression {regression_name}")

    classification_cv = {}
    for name in classification_estimators():
        pipeline = build_classifier(name, feature_columns)
        scored = cross_validate_classifier(
            pipeline, train_raw, train[TARGET_CLASSIFICATION], train[GROUP_COLUMN]
        )
        classification_cv[name] = scored
        print(f"classification {name}: macro F1 {scored['f1_macro']['mean']:.3f}")
    classification_name = _choose(
        classification_cv, "f1_macro", higher_is_better=True, tie=ALGORITHM_TIE_F1
    )
    print(f"selected classification {classification_name}")

    regressor = refit_regressor(
        regression_name, feature_columns, train_raw, train[TARGET_REGRESSION]
    )
    classifier = refit_classifier(
        classification_name, feature_columns, train_raw, train[TARGET_CLASSIFICATION]
    )
    regression_test = regression_metrics(
        test[TARGET_REGRESSION], regressor.predict(test_raw)
    )
    classification_test = classification_metrics(
        test[TARGET_CLASSIFICATION], classifier.predict(test_raw)
    )
    print(
        f"TEST regression MAE {regression_test['mae']:.3f} "
        f"RMSE {regression_test['rmse']:.3f} R2 {regression_test['r2']:.3f}"
    )
    print(
        f"TEST classification macro F1 {classification_test['f1_macro']:.3f} "
        f"accuracy {classification_test['accuracy']:.3f}"
    )

    lobo_regression = _leave_one_battery_out(
        build_regressor(regression_name, feature_columns),
        train_raw,
        train[TARGET_REGRESSION],
        train[GROUP_COLUMN],
    )
    lobo_mae = np.asarray([row["mae"] for row in lobo_regression["batteries"]], dtype=float)
    lobo_classification = _leave_one_battery_out(
        build_classifier(classification_name, feature_columns),
        train_raw,
        train[TARGET_CLASSIFICATION],
        train[GROUP_COLUMN],
    )
    lobo_f1 = np.asarray([row["f1_macro"] for row in lobo_classification["batteries"]], dtype=float)

    ranges = training_ranges(train, raw_columns)
    preprocessing = DischargeFeatureBuilder(columns=feature_columns).fit(train_raw)
    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    limitations = [
        "The model estimates SoH for one lithium-ion cell discharge summarised the way the NASA laboratory file was summarised. It does not assess a full electric-vehicle pack.",
        "Held-out scores use a small number of batteries. A low or negative R² means the model does not generalise reliably to unseen cells.",
        "Ambient temperature and discharge current are confounded with the original experiments. They are not causal maintenance advice.",
        "The 2.0 Ah reference is the assumption already used to label this file. Seven early B0006 capacities exceed 2.0 Ah and their SoH labels are capped at 100%.",
        "Class probabilities are not calibrated and are not shown as confidence.",
        "Laboratory NASA results do not validate real-world vehicle performance.",
        "Assessment is calibrated specifically for lithium-ion battery cells under laboratory discharge test conditions.",
    ]
    metadata = {
        "model_version": version,
        "project": "PowerPulse",
        "battery_chemistry": "lithium_ion",
        "dataset_file": str(data_path.name),
        "dataset_sha256": file_sha256(data_path),
        "dataset_rows": int(len(frame)),
        "dataset_batteries": int(frame[GROUP_COLUMN].nunique()),
        "reference_capacity_ah": REFERENCE_CAPACITY_AH,
        "reference_capacity_note": (
            "SoH labels in the cleaned file are min(100, capacity_ah / 2.0 * 100). "
            "The 2.0 Ah value is a preprocessing assumption."
        ),
        "random_seed": 42,
        "trained_at_utc": trained_at,
        "split": {
            "strategy": (
                "battery-wise holdout of 8 of 32 batteries, stratified by temperature profile. "
                "B0042, B0043 and B0044 are one profile because each contains both 22 °C and 4 °C discharges."
            ),
            "train_batteries": train_ids,
            "test_batteries": test_ids,
            "ambient": ambient_split_summary(frame, train_ids, test_ids),
        },
        "raw_input_columns": raw_columns,
        "feature_columns": feature_columns,
        "engineered_features": [column for column in feature_columns if column in ENGINEERED_FEATURES],
        "discharge_cycle_included": CYCLE_FEATURE in feature_columns,
        "discharge_cycle_rationale": ablation["rationale"],
        "leakage_columns_excluded": list(LEAKAGE_COLUMNS),
        "raw_feature_ranges_training": ranges,
        "hyperparameters_fixed_before_test": _hyperparameters(),
        "regression": {
            "algorithm": regression_name,
            "selection_metric": "lowest GroupKFold MAE on training batteries",
            "test_metrics": regression_test,
            "feature_importance": _importance(regressor, feature_columns),
        },
        "classification": {
            "algorithm": classification_name,
            "selection_metric": "highest GroupKFold macro F1 on training batteries",
            "test_metrics": classification_test,
            "feature_importance": _importance(classifier, feature_columns),
        },
        "class_labels": ["Good", "Fair", "Poor"],
        "condition_thresholds": {
            "poor_at_or_below": 70.0,
            "fair_at_or_below": 85.0,
            "good_above": 85.0,
        },
        "limitations": limitations,
    }
    evaluation = {
        "feature_ablation": ablation["candidates"],
        "regression_cv": regression_cv,
        "classification_cv": classification_cv,
        "selected_regression": regression_name,
        "selected_classification": classification_name,
        "regression_test": regression_test,
        "classification_test": classification_test,
        "regression_lobo": {
            "mae_mean": float(lobo_mae.mean()),
            "mae_std": float(lobo_mae.std(ddof=1)),
            "mae_min": float(lobo_mae.min()),
            "mae_max": float(lobo_mae.max()),
            "batteries": lobo_regression["batteries"],
        },
        "classification_lobo": {
            "f1_macro_mean": float(lobo_f1.mean()),
            "f1_macro_std": float(lobo_f1.std(ddof=1)),
            "f1_macro_min": float(lobo_f1.min()),
            "f1_macro_max": float(lobo_f1.max()),
            "batteries": lobo_classification["batteries"],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, output_dir / "regression_model.pkl")
    joblib.dump(classifier, output_dir / "classification_model.pkl")
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")
    (output_dir / "model_metadata.json").write_text(
        json.dumps(_convert(metadata), indent=2), encoding="utf-8"
    )
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(_convert(evaluation), indent=2), encoding="utf-8"
    )
    if output_dir.resolve() == MODELS_DIR.resolve():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        _write_report(REPORTS_DIR / "model_evaluation.md", metadata, evaluation)
    else:
        _write_report(output_dir / "model_evaluation.md", metadata, evaluation)
    return metadata


def promote(candidate_dir: Path, destination: Path = MODELS_DIR) -> None:
    missing = [
        name
        for name in (
            "regression_model.pkl",
            "classification_model.pkl",
            "preprocessing.pkl",
            "model_metadata.json",
            "evaluation_metrics.json",
        )
        if not (candidate_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Candidate is incomplete: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "regression_model.pkl",
        "classification_model.pkl",
        "preprocessing.pkl",
        "model_metadata.json",
        "evaluation_metrics.json",
    ):
        shutil.copy2(candidate_dir / name, destination / name)
    report = candidate_dir / "model_evaluation.md"
    if report.is_file():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report, REPORTS_DIR / "model_evaluation.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or promote PowerPulse models.")
    parser.add_argument("--data", type=Path, default=DATA_CSV)
    parser.add_argument("--version", default=MODEL_VERSION)
    parser.add_argument("--output", type=Path, default=MODELS_DIR)
    parser.add_argument("--promote", type=Path, default=None, help="Copy a candidate folder into models/.")
    args = parser.parse_args()
    if args.promote is not None:
        promote(args.promote, args.output)
        print(f"promoted {args.promote} -> {args.output}")
        return
    train(args.data, args.output, args.version)


if __name__ == "__main__":
    main()
