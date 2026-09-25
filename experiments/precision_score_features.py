"""Three bounded cross-question score experiments; no new neural inference.

Train with the pinned experiments/requirements-text.txt environment:
  python -m experiments.precision_score_features

Frozen JSON models replay with the standard library:
  python3 -m experiments.precision_score_features --replay PATH/frozen_balanced.json
    --split confirmation --allow-confirmation --output predictions.jsonl

Type-only features are an ablation of cached 25-question outputs, not evidence
that issuing eight live questions gives identical scores or proportional latency.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any, Callable

from experiment import digest, read_json, write_json
from experiments.precision_calibration import load_scores, load_training_data, metrics, validate_scores
from experiments.precision_protocol import file_sha256, load_protocol, select_ids, verify_sources

CONFIGURATIONS = {
    "scores25_logistic": {"kind": "logistic", "feature_scope": "all", "c": 4.0, "score_weight": 0.25},
    "type8_logistic": {"kind": "logistic", "feature_scope": "type", "c": 4.0, "score_weight": 0.25},
    "scores25_trees": {"kind": "extra_trees", "feature_scope": "all", "n_estimators": 128,
                       "max_depth": 5, "min_samples_leaf": 10, "max_features": 1.0,
                       "random_state": 0, "n_jobs": 1},
}


def feature_labels(labels: list[str], scope: str) -> list[str]:
    if scope not in ("all", "type"):
        raise ValueError("Unknown cached-score feature scope.")
    return [label for label in labels if scope == "all" or label.startswith("type/")]


def predict_probabilities(scores: dict[str, float], model: dict[str, Any]) -> dict[str, float]:
    """Portable numeric model replay, with no text, targets, or issue metadata."""
    features = model["feature_labels"]
    validate_scores(scores, features)
    values = [scores[label] for label in features]
    if model["kind"] == "logistic":
        values = [(score - mean) / scale * model["score_weight"]
                  for score, mean, scale in zip(values, model["means"], model["scales"])]
    elif model["kind"] == "extra_trees":
        # sklearn trees compare float32 features against float64 split thresholds.
        values = [struct.unpack("f", struct.pack("f", score))[0] for score in values]
    else:
        raise ValueError("Unknown exported model kind.")
    result = {}
    for label, estimator in zip(model["labels"], model["estimators"]):
        if "constant" in estimator:
            probability = estimator["constant"]
        elif model["kind"] == "logistic":
            logit = estimator["intercept"] + sum(weight * value for weight, value in zip(estimator["coefficients"], values))
            probability = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
        else:
            predictions = []
            for tree in estimator["trees"]:
                node = 0
                while tree[node][2] != -1:
                    feature, threshold, left, right, _ = tree[node]
                    node = left if values[feature] <= threshold else right
                predictions.append(tree[node][4])
            probability = sum(predictions) / len(predictions)
        result[label] = probability
    validate_scores(result, model["labels"])
    return result


def predict(scores: dict[str, float], frozen: dict[str, Any]) -> list[str]:
    probabilities = predict_probabilities(scores, frozen["model"])
    return sorted(label for label, value in probabilities.items() if value >= frozen["policy"]["threshold"])


def train_model(
    fit: list[dict[str, Any]], labels: list[str], config: dict[str, Any]
) -> tuple[dict[str, Any], Callable[[list[dict[str, Any]]], Any]]:
    """Reuse the sibling's masked OVR fit, adding only feature projection/trees."""
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier
    from experiments.precision_text import TextBaseline, TextConfig, target_arrays

    features = feature_labels(labels, config["feature_scope"])
    columns = [labels.index(label) for label in features]

    def array(records: list[dict[str, Any]], full_width: bool = False) -> Any:
        values = np.asarray([[record["scores"][label] for label in features] for record in records])
        if not full_width:
            return values
        projected = np.zeros((len(records), len(labels)))
        projected[:, columns] = values
        return projected

    model = {"kind": config["kind"], "labels": labels, "feature_labels": features,
             "configuration": config, "estimators": []}
    expected = [record["expected"] for record in fit]
    if config["kind"] == "logistic":
        native = TextBaseline(labels, TextConfig(kind="scores", c=config["c"], score_weight=config["score_weight"]))
        native.fit([{} for _ in fit], expected, array(fit, True))
        model.update({
            "means": native.scaler.mean_[columns].tolist(),
            "scales": native.scaler.scale_[columns].tolist(),
            "score_weight": config["score_weight"],
            "training_support": native.training_support,
        })
        for estimator in native.estimators:
            if isinstance(estimator, float):
                model["estimators"].append({"constant": estimator})
            else:
                model["estimators"].append({
                    "coefficients": estimator.coef_[0, columns].tolist(),
                    "intercept": float(estimator.intercept_[0]),
                })
        return model, lambda rows: native.predict([{} for _ in rows], array(rows, True))
    if config["kind"] != "extra_trees":
        raise ValueError("Unknown feature experiment kind.")
    targets, observed = target_arrays(expected, labels)
    matrix = array(fit)
    native_estimators = []
    support = {}
    for column, label in enumerate(labels):
        mask = observed[:, column]
        y = targets[mask, column]
        positives, count = int(y.sum()), len(y)
        constant = not count or positives in (0, count)
        support[label] = {
            "observed_issues": count, "positive_issues": positives, "negative_issues": count - positives,
            "fit_status": "unobserved" if not count else "constant" if constant else "learned",
        }
        if constant:
            value = positives / count if count else 0.0
            native_estimators.append(value)
            model["estimators"].append({"constant": value})
            continue
        estimator = ExtraTreesClassifier(**{key: value for key, value in config.items()
                                           if key not in ("kind", "feature_scope")}).fit(matrix[mask], y)
        native_estimators.append(estimator)
        trees = []
        for fitted in estimator.estimators_:
            tree = fitted.tree_
            trees.append([
                [int(tree.feature[node]), float(tree.threshold[node]), int(tree.children_left[node]),
                 int(tree.children_right[node]), float(tree.value[node][0][1] / tree.value[node][0].sum())]
                for node in range(tree.node_count)
            ])
        model["estimators"].append({"trees": trees})
    model["training_support"] = support

    def native_predict(rows: list[dict[str, Any]]) -> Any:
        values = array(rows)
        return np.column_stack([
            np.full(len(rows), estimator) if isinstance(estimator, float) else estimator.predict_proba(values)[:, 1]
            for estimator in native_estimators
        ])

    return model, native_predict


def exported_scores(records: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, float]]:
    return [predict_probabilities({label: record["scores"][label] for label in model["feature_labels"]}, model)
            for record in records]


def score_metrics(
    records: list[dict[str, Any]], scores: list[dict[str, float]], threshold: float, labels: list[str]
) -> dict[str, Any]:
    return metrics([
        {"number": record["number"], "expected": record["expected"],
         "proposed": sorted(label for label, value in probabilities.items() if value >= threshold)}
        for record, probabilities in zip(records, scores)
    ], labels)


def immutable_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Refusing to overwrite frozen score-feature artifact: {path}")
    write_json(path, value)


def run(args: argparse.Namespace) -> None:
    import numpy as np
    from experiments.precision_text import THRESHOLDS, choose_policies, dependencies
    import experiments.precision_text as text_module

    if args.output.exists():
        raise ValueError("Use a new output directory for immutable score-feature evidence.")
    protocol = load_protocol(args.protocol)
    verify_sources(protocol, args.snapshot, args.baseline)
    snapshot = read_json(args.snapshot)
    fit = load_training_data(snapshot, args.baseline, protocol, "fit")
    development = load_training_data(snapshot, args.baseline, protocol, "development")
    labels = sorted(protocol["taxonomy"])
    metadata = {
        "schema_version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "snapshot_sha256": protocol["snapshot_sha256"],
        "baseline_sha256": protocol["baseline_sha256"],
        "module_sha256": file_sha256(Path(__file__)),
        "text_module_sha256": file_sha256(Path(text_module.__file__)),
        "dependencies": dependencies(),
        "fit_ids": select_ids(protocol, "fit"),
        "development_ids": select_ids(protocol, "development"),
        "confirmation_evaluated": False,
        "selection": "Three model configurations; fixed sibling 37-point scalar threshold grid on development",
        "score_feature_contract": "Only named cached scores enter model features; no titles, bodies, IDs, targets or label-presence metadata",
        "rare_class_rule": "Same masked OVR constants as sibling; no extra positive-support suppression; trees regularized with min_samples_leaf=10",
    }
    results, models = {}, {}
    for name, config in CONFIGURATIONS.items():
        started = time.perf_counter()
        model, native_predict = train_model(fit, labels, config)
        fit_seconds = time.perf_counter() - started
        models[name] = model
        fit_scores, development_scores = exported_scores(fit, model), exported_scores(development, model)
        max_delta = 0.0
        for records, scores in ((fit, fit_scores), (development, development_scores)):
            exported = np.asarray([[row[label] for label in labels] for row in scores])
            native = native_predict(records)
            max_delta = max(max_delta, float(np.max(np.abs(exported - native))))
            if not np.allclose(exported, native, rtol=0, atol=1e-12):
                raise ValueError(f"Portable {name} export does not reproduce sklearn predictions.")
            if any(not np.array_equal(exported >= threshold, native >= threshold) for threshold in THRESHOLDS):
                raise ValueError(f"Portable {name} export changes a threshold decision.")
        scores_array = np.asarray([[row[label] for label in labels] for row in development_scores])
        policies, grid = choose_policies(
            [record["number"] for record in development],
            [record["expected"] for record in development], scores_array, labels,
        )
        results[name] = {
            "configuration": config, "model_sha256": digest(model), "fit_seconds": fit_seconds,
            "feature_labels": model["feature_labels"], "training_support": model["training_support"],
            "native_export_max_absolute_error": max_delta,
            "threshold_grid": grid,
            "policies": {
                objective: {"policy": policy,
                            "fit": score_metrics(fit, fit_scores, policy["threshold"], labels),
                            "development": score_metrics(development, development_scores, policy["threshold"], labels)}
                for objective, policy in policies.items()
            },
        }
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"development_scores_{name}.jsonl").open("w", encoding="utf8") as stream:
            for record, scores in zip(development, development_scores):
                stream.write(json.dumps({"number": record["number"], "scores": scores}, sort_keys=True) + "\n")
        for objective, result in results[name]["policies"].items():
            summary = result["development"]["overall"]
            print(f"{name:22s} {objective:15s} P={summary['precision']:.4f} R={summary['recall']:.4f} "
                  f"F1={summary['micro_f1']:.4f} threshold={result['policy']['threshold']:.3f} fit_s={fit_seconds:.3f}",
                  flush=True)
    finalists = {}
    for objective, metric in (("balanced", "micro_f1"), ("precision_first", "precision")):
        eligible = [name for name in results if objective in results[name]["policies"]]
        if not eligible:
            finalists[objective] = None
            continue
        finalists[objective] = max(
            eligible, key=lambda name: (
                results[name]["policies"][objective]["development"]["overall"][metric] or 0,
                results[name]["policies"][objective]["development"]["overall"]["micro_f1"] or 0, name,
            ),
        )
    report = {**metadata, "results": results, "finalists": finalists}
    report_path = args.output / "summary.json"
    write_json(report_path, report)
    for objective, name in finalists.items():
        if name is None:
            continue
        result = results[name]["policies"][objective]
        frozen = {
            **metadata, "candidate": name, "objective": objective, "policy": result["policy"],
            "model": models[name], "model_sha256": digest(models[name]),
            "development_metrics": result["development"], "source_report_sha256": file_sha256(report_path),
        }
        frozen["artifact_sha256"] = digest(frozen)
        immutable_json(args.output / f"frozen_{objective}.json", frozen)
        with (args.output / f"development_{objective}.jsonl").open("w", encoding="utf8") as stream:
            for record in development:
                scores = {label: record["scores"][label] for label in frozen["model"]["feature_labels"]}
                stream.write(json.dumps({"number": record["number"], "proposed": predict(scores, frozen),
                                         "artifact_sha256": frozen["artifact_sha256"]}, sort_keys=True) + "\n")
    print("Finalists:", json.dumps(finalists))


def load_frozen(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    frozen = read_json(path)
    if digest({key: value for key, value in frozen.items() if key != "artifact_sha256"}) != frozen["artifact_sha256"]:
        raise ValueError("Frozen score-feature artifact hash mismatch.")
    if digest(frozen["model"]) != frozen["model_sha256"]:
        raise ValueError("Frozen numeric model hash mismatch.")
    for key in ("protocol_sha256", "snapshot_sha256", "baseline_sha256"):
        if frozen[key] != protocol[key]:
            raise ValueError(f"Frozen model {key} mismatch.")
    if frozen["module_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("Score-feature replay code differs from frozen implementation.")
    return frozen


def replay(args: argparse.Namespace) -> None:
    if args.split in ("confirmation", "confirmation96") and not args.allow_confirmation:
        raise ValueError("Confirmation requires explicit parent-only --allow-confirmation.")
    protocol = load_protocol(args.protocol)
    verify_sources(protocol, args.snapshot, args.baseline)
    frozen = load_frozen(args.replay, protocol)
    numbers = select_ids(protocol, args.split)
    wanted = set(numbers)
    inputs = {issue["number"]: {"title": issue["title"], "body": issue["body"]}
              for issue in read_json(args.snapshot)["issues"] if issue["number"] in wanted}
    scores = load_scores(args.baseline, numbers, sorted(protocol["taxonomy"]), inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8") as stream:
        for number in numbers:
            selected = {label: scores[number][label] for label in frozen["model"]["feature_labels"]}
            stream.write(json.dumps({"number": number, "proposed": predict(selected, frozen),
                                     "artifact_sha256": frozen["artifact_sha256"]}, sort_keys=True) + "\n")
    print(json.dumps({"prediction_count": len(numbers), "expected_labels_read": False,
                      "features": len(frozen["model"]["feature_labels"]), "output": str(args.output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/snapshot.json"))
    parser.add_argument("--baseline", type=Path, default=Path("runs/baseline/predictions.jsonl"))
    parser.add_argument("--protocol", type=Path, default=Path("runs/precision/protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/precision/score_features"))
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--split", default="development")
    parser.add_argument("--allow-confirmation", action="store_true")
    args = parser.parse_args()
    replay(args) if args.replay else run(args)


if __name__ == "__main__":
    main()
