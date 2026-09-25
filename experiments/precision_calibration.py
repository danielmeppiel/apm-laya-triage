"""Fit-only, zero-inference calibration with development-only policy selection.

Train: python3 -m experiments.precision_calibration --round 1
Replay: python3 -m experiments.precision_calibration --replay POLICY.json
        --split confirmation --allow-confirmation --output predictions.jsonl

Only the parent should invoke confirmation replay. ``predict(scores, policy)``
accepts probabilities, never expected labels. ``metrics(records, labels)`` uses
the repository's observed-dimension silver-label agreement semantics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation import DIMENSIONS, aggregate, canonical_labels, fraction, label_metrics, scored_sets
from experiment import digest, read_json, write_json
from experiments.precision_protocol import (
    file_sha256,
    load_protocol,
    select_ids,
    verify_sources,
)

MIN_SUPPORT = 5


def load_scores(
    path: Path, numbers: list[int], labels: list[str], inputs: dict[int, dict[str, Any]]
) -> dict[int, dict[str, float]]:
    """Extract and validate only requested cached rows, ignoring all others."""
    wanted = set(numbers)
    result = {}
    with path.open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            number = row["number"]
            if number not in wanted:
                continue
            if number in result:
                raise ValueError(f"Duplicate saved prediction: {number}")
            issue = inputs[number]
            if row["input_sha256"] != digest({"title": issue["title"], "body": issue["body"]}):
                raise ValueError(f"Cached prediction input mismatch: {number}")
            if row["evidence"] != "real-local-model-inference":
                raise ValueError("Cached row is not real baseline inference.")
            answers = row["response"]["answers"]
            if set(answers) != set(labels):
                raise ValueError("Saved taxonomy mismatch.")
            probabilities = {label: answers[label]["noul"] for label in labels}
            validate_scores(probabilities, labels)
            result[number] = probabilities
    if set(result) != wanted:
        raise ValueError("Missing requested baseline predictions.")
    return result


def validate_scores(scores: dict[str, float], labels: list[str]) -> None:
    if set(scores) != set(labels):
        raise ValueError("Prediction scores must match the frozen taxonomy.")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in scores.values()):
        raise ValueError("Prediction probabilities must be finite and in [0,1].")


def load_training_data(
    snapshot: dict[str, Any], baseline: Path, protocol: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    if split not in ("fit", "development"):
        raise ValueError("Training and selection cannot access confirmation.")
    numbers = select_ids(protocol, split)
    wanted = set(numbers)
    issues = {issue["number"]: issue for issue in snapshot["issues"] if issue["number"] in wanted}
    scores = load_scores(baseline, numbers, sorted(protocol["taxonomy"]), issues)
    records = []
    for number in numbers:
        expected = sorted(canonical_labels(issues[number]["labels"], snapshot["contract"]))
        if not expected:
            raise ValueError("Protocol control no longer has an observed dimension.")
        records.append({"number": number, "expected": expected, "scores": scores[number]})
    return records


def metrics(records: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    """Include abstention coverage and supports, not precision alone."""
    controls = [record for record in records if record["expected"]]
    observed_pairs = covered_pairs = observed_issue_coverage = full_observed_coverage = 0
    for record in controls:
        observed = {label.split("/")[0] for label in record["expected"]}
        proposed = {label.split("/")[0] for label in record["proposed"]}
        observed_pairs += len(observed)
        covered_pairs += len(observed & proposed)
        observed_issue_coverage += bool(observed & proposed)
        full_observed_coverage += observed <= proposed
    by_dimension = {}
    for dimension in DIMENSIONS:
        eligible = [record for record in controls if scored_sets(record, dimension)[0]]
        by_dimension[dimension] = {
            **aggregate(controls, dimension),
            "label_support": sum(len(scored_sets(record, dimension)[0]) for record in eligible),
            "coverage": fraction(sum(bool(scored_sets(record, dimension)[1]) for record in eligible), len(eligible)),
        }
    return {
        "overall": aggregate(controls),
        "by_dimension": by_dimension,
        "per_label": {label: label_metrics(controls, label) for label in labels},
        "coverage": {
            "whole_issue": fraction(sum(bool(record["proposed"]) for record in controls), len(controls)),
            "observed_issue": fraction(observed_issue_coverage, len(controls)),
            "observed_dimension": fraction(covered_pairs, observed_pairs),
            "all_observed_dimensions": fraction(full_observed_coverage, len(controls)),
            "all_three_dimensions": fraction(sum({label.split("/")[0] for label in record["proposed"]} == set(DIMENSIONS)
                                                  for record in controls), len(controls)),
            "observed_dimension_pairs": observed_pairs,
            "covered_observed_dimension_pairs": covered_pairs,
        },
        "proposed_labels": {
            "total": sum(len(record["proposed"]) for record in controls),
            "mean": fraction(sum(len(record["proposed"]) for record in controls), len(controls)),
            "abstained_issues": sum(not record["proposed"] for record in controls),
        },
    }


def best_threshold(pairs: list[tuple[float, bool]], beta: float) -> float | None:
    """Exact fit F-beta optimum over tied score blocks; ties prefer precision."""
    if not pairs or not any(expected for _, expected in pairs):
        return None
    positives = sum(expected for _, expected in pairs)
    buckets: dict[float, list[bool]] = {}
    for score, expected in pairs:
        buckets.setdefault(score, []).append(expected)
    tp = proposed = 0
    best: tuple[float, float, float] | None = None
    threshold = None
    for score in sorted(buckets, reverse=True):
        tp += sum(buckets[score])
        proposed += len(buckets[score])
        f_beta = (1 + beta**2) * tp / (beta**2 * positives + proposed)
        key = (f_beta, tp / proposed, score)
        if best is None or key > best:
            best, threshold = key, score
    return threshold


def pairs_for(fit: list[dict[str, Any]], label: str) -> list[tuple[float, bool]]:
    dimension = label.split("/")[0]
    return [
        (record["scores"][label], label in record["expected"])
        for record in fit
        if any(expected.startswith(dimension + "/") for expected in record["expected"])
    ]


def support_counts(fit: list[dict[str, Any]], labels: list[str]) -> dict[str, int]:
    counts = Counter(label for record in fit for label in record["expected"])
    return {label: counts[label] for label in labels}


def threshold_policy(
    fit: list[dict[str, Any]], labels: list[str], beta: float, scope: str
) -> dict[str, Any]:
    support = support_counts(fit, labels)
    pairs = {label: pairs_for(fit, label) for label in labels}
    if scope == "label":
        thresholds = {label: best_threshold(pairs[label], beta) if support[label] >= MIN_SUPPORT else None
                      for label in labels}
    else:
        keys = list(DIMENSIONS) if scope == "dimension" else ["pooled"]
        shared = {
            key: best_threshold([pair for label in labels
                                 if key == "pooled" or label.startswith(key + "/")
                                 for pair in pairs[label]], beta)
            for key in keys
        }
        thresholds = {label: shared[label.split("/")[0] if scope == "dimension" else "pooled"]
                      if support[label] >= MIN_SUPPORT else None for label in labels}
    return {"kind": "threshold", "labels": labels, "thresholds": thresholds,
            "fit_beta": beta, "threshold_scope": scope, "min_positive_support": MIN_SUPPORT,
            "rare_class_rule": "Suppress labels with fewer than five fit positives; no dev-dependent fallback",
            "fit_support": support}


def fit_isotonic(pairs: list[tuple[float, bool]]) -> list[dict[str, float]]:
    """Pooled-adjacent-violators on tied score blocks, fitted only on fit."""
    groups: dict[float, list[bool]] = {}
    for score, expected in pairs:
        groups.setdefault(score, []).append(expected)
    blocks = []
    for score in sorted(groups):
        values = groups[score]
        blocks.append({"lower": score, "upper": score, "positive": sum(values), "count": len(values)})
        while len(blocks) > 1:
            left, right = blocks[-2:]
            if left["positive"] / left["count"] <= right["positive"] / right["count"]:
                break
            blocks[-2:] = [{"lower": left["lower"], "upper": right["upper"],
                           "positive": left["positive"] + right["positive"],
                           "count": left["count"] + right["count"]}]
    return [{**block, "probability": block["positive"] / block["count"]} for block in blocks]


def calibrated_probability(score: float, blocks: list[dict[str, float]]) -> float:
    if not blocks:
        raise ValueError("Cannot calibrate with no observed fit cases.")
    for index, block in enumerate(blocks[:-1]):
        boundary = (block["upper"] + blocks[index + 1]["lower"]) / 2
        if score <= boundary:
            return block["probability"]
    return blocks[-1]["probability"]


def predict(scores: dict[str, float], policy: dict[str, Any]) -> list[str]:
    labels = policy["labels"]
    validate_scores(scores, labels)
    kind = policy["kind"]
    if kind == "most_common":
        return sorted(policy["chosen"])
    if kind == "threshold":
        selected = [label for label in labels if policy["thresholds"][label] is not None
                    and scores[label] >= policy["thresholds"][label]]
        ranking = scores
    elif kind == "top_k":
        selected, ranking = labels, scores
    elif kind == "isotonic":
        supported = [label for label in labels if policy["fit_support"][label] >= policy["min_positive_support"]]
        ranking = {label: calibrated_probability(scores[label], policy["calibrators"][label]) for label in supported}
        selected = [label for label in supported if ranking[label] >= policy["probability_floor"]]
        if policy.get("minimum_per_dimension") == 1:
            for dimension in DIMENSIONS:
                candidates = [label for label in supported if label.startswith(dimension + "/") and ranking[label] > 0]
                if candidates:
                    selected.append(min(candidates, key=lambda label: (-ranking[label], label)))
            selected = sorted(set(selected))
    else:
        raise ValueError(f"Unknown frozen policy kind: {kind}")
    if policy.get("top_k") is not None:
        selected = [
            label
            for dimension in DIMENSIONS
            for label in sorted((label for label in selected if label.startswith(dimension + "/")),
                                key=lambda label: (-ranking[label], label))[
                                    :policy["top_k"][dimension] if isinstance(policy["top_k"], dict) else policy["top_k"]]
        ]
    return sorted(selected)


def fit_candidates(fit: list[dict[str, Any]], labels: list[str], round_number: int) -> dict[str, dict[str, Any]]:
    """A small explicit candidate list; no data-dependent hyperparameter sweep."""
    supports = support_counts(fit, labels)
    most_common = []
    for dimension in DIMENSIONS:
        candidates = [label for label in labels if label.startswith(dimension + "/") and supports[label]]
        if candidates:
            most_common.append(min(candidates, key=lambda label: (-supports[label], label)))
    policies = {
        "original_050": {"kind": "threshold", "labels": labels, "thresholds": {label: 0.5 for label in labels}},
        "most_common": {"kind": "most_common", "labels": labels, "chosen": most_common, "fit_support": supports},
        "raw_top1": {"kind": "top_k", "labels": labels, "top_k": 1},
        "raw_top2": {"kind": "top_k", "labels": labels, "top_k": 2},
    }
    for scope in ("pooled", "label"):
        for beta in (1.0, 0.5):
            policies[f"{scope}_f{beta:g}"] = threshold_policy(fit, labels, beta, scope)
    if round_number >= 2:
        for beta in (1.0, 0.5):
            policies[f"dimension_f{beta:g}"] = threshold_policy(fit, labels, beta, "dimension")
        calibrators = {label: fit_isotonic(pairs_for(fit, label)) for label in labels}
        for floor, cap in ((0.25, None), (0.4, None), (0.5, None), (0.0, 1), (0.4, 1), (0.5, 1)):
            policies[f"isotonic_floor{floor:g}_cap{cap}"] = {
                "kind": "isotonic", "labels": labels, "calibrators": calibrators,
                "probability_floor": floor, "top_k": cap, "fit_support": supports,
                "min_positive_support": MIN_SUPPORT,
                "rare_class_rule": "Suppress labels with fewer than five fit positives",
            }
        if round_number >= 3:
            template = policies["isotonic_floor0_cap1"]
            for floor in (0.25, 0.4, 0.5):
                policies[f"isotonic_top1_plus_second{floor:g}"] = {
                    **template, "probability_floor": floor, "minimum_per_dimension": 1, "top_k": 2,
                }
            cardinalities = {}
            for dimension in DIMENSIONS:
                observed = [len(scored_sets({**record, "proposed": []}, dimension)[0]) for record in fit]
                observed = [count for count in observed if count]
                cardinalities[dimension] = sum(observed) / len(observed) if observed else 0
            for name, rounding in (("nearest", lambda value: math.floor(value + 0.5)), ("ceil", math.ceil)):
                policies[f"isotonic_fit_mean_{name}"] = {
                    **template, "top_k": {dimension: rounding(mean) for dimension, mean in cardinalities.items()},
                    "fit_mean_observed_cardinality": cardinalities,
                }
    return policies


def evaluate_policy(records: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    return metrics([{**record, "proposed": predict(record["scores"], policy)} for record in records], policy["labels"])


def select_finalists(results: dict[str, dict[str, Any]]) -> dict[str, str | None]:
    eligible = {name: values for name, values in results.items() if name != "original_050"}
    balanced = max(eligible, key=lambda name: (eligible[name]["development"]["overall"]["micro_f1"] or 0, name))
    precision_eligible = [
        name for name, values in eligible.items()
        if (values["development"]["overall"]["recall"] or 0) >= 0.5
        and (values["development"]["coverage"]["observed_dimension"] or 0) >= 0.5
    ]
    precision = max(precision_eligible, key=lambda name: (
        eligible[name]["development"]["overall"]["precision"] or 0,
        eligible[name]["development"]["overall"]["micro_f1"] or 0, name,
    )) if precision_eligible else None
    return {"balanced": balanced, "precision_first_recall050": precision}


def run(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    verify_sources(protocol, args.snapshot, args.baseline)
    snapshot = read_json(args.snapshot)
    fit = load_training_data(snapshot, args.baseline, protocol, "fit")
    development = load_training_data(snapshot, args.baseline, protocol, "development")
    labels = sorted(protocol["taxonomy"])
    policies = fit_candidates(fit, labels, args.round)
    results = {
        name: {"policy": policy, "fit": evaluate_policy(fit, policy),
               "development": evaluate_policy(development, policy)}
        for name, policy in policies.items()
    }
    metadata = {
        "schema_version": 1,
        "round": args.round,
        "protocol_sha256": protocol["protocol_sha256"],
        "snapshot_sha256": protocol["snapshot_sha256"],
        "baseline_sha256": protocol["baseline_sha256"],
        "module_sha256": file_sha256(Path(__file__)),
        "fit_ids": select_ids(protocol, "fit"),
        "development_ids": select_ids(protocol, "development"),
        "confirmation_evaluated": False,
        "selection": "Max development micro-F1; separately max development precision with recall>=.50 and observed-dimension coverage>=.50",
        "results": results,
        "finalists": select_finalists(results),
    }
    destination = args.output / f"round{args.round}.json"
    write_json(destination, metadata)
    for name, values in results.items():
        overall, coverage = values["development"]["overall"], values["development"]["coverage"]
        print(f"{name:30s} P={overall['precision'] or 0:.4f} R={overall['recall'] or 0:.4f} "
              f"F1={overall['micro_f1'] or 0:.4f} exact={overall['exact_agreement'] or 0:.4f} "
              f"issue={coverage['whole_issue'] or 0:.4f} dim={coverage['observed_dimension'] or 0:.4f} "
              f"labels={values['development']['proposed_labels']['mean']:.2f}")
    print("Finalists:", json.dumps(metadata["finalists"]))
    if args.freeze:
        exports = {
            **{f"frozen_{objective}": name for objective, name in metadata["finalists"].items()},
            "reference_original_050": "original_050",
            "reference_most_common": "most_common",
        }
        for objective, name in exports.items():
            if name is None:
                continue
            frozen = {key: value for key, value in metadata.items() if key not in ("results", "finalists")}
            frozen.update({"objective": objective, "candidate": name, "policy": policies[name],
                           "development_metrics": results[name]["development"],
                           "source_report_sha256": file_sha256(destination)})
            frozen["artifact_sha256"] = digest(frozen)
            path = args.output / f"{objective}.json"
            if path.exists() and read_json(path) != frozen:
                raise ValueError(f"Refusing to replace frozen policy: {path}")
            write_json(path, frozen)


def load_frozen(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    frozen = read_json(path)
    if digest({key: value for key, value in frozen.items() if key != "artifact_sha256"}) != frozen["artifact_sha256"]:
        raise ValueError("Frozen policy hash mismatch.")
    if frozen["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("Frozen policy belongs to a different protocol.")
    for key in ("snapshot_sha256", "baseline_sha256"):
        if frozen[key] != protocol[key]:
            raise ValueError("Frozen policy source hash mismatch.")
    if frozen["module_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("Replay implementation differs from the frozen calibration module.")
    return frozen


def replay(args: argparse.Namespace) -> None:
    if args.split in ("confirmation", "confirmation96") and not args.allow_confirmation:
        raise ValueError("Confirmation is reserved for parent; explicit --allow-confirmation is required.")
    protocol = load_protocol(args.protocol)
    verify_sources(protocol, args.snapshot, args.baseline)
    frozen = load_frozen(args.replay, protocol)
    numbers = select_ids(protocol, args.split)
    wanted = set(numbers)
    # Expected labels are neither extracted nor sent through the prediction boundary.
    inputs = {issue["number"]: {"title": issue["title"], "body": issue["body"]}
              for issue in read_json(args.snapshot)["issues"] if issue["number"] in wanted}
    scores = load_scores(args.baseline, numbers, frozen["policy"]["labels"], inputs)
    rows = [{"number": number, "proposed": predict(scores[number], frozen["policy"]),
             "artifact_sha256": frozen["artifact_sha256"]} for number in numbers]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps({"path": str(args.output), "prediction_count": len(rows), "expected_labels_read": False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/snapshot.json"))
    parser.add_argument("--baseline", type=Path, default=Path("runs/baseline/predictions.jsonl"))
    parser.add_argument("--protocol", type=Path, default=Path("runs/precision/protocol.json"))
    parser.add_argument("--round", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--output", type=Path, default=Path("runs/precision/calibration"))
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--split", default="development")
    parser.add_argument("--allow-confirmation", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
