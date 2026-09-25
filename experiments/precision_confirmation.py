"""Paired, duplicate-group-aware comparison of frozen classification policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from evaluation import aggregate, canonical_labels, fraction, percentile, scored_sets
from experiments.precision_protocol import load_protocol, select_ids

METRICS = ("precision", "recall", "micro_f1", "exact_agreement", "issue_coverage")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predictions(path: Path, numbers: list[int], allowed: set[str]) -> dict:
    """Reject partial, duplicate, or invalid predictions instead of omitting them."""
    predictions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        number, proposed = row["number"], row["proposed"]
        if type(number) is not int or number in predictions:
            raise ValueError("Prediction IDs must be unique integers.")
        if (
            not isinstance(proposed, list)
            or any(not isinstance(label, str) for label in proposed)
            or len(proposed) != len(set(proposed))
            or not set(proposed) <= allowed
        ):
            raise ValueError("Predictions must contain unique canonical labels.")
        predictions[number] = proposed
    if set(predictions) != set(numbers):
        raise ValueError("Predictions must cover exactly the reserved evaluation IDs.")
    return predictions


def counts(record: dict) -> tuple[int, ...]:
    expected, proposed = scored_sets(record, None)
    if not expected:
        return (0,) * 8
    observed = {label.split("/", 1)[0] for label in expected}
    covered = {label.split("/", 1)[0] for label in proposed}
    return (
        len(expected & proposed),
        len(proposed - expected),
        len(expected - proposed),
        int(expected == proposed),
        1,
        int(bool(proposed)),
        len(observed & covered),
        len(observed),
    )


def rates(total: tuple | list) -> dict:
    tp, fp, fn, exact, eligible, covered, covered_dims, observed_dims = total
    return {
        "precision": fraction(tp, tp + fp),
        "recall": fraction(tp, tp + fn),
        "micro_f1": fraction(2 * tp, 2 * tp + fp + fn),
        "exact_agreement": fraction(exact, eligible),
        "issue_coverage": fraction(covered, eligible),
        "observed_dimension_coverage": fraction(covered_dims, observed_dims),
    }


def summarize(records: list[dict]) -> dict:
    result = aggregate(records)
    total = [sum(values) for values in zip(*(counts(row) for row in records))]
    result.update(rates(total))
    result["mean_labels_proposed"] = sum(len(row["proposed"]) for row in records) / len(
        records
    )
    result["by_dimension"] = {
        dim: aggregate(records, dim) for dim in ("type", "area", "theme")
    }
    return result


def compare(
    records_by_model: dict[str, list[dict]],
    groups: dict[int, str],
    reference: str,
    *,
    samples: int = 4000,
    seed: int = 20260925,
) -> dict:
    """Resample shared duplicate groups, not individual labels or repeated timings."""
    if reference not in records_by_model or len(records_by_model) < 2:
        raise ValueError("A reference and at least one candidate are required.")
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples.")
    first = next(iter(records_by_model.values()))
    numbers = [row["number"] for row in first]
    if not numbers or len(set(numbers)) != len(numbers):
        raise ValueError("Evaluation records must have unique IDs and cannot be empty.")
    expected = {row["number"]: set(row["expected"]) for row in first}
    if not set(numbers) <= set(groups):
        raise ValueError("Every issue must belong to a frozen duplicate group.")
    grouped = defaultdict(list)
    for number in numbers:
        grouped[groups[number]].append(number)
    units = list(grouped.values())
    sums = {}
    for name, records in records_by_model.items():
        if len(records) != len(numbers) or {r["number"] for r in records} != set(
            numbers
        ):
            raise ValueError("All methods must use identical paired evaluation IDs.")
        if any(set(r["expected"]) != expected[r["number"]] for r in records):
            raise ValueError("All methods must use identical reference labels.")
        lookup = {row["number"]: counts(row) for row in records}
        sums[name] = [
            [sum(lookup[number][i] for number in members) for i in range(8)]
            for members in units
        ]
    candidates = [name for name in records_by_model if name != reference]
    deltas = {name: {key: [] for key in METRICS} for name in candidates}
    generator = random.Random(seed)
    for _ in range(samples):
        draw = [generator.randrange(len(units)) for _ in units]
        sampled = {}
        for name, rows in sums.items():
            total = [sum(rows[index][i] for index in draw) for i in range(8)]
            sampled[name] = rates(total)
        for name in candidates:
            for key in METRICS:
                left, right = sampled[name][key], sampled[reference][key]
                if left is not None and right is not None:
                    deltas[name][key].append(left - right)
    points = {name: summarize(rows) for name, rows in records_by_model.items()}
    intervals = {}
    tail = 0.025 / len(candidates)
    for name in candidates:
        intervals[name] = {}
        for key, values in deltas[name].items():
            left, right = points[name][key], points[reference][key]
            intervals[name][key] = {
                "difference": left - right
                if left is not None and right is not None
                else None,
                "paired_95ci": {
                    "lower": percentile(values, 0.025),
                    "upper": percentile(values, 0.975),
                }
                if values
                else None,
                "familywise_95ci": {
                    "lower": percentile(values, tail),
                    "upper": percentile(values, 1 - tail),
                }
                if values
                else None,
                "defined_resamples": len(values),
                "undefined_resamples": samples - len(values),
            }
        precision = intervals[name]["precision"]
        f1 = intervals[name]["micro_f1"]
        intervals[name]["clear_precision_win"] = bool(
            precision["undefined_resamples"] == 0
            and f1["undefined_resamples"] == 0
            and precision["familywise_95ci"]["lower"] > 0
            and points[name]["recall"] is not None
            and points[name]["recall"] >= 0.5
            and f1["familywise_95ci"]["lower"] >= -0.02
        )
    return {
        "reference": reference,
        "issue_count": len(numbers),
        "duplicate_group_count": len(units),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "comparison_count": len(candidates),
        "metrics": points,
        "differences": intervals,
        "limitations": [
            "Agreement with incomplete silver labels, not adjudicated truth.",
            "Reserved from new experiments; historical baseline outcomes were seen.",
            "Duplicate-group bootstrap does not eliminate all topic/time dependence.",
            "Familywise intervals correct candidate comparisons, not every metric.",
            "Intervals with undefined resamples are conditional and not win evidence.",
        ],
    }


def evaluate_plan(plan_path: Path) -> dict:
    """Evaluate a coordinator-authored plan with pinned inputs and frozen artifacts."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = plan_path.parent
    for key in ("snapshot", "protocol"):
        item = plan[key]
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise ValueError(f"{key} does not match its frozen hash.")
    snapshot = json.loads((root / plan["snapshot"]["path"]).read_text("utf-8"))
    protocol = load_protocol(root / plan["protocol"]["path"])
    if plan["snapshot"]["sha256"] != protocol["snapshot_sha256"]:
        raise ValueError("Evaluation snapshot differs from the shared protocol.")
    allowed = set(snapshot["contract"]["classification_labels"])
    if plan["partition"] not in {"confirmation", "confirmation96"}:
        raise ValueError("Confirmation evaluation requires a reserved partition.")
    numbers = select_ids(protocol, plan["partition"])
    groups = {
        number: group["group_id"]
        for group in protocol["groups"]
        for number in group["issue_ids"]
    }
    scope = set(plan["dimensions"])
    if not scope or not scope <= {"type", "area", "theme"}:
        raise ValueError("Evaluation must declare valid dimensions.")
    issues = {issue["number"]: issue for issue in snapshot["issues"]}
    records_by_model = {}
    for candidate in plan["candidates"]:
        name = candidate["name"]
        if name in records_by_model:
            raise ValueError("Candidate names must be unique.")
        if candidate["selected_on"] != "development":
            raise ValueError("Candidates must be selected before confirmation.")
        if not candidate["frozen_artifacts"]:
            raise ValueError("Candidates must identify frozen code/policy artifacts.")
        for artifact in candidate["frozen_artifacts"]:
            if file_sha256(root / artifact["path"]) != artifact["sha256"]:
                raise ValueError(f"Frozen artifact changed for {name}.")
        prediction_path = root / candidate["predictions"]["path"]
        if file_sha256(prediction_path) != candidate["predictions"]["sha256"]:
            raise ValueError(f"Prediction evidence changed for {name}.")
        predictions = load_predictions(prediction_path, numbers, allowed)
        records = []
        for number in numbers:
            expected = canonical_labels(issues[number]["labels"], snapshot["contract"])
            records.append(
                {
                    "number": number,
                    "expected": sorted(
                        label for label in expected if label.split("/", 1)[0] in scope
                    ),
                    "proposed": sorted(
                        label
                        for label in predictions[number]
                        if label.split("/", 1)[0] in scope
                    ),
                }
            )
        records_by_model[name] = records
    result = compare(
        records_by_model,
        groups,
        plan["reference"],
        samples=plan.get("bootstrap_samples", 4000),
    )
    result["plan_sha256"] = file_sha256(plan_path)
    result["dimensions"] = sorted(scope)
    result["partition"] = plan["partition"]
    result["issue_numbers"] = numbers
    result["candidate_artifacts"] = plan["candidates"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_plan(args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
