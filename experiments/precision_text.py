"""Small supervised silver-label baselines; no neural inference or remote calls.

The public fit/predict API deliberately separates title/body features from targets.
Split ownership, threshold selection and confirmation access belong to the shared
precision protocol, not this model. Outputs are classifier scores, not validated
probabilities of adjudicated label correctness.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from evaluation import DIMENSIONS, aggregate, canonical_labels, label_metrics, scored_sets
from experiment import digest, read_json, write_json
from experiments.precision_protocol import (
    file_sha256,
    load_protocol,
    prediction_inputs,
    select_ids,
    verify_sources,
)


THRESHOLDS = tuple(round(0.05 + 0.025 * step, 3) for step in range(37))
TITLE_TAGS = re.compile(
    r"^\s*(?:\[(?:bug|feat(?:ure)?|docs?|documentation|enhancement|refactor|"
    r"question|perf(?:ormance)?|release|architecture|automation|test(?:ing)?|"
    r"chore)\]\s*)+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextConfig:
    kind: str = "word"
    title_weight: int = 1
    c: float = 4.0
    score_weight: float = 0.25
    max_features: int = 15000
    min_df: int = 2
    neighbors: int = 5
    strip_title_tags: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"prior", "nearest", "word", "scores", "hybrid"}:
            raise ValueError(f"Unknown baseline kind: {self.kind}")
        if self.title_weight < 1 or self.min_df < 1 or self.max_features < 1:
            raise ValueError("Text configuration values must be positive.")
        if not np.isfinite(self.c) or self.c <= 0:
            raise ValueError("Logistic C must be finite and positive.")
        if not np.isfinite(self.score_weight) or self.score_weight <= 0:
            raise ValueError("Score weight must be finite and positive.")
        if self.neighbors < 1:
            raise ValueError("Neighbor count must be positive.")


def text_features(
    issue: Mapping[str, Any], title_weight: int = 1, strip_title_tags: bool = False
) -> str:
    """Whitelist text fields; never serialize metadata, controls or issue IDs."""
    title, body = issue["title"], issue["body"]
    if not isinstance(title, str) or not isinstance(body, str):
        raise ValueError("Title and body must be strings.")
    if strip_title_tags:
        title = TITLE_TAGS.sub("", title)
    text = "\n".join([title] * title_weight + [body])
    # References in the prose are identifiers too, not semantic text features.
    text = re.sub(r"https?://github\.com/[^\s/)]+/[^\s/)]+/(?:issues|pull)/\d+", " ", text)
    return re.sub(r"#\d+\b", " ", text)


def target_arrays(
    expected: Sequence[Sequence[str]], labels: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("A nonempty unique label order is required.")
    if any("/" not in label for label in labels):
        raise ValueError("Labels must have dimension/name form.")
    allowed = set(labels)
    targets = np.zeros((len(expected), len(labels)), dtype=np.int8)
    observed = np.zeros_like(targets, dtype=bool)
    for row, names in enumerate(expected):
        names = set(names)
        if names - allowed:
            raise ValueError(f"Unknown target labels: {sorted(names - allowed)}")
        dimensions = {label.split("/", 1)[0] for label in names}
        for col, label in enumerate(labels):
            targets[row, col] = label in names
            observed[row, col] = label.split("/", 1)[0] in dimensions
    return targets, observed


class TextBaseline:
    """Deterministic one-vs-rest models with per-dimension training masks."""

    def __init__(self, labels: Sequence[str], config: TextConfig = TextConfig()):
        target_arrays([], labels)
        self.labels = list(labels)
        self.config = config
        self.vectorizer: TfidfVectorizer | None = None
        self.scaler: StandardScaler | None = None
        self.estimators: list[LogisticRegression | float] = []
        self.training_support: dict[str, dict[str, Any]] = {}
        self.fit_seconds = 0.0
        self.fitted = False
        self.empty_vocabulary = False

    def _scores(self, values: Any, count: int, fitting: bool) -> sparse.csr_matrix:
        scores = np.asarray(values, dtype=np.float64)
        if scores.shape != (count, len(self.labels)):
            raise ValueError("Cached scores must align with rows and model label order.")
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Cached scores must be finite and in [0, 1].")
        if fitting:
            self.scaler = StandardScaler().fit(scores)
        if self.scaler is None:
            raise RuntimeError("Score preprocessing was not fitted.")
        return sparse.csr_matrix(self.scaler.transform(scores) * self.config.score_weight)

    def _features(
        self, issues: Sequence[Mapping[str, Any]], scores: Any, fitting: bool
    ) -> sparse.csr_matrix:
        blocks = []
        if self.config.kind in {"word", "nearest", "hybrid"}:
            documents = [
                text_features(issue, self.config.title_weight, self.config.strip_title_tags)
                for issue in issues
            ]
            if fitting:
                vectorizer = TfidfVectorizer(
                    lowercase=True,
                    token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_-]+\b",
                    ngram_range=(1, 2),
                    min_df=self.config.min_df,
                    max_features=self.config.max_features,
                    sublinear_tf=True,
                    dtype=np.float64,
                )
                analyzer = vectorizer.build_analyzer()
                frequencies: dict[str, int] = {}
                for document in documents:
                    for token in set(analyzer(document)):
                        frequencies[token] = frequencies.get(token, 0) + 1
                self.empty_vocabulary = not any(
                    count >= self.config.min_df for count in frequencies.values()
                )
                self.vectorizer = None if self.empty_vocabulary else vectorizer
            if self.vectorizer is None:
                # An explicit zero-feature block leaves only the learned intercept.
                block = sparse.csr_matrix((len(issues), 1), dtype=np.float64)
            else:
                block = (
                    self.vectorizer.fit_transform(documents)
                    if fitting
                    else self.vectorizer.transform(documents)
                )
            blocks.append(block)
        if self.config.kind in {"scores", "hybrid"}:
            blocks.append(self._scores(scores, len(issues), fitting))
        if not blocks:
            return sparse.csr_matrix((len(issues), 1), dtype=np.float64)
        return sparse.hstack(blocks, format="csr")

    def fit(
        self,
        fit_issues: Sequence[Mapping[str, Any]],
        fit_expected: Sequence[Sequence[str]],
        fit_scores: Any = None,
    ) -> TextBaseline:
        """Fit exclusively on supplied fit rows, including vocabulary and scaling."""
        if self.fitted:
            raise RuntimeError("Instantiate a new model rather than mutating a fitted model.")
        if not fit_issues or len(fit_issues) != len(fit_expected):
            raise ValueError("Nonempty aligned fit texts and targets are required.")
        started = time.perf_counter()
        targets, observed = target_arrays(fit_expected, self.labels)
        with threadpool_limits(limits=1):
            features = self._features(fit_issues, fit_scores, fitting=True)
            self.fit_targets, self.fit_observed = targets, observed
            self.fit_features = features
            for col, label in enumerate(self.labels):
                mask = observed[:, col]
                y = targets[mask, col]
                positives = int(y.sum())
                count = len(y)
                constant = count == 0 or positives in {0, count}
                self.training_support[label] = {
                    "observed_issues": count,
                    "positive_issues": positives,
                    "negative_issues": count - positives,
                    "fit_status": "unobserved" if not count else "constant" if constant else "learned",
                }
                if constant or self.config.kind in {"prior", "nearest"}:
                    estimator = positives / count if count else 0.0
                else:
                    estimator = LogisticRegression(
                        C=self.config.c,
                        solver="liblinear",
                        max_iter=1000,
                        tol=1e-6,
                        random_state=0,
                    ).fit(features[mask], y)
                    if int(estimator.n_iter_.max()) >= 1000:
                        raise RuntimeError(f"Logistic solver did not converge for {label}.")
                self.estimators.append(estimator)
        self.fit_seconds = time.perf_counter() - started
        self.fitted = True
        return self

    def predict(
        self, issues: Sequence[Mapping[str, Any]], scores: Any = None
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Fit the baseline before prediction.")
        if not issues:
            return np.zeros((0, len(self.labels)), dtype=np.float64)
        with threadpool_limits(limits=1):
            features = self._features(issues, scores, fitting=False)
            output = np.zeros((len(issues), len(self.labels)), dtype=np.float64)
            similarities = (
                (features @ self.fit_features.T).toarray()
                if self.config.kind == "nearest"
                else None
            )
            for col, estimator in enumerate(self.estimators):
                if similarities is not None:
                    eligible = np.flatnonzero(self.fit_observed[:, col])
                    output[:, col] = float(estimator)
                    if not len(eligible):
                        continue
                    for row in range(len(issues)):
                        values = similarities[row, eligible]
                        top = np.argsort(-values, kind="stable")[: self.config.neighbors]
                        weights = values[top]
                        if weights.sum() > 0:
                            output[row, col] = np.average(
                                self.fit_targets[eligible[top], col], weights=weights
                            )
                elif isinstance(estimator, float):
                    output[:, col] = estimator
                else:
                    output[:, col] = estimator.predict_proba(features)[:, 1]
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "labels": self.labels,
            "fit_seconds": self.fit_seconds,
            "training_support": self.training_support,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            "empty_vocabulary": self.empty_vocabulary,
            "cpu_threads": 1,
            "feature_fields": ["title", "body"],
            "score_semantics": "Supervised silver-reference agreement scores; not adjudicated or calibration-validated.",
            "nearest_zero_similarity": "Fit observed-dimension prevalence",
        }


def policy_predict(
    scores: Mapping[str, float], policy: Mapping[str, Any]
) -> list[str]:
    """Prediction-only frozen policy API, independent of labels or label presence."""
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("Prediction scores must be finite and in [0, 1].")
    if policy["kind"] == "threshold":
        threshold = float(policy["threshold"])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Policy threshold must be finite and in [0, 1].")
        return sorted(label for label, value in scores.items() if value >= threshold)
    if policy["kind"] == "dimension_thresholds":
        thresholds = policy["thresholds"]
        if set(thresholds) != set(DIMENSIONS):
            raise ValueError("A frozen threshold is required for every dimension.")
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds.values()):
            raise ValueError("Dimension thresholds must be finite and in [0, 1].")
        return sorted(
            label for label, value in scores.items()
            if value >= thresholds[label.split("/", 1)[0]]
        )
    if policy["kind"] == "top1_per_dimension":
        selected = []
        for dimension in DIMENSIONS:
            candidates = [label for label in scores if label.startswith(dimension + "/")]
            if candidates and max(scores[label] for label in candidates) > 0:
                selected.append(max(candidates, key=lambda label: (scores[label], label)))
        return sorted(selected)
    raise ValueError(f"Unknown frozen policy: {policy['kind']}")


def agreement_metrics(records: list[dict[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    counts = [len(record["proposed"]) for record in records]
    observed_counts = [len(scored_sets(record, None)[1]) for record in records]
    dimensions = {}
    for dimension in DIMENSIONS:
        eligible = [record for record in records if scored_sets(record, dimension)[0]]
        dimensions[dimension] = {
            **aggregate(records, dimension),
            "prediction_coverage": (
                sum(bool(scored_sets(record, dimension)[1]) for record in eligible) / len(eligible)
                if eligible else None
            ),
        }
    return {
        "overall_observed_dimensions": aggregate(records),
        "by_dimension": dimensions,
        "per_label": {label: label_metrics(records, label) for label in labels},
        "issue_prediction_coverage": sum(count > 0 for count in counts) / len(counts) if counts else None,
        "observed_issue_prediction_coverage": (
            sum(count > 0 for count in observed_counts) / len(observed_counts) if counts else None
        ),
        "mean_labels_all_dimensions": float(np.mean(counts)) if counts else None,
        "mean_labels_observed_dimensions": float(np.mean(observed_counts)) if counts else None,
        "total_labels_all_dimensions": sum(counts),
        "total_labels_observed_dimensions": sum(observed_counts),
        "unscored_proposals": sum(counts) - sum(observed_counts),
    }


def records_for(
    numbers: Sequence[int],
    expected: Sequence[Sequence[str]],
    scores: np.ndarray,
    labels: Sequence[str],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if len(numbers) != len(expected) or scores.shape != (len(numbers), len(labels)):
        raise ValueError("Prediction records require exactly aligned rows and labels.")
    return [
        {"number": number, "expected": sorted(names),
         "proposed": policy_predict(dict(zip(labels, row.tolist())), policy)}
        for number, names, row in zip(numbers, expected, scores)
    ]


def choose_policies(
    numbers: Sequence[int],
    expected: Sequence[Sequence[str]],
    scores: np.ndarray,
    labels: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    for threshold in THRESHOLDS:
        policy = {"kind": "threshold", "threshold": threshold}
        result = aggregate(records_for(numbers, expected, scores, labels, policy))
        candidates.append({"policy": policy, "metrics": result})
    balanced = max(
        candidates,
        key=lambda row: (
            row["metrics"]["micro_f1"] or 0,
            row["metrics"]["precision"] or 0,
            row["metrics"]["recall"] or 0,
            row["policy"]["threshold"],
        ),
    )
    feasible = [row for row in candidates if (row["metrics"]["recall"] or 0) >= 0.5]
    precision = max(
        feasible,
        key=lambda row: (
            row["metrics"]["precision"] or 0,
            row["metrics"]["micro_f1"] or 0,
            row["metrics"]["recall"] or 0,
            row["policy"]["threshold"],
        ),
    ) if feasible else None
    policies = {"balanced": balanced["policy"]}
    if precision is not None:
        policies["precision_first"] = precision["policy"]
    return policies, candidates


def choose_dimension_policy(
    numbers: Sequence[int], expected: Sequence[Sequence[str]], scores: np.ndarray,
    labels: Sequence[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Three scalar dev choices, each requiring recall >= .50 in its dimension."""
    thresholds, evidence = {}, {}
    for dimension in DIMENSIONS:
        candidates = []
        for threshold in THRESHOLDS:
            policy = {"kind": "threshold", "threshold": threshold}
            result = aggregate(records_for(numbers, expected, scores, labels, policy), dimension)
            candidates.append({"threshold": threshold, "metrics": result})
        feasible = [row for row in candidates if (row["metrics"]["recall"] or 0) >= 0.5]
        if not feasible:
            evidence[dimension] = {"feasible": False, "candidates": candidates}
            continue
        winner = max(
            feasible,
            key=lambda row: (
                row["metrics"]["precision"] or 0, row["metrics"]["micro_f1"] or 0,
                row["metrics"]["recall"] or 0, row["threshold"],
            ),
        )
        thresholds[dimension] = winner["threshold"]
        evidence[dimension] = {"feasible": True, "selected": winner, "candidates": candidates}
    if len(thresholds) != len(DIMENSIONS):
        return None, evidence
    return {"kind": "dimension_thresholds", "thresholds": thresholds}, evidence


def read_selected_scores(
    path: Path, issues: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> np.ndarray:
    lookup = {issue["number"]: issue for issue in issues}
    if len(lookup) != len(issues):
        raise ValueError("Input issue IDs must be unique.")
    found = {}
    with path.open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            number = row["number"]
            if number not in lookup:
                continue
            if number in found:
                raise ValueError(f"Duplicate cached score row: {number}")
            issue = lookup[number]
            if row["input_sha256"] != digest({"title": issue["title"], "body": issue["body"]}):
                raise ValueError(f"Cached input text mismatch: {number}")
            answers = row["response"]["answers"]
            if set(answers) != set(labels):
                raise ValueError("Cached score taxonomy mismatch.")
            values = [answers[label]["noul"] for label in labels]
            if any(type(value) not in {int, float} for value in values):
                raise ValueError("Cached scores must be numeric.")
            found[number] = values
    if set(found) != set(lookup):
        raise ValueError(f"Missing cached score rows: {sorted(set(lookup) - set(found))}")
    scores = np.asarray([found[issue["number"]] for issue in issues], dtype=np.float64)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Cached scores must be finite and in [0, 1].")
    return scores


def fit_from_protocol(
    protocol: Mapping[str, Any], snapshot: Mapping[str, Any],
    baseline_path: Path, config: TextConfig,
) -> TextBaseline:
    fit_ids = select_ids(protocol, "fit")
    issues = {issue["number"]: issue for issue in snapshot["issues"]}
    fit_issues = [issues[number] for number in fit_ids]
    labels = list(protocol["taxonomy"])
    expected = [
        sorted(canonical_labels(issue["labels"], snapshot["contract"])) for issue in fit_issues
    ]
    inputs = prediction_inputs(fit_issues)
    scores = (
        read_selected_scores(baseline_path, inputs, labels)
        if config.kind in {"scores", "hybrid"} else None
    )
    return TextBaseline(labels, config).fit(inputs, expected, scores)


def predict_records(
    model: TextBaseline, issues: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any], scores: Any = None,
) -> list[dict[str, Any]]:
    values = model.predict(issues, scores)
    return [
        {"number": issue["number"],
         "proposed": policy_predict(dict(zip(model.labels, row.tolist())), policy)}
        for issue, row in zip(issues, values)
    ]


def dependencies() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")
    }


def write_prediction_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"number": row["number"], "proposed": row["proposed"]}) + "\n"
                for row in records),
        encoding="utf8",
    )


def derive_dimension_policies(root: Path, runs: Sequence[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Choose a new directory for immutable dimension-policy evidence.")
    protocol = load_protocol(root / "runs/precision/protocol.json")
    verify_sources(protocol, root / "data/snapshot.json", root / "runs/baseline/predictions.jsonl")
    labels = list(protocol["taxonomy"])
    ids = select_ids(protocol, "development")
    snapshot = read_json(root / "data/snapshot.json")
    issues = {issue["number"]: issue for issue in snapshot["issues"]}
    expected = [
        sorted(canonical_labels(issues[number]["labels"], snapshot["contract"])) for number in ids
    ]
    results = {}
    for run in runs:
        summary = read_json(run / "summary.json")
        if summary["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("Run belongs to a different experimental protocol.")
        rows = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]
        if [row["number"] for row in rows] != ids:
            raise ValueError("Development score rows do not match exact protocol order.")
        if any(set(row["scores"]) != set(labels) for row in rows):
            raise ValueError("Development score taxonomy mismatch.")
        values = np.asarray([[row["scores"][label] for label in labels] for row in rows])
        policy, trace = choose_dimension_policy(ids, expected, values, labels)
        key = run.name
        if key in results:
            raise ValueError("Each source run must have a unique directory name.")
        result = {
            "source_run": str(run.resolve().relative_to(root.resolve())),
            "source_summary_sha256": file_sha256(run / "summary.json"),
            "source_scores_sha256": file_sha256(run / "scores.jsonl"),
            "config": summary["model"]["config"],
            "feasible": policy is not None,
            "policy": policy,
        }
        if policy is not None:
            records = records_for(ids, expected, values, labels, policy)
            result["metrics"] = agreement_metrics(records, labels)
            write_prediction_records(output / f"{key}.jsonl", records)
        write_json(output / f"{key}-selection.json", trace)
        results[key] = result
    artifact = {
        "protocol_sha256": protocol["protocol_sha256"],
        "code_sha256": file_sha256(Path(__file__)),
        "recall_floor_per_dimension": 0.5,
        "threshold_grid": list(THRESHOLDS),
        "selection": "Maximize precision independently in each dimension subject to its recall floor.",
        "runs": results,
    }
    write_json(output / "summary.json", artifact)
    return artifact


def refit_frozen(
    manifest_path: Path, finalist: str, root: Path
) -> tuple[TextBaseline, dict[str, Any]]:
    """Parent-owned inference refits fit IDs only; never loads evaluation targets."""
    manifest = read_json(manifest_path)
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if digest(payload) != manifest["manifest_sha256"]:
        raise ValueError("Frozen manifest content hash mismatch.")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported text freeze schema.")
    if manifest["dependency_versions"] != dependencies():
        raise ValueError("Refit requires the exact pinned experiment dependency versions.")
    for relative, expected_hash in manifest["source_files"].items():
        if file_sha256(root / relative) != expected_hash:
            raise ValueError(f"Frozen source changed: {relative}")
    protocol = load_protocol(root / "runs/precision/protocol.json")
    if protocol["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("Frozen protocol changed.")
    candidate = manifest["finalists"][finalist]
    model = fit_from_protocol(
        protocol, read_json(root / "data/snapshot.json"),
        root / "runs/baseline/predictions.jsonl", TextConfig(**candidate["config"]),
    )
    return model, candidate["policy"]


def freeze_finalists(root: Path, dimension_directory: Path, output: Path) -> dict[str, Any]:
    """Freeze at most two policies after development selection, with replay proof."""
    if output.exists():
        raise ValueError("Choose a new directory; frozen policies cannot be overwritten.")
    dimension = read_json(dimension_directory / "summary.json")
    protocol = load_protocol(root / "runs/precision/protocol.json")
    if dimension["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("Dimension selection belongs to a different protocol.")
    candidates = []
    for run in dimension["runs"].values():
        path = root / run["source_run"]
        if (file_sha256(path / "summary.json") != run["source_summary_sha256"]
                or file_sha256(path / "scores.jsonl") != run["source_scores_sha256"]):
            raise ValueError("Source development evidence changed.")
        source = read_json(path / "summary.json")
        balanced = source["policies"]["balanced"]
        candidates.append({
            "config": run["config"], "policy": balanced["policy"],
            "metrics": balanced["metrics"], "source_run": run["source_run"],
        })
    balanced = max(candidates, key=lambda row: (
        row["metrics"]["overall_observed_dimensions"]["micro_f1"],
        row["metrics"]["overall_observed_dimensions"]["precision"], row["source_run"],
    ))
    feasible = [row for row in dimension["runs"].values() if row["feasible"]]
    if not feasible:
        raise ValueError("No broad precision policy reaches the per-dimension recall floors.")
    precision = max(feasible, key=lambda row: (
        row["metrics"]["overall_observed_dimensions"]["precision"],
        row["metrics"]["overall_observed_dimensions"]["micro_f1"], row["source_run"],
    ))
    finalists = {
        "balanced": balanced,
        "broad_precision": {
            key: precision[key] for key in ("config", "policy", "metrics", "source_run")
        },
    }
    source_paths = [
        "experiments/precision_text.py", "experiments/precision_protocol.py", "evaluation.py",
        "experiment.py", "experiments/requirements-text.txt",
        "runs/precision/protocol.json", "data/snapshot.json", "runs/baseline/predictions.jsonl",
    ]
    manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_files": {relative: file_sha256(root / relative) for relative in source_paths},
        "dependency_versions": dependencies(),
        "selection_source": str(dimension_directory.resolve().relative_to(root.resolve())),
        "selection_sha256": file_sha256(dimension_directory / "summary.json"),
        "fit_id_digest": digest(select_ids(protocol, "fit")),
        "development_id_digest": digest(select_ids(protocol, "development")),
        "confirmation_accessed": False,
        "finalists": finalists,
    }
    manifest["manifest_sha256"] = digest(manifest)
    snapshot = read_json(root / "data/snapshot.json")
    by_number = {issue["number"]: issue for issue in snapshot["issues"]}
    ids = select_ids(protocol, "development")
    inputs = prediction_inputs([by_number[number] for number in ids])
    replayed = {}
    for name, candidate in finalists.items():
        model = fit_from_protocol(
            protocol, snapshot, root / "runs/baseline/predictions.jsonl",
            TextConfig(**candidate["config"]),
        )
        policy = candidate["policy"]
        cached = (
            read_selected_scores(root / "runs/baseline/predictions.jsonl", inputs, model.labels)
            if model.config.kind in {"hybrid", "scores"} else None
        )
        actual = predict_records(model, inputs, policy, cached)
        source = root / candidate["source_run"]
        source_scores = [json.loads(line) for line in (source / "scores.jsonl").read_text().splitlines()]
        if [row["number"] for row in source_scores] != ids:
            raise ValueError("Frozen source score row alignment changed.")
        expected = [
            {"number": row["number"], "proposed": policy_predict(row["scores"], policy)}
            for row in source_scores
        ]
        if actual != expected:
            raise ValueError("Deterministic refit did not replay development proposals exactly.")
        replayed[name] = actual
    write_json(output / "manifest.json", manifest)
    for name, records in replayed.items():
        write_prediction_records(output / f"{name}-development.jsonl", records)
    return manifest


def run_development(root: Path, output: Path, config: TextConfig) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Choose a new run directory; development evidence is immutable.")
    protocol_path = root / "runs/precision/protocol.json"
    snapshot_path = root / "data/snapshot.json"
    baseline_path = root / "runs/baseline/predictions.jsonl"
    protocol = load_protocol(protocol_path)
    verify_sources(protocol, snapshot_path, baseline_path)
    snapshot = read_json(snapshot_path)
    model = fit_from_protocol(protocol, snapshot, baseline_path, config)
    development_ids = select_ids(protocol, "development")
    by_number = {issue["number"]: issue for issue in snapshot["issues"]}
    development = [by_number[number] for number in development_ids]
    inputs = prediction_inputs(development)
    cached = (
        read_selected_scores(baseline_path, inputs, model.labels)
        if config.kind in {"scores", "hybrid"} else None
    )
    started = time.perf_counter()
    scores = model.predict(inputs, cached)
    batch_seconds = time.perf_counter() - started
    individual_seconds = []
    for index, issue in enumerate(inputs[:16]):
        started = time.perf_counter()
        individual = model.predict([issue], None if cached is None else cached[index:index + 1])
        individual_seconds.append(time.perf_counter() - started)
        np.testing.assert_allclose(individual[0], scores[index], atol=1e-12)
    expected = [
        sorted(canonical_labels(issue["labels"], snapshot["contract"])) for issue in development
    ]
    policies, candidates = choose_policies(development_ids, expected, scores, model.labels)
    if config.kind == "prior":
        policies["top1_reference"] = {"kind": "top1_per_dimension"}
    results = {}
    for name, policy in policies.items():
        records = records_for(development_ids, expected, scores, model.labels, policy)
        results[name] = {"policy": policy, "metrics": agreement_metrics(records, model.labels)}
        write_prediction_records(output / f"{name}.jsonl", records)
    summary = {
        "schema_version": 1,
        "stage": "development_selection_only",
        "fit_count": len(select_ids(protocol, "fit")),
        "development_count": len(development_ids),
        "protocol_sha256": protocol["protocol_sha256"],
        "snapshot_sha256": protocol["snapshot_sha256"],
        "baseline_sha256": protocol["baseline_sha256"],
        "code_sha256": file_sha256(Path(__file__)),
        "dependency_versions": dependencies(),
        "model": model.metadata(),
        "threshold_grid": list(THRESHOLDS),
        "precision_recall_floor": 0.5,
        "latency": {
            "fit_seconds": model.fit_seconds,
            "development_batch_seconds": batch_seconds,
            "amortized_issue_seconds": batch_seconds / len(inputs),
            "single_issue_sample_count": len(individual_seconds),
            "single_issue_median_seconds": float(np.median(individual_seconds)),
            "single_issue_p95_seconds": float(np.quantile(individual_seconds, 0.95)),
            "scope": "In-process preprocessing and classical classifier only; excludes startup, disk IO and cached Laya generation.",
        },
        "policies": results,
        "limitations": protocol["limitations"] + [
            "This model is supervised on current silver labels, not zero-shot Laya.",
            "Selection reuses the full development set; development is not confirmation.",
            "Unobserved dimensions are excluded from target fitting and agreement metrics.",
            "No neural inference is run; hybrid and scores reuse pre-existing Laya outputs.",
        ],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "thresholds.json", candidates)
    (output / "scores.jsonl").write_text(
        "".join(json.dumps({"number": number, "scores": dict(zip(model.labels, row.tolist()))}) + "\n"
                for number, row in zip(development_ids, scores)),
        encoding="utf8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--describe", action="store_true", help="Print the default deterministic model configuration."
    )
    parser.add_argument("--output", type=Path, help="New development run directory; never opens confirmation.")
    parser.add_argument("--kind", choices=["prior", "nearest", "word", "scores", "hybrid"], default="word")
    parser.add_argument("--title-weight", type=int, default=1)
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument("--score-weight", type=float, default=0.25)
    parser.add_argument("--strip-title-tags", action="store_true",
                        help="Ablate leading bracketed type markers; leave body unchanged.")
    parser.add_argument("--dimension-runs", type=Path, nargs="+",
                        help="Derive three dev thresholds from existing run scores; no refitting.")
    parser.add_argument("--freeze-from", type=Path,
                        help="Freeze two finalists from a dimension-policy directory.")
    parser.add_argument("--predict-frozen", type=Path,
                        help="Frozen manifest to refit and predict on parent-supplied input JSONL.")
    parser.add_argument("--finalist", choices=["balanced", "broad_precision"])
    parser.add_argument("--inputs", type=Path, help="Prediction JSONL with number/title/body only.")
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(asdict(TextConfig()), indent=2))
        return
    if args.output is None:
        parser.error("Use --describe or --output for one bounded development experiment.")
    root = Path(__file__).resolve().parents[1]
    if sum(bool(value) for value in (args.dimension_runs, args.freeze_from, args.predict_frozen)) > 1:
        parser.error("Choose one execution mode.")
    if args.dimension_runs:
        summary = derive_dimension_policies(root, args.dimension_runs, args.output)
        print(json.dumps({
            name: {"policy": row["policy"], "feasible": row["feasible"],
                   "overall": row.get("metrics", {}).get("overall_observed_dimensions")}
            for name, row in summary["runs"].items()
        }, indent=2))
        return
    if args.freeze_from:
        manifest = freeze_finalists(root, args.freeze_from, args.output)
        print(json.dumps({
            "manifest_sha256": manifest["manifest_sha256"],
            "finalists": {name: {"config": row["config"], "policy": row["policy"]}
                          for name, row in manifest["finalists"].items()},
            "deterministic_development_replay": "exact",
        }, indent=2))
        return
    if args.predict_frozen:
        if not args.inputs or not args.finalist:
            parser.error("--predict-frozen requires --inputs and --finalist.")
        if args.output.exists():
            parser.error("Prediction output already exists; choose a new path.")
        rows = [json.loads(line) for line in args.inputs.read_text().splitlines()]
        if any(set(row) != {"number", "title", "body"} for row in rows):
            parser.error("Prediction inputs must contain only number/title/body.")
        if not rows or len({row["number"] for row in rows}) != len(rows):
            parser.error("Prediction inputs require nonempty, unique issue IDs.")
        model, policy = refit_frozen(args.predict_frozen, args.finalist, root)
        cached = (
            read_selected_scores(root / "runs/baseline/predictions.jsonl", rows, model.labels)
            if model.config.kind in {"hybrid", "scores"} else None
        )
        started = time.perf_counter()
        predictions = predict_records(model, rows, policy, cached)
        prediction_seconds = time.perf_counter() - started
        write_prediction_records(args.output, predictions)
        print(json.dumps({"rows": len(rows), "fit_seconds": model.fit_seconds,
                          "prediction_seconds": prediction_seconds,
                          "output_sha256": file_sha256(args.output)}))
        return
    config = TextConfig(
        kind=args.kind, title_weight=args.title_weight, c=args.c, score_weight=args.score_weight,
        strip_title_tags=args.strip_title_tags,
    )
    summary = run_development(root, args.output, config)
    print(json.dumps({
        "output": str(args.output),
        "latency": summary["latency"],
        "policies": {
            name: {"policy": result["policy"],
                   **result["metrics"]["overall_observed_dimensions"],
                   "coverage": result["metrics"]["observed_issue_prediction_coverage"],
                   "labels": result["metrics"]["mean_labels_all_dimensions"]}
            for name, result in summary["policies"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
