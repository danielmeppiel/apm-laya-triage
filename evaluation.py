"""Coverage-aware agreement metrics and a plain-English, reproducible report."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiment import (
    check_response,
    digest,
    predicted_labels,
    read_json,
    taxonomy,
    write_json,
)

DIMENSIONS = ("type", "area", "theme")


def canonical_labels(raw: list[str], contract: dict[str, Any]) -> set[str]:
    """Normalize only explicitly documented aliases, never human decisions."""
    allowed = set(contract["classification_labels"])
    aliases = contract["legacy_read_aliases"]
    return {aliases.get(label, label) for label in raw} & allowed


def fraction(numerator: int | float, denominator: int | float) -> float | None:
    """Return undefined, not a fake zero, when no observations are available."""
    return numerator / denominator if denominator else None


def scored_sets(
    record: dict[str, Any], dimension: str | None
) -> tuple[set[str], set[str]]:
    """Score only dimensions for which the issue has at least one control label."""
    expected = set(record["expected"])
    proposed = set(record["proposed"])
    if dimension:
        expected = {label for label in expected if label.startswith(dimension + "/")}
        proposed = {label for label in proposed if label.startswith(dimension + "/")}
    else:
        observed = {label.split("/", 1)[0] for label in expected}
        proposed = {label for label in proposed if label.split("/", 1)[0] in observed}
    return expected, proposed


def aggregate(
    records: list[dict[str, Any]], dimension: str | None = None
) -> dict[str, Any]:
    """Measure exact agreement and positive-label precision/recall, not negative accuracy."""
    tp = fp = fn = exact = controls = 0
    jaccards = []
    for record in records:
        expected, proposed = scored_sets(record, dimension)
        if not expected:
            continue
        controls += 1
        tp += len(expected & proposed)
        fp += len(proposed - expected)
        fn += len(expected - proposed)
        exact += expected == proposed
        jaccards.append(len(expected & proposed) / len(expected | proposed))
    return {
        "control_issues": controls,
        "exact_matches": exact,
        "exact_agreement": fraction(exact, controls),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": fraction(tp, tp + fp),
        "recall": fraction(tp, tp + fn),
        "micro_f1": fraction(2 * tp, 2 * tp + fp + fn),
        "mean_jaccard": statistics.mean(jaccards) if jaccards else None,
    }


def label_metrics(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Treat missing dimensions as unscored, not as negative examples."""
    dim = label.split("/", 1)[0]
    eligible = [record for record in records if scored_sets(record, dim)[0]]
    tp = sum(label in r["expected"] and label in r["proposed"] for r in eligible)
    fp = sum(label not in r["expected"] and label in r["proposed"] for r in eligible)
    fn = sum(label in r["expected"] and label not in r["proposed"] for r in eligible)
    return {
        "control_issues_in_dimension": len(eligible),
        "support": tp + fn,
        "proposed_count": tp + fp,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": fraction(tp, tp + fp),
        "recall": fraction(tp, tp + fn),
        "f1": fraction(2 * tp, 2 * tp + fp + fn),
    }


def percentile(values: list[float], quantile: float) -> float:
    """Use linear interpolation between adjacent sorted observations."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def evaluate(
    snapshot: dict[str, Any],
    predictions: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Join an entire frozen corpus to verified model outputs and compute controls."""
    if manifest["status"] != "complete":
        raise ValueError("Run is incomplete; refusing a full-corpus success report.")
    if manifest["source_snapshot_sha256"] != digest(snapshot):
        raise ValueError("Snapshot does not match the inference manifest.")
    issues = {issue["number"]: issue for issue in snapshot["issues"]}
    if not issues or len(issues) != len(snapshot["issues"]):
        raise ValueError("Snapshot must contain unique issues and cannot be empty.")
    if (
        len(predictions) != len(issues)
        or {row["number"] for row in predictions} != set(issues)
        or manifest["prediction_count"] != len(issues)
        or manifest["target_count"] != len(issues)
    ):
        raise ValueError(
            "Full report requires exactly one real prediction for every issue."
        )
    labels = taxonomy(snapshot)
    records = []
    for prediction in predictions:
        if (
            prediction["fingerprint"] != manifest["fingerprint"]
            or prediction["evidence"] != "real-local-model-inference"
        ):
            raise ValueError(
                "Prediction belongs to a different run or is not real inference."
            )
        issue = issues[prediction["number"]]
        if prediction["input_sha256"] != digest(
            {"title": issue["title"], "body": issue["body"]}
        ):
            raise ValueError("Issue text does not match what was classified.")
        check_response(prediction["response"], manifest["questions"])
        proposed = predicted_labels(
            prediction["response"], manifest["config"]["binary_threshold"]
        )
        if proposed != prediction["predicted_labels"] or not set(proposed) <= set(
            labels
        ):
            raise ValueError(
                "Stored labels do not match the model probabilities and policy."
            )
        if any(
            type(prediction[key]) not in (float, int)
            or not math.isfinite(prediction[key])
            or prediction[key] <= 0
            for key in ("inference_seconds", "processing_seconds")
        ):
            raise ValueError("Inference timings must be finite and positive.")
        records.append(
            {
                **issue,
                "expected": sorted(
                    canonical_labels(issue["labels"], snapshot["contract"])
                ),
                "proposed": proposed,
                "shortened": prediction["shortened"],
                "inference_seconds": prediction["inference_seconds"],
                "processing_seconds": prediction["processing_seconds"],
                "original_state_tokens": prediction["original_state_tokens"],
                "used_state_tokens": prediction["used_state_tokens"],
                "probabilities": {
                    label: prediction["response"]["answers"][label]["noul"]
                    for label in labels
                },
                "label_names_in_original_text": any(
                    label in issue["title"] + "\n" + issue["body"] for label in labels
                ),
            }
        )
    records.sort(key=lambda record: record["number"])
    distribution = Counter(label for record in records for label in record["expected"])
    majority = {
        dim: max(
            (label for label in labels if label.startswith(dim + "/")),
            key=lambda label: (distribution[label], label),
        )
        for dim in DIMENSIONS
    }
    naive = [{**record, "proposed": sorted(majority.values())} for record in records]
    seconds = [record["inference_seconds"] for record in records]
    per_label = {label: label_metrics(records, label) for label in labels}
    supported_f1 = [
        item["f1"]
        for item in per_label.values()
        if item["support"] and item["f1"] is not None
    ]
    excluded = Counter(
        label
        for issue in snapshot["issues"]
        for label in issue["labels"]
        if not canonical_labels([label], snapshot["contract"])
    )
    metrics = {
        "issue_count": len(records),
        "control_issues": sum(bool(record["expected"]) for record in records),
        "unscored_issues": sum(not record["expected"] for record in records),
        "states": dict(Counter(record["state"] for record in records)),
        "shortened_issues": sum(record["shortened"] for record in records),
        "label_names_in_original_text": sum(
            record["label_names_in_original_text"] for record in records
        ),
        "multi_label_controls": {
            dim: sum(len(scored_sets(record, dim)[0]) > 1 for record in records)
            for dim in DIMENSIONS
        },
        "empty_proposals": sum(not record["proposed"] for record in records),
        "overall_observed_dimensions": aggregate(records),
        "by_dimension": {dim: aggregate(records, dim) for dim in DIMENSIONS},
        "by_cohort": {
            "open": aggregate([r for r in records if r["state"] == "open"]),
            "closed": aggregate([r for r in records if r["state"] == "closed"]),
            "full_context": aggregate([r for r in records if not r["shortened"]]),
            "shortened_context": aggregate([r for r in records if r["shortened"]]),
            "without_label_names_in_text": aggregate(
                [r for r in records if not r["label_names_in_original_text"]]
            ),
        },
        "per_label": per_label,
        "macro_f1_supported_labels": statistics.mean(supported_f1)
        if supported_f1
        else None,
        "reference_label_distribution": dict(distribution.most_common()),
        "excluded_label_distribution": dict(excluded.most_common()),
        "naive_baseline": {
            "proposed_labels": sorted(majority.values()),
            "overall_observed_dimensions": aggregate(naive),
            "by_dimension": {dim: aggregate(naive, dim) for dim in DIMENSIONS},
            "caveat": "Descriptive most-common-label reference calculated on this snapshot, not a held-out trained baseline.",
        },
        "latency": {
            "unit": "seconds per issue, all label questions together, synchronized device",
            "mean": statistics.mean(seconds),
            "median": statistics.median(seconds),
            "p90": percentile(seconds, 0.90),
            "p95": percentile(seconds, 0.95),
            "p99": percentile(seconds, 0.99),
            "max": max(seconds),
            "min": min(seconds),
            "first_issue": predictions[0]["inference_seconds"],
            "sum_inference_seconds": sum(seconds),
            "sum_processing_seconds": sum(
                record["processing_seconds"] for record in records
            ),
            "amortized_issues_per_minute": 60 * len(seconds) / sum(seconds),
        },
    }
    return metrics, records


def pct(value: float | None) -> str:
    """Format undefined denominators honestly."""
    return f"{100 * value:.1f}%" if value is not None else "n/a"


def escaped(text: str, limit: int = 180) -> str:
    """Keep issue prose inert in generated Markdown examples."""
    text = html.escape(" ".join(text.split())[:limit])
    for character in ("\\", "`", "*", "_", "[", "]", "!", "|"):
        text = text.replace(character, "\\" + character)
    return text


def report_text(
    snapshot: dict[str, Any],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    artifact_path: str = "runs/baseline",
    readme_path: str = "README.md",
) -> str:
    """Explain the actual result to someone who does not know ML terminology."""
    overall = metrics["overall_observed_dimensions"]
    naive = metrics["naive_baseline"]["overall_observed_dimensions"]
    latency = metrics["latency"]
    lines = [
        "# APM issue labelling experiment: the plain-English report",
        "",
        "## The short answer",
        "",
        f"We ran **real local Laya inference on all {metrics['issue_count']:,} issues**, "
        f"including {metrics['states'].get('open', 0):,} open and "
        f"{metrics['states'].get('closed', 0):,} closed issues. Pull requests were excluded.",
        "**No APM issue, label, comment, status, assignment, or milestone was changed.**",
        "",
        f"On the {metrics['control_issues']:,} issues with usable existing classification labels, "
        f"the model matched every known dimension exactly on **{pct(overall['exact_agreement'])}** "
        f"({overall['exact_matches']:,}/{overall['control_issues']:,}).",
        f"Label precision was **{pct(overall['precision'])}**, recall **{pct(overall['recall'])}**, "
        f"and the combined F1 score **{pct(overall['micro_f1'])}**.",
        f"Median model time was **{latency['median']:.3f} seconds per issue** for all "
        f"{len(manifest['questions'])} label decisions together.",
        "",
        "**This measures agreement with today's labels, not whether the model understood or fixed a bug.** "
        "The controls may be incomplete, debatable, or assigned by automation. Do not read this "
        "as a production accuracy guarantee or permission to auto-label.",
        "",
        "## What we asked the model to do",
        "",
        "- Read the issue title and body, plus a short description of APM.",
        "- Answer one yes/no membership question for every allowed classification label.",
        "- Return a probability for each label. Propose it when that probability is at least "
        f"{manifest['config']['binary_threshold']:.2f}. Multiple types, areas and themes can apply.",
        "- Write the proposals to local JSON/CSV files. Never call a GitHub write API.",
        "",
        "The model never received the issue's current API label list, closed/open state, number, "
        "expected answers, comments, or repository source code. The label names and their "
        "definitions are the shared classification rubric, not per-issue answers.",
        "All labels were evaluated with the same fixed threshold. There was no training, "
        "fine-tuning, threshold fitting, or post-result prompt selection for this baseline.",
        "",
        "## What the numbers mean",
        "",
        "| Measure | In ordinary language |",
        "|---|---|",
        "| Exact agreement | Did the entire proposed label set match, in every dimension with a control? |",
        "| Precision | Of the labels proposed in scorable dimensions, what share already appeared in the controls? |",
        "| Recall | Of the control labels, what share did the model recover? |",
        "| F1 | A combined precision/recall score; high recall alone is not enough if many extra labels are proposed. |",
        "| Unscored | We generated predictions, but there are no relevant existing labels to check them against. |",
        "",
        f"There are **{metrics['unscored_issues']:,} unscored issues**. Their predictions are "
        "included in the output, but excluded from success-rate denominators.",
        "If an issue has a type label but no area labels, we score its type only. Missing an "
        "entire dimension is **not** treated as proof that every label in it should be absent. "
        "Within an observed dimension, additional proposed labels count as disagreements; "
        "some could be reasonable labels that maintainers never added.",
        "",
        "## Results by kind of label",
        "",
        "| Dimension | Control issues | Exact agreement | Precision | Recall | F1 | Controls with multiple labels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dim, result in metrics["by_dimension"].items():
        lines.append(
            f"| {dim} | {result['control_issues']} | {pct(result['exact_agreement'])} | "
            f"{pct(result['precision'])} | {pct(result['recall'])} | {pct(result['micro_f1'])} | "
            f"{metrics['multi_label_controls'][dim]} |"
        )
    comparison = ""
    if overall["micro_f1"] is not None and naive["micro_f1"] is not None:
        difference = overall["micro_f1"] - naive["micro_f1"]
        comparison = (
            f" Laya is {abs(difference) * 100:.1f} percentage points "
            f"{'above' if difference >= 0 else 'below'} the no-reading reference on F1. "
            + (
                "This shows a useful signal, not proof that unattended labelling is safe."
                if difference > 0
                else "This baseline does not demonstrate better F1 than simply using label prevalence."
            )
        )
    lines.extend(
        [
            "",
            "A deliberately simple reference proposes the single most common label in each dimension "
            "without reading the issue. Its F1 is "
            f"**{pct(naive['micro_f1'])}**, versus **{pct(overall['micro_f1'])}** for Laya. "
            "That reference uses this snapshot's label prevalence; it is a descriptive sanity check, "
            "not a separately trained or held-out baseline." + comparison,
            "",
            "## Every individual label",
            "",
            "| Label | Existing positives | Model proposals in scorable issues | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, result in metrics["per_label"].items():
        lines.append(
            f"| `{label}` | {result['support']} | {result['proposed_count']} | "
            f"{pct(result['precision'])} | {pct(result['recall'])} | {pct(result['f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Real examples, including disagreements",
            "",
            "These are the first examples by issue number, not hand-picked best or worst cases. "
            "An extra prediction in an entirely unlabelled dimension is shown but not scored.",
        ]
    )
    for heading, matches, count in [
        ("Matches in observed dimensions", True, 3),
        ("Disagreements", False, 5),
    ]:
        examples = [
            row
            for row in records
            if row["expected"]
            and (scored_sets(row, None)[0] == scored_sets(row, None)[1]) == matches
        ][:count]
        lines.extend(["", f"### {heading}", ""])
        if not examples:
            lines.append("No examples in this category.")
        for row in examples:
            lines.extend(
                [
                    f"**[microsoft/apm#{row['number']}]({row['url']}): {escaped(row['title'])}**",
                    "",
                    f"- Existing classification: {', '.join('`' + label + '`' for label in row['expected'])}.",
                    f"- Model proposal: {', '.join('`' + label + '`' for label in row['proposed']) or '(none)'}.",
                    f"- Context shortened: {'yes' if row['shortened'] else 'no'}.",
                    "",
                ]
            )
    lines.extend(
        [
            "The model does not generate explanations. A disagreement is not evidence of a verified "
            "bug in the control labels or a known reasoning process inside the model.",
            "",
            "## Speed and what machine did the work",
            "",
            f"- Actual inference device: **{manifest['runtime']['device']}**, full precision "
            f"(mixed precision: {manifest['runtime']['mixed_precision']}). "
            f"CPU thread limit: {manifest['runtime']['cpu_threads']}.",
            f"- Platform: `{manifest['runtime']['platform']}`.",
            f"- Median / p95 / p99: **{latency['median']:.3f}s / {latency['p95']:.3f}s / {latency['p99']:.3f}s** per issue.",
            f"- Mean / maximum: **{latency['mean']:.3f}s / {latency['max']:.3f}s**.",
            f"- Total measured inference: **{latency['sum_inference_seconds'] / 60:.1f} minutes**.",
            f"- Amortized throughput: **{latency['amortized_issues_per_minute']:.1f} issues/minute**.",
            f"- Recorded model load: **{manifest['runtime']['model_load_seconds']:.2f}s**; "
            f"download/cache lookup plus checksum: **{manifest['runtime']['download_and_hash_seconds']:.2f}s**.",
            "",
            "GPU timings synchronize the device before and after each call. They are not merely "
            "the time taken to queue GPU work. All label questions run together; multiplying the "
            "per-issue latency by the label count would double-count work. Installation, GitHub "
            "snapshot download, and report generation are not included in model latency. This run "
            "reused cached weights; a cold installation also downloads an approximately 843 MB model.",
            "",
            "A GitHub-hosted CPU runner is a different machine and will have different latency. "
            "The repository includes a manual workflow, but this result is a local run, not a hosted Actions run.",
            "",
            "## Important limitations",
            "",
            f"1. **Short context:** {metrics['shortened_issues']:,}/{metrics['issue_count']:,} issues "
            f"were shortened to at most {manifest['config']['state_token_budget']} state tokens. "
            "The beginning and a smaller tail are retained, with an explicit omission marker. "
            "Useful evidence in the middle may be lost. This is not full-discussion triage.",
            "2. **Silver controls, not gold truth:** labels reflect the repository snapshot, including "
            "legacy aliases and potentially automated decisions. We did not reconstruct who applied "
            "each label or the labels that existed when an issue was originally opened.",
            "3. **Historical and language drift:** the English checkpoint sees old and new issues "
            "through today's taxonomy. Non-English issues are not routed to a multilingual checkpoint.",
            f"4. **Possible text hints:** {metrics['label_names_in_original_text']} issues mention canonical "
            "label names in their original title/body. We did not strip that prose; the sensitivity "
            "table below also reports the corpus without those cases.",
            "5. **Confidence is not authority:** probabilities and the SDK's action head are not "
            "permission to accept, prioritize, assign, implement, or label an issue.",
            "6. **Runtime behavior:** the initial MPS mixed-precision pilot produced non-finite "
            "probabilities. Validation rejected them. This baseline disables mixed precision. "
            "The SDK also warns about a clamped temperature for 11-plus-option choices; this "
            "experiment uses only two-option Noul questions.",
            "7. **Not a held-out production claim:** this is a zero-shot exploratory benchmark "
            "against a frozen public corpus. Future tuning must be evaluated on separately reserved data.",
            "",
            "### Sensitivity checks",
            "",
            "| Subset | Control issues | Exact agreement | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in metrics["by_cohort"].items():
        lines.append(
            f"| {name} | {result['control_issues']} | {pct(result['exact_agreement'])} | "
            f"{pct(result['precision'])} | {pct(result['recall'])} | {pct(result['micro_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Provenance and reproducibility",
            "",
            f"- Snapshot collection: `{snapshot['started_at']}` to `{snapshot['completed_at']}`.",
            f"- Source repository revision for the label contract: `{snapshot['source_revision']}`.",
            f"- Model: `{manifest['config']['model_repo']}` at `{manifest['config']['model_revision']}`.",
            f"- Safetensors SHA-256: `{manifest['runtime']['weights_sha256']}`.",
            f"- Run fingerprint: `{manifest['fingerprint']}`.",
            f"- Run completed: `{manifest['completed_at']}`.",
            "- Current labels, their descriptions, and the canonical alias contract are frozen with the issue snapshot.",
            "- Model weights are data-only Safetensors. Python code is installed from the pinned official SDK source, not executed from the model repository.",
            f"- Raw per-issue probabilities and timing: [{artifact_path}/predictions.jsonl]({artifact_path}/predictions.jsonl).",
            f"- Machine-readable metrics and all proposals: [{artifact_path}/metrics.json]({artifact_path}/metrics.json) and [{artifact_path}/proposals.csv]({artifact_path}/proposals.csv).",
            f"- Exact installation, refresh, resume, and rerun commands are in [README.md]({readme_path}).",
            "",
            "## Recommendation",
            "",
            "**Keep this as a read-only experiment, not an automatic labelling bot.** "
            "Review disagreements, improve the rubric or context strategy, and use a separately "
            "held-out, human-adjudicated sample before authorizing any real label writes. "
            "This repository has no apply-labels command.",
            "",
        ]
    )
    return "\n".join(lines)


def csv_text(value: str) -> str:
    """Prevent spreadsheet applications from treating issue titles as formulas."""
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def build_report(snapshot_path: Path, run_path: Path, output: Path) -> None:
    """Verify complete evidence, then write metrics, proposals and the reader's report."""
    snapshot = read_json(snapshot_path)
    manifest = read_json(run_path / "manifest.json")
    predictions = [
        json.loads(line)
        for line in (run_path / "predictions.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
    ]
    metrics, records = evaluate(snapshot, predictions, manifest)
    write_json(run_path / "metrics.json", metrics)
    with (run_path / "proposals.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "issue_number",
                "url",
                "state",
                "title",
                "existing_classification",
                "proposed_labels",
                "has_controls",
                "observed_dimensions_exact_match",
                "context_shortened",
                "inference_seconds",
            ]
        )
        for row in records:
            expected, proposed = scored_sets(row, None)
            writer.writerow(
                [
                    row["number"],
                    row["url"],
                    row["state"],
                    csv_text(row["title"]),
                    ";".join(row["expected"]),
                    ";".join(row["proposed"]),
                    bool(expected),
                    expected == proposed if expected else "",
                    row["shortened"],
                    row["inference_seconds"],
                ]
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    text = report_text(
        snapshot,
        manifest,
        metrics,
        records,
        artifact_path=Path(
            os.path.relpath(run_path.resolve(), output.resolve().parent)
        ).as_posix(),
        readme_path=Path(
            os.path.relpath(
                Path(__file__).with_name("README.md"), output.resolve().parent
            )
        ).as_posix(),
    )
    output.write_text(text, encoding="utf8")
    print(f"[+] Wrote {output}; full-corpus evidence verified. No GitHub writes.")
