"""Rebuild single-issue speed and full-corpus accuracy comparisons from real evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation import evaluate, pct
from experiment import ROOT, check_response, predicted_labels, read_json, write_json

CONFIGURATIONS = ("fp32-2", "fp32-4", "int8-2", "int8-4")


def validate_profile(profile: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Require all six distinct issues and three real calls per issue."""
    if profile["status"] != "complete":
        raise ValueError("Cannot summarize an incomplete speed configuration.")
    records = profile["records"]
    indexes = {(row["number"], row["repeat"]): row for row in records}
    numbers = {row["number"] for row in records}
    if len(records) != 18 or len(indexes) != 18 or len(numbers) != 6:
        raise ValueError(
            "Speed evidence must contain six issues with three distinct repeats."
        )
    if set(indexes) != {(number, repeat) for number in numbers for repeat in range(3)}:
        raise ValueError("Missing or unexpected speed repetitions.")
    for row in records:
        if row["evidence"] != "real-local-model-inference":
            raise ValueError("Speed evidence is not marked as actual inference.")
        if not math.isfinite(row["inference_seconds"]) or row["inference_seconds"] <= 0:
            raise ValueError("Invalid measured speed.")
        check_response(row["response"], profile["questions"])
        if row["predicted_labels"] != predicted_labels(
            row["response"], profile["config"]["binary_threshold"]
        ):
            raise ValueError("Speed predictions do not match the raw probabilities.")
    return indexes


def profile_summary(root: Path) -> dict[str, Any]:
    """Compare configurations within each machine, not across unrelated VM hardware."""
    results = {name: [] for name in CONFIGURATIONS}
    cohort_hash = None
    for replicate in range(2):
        folder = root / f"single-replicate-{replicate}"
        baseline = read_json(folder / "fp32-2.json")
        expected = validate_profile(baseline)
        if cohort_hash is not None and cohort_hash != baseline["snapshot_sha256"]:
            raise ValueError("The two hardware replicates used different corpora.")
        cohort_hash = baseline["snapshot_sha256"]
        baseline_median = statistics.median(
            row["inference_seconds"] for row in expected.values()
        )
        for name in CONFIGURATIONS:
            profile = read_json(folder / f"{name}.json")
            actual = validate_profile(profile)
            if (
                profile["snapshot_sha256"] != cohort_hash
                or profile["questions"] != baseline["questions"]
                or set(actual) != set(expected)
                or profile["runtime"]["versions"] != baseline["runtime"]["versions"]
                or profile["runtime"]["weights_sha256"]
                != baseline["runtime"]["weights_sha256"]
                or profile["config"]["binary_threshold"]
                != baseline["config"]["binary_threshold"]
                or f"{profile['precision']}-{profile['cpu_threads']}" != name
            ):
                raise ValueError("Speed configurations are not a matched comparison.")
            differences = []
            changed = set()
            for key, row in actual.items():
                control = expected[key]
                if row["state_sha256"] != control["state_sha256"]:
                    raise ValueError(
                        "Speed configurations used different prepared model inputs."
                    )
                if row["predicted_labels"] != control["predicted_labels"]:
                    changed.add(row["number"])
                differences.extend(
                    abs(
                        row["response"]["answers"][label]["noul"]
                        - control["response"]["answers"][label]["noul"]
                    )
                    for label in profile["questions"]
                )
            median = statistics.median(
                row["inference_seconds"] for row in actual.values()
            )
            results[name].append(
                {
                    "replicate": replicate,
                    "median_seconds": median,
                    "speedup_vs_fp32_2": baseline_median / median,
                    "first_issue_process_seconds": profile[
                        "first_issue_process_seconds"
                    ],
                    "optimization_seconds": profile["optimization_seconds"],
                    "changed_distinct_issues": sorted(changed),
                    "max_probability_difference": max(differences),
                }
            )
    return {
        "unique_issues": 6,
        "repeats_per_issue_per_machine": 3,
        "configurations": results,
    }


def corpus_metrics(snapshot: dict[str, Any], folder: Path) -> tuple[dict, list[dict]]:
    """Use the canonical complete-evidence verifier, not a separate accuracy formula."""
    rows = [
        json.loads(line)
        for line in (folder / "predictions.jsonl").read_text().splitlines()
    ]
    return evaluate(snapshot, rows, read_json(folder / "manifest.json"))


def single_summary(folder: Path) -> dict[str, Any]:
    """Separate invocation time, job time and dispatch-to-completion time."""
    result = read_json(folder / "result.json")
    if result["status"] != "complete":
        raise ValueError("A single-issue invocation did not complete successfully.")
    if (
        result["evidence"] != "real-local-model-inference"
        or result["github_writes"] is not False
    ):
        raise ValueError("Single-issue evidence must be real and read-only.")
    check_response(result["response"], result["questions"])
    if result["predicted_labels"] != predicted_labels(
        result["response"], result["config"]["binary_threshold"]
    ):
        raise ValueError("Single-issue labels do not match the raw probabilities.")
    summary = {
        "name": folder.name,
        "number": result["number"],
        "input_sha256": result["input_sha256"],
        "state_sha256": result["state_sha256"],
        "device": result["runtime"]["device"],
        "precision": result["precision"],
        "threads": result["config"]["cpu_threads"],
        "inference_seconds": result["inference_seconds"],
        "invocation_seconds": result["invocation_seconds"],
    }
    for key in ("inference_seconds", "invocation_seconds"):
        if not math.isfinite(summary[key]) or summary[key] <= 0:
            raise ValueError("Invalid single-issue timing.")
    if (folder / "process.json").exists():
        process = read_json(folder / "process.json")
        if process["process_wall_seconds"] < summary["invocation_seconds"]:
            raise ValueError("Process timing cannot be shorter than the invocation.")
        return {
            **summary,
            "kind": "local-process",
            "process_seconds": process["process_wall_seconds"],
            "hardware": process["hardware"],
            "weights_previously_cached": process["weights_previously_cached"],
        }
    workflow = read_json(folder / "workflow.json")
    if workflow["conclusion"] != "success" or len(workflow["jobs"]) != 1:
        raise ValueError("Expected one successful single-issue job.")
    job = workflow["jobs"][0]

    def timestamp(text: str) -> datetime:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))

    install = next(
        step
        for step in job["steps"]
        if step["name"] == "Install pinned CPU dependencies on cache miss"
    )
    return {
        **summary,
        "kind": "github-actions",
        "url": workflow["url"],
        "runtime_cache_hit": install["conclusion"] == "skipped",
        "job_seconds": (
            timestamp(job["completedAt"]) - timestamp(job["startedAt"])
        ).total_seconds(),
        "dispatch_to_complete_seconds": (
            timestamp(job["completedAt"]) - timestamp(workflow["createdAt"])
        ).total_seconds(),
    }


def generate(
    profiles: Path,
    single_runs: list[Path],
    snapshot_path: Path,
    reference: Path | None,
    candidate: Path | None,
    output: Path,
) -> None:
    """Render measured speed, actual workflow overhead and optional full-control accuracy."""
    profiles_result = profile_summary(profiles)
    singles = [single_summary(path) for path in single_runs]
    if (
        len(
            {
                (row["number"], row["input_sha256"], row["state_sha256"])
                for row in singles
            }
        )
        > 1
    ):
        raise ValueError(
            "Single-issue comparisons must use the same issue and prepared input."
        )
    lines = [
        "# Single-issue performance: measured, not estimated",
        "",
        "Production triage handles one issue at a time. Whole-corpus batches are used only "
        "to measure quality and speed on enough cases. Every call retains all 25 label questions.",
        "",
        "## Matched-hardware configuration pilot",
        "",
        "Each of two runners executed all four configurations, in opposite orders. "
        "Six fixed issues were each repeated three times. That means six distinct accuracy "
        "cases, not 36 independent ones. These pilot timings are not a production p95 claim.",
        "INT8 converts only encoder Linear layers; the decision head stays FP32.",
        "",
        "| Configuration | Runner 0 median | Runner 1 median | Changed issues vs FP32/2, runner 0 / 1 | Maximum probability change |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, records in profiles_result["configurations"].items():
        a, b = records
        lines.append(
            f"| {name} | {a['median_seconds']:.3f}s | {b['median_seconds']:.3f}s | "
            f"{len(a['changed_distinct_issues'])}/6 / {len(b['changed_distinct_issues'])}/6 | "
            f"{max(a['max_probability_difference'], b['max_probability_difference']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Model loading and INT8 conversion are separate costs, recorded in summary.json. "
            "The pilot reuses downloaded weights on each runner after its first configuration. "
            "Do not present warm-loop timings as cold workflow latency.",
            "",
            "## Actual one-issue workflow runs",
            "",
            "| Run | Mode / threads | Runtime cache hit | Inference | Invocation including fetch and loading | Job | Dispatch to completion |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in (item for item in singles if item["kind"] == "github-actions"):
        lines.append(
            f"| [{row['name']}]({row['url']}) | {row['precision']} / {row['threads']} | "
            f"{row['runtime_cache_hit']} | {row['inference_seconds']:.3f}s | "
            f"{row['invocation_seconds']:.3f}s | {row['job_seconds']:.0f}s | "
            f"{row['dispatch_to_complete_seconds']:.0f}s |"
        )
    lines.extend(
        [
            "",
            "These are individual observed runs, not latency guarantees. Dispatch time includes "
            "queueing; job time includes setup, caching and artifact upload. A cache miss installs "
            "dependencies, while a hit restores the exact pinned virtualenv.",
        ]
    )
    local = [row for row in singles if row["kind"] == "local-process"]
    if local:
        lines.extend(
            [
                "",
                "### Local GPU on the identical issue",
                "",
                "| Hardware / device | Precision | Inference | Invocation | Fresh process total |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in local:
            lines.append(
                f"| {row['hardware']} / {row['device']} | {row['precision']} | "
                f"{row['inference_seconds']:.3f}s | {row['invocation_seconds']:.3f}s | "
                f"{row['process_seconds']:.3f}s |"
            )
        lines.extend(
            [
                "",
                "Local measurements use already-downloaded model weights and no competing "
                "bulk GPU worker. GPU calls synchronize before/after inference. A fresh "
                "process still imports dependencies, verifies weights and loads the model. "
                "This is an Apple GPU result, not a measurement of an NVIDIA CUDA fast path.",
            ]
        )
    quality = None
    if reference and candidate:
        snapshot = read_json(snapshot_path)
        a, reference_rows = corpus_metrics(snapshot, reference)
        b, candidate_rows = corpus_metrics(snapshot, candidate)
        if (
            read_json(reference / "manifest.json")["questions"]
            != read_json(candidate / "manifest.json")["questions"]
            or read_json(reference / "manifest.json")["config"]["binary_threshold"]
            != read_json(candidate / "manifest.json")["config"]["binary_threshold"]
        ):
            raise ValueError("Cannot compare quality under different question rubrics.")
        reference_by_number = {row["number"]: row for row in reference_rows}
        changed = [
            row["number"]
            for row in candidate_rows
            if row["proposed"] != reference_by_number[row["number"]]["proposed"]
        ]
        quality = {"reference": a, "candidate": b, "changed_label_set_issues": changed}
        lines.extend(
            [
                "",
                "## Accuracy check on the full frozen control group",
                "",
                f"Both configurations classified all **{a['issue_count']} issues** against the same "
                f"**{a['control_issues']} eligible reference issues**. Unlabelled dimensions are "
                "unscored; current labels are a silver reference, not guaranteed truth.",
                "",
                "| Runtime | Exact agreement | Precision | Recall | F1 | Corpus median inference |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, metrics in [("FP32 reference", a), ("Candidate", b)]:
            scores = metrics["overall_observed_dimensions"]
            lines.append(
                f"| {name} | {pct(scores['exact_agreement'])} | {pct(scores['precision'])} | "
                f"{pct(scores['recall'])} | {pct(scores['micro_f1'])} | {metrics['latency']['median']:.3f}s |"
            )
        lines.extend(
            [
                "",
                f"The candidate changed the full proposed label set on **{len(changed)}/{a['issue_count']} "
                "issues** relative to FP32. This is output drift, not automatically an error: use "
                "the control metrics to distinguish improvement from disagreement.",
                "No decision threshold was fitted to these results. Rare labels remain "
                "under-supported, and choosing a runtime from this experiment does not turn "
                "the same corpus into an untouched future test set.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "**Full-corpus candidate accuracy is not included yet. Do not promote an approximate numerical mode based on the pilot alone.**",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf8")
    write_json(
        profiles / "summary.json",
        {"profiles": profiles_result, "single_runs": singles, "quality": quality},
    )
    print(f"[+] Wrote {output} from verified real-inference artifacts.")


def main() -> None:
    """Rebuild the performance report from downloaded or committed experiment evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--single-runs", nargs="*", type=Path, default=[])
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/snapshot.json")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "PERFORMANCE.md")
    args = parser.parse_args()
    if bool(args.reference) != bool(args.candidate):
        parser.error("--reference and --candidate must be supplied together.")
    generate(
        args.profiles,
        args.single_runs,
        args.snapshot,
        args.reference,
        args.candidate,
        args.output,
    )


if __name__ == "__main__":
    main()
