"""Run disjoint issue shards and combine their unchanged inference evidence."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

from evaluation import build_report, evaluate
from experiment import ROOT, digest, now, read_json, run, write_json


def shard_snapshot(parent: dict[str, Any], index: int, count: int) -> dict[str, Any]:
    """Distribute issues round-robin, retaining the same text and control rubric."""
    if not 1 <= count <= len(parent["issues"]) or not 0 <= index < count:
        raise ValueError("Shard count/index must describe a nonempty partition.")
    issues = parent["issues"][index::count]
    return {
        **parent,
        "issues": issues,
        "issue_count": len(issues),
        "shard": {
            "index": index,
            "count": count,
            "parent_snapshot_sha256": digest(parent),
        },
    }


def combine(parent_path: Path, shard_root: Path, output: Path) -> None:
    """Require complete, compatible shards; never relabel or rewrite their predictions."""
    import json

    parent = read_json(parent_path)
    folders = sorted(path.parent for path in shard_root.glob("*/manifest.json"))
    if not folders:
        raise ValueError("No shard manifests were downloaded.")
    manifests = []
    source_runs = {}
    predictions = []
    observed_indexes = set()
    reference = None
    sources = []
    for folder in folders:
        snapshot = read_json(folder / "snapshot.json")
        shard = snapshot["shard"]
        index, count = shard["index"], shard["count"]
        if count != len(folders) or index in observed_indexes:
            raise ValueError("Missing, duplicate, or inconsistent shard indexes.")
        observed_indexes.add(index)
        if digest(snapshot) != digest(shard_snapshot(parent, index, count)):
            raise ValueError(
                "Shard snapshot does not match its exact parent partition."
            )
        manifest = read_json(folder / "manifest.json")
        raw = (folder / "predictions.jsonl").read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf8").splitlines()]
        evaluate(snapshot, rows, manifest)
        compatibility = {
            "config": manifest["config"],
            "questions": manifest["questions"],
            "pipeline_sources": manifest.get("pipeline_sources"),
            "quantization_backend": manifest["runtime"].get("quantization_backend"),
            "runtime": {
                key: manifest["runtime"][key]
                for key in (
                    "device",
                    "mixed_precision",
                    "versions",
                    "weights_sha256",
                    "cpu_threads",
                )
            },
        }
        if reference is not None and compatibility != reference:
            raise ValueError(
                "Cannot combine different models, rubrics, devices or runtimes."
            )
        reference = compatibility
        fingerprint = manifest["fingerprint"]
        if fingerprint in source_runs:
            raise ValueError("Duplicate source run fingerprint.")
        source_runs[fingerprint] = {
            "issue_numbers": [issue["number"] for issue in snapshot["issues"]],
            "manifest_sha256": digest(manifest),
            "snapshot_sha256": digest(snapshot),
            "predictions_sha256": hashlib.sha256(raw).hexdigest(),
            "path": f"shards/{index}",
        }
        sources.append((folder, index))
        manifests.append(manifest)
        predictions.extend(rows)
    if observed_indexes != set(range(len(folders))):
        raise ValueError("Shard indexes do not cover the declared partition.")
    first = manifests[0]
    executions = [item.get("execution", {}) for item in manifests]
    runtime = {
        **first["runtime"],
        "shard_count": len(manifests),
        "platform": "; ".join(
            sorted({item["runtime"]["platform"] for item in manifests})
        ),
        "model_load_seconds": sum(
            item["runtime"]["model_load_seconds"] for item in manifests
        ),
        "download_and_hash_seconds": sum(
            item["runtime"]["download_and_hash_seconds"] for item in manifests
        ),
        "optimization_seconds": sum(
            item["runtime"].get("optimization_seconds", 0.0) for item in manifests
        ),
    }
    manifest = {
        "fingerprint": digest({"source_runs": source_runs, "snapshot": digest(parent)}),
        "source_snapshot_sha256": digest(parent),
        "source_runs": source_runs,
        "status": "complete",
        "real_inference": True,
        "github_writes": False,
        "started_at": min(item["started_at"] for item in manifests),
        "completed_at": now(),
        "target_count": len(parent["issues"]),
        "full_corpus_count": len(parent["issues"]),
        "prediction_count": len(predictions),
        "config": first["config"],
        "questions": first["questions"],
        "runtime": runtime,
        "execution": {
            "kind": "github-actions"
            if all(item.get("kind") == "github-actions" for item in executions)
            else "local-shards",
            "sources": list({digest(item): item for item in executions}.values()),
        },
    }
    evaluate(parent, predictions, manifest)
    if (output / "manifest.json").exists() and read_json(output / "manifest.json")[
        "fingerprint"
    ] != manifest["fingerprint"]:
        raise ValueError("Output directory already belongs to another combined run.")
    output.mkdir(parents=True, exist_ok=True)
    for folder, index in sources:
        archive = output / "shards" / str(index)
        archive.mkdir(parents=True, exist_ok=True)
        for filename in ("snapshot.json", "manifest.json", "predictions.jsonl"):
            shutil.copyfile(folder / filename, archive / filename)
    predictions.sort(key=lambda item: item["number"])
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in predictions),
        encoding="utf8",
    )
    write_json(output / "snapshot.json", parent)
    write_json(output / "manifest.json", manifest)
    print(
        f"[+] Combined {len(predictions)} real predictions from {len(manifests)} verified shards."
    )


def main() -> None:
    """Expose partitioned inference and a fail-closed aggregate/report command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("run")
    infer.add_argument("--snapshot", type=Path, default=ROOT / "data/snapshot.json")
    infer.add_argument("--config", type=Path, default=ROOT / "config.json")
    infer.add_argument("--index", type=int, required=True)
    infer.add_argument("--count", type=int, default=8)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--precision", choices=("fp32", "int8"), default="fp32")
    infer.add_argument("--threads", type=int, choices=(1, 2, 4), default=2)
    aggregate = commands.add_parser("combine")
    aggregate.add_argument("--snapshot", type=Path, default=ROOT / "data/snapshot.json")
    aggregate.add_argument("--shards-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, default=ROOT / "runs/actions")
    aggregate.add_argument("--report", type=Path, default=ROOT / "REPORT-actions.md")
    args = parser.parse_args()
    if args.command == "run":
        selected = shard_snapshot(read_json(args.snapshot), args.index, args.count)
        destination = args.output / "snapshot.json"
        if destination.exists() and digest(read_json(destination)) != digest(selected):
            raise ValueError("Existing shard belongs to another snapshot or partition.")
        write_json(destination, selected)
        config = read_json(args.config)
        config.update(
            device="cpu", cpu_precision=args.precision, cpu_threads=args.threads
        )
        run(config, destination, args.output, None)
        manifest = read_json(args.output / "manifest.json")
        manifest["execution"] = {
            "kind": "github-actions"
            if os.environ.get("GITHUB_ACTIONS") == "true"
            else "local-shards",
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "repository": os.environ.get("GITHUB_REPOSITORY"),
        }
        write_json(args.output / "manifest.json", manifest)
    else:
        combine(args.snapshot, args.shards_root, args.output)
        build_report(args.snapshot, args.output, args.report)


if __name__ == "__main__":
    main()
