"""One explicitly authorized typed-checkpoint comparison; base policy stays frozen.

``download`` is the only network operation: five pinned data files, no Python.
``run`` is offline and reuses the frozen base questions, excerpt policy, strict
response validation, FP32 loader and shared issue subsets.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

from experiment import MODEL_FILES, ROOT, digest, load_agent, now, read_json, synchronize, write_json
from experiments.precision_protocol import file_sha256, load_protocol, prediction_inputs, verify_sources
from experiments.precision_prompts import (
    decode,
    inspect_question_encoding,
    offline_runtime_check,
    prepare_input,
    read_rows,
    selected_ids,
    summarize,
    validate_policy,
    validate_rows,
)

CHECKPOINT = {
    "model_repo": "convaiinnovations/laya-typed-decisions",
    "model_revision": "1a793eb568e6718f15941d08f85432581df534e3",
    "weights_sha256": "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
}
WEIGHT_BYTES = 842609220
DEFAULT_BASE = ROOT / "runs/precision/prompts/frozen-choice-type-theme.json"
DEFAULT_OUTPUT = ROOT / "runs/precision/prompts/typed-screen"


def checkpoint_policy(base: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    validate_policy(base, protocol)
    if base["variant"] != "choice-clean" or base["dimensions_scope"] != ["theme", "type"]:
        raise ValueError("Authorized checkpoint screen is categorical type+theme only.")
    policy = {
        **{key: value for key, value in base.items() if key != "policy_sha256"},
        "config": {**base["config"], **CHECKPOINT},
        "source_hashes": {
            **base["source_hashes"],
            "experiments/precision_typed_checkpoint.py": file_sha256(
                ROOT / "experiments/precision_typed_checkpoint.py"
            ),
        },
        "base_policy_sha256": base["policy_sha256"],
        "checkpoint_profile": "typed-decisions-one-authorized-checkpoint",
        "limitations": [
            *base["limitations"],
            "Checkpoint was trained on four synthetic workflows, not APM labels.",
            "Checkpoint temperatures are inherited; upstream confidence is not calibrated for APM.",
        ],
    }
    policy["policy_sha256"] = digest(policy)
    return policy


def download(output: Path) -> None:
    started = time.perf_counter()
    if not os.environ.get("HF_HOME") or not Path(os.environ["HF_HOME"]).is_dir():
        raise ValueError("Use the existing HF_HOME cache; no implicit new cache.")
    os.environ.update(
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="1", USE_TF="0",
    )
    from huggingface_hub import snapshot_download

    record = {
        "started_at": now(), "status": "downloading", **CHECKPOINT,
        "authorized_weight_bytes": WEIGHT_BYTES,
        "allow_patterns": MODEL_FILES,
        "license": "Apache-2.0",
        "token": False,
        "remote_python": False,
    }
    write_json(output, record)
    try:
        directory = Path(snapshot_download(
            CHECKPOINT["model_repo"], revision=CHECKPOINT["model_revision"],
            allow_patterns=MODEL_FILES, token=False, max_workers=2,
        ))
        if not all((directory / name).is_file() for name in MODEL_FILES):
            raise ValueError("Downloaded checkpoint is incomplete.")
        weights = directory / "model.safetensors"
        if weights.stat().st_size != WEIGHT_BYTES or file_sha256(weights) != CHECKPOINT["weights_sha256"]:
            raise ValueError("Authorized checkpoint size or weight checksum mismatch.")
        record.update(
            status="complete", completed_at=now(), directory=str(directory),
            files={name: {"bytes": (directory / name).stat().st_size,
                          "sha256": file_sha256(directory / name)} for name in MODEL_FILES},
        )
    except BaseException as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                             "traceback": traceback.format_exc()})
        raise
    finally:
        record["download_and_verification_seconds"] = time.perf_counter() - started
        write_json(output, record)
    print(json.dumps(record), flush=True)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    protocol = load_protocol(args.protocol)
    verify_sources(protocol, args.snapshot, args.baseline)
    policy = checkpoint_policy(read_json(args.base_policy), protocol)
    if args.frozen and read_json(args.frozen) != policy:
        raise ValueError("Frozen checkpoint policy differs from current code/config.")
    numbers = selected_ids(protocol, args.partition, args.frozen is not None)
    wanted = set(numbers)
    inputs = {i["number"]: i for i in prediction_inputs(
        [i for i in read_json(args.snapshot)["issues"] if i["number"] in wanted]
    )}
    if set(inputs) != wanted:
        raise ValueError("Missing requested issue inputs.")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    prediction_path = args.output / "predictions.jsonl"
    previous = read_json(manifest_path) if manifest_path.exists() else None
    if previous and (previous["policy"] != policy or previous["issue_numbers"] != numbers):
        raise ValueError("Output belongs to a different policy or subset.")
    rows = read_rows(prediction_path)
    validate_rows(rows, policy, inputs)
    reused = []
    if args.reuse:
        source = read_json(args.reuse / "manifest.json")
        if source["status"] != "complete" or source["policy"] != policy:
            raise ValueError("Reuse requires an identical complete typed policy.")
        reused = read_rows(args.reuse / "predictions.jsonl")
        if {r["number"] for r in reused} != set(source["issue_numbers"]):
            raise ValueError("Reuse source is incomplete.")
        validate_rows(reused, policy, inputs)
    done = {row["number"] for row in rows}
    with prediction_path.open("a", encoding="utf8") as stream:
        for row in reused:
            if row["number"] not in done:
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                rows.append(row)
                done.add(row["number"])
        stream.flush()
        os.fsync(stream.fileno())
    if done == wanted and previous and previous["status"] == "complete":
        print(json.dumps({"status": "already-complete", "count": len(rows)}))
        return
    manifest = {
        "started_at": previous["started_at"] if previous else now(), "status": "running",
        "partition": args.partition, "issue_numbers": numbers, "policy": policy,
        "dimensions_scope": policy["dimensions_scope"], "option_counts": policy["option_counts"],
        "source_snapshot_sha256": protocol["snapshot_sha256"],
        "source_baseline_sha256": protocol["baseline_sha256"],
        "real_inference": True, "github_writes": False, "prior_inference_count": len(rows),
        "reuse_path": str(args.reuse) if args.reuse else None,
        "segments": previous.get("segments", []) if previous else [],
    }
    write_json(manifest_path, manifest)
    try:
        offline = offline_runtime_check(policy["config"])
        with (Path(offline["hf_home"]) / "precision-prompts-gpu.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                agent, runtime = load_agent(policy["config"])
            runtime.update(offline, load_warnings=[str(w.message) for w in captured])
            if runtime["device"] != "mps" or runtime["mixed_precision"] or runtime["cpu_threads"] != 2:
                raise ValueError("Runtime is not the authorized two-thread FP32 MPS configuration.")
            if any(p.is_floating_point() and str(p.dtype) != "torch.float32" for p in agent.model.parameters()):
                raise ValueError("Typed checkpoint was not upcast to FP32.")
            manifest.update(runtime=runtime, question_encoding=inspect_question_encoding(agent, policy["questions"]))
            write_json(manifest_path, manifest)
            print(json.dumps({"status": "model-ready", "checkpoint": CHECKPOINT["model_repo"],
                              "remaining": len(wanted - done), "device": runtime["device"]}), flush=True)
            baseline = {r["number"]: r for r in read_rows(args.baseline) if r["number"] in wanted}
            with prediction_path.open("a", encoding="utf8") as stream:
                for number in numbers:
                    if number in done:
                        continue
                    processing = time.perf_counter()
                    state, metadata = prepare_input(inputs[number], agent.tok, policy["config"], policy["variant"])
                    if metadata["baseline_state_sha256"] != baseline[number]["state_sha256"]:
                        raise ValueError("Alternate tokenizer changed the exact baseline issue excerpt.")
                    synchronize(agent)
                    before = time.perf_counter()
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        response = agent.predict(state, policy["questions"])
                    synchronize(agent)
                    elapsed = time.perf_counter() - before
                    if str(agent.device) != "mps" or agent.amp_enabled:
                        raise ValueError("SDK changed device or precision.")
                    proposed, scores = decode(response, policy["questions"])
                    row = {
                        "number": number, "policy_sha256": policy["policy_sha256"],
                        "evidence": "real-local-model-inference", **metadata,
                        "inference_seconds": elapsed, "processing_seconds": time.perf_counter() - processing,
                        "proposed": proposed, "scores": scores, "response": response,
                        "warnings": [str(w.message) for w in captured],
                    }
                    stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    done.add(number)
                    print(json.dumps({"completed": len(done), "total": len(numbers), "number": number,
                                      "inference_seconds": round(elapsed, 4)}), flush=True)
        manifest.update(status="complete", completed_at=now(), prediction_count=len(rows))
    except BaseException as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc()})
        with (args.output / "errors.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps({"at": now(), **manifest["error"]}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        raise
    finally:
        manifest["segments"].append({"completed_at": now(), "invocation_seconds": time.perf_counter() - started})
        write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "runs/precision/protocol.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/snapshot.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs/baseline/predictions.jsonl")
    parser.add_argument("--base-policy", type=Path, default=DEFAULT_BASE)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--output", type=Path, default=ROOT / "runs/precision/prompts/typed-download.json")
    infer = commands.add_parser("run")
    infer.add_argument("--partition", choices=("screen24", "development64", "confirmation96"), required=True)
    infer.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    infer.add_argument("--reuse", type=Path)
    infer.add_argument("--frozen", type=Path)
    report = commands.add_parser("summarize")
    report.add_argument("--run", type=Path, default=DEFAULT_OUTPUT)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--run", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        download(args.output)
    elif args.command == "run":
        run(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        manifest = read_json(args.run / "manifest.json")
        if manifest["status"] != "complete" or manifest["partition"] != "development64":
            raise ValueError("Freeze requires a complete development64 run.")
        expected = checkpoint_policy(read_json(args.base_policy), load_protocol(args.protocol))
        if manifest["policy"] != expected:
            raise ValueError("Development policy differs from current code/config.")
        if args.output.exists() and read_json(args.output) != expected:
            raise ValueError("Refusing to replace a different frozen policy.")
        write_json(args.output, expected)
        print(json.dumps({"frozen": str(args.output), "policy_sha256": expected["policy_sha256"]}))


if __name__ == "__main__":
    main()
