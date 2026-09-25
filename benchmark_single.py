"""Compare real single-issue CPU latency without changing labels, prompts or context."""

from __future__ import annotations

import argparse
import math
import platform
import time
from pathlib import Path

from experiment import (
    ROOT,
    check_response,
    digest,
    load_agent,
    predicted_labels,
    prepare_state,
    questions_for,
    read_json,
    taxonomy,
    write_json,
)


def benchmark(
    config_path: Path, snapshot_path: Path, output: Path, threads: int, precision: str
) -> None:
    """Measure cold/warm predictions for six fixed issues, not a throughput batch."""
    started = time.perf_counter()
    config = read_json(config_path)
    config.update(
        device="cpu",
        cpu_threads=threads,
        mixed_precision=False,
        cpu_precision=precision,
    )
    snapshot = read_json(snapshot_path)
    questions = questions_for(taxonomy(snapshot))
    issues = snapshot["issues"]
    if len(issues) < 6:
        raise ValueError("The comparison requires at least six source issues.")
    positions = [index * (len(issues) - 1) // 5 for index in range(6)]
    agent, runtime = load_agent(config)
    ready_seconds = time.perf_counter() - started
    records = []
    for sample, position in enumerate(positions):
        prepare_start = time.perf_counter()
        state, metadata = prepare_state(issues[position], agent.tok, config)
        preparation_seconds = time.perf_counter() - prepare_start
        for repeat in range(3):
            before = time.perf_counter()
            response = agent.predict(state, questions)
            elapsed = time.perf_counter() - before
            check_response(response, questions)
            if agent.device.type != "cpu" or agent.amp_enabled:
                raise ValueError(
                    "Runtime changed device or precision during measurement."
                )
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid measured latency.")
            row = {
                "number": issues[positions[sample]]["number"],
                "repeat": repeat,
                "evidence": "real-local-model-inference",
                **metadata,
                "inference_seconds": elapsed,
                "preparation_seconds": preparation_seconds,
                "predicted_labels": predicted_labels(
                    response, config["binary_threshold"]
                ),
                "response": response,
            }
            records.append(row)
            write_json(
                output,
                {
                    "status": "running",
                    "precision": precision,
                    "cpu_threads": threads,
                    "runtime": runtime,
                    "quantized_linear_modules": runtime["quantized_linear_modules"],
                    "quantization_backend": runtime["quantization_backend"],
                    "quantization_scope": runtime["quantization_scope"],
                    "optimization_seconds": runtime["optimization_seconds"],
                    "process_to_model_ready_seconds": ready_seconds,
                    "first_issue_process_seconds": ready_seconds
                    + records[0]["preparation_seconds"]
                    + records[0]["inference_seconds"],
                    "snapshot_sha256": digest(snapshot),
                    "config": config,
                    "questions": questions,
                    "records": records,
                    "machine": platform.machine(),
                },
            )
            print(
                f"[>] {precision}/{threads} threads: issue {row['number']}, repeat {repeat}, {elapsed:.3f}s",
                flush=True,
            )
    result = read_json(output)
    result["status"] = "complete"
    result["total_process_seconds"] = time.perf_counter() - started
    write_json(output, result)


def main() -> None:
    """Run one measured CPU configuration, suitable for an Actions comparison matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/snapshot.json")
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--precision", choices=("fp32", "int8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.config, args.snapshot, args.output, args.threads, args.precision)


if __name__ == "__main__":
    main()
