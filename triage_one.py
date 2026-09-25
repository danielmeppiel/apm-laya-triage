"""Read one live APM issue and propose labels locally; never modify GitHub."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from experiment import (
    ROOT,
    check_response,
    digest,
    gh_get,
    load_agent,
    now,
    predicted_labels,
    prepare_state,
    questions_for,
    read_json,
    synchronize,
    taxonomy,
    write_json,
)


def read_issue(repo: str, number: int) -> dict[str, Any]:
    """Return only model-input fields from the GET response; exclude all controls."""
    if number <= 0:
        raise ValueError("Issue number must be positive.")
    data = gh_get(f"repos/{repo}/issues/{number}")
    if "pull_request" in data:
        raise ValueError("The requested item is a pull request, not an issue.")
    return {
        "number": data["number"],
        "title": data["title"],
        "body": data["body"] or "",
        "url": data["html_url"],
    }


def classify(
    number: int, threads: int, precision: str, output: Path, device: str = "cpu"
) -> None:
    """Measure a complete single-issue invocation, including the source read and model setup."""
    started_at = now()
    started = time.perf_counter()
    config = read_json(ROOT / "config.json")
    config.update(
        device=device,
        cpu_threads=threads,
        mixed_precision=False,
        cpu_precision=precision,
    )
    snapshot = read_json(ROOT / "data/snapshot.json")
    questions = questions_for(taxonomy(snapshot))
    before = time.perf_counter()
    issue = read_issue(config["source_repo"], number)
    github_read_seconds = time.perf_counter() - before
    agent, runtime = load_agent(config)
    state, input_metadata = prepare_state(issue, agent.tok, config)
    synchronize(agent)
    before = time.perf_counter()
    response = agent.predict(state, questions)
    synchronize(agent)
    inference_seconds = time.perf_counter() - before
    check_response(response, questions)
    if (
        str(agent.device) != runtime["device"]
        or agent.amp_enabled != runtime["mixed_precision"]
    ):
        raise ValueError("Runtime changed device or precision during inference.")
    result = {
        "started_at": started_at,
        "completed_at": now(),
        "status": "complete",
        "evidence": "real-local-model-inference",
        "github_writes": False,
        "number": number,
        "url": issue["url"],
        **input_metadata,
        "precision": precision,
        "quantized_linear_modules": runtime["quantized_linear_modules"],
        "quantization_backend": runtime["quantization_backend"],
        "quantization_scope": runtime["quantization_scope"],
        "config": config,
        "questions": questions,
        "runtime": runtime,
        "taxonomy_snapshot_sha256": digest(snapshot),
        "github_read_seconds": github_read_seconds,
        "optimization_seconds": runtime["optimization_seconds"],
        "inference_seconds": inference_seconds,
        "invocation_seconds": time.perf_counter() - started,
        "predicted_labels": predicted_labels(response, config["binary_threshold"]),
        "response": response,
    }
    write_json(output, result)
    print(
        f"[+] Issue {number}: {len(questions)} real label decisions; {result['invocation_seconds']:.3f}s invocation. No GitHub writes."
    )


def main() -> None:
    """Expose explicit CPU settings without disguising INT8 as the FP32 baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--precision", choices=("fp32", "int8"), default="fp32")
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/single-issue/result.json"
    )
    args = parser.parse_args()
    classify(args.issue, args.threads, args.precision, args.output, args.device)


if __name__ == "__main__":
    main()
