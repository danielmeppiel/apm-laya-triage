"""Offline, serialized Laya formulation experiments with frozen protocol IDs.

Run ``python -m experiments.precision_prompts --help`` from the repository root.
No source labels enter the predictor. Confirmation prediction requires a frozen
policy; confirmation scoring belongs to the parent and is deliberately absent.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import statistics
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

from evaluation import aggregate, canonical_labels, label_metrics
from experiment import (
    ROOT,
    check_response,
    digest,
    load_agent,
    now,
    prepare_state,
    questions_for,
    read_json,
    synchronize,
    write_json,
)
from experiments.precision_protocol import (
    file_sha256,
    load_protocol,
    prediction_inputs,
    select_ids,
    verify_sources,
)

VARIANTS = (
    "membership-clean",
    "predicate-context",
    "predicate-clean",
    "choice-clean",
)
PREDICATES = {
    "type/architecture": "Does this issue request a new software module, design pattern, or interface contract?",
    "type/automation": "Does this issue request changes to an automation script or CI workflow?",
    "type/bug": "Does this issue report existing software behaving incorrectly or failing?",
    "type/docs": "Does this issue request changes to documentation, prose, or examples?",
    "type/feature": "Does this issue request a new capability, command flag, or supported operation?",
    "type/performance": "Does this issue request faster execution, higher throughput, or lower memory usage?",
    "type/refactor": "Does this issue request restructuring internal code without changing its behavior?",
    "type/release": "Does this issue request release engineering, a version bump, or distribution packaging?",
    "theme/governance": "Does this issue request policy enforcement, auditing rules, or enterprise governance?",
    "theme/portability": "Does this issue request support for installing or deploying packages across environments or coding assistants?",
    "theme/security": "Does this issue request protection from malicious content, tampering, or untrusted MCP servers?",
}
CHOICE_DESCRIPTIONS = {
    "type/architecture": "new module, design pattern, interface contract",
    "type/automation": "automation scripts, CI workflows, dependabot",
    "type/bug": "existing behavior is broken or incorrect",
    "type/docs": "documentation, prose, examples",
    "type/feature": "new capability, flag, operation",
    "type/performance": "speed, throughput, memory usage",
    "type/refactor": "internal restructuring without behavior change",
    "type/release": "release, version bump, distribution packaging",
    "theme/governance": "policy rules, auditing, enterprise enforcement",
    "theme/portability": "installing or deploying across environments",
    "theme/security": "malicious content, integrity, trust boundaries",
}
SOURCES = (
    "experiments/precision_prompts.py",
    "experiments/precision_protocol.py",
    "experiment.py",
    "cpu_runtime.py",
    "requirements.txt",
)


def dimensions_scope(text: str) -> list[str]:
    dimensions = text.split(",")
    if (
        not dimensions
        or len(set(dimensions)) != len(dimensions)
        or not set(dimensions) <= {"type", "theme"}
    ):
        raise ValueError("Scope must contain unique type/theme dimensions; area is not supported.")
    return sorted(dimensions)


def make_questions(
    labels: dict[str, str], variant: str, dimensions: list[str]
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown formulation: {variant}")
    dimensions_scope(",".join(dimensions))
    scoped = {
        label: description
        for label, description in sorted(labels.items())
        if label.split("/")[0] in dimensions
    }
    if not scoped or any(label not in PREDICATES for label in scoped):
        raise ValueError("Taxonomy is not supported by the frozen concrete predicates.")
    if variant == "membership-clean":
        return questions_for(scoped)
    if variant.startswith("predicate-"):
        return {
            label: {"type": "noul", "instructions": PREDICATES[label]}
            for label in scoped
        }
    questions = {}
    for dimension in dimensions:
        criteria = {
            label: CHOICE_DESCRIPTIONS[label]
            for label in scoped
            if label.startswith(dimension + "/")
        }
        criteria["none"] = "none of these, other, or insufficient information"
        if not 2 <= len(criteria) <= 10:
            raise ValueError("Choice must have 2..10 options, including none.")
        questions[dimension] = {
            "type": "choice",
            "instructions": (
                "What is the primary kind of work requested in this issue?"
                if dimension == "type"
                else "What is the primary theme of the requested work?"
            ),
            "criteria": criteria,
        }
    return questions


def prepare_input(
    issue: dict[str, Any], tokenizer: Any, config: dict[str, Any], variant: str
) -> tuple[str, dict[str, Any]]:
    """Strip context after shortening: no arm receives extra issue content."""
    state, metadata = prepare_state(
        {"title": issue["title"], "body": issue["body"]}, tokenizer, config
    )
    baseline_state = state
    if variant.endswith("-clean"):
        if "\nTitle:" not in state:
            raise ValueError("Prepared baseline state lost the project/title boundary.")
        state = "Title:" + state.split("\nTitle:", 1)[1]
    metadata.update(
        baseline_state_sha256=digest(baseline_state),
        baseline_used_state_tokens=metadata["used_state_tokens"],
        state_sha256=digest(state),
        used_state_tokens=len(tokenizer(state, add_special_tokens=False)["input_ids"]),
        state=state,
        excerpt_policy="baseline-300-token-head-tail-then-strip-project-if-clean",
    )
    if metadata["used_state_tokens"] > config["state_token_budget"]:
        raise ValueError("Ablation exceeded the frozen state budget.")
    return state, metadata


def decode(response: dict[str, Any], questions: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    check_response(response, questions)
    proposed, scores = [], {}
    for key, answer in response["answers"].items():
        if questions[key]["type"] == "noul":
            scores[key] = answer["noul"]
            if answer["noul"] >= 0.5:
                proposed.append(key)
        else:
            scores.update(
                {label: value for label, value in answer["probabilities"].items() if label != "none"}
            )
            if answer["choice"] != "none":
                proposed.append(answer["choice"])
    return sorted(proposed), scores


def source_hashes() -> dict[str, str]:
    return {name: file_sha256(ROOT / name) for name in SOURCES}


def policy_for(protocol: dict[str, Any], variant: str, dimensions: list[str]) -> dict[str, Any]:
    config = read_json(ROOT / "config.json")
    config.update(device="mps", mixed_precision=False, cpu_threads=2, cpu_precision="fp32")
    if config["state_token_budget"] != 300 or config["binary_threshold"] != 0.5:
        raise ValueError("These ablations require the original 300-token budget and 0.5 cutoff.")
    questions = make_questions(protocol["taxonomy"], variant, dimensions)
    policy = {
        "schema_version": 1,
        "variant": variant,
        "dimensions_scope": dimensions,
        "config": config,
        "questions": questions,
        "questions_sha256": digest(questions),
        "option_counts": {
            key: len(q["criteria"]) if q["type"] == "choice" else 2
            for key, q in questions.items()
        },
        "decision_rule": "primary-argmax-with-none" if variant == "choice-clean" else "independent-noul>=0.5",
        "excerpt_policy": "baseline-300-token-head-tail-then-strip-project-if-clean",
        "protocol_sha256": protocol["protocol_sha256"],
        "source_hashes": source_hashes(),
        "model_input_fields": ["title", "body"],
        "limitations": [
            "Silver-label agreement, not verified gold precision.",
            "Missing control dimensions are unscored, not negative.",
            "Development screen is nested in development64, not independent.",
            "Confirmation is reserved from new experiments only; historical baseline saw every issue.",
            "Choice predicts at most one label per dimension; binary baseline permits multiple.",
        ],
    }
    policy["policy_sha256"] = digest(policy)
    return policy


def validate_policy(policy: dict[str, Any], protocol: dict[str, Any]) -> None:
    if digest({k: v for k, v in policy.items() if k != "policy_sha256"}) != policy["policy_sha256"]:
        raise ValueError("Frozen policy hash mismatch.")
    expected = policy_for(protocol, policy["variant"], policy["dimensions_scope"])
    if policy != expected:
        raise ValueError("Frozen policy differs from current protocol, code, or config.")


def selected_ids(protocol: dict[str, Any], partition: str, frozen: bool) -> list[int]:
    limits = {"screen24": 24, "development64": 64, "confirmation96": 96}
    if partition not in limits:
        raise ValueError("Only the shared screen24, development64, or parent confirmation96 is supported.")
    if partition == "confirmation96" and not frozen:
        raise ValueError("Parent confirmation inference requires --frozen.")
    numbers = select_ids(protocol, partition)
    if not numbers or len(numbers) > limits[partition] or len(numbers) != len(set(numbers)):
        raise ValueError("Invalid or oversized protocol subset.")
    return numbers


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def validate_rows(
    rows: list[dict[str, Any]], policy: dict[str, Any], inputs: dict[int, dict[str, Any]]
) -> None:
    seen = set()
    for row in rows:
        number = row["number"]
        if number not in inputs or number in seen:
            raise ValueError("Duplicate or out-of-subset prediction.")
        seen.add(number)
        issue = inputs[number]
        if (
            row["policy_sha256"] != policy["policy_sha256"]
            or row["evidence"] != "real-local-model-inference"
            or row["input_sha256"] != digest({"title": issue["title"], "body": issue["body"]})
            or row["state_sha256"] != digest(row["state"])
        ):
            raise ValueError("Prediction provenance mismatch.")
        proposed, scores = decode(row["response"], policy["questions"])
        if row["proposed"] != proposed or row["scores"] != scores:
            raise ValueError("Stored decisions do not match response probabilities.")
        for key in ("inference_seconds", "processing_seconds"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Prediction timing must be finite and positive.")


def offline_runtime_check(config: dict[str, Any]) -> dict[str, Any]:
    os.environ.update(
        USE_TF="0",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
    )
    home = os.environ.get("HF_HOME")
    if not home or not Path(home).is_dir():
        raise ValueError("HF_HOME must explicitly name the existing model cache.")
    distribution = importlib.metadata.distribution("laya")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != config["runtime_source_revision"]:
        raise ValueError("Installed Laya source revision differs from the pinned runtime.")
    for line in (ROOT / "requirements.txt").read_text(encoding="utf8").splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            if importlib.metadata.version(name) != version:
                raise ValueError(f"Installed {name} differs from requirements.txt.")
    return {"laya_direct_url": direct_url, "offline": True, "hf_home": str(Path(home).resolve())}


def inspect_question_encoding(agent: Any, questions: dict[str, Any]) -> dict[str, Any]:
    """Reject hidden instruction/option shortening before any real forward pass."""
    from laya.common import QTYPES, render_options, temp_bucket

    result = {}
    for key, question in questions.items():
        internal = agent._to_internal(question)
        options = render_options(internal)
        option_tokens = [
            len(agent.tok(" " + option, add_special_tokens=False)["input_ids"])
            for option in options
        ]
        head_tokens = len(
            agent.tok(f"{question['type']} question: {question['instructions']}", add_special_tokens=False)["input_ids"]
        )
        total = head_tokens + sum(count + 1 for count in option_tokens)
        if max(option_tokens) > 48 or total > agent.cfg["head_max_len"]:
            raise ValueError(f"SDK would silently shorten the question/options for {key}.")
        bucket = temp_bucket(QTYPES[question["type"]], len(options))
        temperature = agent.temperature_by_options.get(bucket, agent.temperature[QTYPES[question["type"]]])
        raw_temperature = agent.temperature_by_options_raw.get(bucket, agent.temperature_raw[QTYPES[question["type"]]])
        if temperature != raw_temperature:
            raise ValueError(f"Active temperature bucket is clamped: {bucket}.")
        result[key] = {
            "instruction_tokens": head_tokens,
            "option_tokens": option_tokens,
            "total_head_tokens": total,
            "head_max_len": agent.cfg["head_max_len"],
            "temperature_bucket": bucket,
            "temperature": temperature,
            "rendered_options": options,
        }
    return result


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    protocol = load_protocol(args.protocol)
    verify_sources(protocol, args.snapshot, args.baseline)
    numbers = selected_ids(protocol, args.partition, args.frozen is not None)
    policy = read_json(args.frozen) if args.frozen else policy_for(
        protocol, args.variant, dimensions_scope(args.dimensions)
    )
    validate_policy(policy, protocol)
    wanted = set(numbers)
    inputs = {
        issue["number"]: issue
        for issue in prediction_inputs(
            [issue for issue in read_json(args.snapshot)["issues"] if issue["number"] in wanted]
        )
    }
    if set(inputs) != wanted:
        raise ValueError("Snapshot does not contain the exact requested subset.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, prediction_path = output / "manifest.json", output / "predictions.jsonl"
    previous = read_json(manifest_path) if manifest_path.exists() else None
    if previous and (previous["policy"] != policy or previous["issue_numbers"] != numbers):
        raise ValueError("Output belongs to a different policy or subset.")
    rows = read_rows(prediction_path)
    validate_rows(rows, policy, inputs)
    if args.reuse:
        reused_manifest = read_json(args.reuse / "manifest.json")
        if reused_manifest["status"] != "complete" or reused_manifest["policy"] != policy:
            raise ValueError("Reuse requires a complete run with the identical code-frozen policy.")
        reused = read_rows(args.reuse / "predictions.jsonl")
        validate_rows(reused, policy, inputs)
        done = {row["number"] for row in rows}
        with prediction_path.open("a", encoding="utf8") as stream:
            for row in reused:
                if row["number"] not in done:
                    stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                    rows.append(row)
            stream.flush()
            os.fsync(stream.fileno())
    done = {row["number"] for row in rows}
    if done == wanted and previous and previous["status"] == "complete":
        print(json.dumps({"status": "already-complete", "output": str(output), "count": len(rows)}))
        return
    manifest = {
        "started_at": previous["started_at"] if previous else now(),
        "status": "running",
        "partition": args.partition,
        "issue_numbers": numbers,
        "policy": policy,
        "dimensions_scope": policy["dimensions_scope"],
        "option_counts": policy["option_counts"],
        "real_inference": True,
        "github_writes": False,
        "source_snapshot_sha256": protocol["snapshot_sha256"],
        "source_baseline_sha256": protocol["baseline_sha256"],
        "prior_inference_count": len(rows),
        "reuse_path": str(args.reuse) if args.reuse else None,
        "segments": previous.get("segments", []) if previous else [],
    }
    write_json(manifest_path, manifest)
    try:
        offline = offline_runtime_check(policy["config"])
        lock_path = Path(offline["hf_home"]) / "precision-prompts-gpu.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                agent, runtime = load_agent(policy["config"])
            runtime.update(offline)
            runtime["load_warnings"] = [str(w.message) for w in captured]
            if runtime["device"] != "mps" or runtime["mixed_precision"] or runtime["cpu_threads"] != 2:
                raise ValueError("Runtime is not the authorized two-thread FP32 MPS configuration.")
            for name, parameter in agent.model.named_parameters():
                if parameter.is_floating_point() and str(parameter.dtype) != "torch.float32":
                    raise ValueError(f"Non-FP32 model parameter: {name}")
            manifest["question_encoding"] = inspect_question_encoding(agent, policy["questions"])
            manifest["runtime"] = runtime
            write_json(manifest_path, manifest)
            print(json.dumps({"status": "model-ready", "partition": args.partition, "variant": policy["variant"],
                              "remaining": len(wanted - done), "device": runtime["device"]}), flush=True)
            with prediction_path.open("a", encoding="utf8") as stream:
                for number in numbers:
                    if number in done:
                        continue
                    processing_start = time.perf_counter()
                    state, metadata = prepare_input(inputs[number], agent.tok, policy["config"], policy["variant"])
                    synchronize(agent)
                    inference_start = time.perf_counter()
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        response = agent.predict(state, policy["questions"])
                    synchronize(agent)
                    inference_seconds = time.perf_counter() - inference_start
                    if str(agent.device) != "mps" or agent.amp_enabled:
                        raise ValueError("SDK device or precision changed during inference.")
                    proposed, scores = decode(response, policy["questions"])
                    row = {
                        "number": number,
                        "policy_sha256": policy["policy_sha256"],
                        "evidence": "real-local-model-inference",
                        **metadata,
                        "inference_seconds": inference_seconds,
                        "processing_seconds": time.perf_counter() - processing_start,
                        "proposed": proposed,
                        "scores": scores,
                        "response": response,
                        "warnings": [str(w.message) for w in captured],
                    }
                    stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    done.add(number)
                    print(json.dumps({"completed": len(done), "total": len(numbers), "number": number,
                                      "inference_seconds": round(inference_seconds, 4)}), flush=True)
        manifest.update(status="complete", completed_at=now(), prediction_count=len(rows))
    except BaseException as error:
        manifest.update(status="failed", failed_at=now(), error={"type": type(error).__name__, "message": str(error),
                                                               "traceback": traceback.format_exc()})
        with (output / "errors.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps({"at": now(), **manifest["error"]}, ensure_ascii=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        write_json(manifest_path, manifest)
        raise
    finally:
        manifest["segments"].append({"completed_at": now(), "invocation_seconds": time.perf_counter() - started})
        write_json(manifest_path, manifest)


def metrics(records: list[dict[str, Any]], dimensions: list[str]) -> dict[str, Any]:
    values = aggregate(records)
    values["by_dimension"] = {dimension: aggregate(records, dimension) for dimension in dimensions}
    values["by_label"] = {
        label: label_metrics(records, label)
        for label in sorted(PREDICATES)
        if label.split("/")[0] in dimensions
    }
    values["issue_coverage"] = sum(bool(r["proposed"]) for r in records) / len(records)
    values["average_proposals"] = sum(len(r["proposed"]) for r in records) / len(records)
    values["dimension_coverage"] = {
        dimension: sum(any(label.startswith(dimension + "/") for label in r["proposed"]) for r in records) / len(records)
        for dimension in dimensions
    }
    return values


def summarize(args: argparse.Namespace) -> None:
    protocol = load_protocol(args.protocol)
    verify_sources(protocol, args.snapshot, args.baseline)
    manifest = read_json(args.run / "manifest.json")
    if manifest["partition"] not in ("screen24", "development64"):
        raise ValueError("Confirmation scoring is reserved for the parent.")
    if manifest["status"] != "complete":
        raise ValueError("Cannot summarize an incomplete run.")
    policy = manifest["policy"]
    dimensions = policy["dimensions_scope"]
    numbers = selected_ids(protocol, manifest["partition"], False)
    if manifest["issue_numbers"] != numbers:
        raise ValueError("Manifest subset does not match the protocol.")
    snapshot = read_json(args.snapshot)
    selected = {i["number"]: i for i in snapshot["issues"] if i["number"] in set(numbers)}
    rows = read_rows(args.run / "predictions.jsonl")
    validate_rows(rows, policy, selected)
    if len(rows) != len(numbers):
        raise ValueError("Run has missing predictions.")
    baseline = {r["number"]: r for r in read_rows(args.baseline) if r["number"] in selected}
    records, baseline_records = [], []
    for row in rows:
        issue = selected[row["number"]]
        expected = sorted(
            label for label in canonical_labels(issue["labels"], snapshot["contract"])
            if label.split("/")[0] in dimensions
        )
        records.append({"expected": expected, "proposed": row["proposed"]})
        base = baseline[row["number"]]
        if base["input_sha256"] != row["input_sha256"] or base["state_sha256"] != row["baseline_state_sha256"]:
            raise ValueError("Comparator did not use the exact same issue excerpt.")
        baseline_records.append({
            "expected": expected,
            "proposed": sorted(label for label, answer in base["response"]["answers"].items()
                               if label.split("/")[0] in dimensions and answer["noul"] >= 0.5),
        })
    result = {
        "partition": manifest["partition"],
        "variant": policy["variant"],
        "dimensions_scope": dimensions,
        "issue_count": len(rows),
        "policy_sha256": policy["policy_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "candidate": metrics(records, dimensions),
        "baseline_same_scope": metrics(baseline_records, dimensions),
        "latency": {
            "inference_seconds_total": sum(r["inference_seconds"] for r in rows),
            "inference_seconds_mean": statistics.mean(r["inference_seconds"] for r in rows),
            "inference_seconds_median": statistics.median(r["inference_seconds"] for r in rows),
            "invocation_seconds_segments": sum(s["invocation_seconds"] for s in manifest["segments"]),
            "shortened_count": sum(r["shortened"] for r in rows),
        },
        "limitations": policy["limitations"],
    }
    write_json(args.run / "metrics.json", result)
    print(json.dumps({key: result[key] for key in ("partition", "variant", "dimensions_scope", "issue_count")}))
    print(json.dumps({name: {key: result[name][key] for key in
                           ("precision", "recall", "micro_f1", "issue_coverage", "average_proposals")}
                      for name in ("candidate", "baseline_same_scope")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "runs/precision/protocol.json")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/snapshot.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "runs/baseline/predictions.jsonl")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("run", help="Offline real inference on an authorized shared subset.")
    infer.add_argument("--partition", choices=("screen24", "development64", "confirmation96"), required=True)
    infer.add_argument("--variant", choices=VARIANTS, default="predicate-clean")
    infer.add_argument("--dimensions", default="type")
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--reuse", type=Path, help="Reuse complete screen rows under the identical code-frozen policy.")
    infer.add_argument("--frozen", type=Path, help="Replay this exact policy; required for parent confirmation.")
    report = commands.add_parser("summarize", help="Score screen/development only against same-scope cached baseline.")
    report.add_argument("--run", type=Path, required=True)
    freeze = commands.add_parser("freeze", help="Freeze a completed development policy without evaluating confirmation.")
    freeze.add_argument("--run", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        manifest = read_json(args.run / "manifest.json")
        if manifest["partition"] != "development64" or manifest["status"] != "complete":
            raise ValueError("Freeze requires completed development64 inference.")
        policy = manifest["policy"]
        validate_policy(policy, load_protocol(args.protocol))
        if args.output.exists() and read_json(args.output) != policy:
            raise ValueError("Refusing to overwrite a different frozen policy.")
        write_json(args.output, policy)
        print(json.dumps({"frozen": str(args.output), "policy_sha256": policy["policy_sha256"]}))


if __name__ == "__main__":
    main()
