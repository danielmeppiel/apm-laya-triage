"""Immutable text-only experimental partitions; no model scores are read.

Run from the repository root with ``python3 -m experiments.precision_protocol``.
``load_protocol`` verifies the artifact; ``select_ids`` returns an ordered split
or subset; ``prediction_inputs`` strips labels and metadata from model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation import canonical_labels
from experiment import digest, read_json, taxonomy

SEED = "apm-precision-text-groups-v1-2026-09-25"
SPLITS = ("fit", "development", "confirmation")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def text_groups(issues: list[dict[str, Any]]) -> list[list[int]]:
    """Union exact titles/bodies and >=0.82 Jaccard word-trigram neighbors."""
    inputs = prediction_inputs(issues)
    parents = {issue["number"]: issue["number"] for issue in inputs}
    if len(parents) != len(inputs):
        raise ValueError("Issue IDs must be unique.")

    def root(number: int) -> int:
        while parents[number] != number:
            parents[number] = parents[parents[number]]
            number = parents[number]
        return number

    def union(left: int, right: int) -> None:
        left, right = root(left), root(right)
        parents[max(left, right)] = min(left, right)

    exact: dict[tuple[str, str], int] = {}
    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    shingles: dict[int, set[tuple[str, ...]]] = {}
    for issue in sorted(inputs, key=lambda item: item["number"]):
        number = issue["number"]
        title, body = normalized(issue["title"]), normalized(issue["body"])
        for field, value, minimum in (("title", title, 4), ("body", body, 12)):
            if len(value.split()) >= minimum:
                key = (field, value)
                if key in exact:
                    union(number, exact[key])
                else:
                    exact[key] = number
        tokens = (title + " " + body).split()
        if len(tokens) < 12:
            continue
        current = {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}
        candidates = {other for shingle in current for other in postings[shingle]}
        for other in candidates:
            previous = shingles[other]
            if min(len(current), len(previous)) / max(len(current), len(previous)) < 0.82:
                continue
            if len(current & previous) / len(current | previous) >= 0.82:
                union(number, other)
        shingles[number] = current
        for shingle in current:
            postings[shingle].append(number)
    groups: dict[int, list[int]] = defaultdict(list)
    for number in sorted(parents):
        groups[root(number)].append(number)
    return sorted(groups.values(), key=lambda group: group[0])


def prediction_inputs(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The complete allowed prediction-input schema, with no expected labels."""
    return [
        {"number": issue["number"], "title": issue["title"], "body": issue["body"]}
        for issue in issues
    ]


def order_key(number: int) -> str:
    return digest({"seed": SEED, "purpose": "subset-order", "number": number})


def build_protocol(
    snapshot: dict[str, Any], snapshot_sha256: str, baseline_sha256: str
) -> dict[str, Any]:
    """Assign all issues before consulting control presence, without stratifying."""
    groups = text_groups(snapshot["issues"])
    assignments: dict[str, list[int]] = {name: [] for name in SPLITS}
    group_records = []
    for members in groups:
        group_id = digest({"seed": SEED, "purpose": "split", "members": members})
        bucket = int(group_id[:16], 16) / 2**64
        split = "fit" if bucket < 0.6 else "development" if bucket < 0.8 else "confirmation"
        assignments[split].extend(members)
        group_records.append({"group_id": group_id, "issue_ids": members, "split": split})
    # Control existence filters already-fixed partitions; label values never set a split.
    controls = {
        issue["number"]
        for issue in snapshot["issues"]
        if canonical_labels(issue["labels"], snapshot["contract"])
    }
    partitions = {}
    for name in SPLITS:
        numbers = sorted(assignments[name], key=order_key)
        partitions[name] = {
            "all_ids": numbers,
            "control_ids": [number for number in numbers if number in controls],
            "unlabelled_ids": [number for number in numbers if number not in controls],
        }
    development = partitions["development"]["control_ids"]
    confirmation = partitions["confirmation"]["control_ids"]
    protocol = {
        "schema_version": 1,
        "seed": SEED,
        "snapshot_sha256": snapshot_sha256,
        "baseline_sha256": baseline_sha256,
        "hash_definition": "SHA256 of exact file bytes; protocol_sha256 is experiment.digest without itself",
        "taxonomy": taxonomy(snapshot),
        "legacy_read_aliases": snapshot["contract"]["legacy_read_aliases"],
        "target_fractions": {"fit": 0.6, "development": 0.2, "confirmation": 0.2},
        "assignment": "Hash sorted text-group issue IDs into fixed [0,.6), [.6,.8), [.8,1) ranges",
        "grouping": {
            "normalization": "Unicode NFKC, casefold, word tokens, normalized whitespace",
            "exact_title_min_words": 4,
            "exact_body_min_words": 12,
            "near_duplicate_min_words": 12,
            "near_duplicate_word_trigram_jaccard": 0.82,
            "transitive_connected_components": True,
        },
        "groups": group_records,
        "splits": partitions,
        "subsets": {
            "screen24": development[:24],
            "development64": development[:64],
            "confirmation96": confirmation[:96],
        },
        "counts": {
            "all_issues": len(snapshot["issues"]),
            "controls": len(controls),
            "unlabelled": len(snapshot["issues"]) - len(controls),
            "groups": len(groups),
            "duplicate_groups": sum(len(group) > 1 for group in groups),
            "issues_in_duplicate_groups": sum(len(group) for group in groups if len(group) > 1),
            "splits": {
                name: {key.removesuffix("_ids"): len(value) for key, value in part.items()}
                for name, part in partitions.items()
            },
            "subsets": {"screen24": min(24, len(development)), "development64": min(64, len(development)),
                        "confirmation96": min(96, len(confirmation))},
        },
        "rules": [
            "Fit parameters using fit controls only; choose finite configurations using development only.",
            "screen24 is nested in development64, which is nested in full development.",
            "confirmation96 is nested in full confirmation; parent owns all confirmation evaluation.",
            "Do not inspect confirmation labels or probabilities to design, fit, or select policies.",
            "Predictions may use IDs/title/body and cached probabilities, never expected labels or label-presence metadata.",
            "Unlabelled issues are recorded but excluded from fitting and evaluation.",
            "Entire missing dimensions are unscored, not negative; reuse evaluation.scored_sets/aggregate.",
            "Freeze at most two policies per workstream before parent confirmation.",
            "Report silver-label agreement, not gold true precision, with recall and coverage.",
        ],
        "limitations": [
            "All-corpus results were previously inspected; confirmation is reserved from NEW experiments only, not never-seen data.",
            "Text-only splitting is not label-stratified; label and dimension support can differ or be absent.",
            "Near-duplicate grouping is lexical, not semantic; paraphrases may remain across splits.",
            "Exact shared titles/bodies and transitive closure can overgroup templates or generic reports.",
            "Unbalanced group sizes mean realized fractions need not equal targets exactly.",
            "Control-label presence is used only for eligibility after assignment, not for outcome-based selection.",
            "The development screen and full development are reused for selection, not independent validation.",
        ],
    }
    protocol["protocol_sha256"] = digest(protocol)
    return protocol


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = read_json(path)
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if digest(payload) != protocol["protocol_sha256"]:
        raise ValueError("Protocol content hash mismatch.")
    if protocol["schema_version"] != 1:
        raise ValueError("Unsupported precision protocol schema.")
    return protocol


def select_ids(protocol: dict[str, Any], name: str) -> list[int]:
    if name in protocol["splits"]:
        return list(protocol["splits"][name]["control_ids"])
    if name in protocol["subsets"]:
        return list(protocol["subsets"][name])
    raise ValueError(f"Unknown protocol split/subset: {name}")


def verify_sources(protocol: dict[str, Any], snapshot_path: Path, baseline_path: Path) -> None:
    for key, path in (("snapshot_sha256", snapshot_path), ("baseline_sha256", baseline_path)):
        if file_sha256(path) != protocol[key]:
            raise ValueError(f"Frozen source mismatch: {path}")


def persist_immutable(path: Path, protocol: dict[str, Any]) -> None:
    content = json.dumps(protocol, indent=2, ensure_ascii=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf8") != content:
            raise ValueError("Refusing to replace an existing experimental protocol.")
        return
    with path.open("x", encoding="utf8") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("data/snapshot.json"))
    parser.add_argument("--baseline", type=Path, default=Path("runs/baseline/predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/precision/protocol.json"))
    args = parser.parse_args()
    protocol = build_protocol(read_json(args.snapshot), file_sha256(args.snapshot), file_sha256(args.baseline))
    persist_immutable(args.output, protocol)
    print(json.dumps({"path": str(args.output.resolve()), "counts": protocol["counts"],
                      "protocol_sha256": protocol["protocol_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
