"""Test metric math with invented data, never as model-performance evidence."""

import copy
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluation import (
    aggregate,
    build_report,
    canonical_labels,
    csv_text,
    escaped,
    evaluate,
    fraction,
    percentile,
    wilson,
)
from experiment import digest, questions_for, write_json


def corpus() -> tuple[dict, list[dict], dict]:
    """Construct a toy corpus with known exact precision, recall and coverage."""
    descriptions = {
        "type/bug": "bug",
        "type/feature": "feature",
        "area/cli": "CLI",
        "theme/security": "security",
    }
    snapshot = {
        "labels": [
            {"name": key, "description": value} for key, value in descriptions.items()
        ],
        "contract": {
            "classification_labels": list(descriptions),
            "legacy_read_aliases": {"bug": "type/bug", "accepted": "status/accepted"},
        },
        "issues": [
            {
                "number": 1,
                "title": "A",
                "body": "a",
                "state": "open",
                "labels": ["bug", "area/cli"],
            },
            {
                "number": 2,
                "title": "B",
                "body": "b",
                "state": "closed",
                "labels": ["type/feature"],
            },
            {
                "number": 3,
                "title": "C",
                "body": "c",
                "state": "open",
                "labels": ["accepted"],
            },
            {
                "number": 4,
                "title": "D",
                "body": "d",
                "state": "closed",
                "labels": ["theme/security"],
            },
        ],
    }
    proposals = [
        ["type/bug", "area/cli", "theme/security"],
        ["type/bug", "area/cli"],
        ["type/bug"],
        ["theme/security", "type/feature"],
    ]
    predictions = []
    for issue, proposed in zip(snapshot["issues"], proposals):
        predictions.append(
            {
                "number": issue["number"],
                "fingerprint": "toy",
                "evidence": "real-local-model-inference",
                "input_sha256": digest(
                    {"title": issue["title"], "body": issue["body"]}
                ),
                "predicted_labels": sorted(proposed),
                "shortened": False,
                "inference_seconds": 1.0,
                "processing_seconds": 1.1,
                "original_state_tokens": 20,
                "used_state_tokens": 20,
                "response": {
                    "answers": {
                        label: {
                            "type": "noul",
                            "noul": 0.9 if label in proposed else 0.1,
                        }
                        for label in descriptions
                    }
                },
            }
        )
    manifest = {
        "status": "complete",
        "real_inference": True,
        "github_writes": False,
        "source_snapshot_sha256": digest(snapshot),
        "prediction_count": 4,
        "target_count": 4,
        "fingerprint": "toy",
        "questions": questions_for(descriptions),
        "config": {"binary_threshold": 0.5},
    }
    return snapshot, predictions, manifest


class EvaluationTests(unittest.TestCase):
    def test_intervals_handle_zero_support_and_small_classes_honestly(self) -> None:
        self.assertIsNone(wilson(0, 0))
        interval = wilson(1, 1)
        self.assertAlmostEqual(interval["upper"], 1.0)
        self.assertLess(interval["lower"], 0.21)
        narrow = wilson(50, 100)
        self.assertAlmostEqual(narrow["lower"], 0.4038315, places=6)
        self.assertAlmostEqual(narrow["upper"], 0.5961685, places=6)

    def test_aliases_do_not_turn_human_acceptance_into_a_control(self) -> None:
        snapshot, _, _ = corpus()
        self.assertEqual(
            canonical_labels(["bug", "type/bug", "accepted"], snapshot["contract"]),
            {"type/bug"},
        )

    def test_missing_dimensions_and_unlabelled_issues_are_not_negatives(self) -> None:
        metrics, _ = evaluate(*corpus())
        self.assertEqual(metrics["control_issues"], 3)
        self.assertEqual(metrics["unscored_issues"], 1)
        self.assertEqual(metrics["proposal_cardinality"]["mean"], 2.0)
        self.assertEqual(
            metrics["proposal_cardinality"]["all_allowed_labels_issues"], 0
        )
        overall = metrics["overall_observed_dimensions"]
        self.assertEqual(overall["exact_matches"], 2)
        self.assertAlmostEqual(overall["exact_agreement"], 2 / 3)
        self.assertEqual(overall["true_positives"], 3)
        self.assertEqual(overall["false_positives"], 1)
        self.assertEqual(overall["false_negatives"], 1)
        self.assertEqual(overall["precision"], 0.75)
        self.assertEqual(overall["recall"], 0.75)
        self.assertEqual(overall["micro_f1"], 0.75)
        self.assertEqual(metrics["by_dimension"]["type"]["control_issues"], 2)
        self.assertEqual(metrics["by_dimension"]["area"]["precision"], 1)

    def test_no_denominator_is_reported_as_undefined_not_success(self) -> None:
        self.assertIsNone(fraction(0, 0))
        metrics = aggregate([{"expected": [], "proposed": ["type/bug"]}])
        self.assertEqual(metrics["control_issues"], 0)
        self.assertIsNone(metrics["exact_agreement"])

    def test_more_than_one_existing_type_is_scored(self) -> None:
        result = aggregate(
            [{"expected": ["type/bug", "type/feature"], "proposed": ["type/bug"]}],
            "type",
        )
        self.assertEqual(result["exact_agreement"], 0)
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 0.5)

    def test_entire_corpus_without_controls_has_no_success_score(self) -> None:
        snapshot, rows, manifest = corpus()
        for issue in snapshot["issues"]:
            issue["labels"] = []
        manifest["source_snapshot_sha256"] = digest(snapshot)
        metrics, _ = evaluate(snapshot, rows, manifest)
        self.assertEqual(metrics["unscored_issues"], 4)
        self.assertIsNone(metrics["overall_observed_dimensions"]["exact_agreement"])
        self.assertIsNone(metrics["macro_f1_supported_labels"])

    def test_corrupt_timing_input_or_run_identity_fails(self) -> None:
        snapshot, rows, manifest = corpus()
        for key, value in [
            ("inference_seconds", float("nan")),
            ("processing_seconds", float("inf")),
            ("processing_seconds", -1),
            ("evidence", "offline-fixture"),
            ("fingerprint", "another-run"),
            ("input_sha256", "different-input"),
        ]:
            with self.subTest(key=key, value=value):
                candidate = copy.deepcopy(rows)
                candidate[0][key] = value
                with self.assertRaises(ValueError):
                    evaluate(snapshot, candidate, manifest)

    def test_partial_duplicate_or_modified_evidence_is_rejected(self) -> None:
        snapshot, rows, manifest = corpus()
        for changed in [rows[:-1], rows + [rows[0]], [rows[0]] * 4]:
            with self.assertRaises(ValueError):
                evaluate(snapshot, changed, manifest)
        candidate = copy.deepcopy(rows)
        candidate[0]["predicted_labels"] = ["type/feature"]
        with self.assertRaises(ValueError):
            evaluate(snapshot, candidate, manifest)
        manifest["status"] = "running"
        with self.assertRaises(ValueError):
            evaluate(snapshot, rows, manifest)

    def test_percentiles_and_spreadsheet_formula_safety(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertEqual(csv_text("=1+1"), "'=1+1")
        self.assertEqual(csv_text("  @payload"), "'  @payload")
        self.assertEqual(csv_text("ordinary title"), "ordinary title")

    def test_unsupported_labels_are_not_given_fake_precision(self) -> None:
        metrics, _ = evaluate(*corpus())
        feature = metrics["per_label"]["type/feature"]
        self.assertEqual(feature["support"], 1)
        self.assertEqual(feature["proposed_count"], 0)
        self.assertIsNone(feature["precision"])
        self.assertEqual(feature["recall"], 0)
        self.assertAlmostEqual(metrics["macro_f1_supported_labels"], 2 / 3)

    def test_markdown_body_and_csv_title_remain_inert(self) -> None:
        rendered = escaped("[click](javascript:alert(1)) <script>bad</script> | extra")
        self.assertIn("\\[click\\]", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("\\|", rendered)

    def test_complete_report_persists_metrics_and_every_proposal(self) -> None:
        snapshot, rows, manifest = corpus()
        snapshot.update(started_at="test", completed_at="test", source_revision="test")
        for issue in snapshot["issues"]:
            issue["url"] = f"https://github.com/microsoft/apm/issues/{issue['number']}"
        snapshot["issues"][0]["title"] = "=1+1"
        rows[0]["input_sha256"] = digest({"title": "=1+1", "body": "a"})
        manifest["source_snapshot_sha256"] = digest(snapshot)
        manifest["config"].update(
            state_token_budget=300, model_repo="unit-test-only", model_revision="test"
        )
        manifest.update(
            completed_at="test",
            runtime={
                "device": "unit-test-only",
                "mixed_precision": False,
                "cpu_threads": 2,
                "platform": "test",
                "model_load_seconds": 1,
                "download_and_hash_seconds": 1,
                "weights_sha256": "test",
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_path = root / "runs" / "custom"
            snapshot_path = root / "snapshot.json"
            write_json(snapshot_path, snapshot)
            write_json(run_path / "manifest.json", manifest)
            (run_path / "predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf8"
            )
            output = root / "reports" / "report.md"
            with redirect_stdout(io.StringIO()):
                build_report(snapshot_path, run_path, output)
            report = output.read_text(encoding="utf8")
            self.assertIn("../runs/custom/metrics.json", report)
            self.assertNotIn("runs/baseline", report)
            metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf8"))
            self.assertEqual(metrics["issue_count"], 4)
            with (run_path / "proposals.csv").open(
                encoding="utf8", newline=""
            ) as stream:
                proposals = list(csv.DictReader(stream))
            self.assertEqual(len(proposals), 4)
            self.assertEqual(proposals[0]["title"], "'=1+1")
            self.assertEqual(proposals[2]["observed_dimensions_exact_match"], "")
            self.assertNotIn(b"\r\n", (run_path / "proposals.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
