"""Synthetic checks for the confirmation evaluator, not model evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.precision_confirmation import (
    compare,
    evaluate_plan,
    file_sha256,
    load_predictions,
    summarize,
)


class ConfirmationTests(unittest.TestCase):
    def records(self, proposed):
        return [
            {"number": i + 1, "expected": ["type/bug"], "proposed": list(labels)}
            for i, labels in enumerate(proposed)
        ]

    def test_unobserved_dimension_is_not_a_negative(self):
        result = summarize(self.records([["type/bug", "area/cli"]]))
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["observed_dimension_coverage"], 1)
        self.assertEqual(result["mean_labels_proposed"], 2)

    def test_abstention_is_not_a_precision_win(self):
        baseline = self.records([["type/bug", "type/feature"]] * 4)
        result = compare(
            {"baseline": baseline, "abstain": self.records([[]] * 4)},
            dict.fromkeys(range(1, 5), "duplicate-cluster"),
            "baseline",
            samples=100,
        )
        self.assertEqual(result["duplicate_group_count"], 1)
        self.assertIsNone(result["metrics"]["abstain"]["precision"])
        self.assertEqual(result["metrics"]["abstain"]["recall"], 0)
        self.assertEqual(result["metrics"]["abstain"]["issue_coverage"], 0)
        delta = result["differences"]["abstain"]
        self.assertFalse(delta["clear_precision_win"])
        self.assertEqual(delta["precision"]["undefined_resamples"], 100)

    def test_clear_win_and_bootstrap_are_reproducible(self):
        records = {
            "baseline": self.records([["type/bug", "type/feature"]] * 4),
            "better": self.records([["type/bug"]] * 4),
        }
        groups = {1: "a", 2: "a", 3: "b", 4: "c"}
        result = compare(records, groups, "baseline", samples=100)
        self.assertEqual(result, compare(records, groups, "baseline", samples=100))
        self.assertTrue(result["differences"]["better"]["clear_precision_win"])
        self.assertAlmostEqual(
            result["differences"]["better"]["precision"]["difference"], 0.5
        )
        self.assertEqual(result["duplicate_group_count"], 3)

    def test_global_recall_cannot_hide_abandoning_an_observed_dimension(self):
        expected = ["type/bug", "type/feature", "area/cli"]
        baseline = [
            {
                "number": 1,
                "expected": expected,
                "proposed": [*expected, "type/docs", "area/testing"],
            }
        ]
        narrow = [
            {
                "number": 1,
                "expected": expected,
                "proposed": ["type/bug", "type/feature"],
            }
        ]
        result = compare(
            {"baseline": baseline, "narrow": narrow},
            {1: "one"},
            "baseline",
            samples=100,
        )
        self.assertTrue(result["differences"]["narrow"]["clear_precision_win"])
        self.assertFalse(result["differences"]["narrow"]["broad_precision_win"])

    def test_mismatched_ids_or_reference_labels_fail(self):
        records = self.records([["type/bug"]])
        with self.assertRaisesRegex(ValueError, "identical paired"):
            compare({"base": records, "bad": []}, {1: "a"}, "base", samples=100)
        changed = [{"number": 1, "expected": ["type/feature"], "proposed": []}]
        with self.assertRaisesRegex(ValueError, "identical reference"):
            compare({"base": records, "bad": changed}, {1: "a"}, "base", samples=100)

    def test_predictions_must_be_complete_unique_and_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            row = {"number": 1, "proposed": ["type/bug"]}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(
                load_predictions(path, [1], {"type/bug"}), {1: ["type/bug"]}
            )
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_predictions(path, [1, 2], {"type/bug"})
            path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique integers"):
                load_predictions(path, [1], {"type/bug"})
            path.write_text(
                json.dumps({"number": 1, "proposed": ["invented/label"]}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical"):
                load_predictions(path, [1], {"type/bug"})

    def test_plan_pins_sources_artifacts_and_reserved_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = {
                "contract": {
                    "classification_labels": ["type/bug", "type/feature"],
                    "legacy_read_aliases": {},
                },
                "issues": [{"number": 1, "labels": ["type/bug"]}],
            }
            (root / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
            (root / "protocol.json").write_text("{}", encoding="utf-8")
            (root / "policy.json").write_text("{}", encoding="utf-8")
            artifact = {
                "path": "policy.json",
                "sha256": file_sha256(root / "policy.json"),
            }
            plan = {
                "snapshot": {
                    "path": "snapshot.json",
                    "sha256": file_sha256(root / "snapshot.json"),
                },
                "protocol": {
                    "path": "protocol.json",
                    "sha256": file_sha256(root / "protocol.json"),
                },
                "partition": "confirmation",
                "dimensions": ["type"],
                "reference": "baseline",
                "bootstrap_samples": 100,
                "candidates": [],
            }
            for name, labels in (
                ("baseline", ["type/bug", "type/feature"]),
                ("better", ["type/bug"]),
            ):
                path = root / f"{name}.jsonl"
                path.write_text(
                    json.dumps({"number": 1, "proposed": labels}) + "\n",
                    encoding="utf-8",
                )
                plan["candidates"].append(
                    {
                        "name": name,
                        "selected_on": "development",
                        "frozen_artifacts": [artifact],
                        "predictions": {"path": path.name, "sha256": file_sha256(path)},
                    }
                )
            path = root / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            protocol = {
                "snapshot_sha256": plan["snapshot"]["sha256"],
                "groups": [{"group_id": "one", "issue_ids": [1]}],
            }
            with (
                patch(
                    "experiments.precision_confirmation.load_protocol",
                    return_value=protocol,
                ),
                patch(
                    "experiments.precision_confirmation.select_ids", return_value=[1]
                ) as membership,
            ):
                result = evaluate_plan(path)
                membership.assert_called_once_with(protocol, "confirmation")
                self.assertEqual(result["issue_numbers"], [1])
                self.assertTrue(result["differences"]["better"]["clear_precision_win"])
                (root / "policy.json").write_text('{"changed": true}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Frozen artifact"):
                    evaluate_plan(path)
                plan["partition"] = "development"
                path.write_text(json.dumps(plan), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "reserved partition"):
                    evaluate_plan(path)


if __name__ == "__main__":
    unittest.main()
