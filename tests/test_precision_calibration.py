"""Synthetic calibration and replay tests; no confirmation outcomes are read."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments import precision_calibration
from experiment import digest
from experiments.precision_calibration import (
    best_threshold,
    calibrated_probability,
    evaluate_policy,
    fit_candidates,
    fit_isotonic,
    load_frozen,
    load_scores,
    load_training_data,
    metrics,
    predict,
    select_finalists,
    threshold_policy,
)
from experiments.precision_protocol import file_sha256

LABELS = ["area/cli", "theme/security", "type/bug", "type/feature"]


def records():
    return [
        {"number": number, "expected": ["type/bug", "area/cli"] if number % 2 else ["type/feature"],
         "scores": {"area/cli": 0.8, "theme/security": 0.3,
                    "type/bug": 0.9 if number % 2 else 0.2,
                    "type/feature": 0.2 if number % 2 else 0.9}}
        for number in range(1, 21)
    ]


class PrecisionCalibrationTests(unittest.TestCase):
    def test_missing_dimensions_are_not_fit_negatives(self):
        fit = records()
        policy = threshold_policy(fit, LABELS, 1, "label")
        self.assertEqual(policy["thresholds"]["area/cli"], 0.8)
        self.assertIsNone(policy["thresholds"]["theme/security"])
        result = evaluate_policy(fit, policy)
        self.assertEqual(result["overall"]["precision"], 1)
        self.assertEqual(result["overall"]["recall"], 1)
        self.assertEqual(result["by_dimension"]["area"]["control_issues"], 10)
        self.assertEqual(result["by_dimension"]["theme"]["control_issues"], 0)
        self.assertIsNone(result["by_dimension"]["theme"]["precision"])

    def test_precision_does_not_reward_total_abstention(self):
        data = [{**row, "proposed": []} for row in records()]
        result = metrics(data, LABELS)
        self.assertIsNone(result["overall"]["precision"])
        self.assertEqual(result["overall"]["recall"], 0)
        self.assertEqual(result["coverage"]["whole_issue"], 0)
        self.assertEqual(result["coverage"]["observed_dimension"], 0)
        for record in data:
            record["proposed"] = ["theme/security"]
        result = metrics(data, LABELS)
        self.assertEqual(result["coverage"]["whole_issue"], 1)
        self.assertEqual(result["coverage"]["observed_dimension"], 0)
        self.assertEqual(result["coverage"]["observed_issue"], 0)
        self.assertIsNone(result["overall"]["precision"])

    def test_ties_thresholds_and_rare_classes(self):
        self.assertEqual(best_threshold([(0.9, True), (0.9, False), (0.2, False)], 1), 0.9)
        self.assertIsNone(best_threshold([(0.9, False)], 1))
        fit = records()
        fit[0]["expected"].append("theme/security")
        policy = threshold_policy(fit, LABELS, 1, "label")
        self.assertIsNone(policy["thresholds"]["theme/security"])
        self.assertEqual(policy["fit_support"]["theme/security"], 1)

    def test_isotonic_is_monotone_and_replays_boundaries(self):
        blocks = fit_isotonic([(0.1, False), (0.2, True), (0.3, False), (0.9, True)])
        probabilities = [calibrated_probability(value / 100, blocks) for value in range(101)]
        self.assertEqual(probabilities, sorted(probabilities))
        self.assertEqual(calibrated_probability(0.25, blocks), 0.5)
        self.assertEqual(calibrated_probability(1, blocks), 1)
        with self.assertRaises(ValueError):
            calibrated_probability(0.5, [])
        policy = fit_candidates(records(), LABELS, 2)["isotonic_floor0_cap1"]
        self.assertEqual(predict(records()[0]["scores"], policy), ["area/cli", "type/bug"])

    def test_selection_uses_development_and_never_refits(self):
        policies = fit_candidates(records(), LABELS, 1)
        before = copy.deepcopy(policies)
        results = {name: {"development": evaluate_policy(records(), policy)}
                   for name, policy in policies.items()}
        selected = select_finalists(results)
        self.assertNotEqual(selected["balanced"], "original_050")
        self.assertEqual(policies, before)
        for value in results.values():
            value["fit"] = {"overall": {"micro_f1": -99}}
        self.assertEqual(selected, select_finalists(results))
        for value in results.values():
            value["development"]["overall"]["recall"] = 0.49
        self.assertIsNone(select_finalists(results)["precision_first_recall050"])

    def test_variable_cardinality_does_not_invent_unsupported_dimensions(self):
        policies = fit_candidates(records(), LABELS, 3)
        self.assertEqual(policies["isotonic_fit_mean_nearest"]["top_k"], {"type": 1, "area": 1, "theme": 0})
        policy = policies["isotonic_top1_plus_second0.4"]
        predicted = predict(records()[0]["scores"], policy)
        self.assertEqual(predicted, ["area/cli", "type/bug"])
        self.assertNotIn("theme/security", predicted)
        policy["calibrators"]["type/feature"] = [{"lower": 0, "upper": 1, "probability": 0.6}]
        self.assertEqual(predict(records()[0]["scores"], policy), ["area/cli", "type/bug", "type/feature"])

    def test_confirmation_cannot_be_loaded_for_training(self):
        with self.assertRaisesRegex(ValueError, "cannot access confirmation"):
            load_training_data({}, Path("does-not-exist"), {}, "confirmation")

    def test_saved_score_projection_ignores_unrequested_rows(self):
        issue = {"title": "Test", "body": "body"}
        selected = {"number": 1, "input_sha256": digest(issue),
                    "evidence": "real-local-model-inference",
                    "response": {"answers": {label: {"noul": 0.5} for label in LABELS}}}
        excluded = {"number": 999, "response": "not inspected, deliberately invalid"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text(json.dumps(excluded) + "\n" + json.dumps(selected) + "\n", encoding="utf8")
            result = load_scores(path, [1], LABELS, {1: issue})
            self.assertEqual(result, {1: {label: 0.5 for label in LABELS}})
            with self.assertRaises(ValueError):
                load_scores(path, [1, 2], LABELS, {1: issue})

    def test_replay_survives_json_and_rejects_corruption(self):
        policy = threshold_policy(records(), LABELS, 1, "label")
        frozen = {"policy": policy, "protocol_sha256": "protocol",
                  "snapshot_sha256": "snapshot", "baseline_sha256": "baseline",
                  "module_sha256": file_sha256(Path(precision_calibration.__file__))}
        frozen["artifact_sha256"] = digest(frozen)
        protocol = {key: frozen[key] for key in ("protocol_sha256", "snapshot_sha256", "baseline_sha256")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(frozen), encoding="utf8")
            loaded = load_frozen(path, protocol)
            for row in records():
                self.assertEqual(predict(row["scores"], policy), predict(row["scores"], loaded["policy"]))
            frozen["policy"]["thresholds"]["type/bug"] = 0.1
            path.write_text(json.dumps(frozen), encoding="utf8")
            with self.assertRaises(ValueError):
                load_frozen(path, protocol)
        with self.assertRaises(ValueError):
            predict({"expected": 1.0}, policy)


if __name__ == "__main__":
    unittest.main()
