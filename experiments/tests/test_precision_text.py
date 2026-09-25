"""Synthetic safety checks, never experimental model-performance evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evaluation import aggregate, label_metrics
from experiment import digest
from experiments.precision_text import (
    THRESHOLDS,
    TextBaseline,
    TextConfig,
    agreement_metrics,
    choose_dimension_policy,
    choose_policies,
    load_excerpt_inputs,
    policy_predict,
    predict_records,
    read_selected_scores,
    records_for,
    refit_frozen,
    target_arrays,
    text_features,
)


LABELS = ["area/cli", "area/docs", "type/bug", "type/feature", "theme/security"]
TEXTS = [
    {"number": 1, "title": "terminal crashes", "body": "command error terminal"},
    {"number": 2, "title": "guide pages", "body": "guide documentation pages"},
    {"number": 3, "title": "terminal feature", "body": "new command feature"},
    {"number": 4, "title": "guide error", "body": "guide error pages"},
]
EXPECTED = [
    ["area/cli", "type/bug"],
    ["area/docs", "type/feature"],
    ["area/cli"],
    ["area/docs", "type/bug"],
]


class TextBaselineTests(unittest.TestCase):
    def model(self, **options):
        return TextBaseline(LABELS, TextConfig(min_df=1, **options))

    def test_missing_dimension_is_not_a_negative(self):
        targets, masks = target_arrays(EXPECTED, LABELS)
        self.assertEqual(targets[2, 2], 0)
        self.assertFalse(masks[2, 2])
        fitted = self.model().fit(TEXTS, EXPECTED)
        self.assertEqual(fitted.training_support["type/bug"]["observed_issues"], 3)
        self.assertEqual(fitted.training_support["type/bug"]["negative_issues"], 1)
        self.assertEqual(fitted.training_support["theme/security"]["fit_status"], "unobserved")

    def test_vocabulary_statistics_and_parameters_are_fit_only(self):
        fitted = self.model().fit(TEXTS, EXPECTED)
        before = copy.deepcopy(fitted.vectorizer.vocabulary_)
        idf = fitted.vectorizer.idf_.copy()
        coefficients = fitted.estimators[0].coef_.copy()
        fitted.predict([{"title": "confirmationonlytoken", "body": "newtoken"}])
        self.assertEqual(before, fitted.vectorizer.vocabulary_)
        self.assertNotIn("confirmationonlytoken", before)
        np.testing.assert_array_equal(idf, fitted.vectorizer.idf_)
        np.testing.assert_array_equal(coefficients, fitted.estimators[0].coef_)

    def test_metadata_and_issue_numbers_are_not_features(self):
        changed = [
            {**text, "number": 900000 + i, "labels": ["metadata_canary"], "state": "metadata_canary",
             "comments": "metadata_canary", "url": "metadata_canary"}
            for i, text in enumerate(TEXTS)
        ]
        fitted = self.model().fit(TEXTS, EXPECTED)
        np.testing.assert_array_equal(fitted.predict(TEXTS), fitted.predict(changed))
        self.assertNotIn("metadata_canary", fitted.vectorizer.vocabulary_)
        self.assertEqual(
            text_features({"title": "link #123", "body": "https://github.com/a/b/issues/456"}),
            "link  \n ",
        )

    def test_empty_vocabulary_and_constant_labels_are_explicit(self):
        fitted = self.model().fit(
            [{"title": "", "body": ""}, {"title": "123 #444", "body": ""}],
            [["type/bug"], ["type/bug"]],
        )
        self.assertTrue(fitted.empty_vocabulary)
        self.assertEqual(fitted.training_support["type/bug"]["fit_status"], "constant")
        self.assertEqual(fitted.training_support["type/feature"]["fit_status"], "constant")
        np.testing.assert_array_equal(
            fitted.predict([{"title": "unseen", "body": "anything"}]),
            [[0, 0, 1, 0, 0]],
        )

    def test_fit_only_score_scaling_and_alignment_validation(self):
        scores = np.arange(20).reshape(4, 5) / 20
        fitted = self.model(kind="hybrid").fit(TEXTS, EXPECTED, scores)
        mean = fitted.scaler.mean_.copy()
        fitted.predict(TEXTS[:1], np.ones((1, 5)))
        np.testing.assert_array_equal(mean, scores.mean(axis=0))
        np.testing.assert_array_equal(mean, fitted.scaler.mean_)
        for invalid in (np.zeros((4, 4)), np.full((4, 5), np.nan), np.full((4, 5), 2)):
            with self.assertRaises(ValueError):
                self.model(kind="scores").fit(TEXTS, EXPECTED, invalid)

    def test_replay_and_label_order_are_deterministic(self):
        scores = np.arange(20).reshape(4, 5) / 20
        for kind in ("prior", "word", "scores", "hybrid", "nearest"):
            with self.subTest(kind=kind):
                first = self.model(kind=kind).fit(TEXTS, EXPECTED, scores)
                second = self.model(kind=kind).fit(TEXTS, EXPECTED, scores)
                prediction = first.predict(TEXTS, scores)
                np.testing.assert_array_equal(prediction, second.predict(TEXTS, scores))
                reverse = TextBaseline(
                    LABELS[::-1], TextConfig(kind=kind, min_df=1)
                ).fit(TEXTS, EXPECTED, scores[:, ::-1])
                np.testing.assert_allclose(
                    prediction, reverse.predict(TEXTS, scores[:, ::-1])[:, ::-1],
                    atol=1e-7,
                )

    def test_nearest_uses_only_observed_neighbors_and_explicit_zero_fallback(self):
        fitted = self.model(kind="nearest", neighbors=1).fit(TEXTS, EXPECTED)
        output = fitted.predict([{"title": "terminal feature", "body": "new command feature"}])
        self.assertEqual(output[0, 0], 1)
        self.assertEqual(output[0, 4], 0)
        empty = fitted.predict([{"title": "unseen", "body": ""}])
        self.assertAlmostEqual(empty[0, 2], 2 / 3)

    def test_invalid_targets_and_lifecycle_fail_explicitly(self):
        with self.assertRaises(ValueError):
            target_arrays([["unknown/value"]], LABELS)
        with self.assertRaises(ValueError):
            TextBaseline(["type/bug", "type/bug"])
        with self.assertRaises(RuntimeError):
            self.model().predict(TEXTS)
        with self.assertRaises(ValueError):
            self.model().fit(TEXTS, EXPECTED[:-1])
        with self.assertRaises(RuntimeError):
            self.model().fit(TEXTS, EXPECTED).fit(TEXTS, EXPECTED)

    def test_title_marker_ablation_preserves_body_and_semantic_words(self):
        issue = {"title": "[BUG] [Feature] fix ordinary feature bugs", "body": "[BUG] body unchanged"}
        self.assertEqual(
            text_features(issue, 1, strip_title_tags=True),
            "fix ordinary feature bugs\n[BUG] body unchanged",
        )
        self.assertEqual(
            text_features({"title": "[v1] [BUG] normal", "body": ""}, strip_title_tags=True),
            "[v1] [BUG] normal\n",
        )

    def test_scoring_alignment_ignores_missing_dimensions(self):
        records = [
            {"number": 1, "expected": ["type/bug"], "proposed": ["type/bug", "area/docs"]},
            {"number": 2, "expected": ["type/feature", "area/cli"],
             "proposed": ["type/bug", "area/cli"]},
        ]
        actual = agreement_metrics(records, LABELS)
        self.assertEqual(actual["overall_observed_dimensions"], aggregate(records))
        for label in LABELS:
            self.assertEqual(actual["per_label"][label], label_metrics(records, label))
        self.assertEqual(actual["unscored_proposals"], 1)
        self.assertEqual(actual["by_dimension"]["area"]["control_issues"], 1)
        self.assertEqual(actual["by_dimension"]["area"]["precision"], 1)
        self.assertEqual(actual["by_dimension"]["theme"]["prediction_coverage"], None)

    def test_dimension_thresholds_enforce_each_recall_floor(self):
        labels = ["type/a", "type/b", "area/a", "area/b", "theme/a", "theme/b"]
        expected = [["type/a", "area/a", "theme/a"], ["type/b", "area/b", "theme/b"]]
        values = np.array([[.9, .05, .3, .1, .6, .05], [.7, .8, .2, .4, .3, .7]])
        policy, evidence = choose_dimension_policy([1, 2], expected, values, labels)
        self.assertIsNotNone(policy)
        self.assertEqual(len(THRESHOLDS), 37)
        self.assertEqual(THRESHOLDS[0], .05)
        self.assertEqual(THRESHOLDS[-1], .95)
        records = records_for([1, 2], expected, values, labels, policy)
        for dimension in ("type", "area", "theme"):
            self.assertTrue(evidence[dimension]["feasible"])
            self.assertGreaterEqual(aggregate(records, dimension)["recall"], .5)
        missing, trace = choose_dimension_policy([1, 2], [[], []], values, labels)
        self.assertIsNone(missing)
        self.assertFalse(trace["area"]["feasible"])

    def test_pooled_precision_policy_has_required_recall_or_is_absent(self):
        values = np.array([[.8, .2, .9, .1, 0], [.2, .8, .1, .9, 0],
                           [.9, .1, .5, .5, 0], [.2, .8, .8, .2, 0]])
        policies, trace = choose_policies([1, 2, 3, 4], EXPECTED, values, LABELS)
        self.assertEqual(len(trace), 37)
        records = records_for([1, 2, 3, 4], EXPECTED, values, LABELS, policies["precision_first"])
        self.assertGreaterEqual(aggregate(records)["recall"], .5)
        policies, _ = choose_policies([1, 2, 3, 4], EXPECTED, values * 0, LABELS)
        self.assertNotIn("precision_first", policies)

    def test_prediction_records_never_read_expected_labels(self):
        model = self.model().fit(TEXTS, EXPECTED)
        policy = {"kind": "dimension_thresholds", "thresholds": {"type": .5, "area": .4, "theme": .3}}
        clean = predict_records(model, TEXTS, policy)
        dirty = [{**row, "expected": ["unknown/label"], "labels": ["unknown/label"]} for row in TEXTS]
        self.assertEqual(clean, predict_records(model, dirty, policy))
        self.assertEqual(set(clean[0]), {"number", "proposed"})
        with self.assertRaises(ValueError):
            policy_predict({"type/bug": np.nan}, {"kind": "threshold", "threshold": .5})
        with self.assertRaises(ValueError):
            policy_predict({"type/bug": .5}, {"kind": "dimension_thresholds", "thresholds": {}})
        self.assertEqual(policy_predict({"type/bug": 0.0}, {"kind": "top1_per_dimension"}), [])

    def test_cached_score_rows_validate_text_taxonomy_and_explicit_order(self):
        rows = [
            {"number": issue["number"],
             "input_sha256": digest({"title": issue["title"], "body": issue["body"]}),
             "response": {"answers": {label: {"noul": (i + 1) / 10} for label in LABELS}}}
            for i, issue in enumerate(TEXTS)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"

            def write(values):
                path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf8")

            write(rows[::-1])
            scores = read_selected_scores(path, TEXTS[:2], LABELS)
            np.testing.assert_array_equal(scores[:, 0], [.1, .2])
            write(rows + [rows[0]])
            with self.assertRaises(ValueError):
                read_selected_scores(path, TEXTS, LABELS)
            write(rows[:-1])
            with self.assertRaises(ValueError):
                read_selected_scores(path, TEXTS, LABELS)
            changed = copy.deepcopy(rows)
            changed[0]["input_sha256"] = "wrong"
            write(changed)
            with self.assertRaises(ValueError):
                read_selected_scores(path, TEXTS, LABELS)
            changed = copy.deepcopy(rows)
            del changed[0]["response"]["answers"][LABELS[0]]
            write(changed)
            with self.assertRaises(ValueError):
                read_selected_scores(path, TEXTS, LABELS)

    def test_frozen_refit_rejects_manifest_tampering_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text(json.dumps({"schema_version": 1, "manifest_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "content hash"):
                refit_frozen(path, "balanced", Path(directory))

    def test_excerpt_override_is_exact_budget_and_reserved_ids_are_rejected(self):
        protocol = {"splits": {
            "fit": {"control_ids": [1]}, "development": {"control_ids": [2]},
            "confirmation": {"control_ids": [3]},
        }}
        excerpt_rows = []
        baseline = []
        for issue in TEXTS[:2]:
            state = f"Title: {issue['title']}\nBody: retained exact body"
            excerpt_rows.append({
                "number": issue["number"], "state": state, "state_sha256": digest(state),
                "input_sha256": digest({"title": issue["title"], "body": issue["body"]}),
                "baseline_state_sha256": "baseline-" + str(issue["number"]),
                "used_state_tokens": 20, "shortened": True,
            })
            baseline.append({"number": issue["number"],
                             "state_sha256": "baseline-" + str(issue["number"])})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "excerpts.jsonl"
            baseline_path = Path(directory) / "baseline.jsonl"
            baseline_path.write_text("".join(json.dumps(row) + "\n" for row in baseline))
            path.write_text("".join(json.dumps(row) + "\n" for row in excerpt_rows))
            inputs, compact = load_excerpt_inputs(path, protocol, {"issues": TEXTS}, baseline_path)
            self.assertEqual(set(inputs), {1, 2})
            self.assertEqual(inputs[1]["title"], TEXTS[0]["title"])
            self.assertEqual(inputs[1]["body"], "retained exact body")
            self.assertEqual(len(compact), 2)
            for modification in ({"number": 3}, {"state_sha256": "wrong"},
                                 {"baseline_state_sha256": "wrong"}, {"used_state_tokens": 301}):
                changed = copy.deepcopy(excerpt_rows)
                changed[0].update(modification)
                path.write_text("".join(json.dumps(row) + "\n" for row in changed))
                with self.assertRaises(ValueError):
                    load_excerpt_inputs(path, protocol, {"issues": TEXTS}, baseline_path)


if __name__ == "__main__":
    unittest.main()
