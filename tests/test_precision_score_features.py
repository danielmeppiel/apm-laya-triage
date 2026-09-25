"""Portable score-model replay plus optional pinned-sklearn fit parity."""

import importlib.util
import unittest

from experiments.precision_score_features import (
    CONFIGURATIONS,
    exported_scores,
    feature_labels,
    predict,
    predict_probabilities,
    train_model,
)

LABELS = ["area/cli", "theme/security", "type/bug", "type/feature"]


def toy_records():
    return [
        {
            "number": index,
            "scores": {
                "area/cli": ((index * 13) % 97) / 100,
                "theme/security": ((index * 7) % 89) / 100,
                "type/bug": 0.75 + index / 500 if index % 2 else 0.1 + index / 500,
                "type/feature": 0.15 + index / 500 if index % 2 else 0.65 + index / 500,
            },
            "expected": (["type/bug"] if index % 2 else ["type/feature"])
            + (["area/cli"] if index % 3 else []),
        }
        for index in range(1, 61)
    ]


class PortableScoreFeaturesTests(unittest.TestCase):
    def test_projection_and_rejected_target_metadata(self):
        self.assertEqual(feature_labels(LABELS, "type"), ["type/bug", "type/feature"])
        with self.assertRaises(ValueError):
            feature_labels(LABELS, "unknown")
        model = {
            "kind": "logistic", "feature_labels": ["type/bug"], "labels": ["type/bug"],
            "means": [0.5], "scales": [0.25], "score_weight": 0.25,
            "estimators": [{"coefficients": [4.0], "intercept": 0.0}],
        }
        self.assertEqual(predict_probabilities({"type/bug": 0.5}, model), {"type/bug": 0.5})
        self.assertEqual(predict({"type/bug": 0.5}, {"model": model, "policy": {"threshold": 0.5}}), ["type/bug"])
        with self.assertRaises(ValueError):
            predict_probabilities({"type/bug": 0.5, "expected": 1.0}, model)

    def test_tree_float32_branching_matches_sklearn_contract(self):
        model = {
            "kind": "extra_trees", "feature_labels": ["type/bug"], "labels": ["type/bug", "theme/security"],
            "estimators": [
                {"trees": [[[0, 0.500000005, 1, 2, 0.5], [-2, -2, -1, -1, 0.2], [-2, -2, -1, -1, 0.8]]]},
                {"constant": 0.0},
            ],
        }
        self.assertEqual(predict_probabilities({"type/bug": 0.50000001}, model),
                         {"type/bug": 0.2, "theme/security": 0.0})


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Run with the pinned text environment for model-fit parity")
class FittedScoreFeaturesTests(unittest.TestCase):
    def test_all_three_model_exports_match_native_scores(self):
        import numpy as np

        rows = toy_records()
        for name, config in CONFIGURATIONS.items():
            with self.subTest(name=name):
                model, native_predict = train_model(rows, LABELS, config)
                exported = np.asarray([[scores[label] for label in LABELS] for scores in exported_scores(rows, model)])
                np.testing.assert_allclose(exported, native_predict(rows), rtol=0, atol=1e-12)
                self.assertTrue((exported[:, LABELS.index("theme/security")] == 0).all())
                self.assertEqual(model["training_support"]["area/cli"]["observed_issues"], 40)
                self.assertEqual(model["training_support"]["theme/security"]["observed_issues"], 0)
                if config["feature_scope"] == "type":
                    modified = [{**row, "scores": {**row["scores"], "area/cli": 1, "theme/security": 0},
                                 "title": "Invisible metadata", "expected": []} for row in rows]
                    self.assertEqual(exported_scores(rows, model), exported_scores(modified, model))


if __name__ == "__main__":
    unittest.main()
