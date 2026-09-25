"""Hermetic formulation tests; no neural inference, cache access, or downloads."""

import math
import re
import unittest
from copy import deepcopy

from experiment import digest, prepare_state, read_json, ROOT, taxonomy
from experiments import precision_prompts as prompts
from experiments.precision_protocol import load_protocol


class Tokenizer:
    mask_token = "[MASK]"

    def __call__(self, text, **kwargs):
        return {"input_ids": re.findall(r"\n|\S+", text)}

    def decode(self, tokens, **kwargs):
        return " ".join(tokens).replace(" \n ", "\n")


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.labels = taxonomy(read_json(ROOT / "data/snapshot.json"))
        self.config = read_json(ROOT / "config.json")
        self.issue = {"number": 7, "title": "Example", "body": "hello",
                      "labels": ["CONTROL-CANARY"], "state": "closed"}

    def test_concrete_noul_has_no_generic_criteria(self):
        questions = prompts.make_questions(self.labels, "predicate-clean", ["type"])
        self.assertEqual(len(questions), 8)
        self.assertTrue(all(set(q) == {"type", "instructions"} for q in questions.values()))
        self.assertEqual(questions["type/bug"]["instructions"], prompts.PREDICATES["type/bug"])

    def test_membership_arm_preserves_original_questions(self):
        from experiment import questions_for
        expected = questions_for({k: v for k, v in self.labels.items() if k.startswith("type/")})
        self.assertEqual(prompts.make_questions(self.labels, "membership-clean", ["type"]), expected)

    def test_choices_include_none_without_large_option_bucket(self):
        questions = prompts.make_questions(self.labels, "choice-clean", ["theme", "type"])
        self.assertEqual(len(questions["type"]["criteria"]), 9)
        self.assertEqual(len(questions["theme"]["criteria"]), 4)
        self.assertTrue(all("none" in q["criteria"] for q in questions.values()))

    def test_clean_arm_has_exact_same_retained_issue_excerpt(self):
        self.issue["body"] = "word " * 600 + "TAIL"
        baseline, baseline_info = prepare_state(self.issue, Tokenizer(), self.config)
        clean, info = prompts.prepare_input(self.issue, Tokenizer(), self.config, "predicate-clean")
        self.assertEqual(clean, "Title:" + baseline.split("\nTitle:", 1)[1])
        self.assertEqual(info["baseline_state_sha256"], baseline_info["state_sha256"])
        self.assertLess(info["used_state_tokens"], baseline_info["used_state_tokens"])
        self.assertTrue(info["shortened"])
        self.assertTrue(clean.endswith("TAIL"))

    def test_context_arm_is_byte_identical_to_baseline(self):
        baseline, _ = prepare_state(self.issue, Tokenizer(), self.config)
        state, _ = prompts.prepare_input(self.issue, Tokenizer(), self.config, "predicate-context")
        self.assertEqual(state, baseline)

    def test_no_control_labels_or_metadata_enter_state(self):
        for variant in prompts.VARIANTS:
            state, metadata = prompts.prepare_input(self.issue, Tokenizer(), self.config, variant)
            self.assertNotIn("CONTROL-CANARY", state)
            self.assertNotIn("closed", state)
            self.assertEqual(metadata["input_sha256"], digest({"title": "Example", "body": "hello"}))

    def test_noul_threshold_inclusive_and_multiple_labels(self):
        questions = {"type/bug": {"type": "noul"}, "type/feature": {"type": "noul"}}
        response = {"answers": {k: {"type": "noul", "noul": 0.5} for k in questions}}
        proposed, scores = prompts.decode(response, questions)
        self.assertEqual(proposed, ["type/bug", "type/feature"])
        self.assertEqual(set(scores), set(questions))

    def test_choice_argmax_is_not_binary_half_threshold(self):
        questions = {"type": {"type": "choice", "criteria": {"type/bug": "", "type/docs": "", "none": ""}}}
        response = {"answers": {"type": {"type": "choice", "choice": "type/bug", "confidence": 0.4,
                                       "probabilities": {"type/bug": 0.4, "type/docs": 0.3, "none": 0.3}}}}
        self.assertEqual(prompts.decode(response, questions), (["type/bug"], {"type/bug": 0.4, "type/docs": 0.3}))
        response["answers"]["type"].update(choice="none", confidence=0.6,
                                          probabilities={"type/bug": 0.2, "type/docs": 0.2, "none": 0.6})
        self.assertEqual(prompts.decode(response, questions)[0], [])

    def test_invalid_probabilities_fail(self):
        questions = {"type/bug": {"type": "noul"}}
        for value in (True, "0.5", math.nan, math.inf, -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prompts.decode({"answers": {"type/bug": {"type": "noul", "noul": value}}}, questions)

    def test_scope_rejects_area_duplicates_unknown_and_empty(self):
        for scope in ("", "area", "type,type", "whatever", "type,theme,area"):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                prompts.dimensions_scope(scope)

    def test_confirmation_requires_frozen_policy_and_full_corpus_is_blocked(self):
        protocol = {"splits": {}, "subsets": {"screen24": [1], "development64": [1, 2], "confirmation96": [3]}}
        self.assertEqual(prompts.selected_ids(protocol, "screen24", False), [1])
        with self.assertRaises(ValueError):
            prompts.selected_ids(protocol, "confirmation96", False)
        self.assertEqual(prompts.selected_ids(protocol, "confirmation96", True), [3])
        with self.assertRaises(ValueError):
            prompts.selected_ids(protocol, "development", True)
        protocol["subsets"]["screen24"] = list(range(25))
        with self.assertRaises(ValueError):
            prompts.selected_ids(protocol, "screen24", False)

    def test_policy_binds_code_questions_and_config(self):
        protocol = load_protocol(ROOT / "runs/precision/protocol.json")
        policy = prompts.policy_for(protocol, "choice-clean", ["type"])
        prompts.validate_policy(policy, protocol)
        mutated = deepcopy(policy)
        mutated["option_counts"]["type"] = 15
        with self.assertRaises(ValueError):
            prompts.validate_policy(mutated, protocol)
        mutated["policy_sha256"] = digest({k: v for k, v in mutated.items() if k != "policy_sha256"})
        with self.assertRaises(ValueError):
            prompts.validate_policy(mutated, protocol)

    def test_missing_dimensions_unscored_and_coverage_explicit(self):
        records = [{"expected": ["type/bug"], "proposed": ["type/bug", "theme/security"]},
                   {"expected": [], "proposed": ["type/feature"]}]
        result = prompts.metrics(records, ["type", "theme"])
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["control_issues"], 1)
        self.assertEqual(result["by_dimension"]["theme"]["control_issues"], 0)
        self.assertEqual(result["issue_coverage"], 1)
        self.assertEqual(result["dimension_coverage"]["theme"], 0.5)

    def test_resume_rejects_tampered_duplicated_or_foreign_rows(self):
        questions = {"type/bug": {"type": "noul"}}
        policy = {"policy_sha256": "hash", "questions": questions}
        state = "Title: Example\nBody: hello"
        row = {"number": 7, "policy_sha256": "hash", "evidence": "real-local-model-inference",
               "input_sha256": digest({"title": "Example", "body": "hello"}),
               "state": state, "state_sha256": digest(state), "proposed": ["type/bug"],
               "scores": {"type/bug": 0.7},
               "response": {"answers": {"type/bug": {"type": "noul", "noul": 0.7}}},
               "inference_seconds": 1.0, "processing_seconds": 1.1}
        inputs = {7: self.issue}
        prompts.validate_rows([row], policy, inputs)
        for altered in ([row, row], [{**row, "number": 8}], [{**row, "proposed": []}],
                        [{**row, "state": "tampered"}], [{**row, "processing_seconds": -1}]):
            with self.assertRaises(ValueError):
                prompts.validate_rows(altered, policy, inputs)


if __name__ == "__main__":
    unittest.main()
