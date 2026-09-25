"""The authorized checkpoint must not change prompts, excerpts, or base pins."""

import unittest
from copy import deepcopy

from experiment import ROOT, read_json
from experiments.precision_protocol import load_protocol
from experiments.precision_prompts import policy_for, validate_policy
from experiments.precision_typed_checkpoint import CHECKPOINT, WEIGHT_BYTES, checkpoint_policy


class TypedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(ROOT / "runs/precision/protocol.json")
        self.base = read_json(ROOT / "runs/precision/prompts/frozen-choice-type-theme.json")

    def test_preserves_base_and_exact_questions(self):
        original = deepcopy(self.base)
        typed = checkpoint_policy(self.base, self.protocol)
        self.assertEqual(self.base, original)
        self.assertEqual(typed["questions"], self.base["questions"])
        self.assertEqual(typed["questions_sha256"], self.base["questions_sha256"])
        self.assertEqual(typed["excerpt_policy"], self.base["excerpt_policy"])
        validate_policy(self.base, self.protocol)

    def test_changes_only_authorized_model_config(self):
        typed = checkpoint_policy(self.base, self.protocol)
        changes = {k: v for k, v in typed["config"].items() if v != self.base["config"][k]}
        self.assertEqual(changes, CHECKPOINT)
        self.assertEqual(WEIGHT_BYTES, 842609220)
        self.assertFalse(typed["config"]["mixed_precision"])
        self.assertEqual(typed["config"]["device"], "mps")

    def test_records_both_runner_hashes_and_distinct_identity(self):
        typed = checkpoint_policy(self.base, self.protocol)
        self.assertIn("experiments/precision_prompts.py", typed["source_hashes"])
        self.assertIn("experiments/precision_typed_checkpoint.py", typed["source_hashes"])
        self.assertEqual(typed["base_policy_sha256"], self.base["policy_sha256"])
        self.assertNotEqual(typed["policy_sha256"], self.base["policy_sha256"])

    def test_rejects_unapproved_scope(self):
        with self.assertRaises(ValueError):
            checkpoint_policy(policy_for(self.protocol, "choice-clean", ["type"]), self.protocol)


if __name__ == "__main__":
    unittest.main()
