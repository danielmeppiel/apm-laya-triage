"""Hermetic pipeline tests, separate from actual model inference evidence."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import experiment


class Tokenizer:
    """A word-level stand-in for testing length accounting without model downloads."""

    mask_token = "[MASK]"

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[str]]:
        return {"input_ids": text.split()}

    def decode(self, tokens: list[str], **kwargs: object) -> str:
        return " ".join(tokens)


class PipelineTests(unittest.TestCase):
    def test_cli_routes_precision_to_the_canonical_loader_config(self) -> None:
        with (
            patch(
                "sys.argv",
                ["experiment.py", "run", "--device", "cpu", "--precision", "int8"],
            ),
            patch("experiment.run") as runner,
        ):
            experiment.main()
        config = runner.call_args.args[0]
        self.assertEqual(config["cpu_precision"], "int8")
        self.assertEqual(config["device"], "cpu")

    def test_every_classification_is_an_independent_membership_question(self) -> None:
        questions = experiment.questions_for(
            {"type/bug": "A bug", "type/feature": "A feature", "area/cli": "CLI"}
        )
        self.assertEqual(set(questions), {"type/bug", "type/feature", "area/cli"})
        self.assertTrue(
            all(question["type"] == "noul" for question in questions.values())
        )

    def test_threshold_is_inclusive_and_supports_multiple_types(self) -> None:
        response = {
            "answers": {
                "type/bug": {"type": "noul", "noul": 0.5},
                "type/feature": {"type": "noul", "noul": 0.7},
                "area/cli": {"type": "noul", "noul": 0.4999},
            }
        }
        self.assertEqual(
            experiment.predicted_labels(response, 0.5), ["type/bug", "type/feature"]
        )

    def test_invalid_gpu_numbers_fail_closed(self) -> None:
        questions = experiment.questions_for({"type/bug": "A bug"})
        for probability in (math.nan, math.inf, -0.1, 1.1, True, "0.5"):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                experiment.check_response(
                    {"answers": {"type/bug": {"type": "noul", "noul": probability}}},
                    questions,
                )

    def test_no_control_labels_in_model_input(self) -> None:
        issue = {
            "title": "Example",
            "body": "Some content",
            "labels": ["never-pass-this"],
        }
        state, metadata = experiment.prepare_state(
            issue, Tokenizer(), {"project_context": "APM", "state_token_budget": 100}
        )
        self.assertNotIn("never-pass-this", state)
        self.assertFalse(metadata["shortened"])
        self.assertEqual(
            metadata["input_sha256"],
            experiment.digest({"title": issue["title"], "body": issue["body"]}),
        )

    def test_shortening_preserves_front_tail_and_records_loss(self) -> None:
        state, metadata = experiment.prepare_state(
            {"title": "Start", "body": "word " * 200 + "TAIL"},
            Tokenizer(),
            {"project_context": "APM", "state_token_budget": 64},
        )
        self.assertTrue(metadata["shortened"])
        self.assertLessEqual(metadata["used_state_tokens"], 64)
        self.assertIn("Title: Start", state)
        self.assertTrue(state.endswith("TAIL"))

    def test_github_boundary_is_get_only(self) -> None:
        with patch(
            "experiment.subprocess.run", return_value=SimpleNamespace(stdout="{}")
        ) as runner:
            experiment.gh_get("repos/microsoft/apm")
        command = runner.call_args.args[0]
        self.assertEqual(
            command[:7],
            [
                "gh",
                "api",
                "--hostname",
                "github.com",
                "--method",
                "GET",
                "repos/microsoft/apm",
            ],
        )

    def test_taxonomy_rejects_human_decision_labels(self) -> None:
        with self.assertRaises(ValueError):
            experiment.taxonomy(
                {
                    "labels": [{"name": "status/accepted", "description": "Approval"}],
                    "contract": {"classification_labels": ["status/accepted"]},
                }
            )

    def test_atomic_json_writer_persists_valid_complete_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "test.json"
            experiment.write_json(path, {"text": "caf\u00e9"})
            self.assertEqual(json.loads(path.read_text()), {"text": "caf\u00e9"})
            self.assertTrue(path.read_bytes().isascii())
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
