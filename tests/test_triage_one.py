"""Verify the per-issue read boundary without live GitHub requests."""

import unittest
from unittest.mock import patch

from triage_one import main, read_issue


class ReadOneIssueTests(unittest.TestCase):
    def test_cli_honors_explicit_gpu_request(self) -> None:
        with (
            patch("sys.argv", ["triage_one.py", "--issue", "10", "--device", "mps"]),
            patch("triage_one.classify") as classify,
        ):
            main()
        self.assertEqual(classify.call_args.args[-1], "mps")

    def test_only_input_fields_cross_the_boundary(self) -> None:
        response = {
            "number": 10,
            "title": "Example",
            "body": None,
            "html_url": "https://github.com/microsoft/apm/issues/10",
            "labels": [{"name": "type/bug"}],
            "state": "closed",
        }
        with patch("triage_one.gh_get", return_value=response) as get:
            result = read_issue("microsoft/apm", 10)
        get.assert_called_once_with("repos/microsoft/apm/issues/10")
        self.assertEqual(set(result), {"number", "title", "body", "url"})
        self.assertEqual(result["body"], "")

    def test_nonpositive_identifiers_and_pull_requests_fail(self) -> None:
        with patch("triage_one.gh_get") as get, self.assertRaises(ValueError):
            read_issue("microsoft/apm", 0)
        get.assert_not_called()
        with patch("triage_one.gh_get", return_value={"pull_request": {}}):
            with self.assertRaises(ValueError):
                read_issue("microsoft/apm", 10)


if __name__ == "__main__":
    unittest.main()
