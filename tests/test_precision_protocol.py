"""Text-only split integrity, using synthetic issues only."""

import copy
import tempfile
import unittest
from pathlib import Path

from experiments.precision_protocol import (
    build_protocol,
    load_protocol,
    persist_immutable,
    prediction_inputs,
    select_ids,
    text_groups,
)


def snapshot():
    return {
        "labels": [{"name": "type/bug", "description": "bug"}],
        "contract": {"classification_labels": ["type/bug"], "legacy_read_aliases": {}},
        "issues": [
            {"number": number, "title": f"Unique issue {number}", "body": "",
             "labels": ["type/bug"] if number % 7 else []}
            for number in range(1, 600)
        ],
    }


class PrecisionProtocolTests(unittest.TestCase):
    def test_disjoint_complete_reproducible_nested_partitions(self):
        data = snapshot()
        first = build_protocol(data, "snapshot", "baseline")
        data["issues"].reverse()
        self.assertEqual(first, build_protocol(data, "snapshot", "baseline"))
        parts = [set(part["all_ids"]) for part in first["splits"].values()]
        self.assertEqual(len(set.union(*parts)), 599)
        self.assertEqual(sum(map(len, parts)), 599)
        self.assertEqual(select_ids(first, "screen24"), select_ids(first, "development64")[:24])
        self.assertEqual(select_ids(first, "development64"), select_ids(first, "development")[:64])
        self.assertEqual(select_ids(first, "confirmation96"), select_ids(first, "confirmation")[:96])
        for part in first["splits"].values():
            self.assertFalse(set(part["control_ids"]) & set(part["unlabelled_ids"]))
            self.assertEqual(set(part["all_ids"]), set(part["control_ids"]) | set(part["unlabelled_ids"]))

    def test_labels_never_change_group_assignment(self):
        data = snapshot()
        first = build_protocol(data, "snapshot", "baseline")
        for issue in data["issues"]:
            issue["labels"] = []
            issue["scores"] = {"arbitrary": 1}
        second = build_protocol(data, "snapshot", "baseline")
        self.assertEqual(first["groups"], second["groups"])
        for name in first["splits"]:
            self.assertEqual(first["splits"][name]["all_ids"], second["splits"][name]["all_ids"])

    def test_exact_and_near_duplicates_stay_together(self):
        body = " ".join(f"word{number}" for number in range(80))
        issues = [
            {"number": 1, "title": "A specific shared normalized title", "body": body},
            {"number": 2, "title": "A SPECIFIC shared normalized title!", "body": ""},
            {"number": 3, "title": "Different title", "body": body},
            {"number": 4, "title": "Yet another title", "body": body + " new extra words"},
            {"number": 5, "title": "A short unique issue", "body": ""},
        ]
        self.assertEqual(text_groups(issues), [[1, 2, 3, 4], [5]])
        data = snapshot()
        data["issues"] = [{**issue, "labels": ["type/bug"]} for issue in issues]
        protocol = build_protocol(data, "snapshot", "baseline")
        self.assertEqual(protocol["groups"][0]["issue_ids"], [1, 2, 3, 4])
        owner = protocol["groups"][0]["split"]
        self.assertTrue({1, 2, 3, 4} <= set(select_ids(protocol, owner)))

    def test_prediction_schema_and_immutable_file(self):
        data = snapshot()
        self.assertEqual(set(prediction_inputs(data["issues"])[0]), {"number", "title", "body"})
        protocol = build_protocol(data, "snapshot", "baseline")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            persist_immutable(path, protocol)
            persist_immutable(path, protocol)
            self.assertEqual(load_protocol(path), protocol)
            modified = copy.deepcopy(protocol)
            modified["seed"] = "changed"
            with self.assertRaises(ValueError):
                persist_immutable(path, modified)
            path.write_text('{"protocol_sha256": "invalid"}', encoding="utf8")
            with self.assertRaises(ValueError):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()
