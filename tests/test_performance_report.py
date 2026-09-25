"""Guard real-evidence requirements with explicit fabricated unit-test records."""

import copy
import tempfile
import unittest
from pathlib import Path

from experiment import write_json
from performance_report import (
    CONFIGURATIONS,
    profile_summary,
    single_summary,
    validate_profile,
)


def profile() -> dict:
    """A tiny six-case fake timing fixture, never shipped as a model result."""
    return {
        "status": "complete",
        "snapshot_sha256": "fixture-only",
        "precision": "fp32",
        "cpu_threads": 2,
        "questions": {"type/bug": {"type": "noul"}},
        "config": {"binary_threshold": 0.5},
        "runtime": {"versions": {"laya": "fixture"}, "weights_sha256": "fixture"},
        "first_issue_process_seconds": 10,
        "optimization_seconds": 0,
        "records": [
            {
                "number": number,
                "repeat": repeat,
                "state_sha256": f"state-{number}",
                "evidence": "real-local-model-inference",
                "inference_seconds": 2,
                "predicted_labels": ["type/bug"],
                "response": {"answers": {"type/bug": {"type": "noul", "noul": 0.8}}},
            }
            for number in range(6)
            for repeat in range(3)
        ],
    }


class PerformanceReportTests(unittest.TestCase):
    def test_incomplete_duplicate_and_invalid_timings_fail(self) -> None:
        for mutation in ("incomplete", "duplicate", "nan", "fixture", "labels"):
            candidate = profile()
            if mutation == "incomplete":
                candidate["status"] = "running"
            elif mutation == "duplicate":
                candidate["records"][-1] = candidate["records"][0]
            elif mutation == "nan":
                candidate["records"][0]["inference_seconds"] = float("nan")
            elif mutation == "fixture":
                candidate["records"][0]["evidence"] = "fixture"
            else:
                candidate["records"][0]["predicted_labels"] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_profile(candidate)

    def test_matched_profiles_count_unique_issues_not_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for replicate in range(2):
                for name in CONFIGURATIONS:
                    candidate = profile()
                    precision, threads = name.split("-")
                    candidate.update(precision=precision, cpu_threads=int(threads))
                    if precision == "int8":
                        for row in candidate["records"]:
                            if row["number"] == 0:
                                row["predicted_labels"] = []
                                row["response"]["answers"]["type/bug"]["noul"] = 0.4
                    write_json(
                        root / f"single-replicate-{replicate}" / f"{name}.json",
                        candidate,
                    )
            summary = profile_summary(root)
            self.assertEqual(summary["unique_issues"], 6)
            self.assertEqual(
                summary["configurations"]["int8-4"][0]["changed_distinct_issues"], [0]
            )
            path = root / "single-replicate-1" / "int8-4.json"
            candidate["records"][0]["state_sha256"] = "different"
            write_json(path, candidate)
            with self.assertRaises(ValueError):
                profile_summary(root)

    def test_local_process_requires_positive_consistent_timing(self) -> None:
        item = copy.deepcopy(profile()["records"][0])
        item.update(
            status="complete",
            github_writes=False,
            input_sha256="input",
            precision="fp32",
            config={"cpu_threads": 2, "binary_threshold": 0.5},
            runtime={"device": "mps"},
            invocation_seconds=3,
            questions={"type/bug": {"type": "noul"}},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "result.json", item)
            process = {
                "process_wall_seconds": 4,
                "hardware": "test",
                "weights_previously_cached": True,
            }
            write_json(root / "process.json", process)
            self.assertEqual(single_summary(root)["kind"], "local-process")
            process["process_wall_seconds"] = 2
            write_json(root / "process.json", process)
            with self.assertRaises(ValueError):
                single_summary(root)


if __name__ == "__main__":
    unittest.main()
