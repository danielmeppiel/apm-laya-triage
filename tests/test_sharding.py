"""Validate parallel aggregation with explicitly invented, offline records."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from evaluation import evaluate
from experiment import digest, read_json, write_json
from sharding import combine, shard_snapshot
from test_evaluation import corpus


def write_shards(root: Path) -> tuple[Path, Path]:
    """Persist two tiny compatible shards of the existing scoring fixture."""
    parent, rows, base = corpus()
    parent_path = root / "snapshot.json"
    write_json(parent_path, parent)
    shards = root / "shards"
    for index in range(2):
        folder = shards / f"shard-{index}"
        snapshot = shard_snapshot(parent, index, 2)
        manifest = {
            **base,
            "fingerprint": f"test-only-{index}",
            "source_snapshot_sha256": digest(snapshot),
            "prediction_count": 2,
            "target_count": 2,
            "started_at": "2026-01-01T00:00:00+00:00",
            "runtime": {
                "device": "cpu",
                "mixed_precision": False,
                "versions": {"laya": "test-only"},
                "weights_sha256": "test-only",
                "cpu_threads": 2,
                "platform": "test-only",
                "model_load_seconds": 1.0,
                "download_and_hash_seconds": 1.0,
            },
            "execution": {"kind": "github-actions", "run_id": "test-only"},
        }
        selected = [
            {**row, "fingerprint": manifest["fingerprint"]} for row in rows[index::2]
        ]
        write_json(folder / "snapshot.json", snapshot)
        write_json(folder / "manifest.json", manifest)
        (folder / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in selected), encoding="utf8"
        )
    return parent_path, shards


class ShardingTests(unittest.TestCase):
    def test_partition_is_complete_disjoint_and_balanced(self) -> None:
        parent, _, _ = corpus()
        shards = [shard_snapshot(parent, index, 3) for index in range(3)]
        numbers = [issue["number"] for shard in shards for issue in shard["issues"]]
        self.assertEqual(sorted(numbers), [1, 2, 3, 4])
        self.assertEqual(len(numbers), len(set(numbers)))
        self.assertEqual([len(item["issues"]) for item in shards], [2, 1, 1])
        self.assertEqual(shards[0]["contract"], parent["contract"])

    def test_invalid_partition_parameters_fail(self) -> None:
        parent, _, _ = corpus()
        for index, count in [(-1, 2), (2, 2), (0, 0), (0, 5)]:
            with self.subTest(index=index, count=count), self.assertRaises(ValueError):
                shard_snapshot(parent, index, count)

    def test_merge_preserves_raw_predictions_and_global_metric_denominators(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent_path, shards = write_shards(root)
            output = root / "combined"
            with redirect_stdout(io.StringIO()):
                combine(parent_path, shards, output)
            manifest = read_json(output / "manifest.json")
            predictions = [
                json.loads(line)
                for line in (output / "predictions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(predictions), 4)
            self.assertEqual(manifest["runtime"]["shard_count"], 2)
            self.assertEqual(manifest["execution"]["kind"], "github-actions")
            self.assertEqual(
                {row["fingerprint"] for row in predictions},
                {"test-only-0", "test-only-1"},
            )
            for index in range(2):
                self.assertEqual(
                    (output / "shards" / str(index) / "predictions.jsonl").read_bytes(),
                    (shards / f"shard-{index}" / "predictions.jsonl").read_bytes(),
                )
            metrics, _ = evaluate(read_json(parent_path), predictions, manifest)
            self.assertEqual(metrics["overall_observed_dimensions"]["micro_f1"], 0.75)

    def test_missing_incomplete_changed_or_incompatible_shards_fail(self) -> None:
        for mutation in (
            "missing",
            "incomplete",
            "changed-parent",
            "mixed-device",
            "wrong-index",
        ):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                parent_path, shards = write_shards(root)
                manifest_path = shards / "shard-1" / "manifest.json"
                manifest = read_json(manifest_path)
                if mutation == "missing":
                    manifest_path.unlink()
                elif mutation == "incomplete":
                    manifest["status"] = "running"
                    write_json(manifest_path, manifest)
                elif mutation == "changed-parent":
                    parent = read_json(parent_path)
                    parent["issues"][0]["body"] = "changed after inference"
                    write_json(parent_path, parent)
                elif mutation == "mixed-device":
                    manifest["runtime"]["device"] = "mps"
                    write_json(manifest_path, manifest)
                else:
                    snapshot_path = shards / "shard-1" / "snapshot.json"
                    snapshot = read_json(snapshot_path)
                    snapshot["shard"]["index"] = 0
                    write_json(snapshot_path, snapshot)
                with self.assertRaises(ValueError):
                    combine(parent_path, shards, root / "combined")
                self.assertFalse((root / "combined" / "manifest.json").exists())

    def test_report_rejects_overlap_between_source_manifests(self) -> None:
        parent, rows, manifest = corpus()
        manifest = copy.deepcopy(manifest)
        manifest["source_runs"] = {
            "a": {"issue_numbers": [1, 2]},
            "b": {"issue_numbers": [2, 3, 4]},
        }
        with self.assertRaises(ValueError):
            evaluate(parent, rows, manifest)


if __name__ == "__main__":
    unittest.main()
