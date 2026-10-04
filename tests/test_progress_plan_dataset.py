from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class DatasetBuilderTests(unittest.TestCase):
    def test_builder_writes_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts" / "build_progress_plan_sft_dataset.py"),
                    "--out-dir",
                    str(out),
                    "--train-count",
                    "5",
                    "--eval-count",
                    "2",
                    "--seed",
                    "7",
                ],
                check=True,
                cwd=REPO,
            )
            train_rows = [
                json.loads(line)
                for line in (out / "progress_plan_train.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(train_rows), 5)
            self.assertIn("prompt", train_rows[0])
            self.assertIn("completion", train_rows[0])
            json.loads(train_rows[0]["completion"])


if __name__ == "__main__":
    unittest.main()
