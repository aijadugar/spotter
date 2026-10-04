from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.public_fitness_style_data import (  # noqa: E402
    convert_rows_to_style_corpus,
    is_style_relevant,
    load_chibbss_rows,
    load_haz_rows,
)


class PublicFitnessStyleDataTests(unittest.TestCase):
    def test_is_style_relevant_filters_broad_wellness(self) -> None:
        self.assertTrue(
            is_style_relevant(
                "How can I improve my push-up form?",
                "Use slower reps and keep your body line organized.",
            )
        )
        self.assertFalse(
            is_style_relevant(
                "How do I manage anxiety and insomnia?",
                "Practice breathing and relaxation.",
            )
        )

    def test_loaders_and_conversion_match_real_source_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            haz_path = Path(temp_dir) / "FITNESS.csv"
            haz_path.write_text(
                "Human,Assistant\n"
                "\"how do i improve squat depth\",\"use a slower descent and stay balanced\"\n"
                "\"how do i improve sleep\",\"sleep more\"\n",
                encoding="utf-8",
            )
            chibbss_path = Path(temp_dir) / "fitness.json"
            chibbss_path.write_text(
                json.dumps(
                    [
                        {
                            "instruction": "What push-up cues help beginners?",
                            "output": "Keep a straight line and use repeatable reps.",
                        },
                        {
                            "instruction": "How do I reduce anxiety fast?",
                            "output": "Try meditation.",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            haz_rows = load_haz_rows(haz_path)
            chibbss_rows = load_chibbss_rows(chibbss_path)
            corpus = convert_rows_to_style_corpus(
                haz_rows + chibbss_rows,
                source_dataset="demo/public-style",
            )

        self.assertEqual(len(haz_rows), 2)
        self.assertEqual(len(chibbss_rows), 2)
        self.assertEqual(len(corpus), 2)
        self.assertTrue(all(row["metadata"]["style_only"] for row in corpus))


if __name__ == "__main__":
    unittest.main()
