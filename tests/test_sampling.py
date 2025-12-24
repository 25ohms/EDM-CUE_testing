import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class SamplingTests(unittest.TestCase):
    def test_stratified_sample_spreads_across_buckets(self) -> None:
        from tools.sampling import sample_songs

        df = pd.DataFrame(
            {
                "id": ["1", "2", "3", "4"],
                "title": ["A", "B", "C", "D"],
                "artists": ["AA", "BB", "CC", "DD"],
                "genre": ["chill", "house", "electro", "dubstep"],
                "beat_grid": [
                    {"bpm": 100},
                    {"bpm": 120},
                    {"bpm": 130},
                    {"bpm": 150},
                ],
            }
        )

        sample_df = sample_songs.stratified_bpm_sample(df, sample_size=4, seed=7)
        self.assertEqual(len(sample_df), 4)
        self.assertIn("bpm_bucket", sample_df.columns)
        self.assertEqual(sample_df["bpm_bucket"].nunique(), 4)

    def test_write_sample_respects_output_name(self) -> None:
        from tools.sampling import sample_songs

        df = pd.DataFrame(
            {
                "id": ["1"],
                "title": ["A"],
                "artists": ["AA"],
                "genre": ["chill"],
                "beat_grid": [{"bpm": 100}],
                "bpm": [100],
                "bpm_bucket": ["Chill / Downtempo"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_dir = Path(tmpdir) / "samples"
            with patch.object(sample_songs, "SAMPLES_DIR", samples_dir):
                output_path = sample_songs._write_sample(
                    df=df,
                    split="train",
                    seed=42,
                    output_name="custom.csv",
                )
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.name, "custom.csv")

    def test_view_sample_resolves_default_dir(self) -> None:
        from tools.sampling import view_sample

        with tempfile.TemporaryDirectory() as tmpdir:
            default_dir = Path(tmpdir) / "samples"
            default_dir.mkdir()
            sample_file = default_dir / "sample.csv"
            sample_file.write_text("id,title\n1,Song")

            with patch.object(view_sample, "DEFAULT_DIR", default_dir):
                resolved = view_sample._resolve_file("sample.csv")
            self.assertEqual(resolved, sample_file)


if __name__ == "__main__":
    unittest.main()
