import unittest
from unittest.mock import patch


class VisualizationTests(unittest.TestCase):
    def _has_visual_deps(self) -> bool:
        try:
            import numpy  # noqa: F401
            import matplotlib  # noqa: F401
            import pandas  # noqa: F401
        except ImportError:
            return False
        return True

    def test_extract_bpm_from_dict_and_str(self) -> None:
        if not self._has_visual_deps():
            self.skipTest("Visualization dependencies not installed.")

        from visualization import extract_bpm

        self.assertEqual(extract_bpm({"bpm": 128}), 128.0)
        self.assertEqual(extract_bpm("{'bpm': 140}"), 140.0)
        self.assertEqual(extract_bpm({"tempo": 110}), 110.0)
        self.assertIsNone(extract_bpm("not a dict"))

    def test_bucket_by_bpm(self) -> None:
        if not self._has_visual_deps():
            self.skipTest("Visualization dependencies not installed.")

        from visualization import bucket_by_bpm

        self.assertEqual(bucket_by_bpm(100), "Chill / Downtempo")
        self.assertEqual(bucket_by_bpm(120), "House / Deep House")
        self.assertEqual(bucket_by_bpm(130), "Electro / Dance")

    def test_load_split_uses_hf_paths(self) -> None:
        if not self._has_visual_deps():
            self.skipTest("Visualization dependencies not installed.")

        import pandas as pd
        from visualization import load_split

        with patch("visualization.pd.read_parquet", return_value=pd.DataFrame()) as mock_read:
            load_split("train")
            mock_read.assert_called_once_with(
                "hf://datasets/disco-eth/edm-cue/data/train-00000-of-00001.parquet"
            )

    def test_plots_run_with_minimal_dataframe(self) -> None:
        if not self._has_visual_deps():
            self.skipTest("Visualization dependencies not installed.")

        import pandas as pd
        from visualization import plot_bpm_distribution, plot_genre_distribution

        df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "title": ["Song A", "Song B"],
                "artists": ["Artist A", "Artist B"],
                "genre": ["house/tech house", "trance"],
                "beat_grid": [{"bpm": 120}, {"bpm": 138}],
            }
        )

        with patch("matplotlib.pyplot.show"):
            plot_bpm_distribution(df=df, split="train", bins=5, debug=False)
            plot_genre_distribution(df=df, split="train", debug=False)


if __name__ == "__main__":
    unittest.main()
