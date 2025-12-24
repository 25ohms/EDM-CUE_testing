import tempfile
import unittest
from pathlib import Path

import pandas as pd


class AudioDownloadTests(unittest.TestCase):
    def test_safe_filename_strips_invalid_chars(self) -> None:
        from tools.audio import songdl

        self.assertEqual(songdl.safe_filename('a/b:c*?"<>|'), "abc")

    def test_download_from_dataframe_uses_stub(self) -> None:
        from tools.audio import songdl

        class DummyYDL:
            def __init__(self, opts, temp_dir: Path):
                self.temp_dir = temp_dir
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, query, download=True):
                self.query = query
                return {"entries": [{"id": "dummy"}]}

            def prepare_filename(self, entry):
                raw_path = self.temp_dir / "download.webm"
                mp3_path = raw_path.with_suffix(".mp3")
                mp3_path.write_text("stub audio")
                return str(raw_path)

        df = pd.DataFrame(
            {
                "title": ["Test Track"],
                "artists": ["Test Artist"],
            }
        )

        calls = []

        def tag_writer(path: Path, title: str, artist: str) -> None:
            calls.append((path, title, artist))

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            output_dir = temp_dir / "songs"

            def ydl_factory(opts):
                return DummyYDL(opts, temp_dir=temp_dir)

            downloaded = songdl.download_from_dataframe(
                df,
                output_dir=output_dir,
                ydl_factory=ydl_factory,
                tag_writer=tag_writer,
            )

            self.assertEqual(len(downloaded), 1)
            self.assertTrue(downloaded[0].exists())
            self.assertEqual(calls[0][1:], ("Test Track", "Test Artist"))


if __name__ == "__main__":
    unittest.main()
