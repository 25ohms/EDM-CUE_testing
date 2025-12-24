import subprocess
import unittest


class FixTracksTests(unittest.TestCase):
    def test_fix_tracks_usage_without_args(self) -> None:
        result = subprocess.run(
            ["bash", "fix_tracks.sh"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
