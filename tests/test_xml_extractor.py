import json
import tempfile
import unittest
from pathlib import Path


class XmlExtractorTests(unittest.TestCase):
    def test_process_database_matches_extracts_sections(self) -> None:
        from tools.audio import xml_extractor

        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<ROOT>
  <COLLECTION>
    <TRACK Name="Song A" Artist="Artist A" TotalTime="180" Location="file:///tmp/Song%20A.mp3">
      <POSITION_MARK Name="Intro" Start="0.0" />
      <POSITION_MARK Name="Drop" Start="60.0" />
    </TRACK>
  </COLLECTION>
</ROOT>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            xml_file = tmp_path / "rekordbox.xml"
            csv_file = tmp_path / "sample.csv"
            output_file = tmp_path / "dataset.json"

            xml_file.write_text(xml_content)
            csv_file.write_text("id,title,artists\n1,Song A,Artist A\n")

            processed, missing = xml_extractor.process_database_matches(
                xml_file=xml_file,
                csv_file=csv_file,
                output_file=output_file,
            )

            self.assertEqual(len(processed), 1)
            self.assertEqual(missing, [])
            self.assertTrue(output_file.exists())

            with output_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload[0]["sections"][0]["label"], "Intro")
            self.assertEqual(payload[0]["sections"][-1]["end"], 180.0)


if __name__ == "__main__":
    unittest.main()
