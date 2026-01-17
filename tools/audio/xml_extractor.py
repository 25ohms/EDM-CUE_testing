import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.parse import unquote, urlparse

DEFAULT_XML_FILE = Path("jan16.xml")
DEFAULT_CSV_FILE = Path("data/samples/Jan7-98-songs.csv")
LEGACY_CSV_FILE = Path("samples/data/samples/Jan7-98-songs.csv")
DEFAULT_OUTPUT_FILE = Path("data/exports/dataset.json")
LEGACY_OUTPUT_FILE = Path("dataset.json")


def _default_csv_file() -> Path:
    if DEFAULT_CSV_FILE.exists() or not LEGACY_CSV_FILE.exists():
        return DEFAULT_CSV_FILE
    return LEGACY_CSV_FILE


def _default_output_file() -> Path:
    if DEFAULT_OUTPUT_FILE.exists() or not LEGACY_OUTPUT_FILE.exists():
        return DEFAULT_OUTPUT_FILE
    return LEGACY_OUTPUT_FILE


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return str(text).lower().strip()


def build_xml_index(xml_file: Path) -> dict[str, list[ET.Element]]:
    if not xml_file.exists():
        print(f"Error: XML file not found at {xml_file}")
        return {}

    print("Indexing Rekordbox XML...")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError:
        print("Error: Could not parse XML.")
        return {}

    collection = root.find("COLLECTION")
    if collection is None:
        print("Error: XML missing COLLECTION element.")
        return {}

    xml_index: dict[str, list[ET.Element]] = {}
    for track in collection.findall("TRACK"):
        title = clean_text(track.get("Name"))
        xml_index.setdefault(title, []).append(track)

    print(f"Indexed {len(collection.findall('TRACK'))} tracks.")
    return xml_index


def find_exact_match(title_matches: Iterable[ET.Element], target_artist: str) -> ET.Element:
    target_artist = clean_text(target_artist)
    matches = list(title_matches)

    for track in matches:
        xml_artist = clean_text(track.get("Artist"))
        if xml_artist == target_artist or target_artist in xml_artist or xml_artist in target_artist:
            return track

    return matches[0]


def process_database_matches(
    *,
    xml_file: Path,
    csv_file: Path,
    output_file: Path,
) -> Tuple[list[dict], list[str]]:
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
        return [], []

    xml_db = build_xml_index(xml_file)
    if not xml_db:
        return [], []

    processed_data: list[dict] = []
    missing_tracks: list[str] = []

    print(f"Reading CSV: {csv_file}...")

    with csv_file.open(mode="r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            print("Error: CSV has no headers.")
            return [], []
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        if "title" not in reader.fieldnames:
            print("Error: CSV missing 'title' column.")
            return [], []

        for row in reader:
            csv_id = row.get("id", "N/A")
            csv_title = row.get("title", "")
            csv_artist = row.get("artists", "")

            clean_title = clean_text(csv_title)
            potential_matches = xml_db.get(clean_title)

            if not potential_matches:
                missing_tracks.append(f"{csv_title} (Not found in XML)")
                continue

            match = find_exact_match(potential_matches, csv_artist)

            artist = match.get("Artist")
            title = match.get("Name")
            try:
                total_time = float(match.get("TotalTime"))
            except (TypeError, ValueError):
                continue

            raw_location = match.get("Location") or ""
            file_path = unquote(urlparse(raw_location).path)
            if os.name == "nt" and file_path.startswith("/") and ":" in file_path:
                file_path = file_path.lstrip("/")

            cues: list[dict] = []
            for mark in match.findall("POSITION_MARK"):
                label_name = mark.get("Name")
                if label_name:
                    cues.append(
                        {
                            "time": float(mark.get("Start")),
                            "label": label_name,
                        }
                    )

            if not cues:
                print(f"Skipping '{title}': Match found, but no labeled cues.")
                continue

            cues.sort(key=lambda x: x["time"])
            sections: list[dict] = []
            for i, cue in enumerate(cues):
                start_time = cue["time"]
                label = cue["label"]
                end_time = cues[i + 1]["time"] if i < len(cues) - 1 else total_time
                if end_time > start_time + 0.05:
                    sections.append(
                        {
                            "start": round(start_time, 3),
                            "end": round(end_time, 3),
                            "label": label,
                        }
                    )

            processed_data.append(
                {
                    "id": csv_id,
                    "song": f"{artist} - {title}",
                    "file_path": file_path,
                    "duration": total_time,
                    "sections": sections,
                }
            )

    if processed_data:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(processed_data, handle, indent=2)
        print(f"\nSuccess! Exported {len(processed_data)} tracks to '{output_file}'.")

    if missing_tracks:
        print(f"\nWarning: {len(missing_tracks)} songs from CSV were not found in XML.")
        for entry in missing_tracks[:5]:
            print(f" - {entry}")

    return processed_data, missing_tracks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract labeled cue sections from Rekordbox XML.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_FILE, help="Path to rekordbox.xml.")
    parser.add_argument("--csv", type=Path, default=_default_csv_file(), help="Sample CSV used for lookup.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_file(),
        help="JSON output path for extracted sections.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=None if argv is None else list(argv))

    process_database_matches(xml_file=args.xml, csv_file=args.csv, output_file=args.output)


if __name__ == "__main__":
    main()
