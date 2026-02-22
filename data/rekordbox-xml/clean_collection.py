#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Union
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse


def normalize_path(path: Union[Path, str]) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def location_to_path(location: str) -> str:
    if not location:
        return ""
    parsed = urlparse(location)
    file_path = unquote(parsed.path)
    if os.name == "nt" and file_path.startswith("/") and ":" in file_path:
        file_path = file_path.lstrip("/")
    return file_path


def collect_song_paths(songs_dir: Path) -> set[str]:
    if not songs_dir.exists():
        raise FileNotFoundError(f"Songs directory not found: {songs_dir}")

    paths: set[str] = set()
    for path in songs_dir.rglob("*"):
        if path.is_file():
            paths.add(normalize_path(path))
    return paths


def clean_xml(xml_path: Path, songs_dir: Path, output_path: Path) -> None:
    allowed_paths = collect_song_paths(songs_dir)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    collection = root.find("COLLECTION")
    if collection is None:
        raise ValueError("XML missing COLLECTION element")

    kept_track_ids: set[str] = set()
    removed_tracks = 0

    for track in list(collection.findall("TRACK")):
        location = track.get("Location", "")
        file_path = location_to_path(location)
        if not file_path:
            collection.remove(track)
            removed_tracks += 1
            continue

        normalized = normalize_path(file_path)
        if normalized in allowed_paths:
            track_id = track.get("TrackID")
            if track_id:
                kept_track_ids.add(track_id)
        else:
            collection.remove(track)
            removed_tracks += 1

    collection.set("Entries", str(len(collection.findall("TRACK"))))

    removed_playlist_entries = 0
    removed_playlist_nodes = 0
    playlists = root.find("PLAYLISTS")
    if playlists is not None:
        for node in playlists.iter("NODE"):
            for track_ref in list(node.findall("TRACK")):
                key = track_ref.get("Key")
                if not key or key not in kept_track_ids:
                    node.remove(track_ref)
                    removed_playlist_entries += 1
            if node.get("Entries") is not None:
                node.set("Entries", str(len(node.findall("TRACK"))))

        parent_map = {child: parent for parent in playlists.iter() for child in parent}
        for node in reversed(list(playlists.iter("NODE"))):
            if node.get("Name") == "ROOT" and node.get("Type") == "0":
                continue
            if node.find("TRACK") is None and node.find("NODE") is None:
                parent = parent_map.get(node)
                if parent is not None:
                    parent.remove(node)
                    removed_playlist_nodes += 1

        for node in playlists.iter("NODE"):
            if node.get("Entries") is not None:
                node.set("Entries", str(len(node.findall("TRACK"))))

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    kept_tracks = len(collection.findall("TRACK"))
    print(
        "Cleaned XML written to: {path}\n"
        "Kept tracks: {kept}\n"
        "Removed tracks: {removed}\n"
        "Removed playlist entries: {removed_entries}\n"
        "Removed empty playlist nodes: {removed_nodes}".format(
            path=output_path,
            kept=kept_tracks,
            removed=removed_tracks,
            removed_entries=removed_playlist_entries,
            removed_nodes=removed_playlist_nodes,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_xml = script_dir / "test1-dec24-25.xml"
    default_songs = repo_root / "data" / "songs"

    parser = argparse.ArgumentParser(
        description=(
            "Filter Rekordbox XML to only tracks whose Location lives in data/songs."
        )
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=default_xml,
        help="Path to Rekordbox XML file.",
    )
    parser.add_argument(
        "--songs-dir",
        type=Path,
        default=default_songs,
        help="Directory containing song files to keep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output XML path (defaults to <xml>-filtered.xml).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input XML file instead of writing a new file.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    xml_path = args.xml
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    output_path = args.output
    if args.in_place:
        output_path = xml_path
    elif output_path is None:
        output_path = xml_path.with_name(f"{xml_path.stem}-filtered.xml")

    clean_xml(xml_path=xml_path, songs_dir=args.songs_dir, output_path=output_path)


if __name__ == "__main__":
    main()
