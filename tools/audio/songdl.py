import argparse
import os
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd

DEFAULT_SAMPLE_CSV = Path("data/samples/train_sample_100_seed2025_20251126-194143.csv")
LEGACY_SAMPLE_CSV = Path("samples/train_sample_100_seed2025_20251126-194143.csv")
DEFAULT_OUTPUT_DIR = Path("data/songs")
LEGACY_OUTPUT_DIR = Path("songs")


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "", name)


def _default_csv_path() -> Path:
    if DEFAULT_SAMPLE_CSV.exists() or not LEGACY_SAMPLE_CSV.exists():
        return DEFAULT_SAMPLE_CSV
    return LEGACY_SAMPLE_CSV


def _default_output_dir() -> Path:
    if DEFAULT_OUTPUT_DIR.exists() or not LEGACY_OUTPUT_DIR.exists():
        return DEFAULT_OUTPUT_DIR
    return LEGACY_OUTPUT_DIR


def build_ydl_options(audio_quality: str = "192") -> dict:
    return {
        "format": "bestaudio/best",
        "quiet": False,
        "noplaylist": True,
        "remote_components": ["ejs:github"],
        "js_runtimes": {"deno": {}},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": audio_quality,
            }
        ],
    }


def write_metadata(path: Path, title: str, artist: str) -> None:
    try:
        from mutagen.easyid3 import EasyID3
        from mutagen.id3 import ID3, error
    except ImportError as exc:
        raise RuntimeError("mutagen is required for tagging mp3 metadata.") from exc

    try:
        audio = EasyID3(path)
    except error:
        audio = ID3()

    audio["title"] = title
    audio["artist"] = artist
    audio.save(path)


def download_from_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    ydl_factory: Optional[Callable[[dict], object]] = None,
    ydl_opts: Optional[dict] = None,
    tag_writer: Optional[Callable[[Path, str, str], None]] = None,
    query_template: str = "{artist} {title} audio",
) -> list[Path]:
    if ydl_opts is None:
        ydl_opts = build_ydl_options()
    if ydl_factory is None:
        from yt_dlp import YoutubeDL

        ydl_factory = YoutubeDL
    if tag_writer is None:
        tag_writer = write_metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for _, row in df.iterrows():
        title = str(row["title"])
        artist = str(row["artists"])
        query = query_template.format(title=title, artist=artist)

        print(f"\n🔍 Searching & downloading: {title} – {artist}")

        with ydl_factory(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch:{query}", download=True)
            entry = info["entries"][0]
            files = ydl.prepare_filename(entry)
            temp_file = Path(os.path.splitext(files)[0] + ".mp3")

        final_name = safe_filename(f"{title}.mp3")
        final_path = output_dir / final_name

        if not temp_file.exists():
            print(f"❌ Missing file: {temp_file}")
            continue

        os.rename(temp_file, final_path)
        print(f"✅ Saved as: {final_path}")

        tag_writer(final_path, title, artist)
        print(f"🎵 Added metadata → Title: {title}, Artist: {artist}")
        downloaded.append(final_path)

    print("\n🎉 Done.")
    return downloaded


def download_from_csv(
    csv_path: Path,
    output_dir: Path,
    *,
    ydl_factory: Optional[Callable[[dict], object]] = None,
    ydl_opts: Optional[dict] = None,
    tag_writer: Optional[Callable[[Path, str, str], None]] = None,
) -> list[Path]:
    df = pd.read_csv(csv_path)
    return download_from_dataframe(
        df,
        output_dir=output_dir,
        ydl_factory=ydl_factory,
        ydl_opts=ydl_opts,
        tag_writer=tag_writer,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download songs from a sampled EDM-CUE CSV.")
    parser.add_argument("--csv", type=Path, default=_default_csv_path(), help="Sample CSV to download.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory to store downloaded mp3s.",
    )
    parser.add_argument(
        "--audio_quality",
        default="192",
        help="Target audio quality for ffmpeg postprocessing (e.g., 192).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=None if argv is None else list(argv))

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    ydl_opts = build_ydl_options(audio_quality=args.audio_quality)
    download_from_csv(args.csv, output_dir=args.output_dir, ydl_opts=ydl_opts)


if __name__ == "__main__":
    main()
