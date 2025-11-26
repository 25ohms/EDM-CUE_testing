import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from visualization import plot_bpm_distribution, plot_genre_distribution

DEFAULT_DIR = Path("samples")


def _resolve_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    candidate = DEFAULT_DIR / path_str
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not locate '{path_str}'. Checked '{path.resolve()}' and '{candidate.resolve()}'."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View BPM/genre visualizations for an existing sample CSV.")
    parser.add_argument("--file", required=True, help="Path to the sample CSV (relative paths checked under samples/).")
    parser.add_argument("--split", default="sample", help="Label to display in the visualization titles.")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins for the BPM histogram.")
    parser.add_argument("--show_bpm", action="store_true", help="Display the BPM distribution histogram.")
    parser.add_argument("--show_genre", action="store_true", help="Display the genre distribution chart.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed diagnostics (e.g., tracks missing BPM or taxonomy mappings).",
    )
    return parser


def _print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    print(f"Loaded {total} sampled tracks.")
    if "bpm_bucket" in df.columns:
        counts = df["bpm_bucket"].value_counts()
        print("\nBucket distribution:")
        for bucket, count in counts.items():
            pct = (count / total * 100) if total else 0.0
            print(f"  - {bucket:<22} {count:>4} tracks ({pct:>5.1f}%)")
    else:
        print("No 'bpm_bucket' column present; skipping bucket summary.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=None if argv is None else list(argv))

    if not args.show_bpm and not args.show_genre:
        parser.error("Please specify --show_bpm, --show_genre, or both.")

    csv_path = _resolve_file(args.file)
    df = pd.read_csv(csv_path)
    _print_summary(df)

    if args.show_bpm:
        plot_bpm_distribution(df=df, split=args.split, bins=args.bins, debug=args.debug)
    if args.show_genre:
        plot_genre_distribution(df=df, split=args.split, debug=args.debug)


if __name__ == "__main__":
    main()
