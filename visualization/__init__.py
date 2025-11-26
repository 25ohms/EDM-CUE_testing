import argparse
import ast
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BPM_BUCKETS,
    GENRE_DELIMITER_PATTERN,
    GENRE_FONT_STACK,
    GENRE_SCORING,
    GENRE_TAXONOMY,
)


SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "valid": "data/valid-00000-of-00001.parquet",
}

_GENRE_FONT_CONFIG = {
    "font.family": "sans-serif",
    "font.sans-serif": GENRE_FONT_STACK,
}
_GENRE_SPLIT_PATTERN = re.compile(GENRE_DELIMITER_PATTERN)

__all__ = [
    "SPLITS",
    "load_split",
    "plot_bpm_distribution",
    "plot_genre_distribution",
    "extract_bpm",
    "bucket_by_bpm",
    "build_arg_parser",
    "main",
]

def _coerce_to_dict(value: Any) -> Optional[dict]:
    """Parse beat_grid values that may already be dicts or stored as strings."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_bpm(entry: Any) -> Optional[float]:
    grid = _coerce_to_dict(entry)
    if not grid:
        return None
    bpm = grid.get("bpm")
    # Fall back to the first value if the key is missing for any reason.
    if bpm is None and grid:
        first_key = next(iter(grid))
        bpm = grid.get(first_key)
    try:
        return float(bpm)
    except (TypeError, ValueError):
        return None


def _flatten_genre_source(value: Any) -> List[Any]:
    """Turn arbitrary nested containers / numpy arrays into a plain Python list."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
            return _flatten_genre_source(parsed)
        return [value]
    if isinstance(value, (list, tuple, set)):
        flattened: List[Any] = []
        for item in value:
            flattened.extend(_flatten_genre_source(item))
        return flattened
    if hasattr(value, "tolist"):
        return _flatten_genre_source(value.tolist())
    return [value]


def _parse_genres(value: Any) -> List[str]:
    """Normalize genre values to individual labels, splitting on slashes/backslashes."""
    tokens: List[str] = []
    for item in _flatten_genre_source(value):
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        for part in _GENRE_SPLIT_PATTERN.split(text):
            normalized = part.strip().strip('"\'')
            if normalized:
                tokens.append(normalized)
    return tokens


def _find_taxonomy_label(normalized_tokens: List[str], bpm: Optional[float]) -> Optional[str]:
    if not normalized_tokens:
        return None

    keyword_score = GENRE_SCORING.get("keyword_match", 10)
    bpm_bonus = GENRE_SCORING.get("bpm_match_bonus", 5)
    default_priority = GENRE_SCORING.get("default_priority", 1000)

    best_key = None
    best_label = None

    for idx, entry in enumerate(GENRE_TAXONOMY):
        keywords = entry.get("keywords", ())
        matches_keyword = any(
            keyword in token for token in normalized_tokens for keyword in keywords
        )
        if not matches_keyword:
            continue

        bpm_range = entry.get("bpm_range")
        bpm_matches = _bpm_in_range(bpm, bpm_range)
        # Enforce BPM alignment when data is available and the entry defines a range.
        if bpm is not None and bpm_range and not bpm_matches:
            continue

        score = float(keyword_score)
        if bpm_matches:
            score += float(bpm_bonus)

        priority = float(entry.get("priority", default_priority)) * -1  # negative so lower priority ranks higher
        candidate_key = (score, priority, -idx)

        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_label = entry["label"]

    return best_label


def _normalize_token(token: str) -> str:
    token = token.lower()
    token = token.replace("&", "and")
    token = re.sub(r"[^a-z0-9\s]", " ", token)
    return re.sub(r"\s+", " ", token).strip()


def _bpm_in_range(bpm: Optional[float], bpm_range: Optional[tuple]) -> bool:
    if bpm is None or not bpm_range:
        return True
    lower, upper = bpm_range
    if lower is not None and bpm < lower:
        return False
    if upper is not None and bpm > upper:
        return False
    return True


def _bucket_by_bpm(bpm: float) -> str:
    for label, lower, upper in BPM_BUCKETS:
        lower_ok = lower is None or bpm >= lower
        upper_ok = upper is None or bpm < upper
        if lower_ok and upper_ok:
            return label
    return "Other / Experimental"


def _boxplot_bounds(values: np.ndarray) -> Tuple[float, float]:
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    if iqr == 0:
        # Provide a minimal spread so the whiskers still make sense.
        margin = max(1.0, q3 * 0.01 or 1.0)
        return q1 - margin, q3 + margin
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def extract_bpm(entry: Any) -> Optional[float]:
    """Public wrapper for BPM extraction."""
    return _extract_bpm(entry)


def bucket_by_bpm(bpm: float) -> str:
    """Public wrapper for consistent BPM bucketing."""
    return _bucket_by_bpm(bpm)


def load_split(split: str) -> pd.DataFrame:
    if split not in SPLITS:
        available = ", ".join(sorted(SPLITS))
        raise ValueError(f"Unknown split '{split}'. Available: {available}")
    df = pd.read_parquet(f"hf://datasets/disco-eth/edm-cue/{SPLITS[split]}")
    print(f"Loaded {len(df)} songs from '{split}' split.")
    return df


def plot_bpm_distribution(df: pd.DataFrame, split: str, bins: int = 30, debug: bool = False) -> None:
    raw_bpm = pd.to_numeric(df["beat_grid"].apply(_extract_bpm), errors="coerce")
    bpm_values = raw_bpm.dropna()
    if bpm_values.empty:
        raise ValueError("No BPM values were extracted from beat_grid.")

    total_songs = len(df)
    bpm_coverage = len(bpm_values)
    coverage_pct = (bpm_coverage / total_songs * 100) if total_songs else 0.0
    print(f"BPM coverage: {bpm_coverage}/{total_songs} songs ({coverage_pct:.2f}%).")

    if debug:
        missing_mask = raw_bpm.isna()
        missing_count = int(missing_mask.sum())
        if missing_count:
            print(f"{missing_count} track(s) missing BPM data:")
            debug_cols = ["id", "title", "artists", "genre"]
            for _, row in df.loc[missing_mask, debug_cols].iterrows():
                print(f"  - {row['title']} by {row['artists']} (id={row['id']}, genre={row['genre']})")

    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bin_edges, _ = ax.hist(bpm_values, bins=bins, color="#1f77b4", edgecolor="white")
    ax.set_title(f"EDM-CUE {split.capitalize()} Split BPM Distribution")
    ax.set_xlabel("Beats per Minute (BPM)")
    ax.set_ylabel("Song Count")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plt.show()

    bin_labels = [
        f"{bin_edges[i]:.1f}–{bin_edges[i + 1]:.1f}"
        for i in range(len(bin_edges) - 1)
    ]
    bpm_table = pd.DataFrame(
        {
            "bin": bin_labels,
            "count": counts.astype(int),
            "pct_of_total": np.where(
                total_songs,
                (counts / total_songs) * 100,
                0,
            ),
        }
    )
    print("\nBPM bin summary:")
    print(bpm_table.to_string(index=False, formatters={"pct_of_total": "{:.2f}%".format}))


def plot_genre_distribution(df: pd.DataFrame, split: str, debug: bool = False) -> None:
    bpm_series = pd.to_numeric(df["beat_grid"].apply(_extract_bpm), errors="coerce")
    genre_lists = df["genre"].apply(_parse_genres)
    taxonomy_lookup = {entry["label"]: entry for entry in GENRE_TAXONOMY}
    bpm_bucket_lookup = {label: (low, high) for label, low, high in BPM_BUCKETS}

    def _print_section(title: str) -> None:
        print(f"\n{title}")
        print("-" * len(title))

    primary_labels: List[str] = []
    songs_bucketed = 0
    songs_with_named_bucket = 0
    unmapped_records: List[dict] = []

    for idx, (genres, bpm) in enumerate(zip(genre_lists, bpm_series)):
        bpm_value = None if pd.isna(bpm) else float(bpm)
        normalized_tokens = [_normalize_token(label) for label in genres]
        normalized_tokens = [token for token in normalized_tokens if token]

        primary_label = _find_taxonomy_label(normalized_tokens, bpm_value)
        if not primary_label and bpm_value is not None:
            primary_label = _bucket_by_bpm(bpm_value)
        if not primary_label:
            primary_label = "Other / Experimental"

        primary_labels.append(primary_label)
        songs_bucketed += 1
        if primary_label != "Other / Experimental":
            songs_with_named_bucket += 1
        else:
            row = df.iloc[idx]
            unmapped_records.append(
                {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "artists": row.get("artists"),
                    "raw_genre": row.get("genre"),
                    "parsed_tokens": genres,
                    "bpm": bpm_value,
                }
            )

    total_songs = len(df)
    if total_songs == 0 or not primary_labels:
        raise ValueError("No genre information available to plot.")

    bucket_pct = (songs_bucketed / total_songs * 100) if total_songs else 0.0
    named_pct = (songs_with_named_bucket / total_songs * 100) if total_songs else 0.0
    if debug:
        _print_section("Coverage")
        coverage_rows = [
            ("Total songs", f"{total_songs}"),
            ("Bucketed songs", f"{songs_bucketed} ({bucket_pct:.2f}%)"),
            ("Named taxonomy bucket", f"{songs_with_named_bucket} ({named_pct:.2f}%)"),
        ]
        for label, value in coverage_rows:
            print(f"{label:<22} | {value}")
    else:
        print(
            f"Genre bucket coverage: {songs_bucketed}/{total_songs} songs ({bucket_pct:.2f}%)."
        )
        print(
            f"Mapped to named taxonomy buckets: {songs_with_named_bucket}/{total_songs} songs ({named_pct:.2f}%)."
        )

    if debug and unmapped_records:
        _print_section("Unmapped tracks")
        for record in unmapped_records:
            print(
                f"  - {record['title']} by {record['artists']} "
                f"(id={record['id']}, bpm={record['bpm']}) "
                f"raw_genre={record['raw_genre']} parsed={record['parsed_tokens']}"
            )

    genre_counts = pd.Series(primary_labels).value_counts().sort_values(ascending=False)
    if genre_counts.empty:
        raise ValueError("No genre information available to plot.")

    bucket_series = pd.Series(primary_labels, index=df.index, name="bucket")
    df_with_bucket = df.copy()
    df_with_bucket["bucket"] = bucket_series
    df_with_bucket["bpm"] = bpm_series

    genre_table = pd.DataFrame(
        {
            "genre": genre_counts.index,
            "count": genre_counts.values.astype(int),
            "pct_of_total": np.where(
                total_songs,
                (genre_counts.values / total_songs) * 100,
                0,
            ),
        }
    )
    print("\nGenre summary:")
    print(
        genre_table.to_string(
            index=False,
            formatters={"pct_of_total": "{:.2f}%".format},
        )
    )

    bucket_bpm_data: Dict[str, np.ndarray] = {}
    bucket_outliers: Dict[str, List[dict]] = defaultdict(list)
    bucket_accuracy: Dict[str, Dict[str, int]] = {}

    for label in genre_counts.index:
        bucket_df = df_with_bucket[df_with_bucket["bucket"] == label]
        bpm_col = bucket_df["bpm"]
        valid_bpm = bpm_col.dropna()
        bucket_bpm_data[label] = valid_bpm.values

        if debug:
            expected_range = taxonomy_lookup.get(label, {}).get("bpm_range") or bpm_bucket_lookup.get(label)
            if expected_range and len(valid_bpm) > 0:
                in_range = int(sum(_bpm_in_range(val, expected_range) for val in valid_bpm))
                bucket_accuracy[label] = {"total": int(len(valid_bpm)), "in_range": in_range}
            elif expected_range:
                bucket_accuracy[label] = {"total": 0, "in_range": 0}

            if len(valid_bpm) > 0:
                lower, upper = _boxplot_bounds(valid_bpm.values)
                outlier_mask = bpm_col.lt(lower) | bpm_col.gt(upper)
                outlier_mask = outlier_mask.fillna(False)
                if outlier_mask.any():
                    outlier_rows = bucket_df.loc[outlier_mask, ["title", "artists", "genre", "bpm"]]
                    bucket_outliers[label] = [
                        {
                            "title": row["title"],
                            "artists": row["artists"],
                            "bpm": row["bpm"],
                            "tags": _parse_genres(row["genre"]),
                        }
                        for _, row in outlier_rows.iterrows()
                    ]

    if debug:
        _print_section("BPM Accuracy by Bucket")
        header = f"{'Bucket':<22} | {'In Range':>9} | {'Total':>5} | {'Pct':>7}"
        print(header)
        print("-" * len(header))
        for label in genre_counts.index:
            stats = bucket_accuracy.get(label)
            if not stats:
                print(f"{label:<22} | {'n/a':>9} | {'n/a':>5} | {'n/a':>7}")
                continue
            total = stats["total"]
            in_range = stats["in_range"]
            pct = (in_range / total * 100) if total else 0.0
            print(f"{label:<22} | {in_range:>9} | {total:>5} | {pct:>6.2f}%")

        _print_section("BPM Outliers")
        header = f"{'Bucket':<22} | {'Track':<45} | {'BPM':>6} | Tags"
        print(header)
        print("-" * len(header))

        def _format_track(title: str, artists: str, width: int = 45) -> str:
            text = f"{title} — {artists}"
            if len(text) <= width:
                return text
            return text[: width - 1] + "…"

        any_outliers = False
        for label in genre_counts.index:
            entries = bucket_outliers.get(label, [])
            if not entries:
                print(f"{label:<22} | {'(no outliers)':<45} | {'':>6} | ")
                continue
            any_outliers = True
            for entry in entries:
                tags = entry["tags"] or ["<missing>"]
                tag_text = ", ".join(tags)
                bpm_text = f"{entry['bpm']:.2f}" if entry["bpm"] is not None else "n/a"
                track_text = _format_track(entry["title"], entry["artists"])
                print(f"{label:<22} | {track_text:<45} | {bpm_text:>6} | {tag_text}")
        if not any_outliers:
            print("No BPM outliers detected.")

    with plt.rc_context(_GENRE_FONT_CONFIG):
        num_buckets = len(genre_counts)
        fig_width = max(12, num_buckets * 2)
        fig_height = 10
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = fig.add_gridspec(
            nrows=2,
            ncols=num_buckets,
            height_ratios=[1, 1],
            hspace=0.4,
            wspace=0.3,
        )

        ax_bar = fig.add_subplot(gs[0, :])
        ax_bar.bar(genre_counts.index, genre_counts.values, color="#ff7f0e")
        ax_bar.set_title(f"EDM-CUE {split.capitalize()} Split Genre Distribution")
        ax_bar.set_ylabel("Song Count")
        ax_bar.grid(axis="y", alpha=0.2)
        plt.setp(ax_bar.get_xticklabels(), rotation=45, ha="right")

        for idx, label in enumerate(genre_counts.index):
            ax_box = fig.add_subplot(gs[1, idx])
            bpm_values = bucket_bpm_data.get(label)
            ax_box.set_title(label, fontsize=10, pad=10)
            if bpm_values is None or len(bpm_values) == 0:
                ax_box.text(0.5, 0.5, "No BPM data", transform=ax_box.transAxes, ha="center", va="center")
                ax_box.set_ylabel("BPM")
                ax_box.set_xticks([])
                ax_box.grid(axis="y", alpha=0.2)
                continue

            ax_box.boxplot(
                [bpm_values],
                vert=True,
                patch_artist=True,
                boxprops={"facecolor": "#1f77b4", "alpha": 0.6},
                medianprops={"color": "black"},
                whiskerprops={"color": "#1f77b4"},
                capprops={"color": "#1f77b4"},
                flierprops={"markerfacecolor": "#ff7f0e", "markeredgecolor": "#ff7f0e", "markersize": 5},
            )
            y_min, y_max = float(bpm_values.min()), float(bpm_values.max())
            padding = max(1.0, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
            ax_box.set_ylim(y_min - padding, y_max + padding)
            ax_box.set_ylabel("BPM")
            ax_box.set_xticks([])
            ax_box.grid(axis="y", alpha=0.2)

        fig.tight_layout()
        plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize EDM-CUE dataset distributions.")
    parser.add_argument("--split", default="train", choices=sorted(SPLITS), help="Dataset split to visualize.")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins for the BPM histogram.")
    parser.add_argument("--show_bpm", action="store_true", help="Display the BPM distribution histogram.")
    parser.add_argument("--show_genre", action="store_true", help="Display the genre distribution chart.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed diagnostics (e.g., tracks missing BPM or taxonomy mappings).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=None if argv is None else list(argv))

    if not args.show_bpm and not args.show_genre:
        parser.error("Please specify --show_bpm, --show_genre, or both.")

    df = load_split(args.split)

    if args.show_bpm:
        plot_bpm_distribution(df=df, split=args.split, bins=args.bins, debug=args.debug)
    if args.show_genre:
        plot_genre_distribution(df=df, split=args.split, debug=args.debug)
