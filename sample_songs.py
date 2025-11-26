import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import BPM_BUCKETS
from visualization import (
    SPLITS,
    bucket_by_bpm,
    extract_bpm,
    load_split,
    plot_bpm_distribution,
    plot_genre_distribution,
)

SAMPLES_DIR = Path("samples")


def _ordered_buckets(bucket_counts: Dict[str, int]) -> Sequence[str]:
    ordered = [label for label, _, _ in BPM_BUCKETS if label in bucket_counts]
    extras = [label for label in bucket_counts if label not in ordered]
    return ordered + sorted(extras)


def _build_targets(
    bucket_order: Sequence[str],
    available_counts: Dict[str, int],
    sample_size: int,
) -> Tuple[Dict[str, int], int]:
    bucket_count = len(bucket_order)
    if bucket_count == 0:
        raise ValueError("No BPM buckets available for sampling.")

    base, remainder = divmod(sample_size, bucket_count)
    desired = {bucket: base for bucket in bucket_order}
    for bucket in bucket_order[:remainder]:
        desired[bucket] += 1

    actual: Dict[str, int] = {}
    leftover = 0
    residual_capacity: Dict[str, int] = {}

    for bucket in bucket_order:
        capacity = available_counts.get(bucket, 0)
        take = min(desired[bucket], capacity)
        actual[bucket] = take
        residual_capacity[bucket] = capacity - take
        leftover += desired[bucket] - take

    if leftover > 0:
        for bucket in bucket_order:
            if leftover == 0:
                break
            capacity = residual_capacity.get(bucket, 0)
            if capacity <= 0:
                continue
            take = min(capacity, leftover)
            actual[bucket] += take
            residual_capacity[bucket] -= take
            leftover -= take

    sampled_total = sum(actual.values())
    return actual, sampled_total


def stratified_bpm_sample(df: pd.DataFrame, sample_size: int, seed: Optional[int] = None) -> pd.DataFrame:
    working = df.copy()
    working["bpm"] = pd.to_numeric(working["beat_grid"].apply(extract_bpm), errors="coerce")
    working = working.dropna(subset=["bpm"]).copy()
    if working.empty:
        raise ValueError("No BPM values available to build a sample.")
    working["bpm_bucket"] = working["bpm"].apply(bucket_by_bpm)

    bucket_counts = working["bpm_bucket"].value_counts().to_dict()
    bucket_order = _ordered_buckets(bucket_counts)

    total_available = int(working.shape[0])
    desired_size = min(sample_size, total_available)
    if desired_size < sample_size:
        print(
            f"Requested {sample_size} tracks but only {total_available} contain BPM data. "
            f"Sampling {desired_size} instead."
        )

    targets, achieved = _build_targets(bucket_order, bucket_counts, desired_size)
    if achieved < desired_size:
        raise ValueError("Insufficient tracks to satisfy sampling request.")

    rng = np.random.default_rng(seed)
    samples: list[pd.DataFrame] = []
    for bucket in bucket_order:
        count = targets.get(bucket, 0)
        if count <= 0:
            continue
        bucket_df = working[working["bpm_bucket"] == bucket]
        if bucket_df.empty:
            continue
        if count >= len(bucket_df):
            samples.append(bucket_df)
            continue
        random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        samples.append(bucket_df.sample(n=count, random_state=random_state))

    sample_df = pd.concat(samples, axis=0).reset_index(drop=True)
    return sample_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a stratified BPM sample of EDM-CUE songs.")
    parser.add_argument("--split", default="train", choices=sorted(SPLITS), help="Dataset split to sample from.")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of songs to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random sampling process.")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins for optional BPM histogram.")
    parser.add_argument("--show_bpm", action="store_true", help="Display the BPM distribution for the sample.")
    parser.add_argument("--show_genre", action="store_true", help="Display the genre distribution for the sample.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed diagnostics when rendering optional visualizations.",
    )
    parser.add_argument(
        "--output",
        help="Optional filename for the CSV output (stored inside the samples/ directory).",
    )
    return parser


def _summarize_sample(df: pd.DataFrame) -> None:
    bucket_counts = df["bpm_bucket"].value_counts().sort_index()
    print("\nSample bucket distribution:")
    for bucket, count in bucket_counts.items():
        pct = (count / len(df) * 100) if len(df) else 0.0
        print(f"  - {bucket:<22} {count:>4} tracks ({pct:>5.1f}%)")


def _write_sample(df: pd.DataFrame, split: str, seed: int, output_name: Optional[str]) -> Path:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if output_name:
        filename = output_name
    else:
        filename = f"{split}_sample_{len(df)}_seed{seed}_{timestamp}.csv"
    path = SAMPLES_DIR / filename
    df.to_csv(path, index=False)
    return path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=None if argv is None else list(argv))

    if args.sample_size <= 0:
        parser.error("--sample_size must be positive.")

    df = load_split(args.split)
    sample_df = stratified_bpm_sample(df, sample_size=args.sample_size, seed=args.seed)

    output_path = _write_sample(sample_df, split=args.split, seed=args.seed, output_name=args.output)
    print(f"\nWrote {len(sample_df)} sampled tracks to {output_path}")
    _summarize_sample(sample_df)

    if args.show_bpm:
        plot_bpm_distribution(df=sample_df, split=f"{args.split} sample", bins=args.bins, debug=args.debug)
    if args.show_genre:
        plot_genre_distribution(df=sample_df, split=f"{args.split} sample", debug=args.debug)


if __name__ == "__main__":
    main()
