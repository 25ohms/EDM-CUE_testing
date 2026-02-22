import argparse
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# These imports assume visualization.py and config.py are in the same directory
from config import BPM_BUCKETS
from visualization import (
    SPLITS,
    bucket_by_bpm,
    extract_bpm,
    load_split,
    plot_bpm_distribution,
    plot_genre_distribution,
)

# --- DIRECTORY SETUP ---
def _default_samples_dir() -> Path:
    path = Path("data/samples")
    path.mkdir(parents=True, exist_ok=True)
    return path

SAMPLES_DIR = _default_samples_dir()

# --- DEEZER API CHECK ---
def is_streamable_in_ca(deezer_id: str) -> bool:
    """Queries Deezer to see if the track is available in Canada."""
    try:
        url = f"https://api.deezer.com/track/{deezer_id}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "error" in data: 
                return False
            return "CA" in data.get("available_countries", [])
    except Exception:
        return False
    return False

# --- SAMPLING LOGIC ---
def _build_targets(
    bucket_order: Sequence[str],
    available_counts: Dict[str, int],
    sample_size: int,
) -> Tuple[Dict[str, int], int]:
    bucket_count = len(bucket_order)
    if bucket_count == 0:
        raise ValueError("No BPM buckets available.")

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
            if leftover == 0: break
            capacity = residual_capacity.get(bucket, 0)
            if capacity <= 0: continue
            take = min(capacity, leftover)
            actual[bucket] += take
            residual_capacity[bucket] -= take
            leftover -= take

    return actual, sum(actual.values())

def stratified_bpm_sample(df: pd.DataFrame, sample_size: int, seed: Optional[int] = None) -> pd.DataFrame:
    working = df.copy()
    working["bpm"] = pd.to_numeric(working["beat_grid"].apply(extract_bpm), errors="coerce")
    working = working.dropna(subset=["bpm"]).copy()
    working["bpm_bucket"] = working["bpm"].apply(bucket_by_bpm)

    bucket_counts = working["bpm_bucket"].value_counts().to_dict()
    bucket_order = [label for label, _, _ in BPM_BUCKETS if label in bucket_counts]
    
    targets, _ = _build_targets(bucket_order, bucket_counts, sample_size)
    rng = np.random.default_rng(seed)
    final_samples = []

    print(f"\n--- Starting Canadian Verification (Target Total: {sample_size}) ---")

    for bucket in bucket_order:
        needed = targets.get(bucket, 0)
        if needed <= 0: continue
        
        # Get all songs in this bucket and shuffle
        candidates = working[working["bpm_bucket"] == bucket].sample(frac=1, random_state=seed)
        total_in_bucket = len(candidates)
        found_for_bucket = []
        
        # New Print Statement: Shows required vs available in the whole dataset
        print(f"\nBucket: {bucket}")
        print(f"  Target: {needed} tracks | Available in Dataset: {total_in_bucket} tracks")
        
        candidate_iter = candidates.iterrows()
        
        while len(found_for_bucket) < needed:
            try:
                _, row = next(candidate_iter)
            except StopIteration:
                deficit = needed - len(found_for_bucket)
                print(f"  [!] ALERT: Exhausted all {total_in_bucket} candidates in '{bucket}'.")
                print(f"      Could only find {len(found_for_bucket)}/{needed} CA-available tracks.")
                break
            
            track_id = str(row["id"])
            if is_streamable_in_ca(track_id):
                found_for_bucket.append(row)
                # Show progress within the bucket
                print(f"  [{len(found_for_bucket)}/{needed}] PASS: {row['title']} ({track_id})")
            else:
                print(f"  [----] FAIL: {row['title']} ({track_id}) - Blocked in CA. Searching for replacement...")
            
            time.sleep(0.05)

        final_samples.extend(found_for_bucket)

    return pd.DataFrame(final_samples)

# --- CLI HELPERS ---
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a stratified BPM sample available in Canada.")
    parser.add_argument("--split", default="train", choices=sorted(SPLITS))
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_bpm", action="store_true")
    parser.add_argument("--show_genre", action="store_true")
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    df = load_split(args.split)
    sample_df = stratified_bpm_sample(df, args.sample_size, args.seed)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = SAMPLES_DIR / f"{args.split}_sample_CA_{len(sample_df)}_{timestamp}.csv"
    sample_df.to_csv(output_path, index=False)
    
    print(f"\nFinal Total Sample Size: {len(sample_df)} / {args.sample_size}")
    print(f"Saved to: {output_path}")

    if args.show_bpm:
        plot_bpm_distribution(sample_df, f"{args.split} (Canada)")
    if args.show_genre:
        plot_genre_distribution(sample_df, f"{args.split} (Canada)")

if __name__ == "__main__":
    main()