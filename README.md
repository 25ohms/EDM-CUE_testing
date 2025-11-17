# EDM-CUE Visualization Utilities

Python CLI for inspecting the EDM-CUE Hugging Face dataset. The script reads the train/valid splits directly from the `hf://datasets/disco-eth/edm-cue/…` parquet files and renders simple BPM and genre distribution charts.

## Quickstart
- Install Python 3.9+ dependencies: `pip install pandas numpy matplotlib pyarrow fsspec huggingface_hub` (any parquet-capable engine such as `pyarrow` works).
- From the repo root, run the visualizer and pick the plots you want:
  - BPM histogram: `python visualization.py --show_bpm`
  - Genre distribution + per-bucket BPM boxplots: `python visualization.py --show_genre`
  - Both plots together: `python visualization.py --show_bpm --show_genre`
  - Use `--split valid` to switch from the default train split, and `--bins 40` to change the histogram granularity.

The script downloads data on demand from Hugging Face, so expect network usage on first run. Figures are displayed via Matplotlib; close the window to exit.

## Flags and Behavior
- `--split {train,valid}`: Chooses which parquet shard to load (`data/<split>-00000-of-00001.parquet` on HF).
- `--bins <int>`: Number of histogram bins for the BPM plot (default 30).
- `--show_bpm`: Draws a histogram of BPM values extracted from each track's `beat_grid` field; also prints a per-bin table with counts and percent of total.
- `--show_genre`: Assigns each track to a genre bucket using both tagged genres and BPM ranges (taxonomy in `config/taxonomy.py`), then plots:
  - A bar chart of bucket counts.
  - A boxplot of BPM values per bucket.
  - A textual summary table with counts and percentages.
- `--debug`: Adds verbose stdout diagnostics:
  - For BPM: lists tracks missing BPM and shows coverage.
  - For genres: reports coverage, unmapped tracks, BPM range accuracy by bucket, and BPM outliers per bucket.

If neither `--show_bpm` nor `--show_genre` is provided, the CLI exits with an error reminding you to pick a plot.

## Taxonomy Overview
Genre normalization and bucketing are configured in `config/taxonomy.py`:
- `GENRE_TAXONOMY`: Keyword-based genre entries with optional BPM ranges and priorities.
- `BPM_BUCKETS`: Fallback ranges used when only BPM is available.
- `GENRE_SCORING`: Weights for keyword matches and BPM alignment.
- `GENRE_DELIMITER_PATTERN`: Splits multi-genre tags on `/` or `\`.

You can tweak these settings to adjust label mapping without touching the visualization code.
