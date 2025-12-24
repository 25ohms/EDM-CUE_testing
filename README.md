# EDM-CUE Visualization & Sampling Utilities

CLI tools for inspecting and sampling the EDM-CUE Hugging Face dataset. All commands read the train/valid splits directly from the `hf://datasets/disco-eth/edm-cue/…` parquet files, so the data is always pulled from the source.

## Quickstart
- Install Python 3.9+ dependencies: `pip install pandas numpy matplotlib pyarrow fsspec huggingface_hub` (any parquet-capable engine such as `pyarrow` works).
- Visualizations: `python -m visualization --show_bpm --show_genre`
- Stratified sampling: `python sample_songs.py --show_bpm --show_genre`
- View an existing sample CSV: `python view_sample.py --file <csv> --show_bpm --show_genre`
- TO USE `songdl.py`, ensure you have [deno](https://deno.com/) installed.

All scripts fetch data on demand from Hugging Face and display figures via Matplotlib; close the window(s) to exit.

## Project Layout (New)
- `visualization/`: Core visualization module (`python -m visualization`).
- `tools/sampling/`: Stratified sampling + sample viewer implementations.
- `tools/audio/`: Download + Rekordbox XML extraction + fix script implementations.
- `data/samples/`: Generated sample CSVs (symlinked as `samples/` for compatibility).
- `data/songs/`: Downloaded audio (symlinked as `songs/` for compatibility).
- `data/exports/`: JSON exports such as `dataset.json` (symlinked at repo root).
- Top-level `sample_songs.py`, `view_sample.py`, `songdl.py`, `xml_extractor.py`, and `fix_tracks.sh` remain as compatibility wrappers.

## Visualization Module (`python -m visualization`)
Plots BPM histograms and genre distributions for any split without mutating data.

Flags:
- `--split {train,valid}`: Which parquet shard to load (`data/<split>-00000-of-00001.parquet` on HF).
- `--bins <int>`: Histogram bins for BPM (default 30).
- `--show_bpm`: BPM histogram plus a textual per-bin summary.
- `--show_genre`: Genre bucket bar chart, per-bucket BPM boxplots, and summary table.
- `--debug`: Adds verbose diagnostics (BPM coverage/missing tracks, taxonomy coverage, BPM accuracy per bucket, outliers).

If neither `--show_bpm` nor `--show_genre` is provided the CLI exits with an error reminding you to pick a plot.

## Stratified Sampling (`python sample_songs.py`)
Creates a uniform-as-possible sample across BPM buckets. Each run:
1. Extracts BPM from `beat_grid`, drops rows missing BPM, and tags every track with a bucket (`config/BPM_BUCKETS` order).
2. Allocates as even a distribution as possible across buckets, borrowing from buckets with spare capacity.
3. Samples tracks using a seeded RNG for reproducibility.
4. Writes the sampled metadata to `samples/<split>_sample_<size>_seed<seed>_<timestamp>.csv` (or a custom `--output` name).
5. Optionally reuses the visualization plots to inspect the sampled subset.

Flags:
- `--split {train,valid}`: Source split.
- `--sample_size <int>`: Requested sample size (defaults to 100; automatically capped to the number of rows containing BPM).
- `--seed <int>`: RNG seed (default 42) so repeated runs can reproduce the same sample.
- `--bins`, `--show_bpm`, `--show_genre`, `--debug`: Same behavior as the visualization CLI, but applied to the sampled subset.
- `--output <filename>`: Optional CSV name (still saved under `samples/`).

The CLI prints an on-stdout summary of the sampled bucket distribution in addition to writing the CSV.

## Sample Viewer (`python view_sample.py`)
Loads any existing CSV (paths are resolved relative to `samples/` if not found at the provided location), prints its bucket coverage, and reuses the visualization plots without touching the Hugging Face dataset.

Flags:
- `--file <path>`: Required; sample CSV to load.
- `--split <label>`: Title suffix for the plots (default `sample`).
- `--bins`, `--show_bpm`, `--show_genre`, `--debug`: Same semantics as above; at least one of the `--show_*` flags is required.

Use this when you only need to inspect previously generated samples.

## Audio Pipeline Utilities
These utilities build on the sampled CSVs to download audio and extract labeled cue sections.

### Song Downloader (`python songdl.py`)
Downloads MP3s from YouTube search results based on the sample CSV.

Flags:
- `--csv <path>`: Sample CSV to download (default: `data/samples/train_sample_100_seed2025_20251126-194143.csv`).
- `--output_dir <path>`: Destination for MP3s (default: `data/songs/`).
- `--audio_quality <value>`: FFmpeg audio quality (default: `192`).

Notes:
- Requires `yt-dlp`, `ffmpeg`, and `deno` (for the JS runtime).
- Uses the song title as the filename and writes ID3 metadata for title + artist.

### Rekordbox XML Extractor (`python xml_extractor.py`)
Parses `rekordbox.xml`, matches against the sample CSV, and emits a JSON dataset with labeled cue sections.

Flags:
- `--xml <path>`: Rekordbox XML path (default: `rekordbox.xml`).
- `--csv <path>`: Sample CSV to match against (default: `data/samples/train_sample_100_seed2025_20251126-194143.csv`).
- `--output <path>`: JSON export path (default: `data/exports/dataset.json`).

### Fix Incorrect Downloads (`./fix_tracks.sh <input.txt>`)
Downloads a corrected list of tracks from explicit URLs. Each entry is two lines:
1) URL
2) Desired filename (without extension)

Default input lives at `data/inputs/songs_to_fix.txt`, but any path is accepted.

## Taxonomy Overview
Genre normalization and bucketing are configured in `config/taxonomy.py`:
- `GENRE_TAXONOMY`: Keyword-based genre entries with optional BPM ranges and priorities.
- `BPM_BUCKETS`: Fallback ranges used for both visualization and sampling.
- `GENRE_SCORING`: Weights for keyword matches and BPM alignment.
- `GENRE_DELIMITER_PATTERN`: Splits multi-genre tags on `/` or `\`.

You can tweak these settings to adjust label mapping without touching the CLI scripts. All CLIs automatically pick up any taxonomy changes.
