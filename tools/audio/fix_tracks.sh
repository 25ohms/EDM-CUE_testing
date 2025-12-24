#!/usr/bin/env bash

set -euo pipefail

# ---------------- CONFIG ----------------
OUTPUT_DIR="${OUTPUT_DIR:-./data/songs}"
AUDIO_QUALITY="${AUDIO_QUALITY:-192K}"
# ---------------------------------------

if [[ ! -d "$OUTPUT_DIR" && -d "./songs" ]]; then
  OUTPUT_DIR="./songs"
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <input.txt>"
  exit 1
fi

INPUT_FILE="$1"

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "❌ File not found: $INPUT_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Read file two lines at a time
while IFS= read -r url && IFS= read -r name; do
  if [[ -z "$url" || -z "$name" ]]; then
    echo "⚠️  Skipping incomplete entry"
    continue
  fi

  echo "🎵 Downloading:"
  echo "   URL : $url"
  echo "   Name: $name"

  yt-dlp \
    --remote-components ejs:github \
    --js-runtime deno \
    -f bestaudio/best \
    --extract-audio \
    --audio-format mp3 \
    --audio-quality "$AUDIO_QUALITY" \
    --no-playlist \
    -o "$OUTPUT_DIR/$name.%(ext)s" \
    "$url"

  echo "✅ Saved as: $OUTPUT_DIR/$name.mp3"
  echo
done <"$INPUT_FILE"

echo "🎉 All tracks processed."
