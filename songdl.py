from yt_dlp import YoutubeDL
import pandas as pd
import os
import re
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, APIC, error

CSV_PATH = "./samples/train_sample_100_seed2025_20251126-194143.csv"
OUTPUT_DIR = "./songs"

# ---- Make filenames safe ----


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '', name)


# ---- Load CSV ----
df = pd.read_csv(CSV_PATH)

# ---- yt-dlp options ----
ydl_opts = {
    "format": "bestaudio/best",
    "quiet": False,
    "noplaylist": True,

    "remote_components": ["ejs:github"],

    "js_runtimes": {"deno": {}},
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
}

output_folder = OUTPUT_DIR
os.makedirs(output_folder, exist_ok=True)

# ---- Download Loop ----
for _, row in df.iterrows():
    title = str(row["title"])
    artist = str(row["artists"])

    query = f"{artist} {title} audio"

    print(f"\n🔍 Searching & downloading: {title} – {artist}")

    # Download audio
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch:{query}", download=True)
        entry = info["entries"][0]
        files = ydl.prepare_filename(entry)
        temp_file = os.path.splitext(files)[0] + ".mp3"

    # Rename to title.mp3
    final_name = safe_filename(f"{title}.mp3")
    final_path = os.path.join(output_folder, final_name)

    if not os.path.exists(temp_file):
        print(f"❌ Missing file: {temp_file}")
        continue

    os.rename(temp_file, final_path)
    print(f"✅ Saved as: {final_path}")

    # ---- Insert Metadata ----
    try:
        audio = EasyID3(final_path)
    except error:
        audio = ID3()

    audio["title"] = title
    audio["artist"] = artist
    audio.save(final_path)

    print(f"🎵 Added metadata → Title: {title}, Artist: {artist}")

print("\n🎉 Done.")
