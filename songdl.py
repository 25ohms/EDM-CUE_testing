from yt_dlp import YoutubeDL
import pandas as pd


CSV_PATH = "./samples/train_sample_100_seed2025_20251126-194143.csv"
OUTPUT_DIR = "./songs"

# Read CSV
df = pd.read_csv(CSV_PATH)

# yt-dlp options
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': f'{OUTPUT_DIR}/%(artist)s - %(title)s.%(ext)s',  # filename template
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'quiet': False,
    'noplaylist': True,
    'ignoreerrors': True,  # continue on download errors
}

with YoutubeDL(ydl_opts) as ydl:
    for _, row in df.iterrows():
        search_query = f"{row['title']} {row['artists']}"
        print(f"Downloading: {search_query}")
        try:
            # ytsearch1: only download the first YouTube result
            ydl.download([f"ytsearch1:{search_query}"])
        except Exception as e:
            print(f"Failed to download {search_query}: {e}")

