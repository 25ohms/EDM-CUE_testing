import xml.etree.ElementTree as ET
import json
import csv
import os
import urllib.parse
from urllib.parse import urlparse

# --- Configuration ---
XML_FILE = 'rekordbox.xml'
# Your specific CSV file path
CSV_FILE = './samples/train_sample_100_seed2025_20251126-194143.csv'
OUTPUT_FILE = 'dataset.json'

# --- Helper Functions ---

def clean_text(text):
    """
    Basic normalization: lowercase and strip spaces.
    Since you fixed your audio filenames, we don't need aggressive regex removal.
    """
    if not text: return ""
    return str(text).lower().strip()

def build_xml_index(xml_file):
    """
    Parses XML and builds a hash map for O(1) lookups.
    Structure: { "normalized_title": [List of Track Elements] }
    """
    if not os.path.exists(xml_file):
        print(f"Error: XML file not found at {xml_file}")
        return {}

    print("Indexing Rekordbox XML...")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError:
        print("Error: Could not parse XML.")
        return {}

    collection = root.find('COLLECTION')
    xml_index = {}
    
    for track in collection.findall('TRACK'):
        title = clean_text(track.get('Name'))
        
        # Handle duplicate titles (e.g., Remixes vs Originals) by storing list of matches
        if title not in xml_index:
            xml_index[title] = []
        xml_index[title].append(track)
        
    print(f"Indexed {len(collection.findall('TRACK'))} tracks.")
    return xml_index

def find_exact_match(title_matches, target_artist):
    """
    If multiple songs have the same title, pick the one matching the artist.
    """
    target_artist = clean_text(target_artist)
    
    for track in title_matches:
        xml_artist = clean_text(track.get('Artist'))
        
        # Check if Artist strings overlap (e.g. "Drake" vs "Drake, Future")
        if xml_artist == target_artist or target_artist in xml_artist or xml_artist in target_artist:
            return track
            
    # Fallback: Return the first result if artist mismatch (assume unique title)
    return title_matches[0]

# --- Main Logic ---

def process_database_matches():
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found at {CSV_FILE}")
        return

    # 1. Build Index (Fast)
    xml_db = build_xml_index(XML_FILE)
    if not xml_db: return

    processed_data = []
    missing_tracks = []

    print(f"Reading CSV: {CSV_FILE}...")
    
    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        if 'title' not in reader.fieldnames:
            print("Error: CSV missing 'title' column.")
            return

        for row in reader:
            csv_id = row.get('id', 'N/A')
            csv_title = row.get('title', '')
            csv_artist = row.get('artists', '') # Note: CSV 'artists' vs XML 'Artist'

            clean_title = clean_text(csv_title)
            
            # 2. Fast Lookup
            potential_matches = xml_db.get(clean_title)

            if not potential_matches:
                missing_tracks.append(f"{csv_title} (Not found in XML)")
                continue

            # 3. Refine (if multiple matches exist)
            match = find_exact_match(potential_matches, csv_artist)
            
            # --- Extract Data ---
            artist = match.get('Artist')
            title = match.get('Name')
            try:
                total_time = float(match.get('TotalTime'))
            except:
                continue

            # Decode File Path
            raw_location = match.get('Location')
            file_path = urllib.parse.unquote(urlparse(raw_location).path)
            if os.name == 'nt' and file_path.startswith('/') and ':' in file_path:
                file_path = file_path.lstrip('/')

            # --- 4. EXTRACT CUES (The Fix) ---
            cues = []
            for mark in match.findall('POSITION_MARK'):
                label_name = mark.get('Name')
                
                # IMPORTANT: We accept Hot Cues (0-7) AND Memory Cues (-1)
                # We simply check: Did you name it?
                if label_name:
                    cues.append({
                        "time": float(mark.get('Start')),
                        "label": label_name
                    })

            if not cues:
                print(f"Skipping '{title}': Match found, but no labeled cues.")
                continue

            # Sort cues
            cues.sort(key=lambda x: x['time'])
            
            sections = []
            for i in range(len(cues)):
                start_time = cues[i]['time']
                label = cues[i]['label']
                
                if i < len(cues) - 1:
                    end_time = cues[i+1]['time']
                else:
                    end_time = total_time

                # Filter out zero-length errors
                if end_time > start_time + 0.05:
                    sections.append({
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "label": label
                    })

            processed_data.append({
                "id": csv_id,
                "song": f"{artist} - {title}",
                "file_path": file_path,
                "duration": total_time,
                "sections": sections
            })

    # --- Save Output ---
    if processed_data:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2)
        print(f"\nSuccess! Exported {len(processed_data)} tracks to '{OUTPUT_FILE}'.")
    
    if missing_tracks:
        print(f"\nWarning: {len(missing_tracks)} songs from CSV were not found in XML.")
        for t in missing_tracks[:5]: print(f" - {t}")

# --- Run ---
process_database_matches()