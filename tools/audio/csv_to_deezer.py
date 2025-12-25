import csv
import requests

# Configuration
ARL_TOKEN = '63dfb5a220be452a02c9d8ce183f8c21e73532416f07c6c161dbde3112d77d416da1d9e01d4f4fc447f97a348c9dc3bef41719f50f3b998a0794c6ecbd42cf0de52ceb7283a28e85d077639e8116491862ce89b1d2c540b934db18decfe01802'
CSV_FILE = 'data/samples/train_sample_100_seed42_20251224-185006.csv'
PLAYLIST_NAME = 'WATAI'

def create_deezer_playlist():
    session = requests.Session()
    session.cookies.set('arl', ARL_TOKEN, domain='.deezer.com')

    # 1. Get the 'checkForm' token (required for internal POST actions)
    print("Fetching internal API token...")
    user_data_resp = session.get("https://www.deezer.com/ajax/gw-light.php?method=deezer.getUserData&api_version=1.0&api_token=")
    user_data = user_data_resp.json()
    
    check_form_token = user_data['results']['checkForm']
    user_id = user_data['results']['USER']['USER_ID']
    
    if user_id == 0:
        print("Error: Invalid ARL token or not logged in.")
        return

    # 2. Create the Playlist
    print(f"Creating playlist: {PLAYLIST_NAME}...")
    create_payload = {
        'method': 'playlist.create',
        'api_version': '1.0',
        'api_token': check_form_token
    }
    create_data = {
        'title': PLAYLIST_NAME,
        'songs': []  # We will add tracks in the next step
    }
    
    resp = session.post("https://www.deezer.com/ajax/gw-light.php", params=create_payload, json=create_data)
    playlist_id = resp.json()['results']
    print(f"Created Playlist ID: {playlist_id}")

    # 3. Read IDs from CSV
    track_ids = []
    with open(CSV_FILE, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_ids.append(row['id'])

    # 4. Add tracks to the playlist
    print(f"Adding {len(track_ids)} tracks...")
    add_payload = {
        'method': 'playlist.addSongs',
        'api_version': '1.0',
        'api_token': check_form_token
    }
    add_data = {
        'playlist_id': playlist_id,
        'songs': [[tid, 0] for tid in track_ids]  # Internal API expects [id, offset] pairs
    }
    
    add_resp = session.post("https://www.deezer.com/ajax/gw-light.php", params=add_payload, json=add_data)
    
    if add_resp.json().get('results'):
        print("Success! Check your Deezer account.")
    else:
        print(f"Failed to add tracks: {add_resp.text}")

if __name__ == "__main__":
    create_deezer_playlist()