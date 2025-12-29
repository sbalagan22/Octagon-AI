import requests
import json
import os
import unicodedata

API_KEY = "2e165978c393355b8218845cb6798ede"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_name(name):
    if not name:
        return ""
    # Remove accents/diacritics
    nfkd_form = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return name.lower().replace("'", "").replace("-", " ").strip()

def fetch_mma_odds():
    print("Fetching odds from The Odds API...")
    url = f"https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
    params = {
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "apiKey": API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Successfully fetched odds for {len(data)} fights.")
        return data
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return []

def update_events_with_odds():
    odds_data = fetch_mma_odds()
    if not odds_data:
        return

    # Load currently predicted events
    file_path = os.path.join(BASE_DIR, '..', 'dashboard', 'public', 'upcoming_events.json')
    try:
        with open(file_path, 'r') as f:
            events = json.load(f)
    except Exception as e:
        print(f"Could not load {file_path}: {e}")
        return

    # Create a mapping for quick lookup
    odds_map = {}
    for fight in odds_data:
        home = normalize_name(fight.get('home_team'))
        away = normalize_name(fight.get('away_team'))
        
        # Find outcomes
        for bm in fight.get('bookmakers', []):
            if bm['key'] in ['draftkings', 'fanduel', 'betonlineag']: # Prefer these
                for market in bm.get('markets', []):
                    if market['key'] == 'h2h':
                        outcomes = {normalize_name(o['name']): o['price'] for o in market['outcomes']}
                        odds_map[(home, away)] = outcomes
                        odds_map[(away, home)] = outcomes
                        break
                if (home, away) in odds_map: break

    # Update events
    updated_count = 0
    for event in events:
        for fight in event.get('fights', []):
            f1 = normalize_name(fight['fighter_1'])
            f2 = normalize_name(fight['fighter_2'])
            
            # Look for exact or fuzzy match
            matched_odds = None
            for (h, a), outcomes in odds_map.items():
                if (f1 in h or h in f1) and (f2 in a or a in f2):
                    matched_odds = outcomes
                    break
            
            if matched_odds:
                fight['market_odds'] = {
                    fight['fighter_1']: matched_odds.get(f1, "N/A"),
                    fight['fighter_2']: matched_odds.get(f2, "N/A")
                }
                # Try to get the actual name from outcomes if normalization lost it
                for name, price in matched_odds.items():
                    if f1 in name or name in f1:
                        fight['market_odds'][fight['fighter_1']] = price
                    if f2 in name or name in f2:
                        fight['market_odds'][fight['fighter_2']] = price
                updated_count += 1

    print(f"Updated {updated_count} fights with market odds.")

    # Save back
    with open(file_path, 'w') as f:
        json.dump(events, f, indent=2)
    
    # Also update the root file
    root_path = os.path.join(BASE_DIR, '..', 'upcoming_events.json')
    with open(root_path, 'w') as f:
        json.dump(events, f, indent=2)

if __name__ == "__main__":
    update_events_with_odds()
