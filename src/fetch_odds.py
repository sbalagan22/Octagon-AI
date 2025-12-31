import requests
import json
import os
import unicodedata

API_KEY = "2e165978c393355b8218845cb6798ede"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_name(name):
    if not name: return ""
    import re
    # Convert to string and handle accents
    name = str(name)
    nfkd_form = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    # Standardize Saint/St and handle dashes
    name = name.lower().replace("-", " ")
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    name = name.replace(" saint ", " st ").replace(" saint", " st").replace("saint ", "st ")
    return " ".join(name.split())

def is_match(n1, n2):
    n1_norm = normalize_name(n1)
    n2_norm = normalize_name(n2)
    if not n1_norm or not n2_norm: return False
    
    # Simple common first name mappings
    mappings = {
        "dan": "daniel", "daniel": "dan",
        "jim": "james", "james": "jim",
        "mike": "michael", "michael": "mike",
        "paddy": "patrick", "patrick": "paddy"
    }
    
    # Exact match
    if n1_norm == n2_norm: return True
    
    # Token sets
    t1 = n1_norm.split()
    t2 = n2_norm.split()
    
    # Name Reversal check (e.g. "Rong Zhu" vs "Zhu Rong")
    if set(t1) == set(t2) and len(t1) >= 2:
        return True
        
    # Mapping check for first name
    if len(t1) >= 2 and len(t2) >= 2:
        # Check if first names are mapped and last names match
        if mappings.get(t1[0]) == t2[0] and t1[-1] == t2[-1]:
            return True
        if mappings.get(t2[0]) == t1[0] and t1[-1] == t2[-1]:
            return True

    # Substring / Token overlap
    if n1_norm in n2_norm or n2_norm in n1_norm: return True
    
    return False

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
        for fight in data:
            print(f"DEBUG API FIGHT: {fight.get('home_team')} vs {fight.get('away_team')}")
        return data
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return []

def update_events_with_odds():
    odds_data = fetch_mma_odds()
    if not odds_data:
        return

    # Load currently predicted events
    file_path = os.path.join(BASE_DIR, '..', 'upcoming_events.json')
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
        
        # Prefer specific bookmakers, but take any if not available
        preferred_keys = ['draftkings', 'fanduel', 'betonlineag', 'williamhill', 'betfair']
        best_bm = None
        
        # First pass: preferred
        for bm in fight.get('bookmakers', []):
            if bm['key'] in preferred_keys:
                best_bm = bm
                break
        
        # Second pass: any
        if not best_bm and fight.get('bookmakers'):
            best_bm = fight['bookmakers'][0]
            
        if best_bm:
            for market in best_bm.get('markets', []):
                if market['key'] == 'h2h':
                    outcomes = {normalize_name(o['name']): o['price'] for o in market['outcomes']}
                    odds_map[(home, away)] = outcomes
                    odds_map[(away, home)] = outcomes
                    break

    # Update events
    updated_count = 0
    for event in events:
        for fight in event.get('fights', []):
            f1 = fight['fighter_1']
            f2 = fight['fighter_2']
            
            # Look for exact or fuzzy match
            matched_odds = None
            for (h, a), outcomes in odds_map.items():
                if (is_match(f1, h) and is_match(f2, a)) or (is_match(f1, a) and is_match(f2, h)):
                    matched_odds = outcomes
                    break
            
            if matched_odds:
                f1_price = "N/A"
                f2_price = "N/A"
                for name_raw, price in matched_odds.items():
                    if is_match(f1, name_raw): f1_price = price
                    if is_match(f2, name_raw): f2_price = price
                
                fight['market_odds'] = {
                    f1: f1_price,
                    f2: f2_price
                }
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
