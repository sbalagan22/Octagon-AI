
import requests
from bs4 import BeautifulSoup
import json
import re
import time
from datetime import datetime

# Headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
}

def clean_text(text):
    if not text:
        return ""
    return text.strip()

def get_soup(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def scrape_event_details(event_url, event_id):
    print(f"  Scraping details from {event_url}...")
    soup = get_soup(event_url)
    if not soup:
        return []

    fights = []
    # Strategy 1: Smart JSON Extraction (Hidden Data Source)
    # This is much more reliable than DOM scraping for React/Next.js pages
    # like ESPN's which hydrate from a JSON state.
    
    scripts = soup.find_all('script')
    fights_js_found = False
    
    for script in scripts:
        if script.string and "window['__espnfitt__']=" in script.string:
            try:
                content = script.string
                parts = content.split("window['__espnfitt__']=")
                if len(parts) > 1:
                    json_str = parts[1].strip()
                    if json_str.endswith(';'):
                        json_str = json_str[:-1]
                    
                    data = json.loads(json_str)
                    gp = data.get('page', {}).get('content', {}).get('gamepackage', {})
                    
                    if 'cardSegs' in gp:
                        print("    Found embedded JSON data structure.")
                        segs = gp['cardSegs']
                        for seg in segs:
                            is_main_card_seg = (seg.get('nm') == 'main')
                            matches = seg.get('mtchs', [])
                            
                            for m in matches:
                                # Fighters
                                awy = m.get('awy', {})
                                hme = m.get('hme', {})
                                
                                n1 = awy.get('dspNm')
                                n2 = hme.get('dspNm')
                                
                                if not n1 or not n2: continue # Skip invalid
                                
                                u1 = awy.get('lnk', '')
                                u2 = hme.get('lnk', '')
                                
                                # Ensure URLs are absolute
                                if u1 and not u1.startswith('http'): u1 = "https://www.espn.com" + u1
                                if u2 and not u2.startswith('http'): u2 = "https://www.espn.com" + u2
                                
                                # Title Fight Check
                                note = m.get('nte', '')
                                is_title = False
                                if note and "Title Fight" in note:
                                    is_title = True
                                    
                                fights.append({
                                    "fighter_1": n1,
                                    "fighter_2": n2,
                                    "fighter_1_url": u1,
                                    "fighter_2_url": u2,
                                    "is_main_card": is_main_card_seg,
                                    "is_title_fight": is_title
                                })
                                fights_js_found = True
                                
                    if fights_js_found:
                         break
            except Exception as e:
                print(f"    Error parsing embedded JSON: {e}")
                
    if fights_js_found:
        print(f"    Extracted {len(fights)} fights from JSON.")
        return fights

    # Strategy 2: DOM Scraping (Fallback)
    print("    JSON data not found, falling back to DOM scraping.")
    
    # Sequential Strategy:
    # Iterate through Headers and Fight Rows in document order.
    # This handles both nested and flat structures.
    
    # Select Headers AND Fight Containers
    # Note: Sometimes fights are in 'li.AccordionPanel', sometimes 'div.MMAGamestrip'
    # elements = soup.select('h3.Card__Header__Title, div.MMAGamestrip, li.AccordionPanel')
    # Use a broad selector to catch the flow
    elements = soup.select('h3.Card__Header__Title, div.MMAGamestrip, li.AccordionPanel, h3.Card__Header')
    
    print(f"  Found {len(elements)} structural elements")
    
    is_main_card = True # Default start
    
    # Track processed fights to avoid duplicates if both AccordionPanel and Gamestrip match (unlikely with select distinctness but careful)
    # Actually, AccordionPanel usually Contains Gamestrip. 
    # If we select BOTH, we might double count.
    # Let's prefer AccordionPanel if present, else Gamestrip.
    # To avoid double counting, we can check if an element is a descendant of one we just processed?
    # Or just select the top-level container: 'li.AccordionPanel'
    # If no panels, select 'div.MMAGamestrip'.
    
    # Refined Selector Strategy:
    panels = soup.select('li.AccordionPanel')
    if not panels:
        panels = soup.select('div.MMAGamestrip')
        # If still empty, try MMAFightCard
        if not panels:
            panels = soup.select('div.MMAFightCard')
    
    # Now get Headers + Panels in one list for ordering
    # We can't easily merge two soup lists while preserving DOM order without complex sorting.
    # Easier: Iterate ALL elements and filter?
    # Or: find all 'h3' and 'li/div' and sort by position? 
    # Simple approach: Loop through all 'h3' and 'div' that match class.
    
    all_nodes = soup.find_all(['h3', 'li', 'div'])
    relevant_nodes = []
    for node in all_nodes:
        classes = node.get('class', [])
        if 'Card__Header__Title' in classes or 'Card__Header' in classes:
            relevant_nodes.append(('header', node))
        elif 'AccordionPanel' in classes:
            relevant_nodes.append(('fight', node))
        elif 'MMAGamestrip' in classes:
             # Only add if we haven't already decided on AccordionPanel strategy or if this is not inside a panel?
             # For simplicity, if we found Panels earlier, ignore standalone Gamestrips to avoid dupes?
             if not panels or panels[0].name != 'li': 
                 relevant_nodes.append(('fight', node))
                 
    # Deduplicate: If we have nested matches (unlikely with this logic but possible)
    unique_nodes = []
    seen = set()
    for type_, node in relevant_nodes:
        if node in seen: continue
        seen.add(node)
        unique_nodes.append((type_, node))
        
    print(f"  Processing {len(unique_nodes)} sequential nodes")
    
    for type_, node in unique_nodes:
        if type_ == 'header':
            text = node.get_text(strip=True)
            print(f"    Header: {text}")
            if "Preliminary" in text or "Prelims" in text:
                is_main_card = False
            elif "Main Card" in text:
                is_main_card = True
            continue
            
        # It's a fight container
        competitors = node.find_all('div', class_='MMACompetitor')
        if len(competitors) < 2: continue
        
        # Take first 2
        c1 = competitors[0]
        c2 = competitors[1]
        
        n1 = c1.find('h2').get_text(strip=True) if c1.find('h2') else c1.get_text(strip=True)
        n2 = c2.find('h2').get_text(strip=True) if c2.find('h2') else c2.get_text(strip=True)
        
        n1 = re.sub(r'\d+-\d+-\d+$', '', n1).strip()
        n2 = re.sub(r'\d+-\d+-\d+$', '', n2).strip()
        
        print(f"    Processing Fight: {n1} vs {n2} (MainCard: {is_main_card})")

        # URLs & Champion Status
        u1 = ""
        u2 = ""
        
        # Strategy 1: Data Player UID (Most reliable for static scrape if attrs present)
        # Search in the whole node
        uids = node.find_all(attrs={'data-player-uid': True})
        if len(uids) >= 1:
            u1 = f"https://www.espn.com/mma/fighter/_/id/{uids[0]['data-player-uid']}"
        if len(uids) >= 2:
            u2 = f"https://www.espn.com/mma/fighter/_/id/{uids[1]['data-player-uid']}"
            
        # Strategy 2: Headshot Images (Fallback)
        if not u1 or not u2:
            imgs = node.select('img[src*="headshots/mma/players/full"]')
            
            if len(imgs) >= 1 and not u1:
                m = re.search(r'/(\d+)\.png', imgs[0]['src'])
                if m: u1 = f"https://www.espn.com/mma/fighter/_/id/{m.group(1)}"
            if len(imgs) >= 2 and not u2:
                m = re.search(r'/(\d+)\.png', imgs[1]['src'])
                if m: u2 = f"https://www.espn.com/mma/fighter/_/id/{m.group(1)}"
            
        # Strategy 3: Profile Links (Fallback)
        if not u1: 
             l = c1.find('a', href=lambda h: h and '/fighter/' in h)
             if l: u1 = l['href']
        if not u2:
             l = c2.find('a', href=lambda h: h and '/fighter/' in h)
             if l: u2 = l['href']
             
        if u1 and not u1.startswith('http'): u1 = "https://www.espn.com" + u1
        if u2 and not u2.startswith('http'): u2 = "https://www.espn.com" + u2
        
        # Title Fight Check
        is_title = False
        note = node.find(['h2', 'div', 'span'], class_='MMAFightCard__GameNote')
        if note:
            note_text = note.get_text(strip=True)
            # print(f"      GameNote: {note_text}")
            if "Title Fight" in note_text:
                is_title = True
        
        if node.select('img[alt="Championship Belt"]'):
            is_title = True 
            
        fights.append({
            "fighter_1": n1,
            "fighter_2": n2,
            "fighter_1_url": u1,
            "fighter_2_url": u2,
            "is_main_card": is_main_card,
            "is_title_fight": is_title
        })

    return fights

def scrape_upcoming_events():
    today = datetime.now()
    print(f"Current Date: {today.strftime('%Y-%m-%d')}")
    
    # URLs to scrape: Current year + Next year
    current_year = today.year
    next_year = current_year + 1
    
    urls = [
        (f"https://www.espn.com/mma/schedule/_/year/{current_year}/league/ufc", current_year),
        (f"https://www.espn.com/mma/schedule/_/year/{next_year}/league/ufc", next_year)
    ]
    
    events = []
    seen_ids = set()
    
    for base_url, year in urls:
        print(f"Fetching schedule from {base_url}...")
        soup = get_soup(base_url)
        if not soup:
            continue
            
        tables = soup.find_all('table', class_='Table')
        
        for table in tables:
            rows = table.find_all('tr', class_='Table__TR')
            
            for row in rows:
                event_col = row.find('td', class_='event__col')
                if not event_col:
                    continue
                    
                link = event_col.find('a')
                if not link:
                    continue
                    
                event_name = link.get_text(strip=True)
                event_url = link.get('href', '')
                
                match = re.search(r'/id/(\d+)', event_url)
                event_id = match.group(1) if match else str(int(time.time()))
                
                if event_id in seen_ids:
                    continue
                
                if event_url.startswith('/'):
                    event_url = "https://www.espn.com" + event_url
                    
                date_col = row.find('td', class_='date__col')
                date_text = date_col.get_text(strip=True) if date_col else "TBD"
                
                # Parse date to check if upcoming
                # date_text example: "Sat, Dec 6" or "Dec 6"
                try:
                    # Remove day name if present
                    clean_date = re.sub(r'^[A-Za-z]+,\s*', '', date_text)
                    event_date_obj = datetime.strptime(f"{clean_date} {year}", "%b %d %Y")
                    
                    if event_date_obj < today.replace(hour=0, minute=0, second=0, microsecond=0):
                        continue
                except:
                    # If parsing fails, skip or keep? Let's keep if we can't parse to be safe, or skip
                    # Usually skipping is safer to avoid garbage
                    continue
                
                # Location
                loc_col = row.find('td', class_='location__col')
                location = loc_col.get_text(strip=True) if loc_col else "TBD"
                
                print(f"Found Upcoming Event: {event_name} ({date_text}, {year})")
                seen_ids.add(event_id)
                
                fights = scrape_event_details(event_url, event_id)
                
                if fights:
                    events.append({
                        "event_id": event_id,
                        "event_name": event_name,
                        "date": date_text,
                        "location": location,
                        "url": event_url,
                        "fights": fights
                    })
                    
                time.sleep(1)

    
    # Save to JSON in root, using absolute path resolution
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, '..', 'upcoming_events.json')
    
    with open(output_path, 'w') as f:
        json.dump(events, f, indent=2)
    
    print(f"Saved {len(events)} upcoming events to {output_path}")

if __name__ == "__main__":
    scrape_upcoming_events()
