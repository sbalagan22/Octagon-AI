
import pandas as pd
import os
import unicodedata

def normalize_name(name):
    if not name: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(name))
    name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return name.lower().replace("'", "").replace("-", " ").replace(".", "").strip()

def repair_fights():
    print("Repairing Fights.csv (Fixing ID/Name Mappings with Duplicate Handling)...")
    
    data_dir = "newdata"
    fighters_path = os.path.join(data_dir, 'Fighters.csv')
    fights_path = os.path.join(data_dir, 'Fights.csv')
    
    # 1. Load Fighters to build Name -> ID map with experience logic
    fighters_df = pd.read_csv(fighters_path)
    name_to_id = {}
    fighter_exp = {}
    
    for _, row in fighters_df.iterrows():
        n = normalize_name(row['Full Name'])
        fid = row['Fighter_Id']
        # Metric for "Better" match: total fights
        exp = int(row.get('W', 0)) + int(row.get('L', 0)) + int(row.get('D', 0))
        
        if n not in name_to_id or exp > fighter_exp.get(n, -1):
            name_to_id[n] = fid
            fighter_exp[n] = exp
    
    # 2. Load Fights
    fights_df = pd.read_csv(fights_path)
    
    # 3. Repair IDs based on Names
    fixed_count = 0
    missing_fighters = set()
    
    for idx, row in fights_df.iterrows():
        n1 = normalize_name(row['Fighter_1'])
        n2 = normalize_name(row['Fighter_2'])
        
        id1_real = name_to_id.get(n1)
        id2_real = name_to_id.get(n2)
        
        if id1_real and str(row['Fighter_Id_1']) != str(id1_real):
            fights_df.at[idx, 'Fighter_Id_1'] = id1_real
            fixed_count += 1
        
        if id2_real and str(row['Fighter_Id_2']) != str(id2_real):
            fights_df.at[idx, 'Fighter_Id_2'] = id2_real
            fixed_count += 1
            
        if not id1_real: missing_fighters.add(row['Fighter_1'])
        if not id2_real: missing_fighters.add(row['Fighter_2'])

    print(f"Fixed {fixed_count} ID mismatches (including duplicate name corrections).")
    if missing_fighters:
        print(f"Note: {len(missing_fighters)} fighters not found in Fighters.csv.")

    # 4. Save
    fights_df.to_csv(fights_path, index=False)
    print("Repaired Fights.csv saved.")

if __name__ == "__main__":
    repair_fights()
