import os
import sys
import json
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

BZZOIRO_KEY = config.BZZOIRO_KEY
HEADERS = {"Authorization": f"Token {BZZOIRO_KEY}"}
BASE_URL = config.BZZOIRO_BASE

def get_predictions(league_id):
    url = f"{BASE_URL}/api/v2/predictions/?league_id={league_id}&limit=200"
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()
    return res.json().get('results', [])

def main():
    preds = get_predictions(27)
    print(f"Found {len(preds)} predictions.")
    for p in preds[:2]:
        print(json.dumps(p, indent=2))
        
if __name__ == '__main__':
    main()
