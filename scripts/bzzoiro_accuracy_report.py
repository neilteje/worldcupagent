import os
import sys
import json
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

BZZOIRO_KEY = config.BZZOIRO_KEY
if not BZZOIRO_KEY:
    print("Error: BZZOIRO_KEY not found in config.")
    sys.exit(1)

HEADERS = {"Authorization": f"Token {BZZOIRO_KEY}"}
BASE_URL = config.BZZOIRO_BASE

def get_recent_past_predictions(limit=500):
    url = f"{BASE_URL}/api/v2/predictions/?limit=200&status=past"
    predictions = []
    
    while url and len(predictions) < limit:
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            break
        data = res.json()
        results = data.get('results', [])
        if not results:
            break
        
        predictions.extend(results)
        url = data.get('next')
        
    return predictions[:limit]

def main():
    print("Evaluating Bzzoiro API Predictions Accuracy for recently completed matches (All Leagues)...\n")
    
    preds = get_recent_past_predictions(limit=500)
    
    # Only consider predictions for matches that have a final score
    finished_preds = []
    for p in preds:
        event = p.get('event', {})
        if event.get('status') == 'finished' and event.get('home_score') is not None and event.get('away_score') is not None:
            finished_preds.append(p)
            
    print(f"Fetched {len(preds)} past predictions.")
    print(f"Found {len(finished_preds)} completed matches with scores.\n")
    
    total_evaluated = 0
    correct_predictions = 0
    
    for p in finished_preds:
        event = p['event']
        home_score = event.get('home_score')
        away_score = event.get('away_score')
        
        actual_outcome = "H" if home_score > away_score else ("A" if away_score > home_score else "D")
        
        # Prediction accuracy
        pred_outcome = None
        if 'markets' in p and 'match_result' in p['markets']:
            pred_outcome = p['markets']['match_result'].get('predicted')
        
        if pred_outcome:
            is_correct = (pred_outcome == actual_outcome)
            # Print just the first few for brevity, but tally all of them
            if total_evaluated < 10:
                league_name = event.get('league_name', 'Unknown')
                match_str = f"[{league_name}] {event.get('home_team')} vs {event.get('away_team')}"
                print(f"{match_str[:50]:<50} | Res: {home_score}-{away_score} ({actual_outcome}) | Pred: {pred_outcome} -> {'CORRECT' if is_correct else 'INCORRECT'}")
            elif total_evaluated == 10:
                print("... (truncating individual match output) ...")
                
            total_evaluated += 1
            if is_correct:
                correct_predictions += 1

    if total_evaluated > 0:
        accuracy = (correct_predictions / total_evaluated) * 100
        print(f"\nOverall Prediction Accuracy on recent matches: {correct_predictions}/{total_evaluated} ({accuracy:.2f}%)")
    else:
        print("\nNo finished matches with predictions were found to evaluate.")

if __name__ == '__main__':
    main()
