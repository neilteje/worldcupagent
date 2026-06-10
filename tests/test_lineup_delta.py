from models.lineup_delta import evaluate_lineup_delta

def p(id, pos, **kw): return {'id':id,'name':id,'position':pos,**kw}
BASE=[p('gk','goalkeeper'),p('st','striker',star=True),p('cb','center back'),p('dm','defensive midfielder')]

def test_missing_starting_goalkeeper():
    r=evaluate_lineup_delta(BASE, BASE[1:], BASE, BASE)
    assert r['lineup_shock'] and r['probability_delta']['home'] < 0

def test_missing_striker():
    r=evaluate_lineup_delta(BASE, [BASE[0],BASE[2],BASE[3]], BASE, BASE)
    assert 'st' in r['home_missing_expected_starters'] and r['probability_delta']['away'] > 0

def test_both_teams_missing_similar_players():
    r=evaluate_lineup_delta(BASE, BASE[1:], BASE, BASE[1:])
    assert abs(r['probability_delta']['home']) < .01

def test_no_confirmed_lineup():
    r=evaluate_lineup_delta(BASE, None, BASE, BASE)
    assert 'lineup_unconfirmed' in r['risk_flags']

def test_formation_change_only():
    r=evaluate_lineup_delta(BASE, BASE, BASE, BASE, {'home':'4-3-3','away':'4-3-3'}, {'home':'3-5-2','away':'4-3-3'})
    assert r['formation_change']['home'] and r['lineup_shock']
