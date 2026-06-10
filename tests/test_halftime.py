from models.halftime import evaluate_halftime

def test_favorite_losing_but_dominates_xg():
    r=evaluate_halftime({'home':.6,'draw':.25,'away':.15}, score={'home_goals':0,'away_goals':1}, xg={'home_xg':1.2,'away_xg':.2})
    assert r['ht_label']=='lucky_lead' and r['ht_probs']['home'] > .45

def test_leader_also_dominates_xg():
    assert evaluate_halftime({'home':.45,'draw':.28,'away':.27}, score={'home_goals':1,'away_goals':0}, xg={'home_xg':1.1,'away_xg':.3})['ht_label']=='deserved_lead'

def test_0_0_low_xg():
    r=evaluate_halftime({'home':.45,'draw':.28,'away':.27}, score={'home_goals':0,'away_goals':0}, xg={'home_xg':.2,'away_xg':.2})
    assert r['ht_label']=='dead_match' and r['ht_probs']['draw'] > .38

def test_red_card_to_leading_team():
    r=evaluate_halftime({'home':.45,'draw':.28,'away':.27}, score={'home_goals':1,'away_goals':0}, xg={'home_xg':.6,'away_xg':.5}, cards={'home_red':1})
    assert r['ht_label']=='red_card_distortion'

def test_missing_xg_fallback():
    r=evaluate_halftime({'home':.45,'draw':.28,'away':.27}, score={'home_goals':0,'away_goals':0}, xg=None)
    assert r['ht_label'] in {'data_insufficient','red_card_distortion'} and abs(sum(r['ht_probs'].values())-1)<.001
