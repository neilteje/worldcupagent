from models.calibration import normalize_probs, valid_probs
from models.probability import pre_match_model
from models.sanity_checks import audit_decision
from reasoning.ledger_builder import LedgerBuilder
from agent.config import load_settings


def test_probability_normalization():
    p=normalize_probs({'home':2,'draw':1,'away':1})
    assert abs(sum(p.values())-1)<1e-9 and p['home']==.5

def test_pre_match_valid():
    r=pre_match_model({'home':.5,'draw':.25,'away':.25},{'home':.48,'draw':.27,'away':.25},{'home':.45,'draw':.30,'away':.25},None,None,.8)
    assert valid_probs(r['probabilities'])

def test_no_bet_when_dry_run():
    risk=audit_decision({'home':.5,'draw':.28,'away':.22},{'edge_tier':'strong','edge_type':'model_only_edge'},.7,.2,True,True)
    assert 'dry_run_enabled' in risk['blocking_risk_flags'] and not risk['order_allowed']

def test_ledger_dag_parent_structure():
    lb=LedgerBuilder('F','PRE_MATCH',load_settings(True))
    recs=lb.build_standard_trace(prediction={'fixture_code':'F','window':'PRE_MATCH','probabilities':{'home':.4,'draw':.3,'away':.3},'confidence':.6})
    ids={r['record_id'] for r in recs}
    assert lb.validate_dag() and all(pid in ids for r in recs for pid in r.get('parent_ids',[]))

def test_prediction_payload_validates():
    payload={'home':.4,'draw':.3,'away':.3}
    assert valid_probs(payload)
