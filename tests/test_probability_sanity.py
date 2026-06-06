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

def test_risk_audit_exact_confidence_boundary():
    risk=audit_decision({'home':.5,'draw':.28,'away':.22},{'edge_tier':'medium','edge_type':'model_only_edge','best_edge':.07},.55,.2,False,True)
    assert 'confidence_too_low' not in risk['risk_flags']

def test_risk_audit_adds_critic_suggested_flags():
    risk=audit_decision({'home':.5,'draw':.28,'away':.22},{'edge_tier':'soft','edge_type':'model_only_edge','best_edge':.04},.7,.2,False,True, consensus_case='all_disagree')
    assert 'confidence_insufficient_for_soft_edge' in risk['risk_flags']
    assert 'source_disagreement_unresolved' in risk['risk_flags']
    assert 'multi_source_conflict' in risk['risk_flags']

def test_risk_audit_blocks_edge_tier_confidence_mismatch():
    risk=audit_decision({'home':.5,'draw':.28,'away':.22},{'edge_tier':'strong','edge_type':'model_only_edge','best_edge':.12},.6,.2,False,True)
    assert 'edge_tier_confidence_mismatch' in risk['blocking_risk_flags']

def test_ledger_dag_parent_structure():
    lb=LedgerBuilder('F','PRE_MATCH',load_settings(True))
    recs=lb.build_standard_trace(prediction={'fixture_code':'F','window':'PRE_MATCH','probabilities':{'home':.4,'draw':.3,'away':.3},'confidence':.6})
    ids={r['record_id'] for r in recs}
    assert lb.validate_dag() and all(pid in ids for r in recs for pid in r.get('parent_ids',[]))

def test_prediction_payload_validates():
    payload={'home':.4,'draw':.3,'away':.3}
    assert valid_probs(payload)
