from models.edge_engine import evaluate_edge

def test_edge_thresholds():
    assert evaluate_edge('f','PRE_MATCH',{'home':.45,'draw':.30,'away':.25},{'home':.43,'draw':.31,'away':.26},None,.7,.2,'missing_bookmaker')['edge_tier']=='none'
    assert evaluate_edge('f','PRE_MATCH',{'home':.49,'draw':.27,'away':.24},{'home':.45,'draw':.30,'away':.25},{'home':.48,'draw':.28,'away':.24},.8,.2,'model_bookmaker_vs_polymarket')['edge_tier']=='soft'
    assert evaluate_edge('f','PRE_MATCH',{'home':.53,'draw':.25,'away':.22},{'home':.45,'draw':.30,'away':.25},{'home':.51,'draw':.27,'away':.22},.7,.2,'model_bookmaker_vs_polymarket')['edge_tier']=='medium'
    assert evaluate_edge('f','PRE_MATCH',{'home':.57,'draw':.23,'away':.20},{'home':.45,'draw':.30,'away':.25},{'home':.55,'draw':.25,'away':.20},.7,.2,'model_bookmaker_vs_polymarket')['edge_tier']=='strong'

def test_no_bet_low_confidence_uncertainty_and_against():
    assert not evaluate_edge('f','PRE_MATCH',{'home':.57,'draw':.23,'away':.20},{'home':.45,'draw':.30,'away':.25},{'home':.55,'draw':.25,'away':.20},.5,.2,'model_bookmaker_vs_polymarket')['should_bet']
    assert not evaluate_edge('f','PRE_MATCH',{'home':.57,'draw':.23,'away':.20},{'home':.45,'draw':.30,'away':.25},{'home':.55,'draw':.25,'away':.20},.7,.5,'model_bookmaker_vs_polymarket')['should_bet']
    assert not evaluate_edge('f','PRE_MATCH',{'home':.50,'draw':.24,'away':.26},{'away':.42,'draw':.30,'home':.28},{'away':.43,'draw':.29,'home':.28},.7,.2,'bookmaker_polymarket_vs_model')['should_bet']
