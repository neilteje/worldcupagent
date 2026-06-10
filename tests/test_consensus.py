from models.consensus import consensus_triangle

def test_all_agree():
    assert consensus_triangle({'home':.5,'draw':.25,'away':.25},{'home':.48,'draw':.27,'away':.25},{'home':.47,'draw':.28,'away':.25})['case']=='all_agree'

def test_model_bookmaker_vs_polymarket():
    r=consensus_triangle({'home':.52,'draw':.25,'away':.23},{'home':.49,'draw':.27,'away':.24},{'away':.45,'draw':.28,'home':.27})
    assert r['case']=='model_bookmaker_vs_polymarket' and r['confidence_modifier']>0

def test_model_polymarket_vs_bookmaker():
    assert consensus_triangle({'home':.5,'draw':.25,'away':.25},{'away':.47,'draw':.28,'home':.25},{'home':.48,'draw':.27,'away':.25})['case']=='model_polymarket_vs_bookmaker'

def test_bookmaker_polymarket_vs_model():
    assert consensus_triangle({'away':.5,'draw':.25,'home':.25},{'home':.48,'draw':.27,'away':.25},{'home':.47,'draw':.28,'away':.25})['case']=='bookmaker_polymarket_vs_model'

def test_missing_bookmaker():
    assert consensus_triangle({'home':.5,'draw':.25,'away':.25},None,{'home':.47,'draw':.28,'away':.25})['case']=='missing_bookmaker'
