from reasoning.anthropic_review import SONNET_MODELS, _extract_thinking, _model_candidates, _parse_json, anthropic_key_status


class DummySettings:
    anthropic_key = "x" * 12


def test_parse_json_from_text_wrapper():
    assert _parse_json("Here: {\"ok\": true, \"n\": 1}") == {"ok": True, "n": 1}


def test_anthropic_key_status_is_non_secret():
    status = anthropic_key_status(DummySettings())
    assert status == {"present": True, "length": 12}


def test_sonnet_model_candidates_prefer_sonnet():
    candidates = _model_candidates(SONNET_MODELS)
    assert candidates
    assert "sonnet" in candidates[0]


def test_extract_thinking_blocks():
    body = {"content": [{"type": "thinking", "thinking": "summary"}, {"type": "text", "text": "{}"}]}
    assert _extract_thinking(body) == "summary"
