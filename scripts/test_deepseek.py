#!/usr/bin/env python3
"""
Quick smoke-test for the DeepSeek API.

Usage:
    .venv/bin/python scripts/test_deepseek.py

Requires DEEPSEEK_KEY (or DEEPSEEK_API_KEY) in the environment.
Optionally set DEEPSEEK_BASE_URL (defaults to https://api.deepseek.com).
"""
from __future__ import annotations

import os
import sys
import time

# ── Ensure project root is on sys.path so `import config` works ───────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def main() -> None:
    # ── 1. Check key ───────────────────────────────────────────────────────
    key = config.DEEPSEEK_KEY
    base_url = config.DEEPSEEK_BASE_URL
    model = config.DEVIL_MODEL  # "deepseek-reasoner"

    print("=" * 60)
    print("  DeepSeek API Smoke Test")
    print("=" * 60)
    print(f"  Base URL : {base_url}")
    print(f"  Model    : {model}")
    print(f"  Key      : {'set (' + key[:6] + '…)' if key else 'NOT SET'}")
    print("-" * 60)

    if not key:
        print("\n  ✗ FAIL — No DEEPSEEK_KEY / DEEPSEEK_API_KEY in environment.")
        print("    Set one of those and re-run.")
        sys.exit(1)

    # ── 2. Build client ────────────────────────────────────────────────────
    try:
        from openai import OpenAI
    except ImportError:
        print("\n  ✗ FAIL — `openai` package not installed.")
        print("    Run: pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=key, base_url=base_url)

    # ── 3. Simple completion ───────────────────────────────────────────────
    print("\n  → Sending a short test prompt …")
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Reply in one sentence."},
                {"role": "user", "content": "What is 2+2? Answer briefly."},
            ],
            timeout=config.DEVIL_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  ✗ FAIL — API call errored after {elapsed:.1f}s")
        print(f"    {type(exc).__name__}: {exc}")
        sys.exit(1)

    elapsed = time.time() - t0

    # ── 4. Inspect response ────────────────────────────────────────────────
    msg = resp.choices[0].message
    text = msg.content or ""
    thinking = getattr(msg, "reasoning_content", "") or ""
    usage = getattr(resp, "usage", None)

    print(f"\n  ✓ PASS — Response received in {elapsed:.1f}s\n")
    print(f"  Model      : {resp.model}")
    print(f"  Tokens in  : {getattr(usage, 'prompt_tokens', '?')}")
    print(f"  Tokens out : {getattr(usage, 'completion_tokens', '?')}")
    print(f"  Content    : {text.strip()!r}")

    if thinking:
        # Truncate long chain-of-thought for display
        preview = thinking[:300].replace("\n", " ")
        suffix = " …(truncated)" if len(thinking) > 300 else ""
        print(f"  Reasoning  : {preview}{suffix}")
    else:
        print("  Reasoning  : (empty — model may not expose reasoning_content)")

    print("\n" + "=" * 60)
    print("  DeepSeek API is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
