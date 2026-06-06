# Repo State

## Current status

As reviewed on 2026-06-06, the repo contains two overlapping agent implementations:

- the current package-based agent under `agent/`
- an older top-level `agent.py` flow that reflects an earlier notebook-oriented design

For new development and normal usage, the package-based path should be considered primary.

## What appears healthy

- `python -m agent.main` is a coherent CLI with once, daemon, backtest, and dry-run modes.
- The default behavior is safety-first: `DRY_RUN=true`.
- The main decision path is mostly deterministic and testable.
- Anthropic integration is bounded to health check, claim extraction, analyst, and critique roles.
- The repo already writes useful artifacts under `storage/`.
- The automated test suite currently passes.

Test result from this review:

```text
51 passed in 14.32s
```

## What may confuse a new contributor

- `README.md` mixes older architecture language with newer package-agent behavior.
- `agent.py` and root `config.py` look like runnable entrypoints, but they are not the cleanest current path.
- Some data modules still carry compatibility helpers and demo fallbacks, which is useful operationally but makes the codebase look more ambiguous than it is.

## Architecture summary

The current package agent is:

- scheduler-driven
- deterministic-first
- fallback-tolerant
- artifact-heavy for auditability
- optionally LLM-augmented, but not LLM-authorized

Its strongest internal boundary is that forecasting and risk remain local/deterministic even when Anthropic is enabled.

## Known caveats from code review

- Live-data fallbacks can mask integration problems if you only test without a real `ARENA_KEY`.
- The repo still has legacy files that should be clearly marked or eventually retired.
- Some documentation in the root README predates the current package-centric architecture.

## Suggested next cleanup steps

- Keep new docs and examples centered on `python -m agent.main`.
- Either trim or explicitly mark legacy entrypoints in the root docs.
- Add a small `.env.example` if the team wants onboarding to be less implicit.
- Add one explicit smoke-test section for live arena validation with a real key.
