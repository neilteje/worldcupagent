"""
Paper-trading harness — a live dress rehearsal for the World Cup.

This package simulates exactly what the agent will do during the tournament, but
against arbitrary fixtures (e.g. tomorrow's friendlies). At each match's kickoff
and half-time windows it produces ONE shared prediction and lets multiple
differently-tuned agents (see `profiles.py`) paper-trade it. Nothing is ever sent
to the arena — the broker is a demo "calc sheet" that fills against real
Polymarket mids when available and a clearly-labeled synthetic reference when not.

Entry point: ``python -m harness ...`` (see `runner.py`).
"""
