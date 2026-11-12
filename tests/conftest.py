"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_ledger(monkeypatch):
    """Tests never touch the real cost ledger in data/. test_budget.py turns it back on, pointed at
    a temporary file."""
    monkeypatch.setenv("EDGAR_LEDGER", "off")
