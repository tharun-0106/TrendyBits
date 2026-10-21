"""Shared pytest fixtures. Tests use a throw-away SQLite DB so they never touch data/trendybits.db."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("TRENDYBITS_DB", str(Path(tempfile.mkdtemp(prefix="tb_test_")) / "test.db"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def svc():
    from app.datasets import ensure_datasets
    from app.services import get_services

    ensure_datasets()
    return get_services()
