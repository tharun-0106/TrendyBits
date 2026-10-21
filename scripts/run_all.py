#!/usr/bin/env python
"""One-shot pipeline: datasets → classifiers → XAI verification → ablation → 18-case suite → seeded reports.

  python scripts/run_all.py            # everything (≈1–2 min on a laptop CPU in fast/fallback mode)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ["scripts/import_ieee_dataport.py", "--generate"],
    ["scripts/train_bert_classifier.py", "--mode", "fast"],
    ["scripts/verify_xai.py"],
    ["scripts/run_ablation_study.py"],
    ["scripts/run_llm_test_suite.py"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n{'=' * 100}\n$ python {' '.join(step)}\n{'=' * 100}", flush=True)
        rc = subprocess.call([sys.executable, *step], cwd=ROOT)
        if rc not in (0,):
            print(f"step failed with exit code {rc}")
            sys.exit(rc)
    print("\nAll steps finished. Start the API with:  cd backend && uvicorn app.main:app --port 8000")
