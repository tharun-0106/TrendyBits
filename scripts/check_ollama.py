#!/usr/bin/env python
"""LLM runner smoke test: checks Ollama, lists pulled models and runs one prompt per configured model.

  ollama pull qwen:4b && ollama pull deepseek-r1:8b && ollama pull mistral:7b
  python scripts/check_ollama.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config  # noqa: E402
from app.llm_service import OLLAMA, extract_intent  # noqa: E402

st = OLLAMA.status()
print(f"Ollama at {st['url']}: {'REACHABLE' if st['reachable'] else 'NOT reachable'} (mode={st['mode']})")
print("Pulled models:", ", ".join(st["models"]) or "none")
for role, model in [("Pipeline A summariser/intent", config.SUMMARIZER_MODEL), ("Pipeline B reasoner", config.REASONER_MODEL),
                    ("Judge", config.JUDGE_MODEL)]:
    ok = OLLAMA.has_model(model)
    print(f"\n[{role}] {model}: {'available' if ok else 'missing → deterministic fallback will be used'}")
    if ok:
        t0 = time.time()
        out = OLLAMA.generate(model, 'Reply with the JSON {"ok": true}', json_mode=True)
        print(f"  response in {time.time() - t0:.1f}s: {out!r}"[:200])
    else:
        print(f"  run:  ollama pull {model}")
print("\nIntent extraction check:", extract_intent("Show me what The Hindu wrote about monsoon floods in Kerala this week."))
