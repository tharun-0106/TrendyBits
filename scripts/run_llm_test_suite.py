#!/usr/bin/env python
"""Section VI — 18 LLM test cases (TC-01 … TC-18) across L1 component, L2 RAG, L3 safety/robustness, L4 end-to-end.

Uses Ollama models when reachable (qwen:4b summariser, deepseek-r1:8b reasoner, mistral:7b judge via the
DeepEval-compatible CustomOllamaJudge); otherwise the deterministic fallback engine and scorers. The engine
used is recorded in the JSON output.

Results → data/llm_test_metrics.json ; prints Tables 2, 3, 4 and the Table 5 result-reporting template.

  python scripts/run_llm_test_suite.py
  TRENDYBITS_USE_OLLAMA=0 python scripts/run_llm_test_suite.py     # force the offline fallback engine
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.datasets import ensure_datasets  # noqa: E402
from app.evaluation import format_table, run_test_suite  # noqa: E402
from app.metrics_store import FILES  # noqa: E402
from app.services import get_services  # noqa: E402

CASE_COLS = [("id", "TC"), ("component", "Component"), ("name", "Scenario"), ("metric", "Metric"), ("target", "Target"),
             ("measured", "Measured"), ("status", "Status")]


def main() -> None:
    ensure_datasets()
    res = run_test_suite(get_services())
    titles = {"table2_L1_component": "TABLE 2 — L1 COMPONENT TESTS (TC-01 … TC-07)",
              "table3_L2_rag": "TABLE 3 — L2 RAG / EVIDENCE-REVIEW TESTS (TC-08 … TC-12)",
              "table4_L3_L4_safety_e2e": "TABLE 4 — L3 SAFETY & ROBUSTNESS + L4 END-TO-END (TC-13 … TC-18)"}
    for key, title in titles.items():
        print(f"\n{title}\n")
        print(format_table(res["tables"][key], CASE_COLS))
    print("\nTABLE 5 — RESULT-REPORTING TEMPLATE\n")
    print(format_table(res["table5"], [("metric", "Metric"), ("target", "Target"), ("measured", "Measured"), ("status", "Status")]))
    print(f"\nCitation traceability (TC-18): {res['citation_traceability']}")
    print(f"Passed {res['passed']}/{res['total']} test cases • engine: {res['engine']}")
    print(f"Saved → {FILES['llm_tests']}  ({res['elapsed_seconds']} s)")
    sys.exit(0 if res["passed"] == res["total"] else 1)


if __name__ == "__main__":
    main()
