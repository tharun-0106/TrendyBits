#!/usr/bin/env python
"""Section V — Table 1: ablation study across the complete system and Models 1–6.

Every configuration is evaluated on IDENTICAL held-out samples:
  * retrieval benchmark (data/datasets/retrieval_benchmark.json)  → Precision@5, Recall@5, MRR, ILD, unique-source ratio
  * 24-claim evidence benchmark (seed evidence corpus)            → verdict accuracy, hallucination rate, citation validity
  * gold credibility bands for demo articles                      → band accuracy, score MAE, misleading-framing recall
  * NewsSumm test split                                           → ROUGE-1/2/L, factual coverage

Results → data/ablation_metrics.json (+ complete-system metrics for Models 1, 2, 5 → data/pipeline_model_metrics.json)

  python scripts/run_ablation_study.py [--summaries 70]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.datasets import ensure_datasets  # noqa: E402
from app.evaluation import format_table, run_ablation  # noqa: E402
from app.metrics_store import FILES  # noqa: E402
from app.services import get_services  # noqa: E402

COLS = [("configuration", "Configuration"), ("precision_at_5", "P@5"), ("recall_at_5", "R@5"), ("mrr", "MRR"),
        ("ild", "ILD"), ("unique_source_ratio", "UniqSrc"), ("verdict_accuracy", "VerdictAcc"),
        ("hallucination_rate", "Halluc"), ("citation_validity", "CiteValid"), ("credibility_band_accuracy", "CredBand"),
        ("misleading_recall", "MislRecall"), ("rouge1", "R-1"), ("rouge2", "R-2"), ("rougeL", "R-L"),
        ("factual_coverage", "FactCov")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", type=int, default=70, help="number of NewsSumm test articles for ROUGE")
    a = ap.parse_args()
    ensure_datasets()
    res = run_ablation(get_services(), n_summaries=a.summaries)
    print("\nSECTION V — TABLE 1: ABLATION STUDY (identical held-out samples)\n")
    print(format_table(res["rows"], COLS))
    print("\nBackends:", res["backends"])
    print("Samples :", res["samples"])
    for n in res["notes"]:
        print(" -", n)
    print(f"\nSaved → {FILES['ablation']}  ({res['elapsed_seconds']} s)")


if __name__ == "__main__":
    main()
