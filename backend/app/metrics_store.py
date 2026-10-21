"""Module 6b — Multi-model performance telemetry store.

Merges the JSON telemetry written by the training / evaluation scripts into one view of the
5 executing models, served by /api/admin/metrics and embedded (as benchmark badges) in every
/api/analyze response.

  data/model_metrics.json           ← scripts/train_bert_classifier.py   (Models 3, 4)
  data/pipeline_model_metrics.json  ← scripts/run_ablation_study.py      (Models 1, 2, 5 — complete system)
                                       + scripts/run_llm_test_suite.py   (Model 2 slot F1, Model 5 safety/consistency)
  data/ablation_metrics.json        ← Section V, Table 1
  data/llm_test_metrics.json        ← Section VI, Tables 2–5
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import config

FILES = {
    "classifiers": config.DATA_DIR / "model_metrics.json",
    "pipeline": config.DATA_DIR / "pipeline_model_metrics.json",
    "ablation": config.DATA_DIR / "ablation_metrics.json",
    "llm_tests": config.DATA_DIR / "llm_test_metrics.json",
}

MODEL_CARDS = [
    dict(id=1, key="retriever", name="Sentence-BERT all-MiniLM-L6-v2 + FAISS + MMR Retriever", role="Pipeline A — retrieval",
         llm=None),
    dict(id=2, key="summarizer", name="Qwen 4B Summarizer & Conversational Intent Extractor", role="Pipeline A — summarisation",
         llm=config.SUMMARIZER_MODEL),
    dict(id=3, key="liar", name="BERT-LIAR-6Class Supervised Claim Classifier", role="Pipeline B — classifier signal", llm=None),
    dict(id=4, key="integrated", name="BERT-Integrated-Binary (ISOT + BharatFakeNewsKosh) Article Classifier",
         role="Pipeline B — classifier signal", llm=None),
    dict(id=5, key="reasoner", name="DeepSeek-R1 8B / Gemma2 9B Fake-News Evidence Reasoner", role="Pipeline B — evidence reasoning",
         llm=config.REASONER_MODEL),
]


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(name: str, payload: dict) -> Path:
    path = FILES[name]
    payload = {**payload, "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat()}
    path.write_text(json.dumps(payload, indent=2))
    return path


def update_pipeline_metrics(section: str, values: dict) -> None:
    data = _read(FILES["pipeline"])
    data.setdefault(section, {}).update(values)
    write_json("pipeline", data)


def get_all_model_metrics() -> dict:
    clf = _read(FILES["classifiers"])
    pipe = _read(FILES["pipeline"])
    models = []
    for card in MODEL_CARDS:
        if card["key"] in ("liar", "integrated"):
            section = clf.get(card["key"], {})
            metrics = section.get("primary", {})
            extra = {"baselines": section.get("baselines", {})}
        else:
            metrics = pipe.get(card["key"], {})
            extra = {}
        models.append({**card, "metrics": metrics, "backend": metrics.get("backend", "not evaluated yet"), **extra,
                       "evaluated": bool(metrics)})
    return {"models": models, "files": {k: str(v.name) for k, v in FILES.items()},
            "available": {k: v.exists() for k, v in FILES.items()},
            "classifier_generated_at": clf.get("generated_at"), "pipeline_generated_at": pipe.get("generated_at")}


def benchmark_summary(key: str) -> dict:
    """Compact benchmark badge for the analyzer telemetry strip."""
    m = next((x["metrics"] for x in get_all_model_metrics()["models"] if x["key"] == key), {})
    if not m:
        return {}
    picks = {
        "retriever": [("P@5", "precision_at_k"), ("MRR", "mrr"), ("ILD", "ild")],
        "summarizer": [("ROUGE-L", "rougeL"), ("Faithfulness", "faithfulness"), ("Slot F1", "intent_slot_f1")],
        "liar": [("Macro F1", "macro_f1"), ("Acc", "accuracy")],
        "integrated": [("F1", "f1"), ("ROC-AUC", "roc_auc")],
        "reasoner": [("Verdict Acc", "verdict_accuracy"), ("Halluc.", "hallucination_rate"), ("Citation", "citation_validity")],
    }[key]
    return {label: m.get(field) for label, field in picks if m.get(field) is not None}


def get_ablation() -> dict:
    return _read(FILES["ablation"])


def get_llm_tests() -> dict:
    return _read(FILES["llm_tests"])
