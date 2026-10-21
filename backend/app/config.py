"""Central configuration for TrendyBits.

Every value can be overridden with an environment variable so the same code runs
locally, inside Docker, and inside CI without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repository root (…/TrendyBits). backend/app/config.py -> parents[2]
BASE_DIR = Path(os.getenv("TRENDYBITS_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.getenv("TRENDYBITS_DATA_DIR", BASE_DIR / "data"))
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = Path(os.getenv("TRENDYBITS_MODELS_DIR", BASE_DIR / "models"))
DB_PATH = Path(os.getenv("TRENDYBITS_DB", DATA_DIR / "trendybits.db"))

for _d in (DATA_DIR, DATASETS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Ollama / LLMs
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Pipeline A — summarisation + conversational intent extraction (small, fast).
SUMMARIZER_MODEL = os.getenv("TRENDYBITS_SUMMARIZER_MODEL", "qwen:4b")
# Pipeline B — evidence reasoning + credibility analysis (8B–9B reasoning model).
REASONER_MODEL = os.getenv("TRENDYBITS_REASONER_MODEL", "deepseek-r1:8b")
REASONER_CHOICES = ["deepseek-r1:8b", "gemma2:9b", "llama3.1:8b"]
# LLM-as-a-judge for DeepEval / RAGAs style scoring.
JUDGE_MODEL = os.getenv("TRENDYBITS_JUDGE_MODEL", "mistral:7b")
# auto = use Ollama when reachable, else deterministic fallback; 1 = force; 0 = never.
USE_OLLAMA = os.getenv("TRENDYBITS_USE_OLLAMA", "auto").lower()
OLLAMA_TIMEOUT = float(os.getenv("TRENDYBITS_OLLAMA_TIMEOUT", "120"))

# ---------------------------------------------------------------- Embeddings
EMBEDDING_MODEL = os.getenv("TRENDYBITS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# auto = try sentence-transformers then fall back to TF-IDF+LSA; 0 = always fallback.
USE_SBERT = os.getenv("TRENDYBITS_USE_SBERT", "auto").lower()

# ---------------------------------------------------------------- Pipeline knobs
SUMMARY_CHUNK_WORDS = int(os.getenv("TRENDYBITS_CHUNK_WORDS", "350"))  # fallback context window
LLM_CHUNK_WORDS = int(os.getenv("TRENDYBITS_LLM_CHUNK_WORDS", "1400"))  # qwen context budget
MMR_LAMBDA = float(os.getenv("TRENDYBITS_MMR_LAMBDA", "0.7"))
PUBLISHER_PENALTY = float(os.getenv("TRENDYBITS_PUBLISHER_PENALTY", "0.15"))
LIME_SAMPLES = int(os.getenv("TRENDYBITS_LIME_SAMPLES", "300"))
SEED = int(os.getenv("TRENDYBITS_SEED", "42"))

# A canary string embedded in every system prompt. If it ever appears in an output
# the system prompt has leaked (checked by TC-13 / TC-14).
SYSTEM_CANARY = "TB-SYS-CANARY-7f3a"

TOPICS = ["Tech", "Economy", "Politics", "Society", "World", "Sports", "Opinion", "Entertainment"]
