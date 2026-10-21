# TrendyBits

**LLM-powered conversational platform for personalised news and fake-news analysis.**

TrendyBits draws on the FND-LLM framework, which pairs specialised BERT/SLM classifier signals with LLM reasoning. It's built as two strictly independent pipelines:

| Pipeline | What it does | Models |
|---|---|---|
| **A: Personalised retrieval & faithful summarisation** | Intent extraction, live RSS/URL scraping, Sentence-BERT + FAISS search with personalisation and MMR diversity, and chunked summaries that keep entities, numbers and hedges | `qwen:4b` (Ollama) with a deterministic fallback |
| **B: Independent credibility analysis** | Claim decomposition, RAG evidence retrieval, BERT LIAR (6-class) and ISOT+Bharat (binary) signals, and evidence reasoning with conflict and recency weighting. Output is a multi-dimensional report with LIME + SHAP explanations | `deepseek-r1:8b` / `gemma2:9b` (Ollama), BERT/TF-IDF classifiers, deterministic NLI fallback |

The two pipelines share only article IDs and URLs. **The summariser never says whether an article is true or false.** Pipeline B shows graded reports instead of binary Real/Fake labels, and every citation in a report must come from the evidence it retrieved.

> **About the shipped numbers.** The metrics in `data/*.json` and the manual were produced by running this repository's scripts on **synthetic, schema-faithful benchmark partitions** and a **26-article synthetic demo corpus** (marked `DEMO SEED` in the UI), using the **offline fallback engine** (no Ollama, TF-IDF+LSA instead of SBERT, TF-IDF+LR instead of fine-tuned BERT). To get publishable figures, import the IEEE Dataport files, start Ollama and re-run `python scripts/run_all.py`.

## Quick start (about 2 minutes, CPU only, works offline)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python scripts/run_all.py                                 # datasets → classifiers → XAI checks → ablation → 18 tests
cd backend && uvicorn app.main:app --reload --port 8000   # API + docs at http://localhost:8000/docs
```
In a second terminal:
```bash
cd frontend && npm install && npm run dev                 # UI at http://localhost:5173
```

### Optional: real LLMs and deep-learning models
```bash
ollama pull qwen:4b && ollama pull deepseek-r1:8b && ollama pull mistral:7b
python scripts/check_ollama.py
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements-ml.txt
python scripts/train_bert_classifier.py --mode bert --epochs 3     # fine-tunes bert-base-uncased for both datasets
```
The backend detects Ollama, SBERT, FAISS and fine-tuned BERT automatically and records which back-end produced every metric.

### Docker
```bash
docker compose up --build                     # API :8000, UI :5173 (uses Ollama on the host if it's running)
docker compose --profile ollama up --build    # also starts Ollama and pulls the three models
```

## IEEE Dataport datasets
* News Summarization: https://ieee-dataport.org/documents/trendybits-news-summarization-dataset (`Source, Content, Summary`)
* Fake News & Credibility: https://ieee-dataport.org/documents/trendybits-fake-news-dataset (LIAR 14 columns, 6 labels; Integrated ISOT + BharatFakeNewsKosh `date, title, subject, text, label`)

```bash
python scripts/import_ieee_dataport.py --newssumm NewsSumm.xlsx
python scripts/import_ieee_dataport.py --liar train.tsv valid.tsv test.tsv
python scripts/import_ieee_dataport.py --isot-true True.csv --isot-fake Fake.csv --bharat bharat_en.csv
```
The import pipeline runs these steps in order:
1. Audit: deduplicate, drop rows with missing fields, keep English rows only, normalise labels.
2. Split into stratified 80/10/10 partitions before any fitting.
3. Tune on the validation split only.
4. Fingerprint the test split with SHA-256.

LIAR and Integrated are trained and reported separately.

## Evaluation
| Command | Output |
|---|---|
| `python scripts/train_bert_classifier.py --mode fast\|bert\|both` | `data/model_metrics.json`: accuracy, macro/per-class P/R/F1, ROC-AUC, confusion matrices |
| `python scripts/verify_xai.py` | `data/xai_verification.json`: SHAP efficiency/null-player, LIME fidelity, deletion tests, determinism |
| `python scripts/run_ablation_study.py` | `data/ablation_metrics.json`: Section V Table 1 (complete system plus Models 1–6) |
| `python scripts/run_llm_test_suite.py` | `data/llm_test_metrics.json`: TC-01…TC-18, Tables 2–5 |
| `pytest -q tests` | unit, API and DeepEval-style suites |
| Admin UI ▶️ button / `POST /api/admin/run-tests` | re-runs all of the above from the browser |

Results from the latest offline run: **18 of 18 test cases pass**, the **42 pytest tests pass** (1 skipped because it needs Ollama + deepeval), and all XAI checks hold.

## Explainable AI
* **LIME** (`backend/app/xai_explainer.py`) masks interpretable units (money amounts, cue phrases, entities, words) and also tries leaving each one out. It weights the perturbed samples with an exponential cosine kernel (σ = 0.25) and fits a weighted ridge surrogate. The result is signed weights wᵢ ∈ [−1, 1], shown as red and green token pills.
* **SHAP** computes exact Shapley values over 5 dimensions: source reliability, RAG evidence match, the LIAR signal, the ISOT-Bharat signal and a framing penalty. It enumerates all 32 coalitions against the prior (50, 50, 50, 50, 0), so the contributions always add up exactly: baseline + Σφ = final score.

## Repository map
```
backend/app/   main.py · services.py · llm_service.py · retrieval.py · embeddings.py · scraper.py · credibility.py
               security.py · classifiers.py · xai_explainer.py · metrics_store.py · judge.py · evaluation.py
               datasets.py · database.py · seed_data.py · text_utils.py · config.py
frontend/src/  App.jsx · styles.css · components/ (Header, ReportCard, XaiPanel, TelemetryStrip, ConfusionMatrix)
               pages/ (Home, FactCheck, ArticlePage, Admin, Chatbot, Profile)
scripts/       import_ieee_dataport · train_bert_classifier · verify_xai · run_ablation_study · run_llm_test_suite
               check_ollama · run_all · build_manual
tests/         test_pipelines · test_api · test_deepeval_suite
docs/          TrendyBits_App_Manual_and_Guide.docx (+ figures, sample run log)
```

The full manual (architecture, XAI maths, model metrics, UI guide and every command) is in `docs/TrendyBits_App_Manual_and_Guide.docx`.
