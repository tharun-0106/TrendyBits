"""TrendyBits FastAPI application.

Run:  cd backend && uvicorn app.main:app --reload --port 8000      (docs at http://localhost:8000/docs)
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__, config, database as db, metrics_store
from .credibility import credibility_aggregate
from .llm_service import OLLAMA, summarize
from .seed_data import TICKER
from .services import get_services
from .xai_explainer import LimeTextExplainer, SHAP_BASELINE, explain_dimensions, global_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("trendybits.api")

JOBS: dict[str, dict] = {}
DEMO_REPORT_ARTICLES = ["tb-010", "tb-023", "tb-008", "tb-025", "tb-021", "tb-001"]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    svc = get_services()
    for aid in DEMO_REPORT_ARTICLES:  # pre-compute public checks so the UI is populated offline
        if db.latest_report_for_article(aid) is None:
            try:
                svc.analyze_article(aid, save=True)
            except Exception:  # pragma: no cover - never block startup
                log.exception("demo report for %s failed", aid)
    yield


app = FastAPI(title="TrendyBits API", version=__version__, lifespan=lifespan,
              description="LLM-powered conversational platform for personalised news (Pipeline A) and independent, "
                          "explainable fake-news & credibility analysis (Pipeline B).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================================ schemas
class SearchReq(BaseModel):
    query: str = ""
    user_id: str = "demo"
    k: int = Field(6, ge=1, le=30)
    topic: str | None = None
    source: str | None = None
    time_window: str | None = None
    use_semantic: bool = True
    use_personalization: bool = True
    use_mmr: bool = True
    mmr_lambda: float | None = Field(None, ge=0, le=1)


class ChatReq(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = "demo"
    use_semantic: bool = True
    use_personalization: bool = True
    use_mmr: bool = True


class AnalyzeReq(BaseModel):
    input_type: Literal["link", "text", "image", "article"] = "text"
    content: str = Field("", max_length=20000)
    article_id: str | None = None
    reasoning_model: str | None = None
    instruction: str | None = Field(None, max_length=2000, description="Optional user note/query (treated as untrusted)")
    image_caption: str | None = None
    lime_samples: int | None = Field(None, ge=50, le=2000)


class SummarizeReq(BaseModel):
    article_id: str | None = None
    text: str | None = None
    title: str | None = None
    source: str | None = None


class UrlReq(BaseModel):
    url: str


class ProfileReq(BaseModel):
    name: str | None = None
    topics: list[str] = []
    sources: list[str] = []
    entities: list[str] = []
    language: str = "en"


class FeedbackReq(BaseModel):
    user_id: str = "demo"
    target_type: Literal["summary", "report", "recommendation", "chat"] = "report"
    target_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""


class XaiReq(BaseModel):
    text: str = ""
    dims: dict[str, float] | None = None
    num_samples: int = Field(300, ge=50, le=2000)


class RunTestsReq(BaseModel):
    suite: Literal["all", "classifiers", "ablation", "llm"] = "all"


# ============================================================================ meta
@app.get("/api/health")
def health():
    svc = get_services()
    return dict(status="ok", version=__version__, articles=len(svc.articles), ollama=OLLAMA.status(),
                retriever=svc.retriever.backend, classifiers=dict(liar=svc.classifiers.liar_backend,
                                                                  integrated=svc.classifiers.int_backend))


@app.get("/api/models")
def models():
    st = OLLAMA.status()
    return dict(summarizer=config.SUMMARIZER_MODEL, reasoner=config.REASONER_MODEL, reasoner_choices=config.REASONER_CHOICES,
                judge=config.JUDGE_MODEL, ollama=st,
                active={m: OLLAMA.has_model(m) for m in [config.SUMMARIZER_MODEL, *config.REASONER_CHOICES, config.JUDGE_MODEL]},
                fallback_engine="deterministic NLP fallback is used for any model that is not pulled / reachable")


@app.get("/api/ticker")
def ticker():
    return {"items": TICKER}


# ============================================================================ Pipeline A
@app.get("/api/articles")
def list_articles(topic: str | None = None, limit: int = Query(40, le=500), q: str | None = None, user_id: str = "demo"):
    svc = get_services()
    if q:
        return {"articles": svc.search(q, user_id=user_id, k=min(limit, 20), topic=topic)}
    arts = db.list_articles(topic=topic, limit=limit)
    out = []
    for a in arts:
        rep = db.latest_report_for_article(a["id"])
        out.append({k: a[k] for k in ("id", "title", "source", "domain", "url", "published_at", "topic", "kind", "author", "demo")}
                   | {"snippet": a["body"][:220], "status": rep["status"] if rep else None,
                      "score": rep["overall_score"] if rep else None})
    return {"articles": out}


@app.get("/api/articles/{article_id}")
def get_article(article_id: str):
    svc = get_services()
    art = db.get_article(article_id)
    if not art:
        raise HTTPException(404, "Article not found")
    return dict(article=art, summary=summarize(art), report=db.latest_report_for_article(article_id),
                related=svc.related(article_id), source=db.get_source(art.get("domain")))


@app.post("/api/search")
def search(req: SearchReq):
    t0 = time.perf_counter()
    res = get_services().search(req.query, user_id=req.user_id, k=req.k, topic=req.topic, source=req.source,
                                time_window=req.time_window, use_semantic=req.use_semantic,
                                use_personalization=req.use_personalization, use_mmr=req.use_mmr, mmr_lambda=req.mmr_lambda)
    return {"results": res, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/api/chat")
def chat(req: ChatReq):
    return get_services().chat(req.message, user_id=req.user_id, use_semantic=req.use_semantic,
                               use_personalization=req.use_personalization, use_mmr=req.use_mmr)


@app.post("/api/summarize")
def summarize_ep(req: SummarizeReq):
    if req.article_id:
        s = get_services().summarize_article(req.article_id)
        if not s:
            raise HTTPException(404, "Article not found")
        return s
    if not req.text:
        raise HTTPException(422, "Provide article_id or text")
    return summarize(dict(id=None, title=req.title or "", body=req.text, source=req.source or "Pasted text"))


# ============================================================================ Pipeline B
@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    svc = get_services()
    kw = dict(reasoning_model=req.reasoning_model, user_instruction=req.instruction)
    if req.lime_samples:
        kw["lime_samples"] = req.lime_samples
    try:
        if req.input_type == "article" or req.article_id:
            rep = svc.analyze_article(req.article_id or req.content.strip(), **kw)
            if rep is None:
                raise HTTPException(404, "Article not found")
            return rep
        if not req.content.strip():
            raise HTTPException(422, "Paste a link, claim text or image caption first")
        if req.input_type == "link":
            return svc.analyze_url(req.content.strip(), **kw)
        if req.input_type == "image":
            caption = req.image_caption or req.content
            rep = svc.analyzer.analyze(title="Image claim", body="", image_caption=caption, input_type="image", **kw)
            rep["image_note"] = ("Image forensics are not performed; the caption/context is checked against evidence "
                                 "(e.g. reused or re-captioned photos).")
            db.save_report(rep)
            return rep
        return svc.analyze_text(req.content, **kw)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/reports/recent")
def recent_reports(limit: int = Query(9, le=50)):
    return {"reports": db.recent_reports(limit)}


@app.get("/api/reports/{report_id}")
def get_report(report_id: str):
    rep = db.get_report(report_id)
    if not rep:
        raise HTTPException(404, "Report not found")
    return rep


@app.get("/api/sources/{domain}")
def source(domain: str):
    s = db.get_source(domain)
    if not s:
        raise HTTPException(404, "Source not in registry")
    return s


# ============================================================================ scraping
@app.post("/api/scrape/rss")
def scrape_rss(fetch_full: bool = False, max_per_feed: int = Query(8, le=30)):
    from .scraper import scrape_rss as _scrape

    stats = _scrape(max_per_feed=max_per_feed, fetch_full=fetch_full)
    if stats["inserted"]:
        get_services().rebuild_index()
    return stats


@app.post("/api/scrape/url")
def scrape_url(req: UrlReq):
    from .scraper import scrape_url as _scrape

    res = _scrape(req.url)
    if not res.get("article_id"):
        raise HTTPException(422, res.get("error", "scrape failed"))
    get_services().rebuild_index()
    return res


# ============================================================================ profile / feedback
@app.get("/api/profile/{user_id}")
def get_profile(user_id: str):
    return {**db.get_profile(user_id), "recent_queries": db.recent_queries(user_id, 10)}


@app.put("/api/profile/{user_id}")
def put_profile(user_id: str, req: ProfileReq):
    return db.save_profile({"user_id": user_id, **req.model_dump()})


@app.post("/api/feedback")
def feedback(req: FeedbackReq):
    return {"id": db.add_feedback(req.user_id, req.target_type, req.target_id, req.rating, req.comment)}


# ============================================================================ XAI
@app.post("/api/xai/explain")
def xai_explain(req: XaiReq):
    """XAI sandbox: LIME on arbitrary text and/or SHAP on user-supplied dimension values."""
    svc = get_services()
    out = {}
    if req.text.strip():
        out["lime"] = LimeTextExplainer(num_samples=req.num_samples).explain(req.text, svc.classifiers.text_credibility)
    if req.dims:
        dims = {k: float(req.dims.get(k, SHAP_BASELINE[k])) for k in SHAP_BASELINE}
        out["shap"] = explain_dimensions(dims, credibility_aggregate)
    elif req.text.strip():
        rep = svc.analyze_text(req.text, save=False, with_xai=False)
        out["shap"] = explain_dimensions(rep["signals"]["dimensions"], credibility_aggregate)
        out["dims"] = rep["signals"]["dimensions"]
    out["baseline"] = SHAP_BASELINE
    return out


@app.get("/api/xai/global")
def xai_global():
    return global_importance(db.all_reports())


# ============================================================================ admin
@app.get("/api/admin/metrics")
def admin_metrics():
    return dict(**metrics_store.get_all_model_metrics(), ablation=metrics_store.get_ablation(),
                llm_tests=metrics_store.get_llm_tests(), eval_history=db.eval_history(),
                security_events=db.security_events(20), feedback=db.feedback_summary(),
                xai_global=global_importance(db.all_reports()))


def _run_job(job_id: str, suite: str) -> None:
    from .classifiers import train_all_fast
    from .evaluation import run_ablation, run_test_suite

    job = JOBS[job_id]

    def progress(msg: str):
        job["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")

    try:
        svc = get_services()
        if suite in ("all", "classifiers"):
            train_all_fast(progress=progress)
            svc.classifiers._load()
        if suite in ("all", "ablation"):
            run_ablation(svc, progress=progress)
        if suite in ("all", "llm"):
            run_test_suite(svc, progress=progress)
        job["status"] = "done"
    except Exception as exc:  # surface the failure in the admin UI
        job["status"] = "error"
        job["error"] = f"{exc.__class__.__name__}: {exc}"
        job["log"].append(traceback.format_exc()[-1500:])
    job["finished_at"] = time.time()


@app.post("/api/admin/run-tests")
def run_tests(req: RunTestsReq):
    if any(j["status"] == "running" for j in JOBS.values()):
        raise HTTPException(409, "An evaluation job is already running")
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = dict(id=job_id, suite=req.suite, status="running", log=[], started_at=time.time())
    threading.Thread(target=_run_job, args=(job_id, req.suite), daemon=True).start()
    return JOBS[job_id]


@app.get("/api/admin/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Unknown job")
    return JOBS[job_id]


@app.get("/api/security/events")
def security_events(limit: int = 50):
    return {"events": db.security_events(limit)}
