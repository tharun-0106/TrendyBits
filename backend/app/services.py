"""Service container: wires the SQLite store, retriever, classifiers, Pipeline A and Pipeline B.

Pipeline A (search / chat / summarise) and Pipeline B (analyse) share article ids and URLs only.
"""
from __future__ import annotations

import threading
from urllib.parse import urlparse

from . import database as db
from .classifiers import get_classifier_service
from .credibility import CredibilityAnalyzer
from .llm_service import extract_intent, summarize
from .retrieval import Retriever
from .security import sanitize_untrusted

CLARIFYING_QUESTION = ("Could you tell me what you're interested in — a topic (e.g. Politics, Economy, Tech), a place, "
                       "or a publisher? Meanwhile, here are a few diverse top stories:")


class Services:
    def __init__(self, init_db: bool = True):
        if init_db:
            db.init_db(seed=True)
        self._lock = threading.RLock()
        self.classifiers = get_classifier_service()
        self.rebuild_index()

    def rebuild_index(self) -> None:
        with self._lock:
            self.articles = db.list_articles(limit=5000)
            self.retriever = Retriever(self.articles)
            self.analyzer = CredibilityAnalyzer(self.retriever, self.classifiers)

    # ------------------------------------------------------------------ Pipeline A
    def search(self, query: str, user_id: str = "demo", k: int = 6, topic: str | None = None, source: str | None = None,
               time_window: str | None = None, entities: list[str] | None = None, use_semantic: bool = True,
               use_personalization: bool = True, use_mmr: bool = True, mmr_lambda: float | None = None,
               kinds: set[str] | None = None) -> list[dict]:
        profile = db.get_profile(user_id)
        return self.retriever.search(query, k=k, topic=topic, source=source, time_window=time_window, entities=entities,
                                     profile=profile, use_semantic=use_semantic, use_personalization=use_personalization,
                                     use_mmr=use_mmr, mmr_lambda=mmr_lambda,
                                     kinds=kinds or {"news", "official", "factcheck", "viral", "blog"})

    def chat(self, message: str, user_id: str = "demo", use_semantic: bool = True, use_personalization: bool = True,
             use_mmr: bool = True, k: int = 5) -> dict:
        intent = extract_intent(message)
        db.log_query(user_id, message, {k_: v for k_, v in intent.items() if k_ != "latency_ms"})
        base = dict(intent=intent, toggles=dict(semantic=use_semantic, personalization=use_personalization, mmr=use_mmr))
        if intent["intent"] == "out_of_scope":
            return dict(base, reply=intent["decline_message"], articles=[], clarifying_question=None)
        if intent["intent"] == "clarify_or_top_stories":
            profile = db.get_profile(user_id)
            sparse = not (profile.get("topics") or profile.get("sources"))
            arts = self.retriever.search("top news stories today", k=k, profile=profile, use_semantic=use_semantic,
                                         use_personalization=use_personalization and not sparse, use_mmr=True, mmr_lambda=0.3,
                                         kinds={"news", "official"})
            return dict(base, reply=CLARIFYING_QUESTION, clarifying_question=CLARIFYING_QUESTION, articles=arts,
                        sparse_profile=sparse)
        if intent["intent"] == "fact_check":
            claim = sanitize_untrusted(message, origin="chat").clean_text
            rep = self.analyze_text(claim, input_type="text", with_xai=False, save=True)
            return dict(base, reply=(f"I sent this to the independent Fact-Check pipeline: {rep['overall_display']}. "
                                     "Open the report for per-claim evidence."), articles=[], report_id=rep["report_id"],
                        report_summary=dict(overall=rep["overall_display"], counts=rep["counts"]), clarifying_question=None)
        query = " ".join(intent.get("keywords") or []) or message
        arts = self.search(query, user_id=user_id, k=k, topic=intent.get("topic"), source=intent.get("source_filter"),
                           time_window=intent.get("time_window"), entities=intent.get("entities"),
                           use_semantic=use_semantic, use_personalization=use_personalization, use_mmr=use_mmr)
        for a in arts[:3]:
            art = db.get_article(a["id"])
            if art:
                s = summarize(art)
                a["summary"] = s["summary"]
                a["attribution"] = s["attribution"]
        bits = [f"topic {intent['topic']}" if intent.get("topic") else "", f"source {intent['source_filter']}" if intent.get("source_filter") else "",
                f"time {intent['time_window']}" if intent.get("time_window") else "",
                ("entities " + ", ".join(intent["entities"])) if intent.get("entities") else ""]
        desc = "; ".join(b for b in bits if b)
        reply = f"Here {'is' if len(arts) == 1 else 'are'} {len(arts)} {'story' if len(arts) == 1 else 'stories'}" + (f" matching {desc}" if desc else "") + ". Summaries describe what each outlet reported."
        return dict(base, reply=reply, articles=arts, clarifying_question=None)

    def summarize_article(self, article_id: str) -> dict | None:
        art = db.get_article(article_id)
        return summarize(art) if art else None

    # ------------------------------------------------------------------ Pipeline B
    def analyze_article(self, article_id: str, reasoning_model: str | None = None, user_instruction: str | None = None,
                        save: bool = True, **kw) -> dict | None:
        art = db.get_article(article_id)
        if not art:
            return None
        rep = self.analyzer.analyze(title=art["title"], body=art["body"], url=art["url"], domain=art.get("domain"),
                                    source_name=art.get("source"), article_id=art["id"], image_caption=art.get("image_caption"),
                                    published_at=art.get("published_at"), input_type="article", reasoning_model=reasoning_model,
                                    user_instruction=user_instruction, extra_timeline=(art.get("extra") or {}).get("timeline"), **kw)
        if save:
            db.save_report(rep)
        return rep

    def analyze_text(self, text: str, title: str | None = None, input_type: str = "text", reasoning_model: str | None = None,
                     user_instruction: str | None = None, save: bool = True, **kw) -> dict:
        lines = [l for l in text.strip().splitlines() if l.strip()]
        t = title or (lines[0][:140] if lines else text[:140])
        body = text if title else ("\n".join(lines[1:]) if len(lines) > 1 and len(lines[0]) < 160 else text)
        rep = self.analyzer.analyze(title=t, body=body, input_type=input_type, reasoning_model=reasoning_model,
                                    user_instruction=user_instruction, image_caption=None, **kw)
        if save:
            db.save_report(rep)
        return rep

    def analyze_url(self, url: str, reasoning_model: str | None = None, user_instruction: str | None = None, **kw) -> dict:
        art = db.get_article_by_url(url)
        if not art:
            from .scraper import scrape_url

            res = scrape_url(url)
            if not res.get("article_id"):
                raise ValueError(res.get("error") or "Could not fetch article")
            self.rebuild_index()
            art = db.get_article(res["article_id"])
        return self.analyze_article(art["id"], reasoning_model=reasoning_model, user_instruction=user_instruction, **kw)

    def related(self, article_id: str, k: int = 3) -> list[dict]:
        out = []
        for r in self.retriever.similar(article_id, k=k):
            rep = db.latest_report_for_article(r["id"])
            r["status"] = (rep["status"] if rep else "Unreviewed")
            r["score"] = rep["overall_score"] if rep else None
            out.append(r)
        return out


_SERVICES: Services | None = None
_LOCK = threading.Lock()


def get_services() -> Services:
    global _SERVICES
    with _LOCK:
        if _SERVICES is None:
            _SERVICES = Services()
    return _SERVICES


def domain_of(url: str | None) -> str | None:
    host = urlparse(url).hostname if url else None
    return host.removeprefix("www.") if host else None
