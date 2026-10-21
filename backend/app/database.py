"""SQLite persistence layer (trendybits.db).

Tables: articles, sources, users, queries, reports, model_eval_logs, feedback, security_events.
A short-lived connection is opened per call so the module is safe under FastAPI's thread pool.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .seed_data import ARTICLES, DEFAULT_USERS, SOURCES
from .text_utils import content_hash, detect_language

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL, source TEXT, domain TEXT,
    url TEXT UNIQUE, published_at TEXT, topic TEXT, author TEXT, kind TEXT DEFAULT 'news',
    image_caption TEXT, language TEXT DEFAULT 'en', content_hash TEXT UNIQUE, extra TEXT DEFAULT '{}',
    demo INTEGER DEFAULT 0, created_at TEXT
);
CREATE TABLE IF NOT EXISTS sources (
    domain TEXT PRIMARY KEY, name TEXT, trust_score REAL, domain_age_years REAL, masthead_verified INTEGER,
    corrections_policy INTEGER, owner TEXT, contact TEXT, corrections_record TEXT, history TEXT
);
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY, name TEXT, topics TEXT, sources TEXT, entities TEXT, language TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, query TEXT, intent TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY, article_id TEXT, input_type TEXT, input_text TEXT, report TEXT,
    overall_score REAL, label TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS model_eval_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, suite TEXT, payload TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, target_type TEXT, target_id TEXT, rating INTEGER,
    comment TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS security_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, origin TEXT, detail TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_topic ON articles(topic);
CREATE INDEX IF NOT EXISTS idx_reports_article ON reports(article_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def get_conn(db_path: Path | None = None):
    conn = sqlite3.connect(str(db_path or config.DB_PATH), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(seed: bool = True) -> None:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        if seed:
            for s in SOURCES:
                conn.execute(
                    "INSERT OR IGNORE INTO sources VALUES (:domain,:name,:trust_score,:domain_age_years,"
                    ":masthead_verified,:corrections_policy,:owner,:contact,:corrections_record,:history)", s)
            for u in DEFAULT_USERS:
                conn.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?,?,?,?)",
                             (u["user_id"], u["name"], json.dumps(u["topics"]), json.dumps(u["sources"]),
                              json.dumps(u["entities"]), u["language"], now_iso()))
    if seed:
        for a in ARTICLES:
            upsert_article({**a, "demo": 1})


def _row_to_article(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    d["extra"] = json.loads(d.get("extra") or "{}")
    d["demo"] = bool(d.get("demo"))
    return d


def upsert_article(a: dict[str, Any]) -> tuple[str, bool]:
    """Insert an article, de-duplicating on URL and normalised content hash.

    Returns (article_id, inserted?)."""
    h = content_hash(a["title"], a["body"])
    with get_conn() as conn:
        dup = conn.execute("SELECT id FROM articles WHERE url=? OR content_hash=?", (a.get("url"), h)).fetchone()
        if dup:
            return dup["id"], False
        aid = a.get("id") or f"art-{h[:10]}"
        conn.execute(
            "INSERT INTO articles (id,title,body,source,domain,url,published_at,topic,author,kind,image_caption,"
            "language,content_hash,extra,demo,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, a["title"], a["body"], a.get("source"), a.get("domain"), a.get("url"), a.get("published_at"),
             a.get("topic"), a.get("author"), a.get("kind", "news"), a.get("image_caption"),
             a.get("language") or detect_language(a["body"]), h, json.dumps(a.get("extra") or {}),
             int(a.get("demo", 0)), now_iso()))
        return aid, True


def list_articles(topic: str | None = None, limit: int = 200, kinds: list[str] | None = None) -> list[dict]:
    q, args = "SELECT * FROM articles WHERE 1=1", []
    if topic:
        q += " AND topic=?"
        args.append(topic)
    if kinds:
        q += f" AND kind IN ({','.join('?' * len(kinds))})"
        args.extend(kinds)
    q += " ORDER BY published_at DESC LIMIT ?"
    args.append(limit)
    with get_conn() as conn:
        return [_row_to_article(r) for r in conn.execute(q, args).fetchall()]


def get_article(article_id: str) -> dict | None:
    with get_conn() as conn:
        r = conn.execute("SELECT * FROM articles WHERE id=?", (article_id,)).fetchone()
        return _row_to_article(r) if r else None


def get_article_by_url(url: str) -> dict | None:
    with get_conn() as conn:
        r = conn.execute("SELECT * FROM articles WHERE url=?", (url,)).fetchone()
        return _row_to_article(r) if r else None


def get_source(domain: str | None) -> dict | None:
    if not domain:
        return None
    domain = domain.lower().removeprefix("www.")
    with get_conn() as conn:
        r = conn.execute("SELECT * FROM sources WHERE domain=?", (domain,)).fetchone()
        if not r and domain.count(".") > 1:  # sub-domain → registrable domain
            r = conn.execute("SELECT * FROM sources WHERE domain=?", (".".join(domain.split(".")[-2:]),)).fetchone()
        return dict(r) if r else None


def list_sources() -> list[dict]:
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM sources ORDER BY trust_score DESC").fetchall()]


# ------------------------------------------------------------------ users / queries / feedback
def get_profile(user_id: str) -> dict:
    with get_conn() as conn:
        r = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    if not r:
        return dict(user_id=user_id, name=user_id, topics=[], sources=[], entities=[], language="en")
    d = dict(r)
    for k in ("topics", "sources", "entities"):
        d[k] = json.loads(d[k] or "[]")
    return d


def save_profile(p: dict) -> dict:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (user_id,name,topics,sources,entities,language,created_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET name=excluded.name, topics=excluded.topics, sources=excluded.sources,"
            " entities=excluded.entities, language=excluded.language",
            (p["user_id"], p.get("name") or p["user_id"], json.dumps(p.get("topics", [])),
             json.dumps(p.get("sources", [])), json.dumps(p.get("entities", [])), p.get("language", "en"), now_iso()))
    return get_profile(p["user_id"])


def log_query(user_id: str, query: str, intent: dict) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO queries (user_id,query,intent,created_at) VALUES (?,?,?,?)",
                     (user_id, query, json.dumps(intent), now_iso()))


def recent_queries(user_id: str, limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM queries WHERE user_id=? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    return [{**dict(r), "intent": json.loads(r["intent"])} for r in rows]


def add_feedback(user_id: str, target_type: str, target_id: str, rating: int, comment: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO feedback (user_id,target_type,target_id,rating,comment,created_at) "
                           "VALUES (?,?,?,?,?,?)", (user_id, target_type, target_id, rating, comment, now_iso()))
        return cur.lastrowid


def feedback_summary() -> dict:
    with get_conn() as conn:
        rows = conn.execute("SELECT target_type, COUNT(*) n, AVG(rating) avg FROM feedback GROUP BY target_type").fetchall()
    return {r["target_type"]: {"count": r["n"], "avg_rating": round(r["avg"] or 0, 2)} for r in rows}


# ------------------------------------------------------------------ reports / logs
def save_report(report: dict) -> None:
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO reports VALUES (?,?,?,?,?,?,?,?)",
                     (report["report_id"], report.get("article_id"), report.get("input_type"),
                      (report.get("input_text") or "")[:4000], json.dumps(report), report["overall_score"],
                      report["overall_label"], now_iso()))


def get_report(report_id: str) -> dict | None:
    with get_conn() as conn:
        r = conn.execute("SELECT report FROM reports WHERE id=?", (report_id,)).fetchone()
    return json.loads(r["report"]) if r else None


def latest_report_for_article(article_id: str) -> dict | None:
    with get_conn() as conn:
        r = conn.execute("SELECT report FROM reports WHERE article_id=? ORDER BY created_at DESC LIMIT 1",
                         (article_id,)).fetchone()
    return json.loads(r["report"]) if r else None


def recent_reports(limit: int = 9) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, article_id, input_type, input_text, overall_score, label, created_at, report "
                            "FROM reports ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        rep = json.loads(r["report"])
        out.append(dict(report_id=r["id"], article_id=r["article_id"], input_type=r["input_type"],
                        title=rep.get("title") or (r["input_text"] or "")[:90], overall_score=r["overall_score"],
                        label=r["label"], created_at=r["created_at"],
                        claim_count=len(rep.get("claim_results", []))))
    return out


def all_reports(limit: int = 200) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT report FROM reports ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [json.loads(r["report"]) for r in rows]


def log_eval(suite: str, payload: dict) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO model_eval_logs (suite,payload,created_at) VALUES (?,?,?)",
                     (suite, json.dumps(payload), now_iso()))


def eval_history(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT id, suite, created_at FROM model_eval_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def log_security(kind: str, origin: str, detail: str) -> None:
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO security_events (kind,origin,detail,created_at) VALUES (?,?,?,?)",
                         (kind, origin, detail[:500], now_iso()))
    except sqlite3.Error:
        pass


def security_events(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM security_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
