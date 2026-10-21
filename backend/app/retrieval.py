"""Module 2 — Semantic search (Sentence-BERT + FAISS) with personalisation and MMR re-ranking.

Relevance for document D_i given query Q and user profile P:

    rel(D_i) = cos(e_Q, e_Di) + 0.15·BM25norm(Q, D_i)                      (hybrid semantic score)
             + 0.12·[topic ∈ P.topics] + 0.08·[source ∈ P.sources]        (personalisation boost)
             + 0.05·Σ[entity ∈ D_i] + 0.05·exp(-age_days / 14)            (entity + recency boost)

Diversity-aware re-ranking (Maximal Marginal Relevance) with a same-publisher penalty:

    MMR = argmax_{D_i ∈ R\\S} [ λ·Sim1(D_i, Q) − (1−λ)·max_{D_j ∈ S} Sim2(D_i, D_j) − γ·|{D_j ∈ S : pub(D_j)=pub(D_i)}| ]
"""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime
from typing import Any

import numpy as np

from . import config
from .embeddings import Embedder, VectorIndex
from .text_utils import content_tokens

TIME_WINDOWS = {"today": 1, "yesterday": 2, "this week": 7, "last week": 14, "this month": 31, "recent": 21}


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.N, dtype="float32")
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")[:19]).date()
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def _mentions(blob: str, entity: str) -> bool:
    """Whole-word, case-insensitive entity match ("RBI" must not match "orbit")."""
    return re.search(r"(?<![a-z0-9])" + re.escape(entity.lower()) + r"(?![a-z0-9])", blob) is not None


def doc_text(a: dict) -> str:
    return f"{a.get('title', '')}. {a.get('title', '')}. {a.get('body', '')[:2500]}"


class Retriever:
    def __init__(self, articles: list[dict], embedder: Embedder | None = None, prefer_sbert: bool = True):
        self.articles = list(articles)
        self.embedder = embedder or Embedder(prefer_sbert=prefer_sbert)
        texts = [doc_text(a) for a in self.articles]
        self.embedder.fit(texts)
        self.vectors = self.embedder.encode(texts) if texts else np.zeros((0, 8), dtype="float32")
        self.index = VectorIndex(self.vectors) if len(self.articles) else None
        self.bm25 = BM25([content_tokens(t) for t in texts])
        self.id_to_idx = {a["id"]: i for i, a in enumerate(self.articles)}

    @property
    def backend(self) -> str:
        return f"{self.embedder.backend} + {self.index.backend if self.index else 'empty'}"

    # ---------------------------------------------------------------- core scoring
    def relevance(self, query: str, use_semantic: bool = True) -> np.ndarray:
        bm = self.bm25.scores(content_tokens(query))
        bm_n = bm / bm.max() if bm.max() > 0 else bm
        if not use_semantic:
            return bm_n
        qv = self.embedder.encode([query])[0]
        return self.index.all_scores(qv) + 0.15 * bm_n

    def search(self, query: str, k: int = 5, topic: str | None = None, source: str | None = None,
               time_window: str | None = None, entities: list[str] | None = None, profile: dict | None = None,
               use_semantic: bool = True, use_personalization: bool = True, use_mmr: bool = True,
               mmr_lambda: float | None = None, publisher_penalty: float | None = None, pool: int = 30,
               now: date | None = None, exclude_ids: set[str] | None = None, kinds: set[str] | None = None) -> list[dict]:
        if not self.articles:
            return []
        lam = config.MMR_LAMBDA if mmr_lambda is None else mmr_lambda
        gamma = config.PUBLISHER_PENALTY if publisher_penalty is None else publisher_penalty
        now = now or date.today()
        rel = self.relevance(query or " ".join(entities or []) or "news", use_semantic).astype("float64")
        reasons: list[list[str]] = [[] for _ in self.articles]

        mask = np.ones(len(self.articles), dtype=bool)
        if exclude_ids:
            mask &= np.array([a["id"] not in exclude_ids for a in self.articles])
        if kinds:
            mask &= np.array([a.get("kind", "news") in kinds for a in self.articles])

        def hard_filter(cond: np.ndarray, label: str):
            nonlocal mask
            if (mask & cond).sum() >= 1:
                mask &= cond
            else:  # nothing matches -> degrade gracefully to a soft boost
                rel[cond] += 0.1
                for i in np.where(cond)[0]:
                    reasons[i].append(f"soft {label}")

        if topic:
            hard_filter(np.array([a.get("topic") == topic for a in self.articles]), "topic")
        if source:
            s = source.lower()
            hard_filter(np.array([s in (a.get("source") or "").lower() or s in (a.get("domain") or "")
                                  for a in self.articles]), "source")
        if time_window and time_window in TIME_WINDOWS:
            days = TIME_WINDOWS[time_window]
            ages = [self._age(a, now) for a in self.articles]
            hard_filter(np.array([age is not None and age <= days for age in ages]), "time window")

        for i, a in enumerate(self.articles):
            blob = f"{a.get('title', '')} {a.get('body', '')}".lower()
            for e in entities or []:
                if _mentions(blob, e):
                    rel[i] += 0.05
                    reasons[i].append(f"entity:{e}")
            age = self._age(a, now)
            if age is not None:
                rel[i] += 0.05 * math.exp(-max(age, 0) / 14.0)
            if use_personalization and profile:
                if a.get("topic") in (profile.get("topics") or []):
                    rel[i] += 0.12
                    reasons[i].append("profile topic")
                if a.get("source") in (profile.get("sources") or []):
                    rel[i] += 0.08
                    reasons[i].append("profile source")
                for e in (profile.get("entities") or [])[:5]:
                    if _mentions(blob, e):
                        rel[i] += 0.05
                        reasons[i].append(f"profile entity:{e}")
                        break

        cand = [i for i in np.argsort(-rel) if mask[i]][: max(pool, k)]
        if not cand:
            return []
        if use_mmr:
            chosen = self._mmr(cand, rel, k, lam, gamma)
        else:
            chosen = [(i, float(rel[i])) for i in cand[:k]]
        out = []
        for i, mmr_score in chosen:
            a = self.articles[i]
            out.append({**{kk: a.get(kk) for kk in ("id", "title", "source", "domain", "url", "published_at",
                                                     "topic", "kind", "author", "demo")},
                        "snippet": (a.get("body") or "")[:220], "relevance": round(float(rel[i]), 4),
                        "mmr_score": round(float(mmr_score), 4), "reasons": reasons[i][:4]})
        return out

    def _mmr(self, cand: list[int], rel: np.ndarray, k: int, lam: float, gamma: float) -> list[tuple[int, float]]:
        r = rel[cand]
        span = r.max() - r.min()
        rel_n = {i: (rel[i] - r.min()) / span if span > 0 else 1.0 for i in cand}
        selected: list[int] = []
        scores: list[float] = []
        remaining = list(cand)
        while remaining and len(selected) < k:
            best, best_score = None, -1e9
            for i in remaining:
                max_sim = max((float(self.vectors[i] @ self.vectors[j]) for j in selected), default=0.0)
                same_pub = sum(1 for j in selected if self.articles[j].get("source") == self.articles[i].get("source"))
                s = lam * rel_n[i] - (1 - lam) * max_sim - gamma * same_pub
                if s > best_score:
                    best, best_score = i, s
            selected.append(best)
            scores.append(best_score)
            remaining.remove(best)
        return list(zip(selected, scores))

    @staticmethod
    def _age(a: dict, now: date) -> int | None:
        d = _parse_date(a.get("published_at"))
        return (now - d).days if d else None

    # ---------------------------------------------------------------- helpers
    def similar(self, article_id: str, k: int = 3) -> list[dict]:
        if article_id not in self.id_to_idx:
            return []
        i = self.id_to_idx[article_id]
        sims = self.vectors @ self.vectors[i]
        order = [j for j in np.argsort(-sims) if j != i][:k]
        return [{**{kk: self.articles[j].get(kk) for kk in ("id", "title", "source", "topic", "published_at", "url")},
                 "similarity": round(float(sims[j]), 3)} for j in order]

    def vectors_for(self, ids: list[str]) -> np.ndarray:
        return np.array([self.vectors[self.id_to_idx[i]] for i in ids if i in self.id_to_idx])


def intra_list_diversity(vectors: np.ndarray) -> float:
    """ILD = mean pairwise (1 − cosine) over the recommended list."""
    n = len(vectors)
    if n < 2:
        return 0.0
    sims = vectors @ vectors.T
    iu = np.triu_indices(n, 1)
    return float(np.mean(1 - sims[iu]))


def unique_source_ratio(items: list[dict]) -> float:
    return len({i.get("source") for i in items}) / len(items) if items else 0.0


def precision_recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> tuple[float, float]:
    top = ranked_ids[:k]
    hits = sum(1 for d in top if d in relevant)
    return hits / k if k else 0.0, hits / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    for r, d in enumerate(ranked_ids, 1):
        if d in relevant:
            return 1.0 / r
    return 0.0


def describe(items: list[dict[str, Any]]) -> str:
    return "; ".join(f"{it['title']} ({it['source']})" for it in items)
