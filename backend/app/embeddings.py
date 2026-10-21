"""Sentence embeddings with an automatic offline fallback.

Primary : sentence-transformers `all-MiniLM-L6-v2` (384-d, L2-normalised).
Fallback: TF-IDF (uni+bi-grams, sublinear tf) concatenated with an LSA projection
          (TruncatedSVD) over concept-expanded text. The concept lexicon in text_utils maps
          paraphrases ("downpour", "deluge", "flooding") to a shared token so the fallback still
          captures some semantic similarity that pure keyword matching misses.
The active backend name is reported in every metrics payload so results are never mislabelled.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from . import config
from .text_utils import concept_expand

log = logging.getLogger("trendybits.embeddings")
_SBERT_LOCK = threading.Lock()
_SBERT_MODEL = None
_SBERT_FAILED = False


def _load_sbert():
    global _SBERT_MODEL, _SBERT_FAILED
    if config.USE_SBERT == "0" or _SBERT_FAILED:
        return None
    with _SBERT_LOCK:
        if _SBERT_MODEL is None and not _SBERT_FAILED:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                _SBERT_MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
            except Exception as exc:  # ImportError, offline download failure, …
                log.warning("sentence-transformers unavailable (%s) — using TF-IDF+LSA fallback", exc)
                _SBERT_FAILED = True
    return _SBERT_MODEL


class Embedder:
    def __init__(self, prefer_sbert: bool = True):
        self.sbert = _load_sbert() if prefer_sbert else None
        self.backend = "sbert:all-MiniLM-L6-v2" if self.sbert is not None else "tfidf-lsa-fallback"
        self._tfidf = None
        self._svd = None

    def fit(self, texts: list[str]) -> "Embedder":
        if self.sbert is not None:
            return self
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [concept_expand(t) for t in texts] or ["empty"]
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english",
                                      token_pattern=r"(?u)\b[\w-]+\b")
        X = self._tfidf.fit_transform(docs)
        n_comp = max(2, min(96, X.shape[0] - 1, X.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n_comp, random_state=config.SEED) if X.shape[0] > 3 else None
        if self._svd is not None:
            self._svd.fit(X)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.sbert is not None:
            v = self.sbert.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype="float32")
        if self._tfidf is None:
            self.fit(texts)
        X = self._tfidf.transform([concept_expand(t) for t in texts])
        dense = X.toarray().astype("float32")
        parts = [_l2(dense) * 0.8]
        if self._svd is not None:
            parts.append(_l2(self._svd.transform(X).astype("float32")) * 0.6)
        return _l2(np.hstack(parts))


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype("float32")


class VectorIndex:
    """FAISS IndexFlatIP over L2-normalised vectors (inner product == cosine); NumPy fallback."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors.astype("float32"))
        try:
            import faiss  # type: ignore

            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
            self.backend = "faiss.IndexFlatIP"
        except Exception:
            self.index = None
            self.backend = "numpy-cosine"

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.vectors))
        q = np.ascontiguousarray(q.reshape(1, -1).astype("float32"))
        if self.index is not None:
            scores, idx = self.index.search(q, k)
            return idx[0], scores[0]
        sims = self.vectors @ q[0]
        idx = np.argsort(-sims)[:k]
        return idx, sims[idx]

    def all_scores(self, q: np.ndarray) -> np.ndarray:
        return self.vectors @ q.reshape(-1).astype("float32")
