"""Module 6a — Explainable AI: LIME token attribution + exact Shapley (SHAP) dimension attribution.

LIME (Ribeiro et al., 2016) adapted to news text
------------------------------------------------
1. Segment the text into interpretable units u_1..u_d (money amounts like "Rs 50,000", cue phrases
   like "forward immediately", multi-word entities like "Reserve Bank", then single words).
2. Draw binary masks z ∈ {0,1}^d: the all-ones row, every leave-one-out row, then random masks.
   A 0 removes the unit from the text.
3. Query the black box f (classifier credibility surrogate, P(credible) ∈ [0,1]) on each perturbed text.
4. Weight each sample by the exponential kernel  π(z) = exp(−D(z)² / σ²), with cosine distance
   D(z) = 1 − ‖z‖₁^{1/2} / d^{1/2}  and σ = 0.25.
5. Fit a weighted ridge regression  g(z) = w₀ + Σ w_i z_i  minimising Σ π(z)(f(z) − g(z))² + α‖w‖².
   w_i ∈ [−1, +1] is the signed change in P(credible) contributed by unit i: negative (red) pushes towards
   False/Misleading, positive (green) towards Verified. Local fidelity is reported as weighted R².

SHAP over the 5 pipeline dimensions
-----------------------------------
For the credibility aggregator v(x) with dimensions N = {source, evidence, liar, isot, framing} and a
baseline (prior) x⁰, the exact Shapley value of dimension i is

    φ_i = Σ_{S ⊆ N\\{i}} |S|!(|N|−|S|−1)!/|N|! · [ v(x_S∪{i}, x⁰_rest) − v(x_S, x⁰_rest) ]

computed over all 2⁵ = 32 coalitions. Efficiency holds: v(x⁰) + Σ φ_i = v(x) — every point of the final
score is attributed to exactly one dimension.
"""
from __future__ import annotations

import math
import re
import time
from itertools import combinations
from typing import Callable

import numpy as np

from . import config
from .text_utils import MONEY_RE, STOPWORDS, extract_entities

SHAP_DIMENSIONS = [
    ("source_reliability", "Source Reliability"),
    ("rag_evidence_match", "RAG Evidence Match"),
    ("bert_liar_6class", "BERT-LIAR 6-Class Signal"),
    ("bert_isot_bharat_binary", "BERT ISOT-Bharat Binary Signal"),
    ("framing_omission_penalty", "Framing / Omission Penalty"),
]
SHAP_BASELINE = {"source_reliability": 50.0, "rag_evidence_match": 50.0, "bert_liar_6class": 50.0,
                 "bert_isot_bharat_binary": 50.0, "framing_omission_penalty": 0.0}

CUE_PHRASES = [
    "forward immediately", "share before it is deleted", "before it is deleted", "anonymous ministry source",
    "ministry source", "anonymous source", "anonymous scientist", "official clarification", "official notice",
    "according to", "said in a statement", "no such scheme", "out of context", "missing context",
    "you won't believe", "mainstream media", "Reserve Bank of India", "Reserve Bank", "Ministry of Finance",
    "Election Commission", "India Meteorological Department", "official spokesperson", "press release",
]


# ============================================================================ segmentation
def segment(text: str) -> list[dict]:
    """Split text into interpretable units with character offsets (for the UI heatmap)."""
    spans: list[tuple[int, int, str]] = []

    def add(s: int, e: int, kind: str):
        if not any(s < e2 and e > s2 for s2, e2, _ in spans):
            spans.append((s, e, kind))

    for m in MONEY_RE.finditer(text):
        add(m.start(), m.end(), "quantity")
    low = text.lower()
    for ph in sorted(CUE_PHRASES, key=len, reverse=True):
        for m in re.finditer(re.escape(ph.lower()), low):
            add(m.start(), m.end(), "phrase")
    for ent in extract_entities(text):
        if " " in ent:
            for m in re.finditer(re.escape(ent), text):
                add(m.start(), m.end(), "entity")
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'’\-.,%]*[A-Za-z0-9%]|[A-Za-z0-9]", text):
        add(m.start(), m.end(), "word")
    spans.sort()
    units = []
    for s, e, kind in spans:
        t = text[s:e]
        key = t.lower().strip(".,")
        is_feat = kind != "word" or (key not in STOPWORDS and len(key) > 2 and not key.isdigit())
        units.append(dict(text=t, start=s, end=e, kind=kind, key=key, is_feature=is_feat))
    return units


# ============================================================================ LIME
class LimeTextExplainer:
    def __init__(self, num_samples: int | None = None, kernel_width: float = 0.25, alpha: float = 1.0,
                 max_features: int = 60, seed: int = config.SEED):
        self.num_samples = num_samples or config.LIME_SAMPLES
        self.kernel_width = kernel_width
        self.alpha = alpha
        self.max_features = max_features
        self.seed = seed

    def explain(self, text: str, predict_fn: Callable[[list[str]], np.ndarray], top_k: int = 8) -> dict:
        t0 = time.perf_counter()
        units = segment(text)
        feats: list[str] = []
        for u in units:
            if u["is_feature"] and u["key"] not in feats:
                feats.append(u["key"])
        feats = feats[: self.max_features]
        d = len(feats)
        if d == 0:
            return dict(method="LIME", text=text, tokens=[dict(text=u["text"], start=u["start"], end=u["end"], weight=0.0) for u in units],
                        top_negative=[], top_positive=[], prediction=None, intercept=None, local_fidelity_r2=None,
                        num_samples=0, num_features=0)
        fidx = {f: i for i, f in enumerate(feats)}
        rng = np.random.default_rng(self.seed)
        rows = [np.ones(d, dtype=int)]
        if d <= self.num_samples // 2:
            for i in range(d):
                z = np.ones(d, dtype=int)
                z[i] = 0
                rows.append(z)
        while len(rows) < self.num_samples:
            z = np.ones(d, dtype=int)
            k = rng.integers(1, d) if d > 1 else 1
            z[rng.choice(d, size=k, replace=False)] = 0
            rows.append(z)
        Z = np.array(rows)
        texts = []
        for z in Z:
            kept = [u["text"] for u in units if not (u["key"] in fidx and z[fidx[u["key"]]] == 0)]
            texts.append(" ".join(kept))
        y = np.asarray(predict_fn(texts), dtype=float)
        dist = 1 - np.sqrt(Z.sum(1) / d)
        w = np.exp(-(dist ** 2) / (self.kernel_width ** 2))
        from sklearn.linear_model import Ridge

        reg = Ridge(alpha=self.alpha, fit_intercept=True)
        reg.fit(Z, y, sample_weight=w)
        coef = np.clip(reg.coef_, -1, 1)
        r2 = float(reg.score(Z, y, sample_weight=w))
        max_abs = float(np.abs(coef).max()) or 1.0
        tokens = []
        for u in units:
            if u["key"] in fidx:
                c = float(coef[fidx[u["key"]]])
                tokens.append(dict(text=u["text"], start=u["start"], end=u["end"], weight=round(c, 4),
                                   normalized=round(c / max_abs, 4), kind=u["kind"],
                                   direction="credible" if c > 0 else ("not_credible" if c < 0 else "neutral")))
            else:
                tokens.append(dict(text=u["text"], start=u["start"], end=u["end"], weight=0.0, normalized=0.0,
                                   kind=u["kind"], direction="neutral"))
        ranked = sorted(((f, float(coef[i])) for f, i in fidx.items()), key=lambda x: x[1])
        return dict(
            method="LIME (masking + leave-one-out perturbation, exponential kernel, weighted ridge)",
            black_box="0.6·P_integrated(true) + 0.4·LIAR truth score",
            text=text, prediction=round(float(y[0]), 4), intercept=round(float(reg.intercept_), 4),
            local_fidelity_r2=round(r2, 4), num_samples=int(len(Z)), num_features=d, kernel_width=self.kernel_width,
            tokens=tokens,
            top_negative=[dict(token=f, weight=round(v, 4)) for f, v in ranked[:top_k] if v < 0],
            top_positive=[dict(token=f, weight=round(v, 4)) for f, v in reversed(ranked[-top_k:]) if v > 0],
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )


# ============================================================================ SHAP (exact Shapley)
def shapley_values(fn: Callable[[dict], float], x: dict, baseline: dict) -> dict:
    keys = list(x)
    n = len(keys)
    phi = {k: 0.0 for k in keys}
    cache: dict[tuple, float] = {}

    def v(coalition: frozenset) -> float:
        key = tuple(sorted(coalition))
        if key not in cache:
            cache[key] = fn({k: (x[k] if k in coalition else baseline[k]) for k in keys})
        return cache[key]

    for k in keys:
        others = [o for o in keys if o != k]
        for r in range(n):
            wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in combinations(others, r):
                s = frozenset(S)
                phi[k] += wgt * (v(s | {k}) - v(s))
    return phi


def explain_dimensions(x: dict, fn: Callable[[dict], float], baseline: dict | None = None) -> dict:
    baseline = baseline or SHAP_BASELINE
    phi = shapley_values(fn, x, baseline)
    base_score = fn(dict(baseline))
    final = fn(dict(x))
    contribs = [dict(key=k, label=label, value=round(float(x[k]), 2), baseline_value=baseline[k],
                     contribution=round(float(phi[k]), 2)) for k, label in SHAP_DIMENSIONS]
    return dict(method="Exact Shapley values over 5 pipeline dimensions (32 coalitions)",
                baseline_score=round(base_score, 2), final_score=round(final, 2), contributions=contribs,
                efficiency_gap=round(final - base_score - sum(phi.values()), 6))


def global_importance(reports: list[dict]) -> dict:
    """Global SHAP view: mean |φ| and mean φ per dimension across saved reports."""
    rows = [r["xai_explanation"]["shap"]["contributions"] for r in reports
            if r.get("xai_explanation", {}).get("shap")]
    if not rows:
        return {"n_reports": 0, "dimensions": []}
    out = []
    for key, label in SHAP_DIMENSIONS:
        vals = [c["contribution"] for row in rows for c in row if c["key"] == key]
        out.append(dict(key=key, label=label, mean_abs=round(float(np.mean(np.abs(vals))), 2),
                        mean=round(float(np.mean(vals)), 2)))
    return {"n_reports": len(rows), "dimensions": sorted(out, key=lambda d: -d["mean_abs"])}
