"""Supervised fake-news classifiers (Pipeline B, Step 3).

Model 3 — LIAR 6-class claim classifier   (pants-fire … true)
Model 4 — Integrated ISOT + BharatFakeNewsKosh binary article classifier (1 = true, 0 = false)

Two interchangeable back-ends, chosen automatically at load time:
  * bert  : HuggingFace `bert-base-uncased` fine-tuned by `scripts/train_bert_classifier.py --mode bert`
            (saved in models/bert_liar/ and models/bert_integrated/)
  * fast  : scikit-learn TF-IDF (+ credit-history features for LIAR) → Logistic Regression
            (saved in models/liar_fast.joblib and models/integrated_fast.joblib)
If neither exists the fast models are trained on the stratified train split on first use.
The classifier output is only ONE signal for the credibility report — never the truth authority.
"""
from __future__ import annotations

import json
import logging
import threading
import time

import numpy as np
import pandas as pd

from . import config
from .datasets import LIAR_LABELS, ensure_datasets, load_split

log = logging.getLogger("trendybits.classifiers")
LIAR_TRUTH_WEIGHTS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
HIST_COLS = ["pants_on_fire_counts", "false_counts", "barely_true_counts", "half_true_counts", "mostly_true_counts"]


# ============================================================================ feature prep
def prep_liar(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["liar_text"] = (df["statement"].astype(str) + " | subject " + df.get("subject", "").astype(str) +
                        " | context " + df.get("context", "").astype(str))
    out["party"] = df.get("party", "none").astype(str)
    hist = df[HIST_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
    total = hist.sum(axis=1, keepdims=True)
    props = np.divide(hist, total, out=np.zeros_like(hist), where=total > 0)
    for i, c in enumerate(HIST_COLS):
        out[f"p_{c}"] = props[:, i]
    out["log_total"] = np.log1p(total[:, 0])
    return out


def prep_integrated(df: pd.DataFrame) -> pd.Series:
    return (df["title"].astype(str) + " . " + df["text"].astype(str)).rename("doc")


def liar_frame_from_text(statements: list[str], meta: list[dict] | None = None) -> pd.DataFrame:
    meta = meta or [{} for _ in statements]
    rows = []
    for s, m in zip(statements, meta):
        rows.append(dict(statement=s, subject=m.get("subject", "unknown"), context=m.get("context", "a news article"),
                         party=m.get("party", "none"), **{c: m.get(c, 0) for c in HIST_COLS}))
    return pd.DataFrame(rows)


# ============================================================================ pipelines
def build_liar_pipeline(C: float = 1.0, kind: str = "logreg"):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_cols = [f"p_{c}" for c in HIST_COLS] + ["log_total"]
    feats = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2), "liar_text"),
        ("party", OneHotEncoder(handle_unknown="ignore"), ["party"]),
        ("hist", StandardScaler(), num_cols),
    ])
    if kind == "mlp":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.neural_network import MLPClassifier

        return Pipeline([("features", feats), ("svd", TruncatedSVD(150, random_state=config.SEED)),
                         ("clf", MLPClassifier((128,), max_iter=300, early_stopping=True, random_state=config.SEED))])
    return Pipeline([("features", feats), ("clf", LogisticRegression(C=C, max_iter=3000))])


def build_integrated_pipeline(C: float = 4.0, kind: str = "logreg"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=60000)
    if kind == "mlp":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.neural_network import MLPClassifier

        return Pipeline([("tfidf", vec), ("svd", TruncatedSVD(150, random_state=config.SEED)),
                         ("clf", MLPClassifier((128,), max_iter=300, early_stopping=True, random_state=config.SEED))])
    return Pipeline([("tfidf", vec), ("clf", LogisticRegression(C=C, max_iter=3000))])


def classification_metrics(y_true, y_pred, proba: np.ndarray, labels: list) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                                 roc_auc_score)
    from sklearn.preprocessing import label_binarize

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    per_p, per_r, per_f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(p), 4), "macro_recall": round(float(r), 4), "macro_f1": round(float(f), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)), 4),
        "per_class": {str(l): {"precision": round(float(a), 4), "recall": round(float(b), 4), "f1": round(float(c), 4),
                               "support": int(s)} for l, a, b, c, s in zip(labels, per_p, per_r, per_f, sup)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": [str(l) for l in labels], "n_test": int(len(y_true)),
    }
    try:
        if len(labels) == 2:
            out["roc_auc"] = round(float(roc_auc_score(y_true, proba[:, 1])), 4)
            # binary precision/recall/F1 for the positive ("true") class as conventionally reported
            bp, br, bf, _ = precision_recall_fscore_support(y_true, y_pred, labels=[labels[1]], average=None, zero_division=0)
            out.update(precision=round(float(bp[0]), 4), recall=round(float(br[0]), 4), f1=round(float(bf[0]), 4))
        else:
            yb = label_binarize(y_true, classes=labels)
            out["roc_auc"] = round(float(roc_auc_score(yb, proba, average="macro", multi_class="ovr")), 4)
    except ValueError:
        out["roc_auc"] = None
    return out


def train_fast(name: str, kind: str = "logreg") -> tuple[object, dict]:
    """Fit on train, tune C on VALID only, report on the isolated TEST split."""
    train, valid, test = (load_split(name, s) for s in ("train", "valid", "test"))
    if name == "liar":
        Xtr, Xva, Xte = prep_liar(train), prep_liar(valid), prep_liar(test)
        ytr, yva, yte = train["label"].tolist(), valid["label"].tolist(), test["label"].tolist()
        builder, grid, labels = build_liar_pipeline, [0.25, 0.5, 1.0, 2.0], LIAR_LABELS
    else:
        Xtr, Xva, Xte = prep_integrated(train), prep_integrated(valid), prep_integrated(test)
        ytr, yva, yte = train["label"].astype(int).tolist(), valid["label"].astype(int).tolist(), test["label"].astype(int).tolist()
        builder, grid, labels = build_integrated_pipeline, [1.0, 4.0, 16.0], [0, 1]
    from sklearn.metrics import f1_score

    t0 = time.perf_counter()
    if kind == "mlp":
        best_C, model = None, builder(kind="mlp").fit(Xtr, ytr)
    else:
        best_C, best_f1 = grid[0], -1.0
        for C in grid:
            m = builder(C=C).fit(Xtr, ytr)
            f1 = f1_score(yva, m.predict(Xva), average="macro")
            if f1 > best_f1:
                best_C, best_f1 = C, f1
        model = builder(C=best_C).fit(pd.concat([Xtr, Xva]), ytr + yva)  # refit on train+valid; test untouched
    train_s = time.perf_counter() - t0
    proba = model.predict_proba(Xte)
    order = [list(model.classes_).index(l) for l in labels]
    proba = proba[:, order]
    pred = [labels[i] for i in proba.argmax(1)]
    t1 = time.perf_counter()
    model.predict_proba(Xte[:50] if name == "liar" else Xte.iloc[:50])
    lat = (time.perf_counter() - t1) / 50 * 1000
    metrics = classification_metrics(yte, pred, proba, labels)
    metrics.update(backend=f"tfidf-{kind}" + (f" (C={best_C})" if best_C else ""), train_seconds=round(train_s, 2),
                   latency_ms_per_item=round(lat, 3), n_train=len(ytr), n_valid=len(yva),
                   tuned_on="validation split only", test_split_sha256=_manifest_hash(name))
    return model, metrics


def _manifest_hash(name: str) -> str:
    from .datasets import load_manifest

    return load_manifest(name).get("test_sha256", "")


# ============================================================================ runtime service
class ClassifierService:
    _lock = threading.Lock()

    def __init__(self):
        self.liar_backend = self.int_backend = None
        self._liar = self._int = None
        self._hf = {}
        self._load()

    def _load(self) -> None:
        import joblib

        for name in ("liar", "integrated"):
            hf_dir = config.MODELS_DIR / f"bert_{name}"
            if (hf_dir / "config.json").exists():
                try:
                    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

                    tok = AutoTokenizer.from_pretrained(hf_dir)
                    mdl = AutoModelForSequenceClassification.from_pretrained(hf_dir).eval()
                    self._hf[name] = (tok, mdl)
                    setattr(self, "liar_backend" if name == "liar" else "int_backend", "bert-base-uncased (fine-tuned)")
                    continue
                except Exception as exc:
                    log.warning("Could not load fine-tuned BERT for %s (%s); using fast model", name, exc)
            path = config.MODELS_DIR / f"{name}_fast.joblib"
            with self._lock:
                if path.exists():
                    model = joblib.load(path)
                else:
                    ensure_datasets()
                    model, metrics = train_fast(name)
                    joblib.dump(model, path)
                    (config.MODELS_DIR / f"{name}_fast_meta.json").write_text(json.dumps(metrics, indent=1))
            if name == "liar":
                self._liar, self.liar_backend = model, "tfidf-logreg (fast mode)"
            else:
                self._int, self.int_backend = model, "tfidf-logreg (fast mode)"

    # ------------------------------------------------------------------ inference
    def _hf_proba(self, name: str, texts: list[str]) -> np.ndarray:
        import torch

        tok, mdl = self._hf[name]
        outs = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i + 32], truncation=True, max_length=256, padding=True, return_tensors="pt")
            with torch.no_grad():
                outs.append(torch.softmax(mdl(**enc).logits, dim=-1).numpy())
        return np.vstack(outs)

    def liar_proba(self, statements: list[str], meta: list[dict] | None = None) -> np.ndarray:
        if not statements:
            return np.zeros((0, 6))
        if "liar" in self._hf:
            return self._hf_proba("liar", statements)
        X = prep_liar(liar_frame_from_text(statements, meta))
        p = self._liar.predict_proba(X)
        order = [list(self._liar.classes_).index(l) for l in LIAR_LABELS]
        return p[:, order]

    def liar_truth(self, statements: list[str], meta: list[dict] | None = None) -> np.ndarray:
        return self.liar_proba(statements, meta) @ LIAR_TRUTH_WEIGHTS

    def integrated_proba(self, docs: list[str]) -> np.ndarray:
        """P(label = true) for each 'title . text' document."""
        if not docs:
            return np.zeros(0)
        if "integrated" in self._hf:
            return self._hf_proba("integrated", docs)[:, 1]
        p = self._int.predict_proba(pd.Series(docs, name="doc"))
        return p[:, list(self._int.classes_).index(1)]

    def text_credibility(self, texts: list[str]) -> np.ndarray:
        """Fast text-only surrogate used as the LIME black box: 0.6·P_integrated(true) + 0.4·LIAR truth."""
        return 0.6 * self.integrated_proba(texts) + 0.4 * self.liar_truth(texts)


_SERVICE: ClassifierService | None = None
_SERVICE_LOCK = threading.Lock()


def get_classifier_service() -> ClassifierService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = ClassifierService()
    return _SERVICE


def train_all_fast(write: bool = True, progress=print) -> dict:
    """Train + evaluate the fast production models and MLP baselines; write data/model_metrics.json.

    A fine-tuned BERT model (scripts/train_bert_classifier.py --mode bert) stays the primary entry when present."""
    import joblib

    from .datasets import load_manifest
    from .metrics_store import FILES, _read, write_json

    existing = _read(FILES["classifiers"])
    out = {}
    for name in ("liar", "integrated"):
        progress(f"[classifiers] {name}: TF-IDF + Logistic Regression")
        model, m = train_fast(name, "logreg")
        joblib.dump(model, config.MODELS_DIR / f"{name}_fast.joblib")
        progress(f"[classifiers] {name}: TF-IDF + SVD + MLP baseline")
        _, mlp = train_fast(name, "mlp")
        sec = existing.get(name, {})
        baselines = dict(sec.get("baselines", {}))
        baselines.update({"tfidf-logreg": m, "tfidf-svd-mlp": mlp})
        has_bert = (config.MODELS_DIR / f"bert_{name}" / "config.json").exists()
        prev = sec.get("primary", {})
        primary = prev if has_bert and str(prev.get("backend", "")).startswith("bert") else m
        man = load_manifest(name)
        out[name] = {"primary": primary, "baselines": baselines,
                     "dataset": {"origin": man.get("origin"), "sizes": man.get("sizes"),
                                 "test_sha256": man.get("test_sha256"), "audit": man.get("audit")}}
    if write:
        write_json("classifiers", out)
    return out
