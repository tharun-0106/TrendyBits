#!/usr/bin/env python
"""Train & evaluate the supervised fake-news classifiers (kept SEPARATE per dataset):

  * LIAR 6-class claim classifier            (pants-fire, false, barely-true, half-true, mostly-true, true)
  * Integrated ISOT + BharatFakeNewsKosh      (binary article classifier, 1 = true / 0 = false)

Modes
  --mode fast : TF-IDF (+ credit-history features) → Logistic Regression (C tuned on VALID) and a TF-IDF→SVD→MLP baseline
  --mode bert : HuggingFace bert-base-uncased fine-tuning (PyTorch). Needs `pip install -r backend/requirements-ml.txt`
  --mode both : fast first, then bert

Metrics (Accuracy, macro/per-class Precision/Recall/F1, ROC-AUC, confusion matrix) are computed ONLY on the
held-out test split and written to data/model_metrics.json for the Admin dashboard.

Examples
  python scripts/train_bert_classifier.py --mode fast
  python scripts/train_bert_classifier.py --mode bert --dataset liar --epochs 3 --batch-size 16 --max-len 128
  python scripts/train_bert_classifier.py --mode bert --sample 2000      # quicker CPU run on a stratified subsample
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402

from app import config  # noqa: E402
from app.classifiers import classification_metrics, prep_integrated, prep_liar, train_all_fast  # noqa: E402
from app.datasets import LIAR_LABELS, ensure_datasets, load_split  # noqa: E402
from app.metrics_store import FILES, _read, write_json  # noqa: E402


def bert_texts(name: str, df):
    if name == "liar":
        p = prep_liar(df)
        return (p["liar_text"] + " | party " + p["party"]).tolist()
    return prep_integrated(df).tolist()


def train_bert(name: str, epochs: int, batch_size: int, max_len: int, lr: float, sample: int | None) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit(f"--mode bert needs torch + transformers ({exc}). Install backend/requirements-ml.txt") from exc

    torch.manual_seed(config.SEED)
    labels = LIAR_LABELS if name == "liar" else [0, 1]
    lab2id = {l: i for i, l in enumerate(labels)}
    train, valid, test = (load_split(name, s) for s in ("train", "valid", "test"))
    if sample and len(train) > sample:
        from sklearn.model_selection import train_test_split

        train, _ = train_test_split(train, train_size=sample, stratify=train["label"], random_state=config.SEED)
    ycol = (lambda d: [lab2id[x] for x in d["label"]]) if name == "liar" else (lambda d: [lab2id[int(x)] for x in d["label"]])
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def loader(df, shuffle):
        enc = tok(bert_texts(name, df), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(ycol(df)))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr, va, te = loader(train, True), loader(valid, False), loader(test, False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(tr) * epochs), len(tr) * epochs)

    def predict(dl):
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for ids, mask, y in dl:
                out = model(input_ids=ids.to(device), attention_mask=mask.to(device))
                probs.append(torch.softmax(out.logits, -1).cpu().numpy())
                ys.extend(y.tolist())
        return np.vstack(probs), ys

    best_f1, best_state = -1, None
    t0 = time.time()
    from sklearn.metrics import f1_score

    for ep in range(epochs):
        model.train()
        for step, (ids, mask, y) in enumerate(tr):
            loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            if step % 50 == 0:
                print(f"  [{name}] epoch {ep + 1}/{epochs} step {step}/{len(tr)} loss {loss.item():.4f}")
        pv, yv = predict(va)
        f1 = f1_score(yv, pv.argmax(1), average="macro")
        print(f"  [{name}] epoch {ep + 1} validation macro-F1 = {f1:.4f}")
        if f1 > best_f1:  # model selection on VALID only
            best_f1, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pt, yt = predict(te)
    t_lat = time.time()
    predict(loader(test.head(32), False))
    lat = (time.time() - t_lat) * 1000 / min(len(test), 32)
    m = classification_metrics([labels[i] for i in yt], [labels[i] for i in pt.argmax(1)], pt, labels)
    out_dir = config.MODELS_DIR / f"bert_{name}"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    m.update(backend="bert-base-uncased (fine-tuned)", epochs=epochs, max_len=max_len, lr=lr, batch_size=batch_size,
             n_train=len(train), best_valid_macro_f1=round(best_f1, 4), train_seconds=round(time.time() - t0, 1),
             latency_ms_per_item=round(lat, 2), device=device, tuned_on="validation split only")
    print(f"  [{name}] saved fine-tuned model to {out_dir}")
    return m


def print_report(name: str, m: dict) -> None:
    print(f"\n=== {name.upper()} — {m['backend']} (n_test={m['n_test']}) ===")
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "roc_auc"] + (["precision", "recall", "f1"] if name == "integrated" else [])
    for k in keys:
        print(f"  {k:<16} {m.get(k)}")
    print("  per-class F1:  " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in m["per_class"].items()))
    print("  confusion matrix (rows=gold, cols=pred, order=" + ",".join(m["labels"]) + ")")
    for row in m["confusion_matrix"]:
        print("    " + " ".join(f"{v:>5}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["fast", "bert", "both"], default="fast")
    ap.add_argument("--dataset", choices=["liar", "integrated", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--sample", type=int, default=None, help="stratified subsample of the train split for BERT")
    a = ap.parse_args()

    ensure_datasets()
    names = ["liar", "integrated"] if a.dataset == "all" else [a.dataset]
    if a.mode in ("fast", "both"):
        res = train_all_fast()
        for n in names:
            print_report(n, res[n]["baselines"]["tfidf-logreg"])
            print_report(n, res[n]["baselines"]["tfidf-svd-mlp"])
    if a.mode in ("bert", "both"):
        data = _read(FILES["classifiers"])
        for n in names:
            m = train_bert(n, a.epochs, a.batch_size, a.max_len, a.lr, a.sample)
            sec = data.setdefault(n, {})
            sec.setdefault("baselines", {})["bert-base-uncased"] = m
            sec["primary"] = m
            print_report(n, m)
        data.pop("generated_at", None)
        write_json("classifiers", data)
    print(f"\nMetrics written to {FILES['classifiers']}")
    print(json.dumps({n: {k: _read(FILES['classifiers'])[n]['primary'].get(k) for k in ('backend', 'accuracy', 'macro_f1', 'roc_auc')}
                      for n in names}, indent=1))


if __name__ == "__main__":
    main()
