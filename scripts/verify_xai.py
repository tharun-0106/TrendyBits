#!/usr/bin/env python
"""Verify the XAI module's mathematical properties and save data/xai_verification.json.

Checks
  1. SHAP efficiency:   v(baseline) + Σφ_i == v(x)   (to 1e-6) on every demo article
  2. SHAP symmetry/null: a dimension equal to its baseline value receives φ = 0
  3. LIME local fidelity: weighted R² of the surrogate on its perturbation sample
  4. LIME faithfulness (deletion test): removing the top-3 negative tokens raises the black-box credibility
     and removing the top-3 positive tokens lowers it
  5. Determinism: two LIME runs with the same seed return identical weights
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import config  # noqa: E402
from app.credibility import credibility_aggregate  # noqa: E402
from app.services import get_services  # noqa: E402
from app.xai_explainer import SHAP_BASELINE, LimeTextExplainer, explain_dimensions  # noqa: E402

ARTICLES = ["tb-001", "tb-021", "tb-025", "tb-008", "tb-010", "tb-023"]


def remove_tokens(text: str, toks: list[str]) -> str:
    import re

    for t in toks:
        text = re.sub(re.escape(t), " ", text, flags=re.I)
    return text


def main() -> None:
    svc = get_services()
    f = svc.classifiers.text_credibility
    rows = []
    for aid in ARTICLES:
        rep = svc.analyze_article(aid, save=False)
        shap = rep["xai_explanation"]["shap"]
        lime = rep["xai_explanation"]["lime"]
        text = f"{rep['title']}. {svc.retriever.articles[svc.retriever.id_to_idx[aid]]['body']}"[:1500]
        base = float(f([text])[0])
        neg = [t["token"] for t in lime["top_negative"][:3]]
        pos = [t["token"] for t in lime["top_positive"][:3]]
        d_neg = float(f([remove_tokens(text, neg)])[0]) - base if neg else 0.0
        d_pos = float(f([remove_tokens(text, pos)])[0]) - base if pos else 0.0
        again = LimeTextExplainer().explain(text, f)
        same = [t["weight"] for t in again["tokens"]] == [t["weight"] for t in lime["tokens"]]
        null_dims = {**rep["signals"]["dimensions"], "bert_liar_6class": SHAP_BASELINE["bert_liar_6class"]}
        null_phi = [c["contribution"] for c in explain_dimensions(null_dims, credibility_aggregate)["contributions"]
                    if c["key"] == "bert_liar_6class"][0]
        rows.append(dict(article=aid, overall=rep["overall_score"], shap_baseline=shap["baseline_score"],
                         shap_sum=round(sum(c["contribution"] for c in shap["contributions"]), 4),
                         efficiency_gap=shap["efficiency_gap"], null_player_phi=null_phi,
                         lime_r2=lime["local_fidelity_r2"], lime_samples=lime["num_samples"],
                         delete_top_negative_delta=round(d_neg, 4), delete_top_positive_delta=round(d_pos, 4),
                         deterministic=same, top_negative=neg, top_positive=pos))
    checks = dict(
        shap_efficiency=all(abs(r["efficiency_gap"]) < 1e-6 for r in rows),
        shap_null_player=all(abs(r["null_player_phi"]) < 1e-9 for r in rows),
        lime_mean_r2=round(sum(r["lime_r2"] for r in rows) / len(rows), 4),
        lime_deletion_negative_raises=sum(r["delete_top_negative_delta"] >= 0 for r in rows) / len(rows),
        lime_deletion_positive_lowers=sum(r["delete_top_positive_delta"] <= 0 for r in rows) / len(rows),
        lime_deterministic=all(r["deterministic"] for r in rows))
    out = dict(checks=checks, rows=rows)
    (config.DATA_DIR / "xai_verification.json").write_text(json.dumps(out, indent=2))
    print(f"{'article':<8} {'overall':>7} {'φ0':>6} {'Σφ':>8} {'gap':>9} {'LIME R²':>8} {'Δ(-neg)':>8} {'Δ(-pos)':>8}  top negative tokens")
    for r in rows:
        print(f"{r['article']:<8} {r['overall']:>7} {r['shap_baseline']:>6} {r['shap_sum']:>8} {r['efficiency_gap']:>9} "
              f"{r['lime_r2']:>8} {r['delete_top_negative_delta']:>8} {r['delete_top_positive_delta']:>8}  {r['top_negative']}")
    print("\nchecks:", json.dumps(checks, indent=1))
    print(f"saved → {config.DATA_DIR / 'xai_verification.json'}")


if __name__ == "__main__":
    main()
