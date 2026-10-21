"""Module 4 — Pipeline B: independent fake-news & multi-dimensional credibility analyser.

Step 1  Pre-processing & claim decomposition  (untrusted instructions isolated; opinion / satire /
        calls-to-action separated from check-worthy factual claims)
Step 2  Evidence retrieval (RAG) per claim over the trusted-news / fact-check index
Step 3  Supervised classifier signals (LIAR 6-class, ISOT+Bharat binary) — one signal, never the authority
Step 4  Evidence comparison & conflict resolution (deepseek-r1:8b / gemma2:9b with deterministic NLI fallback)
Step 5  Multi-dimensional report: source trust, fact accuracy, overall estimate, tagged claims, citations,
        highlights, timeline, XAI (LIME + SHAP) and live model telemetry.

Pipeline A (summariser) never calls into this module and this module never summarises.
"""
from __future__ import annotations

import hashlib
import re
import time
from datetime import date, datetime

import numpy as np

from . import config
from .database import get_source
from .llm_service import OLLAMA, llm_reason_claim
from .metrics_store import benchmark_summary
from .retrieval import Retriever, _parse_date
from .security import sanitize_untrusted
from .text_utils import (ABSOLUTE_QUANTIFIERS, ANTONYMS, DEBUNK_CUES, LIMITING_QUALIFIERS, MISLEADING_CUES,
                         SENSATIONAL_CUES, content_tokens, coverage, extract_entities, extract_quantities, find_cues,
                         has_negation, percentages_in, split_sentences)
from .xai_explainer import LimeTextExplainer, explain_dimensions

VERDICT_POINTS = {"SUPPORTED": 95.0, "MISLEADING": 35.0, "UNVERIFIED": 50.0, "FALSE": 5.0}
DIM_WEIGHTS = {"source_reliability": 0.25, "rag_evidence_match": 0.40, "bert_liar_6class": 0.15,
               "bert_isot_bharat_binary": 0.20}
# Items under review (viral posts, unvetted scraped pages) are claims, never evidence.
EVIDENCE_KINDS = {"news", "official", "factcheck", "blog"}
OUTCOME_VERBS = {"won", "lost", "beat", "defeated"}
REL_THRESHOLD = 0.34
MIN_CLAUSE_COVERAGE = 0.3  # the best-matching evidence clause must share ≥30% of the claim's content words
MAX_FRAMING_PENALTY = 15.0
MIN_VERDICT_WEIGHT = 0.12

OPINION_CUES = ["in my view", "i think", "i believe", "in my opinion", "we must", "should", "ought to", "must also",
                "it is time", "shameful", "beautiful", "best ever", "worst ever"]
CTA_CUES = ["forward", "share this", "share before", "click", "subscribe", "send this", "spread the word", "like and share"]
SATIRE_CUES = ["satire", "parody", "satirical", "the onion", "just kidding"]
REPORTED_CLAIM_CUES = ["claims that", "claiming that", "claimed that", "with the claim", "viral post", "viral blog",
                       "post claims", "being shared", "social media post", "circulating", "rumour that", "rumor that"]


def credibility_aggregate(x: dict) -> float:
    """v(x): the multi-dimensional credibility aggregator (0–100) explained by SHAP.

    Weighted evidence/source/classifier blend minus the framing penalty, with a cap when the
    retrieved evidence strongly contradicts the item (non-linear interaction handled by Shapley)."""
    s = sum(DIM_WEIGHTS[k] * x[k] for k in DIM_WEIGHTS) - x["framing_omission_penalty"]
    if x["rag_evidence_match"] < 30:
        s = min(s, 38.0)
    return float(max(0.0, min(100.0, s)))


def classifier_only_score(x: dict) -> float:
    """Ablation (Model 5): no source / evidence / framing dimensions."""
    return float(0.5 * x["bert_liar_6class"] + 0.5 * x["bert_isot_bharat_binary"])


def overall_label(score: float) -> str:
    if score >= 80:
        return "Verified"
    if score >= 65:
        return "Mostly True"
    if score >= 45:
        return "Mixed / Disputed"
    if score >= 25:
        return "Mostly False"
    return "False"


def band(score: float) -> str:
    return "high" if score >= 65 else ("mixed" if score >= 40 else "low")


# ============================================================================ Step 1
def decompose_claims(title: str, body: str, image_caption: str | None = None, max_claims: int = 5) -> tuple[list[dict], list[dict]]:
    claims, non_claims = [], []
    for s in split_sentences(body):
        low = s.lower()
        if image_caption and re.search(r"\b(photo|image|picture|video)\b", low):
            non_claims.append(dict(text=s, type="media_reference (checked via the photo claim)"))
            continue
        if s.rstrip().endswith("?"):
            non_claims.append(dict(text=s, type="question"))
            continue
        if find_cues(low, SATIRE_CUES):
            non_claims.append(dict(text=s, type="satire"))
            continue
        if find_cues(low, CTA_CUES) and not extract_quantities(s):
            non_claims.append(dict(text=s, type="call_to_action"))
            continue
        if find_cues(low, OPINION_CUES) and not extract_quantities(s):
            non_claims.append(dict(text=s, type="opinion"))
            continue
        q, e = extract_quantities(s), extract_entities(s)
        absolute = bool(find_cues(low, ABSOLUTE_QUANTIFIERS))
        assertive = bool(re.search(r"\b(is|are|was|were|will|has|have|had|being|needed|required|confirmed|announced|said)\b", low))
        score = (1.0 * min(len(q), 2) + 0.5 * min(len(e), 2) + 0.5 * absolute + 0.3 * (len(content_tokens(s)) >= 4)
                 + 0.3 * assertive)
        if score < 0.5:
            non_claims.append(dict(text=s, type="background"))
            continue
        claims.append(dict(text=s, type="factual", checkworthiness=round(score, 2)))
    if len(claims) > max_claims:
        keep = sorted(range(len(claims)), key=lambda i: -claims[i]["checkworthiness"])[:max_claims]
        claims = [claims[i] for i in sorted(keep)]
    if image_caption:
        cap = re.sub(r"^\s*(photo|image|picture)\s*:\s*", "", image_caption, flags=re.I)
        claims.append(dict(text=f"Photo: {cap}", type="media", checkworthiness=1.0))
    if not claims and (body.strip() or title.strip()):
        claims.append(dict(text=(body.strip() or title.strip())[:300], type="factual", checkworthiness=0.5))
    for i, c in enumerate(claims, 1):
        c["cid"] = f"C{i}"
    return claims, non_claims


# ============================================================================ Step 4 (deterministic NLI)
def _clauses(sentence: str) -> list[str]:
    parts = re.split(r",\s+and\s+|;\s+|\s+and\s+(?=(?:the|it|will|has|have|is|was|no|its)\b)|,\s+but\s+|\s+but\s+", sentence)
    return [p for p in parts if len(p.split()) >= 3] or [sentence]


def stance(claim: str, passage: str) -> tuple[str, str]:
    """Return (stance, reason) of a passage w.r.t. a claim: supports | contradicts | misleading | unrelated."""
    cl = claim.lower()
    ct = set(content_tokens(claim))
    clauses = [c for s in split_sentences(passage) for c in _clauses(s)]
    # sentences that merely REPORT the claim ("a viral post claims that …") are not evidence for it
    own = [c for c in clauses if not find_cues(c.lower(), REPORTED_CLAIM_CUES)] or clauses
    best = max(own, key=lambda c: coverage(ct, content_tokens(c)), default=passage)
    bl = best.lower()
    # a "reused / out-of-context" cue only applies if its sentence is about this claim, or the claim is about media
    media_claim = bool(re.search(r"\b(photo|image|picture|video|footage)\b", cl))
    mis = [cue for s in split_sentences(passage) for cue in find_cues(s.lower(), MISLEADING_CUES)
           if media_claim or coverage(ct, content_tokens(s)) >= 0.34]
    if mis:
        return "misleading", f"evidence flags reused/out-of-context material ({', '.join(mis[:2])})"
    if coverage(ct, content_tokens(best)) < MIN_CLAUSE_COVERAGE:
        return "unrelated", "no clause in the passage addresses this claim"
    cw, bw =re.findall(r"[A-Za-z0-9'’\-]+", claim), re.findall(r"[A-Za-z0-9'’\-]+", best)

    def verb_at(words: list[str], vocab: set[str]):
        for i, w in enumerate(words):
            if w.lower() in vocab:
                subj = words[i - 1] if i > 0 and words[i - 1][:1].isupper() else None
                return w.lower(), subj
        return None

    for a, b in ANTONYMS:
        vc, vb = verb_at(cw, {a, b}), verb_at(bw, {a, b})
        if vc and vb and vc[0] != vb[0]:
            if vc[1] and vb[1] and vc[1].lower() != vb[1].lower():
                continue  # role reversal: "India won" ≡ "Australia lost"
            return "contradicts", f"evidence states '{vb[0]}' where the claim says '{vc[0]}'"
    vc, vb = verb_at(cw, OUTCOME_VERBS), verb_at(bw, OUTCOME_VERBS)
    if vc and vb and vc[0] == vb[0] and vc[1] and vb[1] and vc[1].lower() != vb[1].lower():
        return "contradicts", f"evidence says {vb[1]} {vb[0]}, not {vc[1]}"
    pc, pb = percentages_in(cl), percentages_in(bl)
    if pc and pb and not (pc & pb):
        return "contradicts", f"figures differ ({', '.join(sorted(pc))}% vs {', '.join(sorted(pb))}%)"
    claim_neg = has_negation(cl) or bool(find_cues(cl, DEBUNK_CUES))
    ev_neg = has_negation(bl) or bool(find_cues(bl, DEBUNK_CUES))
    if claim_neg != ev_neg:
        return "contradicts", "evidence explicitly denies the claim" if ev_neg else "evidence affirms what the claim denies"
    if find_cues(cl, ABSOLUTE_QUANTIFIERS):
        quals = [q for q in find_cues(passage.lower(), LIMITING_QUALIFIERS) if q not in cl]
        if quals:
            return "misleading", f"claim generalises ('{find_cues(cl, ABSOLUTE_QUANTIFIERS)[0]}') but evidence is qualified ('{quals[0]}')"
    return "supports", "evidence states the same facts"


def recency_factor(published: str | None, today: date) -> float:
    d = _parse_date(published)
    if not d:
        return 0.6
    age = (today - d).days
    return 1.0 if age <= 120 else (0.6 if age <= 365 else 0.4)


def source_profile(domain: str | None, source_name: str | None, input_type: str) -> dict:
    reg = get_source(domain) if domain else None
    if reg:
        age = reg["domain_age_years"] or 0
        age_txt = f"{max(1, round(age * 365))} days old" if age < 1 else f"{int(age)} years old"
        known = reg["trust_score"] >= 40 or reg["masthead_verified"]
        parts = ["Registered outlet" if known else "Unknown outlet",
                 "masthead verified" if reg["masthead_verified"] else "no masthead",
                 "corrections policy" if reg["corrections_policy"] else "no corrections policy", age_txt]
        return dict(score=float(reg["trust_score"]), publisher=reg["name"], domain=reg["domain"], domain_age=age_txt,
                    masthead_verified=bool(reg["masthead_verified"]), corrections_policy=bool(reg["corrections_policy"]),
                    owner=reg["owner"], contact=reg["contact"], corrections_record=reg["corrections_record"],
                    history=reg["history"], summary=" • ".join(parts), in_registry=True)
    if domain:
        return dict(score=35.0, publisher=source_name or domain, domain=domain, domain_age="unknown",
                    masthead_verified=False, corrections_policy=False, owner="Unknown", contact="none listed",
                    corrections_record="No record", history="Not in TrendyBits source registry",
                    summary="Unknown outlet • not in source registry • no masthead found", in_registry=False)
    return dict(score=50.0, publisher=source_name or "Pasted text", domain=None, domain_age="n/a", masthead_verified=False,
                corrections_policy=False, owner="n/a", contact="n/a", corrections_record="n/a", history="n/a",
                summary="Source not provided (pasted text) — neutral prior", in_registry=False)


def trust_label(score: float) -> str:
    return "High" if score >= 75 else ("Medium" if score >= 50 else "Low")


def accuracy_label(score: float) -> str:
    return "Accurate" if score >= 75 else ("Mixed" if score >= 50 else ("Mostly False" if score >= 30 else "False"))


# ============================================================================ analyser
class CredibilityAnalyzer:
    def __init__(self, retriever: Retriever, classifiers):
        self.retriever = retriever
        self.clf = classifiers
        self.lime = LimeTextExplainer()

    # ---------------------------------------------------------------- Step 2
    def retrieve_evidence(self, claim: str, exclude_ids: set[str], exclude_domain: str | None, k: int = 6,
                          today: date | None = None) -> list[dict]:
        hits = self.retriever.search(claim, k=k, use_personalization=False, use_mmr=False, exclude_ids=exclude_ids, pool=k,
                                     now=today, kinds=EVIDENCE_KINDS)
        ct = content_tokens(claim)
        out = []
        emb = self.retriever.embedder
        cv = emb.encode([claim])[0]
        for h in hits:
            if exclude_domain and h.get("domain") == exclude_domain:
                continue
            art = self.retriever.articles[self.retriever.id_to_idx[h["id"]]]
            sents = split_sentences(art["body"]) or [art["body"]]
            sv = emb.encode(sents)
            scored = []
            for s, v in zip(sents, sv):
                cov = coverage(ct, content_tokens(s))
                scored.append((0.6 * cov + 0.4 * float(v @ cv), s))
            scored.sort(key=lambda x: -x[0])
            rel = scored[0][0]
            if rel < REL_THRESHOLD:
                continue
            passage = " ".join(s for _, s in scored[:2])
            src = get_source(art.get("domain")) or {}
            out.append(dict(article_id=art["id"], title=art["title"], url=art["url"], publisher=art.get("source"),
                            domain=art.get("domain"), published_at=art.get("published_at"), kind=art.get("kind"),
                            trust=float(src.get("trust_score", 35.0)), passage=passage, relevance=round(rel, 3)))
        out.sort(key=lambda e: -e["relevance"])
        for i, e in enumerate(out[:4], 1):
            e["eid"] = f"E{i}"
        return out[:4]

    # ---------------------------------------------------------------- Step 4
    def judge_claim(self, claim: dict, evidence: list[dict], today: date, model: str, use_llm: bool) -> dict:
        t0 = time.perf_counter()
        if not evidence:
            return dict(**claim, verdict="UNVERIFIED", rationale="No relevant evidence was retrieved from trusted sources or "
                        "fact-check archives, so the claim cannot be verified. No sources are cited.",
                        evidence=[], conflict=False, uncertainty="high", reasoner="deterministic-nli", latency_ms=0.0)
        weights = {"supports": 0.0, "contradicts": 0.0, "misleading": 0.0}
        for e in evidence:
            e["stance"], e["stance_reason"] = stance(claim["text"], e["passage"])
        evidence = [e for e in evidence if e["stance"] != "unrelated"]
        if not evidence:
            return self.judge_claim(claim, [], today, model, use_llm)
        for e in evidence:
            st = e["stance"]
            e["weight"] = round(e["trust"] / 100 * recency_factor(e["published_at"], today) * e["relevance"], 4)
            weights[st] += e["weight"]
        ranked = sorted(weights.items(), key=lambda kv: (-kv[1], ["misleading", "contradicts", "supports"].index(kv[0])))
        top, top_w = ranked[0]
        second, second_w = ranked[1]
        conflict = top_w > 0 and second_w > 0.03 and {top, second} & {"supports"} != set() and top != second
        verdict = {"supports": "SUPPORTED", "contradicts": "FALSE", "misleading": "MISLEADING"}[top]
        cited = [e for e in evidence if e["stance"] == top]
        if top_w < MIN_VERDICT_WEIGHT:
            verdict, cited = "UNVERIFIED", []
            rationale = "Only weak or low-credibility evidence was found; the claim is left unverified."
        elif conflict and top_w / max(second_w, 1e-6) < 1.3:
            verdict, cited = "UNVERIFIED", evidence
            rationale = "Sources of similar weight disagree; the claim is left unverified pending review."
        else:
            best = max(cited, key=lambda e: e["weight"])
            rationale = {"SUPPORTED": f"{best['publisher']} ({best['published_at']}) reports the same facts.",
                         "FALSE": f"{best['publisher']} ({best['published_at']}) contradicts the claim: {best['stance_reason']}.",
                         "MISLEADING": f"{best['publisher']} ({best['published_at']}): {best['stance_reason']}."}[verdict]
        uncertainty = "low"
        if conflict:
            other = [e for e in evidence if e["stance"] == second]
            o = max(other, key=lambda e: e["weight"])
            rationale += (f" Conflict: {o['publisher']} ({o['published_at']}, trust {o['trust']:.0f}) {o['stance']} the claim, "
                          f"but newer / higher-credibility evidence was weighted more (weights {top_w:.2f} vs {second_w:.2f}).")
            uncertainty = "high" if top_w / max(second_w, 1e-6) < 2 else "medium"
        reasoner = "deterministic-nli"
        if use_llm:
            llm = llm_reason_claim(claim["text"], evidence, model)
            if llm:
                verdict, reasoner = llm["verdict"], model
                cited = [e for e in evidence if e["eid"] in llm["cited"]]
                rationale = llm["rationale"] or rationale
                conflict = conflict or llm["conflict"]
                uncertainty = llm["uncertainty"]
        return dict(**claim, verdict=verdict, rationale=rationale, conflict=conflict, uncertainty=uncertainty,
                    evidence=[{k: e[k] for k in ("eid", "title", "url", "publisher", "published_at", "trust", "passage",
                                                 "stance", "relevance", "weight", "kind")} for e in cited],
                    stance_weights={k: round(v, 3) for k, v in weights.items()}, reasoner=reasoner,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2))

    def judge_without_rag(self, claim: dict, liar_truth: float) -> dict:
        """Ablation Model 4: reasoner has no retrieved evidence — must rely on the classifier prior."""
        verdict = "FALSE" if liar_truth < 0.4 else ("SUPPORTED" if liar_truth > 0.6 else "UNVERIFIED")
        return dict(**claim, verdict=verdict, rationale=f"No evidence retrieval; verdict inferred from classifier prior "
                    f"(LIAR truth {liar_truth:.2f}).", conflict=False, uncertainty="high", evidence=[],
                    reasoner="classifier-prior (no RAG)", latency_ms=0.0)

    # ---------------------------------------------------------------- framing
    @staticmethod
    def framing(title: str, body: str, claim_results: list[dict], non_claims: list[dict]) -> dict:
        cues = find_cues(f"{title} {body}".lower(), SENSATIONAL_CUES)
        title_cues = find_cues(title.lower(), SENSATIONAL_CUES)
        letters = [c for c in title if c.isalpha()]
        caps = sum(1 for w in title.split() if len(w) > 3 and w.isupper()) / max(1, len(title.split()))
        sens = min(1.0, 0.25 * len(title_cues) + 0.6 * caps + 0.2 * title.count("!") + (0.0 if letters else 0))
        n_mis = sum(1 for c in claim_results if c["verdict"] == "MISLEADING")
        n_cta = sum(1 for n in non_claims if n["type"] == "call_to_action")
        anonymous = bool(find_cues(body.lower(), ["anonymous", "unnamed source", "sources said"]))
        penalty = min(MAX_FRAMING_PENALTY, 10 * sens + 2 * n_mis + 3 * n_cta + 3 * anonymous)
        return dict(sensationalism=round(sens, 2), cues=cues[:8], misleading_claims=n_mis, calls_to_action=n_cta,
                    anonymous_sourcing=anonymous, penalty=round(penalty, 2),
                    headline_flagged=sens >= 0.4)

    # ---------------------------------------------------------------- main entry
    def analyze(self, title: str, body: str, url: str | None = None, domain: str | None = None,
                source_name: str | None = None, article_id: str | None = None, image_caption: str | None = None,
                published_at: str | None = None, input_type: str = "text", reasoning_model: str | None = None,
                user_instruction: str | None = None, use_rag: bool = True, use_multidim: bool = True,
                use_llm: bool = True, with_xai: bool = True, lime_samples: int | None = None,
                extra_timeline: list | None = None, today: date | None = None) -> dict:
        t_start = time.perf_counter()
        today = today or date.today()
        model = reasoning_model if reasoning_model in config.REASONER_CHOICES else config.REASONER_MODEL
        telemetry: dict[str, dict] = {}

        # Step 1 — sanitise untrusted inputs, decompose claims
        sec_title = sanitize_untrusted(title or "", origin="analyzer:title")
        sec_body = sanitize_untrusted(body or "", origin="analyzer:body")
        sec_user = sanitize_untrusted(user_instruction or "", origin="analyzer:user_query")
        security_flags = sorted({f["kind"] for f in sec_title.flags + sec_body.flags + sec_user.flags})
        title_c, body_c = sec_title.clean_text, sec_body.clean_text
        claims, non_claims = decompose_claims(title_c, body_c, image_caption)

        # Step 3 — classifier signals
        t0 = time.perf_counter()
        liar_probs = self.clf.liar_proba([c["text"] for c in claims]) if claims else np.zeros((0, 6))
        liar_truth = (liar_probs @ np.array([0, .2, .4, .6, .8, 1.0])) if len(claims) else np.array([0.5])
        telemetry["liar"] = dict(latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                 confidence=round(float(liar_probs.max(1).mean()) if len(claims) else 0.0, 3))
        t0 = time.perf_counter()
        p_true = float(self.clf.integrated_proba([f"{title_c} . {body_c}"])[0])
        telemetry["integrated"] = dict(latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                       confidence=round(max(p_true, 1 - p_true), 3))

        # Steps 2 + 4 — evidence retrieval and comparison
        exclude = {article_id} if article_id else set()
        if url:
            exclude |= {a["id"] for a in self.retriever.articles if a.get("url") == url}
        t_ret, t_rea = 0.0, 0.0
        results = []
        for c, lt in zip(claims, liar_truth):
            if use_rag:
                t0 = time.perf_counter()
                ev = self.retrieve_evidence(c["text"], exclude, domain, today=today)
                t_ret += time.perf_counter() - t0
                t0 = time.perf_counter()
                r = self.judge_claim(c, ev, today, model, use_llm)
                r["retrieved_urls"] = [e["url"] for e in ev]
                t_rea += time.perf_counter() - t0
            else:
                r = self.judge_without_rag(c, float(lt))
                r["retrieved_urls"] = []
            r["liar_truth"] = round(float(lt), 3)
            results.append(r)
        telemetry["retriever"] = dict(latency_ms=round(t_ret * 1000, 2),
                                      confidence=round(float(np.mean([e["relevance"] for r in results for e in r["evidence"]]))
                                                       if any(r["evidence"] for r in results) else 0.0, 3))
        reasoner_used = sorted({r["reasoner"] for r in results}) or ["deterministic-nli"]
        telemetry["reasoner"] = dict(latency_ms=round(t_rea * 1000, 2), used=reasoner_used,
                                     confidence=round(1 - float(np.mean([{"low": .1, "medium": .35, "high": .6}[r["uncertainty"]]
                                                                         for r in results])) if results else 0.0, 3))

        # Framing / missing context (multi-dimensional only)
        frame = self.framing(title_c, body_c, results, non_claims) if use_multidim else dict(penalty=0.0, headline_flagged=False)
        if use_multidim and frame["headline_flagged"] and any(r["verdict"] == "SUPPORTED" for r in results):
            sup_ev = [e for r in results if r["verdict"] == "SUPPORTED" for e in r["evidence"]][:2]
            results.append(dict(cid=f"C{len(results) + 1}", text=f"Headline: \"{title_c}\"", type="framing",
                                checkworthiness=1.0, verdict="MISLEADING", conflict=False, uncertainty="medium",
                                rationale=("The body's facts are supported by evidence, but the headline adds sensational framing "
                                           f"({', '.join(frame['cues'][:3]) or 'all-caps / exclamation'}) that the evidence does not support."),
                                evidence=sup_ev, reasoner="framing-analyser", latency_ms=0.0, liar_truth=None,
                                retrieved_urls=[e["url"] for e in sup_ev]))
            frame["misleading_claims"] += 1
            frame["penalty"] = round(min(MAX_FRAMING_PENALTY, frame["penalty"] + 2), 2)

        # Step 5 — scores
        src = source_profile(domain, source_name, input_type)
        fact_acc = float(np.mean([VERDICT_POINTS[r["verdict"]] for r in results])) if results else 50.0
        dims = {"source_reliability": src["score"] if use_multidim else 50.0, "rag_evidence_match": fact_acc,
                "bert_liar_6class": round(float(np.mean(liar_truth)) * 100, 2), "bert_isot_bharat_binary": round(p_true * 100, 2),
                "framing_omission_penalty": frame["penalty"] if use_multidim else 0.0}
        fn = credibility_aggregate if use_multidim else classifier_only_score
        overall = round(fn(dims))
        counts = {v: sum(1 for r in results if r["verdict"] == v) for v in ("FALSE", "MISLEADING", "UNVERIFIED", "SUPPORTED")}

        # XAI
        xai = None
        if with_xai:
            text_for_lime = f"{title_c}. {body_c}"[:1500]
            lime = LimeTextExplainer(num_samples=lime_samples) if lime_samples else self.lime
            xai = dict(lime=lime.explain(text_for_lime, self.clf.text_credibility),
                       shap=explain_dimensions(dims, fn),
                       aggregator="v(x) = clip(0.25·S + 0.40·E + 0.15·L + 0.20·B − F, 0, 100), capped at 38 when E < 30")

        all_ev = {}
        for r in results:
            for e in r["evidence"]:
                all_ev.setdefault(e["url"], e)
        report = dict(
            report_id="rpt-" + hashlib.sha1(f"{title}|{body[:500]}|{url}|{time.time()}".encode()).hexdigest()[:10],
            pipeline="B (independent credibility analysis)", article_id=article_id, input_type=input_type,
            input_text=(body or "")[:4000], title=title_c or (body_c[:80] + "…"), url=url, published_at=published_at,
            language="ne" if re.search(r"[ऀ-ॿ]", body or "") else "en",
            source_trust=dict(score=round(src["score"]), label=trust_label(src["score"]), summary=src["summary"]),
            source_info=src,
            fact_accuracy=dict(score=round(fact_acc), label=accuracy_label(fact_acc),
                               summary=f"{len(results)} claims checked • " + " • ".join(f"{n} {v.lower()}" for v, n in counts.items() if n)),
            overall_score=overall, overall_label=overall_label(overall),
            overall_display=f"{overall_label(overall)} — {overall}/100",
            status="Verified" if overall >= 65 else "Disputed",
            overall_note="System estimate combining source, evidence, classifier and framing signals — not a final truth label.",
            claim_results=results, non_claims=non_claims, counts=counts, framing=frame,
            sources_cited=list(all_ev.values()),
            signals=dict(dimensions=dims, liar_6class_mean_probs=dict(zip(
                ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"],
                [round(float(x), 3) for x in (liar_probs.mean(0) if len(claims) else np.full(6, 1 / 6))])),
                         integrated_p_true=round(p_true, 3)),
            security_flags=security_flags,
            reasoning_model=model, reasoner_used=reasoner_used,
            ablation=dict(use_rag=use_rag, use_multidim=use_multidim, use_llm=use_llm),
            xai_explanation=xai,
        )
        report["highlights"] = self._highlights(report)
        report["timeline"] = self._timeline(report, extra_timeline, source_name, today)
        report["model_execution_metrics"] = self._telemetry(telemetry, model, reasoner_used)
        report["latency_ms"] = round((time.perf_counter() - t_start) * 1000, 1)
        return report

    # ---------------------------------------------------------------- report helpers
    @staticmethod
    def _highlights(rep: dict) -> list[str]:
        c = rep["counts"]
        h = [f"{len(rep['claim_results'])} checkable claims: {c['FALSE']} false, {c['MISLEADING']} misleading, "
             f"{c['UNVERIFIED']} unverified, {c['SUPPORTED']} supported."]
        h.append(f"Source: {rep['source_info']['publisher']} — trust {rep['source_trust']['score']}/100 ({rep['source_trust']['summary']}).")
        n_conf = sum(1 for r in rep["claim_results"] if r.get("conflict"))
        if n_conf:
            h.append(f"Sources conflict on {n_conf} claim(s); newer, higher-credibility evidence was weighted more heavily.")
        if rep["framing"].get("headline_flagged"):
            h.append("Sensational headline framing detected (" + ", ".join(rep["framing"].get("cues", [])[:3]) + ").")
        if rep["framing"].get("anonymous_sourcing"):
            h.append("Relies on anonymous sourcing.")
        if rep["security_flags"]:
            h.append("Embedded instructions were detected in the input and ignored (" + ", ".join(rep["security_flags"]) + ").")
        s = rep["signals"]
        h.append(f"Classifier signals (supporting only): LIAR truth {s['dimensions']['bert_liar_6class']:.0f}/100, "
                 f"ISOT-Bharat P(true) {s['dimensions']['bert_isot_bharat_binary']:.0f}/100.")
        return h

    @staticmethod
    def _timeline(rep: dict, extra: list | None, source_name: str | None, today: date) -> list[dict]:
        if extra:
            return [dict(date=d, event=e) for d, e in extra]
        ev = []
        if rep.get("published_at"):
            ev.append((rep["published_at"], f"Item published by {source_name or 'unknown source'}"))
        for e in rep["sources_cited"]:
            ev.append((e["published_at"] or "", f"{e['publisher']}: {e['title']}"))
        ev.append((today.isoformat(), "TrendyBits credibility report generated"))
        return [dict(date=d, event=e) for d, e in sorted(ev, key=lambda x: x[0])]

    def _telemetry(self, t: dict, model: str, reasoner_used: list[str]) -> list[dict]:
        ollama = OLLAMA.status()["reachable"]
        return [
            dict(model="Sentence-BERT + FAISS + MMR", key="retriever", backend=self.retriever.backend,
                 latency_ms=t["retriever"]["latency_ms"], confidence=t["retriever"]["confidence"],
                 benchmark=benchmark_summary("retriever")),
            dict(model="BERT-LIAR-6Class", key="liar", backend=self.clf.liar_backend, latency_ms=t["liar"]["latency_ms"],
                 confidence=t["liar"]["confidence"], benchmark=benchmark_summary("liar")),
            dict(model="BERT-ISOT-Bharat Binary", key="integrated", backend=self.clf.int_backend,
                 latency_ms=t["integrated"]["latency_ms"], confidence=t["integrated"]["confidence"],
                 benchmark=benchmark_summary("integrated")),
            dict(model=f"{model} evidence reasoner", key="reasoner",
                 backend=", ".join(reasoner_used) + ("" if ollama else " (Ollama offline → deterministic NLI fallback)"),
                 latency_ms=t["reasoner"]["latency_ms"], confidence=t["reasoner"]["confidence"],
                 benchmark=benchmark_summary("reasoner")),
            dict(model=f"{config.SUMMARIZER_MODEL} (Pipeline A, not used for verdicts)", key="summarizer",
                 backend=config.SUMMARIZER_MODEL if ollama else "deterministic fallback", latency_ms=None, confidence=None,
                 benchmark=benchmark_summary("summarizer")),
        ]


def evidence_urls(report: dict) -> set[str]:
    return {e["url"] for r in report["claim_results"] for e in r["evidence"]}


def cited_urls_valid(report: dict, retrieved_urls: set[str]) -> bool:
    return evidence_urls(report) <= retrieved_urls


def parse_iso(d: str) -> date:
    return datetime.fromisoformat(d).date()
