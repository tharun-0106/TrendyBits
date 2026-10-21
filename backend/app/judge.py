"""LLM-as-a-judge utilities (DeepEval / RAGAs compatible) with deterministic fallbacks.

`CustomOllamaJudge` plugs a local Ollama model (default mistral:7b) into DeepEval metrics
(FaithfulnessMetric, AnswerRelevancyMetric). When deepeval is not installed or Ollama is offline,
the deterministic scorers below are used and every result records which scorer produced it.
"""
from __future__ import annotations

import re

import numpy as np

from . import config
from .llm_service import OLLAMA
from .text_utils import content_tokens, coverage, extract_quantities, norm_quantity, split_sentences, tokenize

try:  # pragma: no cover - optional dependency
    from deepeval.models.base_model import DeepEvalBaseLLM  # type: ignore

    HAVE_DEEPEVAL = True
except Exception:  # deepeval missing → light stand-in so the class still imports
    HAVE_DEEPEVAL = False

    class DeepEvalBaseLLM:  # type: ignore
        def __init__(self, model=None, *args, **kwargs):
            self.model = model


class CustomOllamaJudge(DeepEvalBaseLLM):
    """DeepEval-compatible judge backed by a local Ollama model."""

    def __init__(self, model_name: str = config.JUDGE_MODEL):
        self.model_name = model_name
        super().__init__(model_name)

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str, schema=None, *args, **kwargs):
        out = OLLAMA.generate(self.model_name, prompt, system="You are a strict evaluation judge. Answer in JSON when asked.",
                              json_mode=schema is not None) or ""
        if schema is not None:
            import json

            try:
                return schema(**json.loads(re.search(r"\{.*\}", out, re.S).group(0)))
            except Exception:
                return out
        return out

    async def a_generate(self, prompt: str, schema=None, *args, **kwargs):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"ollama:{self.model_name}"

    @staticmethod
    def available() -> bool:
        return HAVE_DEEPEVAL and OLLAMA.has_model(config.JUDGE_MODEL)


# ============================================================================ ROUGE
def _ngrams(toks: list[str], n: int) -> dict:
    out: dict = {}
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i:i + n])
        out[g] = out.get(g, 0) + 1
    return out


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def _lcs(a: list[str], b: list[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge(pred: str, ref: str) -> dict:
    """ROUGE-1/2/L F1 (rouge-score package when installed, else an equivalent implementation)."""
    try:
        from rouge_score import rouge_scorer  # type: ignore

        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(ref, pred)
        return {k: v.fmeasure for k, v in sc.items()} | {"impl": "rouge-score"}
    except ImportError:
        p, r = tokenize(pred), tokenize(ref)
        out = {}
        for n in (1, 2):
            pg, rg = _ngrams(p, n), _ngrams(r, n)
            ov = sum(min(c, rg.get(g, 0)) for g, c in pg.items())
            out[f"rouge{n}"] = _f1(ov / max(sum(pg.values()), 1), ov / max(sum(rg.values()), 1))
        lcs = _lcs(p, r)
        out["rougeL"] = _f1(lcs / max(len(p), 1), lcs / max(len(r), 1))
        out["impl"] = "builtin"
        return out


# ============================================================================ faithfulness / relevancy
def deterministic_faithfulness(output: str, source: str) -> dict:
    """Fraction of output sentences whose content words and every number are grounded in the source."""
    src_toks = set(content_tokens(source))
    src_q = {norm_quantity(q) for q in extract_quantities(source)}
    sents = [s for s in split_sentences(output) if len(s.split()) >= 3]
    if not sents:
        return {"score": 1.0, "supported": 0, "total": 0, "scorer": "deterministic"}
    ok = 0
    for s in sents:
        grounded = coverage(content_tokens(s), src_toks) >= 0.85
        nums = all(norm_quantity(q) in src_q for q in extract_quantities(s))
        ok += grounded and nums
    return {"score": round(ok / len(sents), 4), "supported": ok, "total": len(sents), "scorer": "deterministic"}


def faithfulness(output: str, source: str) -> dict:
    if CustomOllamaJudge.available():  # pragma: no cover - requires deepeval + Ollama
        try:
            from deepeval.metrics import FaithfulnessMetric  # type: ignore
            from deepeval.test_case import LLMTestCase  # type: ignore

            m = FaithfulnessMetric(model=CustomOllamaJudge(), include_reason=False, async_mode=False)
            m.measure(LLMTestCase(input="Summarise the article.", actual_output=output, retrieval_context=[source]))
            return {"score": round(float(m.score), 4), "scorer": f"deepeval+{config.JUDGE_MODEL}"}
        except Exception:
            pass
    return deterministic_faithfulness(output, source)


def embed_cosine(a: str, b: str, embedder=None) -> float:
    if embedder is not None:
        v = embedder.encode([a, b])
        return float(v[0] @ v[1])
    from .text_utils import cosine_counts

    return cosine_counts(content_tokens(a), content_tokens(b))


def answer_relevancy(question: str, answer: str, embedder=None) -> dict:
    if CustomOllamaJudge.available():  # pragma: no cover
        try:
            from deepeval.metrics import AnswerRelevancyMetric  # type: ignore
            from deepeval.test_case import LLMTestCase  # type: ignore

            m = AnswerRelevancyMetric(model=CustomOllamaJudge(), include_reason=False, async_mode=False)
            m.measure(LLMTestCase(input=question, actual_output=answer))
            return {"score": round(float(m.score), 4), "scorer": f"deepeval+{config.JUDGE_MODEL}"}
        except Exception:
            pass
    return {"score": round(embed_cosine(question, answer, embedder), 4), "scorer": "embedding-cosine proxy"}


def contextual_relevancy(claim: str, contexts: list[str], embedder=None, threshold: float = 0.3) -> float:
    """RAGAs-style contextual relevancy proxy: share of retrieved passages relevant to the claim.

    (The full `ragas` ContextRelevance metric needs an LLM backend; run tests/test_deepeval_suite.py
    with ragas installed and Ollama up to use it. This proxy is what the offline suite reports.)"""
    if not contexts:
        return 0.0
    rel =[max(embed_cosine(claim, c, embedder), coverage(content_tokens(claim), content_tokens(c))) >= threshold for c in contexts]
    return float(np.mean(rel))
