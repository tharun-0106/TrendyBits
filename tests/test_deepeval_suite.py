"""Section VI — the 18 LLM test cases as pytest tests (+ optional real DeepEval / RAGAs checks).

  pytest tests/test_deepeval_suite.py -v
  deepeval test run tests/test_deepeval_suite.py        # same file through the DeepEval CLI (if installed)

The suite runs once per session (`suite_results`) and each TC-xx becomes its own test. When Ollama is
running with mistral:7b (the judge), the DeepEval Faithfulness / Answer-Relevancy metrics are evaluated
through `CustomOllamaJudge`; otherwise the deterministic scorers are used (recorded in the results).
"""
from __future__ import annotations

import pytest

from app.judge import HAVE_DEEPEVAL, CustomOllamaJudge, deterministic_faithfulness

TC_IDS = [f"TC-{i:02d}" for i in range(1, 19)]


@pytest.fixture(scope="session")
def suite_results(svc):
    from app.evaluation import run_test_suite

    return run_test_suite(svc, progress=lambda m: None)


@pytest.mark.parametrize("tc", TC_IDS)
def test_llm_case(suite_results, tc):
    case = next(c for c in suite_results["cases"] if c["id"] == tc)
    assert case["status"] == "PASS", f"{tc} {case['name']}: measured {case['measured']} vs target {case['target']}"


def test_table5_all_criteria(suite_results):
    failed = [r for r in suite_results["table5"] if r["status"] != "PASS"]
    assert len(suite_results["table5"]) == 10
    assert not failed, failed


def test_custom_judge_interface():
    judge = CustomOllamaJudge()
    assert judge.get_model_name().startswith("ollama:")
    assert judge.load_model() == judge.model_name


def test_deterministic_faithfulness_detects_invented_number():
    src = "The council approved Rs 40 crore for roads on 3 March."
    assert deterministic_faithfulness("The council approved Rs 40 crore for roads.", src)["score"] == 1.0
    assert deterministic_faithfulness("The council approved Rs 90 crore for roads.", src)["score"] == 0.0


@pytest.mark.skipif(not (HAVE_DEEPEVAL and CustomOllamaJudge.available()), reason="deepeval + Ollama judge not available")
def test_deepeval_faithfulness_with_ollama_judge():  # pragma: no cover - needs local LLM
    from deepeval import assert_test
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    from app.evaluation import TC04_ARTICLE
    from app.llm_service import summarize

    s = summarize(dict(id="tc04", title="", body=TC04_ARTICLE, source="Test"))
    case = LLMTestCase(input="Summarise", actual_output=s["summary"], retrieval_context=[TC04_ARTICLE])
    assert_test(case, [FaithfulnessMetric(threshold=0.9, model=CustomOllamaJudge(), async_mode=False)])

