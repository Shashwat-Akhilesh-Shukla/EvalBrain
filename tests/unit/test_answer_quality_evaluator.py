import pytest
import json
from evalbrain.evaluators.answer_quality import AnswerQualityEvaluator
from evalbrain.evaluators import get_evaluator

def test_initialization():
    evaluator = AnswerQualityEvaluator()
    assert evaluator.method == "llm"
    assert evaluator.metrics == ["correctness", "completeness", "conciseness", "toxicity"]

    with pytest.raises(ValueError, match="Method must be"):
        AnswerQualityEvaluator(method="invalid_method")

def test_get_evaluator_registration():
    evaluator = get_evaluator("answer_quality")
    assert isinstance(evaluator, AnswerQualityEvaluator)

def test_missing_output():
    evaluator = AnswerQualityEvaluator()
    result = evaluator.evaluate(output="", query="test")
    assert result.passed is False
    assert result.score == 0.0
    assert result.metadata["error"] == "missing_output"

def test_rouge_missing_reference():
    evaluator = AnswerQualityEvaluator(method="rouge")
    result = evaluator.evaluate(output="answer", reference="")
    assert result.passed is False
    assert result.score == 0.0
    assert result.metadata["error"] == "missing_reference"

def test_bertscore_missing_reference():
    evaluator = AnswerQualityEvaluator(method="bertscore")
    result = evaluator.evaluate(output="answer", reference="")
    assert result.passed is False
    assert result.score == 0.0
    assert result.metadata["error"] == "missing_reference"

def test_llm_missing_query_and_reference():
    evaluator = AnswerQualityEvaluator(method="llm")
    result = evaluator.evaluate(output="answer")
    assert result.passed is False
    assert result.score == 0.0
    assert result.metadata["error"] == "missing_query_and_reference"

def test_llm_evaluation_success():
    def mock_llm_callable(prompt: str) -> str:
        return json.dumps({
            "correctness": 0.9,
            "completeness": 0.8,
            "conciseness": 0.7,
            "toxicity": 1.0,
            "explanation": "Good answer."
        })

    evaluator = AnswerQualityEvaluator(method="llm", llm_callable=mock_llm_callable)
    result = evaluator.evaluate(output="Paris is the capital.", query="What is the capital of France?", reference="Paris")
    
    assert result.passed is True
    assert result.score == pytest.approx((0.9 + 0.8 + 0.7 + 1.0) / 4)
    assert result.metadata["method"] == "llm"
    assert result.metadata["individual_scores"]["correctness"] == 0.9
    assert result.metadata["individual_scores"]["toxicity"] == 1.0
    assert result.explanation == "Good answer."

def test_llm_evaluation_failure():
    def mock_llm_callable(prompt: str) -> str:
        raise ValueError("API Error")

    evaluator = AnswerQualityEvaluator(method="llm", llm_callable=mock_llm_callable)
    result = evaluator.evaluate(output="Answer", query="Query")
    
    assert result.passed is False
    assert result.score == 0.0
    assert result.metadata["method"] == "llm"
    assert "API Error" in result.metadata["error"]

def test_rouge_evaluation_import_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "rouge_score", None)
    
    evaluator = AnswerQualityEvaluator(method="rouge")
    with pytest.raises(ImportError, match="The 'rouge-score' package is required"):
        evaluator.evaluate(output="test", reference="test")

def test_bertscore_evaluation_import_error(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "evaluate", None)
    
    evaluator = AnswerQualityEvaluator(method="bertscore")
    with pytest.raises(ImportError, match="The 'evaluate' and 'bert_score' packages are required"):
        evaluator.evaluate(output="test", reference="test")
