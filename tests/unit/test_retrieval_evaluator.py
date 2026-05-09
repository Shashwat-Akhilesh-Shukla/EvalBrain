import pytest
from evalbrain.evaluators.retrieval import RetrievalEvaluator

def test_heuristic_evaluator_no_reference():
    evaluator = RetrievalEvaluator(method="heuristic")
    result = evaluator.evaluate(output=["doc1", "doc2"])
    assert result.passed is False
    assert "requires reference documents" in result.explanation
    assert result.metadata["error"] == "missing_reference"

def test_heuristic_evaluator_no_output():
    evaluator = RetrievalEvaluator(method="heuristic")
    result = evaluator.evaluate(output=[], reference=["ref"])
    assert result.passed is False
    assert "No retrieved documents" in result.explanation

def test_heuristic_hit_rate():
    evaluator = RetrievalEvaluator(method="heuristic", metrics=["hit_rate"], k=2)
    # Target is in top 2
    result = evaluator.evaluate(output=["wrong doc", "correct doc info", "another"], reference=["correct doc"])
    assert result.score == 1.0
    assert result.passed is True
    assert result.metadata["individual_scores"]["hit_rate"] == 1.0

    # Target is not in top 2
    result2 = evaluator.evaluate(output=["wrong doc", "another wrong", "correct doc info"], reference=["correct doc"])
    assert result2.score == 0.0
    assert result2.passed is False
    assert result2.metadata["individual_scores"]["hit_rate"] == 0.0

def test_heuristic_mrr():
    evaluator = RetrievalEvaluator(method="heuristic", metrics=["mrr"])
    # Target is at rank 2 (index 1) => MRR = 1/2 = 0.5
    result = evaluator.evaluate(output=["wrong doc", "correct doc info", "another"], reference=["correct doc"])
    assert result.score == 0.5
    assert result.passed is True  # 0.5 >= 0.5
    assert result.metadata["individual_scores"]["mrr"] == 0.5

def test_heuristic_context_precision_and_recall():
    evaluator = RetrievalEvaluator(method="heuristic", metrics=["context_precision", "context_recall"])
    # 2 retrieved docs. 1 is relevant. Precision = 1/2 = 0.5
    # 2 reference docs. 1 is found. Recall = 1/2 = 0.5
    result = evaluator.evaluate(
        output=["correct doc 1 info", "wrong doc"],
        reference=["correct doc 1", "unrelated topic"]
    )
    assert result.score == 0.5
    assert result.metadata["individual_scores"]["context_precision"] == 0.5
    assert result.metadata["individual_scores"]["context_recall"] == 0.5

def test_llm_evaluator_success():
    def mock_llm(prompt: str) -> str:
        return '''```json
        {
            "context_precision": 0.5,
            "context_recall": 1.0,
            "hit_rate": 1.0,
            "mrr": 0.5,
            "explanation": "Found it!"
        }
        ```'''
    
    evaluator = RetrievalEvaluator(method="llm", llm_callable=mock_llm)
    result = evaluator.evaluate(
        output=["doc1", "doc2"],
        input="what is doc?",
        reference=["doc1 is a doc"]
    )
    
    assert result.score == 0.75  # (0.5 + 1.0 + 1.0 + 0.5) / 4
    assert result.passed is True
    assert result.metadata["individual_scores"]["context_precision"] == 0.5
    assert result.explanation == "Found it!"
    
def test_llm_evaluator_failure():
    def mock_llm_error(prompt: str) -> str:
        raise ValueError("API Error")
        
    evaluator = RetrievalEvaluator(method="llm", llm_callable=mock_llm_error)
    result = evaluator.evaluate(
        output=["doc1", "doc2"],
        input="what is doc?",
    )
    assert result.passed is False
    assert result.score == 0.0
    assert "LLM Evaluation failed: API Error" in result.explanation
