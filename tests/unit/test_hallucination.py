import json
import pytest
from unittest.mock import patch, MagicMock
from evalbrain.evaluators.hallucination import HallucinationEvaluator
from evalbrain.models import EvalResult

def test_hallucination_llm_method():
    # Mock LLM callable
    def mock_llm(prompt):
        return json.dumps({
            "score": 0.9,
            "unsupported_claims": [],
            "explanation": "Perfectly faithful."
        })

    evaluator = HallucinationEvaluator(method="llm", llm_callable=mock_llm, threshold=0.8)
    
    result = evaluator.evaluate(
        output="The Eiffel Tower is in Paris.",
        context="Paris is the capital of France, home to the Eiffel Tower."
    )
    
    assert isinstance(result, EvalResult)
    assert result.metric_name == "faithfulness"
    assert result.score == 0.9
    assert result.passed is True
    assert result.metadata["unsupported_claims"] == []

def test_hallucination_llm_method_failure():
    def mock_llm(prompt):
        return '```json\n{"score": 0.4, "unsupported_claims": ["London"], "explanation": "Mentions London."}\n```'

    evaluator = HallucinationEvaluator(method="llm", llm_callable=mock_llm, threshold=0.8)
    
    result = evaluator.evaluate(
        output="The Eiffel Tower is in London.",
        context="Paris is the capital of France, home to the Eiffel Tower."
    )
    
    assert result.score == 0.4
    assert result.passed is False
    assert "London" in result.metadata["unsupported_claims"]

@patch('evalbrain.evaluators.hallucination.HallucinationEvaluator._load_nli_model')
def test_hallucination_nli_method(mock_load):
    # Instantiate with NLI
    evaluator = HallucinationEvaluator(method="nli", threshold=0.8)
    
    # Mock the cross encoder model
    mock_model = MagicMock()
    # CrossEncoder returns logits: [Contradiction, Entailment, Neutral]
    # For high entailment, the middle logit should be highest
    import numpy as np
    # e.g., scores = [-2.0, 3.0, -1.0] -> Entailment is high
    mock_model.predict.return_value = np.array([[-2.0, 3.0, -1.0]])
    evaluator._nli_model = mock_model
    
    result = evaluator.evaluate(
        output="The sky is blue.",
        context="The color of the sky is blue."
    )
    
    assert result.metric_name == "faithfulness"
    assert result.score > 0.8  # Softmax of [ -2, 3, -1 ] will give ~0.98 for index 1
    assert result.passed is True
    assert result.metadata["method"] == "nli"

def test_hallucination_no_context():
    evaluator = HallucinationEvaluator(method="llm", llm_callable=lambda x: "{}")
    result = evaluator.evaluate(output="Hello")
    
    assert result.score == 0.0
    assert result.passed is False
    assert result.metadata["error"] == "missing_context"
