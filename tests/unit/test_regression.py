import os
import yaml  # type: ignore
import pytest
from evalbrain import EvalBrain
from evalbrain.models import EvalResult
from evalbrain.evaluators.base import BaseEvaluator
from evalbrain.regression.suite import SuiteResult, load_suite

class DummyEvaluator(BaseEvaluator):
    def evaluate(self, output, context=None, reference=None, **kwargs):
        # Dummy evaluator returns 0.9 if output matches reference, else 0.5
        score = 0.9 if output == reference else 0.5
        return EvalResult(
            metric_name="correctness",
            score=score,
            passed=score >= 0.8,
            explanation="Dummy eval",
            metadata={}
        )

@pytest.fixture
def dummy_yaml_path(tmp_path):
    yaml_content = [
        {
            "id": "tc_001",
            "query": "What is 2+2?",
            "context": "Math context.",
            "reference": "4",
            "thresholds": {"correctness": 0.8}
        },
        {
            "id": "tc_002",
            "query": "What is 3+3?",
            "context": "Math context.",
            "reference": "6",
            "thresholds": {"correctness": 0.8}
        }
    ]
    path = tmp_path / "test_suite.yaml"
    with open(path, "w") as f:
        yaml.dump(yaml_content, f)
    return str(path)

def test_load_suite(dummy_yaml_path):
    suite = load_suite(dummy_yaml_path)
    assert suite.id == "default-suite"
    assert len(suite.test_cases) == 2
    assert suite.test_cases[0].id == "tc_001"
    assert suite.test_cases[0].thresholds["correctness"] == 0.8

def test_run_suite_all_pass(dummy_yaml_path):
    brain = EvalBrain(project="test-project")
    
    # Target function that always gives the correct answer based on reference
    # Wait, target_func doesn't receive reference. It receives query and context.
    def target_func(query, context=None):
        if "2+2" in query: return "4"
        if "3+3" in query: return "6"
        return "unknown"
        
    result = brain.run_suite(
        suite_source=dummy_yaml_path,
        target_func=target_func,
        evaluators=[DummyEvaluator()]
    )
    
    assert isinstance(result, SuiteResult)
    assert result.passed is True
    assert result.total_cases == 2
    assert result.passed_cases == 2
    assert result.failed_cases == 0

def test_run_suite_one_fails(dummy_yaml_path):
    brain = EvalBrain(project="test-project")
    
    # Target function that gets the second one wrong
    def target_func(query, context=None):
        if "2+2" in query: return "4"
        if "3+3" in query: return "5" # Wrong answer
        return "unknown"
        
    result = brain.run_suite(
        suite_source=dummy_yaml_path,
        target_func=target_func,
        evaluators=[DummyEvaluator()]
    )
    
    assert isinstance(result, SuiteResult)
    assert result.passed is False
    assert result.total_cases == 2
    assert result.passed_cases == 1
    assert result.failed_cases == 1
    
    tc2_result = [r for r in result.results if r.test_case_id == "tc_002"][0]
    assert tc2_result.passed is False
    assert tc2_result.metrics["correctness"] == 0.5
