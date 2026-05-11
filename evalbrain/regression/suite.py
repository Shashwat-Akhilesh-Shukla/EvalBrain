import json
import yaml  # type: ignore
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, ConfigDict
from evalbrain.models import RegressionSuite, TestCase

if TYPE_CHECKING:
    from evalbrain.core.tracer import EvalBrain


class TestCaseResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())
    
    test_case_id: str
    passed: bool
    metrics: Dict[str, float]
    errors: List[str]


class SuiteResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())
    
    suite_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    results: List[TestCaseResult]
    passed: bool


def load_suite(file_path: str) -> RegressionSuite:
    """Loads a RegressionSuite from a YAML or JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".yaml") or file_path.endswith(".yml"):
            data = yaml.safe_load(f)
        elif file_path.endswith(".json"):
            data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
            
    # Assuming data is a list of test cases or a dict with suite info
    if isinstance(data, list):
        # Infer suite from list of test cases
        return RegressionSuite(
            id="default-suite",
            name="Default Suite",
            test_cases=[TestCase(**tc) for tc in data]
        )
    elif isinstance(data, dict):
        return RegressionSuite(**data)
    else:
        raise ValueError("Invalid suite data format")


class RegressionRunner:
    def __init__(self, brain: "EvalBrain"):
        self.brain = brain
        
    def run(
        self, 
        suite: RegressionSuite, 
        target_func: Callable[..., Any], 
        evaluators: Optional[List[Union[str, Any]]] = None
    ) -> SuiteResult:
        """Runs the regression suite and returns the results."""
        results = []
        passed_count = 0
        
        for case in suite.test_cases:
            case_passed = True
            case_metrics = {}
            case_errors = []
            
            # Trace the execution of the target function
            with self.brain.trace(f"test_case_{case.id}") as span:
                try:
                    # Provide query and context to target function
                    sig = inspect.signature(target_func)
                    kwargs = {}
                    
                    if "query" in sig.parameters:
                        kwargs["query"] = case.query
                    elif len(sig.parameters) > 0:
                        # If first arg is not named query, try passing query positionally
                        first_param = list(sig.parameters.values())[0]
                        if first_param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
                            pass # We'll handle this specially below
                        
                    if "context" in sig.parameters and case.context is not None:
                        kwargs["context"] = case.context
                        
                    if kwargs:
                        output = target_func(**kwargs)
                    else:
                        # Fallback for simple single-arg functions
                        output = target_func(case.query)
                        
                    span.evaluate(
                        output=output, 
                        context=case.context, 
                        reference=case.reference, 
                        evaluators=evaluators
                    )
                    
                    # Collect metrics from eval_results
                    # If multiple evaluators return the same metric_name, we overwrite (last wins)
                    # Ideally, evaluators should return distinct metric_names.
                    for eval_result in span.span.eval_results:
                        metric = eval_result.metric_name
                        score = eval_result.score
                        case_metrics[metric] = score
                        
                        # Also extract individual scores if they were returned in metadata
                        if "individual_scores" in eval_result.metadata:
                            for ind_metric, ind_score in eval_result.metadata["individual_scores"].items():
                                case_metrics[ind_metric] = ind_score
                        
                    # Check thresholds
                    for metric, threshold in case.thresholds.items():
                        if metric in case_metrics:
                            score = case_metrics[metric]
                            if score < threshold:
                                case_passed = False
                                case_errors.append(f"Metric '{metric}' scored {score:.2f}, below threshold {threshold:.2f}")
                        else:
                            case_passed = False
                            case_errors.append(f"Metric '{metric}' was not computed.")
                                
                except Exception as e:
                    case_passed = False
                    case_errors.append(f"Execution failed: {str(e)}")
                    
            if case_passed:
                passed_count += 1
                
            results.append(
                TestCaseResult(
                    test_case_id=case.id,
                    passed=case_passed,
                    metrics=case_metrics,
                    errors=case_errors
                )
            )
            
        total_cases = len(suite.test_cases)
        return SuiteResult(
            suite_id=suite.id,
            total_cases=total_cases,
            passed_cases=passed_count,
            failed_cases=total_cases - passed_count,
            results=results,
            passed=passed_count == total_cases
        )
