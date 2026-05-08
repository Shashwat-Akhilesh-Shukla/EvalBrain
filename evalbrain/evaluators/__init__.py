from typing import Dict, Type, Any
from evalbrain.evaluators.base import BaseEvaluator
from evalbrain.evaluators.hallucination import HallucinationEvaluator

# Evaluator registry
EVALUATORS: Dict[str, Type[BaseEvaluator]] = {
    "hallucination": HallucinationEvaluator,
}

def get_evaluator(name: str, **kwargs) -> BaseEvaluator:
    """
    Factory function to get an evaluator instance by name.
    
    Args:
        name: The string identifier of the evaluator (e.g., 'hallucination').
        **kwargs: Configuration arguments passed to the evaluator constructor.
        
    Returns:
        BaseEvaluator instance.
    """
    if name not in EVALUATORS:
        raise ValueError(f"Evaluator '{name}' is not registered. Available evaluators: {list(EVALUATORS.keys())}")
    
    evaluator_class = EVALUATORS[name]
    return evaluator_class(**kwargs)

__all__ = ["BaseEvaluator", "HallucinationEvaluator", "get_evaluator"]
