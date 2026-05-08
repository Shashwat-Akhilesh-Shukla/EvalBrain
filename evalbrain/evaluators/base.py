from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from evalbrain.models import EvalResult

class BaseEvaluator(ABC):
    """Abstract base class for all EvalBrain evaluators."""

    @abstractmethod
    def evaluate(
        self,
        output: Any,
        context: Optional[Any] = None,
        reference: Optional[Any] = None,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate the given output against context and/or reference.
        
        Args:
            output: The generated text or output to evaluate.
            context: The retrieved context used to generate the output.
            reference: The ground truth reference text.
            **kwargs: Additional evaluator-specific parameters.
            
        Returns:
            EvalResult: The result of the evaluation.
        """
        pass
