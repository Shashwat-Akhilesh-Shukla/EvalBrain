import json
from typing import Any, Callable, Dict, List, Optional
from evalbrain.models import EvalResult
from evalbrain.evaluators.base import BaseEvaluator

class HallucinationEvaluator(BaseEvaluator):
    """
    Evaluates whether the generated output contains hallucinations
    (claims not supported by the retrieved context).
    """
    
    def __init__(
        self,
        method: str = "llm",
        threshold: float = 0.5,
        llm_callable: Optional[Callable[[str], str]] = None,
        model_name: str = "cross-encoder/nli-deberta-v3-base"
    ):
        """
        Args:
            method: 'llm' or 'nli'.
            threshold: Minimum score to pass (0.0 to 1.0).
            llm_callable: A function that takes a prompt string and returns an LLM response string.
                          Required if method='llm'.
            model_name: The HuggingFace model name to use if method='nli'.
        """
        self.method = method
        self.threshold = threshold
        self.llm_callable = llm_callable
        self.model_name = model_name
        self._nli_model = None
        
        if self.method not in ("llm", "nli"):
            raise ValueError("Method must be 'llm' or 'nli'")
            
        if self.method == "llm" and not self.llm_callable:
            raise ValueError("llm_callable must be provided when using method='llm'")
            
        if self.method == "nli":
            self._load_nli_model()
            
    def _load_nli_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self._nli_model = CrossEncoder(self.model_name)
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' package is required for the NLI method. "
                "Install it with: pip install sentence-transformers"
            )

    def evaluate(
        self,
        output: Any,
        context: Optional[Any] = None,
        reference: Optional[Any] = None,
        **kwargs
    ) -> EvalResult:
        if not context:
            return EvalResult(
                metric_name="faithfulness",
                score=0.0,
                passed=False,
                explanation="No context provided to evaluate hallucination against.",
                metadata={"error": "missing_context"}
            )
            
        output_str = str(output)
        context_str = str(context)

        if self.method == "llm":
            return self._evaluate_llm(output_str, context_str)
        else:
            return self._evaluate_nli(output_str, context_str)

    def _evaluate_llm(self, output: str, context: str) -> EvalResult:
        prompt = f"""
You are an expert evaluator. Your task is to determine if the given OUTPUT contains any hallucinations based on the provided CONTEXT.
A hallucination is any claim, fact, or statement in the OUTPUT that cannot be logically deduced or supported by the CONTEXT.

CONTEXT:
{context}

OUTPUT:
{output}

Analyze the OUTPUT sentence by sentence.
Calculate a faithfulness score from 0.0 (completely hallucinated) to 1.0 (completely supported).
List any unsupported claims.

Respond ONLY with a valid JSON object in the following format:
{{
    "score": 0.8,
    "unsupported_claims": ["claim 1", "claim 2"],
    "explanation": "Brief reasoning for the score."
}}
"""
        try:
            response_text = self.llm_callable(prompt)
            # Basic cleanup in case the LLM returned markdown blocks
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            result_data = json.loads(response_text.strip())
            
            score = float(result_data.get("score", 0.0))
            unsupported_claims = result_data.get("unsupported_claims", [])
            explanation = result_data.get("explanation", "")
            
            return EvalResult(
                metric_name="faithfulness",
                score=score,
                passed=score >= self.threshold,
                explanation=explanation,
                metadata={"unsupported_claims": unsupported_claims, "method": "llm"}
            )
        except Exception as e:
            return EvalResult(
                metric_name="faithfulness",
                score=0.0,
                passed=False,
                explanation=f"LLM Evaluation failed: {str(e)}",
                metadata={"method": "llm", "error": str(e)}
            )

    def _evaluate_nli(self, output: str, context: str) -> EvalResult:
        # NLI typically takes (premise, hypothesis)
        # Here premise = context, hypothesis = output
        # CrossEncoder output for nli-deberta-v3-base: [Contradiction, Entailment, Neutral]
        try:
            import numpy as np
            # To handle long outputs, ideally we'd split into sentences, but for simplicity we'll pass it as is
            # or you can chunk it. Here we do a single pass.
            scores = self._nli_model.predict([(context, output)])
            # Apply softmax to get probabilities
            exp_scores = np.exp(scores[0])
            probs = exp_scores / np.sum(exp_scores)
            
            # Index 1 is Entailment for cross-encoder/nli-deberta-v3-base
            entailment_prob = float(probs[1])
            contradiction_prob = float(probs[0])
            neutral_prob = float(probs[2])
            
            # Using entailment prob as the faithfulness score
            score = entailment_prob
            
            explanation = f"Entailment: {entailment_prob:.2f}, Contradiction: {contradiction_prob:.2f}, Neutral: {neutral_prob:.2f}"
            
            return EvalResult(
                metric_name="faithfulness",
                score=score,
                passed=score >= self.threshold,
                explanation=explanation,
                metadata={
                    "method": "nli",
                    "entailment": entailment_prob,
                    "contradiction": contradiction_prob,
                    "neutral": neutral_prob
                }
            )
        except Exception as e:
            return EvalResult(
                metric_name="faithfulness",
                score=0.0,
                passed=False,
                explanation=f"NLI Evaluation failed: {str(e)}",
                metadata={"method": "nli", "error": str(e)}
            )
