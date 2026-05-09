import json
from typing import Any, Callable, Dict, List, Optional
from evalbrain.models import EvalResult
from evalbrain.evaluators.base import BaseEvaluator

class RetrievalEvaluator(BaseEvaluator):
    """
    Evaluates the quality of retrieved documents.
    """
    
    def __init__(
        self,
        method: str = "heuristic",
        metrics: Optional[List[str]] = None,
        k: int = 5,
        threshold: float = 0.5,
        llm_callable: Optional[Callable[[str], str]] = None
    ):
        """
        Args:
            method: 'heuristic' (token overlap/exact match) or 'llm' (LLM as judge).
            metrics: List of metrics to compute. Defaults to ["hit_rate", "mrr", "context_precision", "context_recall"].
            k: Top-K value for hit_rate@k.
            threshold: Minimum score to pass (0.0 to 1.0).
            llm_callable: A function that takes a prompt string and returns an LLM response string.
                          Required if method='llm'.
        """
        self.method = method
        self.metrics = metrics or ["hit_rate", "mrr", "context_precision", "context_recall"]
        self.k = k
        self.threshold = threshold
        self.llm_callable = llm_callable
        
        if self.method not in ("heuristic", "llm"):
            raise ValueError("Method must be 'heuristic' or 'llm'")

    def _get_llm_callable(self):
        if self.llm_callable:
            return self.llm_callable
        
        try:
            import openai
            import os
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy"))
            def default_openai_callable(prompt: str) -> str:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return response.choices[0].message.content
            return default_openai_callable
        except ImportError:
            raise ValueError(
                "No llm_callable provided and 'openai' package is not installed. "
                "Either provide an llm_callable or run `pip install openai`."
            )

    def _to_list(self, val: Any) -> List[str]:
        if not val:
            return []
        if isinstance(val, str):
            return [val]
        if isinstance(val, list):
            return [str(v) for v in val]
        return [str(val)]

    def evaluate(
        self,
        output: Any,
        context: Optional[Any] = None,
        reference: Optional[Any] = None,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate retrieved documents (output) against a query (input in kwargs) or reference documents (reference).
        
        In retrieval, 'output' usually represents the retrieved documents.
        """
        retrieved_docs = self._to_list(output)
        reference_docs = self._to_list(reference)
        query = kwargs.get("input", "") or kwargs.get("query", "")
        
        if not retrieved_docs:
            return EvalResult(
                metric_name="retrieval_quality",
                score=0.0,
                passed=False,
                explanation="No retrieved documents provided.",
                metadata={"error": "missing_output"}
            )
            
        if self.method == "heuristic":
            if not reference_docs:
                return EvalResult(
                    metric_name="retrieval_quality",
                    score=0.0,
                    passed=False,
                    explanation="Heuristic method requires reference documents to evaluate against.",
                    metadata={"error": "missing_reference"}
                )
            return self._evaluate_heuristic(retrieved_docs, reference_docs)
        else:
            if not query and not reference_docs:
                 return EvalResult(
                    metric_name="retrieval_quality",
                    score=0.0,
                    passed=False,
                    explanation="LLM method requires either a query or reference documents.",
                    metadata={"error": "missing_query_and_reference"}
                )
            return self._evaluate_llm(retrieved_docs, query=str(query), reference_docs=reference_docs)

    def _is_match(self, retrieved: str, reference: str) -> bool:
        """Simple heuristic match: token overlap > 50% or reference is substring of retrieved."""
        if reference.lower() in retrieved.lower():
            return True
            
        ret_tokens = set(retrieved.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not ref_tokens:
            return False
            
        overlap = len(ret_tokens.intersection(ref_tokens))
        if overlap / len(ref_tokens) > 0.5:
            return True
            
        return False

    def _evaluate_heuristic(self, retrieved_docs: List[str], reference_docs: List[str]) -> EvalResult:
        scores = {}
        
        # Calculate Hit Rate @ K
        if "hit_rate" in self.metrics:
            hit = False
            for i, doc in enumerate(retrieved_docs[:self.k]):
                if any(self._is_match(doc, ref) for ref in reference_docs):
                    hit = True
                    break
            scores["hit_rate"] = 1.0 if hit else 0.0

        # Calculate MRR
        if "mrr" in self.metrics:
            mrr = 0.0
            for i, doc in enumerate(retrieved_docs):
                if any(self._is_match(doc, ref) for ref in reference_docs):
                    mrr = 1.0 / (i + 1)
                    break
            scores["mrr"] = mrr

        # Calculate Context Precision
        if "context_precision" in self.metrics:
            relevant_retrieved = sum(1 for doc in retrieved_docs if any(self._is_match(doc, ref) for ref in reference_docs))
            scores["context_precision"] = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0.0

        # Calculate Context Recall
        if "context_recall" in self.metrics:
            found_refs = sum(1 for ref in reference_docs if any(self._is_match(doc, ref) for doc in retrieved_docs))
            scores["context_recall"] = found_refs / len(reference_docs) if reference_docs else 0.0

        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        explanation = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
        
        return EvalResult(
            metric_name="retrieval_quality",
            score=overall_score,
            passed=overall_score >= self.threshold,
            explanation=f"Heuristic evaluation - {explanation}",
            metadata={"method": "heuristic", "individual_scores": scores}
        )

    def _evaluate_llm(self, retrieved_docs: List[str], query: str, reference_docs: List[str]) -> EvalResult:
        prompt = f"""
You are an expert relevance evaluator. 
Your task is to evaluate the quality of retrieved documents for a given query and/or reference answer.

QUERY: {query}
REFERENCE DOCUMENTS/ANSWER: {json.dumps(reference_docs)}

RETRIEVED DOCUMENTS:
"""
        for i, doc in enumerate(retrieved_docs):
            prompt += f"[{i+1}] {doc}\n"

        prompt += f"""
Calculate the following metrics (0.0 to 1.0) based ONLY on the semantic relevance to the query/reference:
- "context_precision": Fraction of retrieved documents that are relevant.
- "context_recall": Fraction of the reference information/query intent covered by the retrieved documents.

Also determine if there is a 'hit' (a highly relevant document) in the top {self.k} docs (return 1.0 for true, 0.0 for false) as "hit_rate", and calculate the Mean Reciprocal Rank ("mrr") as 1.0 / rank of the first relevant document (or 0.0 if none).

Respond ONLY with a valid JSON object in the following format:
{{
    "context_precision": 0.8,
    "context_recall": 0.7,
    "hit_rate": 1.0,
    "mrr": 0.5,
    "explanation": "Brief reasoning for the scores."
}}
"""
        try:
            callable_func = self._get_llm_callable()
            response_text = callable_func(prompt)
            
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
                
            response_text = response_text.strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            result_data = json.loads(response_text)
            
            scores = {}
            for m in self.metrics:
                if m in result_data:
                    scores[m] = float(result_data[m])
                    
            explanation = result_data.get("explanation", "")
            overall_score = sum(scores.values()) / len(scores) if scores else 0.0

            return EvalResult(
                metric_name="retrieval_quality",
                score=overall_score,
                passed=overall_score >= self.threshold,
                explanation=explanation,
                metadata={"method": "llm", "individual_scores": scores}
            )
        except Exception as e:
            return EvalResult(
                metric_name="retrieval_quality",
                score=0.0,
                passed=False,
                explanation=f"LLM Evaluation failed: {str(e)}",
                metadata={"method": "llm", "error": str(e)}
            )
