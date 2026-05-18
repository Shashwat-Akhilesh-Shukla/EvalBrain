from typing import Any, Dict, List, Optional

try:
    from langchain_core.callbacks import BaseCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore

class EvalBrainCallbackHandler(BaseCallbackHandler): # type: ignore
    """
    LangChain callback handler that traces LLM calls using EvalBrain.
    """
    def __init__(self, brain: Any):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required to use EvalBrainCallbackHandler. "
                "Please install it with `pip install langchain-core`."
            )
        super().__init__()
        self.brain = brain
        self.run_spans: Dict[Any, Any] = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""
        span_ctx = self.brain.trace("langchain_llm", tags={"run_id": str(run_id)})
        span_ctx.__enter__()
        
        span_ctx.set_input({"prompts": prompts, "serialized": serialized})
        if metadata:
            for k, v in metadata.items():
                span_ctx.set_metadata(k, v)
                
        self.run_spans[run_id] = span_ctx

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running."""
        span_ctx = self.run_spans.get(run_id)
        if span_ctx:
            outputs = [[gen.text for gen in gens] for gens in response.generations]
            span_ctx.set_output(outputs)
            
            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                span_ctx.set_tokens(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    model_name=response.llm_output.get("model_name", "unknown")
                )
                
            span_ctx.__exit__(None, None, None)
            del self.run_spans[run_id]

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""
        span_ctx = self.run_spans.get(run_id)
        if span_ctx:
            span_ctx.__exit__(type(error), error, error.__traceback__)
            del self.run_spans[run_id]
