from typing import Any, Dict, List, Optional

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore

class EvalBrainCallbackHandler(BaseCallbackHandler): # type: ignore
    """
    LlamaIndex callback handler that traces LLM calls using EvalBrain.
    """
    def __init__(self, brain: Any):
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError(
                "llama-index-core is required to use EvalBrainCallbackHandler. "
                "Please install it with `pip install llama-index-core`."
            )
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.brain = brain
        self.event_spans = {}

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        if getattr(event_type, "name", str(event_type)) == "LLM":
            span_ctx = self.brain.trace("llamaindex_llm", tags={"event_id": event_id})
            span_ctx.__enter__()
            
            if payload:
                inputs = {}
                for k, v in payload.items():
                    # Check string representation of keys (often enums like EventPayload.PROMPT)
                    key_str = str(k).lower()
                    if "prompt" in key_str or "message" in key_str:
                        inputs[str(k)] = str(v)
                span_ctx.set_input(inputs)
                
            self.event_spans[event_id] = span_ctx
            
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if getattr(event_type, "name", str(event_type)) == "LLM":
            span_ctx = self.event_spans.get(event_id)
            if span_ctx:
                if payload:
                    for k, v in payload.items():
                        if "response" in str(k).lower():
                            span_ctx.set_output(str(v))
                
                span_ctx.__exit__(None, None, None)
                del self.event_spans[event_id]

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass
