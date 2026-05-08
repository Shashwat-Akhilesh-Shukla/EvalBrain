import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, List, Optional, Union
from contextvars import ContextVar

from evalbrain.models import Trace, Span, EvalResult

# Context variables for thread/async safety
_current_trace: ContextVar[Optional[Trace]] = ContextVar("current_trace", default=None)
_current_spans: ContextVar[List[Span]] = ContextVar("current_spans", default=[])


class SpanContext:
    def __init__(self, span: Span, brain: "EvalBrain"):
        self.span = span
        self.brain = brain

    def __enter__(self):
        # Add this span to the current stack
        stack = _current_spans.get().copy()
        stack.append(self.span)
        _current_spans.set(stack)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Finalize span
        self.span.end_time = datetime.now(timezone.utc)
        self.span.latency_ms = (self.span.end_time - self.span.start_time).total_seconds() * 1000
        
        # Pop from stack
        stack = _current_spans.get().copy()
        if stack:
            stack.pop()
        _current_spans.set(stack)

        # If stack is empty, it means we finished the top-level span of this trace
        # We should save the trace
        if not stack:
            trace = _current_trace.get()
            if trace:
                self.brain._save_trace(trace)
                _current_trace.set(None)

    def set_input(self, data: Any):
        self.span.input = data

    def set_output(self, data: Any):
        self.span.output = data

    def set_context(self, data: Any):
        self.span.context = data

    def set_reference(self, data: Any):
        self.span.reference = data

    def set_metadata(self, key: str, value: Any):
        self.span.metadata[key] = value

    def set_tokens(self, input_tokens: int, output_tokens: int, model_name: Optional[str] = None):
        """Record token usage and calculate cost if a cost tracker is configured."""
        self.span.token_counts = {
            "input": input_tokens,
            "output": output_tokens,
            "total": input_tokens + output_tokens
        }
        if model_name:
            self.span.model_name = model_name
            
        # Calculate cost if a cost tracker is configured
        if hasattr(self.brain, "cost_tracker") and self.brain.cost_tracker and self.span.model_name:
            self.span.cost_usd = self.brain.cost_tracker.calculate_cost(
                self.span.model_name, input_tokens, output_tokens
            )

    def evaluate(self, output: Any = None, context: Any = None, reference: Any = None, evaluators: List[Union[str, Any]] = None):
        """
        Manually trigger evaluation for this span.
        """
        if output is not None: self.span.output = output
        if context is not None: self.span.context = context
        if reference is not None: self.span.reference = reference
        
        if not evaluators:
            return
            
        from evalbrain.evaluators import get_evaluator
        
        for evaluator_item in evaluators:
            if isinstance(evaluator_item, str):
                evaluator_inst = get_evaluator(evaluator_item)
            else:
                evaluator_inst = evaluator_item
                
            result = evaluator_inst.evaluate(
                output=self.span.output,
                context=self.span.context,
                reference=self.span.reference
            )
            self.span.eval_results.append(result)


class EvalBrain:
    def __init__(self, project: str = "default", storage=None, cost_config=None):
        self.project = project
        self.storage = storage  # To be implemented in Step 4
        
        # Initialize Cost Tracker
        from evalbrain.trackers.cost import CostTracker
        self.cost_tracker = CostTracker(config=cost_config)
        
        self._local_traces: List[Trace] = [] # Temporary storage for development

    def trace(self, name: str, tags: Dict[str, str] = None) -> SpanContext:
        """Context manager for tracing a block of code."""
        trace = _current_trace.get()
        if not trace:
            trace = Trace(
                trace_id=str(uuid.uuid4()),
                project=self.project,
                tags=tags or {},
                created_at=datetime.now(timezone.utc)
            )
            _current_trace.set(trace)

        span = Span(
            span_id=str(uuid.uuid4()),
            name=name,
            start_time=datetime.now(timezone.utc)
        )
        trace.spans.append(span)
        
        return SpanContext(span, self)

    def eval(self, name: Optional[str] = None, tags: Dict[str, str] = None):
        """Decorator for tracing a function."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                span_name = name or func.__name__
                # Combine tags?
                with self.trace(span_name, tags=tags) as span:
                    span.set_input({"args": args, "kwargs": kwargs})
                    result = func(*args, **kwargs)
                    span.set_output(result)
                    return result
            return wrapper
        return decorator

    def _save_trace(self, trace: Trace):
        """Internal method to persist the trace."""
        if self.storage:
            # self.storage.save(trace)
            pass
        else:
            # Fallback to local list for now
            self._local_traces.append(trace)
            print(f"[EvalBrain] Trace saved: {trace.trace_id} ({len(trace.spans)} spans)")

    def get_traces(self) -> List[Trace]:
        """Return captured traces (useful for testing)."""
        return self._local_traces
