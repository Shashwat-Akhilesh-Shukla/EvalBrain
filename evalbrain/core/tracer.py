import uuid
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, List, Optional, Union
from contextvars import ContextVar

from evalbrain.models import Trace, Span, EvalResult

# Context variables for thread/async safety
_current_trace: ContextVar[Optional[Trace]] = ContextVar("current_trace", default=None)
_current_spans: ContextVar[List[Span]] = ContextVar("current_spans", default_factory=list)


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
        self.span.end_time = datetime.utcnow()
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

    def evaluate(self, output: Any = None, context: Any = None, reference: Any = None, evaluators: List[str] = None):
        """
        Manually trigger evaluation for this span.
        Step 6-8 will implement the actual evaluators.
        """
        if output is not None: self.span.output = output
        if context is not None: self.span.context = context
        if reference is not None: self.span.reference = reference
        
        # Placeholder for evaluator logic (Step 6+)
        pass


class EvalBrain:
    def __init__(self, project: str = "default", storage=None):
        self.project = project
        self.storage = storage  # To be implemented in Step 4
        self._local_traces: List[Trace] = [] # Temporary storage for development

    def trace(self, name: str, tags: Dict[str, str] = None) -> SpanContext:
        """Context manager for tracing a block of code."""
        trace = _current_trace.get()
        if not trace:
            trace = Trace(
                trace_id=str(uuid.uuid4()),
                project=self.project,
                tags=tags or {},
                created_at=datetime.utcnow()
            )
            _current_trace.set(trace)

        span = Span(
            span_id=str(uuid.uuid4()),
            name=name,
            start_time=datetime.utcnow()
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
