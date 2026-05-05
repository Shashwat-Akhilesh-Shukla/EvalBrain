from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class EvalResult(BaseModel):
    """Output of any evaluator."""
    model_config = ConfigDict(populate_by_name=True)

    metric_name: str
    score: float  # 0.0 – 1.0
    passed: bool
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    """A named unit within a trace (retrieval, generation, rerank)."""
    model_config = ConfigDict(populate_by_name=True)

    span_id: str
    name: str
    input: Optional[Any] = None
    output: Optional[Any] = None
    context: Optional[Any] = None
    reference: Optional[Any] = None
    eval_results: List[EvalResult] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    token_counts: Optional[Dict[str, int]] = None  # {"input": X, "output": Y, "total": Z}
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """A single LLM call record comprising multiple spans."""
    model_config = ConfigDict(populate_by_name=True)

    trace_id: str
    project: str
    tags: Dict[str, str] = Field(default_factory=dict)
    spans: List[Span] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PromptVersion(BaseModel):
    """Snapshot of a prompt template."""
    model_config = ConfigDict(populate_by_name=True)

    prompt_id: str
    version: str
    template: str
    prompt_hash: str  # SHA-256 hash of the template
    tags: Dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):
    """A single test case for regression testing."""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    query: str
    context: Optional[str] = None
    reference: Optional[str] = None
    thresholds: Dict[str, float] = Field(default_factory=dict)  # e.g., {"faithfulness": 0.8}
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegressionSuite(BaseModel):
    """A collection of test cases."""
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    test_cases: List[TestCase] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
