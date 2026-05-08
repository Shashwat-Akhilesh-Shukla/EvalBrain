from typing import List, Optional
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, JSON, ForeignKey, Boolean, Integer
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class TraceTable(Base):
    __tablename__ = "traces"
    
    trace_id: Mapped[str] = mapped_column(String, primary_key=True)
    project: Mapped[str] = mapped_column(String, index=True)
    tags: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    
    spans: Mapped[List["SpanTable"]] = relationship(
        "SpanTable", back_populates="trace", cascade="all, delete-orphan"
    )

class SpanTable(Base):
    __tablename__ = "spans"
    
    span_id: Mapped[str] = mapped_column(String, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String, ForeignKey("traces.trace_id"), index=True)
    name: Mapped[str] = mapped_column(String)
    input: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    context: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    reference: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    start_time: Mapped[datetime] = mapped_column(DateTime)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    token_counts: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    span_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    
    trace: Mapped["TraceTable"] = relationship("TraceTable", back_populates="spans")
    eval_results: Mapped[List["EvalResultTable"]] = relationship(
        "EvalResultTable", back_populates="span", cascade="all, delete-orphan"
    )

class EvalResultTable(Base):
    __tablename__ = "eval_results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    span_id: Mapped[str] = mapped_column(String, ForeignKey("spans.span_id"), index=True)
    metric_name: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)
    explanation: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    eval_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    
    span: Mapped["SpanTable"] = relationship("SpanTable", back_populates="eval_results")

class PromptVersionTable(Base):
    __tablename__ = "prompt_versions"
    
    prompt_id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String, primary_key=True)
    template: Mapped[str] = mapped_column(String)
    prompt_hash: Mapped[str] = mapped_column(String, index=True)
    tags: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    prompt_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
