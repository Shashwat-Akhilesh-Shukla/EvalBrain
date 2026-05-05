# Changelog

All notable changes to EvalBrain will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Project scaffolding and package structure
- Core Pydantic v2 data models (`Trace`, `Span`, `EvalResult`, `PromptVersion`)
- Storage backends: SQLite, PostgreSQL, JSON file
- Tracer with context manager and decorator APIs
- Latency and cost tracking
- Hallucination evaluator (LLM-judge + NLI-based)
- Retrieval quality evaluator (precision, recall, MRR, Hit@K)
- Answer quality evaluator (correctness, completeness, ROUGE-L, BERTScore)
- Prompt version registry with diff support
- Regression test suite runner (YAML/JSON/CSV)
- LLM provider integrations (OpenAI, LangChain, LlamaIndex, Anthropic, LiteLLM)
- CLI tool (`evalbrain` command)
- REST API server (FastAPI)
- Web dashboard (HTML + Vanilla JS)
- Alerting and webhooks (Slack, email, HTTP)
- Configuration system (code → env → YAML → defaults)
- Full test suite (>85% coverage)
- PyPI packaging with optional extras

---

## [0.1.0] - Unreleased

Initial release.
