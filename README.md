# EvalBrain 🧠

**The pluggable LLM/RAG Evaluation and Observability Platform.**

EvalBrain is a framework-agnostic, batteries-included evaluation layer for LLM and RAG pipelines. It helps you track prompt versions, retrieval quality, hallucinations, latency, cost, and run regression tests with zero lock-in.

## Key Features

- 🎯 **Pluggable**: Drop it into any Python system with a context manager or decorator.
- 📊 **Observability**: Track latency, token counts, and cost across providers.
- ✅ **Evaluation**: Out-of-the-box metrics for hallucinations, retrieval precision/recall, and answer correctness.
- 📝 **Prompt Registry**: Version control for your prompts with SHA-256 hashing.
- 🧪 **Regression Testing**: Run golden test suites and catch quality drops before they hit production.
- 🖥️ **Dashboard**: A beautiful, real-time observability dashboard (FastAPI + Vanilla JS).

## Quick Start

```bash
pip install evalbrain
```

```python
from evalbrain import EvalBrain

# Initialize EvalBrain
brain = EvalBrain(project="my-awesome-rag")

# Trace your LLM calls
with brain.trace("generation") as span:
    # Your RAG logic here
    result = "The capital of France is Paris."
    context = ["Paris is the capital and most populous city of France."]
    
    # Evaluate on the fly
    span.evaluate(
        output=result, 
        context=context, 
        evaluators=["hallucination", "answer_quality"]
    )
```

## Installation

```bash
# Core only
pip install evalbrain

# With everything (Dashboard, NLI models, Integrations)
pip install evalbrain[all]
```

## License

MIT License. See [LICENSE](LICENSE) for details.
