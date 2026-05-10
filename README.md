# EvalBrain 🧠

**The pluggable LLM/RAG Evaluation and Observability Platform.**

EvalBrain is a framework-agnostic, batteries-included evaluation layer for LLM and RAG pipelines. It helps you track prompt versions, retrieval quality, hallucinations, latency, cost, and run regression tests with zero lock-in.

---

## 🚀 Features Implemented So Far

### 1. Tracer & Context Management
Wrap your LLM or RAG functions using context managers or decorators. Automatically capture trace hierarchies, inputs, outputs, and metadata.
- **`brain.trace("span_name")`**: A context manager to track a specific execution block.
- **`@brain.eval`**: A decorator to seamlessly trace function calls.

### 2. Latency & Cost Tracking
Every span automatically tracks wall-clock execution time (ms precision). By providing token usage data to the span, EvalBrain automatically calculates provider-specific costs.
- Track **Latency** out of the box.
- Configure custom **Pricing Tables** via `CostTracker` (supports OpenAI, Anthropic, Gemini, etc.).

### 3. Comprehensive Evaluators
Plug in any of our out-of-the-box evaluators to instantly grade the quality of your LLM application:
- **Hallucination Evaluator**: Detects ungrounded claims using either an LLM-as-judge or a fast NLI heuristic approach.
- **Retrieval Evaluator**: Measures context precision, context recall, Mean Reciprocal Rank (MRR), and Hit Rate@K to tune your vector DB pipelines.
- **Answer Quality Evaluator**: Grades the final output on Correctness, Completeness, Conciseness, and Toxicity. Supports optional string-matching fallbacks via `rouge-score` and `bert_score`.

### 4. Prompt Version Registry
Version control your prompt templates natively inside your application.
- **Auto-Hashing**: Tracks changes safely via SHA-256.
- **Auto-Tracking**: Fetch a prompt inside a trace span (`brain.prompt.get("my_prompt")`), and EvalBrain will automatically log the exact version used in the trace.
- **String Diffing**: Generate immediate textual diffs between prompt versions.
- **Fallback Caching**: Uses a robust in-memory dictionary cache when a persistent SQL database isn't configured yet.

---

## 🛠️ Quick Start

```bash
pip install evalbrain
```

### Basic Tracing & Evaluation

```python
from evalbrain import EvalBrain

# Initialize EvalBrain
brain = EvalBrain(project="my-awesome-rag")

# Trace your LLM calls
with brain.trace("generation") as span:
    # 1. Track Tokens and Costs
    span.set_tokens(input_tokens=150, output_tokens=50, model_name="gpt-4o-mini")
    
    # 2. Your RAG logic here
    result = "The capital of France is Paris."
    context = ["Paris is the capital and most populous city of France."]
    reference = "Paris"
    
    # 3. Evaluate on the fly
    span.evaluate(
        output=result, 
        context=context,
        reference=reference,
        evaluators=["hallucination", "answer_quality"]
    )
```

### Using the Prompt Registry

```python
# Register a prompt
brain.prompt.register(
    prompt_id="sys_prompt", 
    template="You are a helpful assistant. Answer concisely.", 
    version="v1"
)

# Fetch it inside a trace to magically link the span to the prompt version!
with brain.trace("chat_call"):
    prompt = brain.prompt.get("sys_prompt", version="v1")
    # ... execute LLM ...
```

---

## 📦 Installation

```bash
# Core only
pip install evalbrain

# With everything (Dashboard, NLI models, Evaluator Integrations)
pip install evalbrain[all]

# Optional Evaluator dependencies
pip install rouge-score evaluate bert_score
```

## License

MIT License. See [LICENSE](LICENSE) for details.
