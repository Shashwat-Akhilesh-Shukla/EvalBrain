# EvalBrain 🧠

**Stop guessing if your LLM works. Start measuring it.**

EvalBrain is a **framework-agnostic evaluation and observability layer** for LLM and RAG systems.
It plugs directly into your pipeline and tells you exactly what’s broken, where, and why.

No lock-in. No magic dashboards hiding reality. Just hard metrics.

---

## Why EvalBrain Exists

Most “AI systems” today are fragile.

* Prompts change silently
* Retrieval quality is unknown
* Hallucinations go unnoticed
* Latency and cost spiral out of control
* Nobody knows when things regress

EvalBrain fixes that.

It gives you **trace-level visibility + automated evaluation** so your system doesn’t just *work in demos*.

---

## Core Capabilities

### 1. Tracing That Actually Matters

Wrap anything. See everything.

* Hierarchical spans for full pipeline visibility
* Capture inputs, outputs, metadata automatically
* Zero friction integration

```python
with brain.trace("retrieval"):
    docs = retriever(query)
```

Or just:

```python
@brain.eval
def generate_answer(query):
    ...
```

You get **full execution traces without rewriting your system**.

---

### 2. Latency & Cost — No Surprises

Every span tracks:

* Execution time (ms precision)
* Token usage
* Model-level cost

Supports all major providers via configurable pricing:

* OpenAI
* Anthropic
* Gemini
* Custom providers

```python
span.set_tokens(
    input_tokens=150,
    output_tokens=50,
    model_name="gpt-4o-mini"
)
```

If you're not tracking cost, you're flying blind.

---

### 3. Real Evaluations, Not Vibes

EvalBrain doesn’t “feel” correct. It **measures** correctness.

#### Hallucination Detection

* LLM-as-judge
* Fast NLI-based fallback
* Flags ungrounded claims instantly

#### Retrieval Quality

* Context Precision
* Context Recall
* MRR (Mean Reciprocal Rank)
* Hit Rate@K

If your RAG sucks, this will expose it.

#### Answer Quality

Grades outputs on:

* Correctness
* Completeness
* Conciseness
* Toxicity

Optional classical metrics:

* `ROUGE`
* `BERTScore`

---

### 4. Prompt Versioning (Finally Done Right)

Stop losing track of what prompt produced what output.

* SHA-256 auto-hashing
* Version control built-in
* Automatic trace linking
* String-level diffs between versions
* In-memory fallback (no DB required)

```python
brain.prompt.register(
    prompt_id="sys_prompt",
    template="You are a helpful assistant. Answer concisely.",
    version="v1"
)
```

Then:

```python
with brain.trace("chat"):
    prompt = brain.prompt.get("sys_prompt", version="v1")
```

Every trace now knows exactly which prompt created it.

No ambiguity. No guessing.

---

## Quick Start

### Install

```bash
pip install evalbrain
```

---

### Minimal Example

```python
from evalbrain import EvalBrain

brain = EvalBrain(project="my-rag-system")

with brain.trace("generation") as span:

    span.set_tokens(
        input_tokens=150,
        output_tokens=50,
        model_name="gpt-4o-mini"
    )

    result = "The capital of France is Paris."
    context = ["Paris is the capital and most populous city of France."]
    reference = "Paris"

    span.evaluate(
        output=result,
        context=context,
        reference=reference,
        evaluators=[
            "hallucination",
            "answer_quality"
        ]
    )
```

---

## Installation Options

```bash
# Core
pip install evalbrain

# Full stack (dashboard, models, integrations)
pip install evalbrain[all]

# Optional metrics
pip install rouge-score evaluate bert_score
```

---

## What You Actually Get

* End-to-end trace visibility
* Grounded evaluation metrics
* Prompt version control
* Cost + latency monitoring
* Zero framework lock-in

This is not another wrapper.

This is the layer your LLM system is missing.

---

## License

MIT License. See `LICENSE` for details.
