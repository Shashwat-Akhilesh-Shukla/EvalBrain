"""
Basic usage example for EvalBrain tracing.
"""
import time
from evalbrain import EvalBrain

# Initialize EvalBrain
brain = EvalBrain(project="example-app")

@brain.eval(tags={"version": "1.0"})
def process_data(query: str):
    print(f"Processing query: {query}")
    
    # Nested span using context manager
    with brain.trace("retrieval") as span:
        span.set_input(query)
        time.sleep(0.5)  # Simulate search
        docs = ["Doc 1: AI is cool", "Doc 2: RAG is powerful"]
        span.set_output(docs)
        span.set_metadata("retrieved_count", len(docs))
    
    # Another nested span
    with brain.trace("generation") as span:
        span.set_input({"query": query, "docs": docs})
        time.sleep(0.8)  # Simulate LLM call
        answer = "AI and RAG are great technologies for building intelligent apps."
        span.set_output(answer)
        
        # Manual evaluation placeholder
        span.evaluate(
            output=answer,
            context=docs,
            reference="AI and RAG are useful."
        )

    return answer

if __name__ == "__main__":
    print("--- Starting EvalBrain Example ---")
    result = process_data("Tell me about AI and RAG")
    print(f"\nFinal Result: {result}")
    
    # Inspect captured traces
    traces = brain.get_traces()
    print(f"\nCaptured {len(traces)} trace(s).")
    for t in traces:
        print(f"Trace ID: {t.trace_id}")
        for s in t.spans:
            print(f"  Span: {s.name:<12} | Latency: {s.latency_ms:>7.2f}ms")
