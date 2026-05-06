import pytest
import time
from evalbrain import EvalBrain

def test_basic_tracing():
    brain = EvalBrain(project="test-project")
    
    with brain.trace("root_span") as span:
        span.set_input("hello")
        time.sleep(0.1)
        span.set_output("world")
        
        with brain.trace("child_span") as child:
            child.set_input({"data": 123})
            time.sleep(0.05)
            child.set_output({"result": "ok"})

    traces = brain.get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.project == "test-project"
    assert len(trace.spans) == 2
    
    root_span = next(s for s in trace.spans if s.name == "root_span")
    child_span = next(s for s in trace.spans if s.name == "child_span")
    
    assert root_span.input == "hello"
    assert root_span.output == "world"
    assert child_span.input == {"data": 123}
    assert child_span.latency_ms >= 50
    assert root_span.latency_ms >= 150

def test_decorator_tracing():
    brain = EvalBrain(project="decorator-project")
    
    @brain.eval(name="decorated_func", tags={"env": "test"})
    def my_func(x, y):
        return x + y
    
    result = my_func(10, 20)
    assert result == 30
    
    traces = brain.get_traces()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.tags == {"env": "test"}
    assert len(trace.spans) == 1
    assert trace.spans[0].name == "decorated_func"
    assert trace.spans[0].output == 30

def test_nested_decorators():
    brain = EvalBrain(project="nested-decorators")
    
    @brain.eval()
    def inner(a):
        return a * 2
        
    @brain.eval()
    def outer(b):
        return inner(b) + 1
        
    result = outer(5)
    assert result == 11
    
    traces = brain.get_traces()
    assert len(traces) == 1
    assert len(traces[0].spans) == 2
