import pytest
from unittest.mock import MagicMock
from evalbrain.core.tracer import EvalBrain

class MockUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.input_tokens = prompt_tokens
        self.output_tokens = completion_tokens

class MockResponse:
    def __init__(self, content="Hello", prompt_tokens=10, completion_tokens=20, model="gpt-4"):
        self.content = content
        self.usage = MockUsage(prompt_tokens, completion_tokens)
        self.model = model
    
    def model_dump(self):
        return {"content": self.content, "model": self.model}

def test_openai_integration():
    from evalbrain.integrations.openai import instrument_openai
    brain = EvalBrain()
    
    mock_client = MagicMock()
    # Need to properly mock inspect.iscoroutinefunction by setting __code__ or similar, 
    # but a simple MagicMock is treated as synchronous by default
    mock_client.chat.completions.create = MagicMock(return_value=MockResponse())
    
    instrument_openai(mock_client, brain)
    mock_client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])
    
    traces = brain.get_traces()
    assert len(traces) == 1
    assert len(traces[0].spans) == 1
    span = traces[0].spans[0]
    
    assert span.name == "openai_chat_completion"
    assert span.token_counts["input"] == 10
    assert span.token_counts["output"] == 20
    assert span.model_name == "gpt-4"

def test_anthropic_integration():
    from evalbrain.integrations.anthropic import instrument_anthropic
    brain = EvalBrain()
    
    mock_client = MagicMock()
    mock_client.messages.create = MagicMock(return_value=MockResponse())
    
    instrument_anthropic(mock_client, brain)
    mock_client.messages.create(model="claude-3", messages=[{"role": "user", "content": "Hi"}])
    
    traces = brain.get_traces()
    assert len(traces) == 1
    span = traces[0].spans[0]
    
    assert span.name == "anthropic_message"
    assert span.token_counts["input"] == 10
    assert span.token_counts["output"] == 20

def test_litellm_integration():
    from evalbrain.integrations.litellm import get_litellm_callback
    brain = EvalBrain()
    
    cb = get_litellm_callback(brain)
    
    cb(
        kwargs={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        completion_response=MockResponse(),
        start_time=None,
        end_time=None
    )
    
    traces = brain.get_traces()
    assert len(traces) == 1
    span = traces[0].spans[0]
    
    assert span.name == "litellm_completion"
    assert span.token_counts["input"] == 10
    assert span.token_counts["output"] == 20
    assert span.model_name == "gpt-4"
