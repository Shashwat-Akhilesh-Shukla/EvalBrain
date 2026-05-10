import pytest
from evalbrain.core.tracer import EvalBrain
from evalbrain.models import PromptVersion

def test_prompt_registry_initialization():
    brain = EvalBrain()
    assert brain.prompt is not None

def test_prompt_hashing():
    brain = EvalBrain()
    template = "You are a helpful assistant. Answer the user query: {query}"
    # Calculate hash manually to compare
    import hashlib
    expected_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    
    version = brain.prompt.register(prompt_id="test_prompt", template=template, version="v1")
    assert version.prompt_hash == expected_hash

def test_prompt_registration_and_retrieval():
    brain = EvalBrain()
    brain.prompt.register(prompt_id="greet", template="Hello, {name}!", version="v1")
    brain.prompt.register(prompt_id="greet", template="Hi, {name}!", version="v2")
    
    # Retrieve
    v1 = brain.prompt.get("greet", "v1")
    v2 = brain.prompt.get("greet", "v2")
    
    assert v1 is not None
    assert v2 is not None
    assert v1.template == "Hello, {name}!"
    assert v2.template == "Hi, {name}!"
    
    # List prompts
    prompts = brain.prompt.list_prompts("greet")
    assert len(prompts) == 2

def test_prompt_diff():
    brain = EvalBrain()
    brain.prompt.register(prompt_id="agent", template="You are an AI.\nAct smart.", version="v1")
    brain.prompt.register(prompt_id="agent", template="You are an AI assistant.\nAct smart.", version="v2")
    
    diff_text = brain.prompt.diff("agent", "v1", "v2")
    assert "-You are an AI.\n" in diff_text
    assert "+You are an AI assistant.\n" in diff_text

def test_prompt_auto_tracking():
    brain = EvalBrain()
    brain.prompt.register(prompt_id="search", template="Find: {query}", version="latest")
    
    with brain.trace("retrieval_step") as span:
        # Fetching the prompt should automatically log it to the current span's metadata
        fetched_prompt = brain.prompt.get("search", "latest")
        assert fetched_prompt is not None
        
        # Check if span metadata got updated
        assert "prompt" in span.span.metadata
        assert span.span.metadata["prompt"]["prompt_id"] == "search"
        assert span.span.metadata["prompt"]["version"] == "latest"
