"""
EvalBrain integrations with popular LLM providers and frameworks.
"""

from .openai import instrument_openai
from .anthropic import instrument_anthropic
from .langchain import EvalBrainCallbackHandler as LangChainCallbackHandler
from .llamaindex import EvalBrainCallbackHandler as LlamaIndexCallbackHandler
from .litellm import get_litellm_callback

__all__ = [
    "instrument_openai",
    "instrument_anthropic",
    "LangChainCallbackHandler",
    "LlamaIndexCallbackHandler",
    "get_litellm_callback",
]
