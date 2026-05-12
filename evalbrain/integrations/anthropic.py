import inspect
import functools
from typing import Any

def instrument_anthropic(client: Any, brain: Any) -> Any:
    """
    Instrument an Anthropic client with EvalBrain tracing.
    This modifies the client.messages.create method to automatically
    wrap calls in a trace and capture usage statistics.
    """
    if getattr(client, "_evalbrain_instrumented", False) is True:
        return client

    if hasattr(client, "messages"):
        original_create = client.messages.create
        
        if inspect.iscoroutinefunction(original_create):
            @functools.wraps(original_create)
            async def async_wrapper(*args, **kwargs):
                with brain.trace("anthropic_message") as span:
                    span.set_input({"args": args, "kwargs": kwargs})
                    
                    response = await original_create(*args, **kwargs)
                    span.set_output(response.model_dump() if hasattr(response, "model_dump") else response)
                    
                    if hasattr(response, "usage") and response.usage:
                        span.set_tokens(
                            input_tokens=getattr(response.usage, "input_tokens", 0),
                            output_tokens=getattr(response.usage, "output_tokens", 0),
                            model_name=getattr(response, "model", kwargs.get("model", "unknown"))
                        )
                    return response
            client.messages.create = async_wrapper
        else:
            @functools.wraps(original_create)
            def sync_wrapper(*args, **kwargs):
                with brain.trace("anthropic_message") as span:
                    span.set_input({"args": args, "kwargs": kwargs})
                    
                    response = original_create(*args, **kwargs)
                    span.set_output(response.model_dump() if hasattr(response, "model_dump") else response)
                    
                    if hasattr(response, "usage") and response.usage:
                        span.set_tokens(
                            input_tokens=getattr(response.usage, "input_tokens", 0),
                            output_tokens=getattr(response.usage, "output_tokens", 0),
                            model_name=getattr(response, "model", kwargs.get("model", "unknown"))
                        )
                    return response
            client.messages.create = sync_wrapper
            
    client._evalbrain_instrumented = True
    return client
