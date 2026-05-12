from typing import Any, Dict

def get_litellm_callback(brain: Any) -> Any:
    """
    Returns a callback function for LiteLLM that traces requests via EvalBrain.
    
    Usage:
        import litellm
        from evalbrain.integrations.litellm import get_litellm_callback
        
        litellm.success_callback = [get_litellm_callback(brain)]
    """
    def evalbrain_litellm_callback(
        kwargs: Dict[str, Any],
        completion_response: Any,
        start_time: Any,
        end_time: Any
    ) -> None:
        with brain.trace("litellm_completion") as span:
            inputs = {}
            if "messages" in kwargs:
                inputs["messages"] = kwargs["messages"]
            elif "prompt" in kwargs:
                inputs["prompt"] = kwargs["prompt"]
            span.set_input(inputs)
            
            if hasattr(completion_response, "model_dump"):
                span.set_output(completion_response.model_dump())
            elif isinstance(completion_response, dict):
                span.set_output(completion_response)
            else:
                span.set_output(str(completion_response))
                
            if hasattr(completion_response, "usage") and completion_response.usage:
                span.set_tokens(
                    input_tokens=getattr(completion_response.usage, "prompt_tokens", 0),
                    output_tokens=getattr(completion_response.usage, "completion_tokens", 0),
                    model_name=getattr(completion_response, "model", kwargs.get("model", "unknown"))
                )
            elif isinstance(completion_response, dict) and "usage" in completion_response:
                usage = completion_response["usage"]
                span.set_tokens(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    model_name=completion_response.get("model", kwargs.get("model", "unknown"))
                )

    return evalbrain_litellm_callback
