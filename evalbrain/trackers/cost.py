from typing import Dict, Optional

# Default prices per 1k tokens (input, output) in USD
DEFAULT_PRICES = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "cohere-command-r-plus": {"input": 0.003, "output": 0.015},
    "cohere-command-r": {"input": 0.0005, "output": 0.0015},
}


class CostConfig:
    """Configuration for token prices. Allows overriding default prices."""
    
    def __init__(self, custom_prices: Optional[Dict[str, Dict[str, float]]] = None):
        self.prices = DEFAULT_PRICES.copy()
        if custom_prices:
            for model, rates in custom_prices.items():
                if model in self.prices:
                    self.prices[model].update(rates)
                else:
                    self.prices[model] = rates


class CostTracker:
    """Tracks token usage and calculates cost for LLM calls."""
    
    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()

    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost in USD given a model name and token counts.
        """
        if not model_name:
            return 0.0
            
        rates = self.config.prices.get(model_name)
        if not rates:
            # Fallback to zero if unknown model
            return 0.0
            
        input_cost = (input_tokens / 1000) * rates.get("input", 0.0)
        output_cost = (output_tokens / 1000) * rates.get("output", 0.0)
        
        return input_cost + output_cost
