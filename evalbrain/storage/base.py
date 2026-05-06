from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from evalbrain.models import Trace, Span, EvalResult, PromptVersion

class BaseStorage(ABC):
    """Abstract base class for all storage backends."""
    
    @abstractmethod
    def save_trace(self, trace: Trace) -> None:
        """Persist a complete trace with all its spans and eval results."""
        pass

    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a single trace by ID."""
        pass

    @abstractmethod
    def list_traces(self, project: Optional[str] = None, limit: int = 50) -> List[Trace]:
        """List recent traces, optionally filtered by project."""
        pass

    @abstractmethod
    def save_prompt_version(self, prompt_version: PromptVersion) -> None:
        """Persist a prompt version."""
        pass

    @abstractmethod
    def get_prompt_version(self, prompt_id: str, version: str) -> Optional[PromptVersion]:
        """Retrieve a specific prompt version."""
        pass
