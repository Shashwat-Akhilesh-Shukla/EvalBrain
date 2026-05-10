import hashlib
import difflib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from evalbrain.models import PromptVersion
from evalbrain.core.tracer import _current_spans

class PromptRegistry:
    """
    Manages, versions, and tracks prompt templates.
    """
    
    def __init__(self, brain: "EvalBrain"):
        self.brain = brain
        # Fallback local cache if no storage is provided
        # Format: { "prompt_id": { "version_tag": PromptVersion } }
        self._local_cache: Dict[str, Dict[str, PromptVersion]] = {}

    def _hash_template(self, template: str) -> str:
        """Generate a SHA-256 hash of the template string."""
        return hashlib.sha256(template.encode("utf-8")).hexdigest()

    def register(
        self, 
        prompt_id: str, 
        template: str, 
        version: str = "latest", 
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptVersion:
        """
        Register a new prompt version or update an existing one.
        """
        prompt_hash = self._hash_template(template)
        
        prompt_version = PromptVersion(
            prompt_id=prompt_id,
            version=version,
            template=template,
            prompt_hash=prompt_hash,
            tags=tags or {},
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc)
        )
        
        # Save to storage or local cache
        if hasattr(self.brain, "storage") and self.brain.storage and hasattr(self.brain.storage, "save_prompt_version"):
            self.brain.storage.save_prompt_version(prompt_version)
        else:
            if prompt_id not in self._local_cache:
                self._local_cache[prompt_id] = {}
            self._local_cache[prompt_id][version] = prompt_version
            
        return prompt_version

    def get(self, prompt_id: str, version: str = "latest", auto_track: bool = True) -> Optional[PromptVersion]:
        """
        Retrieve a specific prompt version.
        If auto_track is True and there's an active Span, injects the prompt info into the span metadata.
        """
        prompt_version = None
        
        # Try to get from storage
        if hasattr(self.brain, "storage") and self.brain.storage and hasattr(self.brain.storage, "get_prompt_version"):
            prompt_version = self.brain.storage.get_prompt_version(prompt_id, version)
            
        # Fallback to local cache
        if not prompt_version:
            if prompt_id in self._local_cache and version in self._local_cache[prompt_id]:
                prompt_version = self._local_cache[prompt_id][version]
                
        if not prompt_version:
            return None
            
        # Auto-tracking
        if auto_track:
            stack = _current_spans.get()
            if stack:
                active_span = stack[-1]
                active_span.metadata["prompt"] = {
                    "prompt_id": prompt_version.prompt_id,
                    "version": prompt_version.version,
                    "prompt_hash": prompt_version.prompt_hash
                }
                
        return prompt_version

    def list_prompts(self, prompt_id: str) -> List[PromptVersion]:
        """
        List all versions of a given prompt ID.
        """
        # Since base storage interface doesn't have list_prompt_versions, 
        # we will primarily rely on local cache or assume it's added later.
        # For now, return what we have in cache.
        if prompt_id in self._local_cache:
            return list(self._local_cache[prompt_id].values())
        return []

    def diff(self, prompt_id: str, version1: str, version2: str) -> str:
        """
        Return a unified diff between two prompt versions.
        """
        p1 = self.get(prompt_id, version1, auto_track=False)
        p2 = self.get(prompt_id, version2, auto_track=False)
        
        if not p1:
            raise ValueError(f"Prompt {prompt_id} version {version1} not found.")
        if not p2:
            raise ValueError(f"Prompt {prompt_id} version {version2} not found.")
            
        t1_lines = p1.template.splitlines(keepends=True)
        t2_lines = p2.template.splitlines(keepends=True)
        
        diff_lines = difflib.unified_diff(
            t1_lines, 
            t2_lines, 
            fromfile=f"{prompt_id}@{version1}", 
            tofile=f"{prompt_id}@{version2}"
        )
        
        return "".join(diff_lines)
