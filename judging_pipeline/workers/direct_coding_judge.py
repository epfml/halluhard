"""Direct coding hallucination judge using OpenAI websearch.

This worker takes an entire assistant turn and uses OpenAI's websearch capability
to detect hallucinations in imports, installs, and function calls - all at once,
without extracting individual claims first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..core.worker import Worker
from ..core.queue import MonitoredQueue, QueueItem
from ..logging_config import get_logger
from libs.types import SamplerBase

logger = get_logger()


@dataclass
class TurnItem:
    """A single assistant turn to be judged for coding hallucinations."""
    
    conversation_id: int
    turn_number: int
    content: str  # The assistant's response content
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectCodingResult:
    """Result from direct coding hallucination detection."""
    
    conversation_id: int
    turn_number: int
    
    # Overall hallucination detection
    has_hallucination: bool = False
    
    # Specific hallucination types detected
    hallucinated_imports: List[Dict[str, str]] = field(default_factory=list)
    hallucinated_installs: List[Dict[str, str]] = field(default_factory=list)
    hallucinated_function_calls: List[Dict[str, str]] = field(default_factory=list)
    
    # Counts for backward compatibility
    hallucinated_import_detected: bool = False
    hallucinated_install_detected: bool = False
    hallucinated_function_usage_detected: bool = False
    
    # Reasoning/evidence
    reasoning: str = ""
    raw_response: str = ""
    
    # Metadata
    error: str | None = None
    token_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "conversation_id": self.conversation_id,
            "turn_number": self.turn_number,
            "has_hallucination": self.has_hallucination,
            "hallucinated_imports": self.hallucinated_imports,
            "hallucinated_installs": self.hallucinated_installs,
            "hallucinated_function_calls": self.hallucinated_function_calls,
            "hallucinated_import_detected": self.hallucinated_import_detected,
            "hallucinated_install_detected": self.hallucinated_install_detected,
            "hallucinated_function_usage_detected": self.hallucinated_function_usage_detected,
            "reasoning": self.reasoning,
            "error": self.error,
            "token_usage": self.token_usage,
        }


DIRECT_CODING_SYSTEM_PROMPT = """You are an expert code reviewer with web search capabilities. Your task is to analyze code responses and identify hallucinated (non-existent) packages, imports, and function calls.

## Your Task

Analyze the given code and use web search to verify:
1. **Import Hallucinations**: Packages that are imported but don't exist
2. **Install Hallucinations**: Packages in install commands (pip, npm, etc.) that don't exist
3. **Function Call Hallucinations**: Functions/methods called on packages that don't exist in those packages' APIs

## Verification Strategy

For IMPORTS and INSTALLS:
- Search for the package on PyPI (Python), npm (JavaScript), crates.io (Rust), etc.
- If the package doesn't exist on the official registry, it's hallucinated

For FUNCTION CALLS:
- Search for "[package_name] [function_name] documentation" or "[package_name] API reference"
- Verify the specific function/method exists in the package's public API
- IMPORTANT: A package may exist but the function may not - this is still a hallucination!

## What to IGNORE

- Standard library modules (os, sys, json, datetime, etc. for Python)
- Well-known packages that definitely exist (numpy, pandas, requests, react, lodash, etc.)
- Minor syntax variations that would still work
- Deprecated but still functional APIs

## Output Format

Return a JSON object with:
```json
{
  "has_hallucination": boolean,
  "hallucinated_imports": [
    {"package": "package_name", "code": "import statement", "reason": "why it's hallucinated"}
  ],
  "hallucinated_installs": [
    {"package": "package_name", "code": "install command", "reason": "why it's hallucinated"}
  ],
  "hallucinated_function_calls": [
    {"package": "package_name", "function": "function_name", "code": "the call", "reason": "why it's hallucinated"}
  ],
  "reasoning": "Overall summary of your analysis"
}
```

If there are no hallucinations, return:
```json
{
  "has_hallucination": false,
  "hallucinated_imports": [],
  "hallucinated_installs": [],
  "hallucinated_function_calls": [],
  "reasoning": "All packages and function calls verified as valid"
}
```

Be thorough but accurate. Only flag something as hallucinated if you've verified it doesn't exist."""


class DirectCodingJudgeWorker(Worker[TurnItem, DirectCodingResult]):
    """Judge coding hallucinations directly using OpenAI websearch.
    
    This worker takes entire assistant turns and uses OpenAI's websearch
    capability to detect hallucinations in imports, installs, and function calls
    without extracting individual claims first.
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[TurnItem],
        output_queue: MonitoredQueue[DirectCodingResult],
        sampler: SamplerBase,
        num_workers: int = 10,
        rate_limit_delay: float = 0.0,
    ):
        super().__init__(
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
            rate_limit_delay=rate_limit_delay,
            name="DirectCodingJudge",
        )
        self.sampler = sampler
        self.system_prompt = DIRECT_CODING_SYSTEM_PROMPT
    
    async def process(
        self,
        item: TurnItem,
        item_wrapper: QueueItem[TurnItem],
    ) -> DirectCodingResult:
        """Judge a turn for coding hallucinations."""
        
        # Build the user prompt with the code content
        user_prompt = f"""Please analyze the following code response for hallucinations.

Use web search to verify that all imported packages exist, all installed packages are real,
and all function calls are valid for their respective packages.

<CODE_RESPONSE>
{item.content}
</CODE_RESPONSE>

Search for packages on their official registries (PyPI, npm, etc.) and verify function calls
against official documentation. Report any hallucinations you find."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = await self.sampler(messages)
            raw_text = response.response_text.strip()
            
            # Parse JSON response
            result_data = self._parse_response(raw_text)
            
            # Build result
            return DirectCodingResult(
                conversation_id=item.conversation_id,
                turn_number=item.turn_number,
                has_hallucination=result_data.get("has_hallucination", False),
                hallucinated_imports=result_data.get("hallucinated_imports", []),
                hallucinated_installs=result_data.get("hallucinated_installs", []),
                hallucinated_function_calls=result_data.get("hallucinated_function_calls", []),
                hallucinated_import_detected=len(result_data.get("hallucinated_imports", [])) > 0,
                hallucinated_install_detected=len(result_data.get("hallucinated_installs", [])) > 0,
                hallucinated_function_usage_detected=len(result_data.get("hallucinated_function_calls", [])) > 0,
                reasoning=result_data.get("reasoning", ""),
                raw_response=raw_text,
                token_usage=response.token_usage if hasattr(response, "token_usage") else {},
            )
            
        except Exception as e:
            logger.error(f"Error judging turn {item.turn_number} in conv {item.conversation_id}: {e}")
            return DirectCodingResult(
                conversation_id=item.conversation_id,
                turn_number=item.turn_number,
                error=str(e),
            )
    
    def _parse_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response
        # First, try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON
        json_match = re.search(r'\{[^{}]*"has_hallucination"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to parse the entire response as JSON
        try:
            # Find the first { and last }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
        
        # Fallback: return empty result
        logger.warning(f"Failed to parse JSON response: {text[:200]}...")
        return {
            "has_hallucination": False,
            "hallucinated_imports": [],
            "hallucinated_installs": [],
            "hallucinated_function_calls": [],
            "reasoning": f"Failed to parse response: {text[:500]}"
        }

