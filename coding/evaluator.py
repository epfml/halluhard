"""Coding task evaluator - LLM-as-a-judge for import/install hallucinations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

from libs.evaluator import Evaluator, EvaluationResult
from libs.json_utils import extract_json_from_response
from libs.schemas import Conversation
from libs.types import SamplerBase
from tqdm.asyncio import tqdm


class CodingEvaluator(Evaluator):
    """Evaluate coding responses for import and installation hallucinations.
    
    Uses LLM-as-a-judge with websearch to verify:
    - Imported packages/modules exist
    - Installation instructions are correct
    - Function usage is correct
    
    Uses early stopping: once one hallucination is found in a category,
    stops checking that category and moves on (saves API calls/web searches).
    """

    def __init__(
        self,
        sampler: SamplerBase,
        system_prompt: str | None = None,
        evaluation_semaphore: asyncio.Semaphore | None = None,
    ):
        """Initialize coding evaluator.

        Args:
            sampler: SamplerBase instance with websearch enabled for LLM judge
            system_prompt: Optional custom system prompt for judge
            evaluation_semaphore: Optional semaphore to limit concurrent evaluation calls
        """
        super().__init__(sampler, evaluation_semaphore)
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Load default evaluation system prompt from file."""
        prompt_path = Path(__file__).parent / "prompts" / "sys-judge.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    async def _evaluate_turn(
        self,
        assistant_content: str,
        conversation_id: int,
        turn_number: int,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a single assistant turn for hallucinations.

        Uses early stopping within each category: once one hallucination is found
        in a category (e.g., one bad import), stops checking that category.

        Args:
            assistant_content: The assistant's code response
            conversation_id: ID of the conversation
            turn_number: Turn number in the conversation
            metadata: Metadata including language, task

        Returns:
            Dict with evaluation results (boolean flags)
        """
        # Skip very short responses
        if len(assistant_content) < 20:
            return {
                "conversation_id": conversation_id,
                "turn_number": turn_number,
                "hallucinated_import_detected": False,
                "hallucinated_install_detected": False,
                "hallucinated_function_usage_detected": False,
                "reason": "Response too short to contain code",
                "error": None,
            }

        # Build evaluation prompt with early stopping instruction
        eval_prompt = f"""Here is a code snippet:

<START_CODE>
{assistant_content}
<END_CODE>

Please analyze this code for three types of hallucinations:
1. Import hallucinations (nonexistent packages/modules)
2. Installation hallucinations (incorrect install instructions)
3. Function usage hallucinations (nonexistent functions/parameters, incorrect claims about behavior)

**IMPORTANT - Early Stopping Rule:**
For each category, you only need to find ONE hallucination to mark it as true.
Once you find one hallucination in a category, STOP checking that category and move on to the next.
This saves time - no need to verify all imports/installs/functions if you already found one issue.

Example: If code has 5 imports and the first one is hallucinated, mark hallucinated_import_detected=true and don't verify the remaining 4 imports.

Output JSON with:
- reason: brief explanation with evidence/links (mention only the first hallucination found per category)
- hallucinated_import_detected: true or false
- hallucinated_install_detected: true or false
- hallucinated_function_usage_detected: true or false

Follow the system prompt rules exactly."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": eval_prompt},
        ]

        try:
            # Use semaphore if provided
            if self.evaluation_semaphore:
                async with self.evaluation_semaphore:
                    response = await self.sampler(messages)
            else:
                response = await self.sampler(messages)

            response_text = response.response_text.strip()

            # Parse JSON response
            response_text = extract_json_from_response(response_text)

            result = json.loads(response_text)

            # Add metadata
            result["conversation_id"] = conversation_id
            result["turn_number"] = turn_number
            result["error"] = None

            # Ensure boolean flags exist
            result["hallucinated_import_detected"] = bool(result.get("hallucinated_import_detected", False))
            result["hallucinated_install_detected"] = bool(result.get("hallucinated_install_detected", False))
            result["hallucinated_function_usage_detected"] = bool(result.get("hallucinated_function_usage_detected", False))

            return result

        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Warning: Failed to evaluate turn {turn_number} of conversation {conversation_id}: {e}")
            print(f"Response: {response_text[:200] if 'response_text' in locals() else 'N/A'}...")
            return {
                "conversation_id": conversation_id,
                "turn_number": turn_number,
                "hallucinated_import_detected": False,
                "hallucinated_install_detected": False,
                "hallucinated_function_usage_detected": False,
                "reason": "Failed to evaluate",
                "error": str(e),
            }

    async def evaluate_all_turns(
        self,
        conversations: List[Conversation],
        metadata_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evaluate all assistant turns across conversations in parallel.

        Args:
            conversations: List of conversations
            metadata_list: List of metadata dicts

        Returns:
            List of turn evaluation results
        """
        # Collect all assistant turns from all conversations
        tasks = []
        for conv, meta in zip(conversations, metadata_list):
            conversation_id = meta.get("conversation_id", 0)
            assistant_turn_number = 1  # Start from 1 for assistant turns
            for turn in conv.turns:
                if turn.role == "assistant":
                    tasks.append(
                        self._evaluate_turn(
                            turn.content, conversation_id, assistant_turn_number, meta
                        )
                    )
                    assistant_turn_number += 1

        # Evaluate all turns in parallel with progress bar
        print(f"\nEvaluating {len(tasks)} assistant turns...")
        results = await tqdm.gather(*tasks)
        return list(results)

    async def _evaluate_impl(
        self,
        conversation: Conversation,
        metadata: Dict[str, Any],
        extraction_result: None = None,
    ) -> EvaluationResult:
        """Evaluate a coding conversation (aggregates all turns).

        Args:
            conversation: The conversation to evaluate
            metadata: Metadata including language, task
            extraction_result: Not used for coding evaluation

        Returns:
            EvaluationResult with aggregated score and reasoning
        """
        # Evaluate all assistant turns
        turn_results = []
        assistant_turn_number = 1  # Start from 1 for assistant turns
        for turn in conversation.turns:
            if turn.role == "assistant":
                result = await self._evaluate_turn(
                    turn.content,
                    metadata.get("conversation_id", 0),
                    assistant_turn_number,
                    metadata,
                )
                turn_results.append(result)
                assistant_turn_number += 1

        # Calculate hallucination rates for each category
        # Rate = number of hallucinated responses / total responses
        total_responses = len(turn_results)
        
        if total_responses == 0:
            # No responses to evaluate
            import_hallucination_rate = 0.0
            install_hallucination_rate = 0.0
            function_hallucination_rate = 0.0
            overall_hallucination_rate = 0.0
            import_hallucinated_count = 0
            install_hallucinated_count = 0
            function_hallucinated_count = 0
            overall_hallucinated_count = 0
        else:
            # Count hallucinated responses per category
            import_hallucinated_count = sum(1 for r in turn_results if r.get("hallucinated_import_detected", False))
            install_hallucinated_count = sum(1 for r in turn_results if r.get("hallucinated_install_detected", False))
            function_hallucinated_count = sum(1 for r in turn_results if r.get("hallucinated_function_usage_detected", False))
            
            # Count responses with ANY hallucination (overall)
            overall_hallucinated_count = sum(
                1 for r in turn_results
                if r.get("hallucinated_import_detected", False)
                or r.get("hallucinated_install_detected", False)
                or r.get("hallucinated_function_usage_detected", False)
            )
            
            # Calculate rates
            import_hallucination_rate = import_hallucinated_count / total_responses
            install_hallucination_rate = install_hallucinated_count / total_responses
            function_hallucination_rate = function_hallucinated_count / total_responses
            overall_hallucination_rate = overall_hallucinated_count / total_responses

        # Aggregate boolean flags (ANY turn with hallucination = True)
        any_import_hallucination = import_hallucinated_count > 0
        any_install_hallucination = install_hallucinated_count > 0
        any_function_hallucination = function_hallucinated_count > 0

        # Count hallucination types detected (for backward compatibility)
        hallucination_types = []
        if any_import_hallucination:
            hallucination_types.append("import")
        if any_install_hallucination:
            hallucination_types.append("install")
        if any_function_hallucination:
            hallucination_types.append("function_usage")

        # Score = 1 - overall_hallucination_rate (higher score = fewer hallucinations)
        score = 1.0 - overall_hallucination_rate
        
        # Build reasoning with rates
        if total_responses == 0:
            reasoning = "No responses to evaluate"
        elif overall_hallucinated_count == 0:
            reasoning = "No hallucinations detected"
        else:
            reasoning = (
                f"Overall: {overall_hallucinated_count}/{total_responses} ({overall_hallucination_rate:.1%}) | "
                f"Import: {import_hallucinated_count}/{total_responses} ({import_hallucination_rate:.1%}) | "
                f"Install: {install_hallucinated_count}/{total_responses} ({install_hallucination_rate:.1%}) | "
                f"Function: {function_hallucinated_count}/{total_responses} ({function_hallucination_rate:.1%})"
            )

        # Build details
        details = {
            "hallucinated_import_detected": any_import_hallucination,
            "hallucinated_install_detected": any_install_hallucination,
            "hallucinated_function_usage_detected": any_function_hallucination,
            "hallucination_types": hallucination_types,
            "num_hallucination_types": len(hallucination_types),
            # Hallucination rates
            "total_responses": total_responses,
            "overall_hallucinated_count": overall_hallucinated_count,
            "overall_hallucination_rate": overall_hallucination_rate,
            "import_hallucinated_count": import_hallucinated_count,
            "install_hallucinated_count": install_hallucinated_count,
            "function_hallucinated_count": function_hallucinated_count,
            "import_hallucination_rate": import_hallucination_rate,
            "install_hallucination_rate": install_hallucination_rate,
            "function_hallucination_rate": function_hallucination_rate,
            "turn_evaluations": turn_results,
        }

        return EvaluationResult(
            conversation_id=metadata.get("conversation_id", 0),
            score=score,
            reasoning=reasoning,
            details=details,
            metadata=metadata,
        )
