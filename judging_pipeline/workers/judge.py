"""Worker for LLM-based hallucination judgment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from libs.json_utils import extract_json_from_response
from libs.types import SamplerBase

from ..core.queue import MonitoredQueue, QueueItem
from ..core.worker import Worker
from ..core.domain_strategy import DomainStrategy
from ..models.work_items import FilteredContent, JudgmentResult
from ..logging_config import get_logger

if TYPE_CHECKING:
    from .early_stopping import CodingEarlyStoppingState
    from .package_cache import PackageVerdictCache

logger = get_logger()


class JudgeWorker(Worker[FilteredContent, JudgmentResult]):
    """Judge claims for hallucination using LLM.
    
    Input: FilteredContent (with relevant passages)
    Output: JudgmentResult (with hallucination verdict)
    
    This worker:
    1. Builds judgment prompt with filtered content
    2. Calls LLM for hallucination assessment
    3. Parses and outputs judgment result
    """
    
    def __init__(
        self,
        input_queue: MonitoredQueue[FilteredContent],
        output_queue: MonitoredQueue[JudgmentResult],
        sampler: SamplerBase,
        strategy: DomainStrategy,
        sampler_fallback: SamplerBase | None = None,
        system_prompt: str | None = None,
        num_workers: int = 20,
        rate_limit_delay: float = 0.0,
        early_stopping_state: "CodingEarlyStoppingState | None" = None,
        package_cache: "PackageVerdictCache | None" = None,
    ):
        """Initialize judge.
        
        Args:
            input_queue: Queue of filtered content
            output_queue: Queue for judgments
            sampler: LLM sampler for judgment
            strategy: Domain strategy for prompt generation
            sampler_fallback: Fallback sampler with websearch (for failed searches)
            system_prompt: Custom judgment prompt
            num_workers: Number of concurrent workers
            rate_limit_delay: Delay between judgments
            early_stopping_state: Optional early stopping state for coding task.
                If provided, claims in already-detected categories will be skipped.
            package_cache: Optional package verdict cache for coding task.
                If provided, will cache verification results to skip future searches.
        """
        super().__init__(
            name="Judge",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
            rate_limit_delay=rate_limit_delay,
        )
        
        self.sampler = sampler
        self.strategy = strategy
        self.sampler_fallback = sampler_fallback or sampler
        self.system_prompt = system_prompt or self._load_default_prompt()
        self.early_stopping_state = early_stopping_state
        self.package_cache = package_cache
        self._is_coding_task = strategy.task_name == "coding"
    
    def _load_default_prompt(self) -> str:
        """Load default judgment system prompt."""
        prompt_path = self.strategy.evaluator_prompt_path
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    
    async def process(
        self,
        item: FilteredContent,
        item_wrapper: QueueItem[FilteredContent],
    ) -> JudgmentResult:
        """Judge a claim for hallucination."""
        # Skip judging for whitelisted packages (coding task optimization)
        # If a package is in the whitelist, we KNOW it exists - no LLM call needed
        if self._is_coding_task and item.whitelist_skip:
            element_type = item.claim.data.get("element_type", "unknown")
            package_name = item.claim.data.get("package_name", "unknown")
            
            # Check if this is a dynamic cache hit (vs static whitelist)
            if item.dynamic_cache_hit:
                logger.info(
                    f"🔄 DYNAMIC CACHE [Judge]: {element_type} '{package_name}' (conv {item.conversation_id}) - using cached verdict"
                )
                return self._build_dynamic_cache_result(item, element_type, package_name, item.cached_verdict_exists)
            else:
                logger.info(
                    f"⚡ WHITELIST SKIP: {element_type} '{package_name}' (conv {item.conversation_id}) - no LLM needed"
                )
                return self._build_whitelist_result(item, element_type, package_name)
        
        # Early stopping check for coding task (per-turn: only skip within same turn)
        if self._is_coding_task and self.early_stopping_state:
            element_type = item.claim.data.get("element_type", "unknown")
            should_skip = await self.early_stopping_state.should_skip(
                item.conversation_id, item.claim.turn_number, element_type
            )
            if should_skip:
                logger.info(
                    f"⏭️  EARLY STOP: {element_type} claim skipped (conv {item.conversation_id}, turn {item.claim.turn_number}) - hallucination already found in this turn"
                )
                return self._build_skipped_result(item, element_type)
        
        # Use strategy to build textual claim
        claim_text = self.strategy.build_textual_claim_for_judging(item.claim)
        
        # Determine which prompt/sampler to use:
        # 1. If we have filtered content -> use normal judgment with filtered content
        # 2. If no filtered content but have snippets -> use snippets-only judgment  
        # 3. If no content at all -> use LLM websearch fallback
        has_snippets = item.search_results_text and item.search_results_text != "No search results found."
        filtered_content_empty = not (item.filtered_content or "").strip()
        snippets_only = filtered_content_empty and bool(has_snippets)
        
        if not item.use_fallback:
            # Normal path: use filtered content
            prompt = self.strategy.build_judgment_prompt(
                search_results=item.search_results_text,
                filtered_content=item.filtered_content,
                claim_text=claim_text,
            )
            use_fallback = False
        elif has_snippets:
            # Fallback with snippets: web fetch failed but we have search snippets
            prompt = self.strategy.build_snippets_only_judgment_prompt(
                search_results=item.search_results_text,
                claim_text=claim_text,
            )
            use_fallback = False  # Use normal sampler, snippets are enough
        else:
            # Full fallback: no content at all, use LLM websearch
            prompt = self.strategy.build_fallback_judgment_prompt(claim_text)
            use_fallback = True
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        try:
            # Use appropriate sampler
            sampler = self.sampler_fallback if use_fallback else self.sampler
            response = await sampler(messages)
            response_text = response.response_text.strip()
            
            # Parse JSON response
            json_text = extract_json_from_response(response_text)
            result = json.loads(json_text)
            
            # Extract hallucination flags
            import_halluc = bool(result.get("hallucinated_import_detected", False))
            install_halluc = bool(result.get("hallucinated_install_detected", False))
            function_halluc = bool(result.get("hallucinated_function_usage_detected", False))
            
            # Record hallucinations for early stopping (coding task only, per-turn)
            if self._is_coding_task and self.early_stopping_state:
                if import_halluc or install_halluc or function_halluc:
                    await self.early_stopping_state.record_hallucination(
                        conversation_id=item.conversation_id,
                        turn_number=item.claim.turn_number,
                        import_halluc=import_halluc,
                        install_halluc=install_halluc,
                        function_halluc=function_halluc,
                    )
            
            # Cache package verdict for future claims (coding task only)
            # This allows skipping web search for the same package in future claims
            if self._is_coding_task and self.package_cache:
                element_type = item.claim.data.get("element_type", "")
                package_name = item.claim.data.get("package_name") or item.claim.data.get("module_name", "")
                
                if element_type in ("import", "install") and package_name:
                    # Determine if package exists based on hallucination flags
                    if element_type == "import":
                        package_exists = not import_halluc
                    else:  # install
                        package_exists = not install_halluc
                    
                    # Build reason based on judgment
                    if package_exists:
                        reason = f"Verified via web search - package exists"
                    else:
                        reason = f"Verified via web search - package does NOT exist (hallucinated)"
                    
                    await self.package_cache.set(package_name, package_exists, reason)
                    logger.debug(
                        f"📦 CACHE SET: {element_type} '{package_name}' exists={package_exists}"
                    )
            
            return JudgmentResult(
                claim_id=item.claim_id,
                conversation_id=item.conversation_id,
                turn_number=item.claim.turn_number,
                claim=item.claim,
                reference_name=result.get("reference_name", "Unknown"),
                reference_grounding=result.get("reference_grounding", "Unknown"),
                content_grounding=result.get("content_grounding", "Unknown"),
                hallucination=result.get("hallucination", "Unknown"),
                abstention=result.get("abstention", "Unknown"),
                verification_error=result.get("verification_error", "No"),
                input_use_fallback=item.use_fallback,
                judge_used_websearch_fallback=use_fallback,
                snippets_only=snippets_only,
                # Coding-specific hallucination flags
                hallucinated_import_detected=import_halluc,
                hallucinated_install_detected=install_halluc,
                hallucinated_function_usage_detected=function_halluc,
                # Reasoning/evidence from judge
                reason=result.get("reason", ""),
                # Search queries executed
                search_queries=item.queries,
                token_usage=response.token_usage if hasattr(response, "token_usage") else {},
            )
        
        except (json.JSONDecodeError, ValueError) as e:
            return self._build_error_result(item, f"JSON parse error: {e}")
        except Exception as e:
            return self._build_error_result(item, str(e))
    
    def _build_error_result(self, item: FilteredContent, error: str) -> JudgmentResult:
        """Build error result."""
        # Try to get a reference name from various possible fields
        data = item.claim.data
        reference_name = (
            data.get("authority") or 
            data.get("reference_name") or 
            data.get("claimed_title") or 
            "Unknown"
        )
        return JudgmentResult(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            turn_number=item.claim.turn_number,
            claim=item.claim,
            reference_name=reference_name,
            reference_grounding="Error - Failed to evaluate",
            content_grounding="Error - Failed to evaluate",
            hallucination="Unknown",
            abstention="Unknown",
            verification_error="Yes",
            search_queries=item.queries,
            error=error,
        )

    def _build_skipped_result(self, item: FilteredContent, element_type: str) -> JudgmentResult:
        """Build result for a skipped claim due to early stopping.
        
        The claim is not evaluated because its category already has a hallucination
        detected in this conversation.
        """
        data = item.claim.data
        reference_name = (
            data.get("package_name") or 
            data.get("reference_name") or 
            "Unknown"
        )
        return JudgmentResult(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            turn_number=item.claim.turn_number,
            claim=item.claim,
            reference_name=reference_name,
            reference_grounding="Skipped - Early stopping",
            content_grounding="Skipped - Early stopping",
            hallucination="Skipped",  # Not evaluated due to early stopping
            abstention="No",
            verification_error="No",
            search_queries=item.queries,
            reason=f"Skipped due to early stopping: {element_type} category already has hallucination detected",
            skipped_early_stopping=True,
        )
    
    def _build_whitelist_result(
        self, item: FilteredContent, element_type: str, package_name: str
    ) -> JudgmentResult:
        """Build a 'not hallucinated' result for whitelisted packages.
        
        Whitelisted packages are known to exist, so no LLM call is needed.
        """
        return JudgmentResult(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            turn_number=item.claim.turn_number,
            claim=item.claim,
            reference_name=package_name,
            reference_grounding="Verified - Whitelisted package",
            content_grounding="Package is in known packages list",
            hallucination="No",  # Known package = not a hallucination
            abstention="No",
            verification_error="No",
            search_queries=item.queries,
            reason=f"Package '{package_name}' is a well-known, verified {element_type}. No LLM verification needed.",
            skipped_whitelist=True,
        )
    
    def _build_dynamic_cache_result(
        self, item: FilteredContent, element_type: str, package_name: str, package_exists: bool
    ) -> JudgmentResult:
        """Build a result for dynamically cached package verdicts.
        
        These are packages that were verified via web search earlier in this run
        and cached to skip future verifications.
        """
        if package_exists:
            hallucination = "No"
            reference_grounding = "Verified - Cached verdict (exists)"
            content_grounding = "Package was verified via web search earlier in this run"
            reason = f"Package '{package_name}' was verified earlier in this run - exists. No repeat verification needed."
        else:
            hallucination = "Yes"
            reference_grounding = "Verified - Cached verdict (does NOT exist)"
            content_grounding = "Package was verified via web search earlier in this run"
            reason = f"Package '{package_name}' was verified earlier in this run - does NOT exist (hallucinated). No repeat verification needed."
        
        return JudgmentResult(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            turn_number=item.claim.turn_number,
            claim=item.claim,
            reference_name=package_name,
            reference_grounding=reference_grounding,
            content_grounding=content_grounding,
            hallucination=hallucination,
            abstention="No",
            verification_error="No",
            search_queries=item.queries,
            reason=reason,
            skipped_whitelist=True,  # Reuse flag for reporting
            # Set hallucination flags based on element type
            hallucinated_import_detected=(element_type == "import" and not package_exists),
            hallucinated_install_detected=(element_type == "install" and not package_exists),
        )
