"""Main entry point for queue-based evaluation pipeline.

Usage:
    python -m judging_pipeline.run_pipeline --input data/conversations.jsonl --type webscraper --task research_questions

Features:
    - Clear visibility into queue depths (see where bottlenecks are)
    - Easy to scale individual workers
    - Each external API has its own rate limiting
    - Real-time progress monitoring
    
Pipeline types:
    - openai: Claims → Judge (with OpenAI websearch)
    - serper: Claims → Search → Fetch → Filter → Judge (Serper snippets only)
    - webscraper: Claims → Search → Fetch → PDF → Filter → Judge (full scraping)
    - coding_direct: Turns → Judge (OpenAI websearch, no claim extraction, coding only)
"""

from __future__ import annotations

import asyncio
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Literal, Tuple, Set
import random
import logging

from libs.evaluator import EvaluationResult
from libs.storage import load_conversations
from libs.models import get_sampler
from libs.types import SamplerBase
from libs.browser_fetcher import close_shared_client
from libs.information_extraction import close_pdf_session

from .logging_config import get_logger, configure_logging
from .core import MonitoredQueue, Pipeline, QueueMonitor
from .core.worker import Worker
from .core.domain_strategy import DomainStrategy
from .strategies import get_strategy
from .models import (
    ConversationItem,
    ClaimItem,
    SearchTask,
    ContentItem,
    FilteredContent,
    JudgmentResult,
)
from .workers import (
    ClaimExtractorWorker,
    WebSearcherWorker,
    WebFetcherWorker,
    SnippetExtractorWorker,
    PDFConverterWorker,
    ContentFilterWorker,
    ContentAggregatorWorker,
    JudgeWorker,
    CodingEarlyStoppingState,
    PackageVerdictCache,
)
from .models.work_items import PDFTask, PDFResult

# Set up logging for the entire package
configure_logging()
logger = get_logger()


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class WorkerConfig:
    """Configuration for worker counts."""
    num_extractors: int = 5
    num_searchers: int = 50
    num_fetchers: int = 20
    num_pdf_converters: int = 10
    num_filters: int = 10
    num_judges: int = 20


@dataclass 
class ModelConfig:
    """Configuration for model names."""
    extractor: str
    judge: str
    judge_fallback: str
    search: str | None = None  # Only for search-based pipelines


# =============================================================================
# Base Pipeline Class
# =============================================================================

class BasePipeline(ABC):
    """Abstract base class for evaluation pipelines.
    
    Provides common functionality for:
    - Loading and sampling conversations
    - Running the pipeline with monitoring
    - Collecting and saving results
    
    Subclasses must implement:
    - _get_model_config(): Return model configuration
    - _create_queues(): Create pipeline-specific queues
    - _create_workers(): Create pipeline-specific workers
    - _get_intermediate_queues(): Return queues to wait on before shutdown
    - _log_config(): Log pipeline-specific configuration
    """
    
    name: str = "BasePipeline"
    
    def __init__(
        self,
        input_path: Path,
        output_path: Path | None,
        worker_config: WorkerConfig,
        task_name: str,
        base_path: Path | str | None = None,
        n_conversations: int | None = None,
        max_claims_per_turn: int | None = None,
        seed: int = 42,
        monitor_interval: float = 3.0,
        claims_cache_path: Path | str | None = None,
        checkpoint_interval: int = 100,
        max_claims_per_category: int | None = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.worker_config = worker_config
        self.task_name = task_name
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.n_conversations = n_conversations
        self.max_claims_per_turn = max_claims_per_turn
        self.max_claims_per_category = max_claims_per_category
        self.seed = seed
        self.monitor_interval = monitor_interval
        self.checkpoint_interval = checkpoint_interval
        
        # Initialize strategy
        self.strategy = get_strategy(task_name, self.base_path)
        
        # Claims cache path - if None, will use default based on input path
        self._claims_cache_path = Path(claims_cache_path) if claims_cache_path else None
        
        # Will be populated during setup
        self.conversations: List[Any] = []
        self.metadata_list: List[Dict[str, Any]] = []
        self.model_config: ModelConfig | None = None
        
        # Core queues (always needed)
        self.conversation_queue: MonitoredQueue[ConversationItem] | None = None
        self.claims_queue: MonitoredQueue[ClaimItem] | None = None
        self.results_queue: MonitoredQueue[JudgmentResult] | None = None
        
        # Pipeline infrastructure
        self.pipeline: Pipeline | None = None
        self.samplers: Dict[str, SamplerBase] = {}
        
        # Cache state
        self._cached_claims: List[ClaimItem] | None = None
        
        # Result cache state (for skipping already-evaluated conversations)
        self._cached_results: List[EvaluationResult] | None = None
        self._cached_result_conv_ids: Set[int] = set()
        
        # Checkpoint state for judgments
        self._checkpoint_results: List[JudgmentResult] = []
        self._checkpoint_lock = asyncio.Lock()
        self._last_checkpoint_count: int = 0
        
        # Checkpoint state for claims extraction
        self._claims_checkpoint_lock = asyncio.Lock()
        self._last_claims_checkpoint_count: int = 0
        
        # Early stopping for coding task (saves API calls by skipping claims
        # in categories that already have a hallucination detected)
        self.early_stopping_state: CodingEarlyStoppingState | None = None
        self.package_cache: PackageVerdictCache | None = None
        if task_name == "coding":
            self.early_stopping_state = CodingEarlyStoppingState()
            self.package_cache = PackageVerdictCache()
    
    @abstractmethod
    def _get_model_config(self) -> ModelConfig:
        """Return model configuration for this pipeline type."""
        raise NotImplementedError
    
    @abstractmethod
    def _create_queues(self) -> Dict[str, MonitoredQueue]:
        """Create pipeline-specific queues. Must include 'claims' key."""
        raise NotImplementedError
    
    @abstractmethod
    def _create_workers(self, queues: Dict[str, MonitoredQueue]) -> List[Worker]:
        """Create pipeline-specific workers."""
        raise NotImplementedError
    
    @abstractmethod
    def _get_intermediate_queues(self, queues: Dict[str, MonitoredQueue]) -> List[MonitoredQueue]:
        """Return intermediate queues in pipeline order for join().
        
        These queues are waited on sequentially using join(), so they must
        be returned in the order items flow through the pipeline.
        When queue[i].join() completes, all its items have been processed
        and sent to queue[i+1].
        """
        raise NotImplementedError
    
    @abstractmethod
    def _log_config(self) -> None:
        """Log pipeline-specific configuration."""
        raise NotImplementedError
    
    def _load_conversations(self) -> None:
        """Load and optionally sample conversations.
        
        For coding tasks: samples N conversations per language (stratified sampling).
        For other tasks: samples first N conversations.
        """
        logger.info(f"Loading conversations from: {self.input_path}")
        self.conversations, self.metadata_list = load_conversations(self.input_path)
        logger.info(f"✓ Loaded {len(self.conversations)} conversations")

        if self.n_conversations is not None and self.n_conversations < len(self.conversations):
            if self.task_name == "coding":
                # Stratified sampling: N conversations per language
                self._sample_per_language()
            else:
                # Simple sampling: first N conversations
                self.conversations = self.conversations[:self.n_conversations]
                self.metadata_list = self.metadata_list[:self.n_conversations]
                logger.info(f"✓ Sampled first {self.n_conversations} conversations")
    
    def _sample_per_language(self) -> None:
        """Take first N conversations per language for coding tasks."""
        from collections import defaultdict
        
        # Group indices by language
        language_groups: dict[str, list[int]] = defaultdict(list)
        for i, meta in enumerate(self.metadata_list):
            lang = meta.get("language", "unknown")
            language_groups[lang].append(i)
        
        # Take first N from each language group
        sampled_indices = []
        
        for lang, indices in sorted(language_groups.items()):
            # Take first N (indices are already in order)
            sampled = indices[:self.n_conversations]
            sampled_indices.extend(sampled)
            logger.info(f"  {lang}: {len(sampled)}/{len(indices)} conversations")
        
        # Sort to maintain original order
        sampled_indices.sort()
        
        # Apply sampling
        self.conversations = [self.conversations[i] for i in sampled_indices]
        self.metadata_list = [self.metadata_list[i] for i in sampled_indices]
        
        logger.info(f"✓ Selected {len(self.conversations)} conversations "
                   f"(first {self.n_conversations} per language, {len(language_groups)} languages)")
    
    def _create_samplers(self) -> None:
        """Create LLM samplers based on model config."""
        self.samplers["extractor"] = get_sampler(self.model_config.extractor)
        self.samplers["judge"] = get_sampler(self.model_config.judge)
        self.samplers["judge_fallback"] = get_sampler(self.model_config.judge_fallback)
        
        if self.model_config.search:
            self.samplers["search"] = get_sampler(self.model_config.search)
    
    def _get_claims_cache_path(self) -> Path:
        """Get the path for claims cache file."""
        if self._claims_cache_path:
            return self._claims_cache_path
        # Default: same directory and name as input, with _extracted_claims_cache suffix
        return self.input_path.parent / f"{self.input_path.stem}_extracted_claims_cache.jsonl"
    
    def _load_claims_cache(self) -> List[ClaimItem] | None:
        """Load claims from cache file if it exists.
        
        Returns:
            List of ClaimItem if cache exists and is valid, None otherwise.
        """
        cache_path = self._get_claims_cache_path()
        
        if not cache_path.exists():
            logger.info(f"No claims cache found at: {cache_path}")
            return None
        
        logger.info(f"Loading claims from cache: {cache_path}")
        
        try:
            claims = []
            with open(cache_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        claim = ClaimItem.from_cache_dict(data)
                        claims.append(claim)
                    except json.JSONDecodeError as e:
                        # Stop at corrupted line (likely partial write from interruption)
                        logger.warning(f"Corrupted cache entry at line {line_num}, "
                                      f"using {len(claims)} valid entries. Error: {e}")
                        break
            
            logger.info(f"✓ Loaded {len(claims)} claims from cache")
            return claims
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load claims cache: {e}")
            return None
    
    def _filter_cache_for_current_conversations(self, cached_claims: List[ClaimItem]) -> Tuple[List[ClaimItem], Set[int]]:
        """Filter cached claims to only include conversations in current input.
        
        Returns:
            Tuple of (filtered_claims, cached_conversation_ids)
        """
        # Get current conversation IDs from loaded data
        current_conv_ids = set()
        for i, meta in enumerate(self.metadata_list):
            conv_id = meta.get("conversation_id", i)
            current_conv_ids.add(conv_id)
        
        # Filter cached claims to only include current conversations
        filtered_claims = []
        cached_conv_ids = set()
        
        for claim in cached_claims:
            if claim.conversation_id in current_conv_ids:
                filtered_claims.append(claim)
                cached_conv_ids.add(claim.conversation_id)
        
        removed_count = len(cached_claims) - len(filtered_claims)
        if removed_count > 0:
            logger.info(f"  Removed {removed_count} claims from removed conversations")
        
        return filtered_claims, cached_conv_ids
    
    def _filter_out_judged_claims(self, claims: List[ClaimItem]) -> List[ClaimItem]:
        """Filter out claims that have already been judged.
        
        Uses the cached results to identify which claims have been evaluated.
        
        Returns:
            List of claims that still need to be judged.
        """
        if not self._cached_results:
            return claims
        
        # Build set of claim IDs that have been judged
        judged_claim_ids = set()
        for result in self._cached_results:
            if hasattr(result, 'details') and result.details:
                claim_evals = result.details.get("claim_evaluations", [])
                for eval_item in claim_evals:
                    if isinstance(eval_item, dict):
                        claim_id = eval_item.get("claim_id")
                        if claim_id:
                            judged_claim_ids.add(claim_id)
        
        if not judged_claim_ids:
            return claims
        
        # Filter out already-judged claims
        unjudged_claims = []
        for claim in claims:
            if claim.claim_id not in judged_claim_ids:
                unjudged_claims.append(claim)
        
        removed_count = len(claims) - len(unjudged_claims)
        if removed_count > 0:
            logger.info(f"  Filtered out {removed_count} already-judged claims")
            logger.info(f"  Remaining claims to judge: {len(unjudged_claims)}")
        
        return unjudged_claims
    
    def _get_uncached_conversations(self, cached_conv_ids: Set[int]) -> List[Tuple[int, Any, Dict]]:
        """Get list of conversations that are not in cache.
        
        Returns:
            List of (index, conversation, metadata) tuples for uncached conversations
        """
        uncached = []
        for i, (conv, meta) in enumerate(zip(self.conversations, self.metadata_list)):
            conv_id = meta.get("conversation_id", i)
            if conv_id not in cached_conv_ids:
                uncached.append((i, conv, meta))
        return uncached
    
    def _save_claims_cache(self, claims: List[ClaimItem]) -> None:
        """Save claims to cache file."""
        cache_path = self._get_claims_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(claims)} claims to cache: {cache_path}")
        
        with open(cache_path, "w", encoding="utf-8") as f:
            for claim in claims:
                f.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"✓ Saved claims cache")
    
    async def _save_claims_checkpoint(self, extractor, force: bool = False) -> None:
        """Save incremental checkpoint of extracted claims.
        
        Args:
            extractor: The ClaimExtractorWorker with output tracking enabled
            force: If True, save regardless of checkpoint interval
        """
        async with self._claims_checkpoint_lock:
            # Get current extracted claims
            extracted_claims = extractor.get_all_outputs()
            current_count = len(extracted_claims)
            claims_since_last = current_count - self._last_claims_checkpoint_count
            
            # Only save if we have enough new claims or forced
            if not force and claims_since_last < self.checkpoint_interval:
                return
            
            if current_count == 0:
                return
            
            # Merge with cached claims if any
            if self._cached_claims:
                all_claims = self._cached_claims + extracted_claims
            else:
                all_claims = extracted_claims
            
            # Save to cache file
            cache_path = self._get_claims_cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, "w", encoding="utf-8") as f:
                for claim in all_claims:
                    f.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")
            
            self._last_claims_checkpoint_count = current_count
            logger.info(f"📁 Claims checkpoint saved: {len(all_claims)} claims ({claims_since_last} new)")
    
    async def _claims_checkpoint_collector(self, extractor) -> None:
        """Background task that periodically saves claims during extraction."""
        while True:
            try:
                # Wait for checkpoint interval
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Save checkpoint if enough new claims
                await self._save_claims_checkpoint(extractor)
                
            except asyncio.CancelledError:
                # Task cancelled, save final claims checkpoint
                await self._save_claims_checkpoint(extractor, force=True)
                raise
    
    def _get_output_path(self) -> Path:
        """Get the output path for results."""
        if self.output_path is None:
            return self.input_path.parent / f"{self.input_path.stem}_eval_{self.name}.jsonl"
        return Path(self.output_path)
    
    def _load_results_cache(self) -> List[EvaluationResult] | None:
        """Load existing evaluation results from output file if it exists.
        
        Returns:
            List of EvaluationResult if file exists and is valid, None otherwise.
        """
        output_path = self._get_output_path()
        
        if not output_path.exists():
            logger.info(f"No existing results found at: {output_path}")
            return None
        
        logger.info(f"Loading existing results from: {output_path}")
        
        try:
            results = []
            with open(output_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Skip non-result records
                        if data.get("_type") != "evaluation_result":
                            continue
                        # Reconstruct EvaluationResult
                        result = EvaluationResult(
                            conversation_id=data["conversation_id"],
                            score=data["score"],
                            reasoning=data["reasoning"],
                            details=data.get("details"),
                            metadata=data.get("metadata"),
                        )
                        results.append(result)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Corrupted result entry at line {line_num}: {e}")
                        continue
            
            if results:
                logger.info(f"✓ Loaded {len(results)} cached evaluation results")
                return results
            return None
        
        except Exception as e:
            logger.warning(f"Failed to load results cache: {e}")
            return None
    
    def _filter_conversations_for_evaluation(self) -> None:
        """Filter out conversations that already have cached results.
        
        Updates self.conversations and self.metadata_list to only include
        conversations that need evaluation.
        
        A conversation is considered "complete" only if:
        1. It has results in the cache, AND
        2. The number of judged claims matches the number of extracted claims
           (if claims cache exists)
        """
        if not self._cached_results:
            return
        
        # Build map of conversation IDs to their judged claim counts
        cached_result_claims = {}
        for r in self._cached_results:
            cached_result_claims[r.conversation_id] = r.details.get("total_claims", 0) if hasattr(r, 'details') else 0
        
        self._cached_result_conv_ids = set(cached_result_claims.keys())
        
        # Load claims cache to get expected claim counts per conversation
        claims_cache_path = self._get_claims_cache_path()
        expected_claims_per_conv = {}
        
        if claims_cache_path.exists():
            try:
                with open(claims_cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            conv_id = data.get("conversation_id", -1)
                            expected_claims_per_conv[conv_id] = expected_claims_per_conv.get(conv_id, 0) + 1
                        except json.JSONDecodeError:
                            continue
                logger.debug(f"Loaded expected claim counts for {len(expected_claims_per_conv)} conversations from claims cache")
            except Exception as e:
                logger.warning(f"Could not load claims cache for comparison: {e}")
        
        # Filter to only include conversations that are incomplete
        filtered_convs = []
        filtered_meta = []
        complete_count = 0
        incomplete_count = 0
        
        for conv, meta in zip(self.conversations, self.metadata_list):
            conv_id = meta.get("conversation_id", 0)
            
            # Check if conversation has results
            if conv_id not in self._cached_result_conv_ids:
                # No results at all - needs evaluation
                filtered_convs.append(conv)
                filtered_meta.append(meta)
                continue
            
            # Has results - check if complete
            judged_claims = cached_result_claims.get(conv_id, 0)
            expected_claims = expected_claims_per_conv.get(conv_id, 0)
            
            if expected_claims > 0 and judged_claims < expected_claims:
                # Incomplete - some claims still need judging
                incomplete_count += 1
                filtered_convs.append(conv)
                filtered_meta.append(meta)
                logger.debug(f"  Conv {conv_id}: incomplete ({judged_claims}/{expected_claims} claims)")
            else:
                # Complete - skip this conversation
                complete_count += 1
        
        skipped_count = len(self.conversations) - len(filtered_convs)
        
        if skipped_count > 0 or incomplete_count > 0:
            logger.info(f"  Skipping {complete_count} fully completed conversations")
            if incomplete_count > 0:
                logger.info(f"  Re-processing {incomplete_count} incomplete conversations (partial results)")
            logger.info(f"  Processing {len(filtered_convs)} conversations total")
            
            # Store original lists for final result building
            self._all_conversations = self.conversations
            self._all_metadata_list = self.metadata_list
            
            # Update to filtered lists
            self.conversations = filtered_convs
            self.metadata_list = filtered_meta
        else:
            self._all_conversations = self.conversations
            self._all_metadata_list = self.metadata_list
    
    async def _load_input_queue(self) -> None:
        """Load conversations into input queue."""
        for i, (conv, meta) in enumerate(zip(self.conversations, self.metadata_list)):
            conv_id = meta.get("conversation_id", i)
            await self.conversation_queue.put(
                ConversationItem(
                    conversation_id=conv_id,
                    conversation=conv.to_message_list(),  # Convert Conversation to list[dict]
                    metadata=meta,
                    max_claims_per_turn=self.max_claims_per_turn,
                ),
                claim_id=f"conv-{conv_id}",
                conversation_id=conv_id,
            )
    
        self.conversation_queue.close()
    
    async def _collect_results(self) -> List[JudgmentResult]:
        """Collect all results from results queue."""
        all_results: List[JudgmentResult] = []
        while True:
            item = await self.results_queue.get_nowait()

            # Use this breaking instead of not self.results_queue.empty to avoid race condition
            if item is None:
                break

            all_results.append(item.data)
            self.results_queue.task_done()
    
        return all_results
    
    def _build_evaluation_results(self, judgments: List[JudgmentResult]) -> List[EvaluationResult]:
        """Build EvaluationResult objects from judgments."""
        # Group by conversation
        results_by_conv: Dict[int, List[JudgmentResult]] = {}
        for result in judgments:
            conv_id = result.conversation_id
            if conv_id not in results_by_conv:
                results_by_conv[conv_id] = []
            results_by_conv[conv_id].append(result)

        # Build EvaluationResults
        final_results: List[EvaluationResult] = []
        for meta in self.metadata_list:
            conv_id = meta.get("conversation_id", 0)
            conv_judgments = results_by_conv.get(conv_id, [])

            total_claims = len(conv_judgments)
            if total_claims == 0:
                score = 1.0
                reasoning = "No verifiable claims found"
                hallucinations = 0
                input_use_fallback_count = 0
                judge_used_websearch_fallback_count = 0
                snippets_only_count = 0
                # Coding-specific
                import_hallucinations = 0
                install_hallucinations = 0
                function_hallucinations = 0
            else:
                hallucinations = sum(1 for j in conv_judgments if j.hallucination.lower() == "yes")
                score = 1.0 - (hallucinations / total_claims)
                reasoning = f"Found {hallucinations}/{total_claims} hallucinated claims"
                input_use_fallback_count = sum(1 for j in conv_judgments if j.input_use_fallback)
                judge_used_websearch_fallback_count = sum(1 for j in conv_judgments if j.judge_used_websearch_fallback)
                snippets_only_count = sum(1 for j in conv_judgments if j.snippets_only)
                # Coding-specific hallucination counts
                import_hallucinations = sum(1 for j in conv_judgments if j.hallucinated_import_detected)
                install_hallucinations = sum(1 for j in conv_judgments if j.hallucinated_install_detected)
                function_hallucinations = sum(1 for j in conv_judgments if j.hallucinated_function_usage_detected)

            # Build details dict
            details = {
                    "total_claims": total_claims,
                    "hallucinations": hallucinations,
                    "input_use_fallback_count": input_use_fallback_count,
                    "judge_used_websearch_fallback_count": judge_used_websearch_fallback_count,
                    "snippets_only_count": snippets_only_count,
                    "claim_evaluations": [j.to_dict() for j in conv_judgments],
            }
            
            # Add coding-specific details if this is a coding task
            if self.task_name == "coding":
                # Aggregate boolean flags (ANY element with hallucination = True)
                any_import_halluc = any(j.hallucinated_import_detected for j in conv_judgments)
                any_install_halluc = any(j.hallucinated_install_detected for j in conv_judgments)
                any_function_halluc = any(j.hallucinated_function_usage_detected for j in conv_judgments)
                
                # Calculate RESPONSE-LEVEL (turn-level) hallucination rates
                # Group claims by turn_number
                turns_data: Dict[int, Dict[str, bool]] = {}
                for j in conv_judgments:
                    turn = j.turn_number
                    if turn not in turns_data:
                        turns_data[turn] = {"import": False, "install": False, "function": False, "any": False}
                    if j.hallucinated_import_detected:
                        turns_data[turn]["import"] = True
                        turns_data[turn]["any"] = True
                    if j.hallucinated_install_detected:
                        turns_data[turn]["install"] = True
                        turns_data[turn]["any"] = True
                    if j.hallucinated_function_usage_detected:
                        turns_data[turn]["function"] = True
                        turns_data[turn]["any"] = True
                
                total_responses = len(turns_data)
                if total_responses > 0:
                    # Count hallucinated responses per category
                    import_halluc_responses = sum(1 for t in turns_data.values() if t["import"])
                    install_halluc_responses = sum(1 for t in turns_data.values() if t["install"])
                    function_halluc_responses = sum(1 for t in turns_data.values() if t["function"])
                    overall_halluc_responses = sum(1 for t in turns_data.values() if t["any"])
                    
                    # Calculate rates
                    import_halluc_rate = import_halluc_responses / total_responses
                    install_halluc_rate = install_halluc_responses / total_responses
                    function_halluc_rate = function_halluc_responses / total_responses
                    overall_halluc_rate = overall_halluc_responses / total_responses
                else:
                    import_halluc_responses = install_halluc_responses = function_halluc_responses = overall_halluc_responses = 0
                    import_halluc_rate = install_halluc_rate = function_halluc_rate = overall_halluc_rate = 0.0
                
                details.update({
                    "hallucinated_import_detected": any_import_halluc,
                    "hallucinated_install_detected": any_install_halluc,
                    "hallucinated_function_usage_detected": any_function_halluc,
                    # Claim-level counts (for reference)
                    "import_hallucination_count": import_hallucinations,
                    "install_hallucination_count": install_hallucinations,
                    "function_hallucination_count": function_hallucinations,
                    # Response-level (turn-level) stats
                    "total_responses": total_responses,
                    "import_hallucinated_responses": import_halluc_responses,
                    "install_hallucinated_responses": install_halluc_responses,
                    "function_hallucinated_responses": function_halluc_responses,
                    "overall_hallucinated_responses": overall_halluc_responses,
                    # Hallucination rates (hallucinated responses / total responses)
                    "import_hallucination_rate": import_halluc_rate,
                    "install_hallucination_rate": install_halluc_rate,
                    "function_hallucination_rate": function_halluc_rate,
                    "overall_hallucination_rate": overall_halluc_rate,
                })
                
                # Update reasoning for coding task with response-level rates
                if overall_halluc_responses > 0:
                    rate_parts = [f"Overall: {overall_halluc_responses}/{total_responses} ({overall_halluc_rate:.1%})"]
                    if import_halluc_responses > 0:
                        rate_parts.append(f"Import: {import_halluc_responses}/{total_responses} ({import_halluc_rate:.1%})")
                    if install_halluc_responses > 0:
                        rate_parts.append(f"Install: {install_halluc_responses}/{total_responses} ({install_halluc_rate:.1%})")
                    if function_halluc_responses > 0:
                        rate_parts.append(f"Function: {function_halluc_responses}/{total_responses} ({function_halluc_rate:.1%})")
                    reasoning = f"Response hallucination rates - {'; '.join(rate_parts)}"
                else:
                    reasoning = f"No hallucinations detected in {total_responses} responses"

            final_results.append(EvaluationResult(
                conversation_id=conv_id,
                score=score,
                reasoning=reasoning,
                details=details,
                metadata=meta,
            ))

        return final_results
    
    def _save_results(self, results: List[EvaluationResult]) -> Path:
        """Save evaluation results to file."""
        if self.output_path is None:
            output_path = self.input_path.parent / f"{self.input_path.stem}_eval_{self.name}.jsonl"
        else:
            output_path = Path(self.output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to: {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                record = {
                    "_type": "evaluation_result",
                    **asdict(result),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"✓ Saved {len(results)} evaluation results")
        return output_path
    
    def _get_checkpoint_path(self) -> Path:
        """Get the path for checkpoint file (same as output file)."""
        if self.output_path is None:
            return self.input_path.parent / f"{self.input_path.stem}_eval_{self.name}.jsonl"
        return Path(self.output_path)
    
    async def _save_checkpoint(self, force: bool = False) -> None:
        """Save incremental checkpoint of current results.
        
        Args:
            force: If True, save regardless of checkpoint interval
        """
        async with self._checkpoint_lock:
            current_count = len(self._checkpoint_results)
            results_since_last = current_count - self._last_checkpoint_count
            
            # Only save if we have enough new results or forced
            if not force and results_since_last < self.checkpoint_interval:
                return
            
            if current_count == 0:
                return
            
            # Build evaluation results from collected judgments
            checkpoint_eval_results = self._build_evaluation_results(self._checkpoint_results)
            
            # Merge with cached results if any
            if self._cached_results:
                all_results = self._cached_results + checkpoint_eval_results
                all_results.sort(key=lambda r: r.conversation_id)
            else:
                all_results = checkpoint_eval_results
            
            # Save to checkpoint file
            checkpoint_path = self._get_checkpoint_path()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for result in all_results:
                    record = {
                        "_type": "evaluation_result",
                        **asdict(result),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            self._last_checkpoint_count = current_count
            logger.info(f"📁 Checkpoint saved: {len(all_results)} results ({results_since_last} new)")
    
    async def _checkpoint_collector(self) -> None:
        """Background task that collects results and saves checkpoints."""
        while True:
            try:
                # Try to get a result with timeout
                try:
                    item = await asyncio.wait_for(
                        self.results_queue.get(),
                        timeout=5.0
                    )
                    # Extract the actual JudgmentResult from the QueueItem wrapper
                    self._checkpoint_results.append(item.data)
                    self.results_queue.task_done()
                    
                    # Check if we should save a checkpoint
                    await self._save_checkpoint()
                    
                except asyncio.TimeoutError:
                    # No result available, check if queue is closed and empty
                    if self.results_queue._closed and self.results_queue.empty():
                        break
                    continue
                    
            except asyncio.CancelledError:
                # Task cancelled, save final checkpoint
                await self._save_checkpoint(force=True)
                raise
    
    async def _log_summary(self, results: List[EvaluationResult], output_path: Path) -> None:
        """Log final summary."""
        total_claims = sum(r.details.get("total_claims", 0) for r in results)
        total_hallucinations = sum(r.details.get("hallucinations", 0) for r in results)
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Conversations: {len(results)}")
        logger.info(f"Total claims: {total_claims}")
        logger.info(f"Hallucinations: {total_hallucinations}")
        logger.info(f"Overall accuracy: {1 - (total_hallucinations / max(total_claims, 1)):.1%}")
        
        # Early stopping stats for coding task
        if self.early_stopping_state:
            early_stop_stats = await self.early_stopping_state.get_stats()
            skipped = early_stop_stats.get("total_claims_skipped", 0)
            fully_stopped = early_stop_stats.get("conversations_fully_stopped", 0)
            if skipped > 0:
                logger.info(f"Early stopping: {skipped} claims skipped, {fully_stopped} conversations fully stopped")
        
        # Package cache stats for coding task
        if self.package_cache:
            cache_stats = await self.package_cache.get_stats()
            whitelist_skips = cache_stats.get("whitelist_skips", 0)
            cache_hits = cache_stats.get("cache_hits", 0)
            if whitelist_skips > 0 or cache_hits > 0:
                logger.info(f"Package cache: {whitelist_skips} whitelist skips, {cache_hits} cache hits")
        
        logger.info(f"Results: {output_path}")
        logger.info("=" * 70)
    
    async def run(self) -> Path:
        """Run the full pipeline."""
        # Validate input
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Setup
        self.model_config = self._get_model_config()
        
        # Log configuration
        logger.info("=" * 70)
        logger.info(f"QUEUE-BASED EVALUATION PIPELINE: {self.name}")
        logger.info(f"Task: {self.task_name}")
        logger.info("=" * 70)
        logger.info(f"Input: {self.input_path}")
        self._log_config()
        if self.checkpoint_interval > 0:
            logger.info(f"Checkpointing: Every {self.checkpoint_interval} judgments")
        else:
            logger.info("Checkpointing: Disabled")
        logger.info("=" * 70 + "\n")
        
        # Load data
        self._load_conversations()
        logger.info("")
        
        # Try to load existing results and filter out already-evaluated conversations
        self._cached_results = self._load_results_cache()
        if self._cached_results:
            self._filter_conversations_for_evaluation()
            
            # If all conversations are cached, skip everything
            if len(self.conversations) == 0:
                logger.info(f"✓ All conversations have cached results - nothing to evaluate")
                output_path = self._get_output_path()
                await self._log_summary(self._cached_results, output_path)
                return output_path
        else:
            # Initialize for later use
            self._all_conversations = self.conversations
            self._all_metadata_list = self.metadata_list
        
        logger.info("")
        
        # Try to load claims from cache and determine what needs extraction
        raw_cached_claims = self._load_claims_cache()
        
        # Determine caching strategy
        self._cached_claims = None
        self._cached_conv_ids: Set[int] = set()
        self._uncached_conversations: List[Tuple[int, Any, Dict]] = []
        need_extraction = True
        
        if raw_cached_claims is not None:
            # Filter cache to only include current conversations
            self._cached_claims, self._cached_conv_ids = self._filter_cache_for_current_conversations(raw_cached_claims)
            
            # Filter out claims that have already been judged (for incomplete conversations)
            if self._cached_results:
                self._cached_claims = self._filter_out_judged_claims(self._cached_claims)
            
            # Find conversations that need extraction
            self._uncached_conversations = self._get_uncached_conversations(self._cached_conv_ids)
            
            if len(self._uncached_conversations) == 0:
                # All conversations are cached - no extraction needed
                need_extraction = False
                logger.info(f"✓ All {len(self.conversations)} conversations have cached claims")
            else:
                logger.info(f"  {len(self._cached_conv_ids)} conversations cached, "
                           f"{len(self._uncached_conversations)} need extraction")
        
        # Create samplers
        self._create_samplers()
        
        # Create core queues
        self.conversation_queue = MonitoredQueue("conversations")
        self.results_queue = MonitoredQueue("results")
        
        # Create pipeline-specific queues
        queues = self._create_queues()
        self.claims_queue = queues["claims"]
        
        # Create workers list - extractor only if we need extraction
        all_workers = []
        extractor = None
        
        if need_extraction:
            # Create extractor (common to all pipelines)
            # For coding tasks, limit claims per category to speed up evaluation
            max_per_cat = self.max_claims_per_category if self.task_name == "coding" else None
            extractor = ClaimExtractorWorker(
                input_queue=self.conversation_queue,
                output_queue=self.claims_queue,
                sampler=self.samplers["extractor"],
                strategy=self.strategy,
                num_workers=self.worker_config.num_extractors,
                max_claims_per_category=max_per_cat,
            )
            # Enable output tracking for caching
            extractor.enable_output_tracking()
            all_workers.append(extractor)
        
        # Create pipeline-specific workers
        workers = self._create_workers(queues)
        all_workers.extend(workers)
        
        # Create pipeline
        self.pipeline = Pipeline(name=self.name)
        
        # Register queues
        self.pipeline.add_queue(self.conversation_queue)
        for queue in queues.values():
            self.pipeline.add_queue(queue)
        self.pipeline.add_queue(self.results_queue)
        
        # Register workers
        for worker in all_workers:
            self.pipeline.add_worker(worker)
        
        self.pipeline.set_results_queue(self.results_queue)
        
        if not need_extraction:
            # All claims are cached - load directly into claims queue
            logger.info(f"Using cached claims, skipping extraction...")
            await self._load_cached_claims_to_queue()
            total_items = len(self._cached_claims)
        elif self._cached_claims:
            # Partial cache - load cached claims AND run extraction for uncached
            logger.info(f"Loading {len(self._cached_claims)} cached claims and extracting {len(self._uncached_conversations)} new conversations...")
            await self._load_cached_claims_to_queue_partial()
            await self._load_uncached_conversations_to_queue()
            total_items = len(self._cached_claims) + len(self._uncached_conversations)
        else:
            # No cache - extract all conversations
            await self._load_input_queue()
            total_items = len(self.conversations)
        logger.info(f"Starting pipeline with {total_items} {'claims' if not need_extraction else 'items'}...")
        
        # Create monitor
        monitor = QueueMonitor(
            queues=self.pipeline.queues,
            workers=self.pipeline.workers,
            total_items=total_items,
        )
        monitor.set_final_queue(self.results_queue)
        monitor.on_update(lambda s: print(str(s)))
        
        # Start everything
        await monitor.start(interval=self.monitor_interval)
        
        for worker in all_workers:
            await worker.start()
        
        # Start checkpoint collector as background task (if enabled)
        checkpoint_task = None
        claims_checkpoint_task = None
        if self.checkpoint_interval > 0:
            checkpoint_task = asyncio.create_task(self._checkpoint_collector())
        
        if need_extraction:
            # Start claims checkpoint collector during extraction (if enabled)
            if self.checkpoint_interval > 0:
                claims_checkpoint_task = asyncio.create_task(
                    self._claims_checkpoint_collector(extractor)
                )
            
            # Wait for extraction pipeline to complete using queue.join()
            await self.conversation_queue.join()
            
            # Stop claims checkpoint collector
            if claims_checkpoint_task:
                try:
                    claims_checkpoint_task.cancel()
                    await claims_checkpoint_task
                except asyncio.CancelledError:
                    pass
            
            # Save final claims cache after extraction completes
            extracted_claims = extractor.get_all_outputs()
            
            if self._cached_claims:
                all_claims = self._cached_claims + extracted_claims
                logger.info(f"Merged {len(self._cached_claims)} cached + {len(extracted_claims)} new = {len(all_claims)} total claims")
            else:
                all_claims = extracted_claims
            
            self._save_claims_cache(all_claims)
            logger.info(f"📁 Claims cache saved ({len(all_claims)} total claims)")
        
        # Wait for each intermediate queue in order
        for queue in self._get_intermediate_queues(queues):
            await queue.join()
        
        # Stop workers
        for worker in all_workers:
            await worker.stop()
        
        await monitor.stop()
        
        # Collect results - either from checkpoint collector or directly from queue
        if checkpoint_task:
            # Wait for checkpoint collector to finish and save final checkpoint
            try:
                checkpoint_task.cancel()
                await checkpoint_task
            except asyncio.CancelledError:
                pass
            
            # Final save with all results
            await self._save_checkpoint(force=True)
            
            # Build final results from checkpoint
            new_results = self._build_evaluation_results(self._checkpoint_results)
            logger.info(f"\n✓ Collected {len(self._checkpoint_results)} judgment results")
        else:
            # No checkpointing - collect results directly from queue
            all_judgments = await self._collect_results()
            logger.info(f"\n✓ Collected {len(all_judgments)} judgment results")
            new_results = self._build_evaluation_results(all_judgments)
        
        # Merge with cached results if any
        if self._cached_results:
            # Combine cached results with new results
            final_results = self._cached_results + new_results
            logger.info(f"Merged {len(self._cached_results)} cached + {len(new_results)} new = {len(final_results)} total results")
            
            # Sort by conversation_id for consistent output
            final_results.sort(key=lambda r: r.conversation_id)
        else:
            final_results = new_results
        
        output_path = self._save_results(final_results)
        await self._log_summary(final_results, output_path)
        
        # Clean up shared HTTP clients
        await close_shared_client()
        await close_pdf_session()
        
        return output_path
    
    async def _load_cached_claims_to_queue(self) -> None:
        """Load cached claims directly into the claims queue (full cache mode)."""
        for claim in self._cached_claims:
            await self.claims_queue.put(
                claim,
                claim_id=claim.claim_id,
                conversation_id=claim.conversation_id,
            )
        self.claims_queue.close()
    
    async def _load_cached_claims_to_queue_partial(self) -> None:
        """Load cached claims to queue without closing (partial cache mode).
        
        Used when we also need to extract claims for uncached conversations.
        """
        for claim in self._cached_claims:
            await self.claims_queue.put(
                claim,
                claim_id=claim.claim_id,
                conversation_id=claim.conversation_id,
            )
    
    async def _load_uncached_conversations_to_queue(self) -> None:
        """Load only uncached conversations into input queue for extraction."""
        for i, conv, meta in self._uncached_conversations:
            conv_id = meta.get("conversation_id", i)
            await self.conversation_queue.put(
                ConversationItem(
                    conversation_id=conv_id,
                    conversation=conv.to_message_list(),
                    metadata=meta,
                    max_claims_per_turn=self.max_claims_per_turn,
                ),
                claim_id=f"conv-{conv_id}",
                conversation_id=conv_id,
            )
        
        self.conversation_queue.close()


# =============================================================================
# OpenAI Pipeline (Direct websearch via LLM)
# =============================================================================

class OpenAIPipeline(BasePipeline):
    """Pipeline using OpenAI's built-in websearch for judgment."""
    
    name = "openai"
    
    def _get_model_config(self) -> ModelConfig:
        return ModelConfig(
            extractor="gpt-5-mini-minimal",
            judge="gpt-5-mini-medium-websearch",
            judge_fallback="gpt-5-mini-medium-websearch",
            search=None,
        )
    
    def _log_config(self) -> None:
        logger.info(f"Models: Extractor={self.model_config.extractor}, Judge={self.model_config.judge}")
        logger.info(f"Workers: Extract={self.worker_config.num_extractors}, Judge={self.worker_config.num_judges}")
    
    def _create_queues(self) -> Dict[str, MonitoredQueue]:
        return {
            "claims": MonitoredQueue[ClaimItem]("claims"),
            "filtered": MonitoredQueue[FilteredContent]("filtered"),
        }
    
    def _create_workers(self, queues: Dict[str, MonitoredQueue]) -> List[Worker]:
        # ClaimToFiltered passthrough worker
        passthrough = ClaimToFilteredWorker(
            input_queue=queues["claims"],
            output_queue=queues["filtered"],
            num_workers=self.worker_config.num_extractors,
        )
        
        # Judge with websearch
        judge = JudgeWorker(
            input_queue=queues["filtered"],
            output_queue=self.results_queue,
            sampler=self.samplers["judge"],
            strategy=self.strategy,
            sampler_fallback=self.samplers["judge_fallback"],
            num_workers=self.worker_config.num_judges,
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        return [passthrough, judge]
    
    def _get_intermediate_queues(self, queues: Dict[str, MonitoredQueue]) -> List[MonitoredQueue]:
        return [queues["claims"], queues["filtered"]]


# =============================================================================
# Serper Pipeline (Search API with snippets)
# =============================================================================

class SerperPipeline(BasePipeline):
    """Pipeline using Serper API for web search (snippets only)."""
    
    name = "serper"
    
    def _get_model_config(self) -> ModelConfig:
        return ModelConfig(
            extractor="gpt-5-mini-minimal",
            search="gpt-5-mini-minimal",
            judge="gpt-5-mini-medium",
            judge_fallback="gpt-5-mini-medium-websearch",
        )
    
    def _log_config(self) -> None:
        logger.info(f"Models: Extractor={self.model_config.extractor}, "
                   f"Search={self.model_config.search}, Judge={self.model_config.judge}")
        logger.info(f"Workers: Extract={self.worker_config.num_extractors}, "
                   f"Search={self.worker_config.num_searchers}, "
                   f"Judge={self.worker_config.num_judges}")
        logger.info("Mode: Serper snippets only (direct to judge)")
    
    def _create_queues(self) -> Dict[str, MonitoredQueue]:
        return {
            "claims": MonitoredQueue[ClaimItem]("claims"),
            "search": MonitoredQueue[SearchTask]("search_tasks"),
            "filtered": MonitoredQueue[FilteredContent]("filtered"),
        }
    
    def _create_workers(self, queues: Dict[str, MonitoredQueue]) -> List[Worker]:
        # Search via Serper API
        searcher = WebSearcherWorker(
            input_queue=queues["claims"],
            output_queue=queues["search"],
            search_sampler=self.samplers["search"],
            claim_text_builder=self.strategy.build_textual_claim_for_websearch,
            strategy=self.strategy,
            num_workers=self.worker_config.num_searchers,
            rate_limit_delay=0.1,
            max_searches=1 if self.task_name == "coding" else 3,  # Coding needs fewer iterations
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        # Convert SearchTask directly to FilteredContent (skipping extraction/filtering)
        search_to_filtered = SearchToFilteredWorker(
            input_queue=queues["search"],
            output_queue=queues["filtered"],
            num_workers=self.worker_config.num_searchers,
        )
        
        # Judge
        judge = JudgeWorker(
            input_queue=queues["filtered"],
            output_queue=self.results_queue,
            sampler=self.samplers["judge"],
            strategy=self.strategy,
            sampler_fallback=self.samplers["judge_fallback"],
            num_workers=self.worker_config.num_judges,
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        return [searcher, search_to_filtered, judge]
    
    def _get_intermediate_queues(self, queues: Dict[str, MonitoredQueue]) -> List[MonitoredQueue]:
        return [queues["claims"], queues["search"], queues["filtered"]]


# =============================================================================
# Webscraper Pipeline (Full scraping with PDF support)
# =============================================================================

class WebscraperPipeline(BasePipeline):
    """Pipeline with full web scraping and PDF extraction."""
    
    name = "webscraper"
    
    # Store aggregator separately since it's not a standard Worker
    _aggregator: ContentAggregatorWorker | None = None
    
    def _get_model_config(self) -> ModelConfig:
        return ModelConfig(
            extractor="gpt-5-mini-minimal",
            search="gpt-5-mini-minimal",
            judge="gpt-5-mini-medium",
            judge_fallback="gpt-5-mini-medium-websearch",
        )
    
    def _log_config(self) -> None:
        logger.info(f"Models: Extractor={self.model_config.extractor}, "
                   f"Search={self.model_config.search}, Judge={self.model_config.judge}")
        logger.info(f"Workers: Extract={self.worker_config.num_extractors}, "
                   f"Search={self.worker_config.num_searchers}, "
                   f"Fetch={self.worker_config.num_fetchers}, "
                   f"PDF={self.worker_config.num_pdf_converters}, "
                   f"Filter={self.worker_config.num_filters}, "
                   f"Judge={self.worker_config.num_judges}")
    
    def _create_queues(self) -> Dict[str, MonitoredQueue]:
        return {
            "claims": MonitoredQueue[ClaimItem]("claims"),
            "search": MonitoredQueue[SearchTask]("search_tasks"),
            "content": MonitoredQueue[ContentItem]("content"),
            "pdf": MonitoredQueue[PDFTask]("pdf_tasks"),
            "pdf_result": MonitoredQueue[PDFResult]("pdf_results"),
            "aggregated": MonitoredQueue[ContentItem]("aggregated"),
            "filtered": MonitoredQueue[FilteredContent]("filtered"),
        }
    
    def _create_workers(self, queues: Dict[str, MonitoredQueue]) -> List[Worker]:
        # Search
        searcher = WebSearcherWorker(
            input_queue=queues["claims"],
            output_queue=queues["search"],
            search_sampler=self.samplers["search"],
            claim_text_builder=self.strategy.build_textual_claim_for_websearch,
            strategy=self.strategy,
            num_workers=self.worker_config.num_searchers,
            rate_limit_delay=0.1,
            max_searches=1 if self.task_name == "coding" else 3,  # Coding needs fewer iterations
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        # Fetch with PDF queue
        fetcher = WebFetcherWorker(
            input_queue=queues["search"],
            output_queue=queues["content"],
            pdf_queue=queues["pdf"],
            num_workers=self.worker_config.num_fetchers,
            early_stopping_state=self.early_stopping_state,
        )
        
        # PDF conversion - outputs to pdf_result queue
        pdf_converter = PDFConverterWorker(
            input_queue=queues["pdf"],
            output_queue=queues["pdf_result"],
            num_workers=self.worker_config.num_pdf_converters,
        )
        
        # Content aggregator (merges HTML content with PDF results)
        self._aggregator = ContentAggregatorWorker(
            content_queue=queues["content"],
            pdf_queue=queues["pdf_result"],
            output_queue=queues["aggregated"],
        )
        
        # Filter - now reads from aggregated queue
        content_filter = ContentFilterWorker(
            input_queue=queues["aggregated"],
            output_queue=queues["filtered"],
            claim_text_builder=self.strategy.build_textual_claim_for_judging,
            num_workers=self.worker_config.num_filters,
            early_stopping_state=self.early_stopping_state,
        )
        
        # Judge
        judge = JudgeWorker(
            input_queue=queues["filtered"],
            output_queue=self.results_queue,
            sampler=self.samplers["judge"],
            strategy=self.strategy,
            sampler_fallback=self.samplers["judge_fallback"],
            num_workers=self.worker_config.num_judges,
            early_stopping_state=self.early_stopping_state,
            package_cache=self.package_cache,
        )
        
        return [searcher, fetcher, pdf_converter, self._aggregator, content_filter, judge]
    
    def _get_intermediate_queues(self, queues: Dict[str, MonitoredQueue]) -> List[MonitoredQueue]:
        return [
            queues["claims"], 
            queues["search"], 
            queues["content"],
            queues["pdf"],
            queues["pdf_result"],
            queues["aggregated"],
            queues["filtered"],
        ]
    
    async def run(self) -> Path:
        """Run the full pipeline with aggregator support."""
        # Standard run but with aggregator start/stop handling
        path = await super().run()
        return path


# =============================================================================
# Helper Worker for OpenAI Pipeline
# =============================================================================

class ClaimToFilteredWorker(Worker[ClaimItem, FilteredContent]):
    """Passthrough worker that converts ClaimItem to FilteredContent."""
    
    def __init__(
        self,
        input_queue: MonitoredQueue[ClaimItem],
        output_queue: MonitoredQueue[FilteredContent],
        num_workers: int = 5,
    ):
        super().__init__(
            name="ClaimToFiltered",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
        )
    
    async def process(self, item: ClaimItem, item_wrapper) -> FilteredContent:
        return FilteredContent(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            claim=item,
            filtered_content="",
            search_results_text="",
            use_fallback=True,  # Use websearch-enabled judge
        )


class SearchToFilteredWorker(Worker[SearchTask, FilteredContent]):
    """Worker that converts SearchTask directly to FilteredContent."""
    
    def __init__(
        self,
        input_queue: MonitoredQueue[SearchTask],
        output_queue: MonitoredQueue[FilteredContent],
        num_workers: int = 5,
    ):
        super().__init__(
            name="SearchToFiltered",
            input_queue=input_queue,
            output_queue=output_queue,
            num_workers=num_workers,
        )
    
    async def process(self, item: SearchTask, item_wrapper) -> FilteredContent:
        return FilteredContent(
            claim_id=item.claim_id,
            conversation_id=item.conversation_id,
            claim=item.claim,
            filtered_content="",
            search_results_text=item.search_results_text,
            queries=item.queries_executed,  # Pass through search queries
            use_fallback=True,  # Use snippet-based judgment
            whitelist_skip=item.whitelist_skip,  # Propagate whitelist skip
        )


# =============================================================================
# Coding Direct Pipeline (OpenAI Websearch without claim extraction)
# =============================================================================

class CodingDirectPipeline:
    """Pipeline for coding hallucination detection using OpenAI websearch directly.
    
    This pipeline does NOT extract individual claims. Instead, it:
    1. Takes each assistant turn as a whole
    2. Uses OpenAI with websearch to detect all hallucinations at once
    3. Returns per-turn results with detected hallucinations
    
    This is faster and simpler but may miss some nuanced hallucinations that
    claim-by-claim evaluation would catch.
    """
    
    name = "coding_direct"
    
    def __init__(
        self,
        input_path: Path,
        output_path: Path | None = None,
        worker_config: WorkerConfig | None = None,
        task_name: str = "coding",
        n_conversations: int | None = None,
        seed: int = 42,
        monitor_interval: float = 3.0,
        base_path: Path | str | None = None,
        # These are ignored but kept for API compatibility
        max_claims_per_turn: int | None = None,
        claims_cache_path: Path | str | None = None,
        checkpoint_interval: int = 100,
        max_claims_per_category: int | None = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.worker_config = worker_config or WorkerConfig()
        self.task_name = task_name
        self.n_conversations = n_conversations
        self.seed = seed
        self.monitor_interval = monitor_interval
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        
        # State
        self.conversations = []
        self.metadata_list = []
    
    def _load_conversations(self) -> None:
        """Load conversations from input file with stratified sampling for coding tasks."""
        self.conversations, self.metadata_list = load_conversations(self.input_path)
        
        if self.n_conversations and self.n_conversations < len(self.conversations):
            if self.task_name == "coding":
                # Stratified sampling: N conversations per language (same as base pipeline)
                self._sample_per_language()
            else:
                # Simple sampling: first N conversations
                random.seed(self.seed)
                indices = list(range(len(self.conversations)))
                random.shuffle(indices)
                indices = sorted(indices[:self.n_conversations])
                self.conversations = [self.conversations[i] for i in indices]
                self.metadata_list = [self.metadata_list[i] for i in indices]
                logger.info(f"✓ Sampled first {self.n_conversations} conversations")
        else:
            logger.info(f"Loaded {len(self.conversations)} conversations")
    
    def _sample_per_language(self) -> None:
        """Take first N conversations per language for coding tasks (same as base pipeline)."""
        from collections import defaultdict
        
        # Group indices by language
        language_groups: dict[str, list[int]] = defaultdict(list)
        for i, meta in enumerate(self.metadata_list):
            lang = meta.get("language", "unknown")
            language_groups[lang].append(i)
        
        # Take first N from each language group
        sampled_indices = []
        
        for lang, indices in sorted(language_groups.items()):
            # Take first N (indices are already in order)
            sampled = indices[:self.n_conversations]
            sampled_indices.extend(sampled)
            logger.info(f"  {lang}: {len(sampled)}/{len(indices)} conversations")
        
        # Sort to maintain original order
        sampled_indices.sort()
        
        # Apply sampling
        self.conversations = [self.conversations[i] for i in sampled_indices]
        self.metadata_list = [self.metadata_list[i] for i in sampled_indices]
        
        logger.info(f"✓ Selected {len(self.conversations)} conversations "
                   f"(first {self.n_conversations} per language, {len(language_groups)} languages)")
    
    def _get_output_path(self) -> Path:
        """Get the output path for results."""
        if self.output_path is None:
            return self.input_path.parent / f"{self.input_path.stem}_eval_{self.name}.jsonl"
        return Path(self.output_path)
    
    def _extract_assistant_turns(self) -> List[Tuple[int, int, str]]:
        """Extract all assistant turns from conversations.
        
        Returns list of (conversation_id, turn_number, content) tuples.
        """
        from .workers import TurnItem
        
        turns = []
        for conv, meta in zip(self.conversations, self.metadata_list):
            conv_id = meta.get("conversation_id", 0)
            messages = conv.to_message_list()
            
            for i, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and len(content.strip()) > 0:
                        turns.append((conv_id, i, content))
        
        return turns
    
    def _build_evaluation_results(
        self, 
        results: List["DirectCodingResult"],
    ) -> List[EvaluationResult]:
        """Build EvaluationResult objects from direct coding results."""
        from .workers import DirectCodingResult
        
        # Group by conversation
        results_by_conv: Dict[int, List[DirectCodingResult]] = {}
        for result in results:
            conv_id = result.conversation_id
            if conv_id not in results_by_conv:
                results_by_conv[conv_id] = []
            results_by_conv[conv_id].append(result)
        
        # Build EvaluationResults
        final_results: List[EvaluationResult] = []
        for meta in self.metadata_list:
            conv_id = meta.get("conversation_id", 0)
            conv_results = results_by_conv.get(conv_id, [])
            
            total_turns = len(conv_results)
            if total_turns == 0:
                score = 1.0
                reasoning = "No assistant turns found"
                import_hallucinations = 0
                install_hallucinations = 0
                function_hallucinations = 0
            else:
                # Count hallucinations
                import_hallucinations = sum(1 for r in conv_results if r.hallucinated_import_detected)
                install_hallucinations = sum(1 for r in conv_results if r.hallucinated_install_detected)
                function_hallucinations = sum(1 for r in conv_results if r.hallucinated_function_usage_detected)
                
                total_hallucinations = import_hallucinations + install_hallucinations + function_hallucinations
                turns_with_hallucinations = sum(1 for r in conv_results if r.has_hallucination)
                
                score = 1.0 - (turns_with_hallucinations / total_turns) if total_turns > 0 else 1.0
                
                # Build detailed reasoning with breakdown
                reasoning_parts = [f"Evaluated {total_turns} turn{'s' if total_turns != 1 else ''}"]
                
                if total_hallucinations == 0:
                    reasoning_parts.append(f"0 hallucinations ({import_hallucinations} import, {install_hallucinations} install, {function_hallucinations} function)")
                    reasoning_parts.append("All packages and API calls verified against official documentation")
                else:
                    reasoning_parts.append(f"{total_hallucinations} hallucination{'s' if total_hallucinations != 1 else ''} ({import_hallucinations} import, {install_hallucinations} install, {function_hallucinations} function)")
                    
                    # Collect specific issues from turn evaluations
                    issues = []
                    for r in conv_results:
                        if r.hallucinated_import_detected and r.hallucinated_imports:
                            for imp in r.hallucinated_imports[:2]:  # Limit to first 2
                                pkg = imp.get("package", "unknown")
                                issues.append(f"{pkg} import")
                        if r.hallucinated_install_detected and r.hallucinated_installs:
                            for inst in r.hallucinated_installs[:2]:  # Limit to first 2
                                pkg = inst.get("package", "unknown")
                                issues.append(f"{pkg} install")
                        if r.hallucinated_function_usage_detected and r.hallucinated_function_calls:
                            for func in r.hallucinated_function_calls[:2]:  # Limit to first 2
                                pkg = func.get("package", "unknown")
                                func_name = func.get("function", "unknown")
                                issues.append(f"{pkg}.{func_name}()")
                    
                    if issues:
                        issues_str = ", ".join(issues[:3])  # Limit to 3 issues
                        if len(issues) > 3:
                            issues_str += f" (+{len(issues) - 3} more)"
                        reasoning_parts.append(f"Issues: {issues_str}")
                
                reasoning = ". ".join(reasoning_parts) + "."
            
            # Convert turn_evaluations to claim_evaluations format for report compatibility
            claim_evaluations = []
            for turn_result in conv_results:
                turn_dict = turn_result.to_dict()
                turn_number = turn_dict.get("turn_number", 0)
                
                # Expand hallucinated items into individual claim evaluations
                # Each hallucinated item becomes a separate claim
                has_any_hallucination = turn_result.has_hallucination
                
                if has_any_hallucination:
                    # Create claim evaluation for each hallucinated import
                    for imp in turn_result.hallucinated_imports:
                        claim_evaluations.append({
                            "claim_id": f"turn-{conv_id}-{turn_number}-import-{imp.get('package', 'unknown')}",
                            "conversation_id": conv_id,
                            "turn_idx": turn_number,
                            "turn_number": turn_number,
                            "hallucination": "Yes",
                            "hallucinated_import_detected": True,
                            "hallucinated_install_detected": False,
                            "hallucinated_function_usage_detected": False,
                            "claim": {
                                "element_type": "import",
                                "package_name": imp.get("package", "unknown"),
                                "code_snippet": imp.get("code", ""),
                            },
                            "reason": imp.get("reason", turn_dict.get("reasoning", "")),
                            "reasoning": turn_dict.get("reasoning", ""),
                        })
                    
                    # Create claim evaluation for each hallucinated install
                    for inst in turn_result.hallucinated_installs:
                        claim_evaluations.append({
                            "claim_id": f"turn-{conv_id}-{turn_number}-install-{inst.get('package', 'unknown')}",
                            "conversation_id": conv_id,
                            "turn_idx": turn_number,
                            "turn_number": turn_number,
                            "hallucination": "Yes",
                            "hallucinated_import_detected": False,
                            "hallucinated_install_detected": True,
                            "hallucinated_function_usage_detected": False,
                            "claim": {
                                "element_type": "install",
                                "package_name": inst.get("package", "unknown"),
                                "code_snippet": inst.get("code", ""),
                            },
                            "reason": inst.get("reason", turn_dict.get("reasoning", "")),
                            "reasoning": turn_dict.get("reasoning", ""),
                        })
                    
                    # Create claim evaluation for each hallucinated function call
                    for func in turn_result.hallucinated_function_calls:
                        claim_evaluations.append({
                            "claim_id": f"turn-{conv_id}-{turn_number}-function-{func.get('package', 'unknown')}-{func.get('function', 'unknown')}",
                            "conversation_id": conv_id,
                            "turn_idx": turn_number,
                            "turn_number": turn_number,
                            "hallucination": "Yes",
                            "hallucinated_import_detected": False,
                            "hallucinated_install_detected": False,
                            "hallucinated_function_usage_detected": True,
                            "claim": {
                                "element_type": "function_call",
                                "package_name": func.get("package", "unknown"),
                                "function_name": func.get("function", "unknown"),
                                "code_snippet": func.get("code", ""),
                            },
                            "reason": func.get("reason", turn_dict.get("reasoning", "")),
                            "reasoning": turn_dict.get("reasoning", ""),
                        })
                else:
                    # For turns with no hallucinations, create a single "verified" claim entry
                    # This represents that the entire turn was verified
                    claim_evaluations.append({
                        "claim_id": f"turn-{conv_id}-{turn_number}-verified",
                        "conversation_id": conv_id,
                        "turn_idx": turn_number,
                        "turn_number": turn_number,
                        "hallucination": "No",
                        "hallucinated_import_detected": False,
                        "hallucinated_install_detected": False,
                        "hallucinated_function_usage_detected": False,
                        "claim": {
                            "element_type": "unknown",  # Will be inferred as non-hallucinated
                        },
                        "reason": turn_dict.get("reasoning", "All packages and API calls verified"),
                        "reasoning": turn_dict.get("reasoning", ""),
                    })
            
            # Calculate response-level stats
            total_responses = total_turns
            import_hallucinated_responses = sum(1 for r in conv_results if r.hallucinated_import_detected)
            install_hallucinated_responses = sum(1 for r in conv_results if r.hallucinated_install_detected)
            function_hallucinated_responses = sum(1 for r in conv_results if r.hallucinated_function_usage_detected)
            overall_hallucinated_responses = turns_with_hallucinations
            
            # Build details dict with both formats for compatibility
            details = {
                "total_turns": total_turns,
                "total_responses": total_responses,  # For report generator
                "total_claims": len(claim_evaluations),  # For report generator
                "import_hallucinations": import_hallucinations,
                "install_hallucinations": install_hallucinations,
                "function_hallucinations": function_hallucinations,
                "import_hallucinated_responses": import_hallucinated_responses,  # For report generator
                "install_hallucinated_responses": install_hallucinated_responses,  # For report generator
                "function_hallucinated_responses": function_hallucinated_responses,  # For report generator
                "overall_hallucinated_responses": overall_hallucinated_responses,  # For report generator
                # Conversation-level flags (for report generator)
                "hallucinated_import_detected": import_hallucinations > 0,
                "hallucinated_install_detected": install_hallucinations > 0,
                "hallucinated_function_usage_detected": function_hallucinations > 0,
                # Keep turn_evaluations for backward compatibility
                "turn_evaluations": [r.to_dict() for r in conv_results],
                # Add claim_evaluations for report generator compatibility
                "claim_evaluations": claim_evaluations,
            }
            
            final_results.append(EvaluationResult(
                conversation_id=conv_id,
                score=score,
                reasoning=reasoning,
                details=details,
                metadata=meta,
            ))
        
        return final_results
    
    def _save_results(self, results: List[EvaluationResult]) -> Path:
        """Save evaluation results to file."""
        output_path = self._get_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to: {output_path}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                record = {
                    "_type": "evaluation_result",
                    **asdict(result),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"✓ Saved {len(results)} evaluation results")
        return output_path
    
    async def run(self) -> Path:
        """Run the direct coding pipeline."""
        from .workers import DirectCodingJudgeWorker, TurnItem, DirectCodingResult
        
        # Validate input
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Log configuration
        logger.info("=" * 70)
        logger.info(f"CODING DIRECT PIPELINE (OpenAI Websearch)")
        logger.info(f"Task: {self.task_name}")
        logger.info("=" * 70)
        logger.info(f"Input: {self.input_path}")
        logger.info(f"Model: gpt-5-mini-medium-websearch")
        logger.info(f"Workers: {self.worker_config.num_judges} judges")
        logger.info("=" * 70 + "\n")
        
        # Load data
        self._load_conversations()
        
        # Extract assistant turns
        turns_data = self._extract_assistant_turns()
        logger.info(f"Extracted {len(turns_data)} assistant turns to evaluate")
        
        if not turns_data:
            logger.warning("No assistant turns found to evaluate")
            return self._save_results([])
        
        # Create sampler
        sampler = get_sampler("gpt-5-mini-medium-websearch")
        
        # Create queues
        input_queue: MonitoredQueue[TurnItem] = MonitoredQueue("turns")
        results_queue: MonitoredQueue[DirectCodingResult] = MonitoredQueue("results")
        
        # Create worker
        judge = DirectCodingJudgeWorker(
            input_queue=input_queue,
            output_queue=results_queue,
            sampler=sampler,
            num_workers=self.worker_config.num_judges,
        )
        
        # Create pipeline
        # Note: Don't add results_queue to queues list - it would cause join() deadlock
        # Results are collected after pipeline completes, not consumed during run
        pipeline = Pipeline(name=self.name)
        pipeline.add_queue(input_queue)
        pipeline.add_worker(judge)
        pipeline.set_results_queue(results_queue)
        
        # Load turns into queue
        for conv_id, turn_num, content in turns_data:
            await input_queue.put(
                TurnItem(
                    conversation_id=conv_id,
                    turn_number=turn_num,
                    content=content,
                ),
                claim_id=f"turn-{conv_id}-{turn_num}",
                conversation_id=conv_id,
            )
        input_queue.close()
        
        # Run pipeline (handles starting workers, monitoring, and completion)
        logger.info(f"Starting pipeline with {len(turns_data)} turns...")
        
        try:
            raw_results = await pipeline.run(
                input_queue=input_queue,
                total_items=len(turns_data),
                monitor_interval=self.monitor_interval,
            )
        except KeyboardInterrupt:
            logger.warning("\nInterrupted by user")
            raw_results = []
        
        # Convert raw results to DirectCodingResult
        all_results: List[DirectCodingResult] = []
        for result in raw_results:
            if isinstance(result, DirectCodingResult):
                all_results.append(result)
            elif hasattr(result, 'item'):
                all_results.append(result.item)
        
        logger.info(f"\n✓ Collected {len(all_results)} results")
        
        # Build evaluation results
        eval_results = self._build_evaluation_results(all_results)
        
        # Save
        output_path = self._save_results(eval_results)
        
        # Log summary
        total_import_h = sum(r.details.get("import_hallucinations", 0) for r in eval_results)
        total_install_h = sum(r.details.get("install_hallucinations", 0) for r in eval_results)
        total_function_h = sum(r.details.get("function_hallucinations", 0) for r in eval_results)
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Conversations: {len(eval_results)}")
        logger.info(f"Turns evaluated: {len(all_results)}")
        logger.info(f"Import hallucinations: {total_import_h}")
        logger.info(f"Install hallucinations: {total_install_h}")
        logger.info(f"Function call hallucinations: {total_function_h}")
        logger.info(f"Results: {output_path}")
        logger.info("=" * 70)
        
        return output_path


# =============================================================================
# Factory Function
# =============================================================================

PIPELINE_CLASSES = {
    "openai": OpenAIPipeline,
    "serper": SerperPipeline,
    "webscraper": WebscraperPipeline,
    "coding_direct": CodingDirectPipeline,
}


def create_pipeline(
    judging_type: Literal["openai", "serper", "webscraper", "coding_direct"],
    input_path: str | Path,
    task_name: str,
    output_path: str | Path | None = None,
    worker_config: WorkerConfig | None = None,
    n_conversations: int | None = None,
    max_claims_per_turn: int | None = None,
    seed: int = 42,
    monitor_interval: float = 3.0,
    base_path: str | Path | None = None,
    claims_cache_path: str | Path | None = None,
    checkpoint_interval: int = 100,
    max_claims_per_category: int | None = None,
) -> BasePipeline:
    """Factory function to create the appropriate pipeline."""

    base_path = Path(base_path) if base_path else Path(__file__).parent.parent

    if judging_type not in PIPELINE_CLASSES:
        raise ValueError(f"Unknown judging type: {judging_type}. Must be one of {list(PIPELINE_CLASSES.keys())}")
    
    pipeline_class = PIPELINE_CLASSES[judging_type]
    
    return pipeline_class(
        input_path=Path(input_path),
        output_path=Path(output_path) if output_path else None,
        worker_config=worker_config or WorkerConfig(),
        task_name=task_name,
        n_conversations=n_conversations,
        max_claims_per_turn=max_claims_per_turn,
        seed=seed,
        monitor_interval=monitor_interval,
        base_path=base_path,
        claims_cache_path=Path(claims_cache_path) if claims_cache_path else None,
        checkpoint_interval=checkpoint_interval,
        max_claims_per_category=max_claims_per_category,
    )


async def run_evaluation_pipeline(
    input_path: str | Path,
    judging_type: Literal["openai", "serper", "webscraper", "coding_direct"],
    task_name: str,
    output_path: str | Path | None = None,
    num_extractors: int = 5,
    num_searchers: int = 10,
    num_fetchers: int = 20,
    num_pdf_converters: int = 3,
    num_filters: int = 10,
    num_judges: int = 20,
    n_conversations: int | None = None,
    max_claims_per_turn: int | None = None,
    seed: int = 42,
    monitor_interval: float = 3.0,
    base_path: str | Path | None = None,
    claims_cache_path: str | Path | None = None,
    checkpoint_interval: int = 100,
    max_claims_per_category: int | None = None,
) -> Path:
    """Convenience function to create and run a pipeline."""
    worker_config = WorkerConfig(
        num_extractors=num_extractors,
        num_searchers=num_searchers,
        num_fetchers=num_fetchers,
        num_pdf_converters=num_pdf_converters,
        num_filters=num_filters,
        num_judges=num_judges,
    )
    
    pipeline = create_pipeline(
        judging_type=judging_type,
        input_path=input_path,
        task_name=task_name,
        output_path=output_path,
        worker_config=worker_config,
        n_conversations=n_conversations,
        max_claims_per_turn=max_claims_per_turn,
        seed=seed,
        monitor_interval=monitor_interval,
        base_path=base_path,
        claims_cache_path=claims_cache_path,
        checkpoint_interval=checkpoint_interval,
        max_claims_per_category=max_claims_per_category,
    )
    
    return await pipeline.run()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run queue-based evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to conversations JSONL file",
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        choices=["openai", "serper", "webscraper", "coding_direct"],
        default="webscraper",
        help="Pipeline type. 'coding_direct' uses OpenAI websearch to judge entire turns without claim extraction.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["research_questions", "medical_guidelines", "legal_cases", "coding"],
        help="Task domain (determines prompts and logic)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--n_conversations", "-n",
        type=int,
        default=None,
        help="Number of conversations to process. For coding tasks: first N per language. For others: first N total.",
    )
    parser.add_argument(
        "--max_claims_per_turn",
        type=int,
        default=None,
        help="Maximum claims per turn to evaluate",
    )
    parser.add_argument(
        "--max-claims-per-category",
        type=int,
        default=0,
        help="For coding tasks: max claims per category (import/install/function). Default 0 (disabled). Use early stopping instead.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Base path for loading system prompts",
    )
    parser.add_argument(
        "--claims-cache",
        type=str,
        default=None,
        help="Path to claims cache file. If not provided, looks for <input>_extracted_claims_cache.jsonl",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N judgments (default: 100). Set to 0 to disable checkpointing.",
    )
    
    # Worker counts
    parser.add_argument("--extractors", type=int, default=20, help="Claim extractor workers")
    parser.add_argument("--searchers", type=int, default=100, help="Web search workers")
    parser.add_argument("--fetchers", type=int, default=50, help="Web fetch workers")
    parser.add_argument("--pdf-converters", type=int, default=10, help="PDF converter workers")
    parser.add_argument("--filters", type=int, default=100, help="Content filter workers")
    parser.add_argument("--judges", type=int, default=200, help="Judge workers")
    
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=3.0,
        help="Seconds between progress updates",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(level=log_level)
    
    # Fix Windows asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Handle max_claims_per_category: 0 means disabled (None)
    max_per_cat = args.max_claims_per_category if args.max_claims_per_category > 0 else None
    
    asyncio.run(
        run_evaluation_pipeline(
            input_path=args.input,
            judging_type=args.type,
            task_name=args.task,
            output_path=args.output,
            num_extractors=args.extractors,
            num_searchers=args.searchers,
            num_fetchers=args.fetchers,
            num_pdf_converters=args.pdf_converters,
            num_filters=args.filters,
            num_judges=args.judges,
            n_conversations=args.n_conversations,
            max_claims_per_turn=args.max_claims_per_turn,
            seed=args.seed,
            monitor_interval=args.monitor_interval,
            base_path=args.base_path,
            claims_cache_path=args.claims_cache,
            checkpoint_interval=args.checkpoint_interval,
            max_claims_per_category=max_per_cat,
        )
    )
