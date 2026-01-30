"""Evaluation runner that orchestrates extract and evaluate steps."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

from libs.evaluator import Evaluator, EvaluationResult, Extractor, ExtractionResult
from libs.schemas import Conversation
from libs.storage import load_conversations

from tqdm import tqdm


class EvaluationRunner:
    """Orchestrates the evaluation pipeline: load → extract → evaluate → save."""

    def __init__(
        self,
        evaluator: Evaluator,
        extractor: Extractor | None = None,
    ):
        """Initialize evaluation runner.

        Args:
            evaluator: Evaluator instance (required)
            extractor: Optional extractor instance (None = direct evaluation)

        Note: Semaphores should be passed when creating the evaluator/extractor instances.
        """
        self.evaluator = evaluator
        self.extractor = extractor

    def _get_extraction_cache_path(self, input_path: Path) -> Path:
        """Get the path for extraction cache file."""
        return input_path.parent / f"{input_path.stem}_extraction_cache.jsonl"

    def _get_evaluation_cache_path(self, input_path: Path) -> Path:
        """Get the path for evaluation cache file."""
        return input_path.parent / f"{input_path.stem}_evaluation_cache.jsonl"

    def _load_extraction_cache(self, cache_path: Path) -> List[ExtractionResult]:
        """Load cached extraction results from file.

        Handles corrupted/partial lines gracefully by stopping at the first
        invalid line (which may have been partially written during interruption).

        Args:
            cache_path: Path to the cache file

        Returns:
            List of ExtractionResult objects
        """
        results = []
        if not cache_path.exists():
            return results

        with open(cache_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Remove _type field if present
                    record.pop("_type", None)
                    results.append(ExtractionResult(**record))
                except json.JSONDecodeError as e:
                    # Stop at corrupted line (likely partial write from interruption)
                    print(f"  Warning: Corrupted cache entry at line {line_num}, "
                          f"using {len(results)} valid entries. Error: {e}")
                    break

        # Rewrite cache with only valid entries to fix corruption
        if results:
            self._rewrite_extraction_cache(results, cache_path)

        return results

    def _rewrite_extraction_cache(self, results: List[ExtractionResult], cache_path: Path):
        """Rewrite extraction cache with valid entries only."""
        with open(cache_path, "w", encoding="utf-8") as f:
            for result in results:
                record = {
                    "_type": "extraction_result",
                    **asdict(result),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_evaluation_cache(self, cache_path: Path) -> List[EvaluationResult]:
        """Load cached evaluation results from file.

        Handles corrupted/partial lines gracefully by stopping at the first
        invalid line (which may have been partially written during interruption).

        Args:
            cache_path: Path to the cache file

        Returns:
            List of EvaluationResult objects
        """
        results = []
        if not cache_path.exists():
            return results

        with open(cache_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Remove _type field if present
                    record.pop("_type", None)
                    results.append(EvaluationResult(**record))
                except json.JSONDecodeError as e:
                    # Stop at corrupted line (likely partial write from interruption)
                    print(f"  Warning: Corrupted cache entry at line {line_num}, "
                          f"using {len(results)} valid entries. Error: {e}")
                    break

        # Rewrite cache with only valid entries to fix corruption
        if results:
            self._rewrite_evaluation_cache(results, cache_path)

        return results

    def _rewrite_evaluation_cache(self, results: List[EvaluationResult], cache_path: Path):
        """Rewrite evaluation cache with valid entries only."""
        with open(cache_path, "w", encoding="utf-8") as f:
            for result in results:
                record = {
                    "_type": "evaluation_result",
                    **asdict(result),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_extraction_to_cache(
        self,
        result: ExtractionResult,
        cache_path: Path,
    ):
        """Append a single extraction result to cache file.

        Args:
            result: Extraction result to append
            cache_path: Path to the cache file
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "a", encoding="utf-8") as f:
            record = {
                "_type": "extraction_result",
                **asdict(result),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_evaluation_to_cache(
        self,
        result: EvaluationResult,
        cache_path: Path,
    ):
        """Append a single evaluation result to cache file.

        Args:
            result: Evaluation result to append
            cache_path: Path to the cache file
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "a", encoding="utf-8") as f:
            record = {
                "_type": "evaluation_result",
                **asdict(result),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _delete_cache_files(self, input_path: Path):
        """Delete cache files after successful completion.

        Args:
            input_path: Input path used to derive cache paths
        """
        extraction_cache = self._get_extraction_cache_path(input_path)
        evaluation_cache = self._get_evaluation_cache_path(input_path)

        if extraction_cache.exists():
            extraction_cache.unlink()
            print(f"✓ Cleaned up extraction cache: {extraction_cache}")

        if evaluation_cache.exists():
            evaluation_cache.unlink()
            print(f"✓ Cleaned up evaluation cache: {evaluation_cache}")

    async def _extract_with_checkpointing(
        self,
        conversations: List[Conversation],
        metadata_list: List[Dict[str, Any]],
        cache_path: Path,
    ) -> List[ExtractionResult]:
        """Extract claims with incremental checkpointing.

        Each result is saved to cache immediately after completion.
        If interrupted, all completed results are preserved.

        Args:
            conversations: List of conversations to process
            metadata_list: List of metadata dicts
            cache_path: Path to save intermediate results

        Returns:
            List of ExtractionResult objects (in order)
        """
        # Load existing cache
        cached_results = self._load_extraction_cache(cache_path)
        start_idx = len(cached_results)

        if start_idx > 0:
            print(f"  Resuming from cached results: {start_idx} items already processed")

        # Collect all assistant turns that need processing
        all_tasks_info = []  # (content, conv_id, turn_num, meta, original_idx)
        for conv, meta in zip(conversations, metadata_list):
            conversation_id = meta.get("conversation_id", 0)
            assistant_turn_number = 1
            for turn in conv.turns:
                if turn.role == "assistant":
                    all_tasks_info.append((
                        turn.content,
                        conversation_id,
                        assistant_turn_number,
                        meta,
                    ))
                    assistant_turn_number += 1

        total_items = len(all_tasks_info)
        remaining_items = all_tasks_info[start_idx:]

        if not remaining_items:
            print(f"  All {total_items} items already extracted from cache")
            return cached_results

        print(f"  Processing {len(remaining_items)} remaining items (of {total_items} total)")

        results = list(cached_results)

        # Create wrapper task that saves result immediately upon completion
        async def extract_and_save(idx: int, content: str, conv_id: int, turn_num: int, meta: Dict[str, Any]) -> Tuple[int, ExtractionResult]:
            result = await self.extractor._extract_turn(content, conv_id, turn_num, meta)
            return (idx, result)

        # Create all tasks with their original indices
        tasks = [
            extract_and_save(i, content, conv_id, turn_num, meta)
            for i, (content, conv_id, turn_num, meta) in enumerate(remaining_items)
        ]

        # Process results as they complete and save incrementally
        # We need to maintain order, so collect results in a dict first
        pending_results: Dict[int, ExtractionResult] = {}
        next_idx_to_save = 0  # Next index we need to save (to maintain order)

        with tqdm(total=len(remaining_items), desc="  Extracting") as pbar:
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                pending_results[idx] = result
                pbar.update(1)

                # Save all consecutive completed results starting from next_idx_to_save
                while next_idx_to_save in pending_results:
                    result_to_save = pending_results.pop(next_idx_to_save)
                    self._append_extraction_to_cache(result_to_save, cache_path)
                    results.append(result_to_save)
                    next_idx_to_save += 1

        print(f"  ✓ All {total_items} extractions complete and saved")
        return results

    async def _evaluate_with_checkpointing(
        self,
        conversations: List[Conversation],
        metadata_list: List[Dict[str, Any]],
        extraction_results: List[ExtractionResult] | None,
        cache_path: Path,
    ) -> List[EvaluationResult]:
        """Evaluate conversations with incremental checkpointing.

        Each result is saved to cache immediately after completion.
        If interrupted, all completed results are preserved.

        Args:
            conversations: List of conversations to evaluate
            metadata_list: List of metadata dicts
            extraction_results: Optional extraction results
            cache_path: Path to save intermediate results

        Returns:
            List of EvaluationResult objects (in order)
        """
        # Load existing cache
        cached_results = self._load_evaluation_cache(cache_path)
        start_idx = len(cached_results)

        if start_idx > 0:
            print(f"  Resuming from cached results: {start_idx} items already evaluated")

        total_items = len(conversations)

        # Get remaining items to process
        remaining_conversations = conversations[start_idx:]
        remaining_metadata = metadata_list[start_idx:]
        if extraction_results:
            remaining_extractions = extraction_results[start_idx:]
        else:
            remaining_extractions = [None] * len(remaining_conversations)

        if not remaining_conversations:
            print(f"  All {total_items} items already evaluated from cache")
            return cached_results

        print(f"  Processing {len(remaining_conversations)} remaining items (of {total_items} total)")

        results = list(cached_results)

        # Create wrapper task that returns result with its index
        async def evaluate_with_idx(
            idx: int,
            conv: Conversation,
            meta: Dict[str, Any],
            extraction: ExtractionResult | None,
        ) -> Tuple[int, EvaluationResult]:
            result = await self.evaluator.evaluate(conv, meta, extraction)
            return (idx, result)

        # Create all tasks with their original indices
        tasks = [
            evaluate_with_idx(i, conv, meta, extraction)
            for i, (conv, meta, extraction) in enumerate(
                zip(remaining_conversations, remaining_metadata, remaining_extractions)
            )
        ]

        # Process results as they complete and save incrementally
        # We need to maintain order, so collect results in a dict first
        pending_results: Dict[int, EvaluationResult] = {}
        next_idx_to_save = 0  # Next index we need to save (to maintain order)

        with tqdm(total=len(remaining_conversations), desc="  Evaluating") as pbar:
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                pending_results[idx] = result
                pbar.update(1)

                # Save all consecutive completed results starting from next_idx_to_save
                while next_idx_to_save in pending_results:
                    result_to_save = pending_results.pop(next_idx_to_save)
                    self._append_evaluation_to_cache(result_to_save, cache_path)
                    results.append(result_to_save)
                    next_idx_to_save += 1

        print(f"  ✓ All {total_items} evaluations complete and saved")
        return results

    async def run(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        cached_extractions: List[ExtractionResult] | None = None,
        resume: bool = True,
        cleanup_cache: bool = True,
    ) -> List[EvaluationResult]:
        """Run the full evaluation pipeline.

        Args:
            input_path: Path to conversations JSONL file
            output_path: Optional path to save results (defaults to input_path with _eval suffix)
            cached_extractions: Optional list of cached extraction results to skip extraction step
            resume: If True, resume from cached intermediate results (default: True)
            cleanup_cache: If True, delete cache files after successful completion (default: True)

        Returns:
            List of EvaluationResults
        """
        input_path = Path(input_path)

        # Load conversations and metadata
        print(f"Loading conversations from: {input_path}")
        conversations, metadata_list = load_conversations(input_path)
        print(f"Loaded {len(conversations)} conversations")

        # Get cache paths
        extraction_cache_path = self._get_extraction_cache_path(input_path)
        evaluation_cache_path = self._get_evaluation_cache_path(input_path)

        # Step 1: Extract (if extractor provided and no cache)
        extraction_results = None
        if cached_extractions:
            print("\nUsing provided cached extraction results")
            extraction_results = cached_extractions
            if len(extraction_results) != len(conversations):
                print(f"Warning: Cached extractions count ({len(extraction_results)}) "
                      f"does not match conversations count ({len(conversations)})")
        elif self.extractor:
            print(f"\nExtracting claims from {len(conversations)} conversations...")
            if resume:
                extraction_results = await self._extract_with_checkpointing(
                    conversations, metadata_list, extraction_cache_path
                )
            else:
                extraction_results = await self.extractor.extract_batch(
                    conversations, metadata_list
                )
            print(f"✓ Extraction complete")

            # Save extractions for future use (final version)
            self._save_extractions(extraction_results, input_path)
        else:
            print("\nSkipping extraction (evaluating directly)")

        # Step 2: Evaluate
        print(f"\nEvaluating {len(conversations)} conversations...")
        if resume:
            eval_results = await self._evaluate_with_checkpointing(
                conversations, metadata_list, extraction_results, evaluation_cache_path
            )
        else:
            eval_results = await self.evaluator.evaluate_batch(
                conversations, metadata_list, extraction_results
            )
        print(f"✓ Evaluation complete")

        # Step 3: Save results
        if output_path is None:
            # Default: same path with _eval suffix
            output_path = (
                input_path.parent / f"{input_path.stem}_eval{input_path.suffix}"
            )

        self._save_results(eval_results, output_path)
        print(f"\n✓ Results saved to: {output_path}")
        print(f"Analyzed {len(eval_results)} conversations")
        print("\nNote: Aggregate metrics are task-specific.")
        print("Use task-specific analysis tools to compute metrics from the results.")

        # Cleanup cache files after successful completion
        if cleanup_cache:
            self._delete_cache_files(input_path)

        return eval_results

    def _save_extractions(
        self,
        extractions: List[ExtractionResult],
        input_path: Path,
    ):
        """Save extraction results to JSONL file for caching.

        Args:
            extractions: List of extraction results
            input_path: Original input path (used to generate extraction cache path)
        """
        input_path = Path(input_path)
        extraction_path = input_path.parent / f"{input_path.stem}_extractions{input_path.suffix}"
        
        extraction_path.parent.mkdir(parents=True, exist_ok=True)

        with open(extraction_path, "w", encoding="utf-8") as f:
            for extraction in extractions:
                record = {
                    "_type": "extraction_result",
                    **asdict(extraction),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"✓ Extractions saved to: {extraction_path}")

    def _save_results(
        self,
        results: List[EvaluationResult],
        output_path: Path,
    ):
        """Save evaluation results to JSONL file.

        Args:
            results: List of evaluation results
            output_path: Path to save results

        Note: No aggregate metrics are saved. Task-specific analysis tools
        should compute metrics from the individual results.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # Write individual results (one per line)
            for result in results:
                record = {
                    "_type": "evaluation_result",
                    **asdict(result),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
