"""Medical guidelines data fetcher - parses guidelines and generates questions.

This module:
1. Fetches guidelines from Hugging Face dataset (epfl-llm/guidelines)
2. Uses an LLM to generate exam questions from guidelines
3. Stores source metadata for each guideline
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import argparse

from tqdm.asyncio import tqdm
from datasets import load_dataset

from libs.sampler.openai_sampler import ResponsesSampler


@dataclass
class MedicalGuidelineTemplate:
    """Template for medical guideline conversations with metadata."""

    guideline_text: str
    question: str
    source: str  # Original source of the guideline

    def to_metadata(self) -> Dict[str, Any]:
        """Convert template to metadata dict for storage."""
        return {
            "guideline_text": self.guideline_text,
            "question": self.question,
            "source": self.source,
        }


def load_guidelines_from_huggingface(
    dataset_name: str = "epfl-llm/guidelines",
    n: int | None = None,
) -> list[Dict[str, Any]]:
    """Load guidelines from Hugging Face dataset with uniform sampling across sources.

    Args:
        dataset_name: Name of the Hugging Face dataset
        n: Maximum number of guidelines to load (None = all). If specified, 
           samples uniformly across different sources.

    Returns:
        List of guideline dictionaries with text and source
    """
    print(f"Loading guidelines from Hugging Face dataset: {dataset_name}")
    
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name, split="train")
    
    print(f"Loaded {len(dataset)} guidelines from dataset")
    
    # Group guidelines by source
    guidelines_by_source = {}
    
    for item in dataset:
        # Get clean_text and source from the dataset
        guideline_text = item.get("clean_text", "")
        source = item.get("source", "unknown")
        
        # Skip if guideline text is too short
        if len(guideline_text.strip()) < 50:
            continue
        
        if source not in guidelines_by_source:
            guidelines_by_source[source] = []
        
        guidelines_by_source[source].append({
            "guideline_text": guideline_text.strip(),
            "source": source,
        })
    
    # Print source distribution
    print(f"\nSource distribution:")
    for source, items in sorted(guidelines_by_source.items()):
        print(f"  {source}: {len(items)} guidelines")
    
    # If n is specified, sample uniformly across sources
    if n is not None and n < sum(len(items) for items in guidelines_by_source.values()):
        print(f"\nSampling {n} guidelines uniformly across {len(guidelines_by_source)} sources...")
        guidelines = _sample_uniformly_across_sources(guidelines_by_source, n)
    else:
        # Return all guidelines
        guidelines = []
        for items in guidelines_by_source.values():
            guidelines.extend(items)
    
    print(f"[OK] Loaded {len(guidelines)} guidelines")
    return guidelines


def _sample_uniformly_across_sources(
    guidelines_by_source: Dict[str, list[Dict[str, Any]]], 
    n: int
) -> list[Dict[str, Any]]:
    """Sample guidelines uniformly across sources.
    
    If a source has fewer items than its fair share, take all from that source
    and compensate by sampling more from other sources.
    
    Args:
        guidelines_by_source: Dictionary mapping source to list of guidelines
        n: Total number of guidelines to sample
        
    Returns:
        List of sampled guidelines
    """
    from collections import defaultdict
    
    num_sources = len(guidelines_by_source)
    per_source = n // num_sources
    
    # First pass: sample from each source (or take all if fewer)
    sampled = []
    shortfall = 0
    sources_with_capacity = []
    
    for source, items in guidelines_by_source.items():
        available = len(items)
        take = min(per_source, available)
        sampled.extend(random.sample(items, take) if take < available else items)
        
        if available < per_source:
            shortfall += per_source - available
        elif available > per_source:
            sources_with_capacity.append((source, available - per_source))
    
    # Redistribute shortfall + remainder from sources with capacity
    needed = shortfall + (n % num_sources)
    
    if needed > 0 and sources_with_capacity:
        # Create pool of available items from sources with capacity
        available_pool = []
        sampled_set = {id(item) for item in sampled}  # Track what's already sampled
        
        for source, capacity in sources_with_capacity:
            remaining = [item for item in guidelines_by_source[source] 
                        if id(item) not in sampled_set]
            available_pool.extend(remaining[:capacity])
        
        # Sample from the pool
        extra = min(needed, len(available_pool))
        sampled.extend(random.sample(available_pool, extra))
    
    # Shuffle to mix sources
    random.shuffle(sampled)
    
    # Print sampling result
    sampled_by_source = defaultdict(int)
    for item in sampled:
        sampled_by_source[item['source']] += 1
    
    print(f"\nSampled distribution:")
    for source in sorted(sampled_by_source.keys()):
        print(f"  {source}: {sampled_by_source[source]} guidelines")
    
    return sampled


async def generate_question_for_guideline(
    guideline: Dict[str, Any],
    sampler: ResponsesSampler,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any] | None:
    """Generate an exam question for a single guideline.

    Args:
        guideline: Guideline dict with 'guideline_text' and 'year_category'
        sampler: ResponsesSampler instance
        system_prompt: System prompt for question generation
        semaphore: Semaphore for rate limiting

    Returns:
        Guideline dict with added 'question' field, or None on error
    """
    async with semaphore:
        prompt_template = (
            "Given the following clinical recommendation from {source}, craft one high-quality, open-ended written exam question "
            "that can be answered accurately and comprehensively *according to authoritative guidelines*, even though the examinee "
            "will not have access to the guideline text. The question should:\n"
            "- Be fully grounded in the provided recommendation (no external knowledge required)\n"
            "- Be phrased clearly and formally, as in a written clinical exam\n"
            "- Keep it concise and to the point\n"
            "- Require structured, discursive reasoning rather than simple recall\n\n"
            "Recommendation from {source} for inspiration:\n{recommendation}\n\n"
            "Exam Question:"
        )
        
        prompt = prompt_template.format(
            source=guideline['source'],
            recommendation=guideline['guideline_text']
        )

        try:
            message_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = await sampler(message_list)
            guideline["question"] = response.response_text.strip()
            return guideline
        except Exception as e:
            print(f"✗ Error generating question for guideline: {e}")
            print(f"  Guideline preview: {guideline['guideline_text'][:100]}...")
            return None


async def generate_questions_for_guidelines(
    guidelines: list[Dict[str, Any]],
    model: str,
    system_prompt: str,
    max_concurrent: int = 5,
) -> list[Dict[str, Any]]:
    """Generate exam questions for guidelines using LLM.

    Args:
        guidelines: List of guideline dicts with 'guideline_text' and 'year_category'
        model: Model name for question generation
        system_prompt: System prompt for question generation
        max_concurrent: Max concurrent API calls

    Returns:
        List of guidelines with added 'question' field
    """
    print(f"Generating exam questions using {model}...")

    # Create sampler for question generation
    sampler = ResponsesSampler(
        model=model,
        reasoning_effort="low",
    )

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        generate_question_for_guideline(guideline, sampler, system_prompt, semaphore)
        for guideline in guidelines
    ]
    results = await tqdm.gather(*tasks, desc="Generating questions")

    # Filter out failed generations
    guidelines_with_questions = [r for r in results if r is not None]
    print(f"\n[OK] Generated {len(guidelines_with_questions)} questions\n")
    return guidelines_with_questions


async def main_pipeline(
    dataset_name: str,
    n: int | None,
    model: str,
    question_system_prompt: str,
    output_path: Path,
    max_concurrent: int,
) -> None:
    """Main pipeline to load guidelines and generate questions.

    Args:
        dataset_name: Name of the Hugging Face dataset
        n: Maximum number of guidelines to process (None = all). 
           If specified, samples uniformly across different sources.
        model: Model name for question generation
        question_system_prompt: System prompt for question generation
        output_path: Output file path
        max_concurrent: Max concurrent API calls
    """
    target_count = n
    
    # Load all guidelines upfront for retry pool
    all_guidelines = load_guidelines_from_huggingface(dataset_name, n=None)
    random.shuffle(all_guidelines)
    
    # Track used guidelines and successful results
    used_indices = set()
    guidelines_with_questions = []
    
    # Initial batch
    batch_size = target_count if target_count else len(all_guidelines)
    
    while True:
        # How many more do we need?
        needed = (target_count - len(guidelines_with_questions)) if target_count else (len(all_guidelines) - len(used_indices))
        if needed <= 0:
            break
        
        # Get next batch of unused guidelines
        batch = []
        for i, g in enumerate(all_guidelines):
            if i not in used_indices:
                batch.append((i, g))
                if len(batch) >= needed:
                    break
        
        if not batch:
            print(f"[!] Ran out of guidelines. Got {len(guidelines_with_questions)}/{target_count}")
            break
        
        # Mark as used
        for i, _ in batch:
            used_indices.add(i)
        
        guidelines_batch = [g for _, g in batch]
        print(f"\nProcessing batch of {len(guidelines_batch)} guidelines (need {needed} more)...")
        
        # Generate questions for this batch
        results = await generate_questions_for_guidelines(
            guidelines_batch, model, question_system_prompt, max_concurrent
        )
        guidelines_with_questions.extend(results)
        
        # Trim to target if we got more than needed
        if target_count and len(guidelines_with_questions) >= target_count:
            guidelines_with_questions = guidelines_with_questions[:target_count]
            break

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for guideline in guidelines_with_questions:
            f.write(json.dumps(guideline, ensure_ascii=False) + "\n")

    print(f"[OK] Saved {len(guidelines_with_questions)} guidelines with questions to: {output_path}")


if __name__ == "__main__":
    # CLI script to load guidelines from HuggingFace and generate questions
    parser = argparse.ArgumentParser(
        description="Load NICE guidelines from HuggingFace and generate exam questions"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="epfl-llm/guidelines",
        help="Name of the Hugging Face dataset (default: epfl-llm/guidelines)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Maximum number of guidelines to process (default: None = all). When specified, samples uniformly across different sources.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model for question generation (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--question-prompt",
        type=str,
        default="question-creator.txt",
        help="System prompt file for question generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="medical_guidelines/data/guidelines.jsonl",
        help="Output file path (jsonl format)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5, help="Max concurrent API calls"
    )

    args = parser.parse_args()

    output_file = Path(args.output)

    # Load question generation system prompt
    prompt_path = Path(__file__).parent / "prompts" / args.question_prompt
    if not prompt_path.exists():
        raise FileNotFoundError(f"Question prompt not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # Run the main pipeline
    asyncio.run(
        main_pipeline(
            dataset_name=args.dataset_name,
            n=args.n,
            model=args.model,
            question_system_prompt=system_prompt,
            output_path=output_file,
            max_concurrent=args.max_concurrent,
        )
    )
