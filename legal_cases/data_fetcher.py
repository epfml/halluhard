"""Legal cases data fetcher - fetches legal case questions from multiple sources.

This module:
1. Fetches legal case questions from reglab/legal_rag_hallucinations dataset
2. Fetches legal case questions from local JSONL file (California Bar Exam)
3. Filters by question categories
4. Saves questions to JSONL format for inference
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import argparse

from datasets import load_dataset


@dataclass
class LegalCaseTemplate:
    """Template for legal case conversations with metadata."""

    question: str
    question_category: str
    
    # Optional metadata
    answer: str | None = None  # Ground truth response
    source: str | None = None  # Source model
    case_id: str | None = None  # Question ID
    correctness: str | None = None  # Correctness label from dataset
    groundedness: str | None = None  # Groundedness label from dataset
    label: str | None = None  # Label from dataset

    def to_metadata(self) -> Dict[str, Any]:
        """Convert template to metadata dict for storage."""
        return {
            "question": self.question,
            "question_category": self.question_category,
            "source": self.source,
            "case_id": self.case_id,
        }


def fetch_legal_cases_from_hf(
    categories: list[str] | None = None,
    n: int | None = None,
    split: str = "train",
) -> list[Dict[str, Any]]:
    """Fetch legal case questions from Hugging Face dataset.

    Args:
        categories: List of question categories to filter by (None = all categories)
        n: Maximum number of cases to fetch (None = fetch all)
        split: Dataset split to use (default: "train")

    Returns:
        List of legal case dictionaries
    """
    print(f"Loading legal cases from reglab/legal_rag_hallucinations (split: {split})...")
    ds = load_dataset("reglab/legal_rag_hallucinations")
    
    # Get the specified split
    if split not in ds:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")
    
    dataset = ds[split]
    
    # Filter by categories if specified
    if categories:
        print(f"Filtering by categories: {categories}")
        dataset = dataset.filter(lambda x: x['Question Category'] in categories)
    
    # Convert to list and limit if needed
    cases = []
    for item in dataset:
        case_data = {
            "question": item.get("Question", ""),
            "question_category": item.get("Question Category", "Unknown"),
            "source": item.get("Model"),  # Source model (e.g., "Westlaw")
            "case_id": item.get("Question ID"),  # Unique question identifier
        }
        cases.append(case_data)
        
        # Check if we've collected enough cases
        if n is not None and len(cases) >= n:
            break
    
    print(f"[OK] Fetched {len(cases)} legal case questions from Hugging Face")
    if categories:
        print(f"  Categories: {categories}")
    
    return cases


def fetch_legal_cases_from_local(
    local_path: Path,
    categories: list[str] | None = None,
    n: int | None = None,
) -> list[Dict[str, Any]]:
    """Fetch legal case questions from local JSONL file.

    Args:
        local_path: Path to the local JSONL file
        categories: List of question categories to filter by (None = all categories)
        n: Maximum number of cases to fetch (None = fetch all)

    Returns:
        List of legal case dictionaries
    """
    print(f"Loading legal cases from local file: {local_path}...")
    
    if not local_path.exists():
        print(f"[WARNING] Local file not found: {local_path}")
        return []
    
    cases = []
    with open(local_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                # breakpoint()
                # Filter by categories if specified
                if categories and item.get("question_category") not in categories:
                    continue
                
                # Use the data as-is since it's already in the correct format
                case_data = {
                    "question": item.get("question", ""),
                    "question_category": item.get("question_category", "Unknown"),
                    "source": item.get("source"),
                    "case_id": item.get("case_id"),
                }
                cases.append(case_data)
                
                # Check if we've collected enough cases
                if n is not None and len(cases) >= n:
                    break
                    
            except json.JSONDecodeError as e:
                print(f"[WARNING] Failed to parse line {idx + 1}: {e}")
                continue
    
    print(f"[OK] Fetched {len(cases)} legal case questions from local file")
    if categories:
        print(f"  Categories: {categories}")
    
    return cases


def fetch_legal_cases(
    categories: list[str] | None = None,
    n: int | None = None,
    split: str = "train",
    include_local: bool = True,
    local_path: Path | None = None,
) -> list[Dict[str, Any]]:
    """Fetch legal case questions from both Hugging Face and local sources.

    Args:
        categories: List of question categories to filter by (None = all categories)
        n: Maximum number of cases to fetch per source (None = fetch all)
        split: Dataset split to use for Hugging Face (default: "train")
        include_local: Whether to include local JSONL file (default: True)
        local_path: Path to local JSONL file (default: ca_bar_50_cleaned.jsonl)

    Returns:
        Combined list of legal case dictionaries from both sources
    """
    all_cases = []
    
    # Fetch from Hugging Face
    hf_cases = fetch_legal_cases_from_hf(categories=categories, n=n, split=split)
    all_cases.extend(hf_cases)
    
    # Fetch from local file if enabled
    if include_local:
        if local_path is None:
            # Default path to the ca_bar_50_cleaned.jsonl
            local_path = Path(__file__).parent / "data" / "california_bar_practice_questions.jsonl"
        
        local_cases = fetch_legal_cases_from_local(
            local_path=local_path,
            categories=categories,
            n=n,
        )
        all_cases.extend(local_cases)
    
    print(f"\n[TOTAL] Combined {len(all_cases)} legal case questions from all sources")
    
    return all_cases


def main_pipeline(
    categories: list[str] | None,
    n: int | None,
    split: str,
    output_path: Path,
    include_local: bool = True,
    local_path: Path | None = None,
) -> None:
    """Main pipeline to fetch and save legal case questions.

    Args:
        categories: List of question categories to filter by
        n: Maximum number of cases to fetch per source
        split: Dataset split to use for Hugging Face
        output_path: Output file path
        include_local: Whether to include local JSONL file
        local_path: Path to local JSONL file
    """
    # Fetch legal cases from both sources
    cases = fetch_legal_cases(
        categories=categories,
        n=n,
        split=split,
        include_local=include_local,
        local_path=local_path,
    )
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    
    print(f"\n[OK] Saved {len(cases)} legal case questions to: {output_path}")


if __name__ == "__main__":
    # CLI script to fetch legal case questions
    parser = argparse.ArgumentParser(
        description="Fetch legal case questions from multiple sources (Hugging Face + local JSONL)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Question categories to filter by (e.g., 'Bar Exam' 'Civil Procedure'). If not specified, fetches all categories.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Maximum number of cases to fetch per source (default: fetch all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for Hugging Face (default: train)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (jsonl format)",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Disable fetching from local JSONL file (only fetch from Hugging Face)",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to local JSONL file (default: data/ca_bar_50_cleaned.jsonl)",
    )

    args = parser.parse_args()

    if args.output is None:
        # Create default output filename based on categories
        if args.categories:
            category_str = "_".join([cat.replace(" ", "_") for cat in args.categories])
            output = f"legal_cases/data/legal_cases_{category_str}.jsonl"
        else:
            output = "legal_cases/data/legal_cases_all.jsonl"
    else:
        output = args.output

    output_file = Path(output)
    
    # Parse local path if provided
    local_path = Path(args.local_path) if args.local_path else None

    # Run the main pipeline
    main_pipeline(
        categories=args.categories,
        n=args.n,
        split=args.split,
        output_path=output_file,
        include_local=not args.no_local,
        local_path=local_path,
    )

