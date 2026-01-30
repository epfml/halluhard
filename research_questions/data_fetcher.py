"""Research questions data fetcher - fetches papers and generates research questions.

This module:
1. Fetches papers from OpenAlex
2. Retrieves abstracts from arXiv
3. Uses an LLM to generate challenging research questions from abstracts
"""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import argparse

import aiohttp
import requests

from tqdm.asyncio import tqdm

from libs.sampler.openai_sampler import ResponsesSampler
from libs.openalex_fetcher import get_random_works


# arXiv API endpoints
ARXIV_API = "https://export.arxiv.org/api/query?id_list={id}"
ARXIV_SEARCH = (
    "https://export.arxiv.org/api/query?search_query=ti:{title}&max_results=1"
)


def extract_arxiv_id(s: str) -> str | None:
    """Extract arXiv ID from various formats (DOI, URL, or raw ID)."""
    s = s.strip()
    # DOI form
    m = re.search(
        r"(?:https?://doi\.org/)?10\.48550/arXiv\.([A-Za-z\-]+/\d{7}|\d{4}\.\d{4,5})(v\d+)?",
        s,
    )
    if m:
        return m.group(1) + (m.group(2) or "")
    # URL form
    m = re.search(
        r"https?://arxiv\.org/(?:abs|pdf)/([A-Za-z\-]+/\d{7}|\d{4}\.\d{4,5})(v\d+)?", s
    )
    if m:
        return m.group(1) + (m.group(2) or "")
    # raw ID
    m = re.fullmatch(r"([A-Za-z\-]+/\d{7}|\d{4}\.\d{4,5})(v\d+)?", s)
    if m:
        return m.group(1) + (m.group(2) or "")
    return None


async def search_arxiv_by_title_async(
    title: str, timeout: int = 15
) -> Dict[str, Any] | None:
    """Search arXiv by paper title and return the first result with abstract."""
    url = ARXIV_SEARCH.format(title=requests.utils.quote(title))

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(url) as resp:
                text = await resp.text()

        root = ET.fromstring(text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)

        if entry is None:
            return None

        arxiv_id_elem = entry.find("a:id", ns)
        if arxiv_id_elem is None:
            return None

        arxiv_id = arxiv_id_elem.text.split("/")[-1]
        summary = entry.find("a:summary", ns)
        abstract = (summary.text or "").strip() if summary is not None else None

        return {"arxiv_id": arxiv_id, "abstract": abstract}
    except Exception as e:
        # print(f"Error fetching abstract for '{title}': {e}")
        return None


async def fetch_abstracts_for_papers(papers: list, max_concurrent: int = 10, verbose: bool = False) -> list:
    """Fetch abstracts from arXiv for a list of papers.

    Args:
        papers: List of paper dicts with 'title' key
        max_concurrent: Maximum concurrent arXiv requests (default: 10)
        verbose: If True, print debug info about failed lookups

    Returns:
        List of papers with added 'abstract' and 'arxiv_id' fields
    """
    print("Fetching abstracts from arXiv...")
    # Add rate limiting to avoid overwhelming arXiv API
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(title: str):
        async with semaphore:
            # Small delay to avoid overwhelming arXiv API
            await asyncio.sleep(0.1)
            return await search_arxiv_by_title_async(title)

    tasks = [fetch_with_limit(paper["title"]) for paper in papers]
    results = await tqdm.gather(*tasks)

    papers_with_abstracts = []
    no_match_count = 0
    no_abstract_count = 0
    failed_titles = []
    
    for paper, arxiv_data in zip(papers, results):
        if arxiv_data and arxiv_data.get("abstract"):
            paper["abstract"] = arxiv_data["abstract"]
            paper["arxiv_id"] = arxiv_data["arxiv_id"]
            papers_with_abstracts.append(paper)
        else:
            if arxiv_data is None:
                no_match_count += 1
            else:
                no_abstract_count += 1
            failed_titles.append(paper["title"])
    
    # Print debug summary
    if verbose or len(papers_with_abstracts) == 0:
        print(f"\n📊 arXiv lookup breakdown:")
        print(f"   ✓ Found with abstract: {len(papers_with_abstracts)}")
        print(f"   ✗ No arXiv match: {no_match_count}")
        print(f"   ✗ Match but no abstract: {no_abstract_count}")
        
        if failed_titles and (verbose or len(papers_with_abstracts) == 0):
            print(f"\n   Sample failed titles (first 5):")
            for title in failed_titles[:5]:
                print(f"     - {title[:80]}{'...' if len(title) > 80 else ''}")

    return papers_with_abstracts


async def generate_question_for_paper(
    paper: Dict[str, Any],
    sampler: Any,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any] | None:
    """Generate a research question for a single paper.

    Args:
        paper: Paper dict with 'abstract' key
        sampler: ResponsesSampler instance
        system_prompt: System prompt for question generation
        semaphore: Semaphore for rate limiting

    Returns:
        Paper dict with added 'research_question' field, or None on error
    """
    async with semaphore:
        prompt = f"Given the following abstract of a research paper, craft one clear and realistic question that an interested reader might naturally ask after reading it. Please only output the question.\n\nAbstract:\n{paper['abstract']}\n\nQuestion:"

        try:
            message_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = await sampler(message_list)
            paper["research_question"] = response.response_text.strip()
            return paper
        except Exception as e:
            print(f"✗ Error generating question for '{paper['title']}': {e}")
            return None


async def generate_questions_for_papers(
    papers_with_abstracts: list,
    model: str,
    system_prompt: str,
    max_concurrent: int = 5,
) -> list:
    """Generate research questions for papers using LLM.

    Args:
        papers_with_abstracts: List of paper dicts with 'abstract' key
        model: Model name for question generation
        system_prompt: System prompt for question generation
        max_concurrent: Max concurrent API calls

    Returns:
        List of papers with added 'research_question' field
    """
    print(f"Generating research questions using {model}...")

    # Create sampler for question generation
    sampler = ResponsesSampler(
        model=model,
        max_tokens=500,
        reasoning_effort="low",
    )

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        generate_question_for_paper(paper, sampler, system_prompt, semaphore)
        for paper in papers_with_abstracts
    ]
    results = await tqdm.gather(*tasks)

    # Filter out failed generations
    papers_with_questions = [r for r in results if r is not None]
    print(f"\n✓ Generated {len(papers_with_questions)} research questions\n")
    return papers_with_questions


@dataclass
class ResearchQuestionTemplate:
    """Template for research question conversations with metadata."""

    paper_title: str
    abstract: str
    research_question: str

    # Optional metadata
    doi: str | None = None
    arxiv_id: str | None = None
    publication_year: int | None = None
    cited_by_count: int | None = None
    language: str | None = None
    created_date: str | None = None

    def to_metadata(self) -> Dict[str, Any]:
        """Convert template to metadata dict for storage."""
        return {
            "paper_title": self.paper_title,
            "abstract": self.abstract,
            "research_question": self.research_question,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "publication_year": self.publication_year,
            "cited_by_count": self.cited_by_count,
            "language": self.language,
            "created_date": self.created_date,
        }


async def main_pipeline(
    n: int,
    seed: int,
    cited_by_min: int,
    cited_by_max: int,
    min_words_in_title: int,
    publication_year: int | None,
    model: str,
    question_system_prompt: str,
    output_path: Path,
    max_concurrent: int,
    verbose: bool = False,
    use_openalex_abstracts: bool = False,
) -> None:
    """Main pipeline to fetch papers, abstracts, and generate questions.

    Args:
        n: Number of papers to fetch
        seed: Random seed
        cited_by_min: Minimum citations
        cited_by_max: Maximum citations
        min_words_in_title: Minimum words in title
        publication_year: Optional publication year filter
        model: Model name for question generation
        question_system_prompt: System prompt for question generation
        output_path: Output file path
        max_concurrent: Max concurrent API calls
        use_openalex_abstracts: If True, use abstracts from OpenAlex instead of arXiv
    """
    # Step 1: Accumulate papers with abstracts until we have enough
    all_papers_with_abstracts = []
    try_nb = 0
    fetch_nb = n

    print(f"Target: {n} papers with abstracts and questions")

    while len(all_papers_with_abstracts) < n:
        # Fetch papers from OpenAlex
        print(f"\nTry #{try_nb + 1}: Fetching {fetch_nb} papers from OpenAlex...")
        papers = get_random_works(
            n=fetch_nb,
            seed=try_nb + (seed if try_nb == 0 else 0),  # Only use seed on first try
            cited_by_min=cited_by_min,
            cited_by_max=cited_by_max,
            publication_year=publication_year,
            min_words_in_title=min_words_in_title,
        )
        print(f"✓ Fetched {len(papers)} papers")

        if use_openalex_abstracts:
            # Use abstracts already provided by OpenAlex (reconstructed from inverted index)
            papers_with_abstracts = [
                p for p in papers
                if p.get("abstract") and len(p.get("abstract", "")) > 200
            ]
            print(f"✓ Found {len(papers_with_abstracts)} papers with OpenAlex abstracts")
        else:
            # Fetch abstracts from arXiv (with rate limiting)
            papers_with_abstracts = await fetch_abstracts_for_papers(
                papers, max_concurrent=max_concurrent, verbose=verbose
            )
            print(f"✓ Found {len(papers_with_abstracts)} papers with arXiv abstracts")

        # Filter: keep only papers with DOI and abstract > 200 characters
        papers_with_abstracts = [
            p for p in papers_with_abstracts
            if p.get("doi") and len(p.get("abstract", "")) > 200
        ]
        print(f"✓ After filtering (DOI + abstract > 200 chars): {len(papers_with_abstracts)} papers")

        # Accumulate papers with abstracts (deduplicate by DOI or title)
        # Track existing DOIs and titles to avoid duplicates
        existing_dois = {
            p.get("doi") for p in all_papers_with_abstracts if p.get("doi")
        }
        existing_titles = {
            p.get("title") for p in all_papers_with_abstracts if p.get("title")
        }

        new_papers = []
        for p in papers_with_abstracts:
            doi = p.get("doi")
            title = p.get("title")

            # Skip if duplicate DOI or duplicate title
            if (doi and doi in existing_dois) or (title and title in existing_titles):
                continue

            new_papers.append(p)
            if doi:
                existing_dois.add(doi)
            if title:
                existing_titles.add(title)

        all_papers_with_abstracts.extend(new_papers)

        if len(new_papers) < len(papers_with_abstracts):
            duplicates = len(papers_with_abstracts) - len(new_papers)
            print(f"⚠ Skipped {duplicates} duplicate papers")

        print(f"📊 Total accumulated abstracts: {len(all_papers_with_abstracts)}/{n}")

        # Check if we have enough abstracts
        if len(all_papers_with_abstracts) >= n:
            print(
                f"\n✓ Successfully obtained {len(all_papers_with_abstracts)} papers with abstracts (target: {n})"
            )
            # Trim to exactly n papers
            all_papers_with_abstracts = all_papers_with_abstracts[:n]
            break

        # If not enough, increase the fetch size for next try
        shortage = n - len(all_papers_with_abstracts)
        # Fix exponential backoff: Multiply current fetch_nb by 2, but cap at reasonable limit
        fetch_nb = min(fetch_nb * 2, 2000)
        print(f"⚠ Need {shortage} more papers. Increasing fetch size to {fetch_nb}...")
        try_nb += 1

        # Safety check: limit to 10 tries
        if try_nb >= 10:
            print(
                f"⚠ Warning: Reached maximum retry attempts. Proceeding with {len(all_papers_with_abstracts)} papers."
            )
            break

    # Step 2: Generate research questions using LLM (only once, after we have enough abstracts)
    print(
        f"\n💰 Generating research questions for {len(all_papers_with_abstracts)} papers..."
    )
    all_papers_with_questions = await generate_questions_for_papers(
        all_papers_with_abstracts, model, question_system_prompt, max_concurrent
    )
    print(f"✓ Generated {len(all_papers_with_questions)} research questions")

    # Step 4: Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for paper in all_papers_with_questions:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(
        f"\n✓ Saved {len(all_papers_with_questions)} research questions to: {output_path}"
    )


if __name__ == "__main__":
    # CLI script to fetch papers, get abstracts, and generate research questions
    parser = argparse.ArgumentParser(
        description="Fetch papers, abstracts, and generate research questions"
    )
    parser.add_argument("--n", type=int, default=10, help="Number of papers to fetch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cited-by-min", type=int, default=5, help="Minimum citations for papers"
    )
    parser.add_argument(
        "--cited-by-max", type=int, default=None, help="Maximum citations for papers (default: no limit)"
    )
    parser.add_argument(
        "--min-words-in-title", type=int, default=4, help="Minimum words in title"
    )
    parser.add_argument(
        "--publication-year",
        type=int,
        default=None,
        help="Filter by publication year (optional)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model for question generation",
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
        default=None,
        help="Output file path (jsonl format)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5, help="Max concurrent API calls"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed debug info about arXiv lookups"
    )
    parser.add_argument(
        "--use-openalex-abstracts", action="store_true", 
        help="Use abstracts from OpenAlex instead of fetching from arXiv (faster, better coverage)"
    )

    args = parser.parse_args()

    if args.output is None:
        output = f"research_questions/data/papers_{args.cited_by_min}_{args.cited_by_max}_{args.min_words_in_title}.jsonl"
    else:
        output = args.output

    output_file = Path(output)

    # Load question generation system prompt
    prompt_path = Path(__file__).parent / "prompts" / args.question_prompt
    if not prompt_path.exists():
        raise FileNotFoundError(f"Question prompt not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # Run the main pipeline
    asyncio.run(
        main_pipeline(
            n=args.n,
            seed=args.seed,
            cited_by_min=args.cited_by_min,
            cited_by_max=args.cited_by_max,
            min_words_in_title=args.min_words_in_title,
            publication_year=args.publication_year,
            model=args.model,
            question_system_prompt=system_prompt,
            output_path=output_file,
            max_concurrent=args.max_concurrent,
            verbose=args.verbose,
            use_openalex_abstracts=args.use_openalex_abstracts,
        )
    )
