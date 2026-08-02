"""Legal cases data fetcher and synthetic question generator.

Fetch subcommand
----------------
1. Fetches legal case questions from reglab/legal_rag_hallucinations dataset
2. Fetches legal case questions from local JSONL file (California Bar Exam)
3. Fetches an authored set of grounded questions (data/legal_cases_new160.jsonl)
4. Deduplicates the combined set (the HF source repeats each question ~5x)
5. Filters by question categories
6. Saves questions to JSONL format for inference

Generate subcommand
-------------------
Generates authored ("synthetic") grounded legal questions for HalluHard.

This reproduces the methodology used to create `data/legal_cases_new160.jsonl`.
It prompts an LLM to produce questions in each of the four existing styles,
grounded in real U.S. law, then deduplicates (against the existing dataset and
within the new set), validates the format, assigns ids/source, and writes JSONL:

    {"question": str, "question_category": str, "source": "synthetic",
     "case_id": "<category>-gen-<N>"}

Web grounding: by default the generator first searches four authoritative sites
(oyez.org, law.cornell.edu, ballotpedia.org, britannica.com — the same references
used for the original set) via the repo's SerperSearchClient, and feeds the real
fetched snippets into the prompt so questions are anchored in real cases/doctrine
rather than model memory. Disable with --no-ground. Grounding degrades gracefully
if no search API key is configured.

IMPORTANT — this generates a *fresh* set each run; it does not regenerate the
exact shipped 160 (those were authored interactively). Treat the output as
candidates: the LLM can still hallucinate cases/holdings, so always run a
legal-accuracy verification pass over the non-Bar-Exam items before use
(the shipped set was fact-checked and 0/60 were flagged).

Synthetic question schema
-------------------------
    {"question": str, "question_category": str, "source": "synthetic",
     "case_id": "<category>-gen-<N>"}

Category targets (default, overridable via --counts)
-----------------------------------------------------
    Bar Exam        100
    SCALR            30
    Rule QA          20
    Changes in Law   10
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset
from libs.models import get_sampler
from libs.serper import SerperSearchClient


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
    synthetic_cases: list[Dict[str, Any]] | None = None,
    dedupe: bool = True,
) -> list[Dict[str, Any]]:
    """Fetch legal case questions from Hugging Face and local sources.

    Note: the Hugging Face source stores one row per (question x research tool),
    so each question repeats ~5x. With dedupe=True the combined set is collapsed
    to unique questions (by case_id, falling back to question text).

    Args:
        categories: List of question categories to filter by (None = all categories)
        n: Maximum number of cases to fetch per source (None = all). Applied
            independently to each source before deduplication, so passing n=50
            caps each source at 50 rows, not the final combined total.
        split: Dataset split to use for Hugging Face (default: "train")
        include_local: Whether to include local JSONL file (default: True)
        local_path: Path to local JSONL file
            (default: data/california_bar_practice_questions.jsonl)
        synthetic_cases: Pre-generated synthetic question records to include.
            Pass the return value of generate_questions(). If None, no synthetic
            questions are added.
        dedupe: Collapse duplicate questions by case_id / question text (default: True)

    Returns:
        Combined list of legal case dictionaries from all sources
    """
    all_cases = []

    # Fetch from Hugging Face
    hf_cases = fetch_legal_cases_from_hf(categories=categories, n=n, split=split)
    all_cases.extend(hf_cases)

    # Fetch from local file if enabled
    if include_local:
        if local_path is None:
            local_path = Path(__file__).parent / "data" / "california_bar_practice_questions.jsonl"

        local_cases = fetch_legal_cases_from_local(
            local_path=local_path,
            categories=categories,
            n=n,
        )
        all_cases.extend(local_cases)

    # Include pre-generated synthetic questions if provided
    if synthetic_cases:
        if categories:
            synthetic_cases = [c for c in synthetic_cases if c.get("question_category") in categories]
        all_cases.extend(synthetic_cases)
        print(f"[synthetic] Added {len(synthetic_cases)} generated questions")

    print(f"\n[TOTAL] Combined {len(all_cases)} legal case questions from all sources")

    # Deduplicate: the HF source repeats each question once per research tool,
    # so collapse to unique questions by case_id (fallback to question text).
    if dedupe:
        seen: set = set()
        unique: list[Dict[str, Any]] = []
        for case in all_cases:
            key = case.get("case_id") or case.get("question", "").strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(case)
        print(f"[DEDUP] {len(all_cases)} -> {len(unique)} unique questions")
        all_cases = unique

    return all_cases


# ---------------------------------------------------------------------------
# Synthetic question generation
# ---------------------------------------------------------------------------

GROUNDING_SOURCES = [
    "oyez.org",
    "law.cornell.edu",
    "ballotpedia.org",
    "britannica.com",
]
_SITE_FILTER = " (" + " OR ".join(f"site:{d}" for d in GROUNDING_SOURCES) + ")"

GROUNDING_QUERIES = {
    "SCALR": [
        "landmark Supreme Court Fourth Amendment search and seizure cases",
        "landmark Supreme Court First Amendment free speech cases",
        "Supreme Court administrative law and criminal procedure landmark decisions",
    ],
    "Changes in Law": [
        "Supreme Court overruled precedent 2022 2023 2024 2025 term",
        "major Supreme Court decisions changing doctrine Chevron Dobbs Bruen",
    ],
    "Rule QA": [
        "famous legal tests and doctrines elements Miller Strickland Mathews Central Hudson",
    ],
    "Bar Exam": [
        "MBE bar exam black letter law rules evidence torts contracts constitutional law",
    ],
}

CATEGORY_SPECS: dict[str, dict[str, Any]] = {
    "Bar Exam": {
        "count": 100,
        "id_prefix": "bar-exam",
        "style": (
            "Multi-sentence bar-exam-style fact-pattern hypotheticals. Invent a short "
            "(2-4 sentence) fact scenario, then end with a single yes/no or short legal "
            "question. The facts are fictional, but the legal rule being tested must be "
            "real black-letter law. Spread across MBE subjects: Contracts/Sales (UCC), "
            "Torts, Evidence (FRE), Criminal Law, Criminal Procedure, Constitutional Law, "
            "Civil Procedure, Real Property, Agency/Partnership/Corporations, Wills/Trusts."
        ),
    },
    "SCALR": {
        "count": 30,
        "id_prefix": "scalr",
        "style": (
            "State the precise legal ISSUE decided by a REAL U.S. Supreme Court case, "
            "phrased as a yes/no question (you need not name the case). Cover varied areas "
            "(4th/1st/2nd Amendment, administrative law, criminal procedure, civil "
            "procedure/jurisdiction, employment, takings). Frame the genuine question the "
            "Court resolved; do NOT assert the holding as settled fact."
        ),
    },
    "Rule QA": {
        "count": 20,
        "id_prefix": "rule-qa",
        "style": (
            "Short black-letter rule/test questions with a single correct answer, e.g. "
            "naming a canonical multi-factor test or doctrine ('What are the four fair use "
            "factors?', 'What is the Miller test?'). Reference only real, correctly-named tests."
        ),
    },
    "Changes in Law": {
        "count": 10,
        "id_prefix": "changes-in-law",
        "style": (
            "Questions about the CURRENT state of a legal doctrine after a recent, real "
            "decision that changed it (e.g. Chevron after Loper Bright; abortion after "
            "Dobbs; 2nd Amendment after Bruen). Phrase as a genuine question about present law."
        ),
    },
}

SYSTEM_PROMPT = (
    "You are a U.S. legal expert writing questions for a hallucination benchmark. "
    "Every question MUST be grounded in real, verifiable U.S. law (real Supreme Court "
    "issues, real statutes, or real black-letter rules). Never invent fake cases, "
    "citations, statutes, or doctrines. Output ONLY a JSON array of question strings."
)


def load_existing(paths: list[Path]) -> tuple[set[str], dict[str, list[str]]]:
    """Return (normalized existing question set, {category: [example questions]})."""
    seen: set[str] = set()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    for p in paths:
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = (r.get("question") or "").strip()
            if not q:
                continue
            seen.add(q.lower())
            cat = r.get("question_category")
            if cat in CATEGORY_SPECS and len(examples[cat]) < 4:
                examples[cat].append(q)
    return seen, examples


async def gather_grounding(category: str, num_results: int = 6, max_chars: int = 3500) -> str:
    """Fetch real legal reference material for a category from GROUNDING_SOURCES.

    Returns a context block of (title / snippet / url) drawn only from the four
    grounding sites, or "" if search is unavailable, in which case generation
    falls back to the model's own (ungrounded) knowledge.
    """
    queries = GROUNDING_QUERIES.get(category, [])
    if not queries:
        return ""
    try:
        client = SerperSearchClient()
    except ValueError as e:
        print(f"  [grounding] disabled ({e}); falling back to model knowledge.")
        return ""

    lines: list[str] = []
    seen_urls: set[str] = set()
    async with client:
        for q in queries:
            try:
                result, _ = await client.search(q + _SITE_FILTER, num_results=num_results)
            except Exception as e:
                print(f"  [grounding] search failed for {q[:40]!r}: {type(e).__name__}")
                continue
            for item in (result.get("organic") or []):
                url = item.get("link", "")
                if not url or url in seen_urls:
                    continue
                if not any(d in url for d in GROUNDING_SOURCES):
                    continue
                seen_urls.add(url)
                title = item.get("title", "").strip()
                snippet = item.get("snippet", "").strip()
                if title or snippet:
                    lines.append(f"- {title} — {snippet} [{url}]")
    ctx = "\n".join(lines)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars]
    print(f"  [grounding] {category}: {len(seen_urls)} sources fetched ({len(ctx)} chars)")
    return ctx


def build_prompt(category: str, k: int, examples: list[str], avoid: list[str],
                 grounding: str = "") -> str:
    spec = CATEGORY_SPECS[category]
    ex = "\n".join(f"- {e}" for e in examples) or "(no examples available)"
    avoid_block = "\n".join(f"- {a[:160]}" for a in avoid[:30])
    ground_block = (
        f"GROUNDING MATERIAL — real references fetched from {', '.join(GROUNDING_SOURCES)}. "
        f"Base your questions on real cases/doctrines that appear here; do NOT invent any "
        f"case, holding, statute, or test not supported by this material or well-established "
        f"black-letter law:\n{grounding}\n\n"
        if grounding else ""
    )
    return (
        f"Write {k} NEW '{category}' questions for a legal hallucination benchmark.\n\n"
        f"STYLE:\n{spec['style']}\n\n"
        f"{ground_block}"
        f"EXAMPLES of this category's style (match it, do not copy):\n{ex}\n\n"
        f"Each question must be distinct from these already-used questions:\n{avoid_block}\n\n"
        f"Requirements: each grounded in real law; each ends with '?'; no numbering, no "
        f"explanations. Return ONLY a JSON array of exactly {k} strings."
    )


def parse_array(text: str) -> list[str]:
    """Extract a JSON array of strings from model output (tolerant)."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in arr if isinstance(x, str) and str(x).strip()]


def _valid_question(q: str) -> bool:
    return q.endswith("?") and len(q) >= 25


async def generate_category(
    sampler, category: str, target: int, examples: list[str],
    seen: set[str], grounding: str = "", max_rounds: int = 6,
) -> list[str]:
    """Generate `target` unique, valid questions for one category."""
    out: list[str] = []
    local_seen = set(seen)
    rounds = 0
    while len(out) < target and rounds < max_rounds:
        rounds += 1
        need = target - len(out)
        ask = min(need + 5, 40)
        prompt = build_prompt(category, ask, examples, list(local_seen)[-30:], grounding)
        resp = await sampler([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        for q in parse_array(resp.response_text):
            norm = q.lower()
            if norm in local_seen or not _valid_question(q):
                continue
            local_seen.add(norm)
            out.append(q)
            if len(out) >= target:
                break
        print(f"  [{category}] round {rounds}: {len(out)}/{target}")
    return out[:target]


async def generate_questions(
    model: str,
    counts: dict[str, int] | None = None,
    ground: bool = True,
    dedup_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Generate synthetic grounded legal questions and return them as records.

    Args:
        model: Model registry id (see libs/models.py).
        counts: Per-category override, e.g. {"Bar Exam": 50}. Defaults to CATEGORY_SPECS counts.
        ground: Whether to fetch grounding snippets from GROUNDING_SOURCES (default: True).
        dedup_paths: Existing JSONL files to dedup against. Defaults to the two standard files.

    Returns:
        List of question records ready to pass to fetch_legal_cases(synthetic_cases=...).
    """
    if dedup_paths is None:
        base = Path(__file__).parent / "data"
        dedup_paths = [base / p for p in ("legal_cases_dedup.jsonl", "legal_cases_all.jsonl")]

    seen, examples = load_existing(dedup_paths)
    print(f"Loaded {len(seen)} existing question texts to dedup against.")

    sampler = get_sampler(model)
    counters: collections.Counter = collections.Counter()
    records: list[dict[str, Any]] = []
    _counts = counts or {}

    for category, spec in CATEGORY_SPECS.items():
        target = _counts.get(category, spec["count"])
        print(f"Generating {target} '{category}' questions with {model}...")
        grounding = await gather_grounding(category) if ground else ""
        qs = await generate_category(sampler, category, target, examples[category], seen, grounding)
        for q in qs:
            seen.add(q.lower())
            counters[category] += 1
            records.append({
                "question": q,
                "question_category": category,
                "source": "synthetic",
                "case_id": f"{spec['id_prefix']}-gen-{counters[category]}",
            })

    print(f"\n[generate] {len(records)} questions generated. Category counts: {dict(counters)}")
    print("NOTE: run a legal-accuracy verification pass over the SCALR / Rule QA / "
          "Changes-in-Law items before use.")
    return records


async def generate_questions_async(args: argparse.Namespace) -> None:
    """Async entry point for the generate subcommand (generates and writes to disk)."""
    records = await generate_questions(
        model=args.model,
        counts=args.counts,
        ground=args.ground,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(records)} questions to {out}")


async def main_pipeline(
    categories: list[str] | None,
    n: int | None,
    split: str,
    output_path: Path,
    include_local: bool = True,
    local_path: Path | None = None,
    model: str | None = None,
    counts: dict[str, int] | None = None,
    ground: bool = True,
    dedupe: bool = True,
) -> None:
    """Fetch legal case questions, optionally generating synthetic ones, and save.

    Args:
        categories: List of question categories to filter by
        n: Maximum number of cases to fetch per source
        split: Dataset split to use for Hugging Face
        output_path: Output file path
        include_local: Whether to include local JSONL file
        local_path: Path to local JSONL file
        model: Model registry id for synthetic generation (None = skip generation)
        counts: Per-category count overrides for generation
        ground: Whether to use web grounding during generation
        dedupe: Collapse duplicate questions by case_id / question text
    """
    synthetic: list[Dict[str, Any]] | None = None
    if model:
        synthetic = await generate_questions(model=model, counts=counts, ground=ground)

    cases = fetch_legal_cases(
        categories=categories,
        n=n,
        split=split,
        include_local=include_local,
        local_path=local_path,
        synthetic_cases=synthetic,
        dedupe=dedupe,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"\n[OK] Saved {len(cases)} legal case questions to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Legal cases data fetcher and synthetic question generator."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # fetch subcommand
    # ------------------------------------------------------------------
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch legal case questions from Hugging Face + local JSONL sources.",
    )
    fetch_parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Question categories to filter by (e.g., 'Bar Exam' 'Civil Procedure'). "
             "If not specified, fetches all categories.",
    )
    fetch_parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Maximum number of cases to fetch per source (default: fetch all)",
    )
    fetch_parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for Hugging Face (default: train)",
    )
    fetch_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (jsonl format)",
    )
    fetch_parser.add_argument(
        "--no-local",
        action="store_true",
        help="Disable fetching from local JSONL file (only fetch from Hugging Face)",
    )
    fetch_parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to local JSONL file (default: data/california_bar_practice_questions.jsonl)",
    )
    fetch_parser.add_argument(
        "--model",
        default=None,
        help="Model registry id for live synthetic generation (see libs/models.py). "
             "If omitted, no synthetic questions are added.",
    )
    fetch_parser.add_argument(
        "--counts",
        default=None,
        help="Optional JSON overriding per-category generation counts, e.g. "
             "'{\"Bar Exam\": 100, \"SCALR\": 30}'. Only used when --model is set.",
    )
    fetch_parser.add_argument(
        "--no-ground",
        dest="ground",
        action="store_false",
        help="Disable web grounding during synthetic generation.",
    )
    fetch_parser.set_defaults(ground=True)
    fetch_parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate questions (by default deduplicated by case_id)",
    )

    # ------------------------------------------------------------------
    # generate subcommand
    # ------------------------------------------------------------------
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate grounded synthetic legal questions via LLM.",
    )
    gen_parser.add_argument(
        "--model",
        help="Model registry id (see libs/models.py). Use a strong model.",
    )
    gen_parser.add_argument(
        "--out",
        default="legal_cases/data/legal_cases_new160.jsonl",
        help="Output JSONL path.",
    )
    gen_parser.add_argument(
        "--counts",
        default=None,
        help="Optional JSON overriding per-category counts, e.g. "
             "'{\"Bar Exam\": 100, \"SCALR\": 30}'.",
    )
    gen_parser.add_argument(
        "--no-ground",
        dest="ground",
        action="store_false",
        help="Disable web grounding (oyez.org / law.cornell.edu / "
             "ballotpedia.org / britannica.com); use model knowledge only.",
    )
    gen_parser.set_defaults(ground=True)

    args = parser.parse_args()

    if args.command == "fetch":
        if args.output is None:
            if args.categories:
                category_str = "_".join(cat.replace(" ", "_") for cat in args.categories)
                output = f"legal_cases/data/legal_cases_{category_str}.jsonl"
            else:
                output = "legal_cases/data/legal_cases_all.jsonl"
        else:
            output = args.output

        counts = json.loads(args.counts) if args.counts else None
        asyncio.run(main_pipeline(
            categories=args.categories,
            n=args.n,
            split=args.split,
            output_path=Path(output),
            include_local=not args.no_local,
            local_path=Path(args.local_path) if args.local_path else None,
            model=args.model,
            counts=counts,
            ground=args.ground,
            dedupe=not args.no_dedupe,
        ))

    elif args.command == "generate":
        args.counts = json.loads(args.counts) if args.counts else {}
        asyncio.run(generate_questions_async(args))

