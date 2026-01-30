"""Fetch paper data from OpenAlex for paper author questions."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
from pyalex import Works, config

# Load OpenAlex configuration from environment
MAILTO = os.getenv("OPENALEX_MAILTO", "example@gmail.com")
config.mailto = MAILTO


def fetch_n(q, n: int) -> List[Any]:
    """Fetch n works from a query using pagination."""
    works = []
    page = 1
    per_page = min(n, 200)  # OpenAlex max page size is 200

    while len(works) < n:
        page_works = q.paginate(method="page", per_page=per_page, page=page).__next__()

        if not page_works:
            break

        works.extend(page_works)
        page += 1

    return works[:n]


def get_random_works(
    n: int,
    seed: int | None = None,
    cited_by_min: int | None = None,
    cited_by_max: int | None = None,
    publication_year: int | None = None,
    min_words_in_title: int | None = None,
) -> List[Dict[str, Any]]:
    """Return n random OpenAlex works with paper and author information.

    Args:
        n: Number of works to fetch
        seed: Random seed for reproducibility
        cited_by_min: Minimum citation count
        cited_by_max: Maximum citation count
        publication_year: Filter by publication year
        min_words_in_title: Minimum number of words in title

    Returns:
        List of dicts with keys: doi, title, language, created_date, cited_by_count, authors
    """
    # Ask OpenAlex only for the fields we need
    fields = [
        "doi",
        "display_name",
        "language",
        "created_date",
        "publication_year",
        "cited_by_count",
        "authorships",
        "abstract_inverted_index",
    ]

    q = Works()

    if publication_year is not None:
        q = q.filter(publication_year=publication_year)

    if cited_by_min is not None:
        q = q.filter(cited_by_count=f">{cited_by_min}")

    # Helper to reconstruct abstract from inverted index
    def reconstruct_abstract(inverted_index: Dict[str, List[int]] | None) -> str | None:
        if not inverted_index:
            return None
        try:
            # Create a list of (position, word) tuples
            word_list = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_list.append((pos, word))
            # Sort by position
            word_list.sort()
            # Join words
            return " ".join(word for _, word in word_list)
        except Exception:
            return None

    if cited_by_max is None and min_words_in_title is None:
        # Cap sample size at 10000 (OpenAlex limit)
        sample_size = min(n, 10000)
        q = q.select(fields).sample(sample_size, seed=seed)
        works = fetch_n(q, sample_size)
    else:
        number_of_results = 0
        try_nb = 0
        fetch_nb = min(n, 10000)  # Start capped at 10000

        while number_of_results < n:
            if try_nb > 0:
                fetch_nb *= 2  # double the fetch size each try

            # Cap fetch_nb at 10000 (OpenAlex limit)
            if fetch_nb > 10000:
                fetch_nb = 10000
                print(f"Warning: Capping fetch size at 10,000 (OpenAlex limit)")

            print(f"Try #{try_nb+1}: fetching {fetch_nb} works...")

            # Vary seed by attempt so we don't keep sampling the same pool when retrying
            seed_for_try = None if seed is None else int(seed) + int(try_nb)
            sampled_query = q.select(fields).sample(fetch_nb, seed=seed_for_try)
            works = fetch_n(sampled_query, fetch_nb)

            print(f"Fetched {len(works)} works.")

            # Remove works with cited_by_count > cited_by_max
            if cited_by_max is not None:
                works = [w for w in works if w.get("cited_by_count", 0) <= cited_by_max]
            if min_words_in_title is not None:
                works = [
                    w
                    for w in works
                    if len((w.get("display_name") or "").split()) >= min_words_in_title
                ]

            number_of_results = len(works)
            print(f"Selecting {n} works from {number_of_results} candidates...")
            
            # Break if we're not making progress or hit the limit
            if fetch_nb >= 10000:
                break
            
            try_nb += 1

        # Sample randomly n
        if seed is not None:
            np.random.seed(seed)

        if number_of_results == 0:
            # No candidates matched the filter criteria
            works = []
        else:
            if number_of_results < n:
                print(
                    f"Warning: Only {number_of_results} candidates available; "
                    f"returning {number_of_results} instead of requested {n}."
                )
            n_select = min(n, number_of_results)
            print(f"Selecting {n_select} works from {number_of_results} candidates...")
            works = list(np.random.choice(works, n_select, replace=False))

    # Normalize authors from the authorships array
    out = []
    for w in works:
        authorships = w.get("authorships") or []
        authors = []
        for a in authorships:
            author_obj = a.get("author") or {}
            name = author_obj.get("display_name")
            if name:
                authors.append(name)

        authors = list(set(authors))  # remove duplicates
        
        # Reconstruct abstract
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
        
        out.append(
            {
                "doi": w.get("doi"),
                "title": w.get("display_name"),
                "language": w.get("language"),
                "created_date": w.get("created_date"),
                "publication_year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count"),
                "authors": authors,
                "abstract": abstract,  # Include OpenAlex abstract
            }
        )
    return out
