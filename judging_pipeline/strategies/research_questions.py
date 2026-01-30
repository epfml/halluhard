from __future__ import annotations

from typing import Any, Dict

from ..core.domain_strategy import DomainStrategy
from ..models.work_items import ClaimItem


class ResearchQuestionsStrategy(DomainStrategy):
    """Strategy for Research Questions domain."""

    @property
    def task_name(self) -> str:
        return "research_questions"

    def get_extraction_user_prompt(self, content: str) -> str:
        return f"""Here is a statement:
<START_STATEMENT>
{content}
<END_STATEMENT>

Please output JSON with:
    - inferred_source_type (<paper|book|dataset|website|report|other>)
    - claimed_content (faithful snippet from user text)
    - full_citation (verbatim bibliography entry if present in user text, empty string if none)
    - claimed_title (title of the source from user text)
    - claimed_authors (comma separated authors if present in user text, empty string if none)
    - claimed_year (publication year if present in user text, empty string if none)
    - claimed_institution (related institution if present in user text, empty string if none)
    - claimed_url (URL/DOI/arXiv ID if present in user text, empty string if none)

Follow the system prompt rules."""

    def is_valid_claim(self, claim: dict) -> bool:
        # For research questions, we just need content
        return bool(claim.get("claimed_content") and claim["claimed_content"] not in ("", "Unknown"))

    def map_to_claim_item(self, data: dict, conversation_id: int, turn_number: int) -> ClaimItem:
        return ClaimItem.from_dict(data, conversation_id, turn_number)

    def build_textual_claim_for_websearch(self, claim: ClaimItem) -> str:
        parts = []
        
        claimed_content = claim.data.get("claimed_content", "")
        full_citation = claim.data.get("full_citation", "")
        claimed_title = claim.data.get("claimed_title", "")
        claimed_authors = claim.data.get("claimed_authors", "")
        claimed_year = claim.data.get("claimed_year", "")
        claimed_url = claim.data.get("claimed_url", "")
        
        if claimed_content:
            parts.append(f"Content: {claimed_content}")
        
        if full_citation:
            parts.append(f"Citation: {full_citation}")
        else:
            if claimed_title:
                parts.append(f"Title: {claimed_title}")
            if claimed_authors:
                parts.append(f"Authors: {claimed_authors}")
            if claimed_year:
                parts.append(f"Year: {claimed_year}")
            if claimed_url:
                parts.append(f"URL/DOI: {claimed_url}")
        
        return "\n".join(parts) if parts else ""

    def build_textual_claim_for_judging(self, claim: ClaimItem) -> str:
        parts = []
        
        claimed_content = claim.data.get("claimed_content", "")
        full_citation = claim.data.get("full_citation", "")
        claimed_title = claim.data.get("claimed_title", "")
        claimed_authors = claim.data.get("claimed_authors", "")
        claimed_year = claim.data.get("claimed_year", "")
        claimed_url = claim.data.get("claimed_url", "")
        
        if claimed_content:
            parts.append(f"Content: {claimed_content}")
        
        if full_citation:
            parts.append(f"Citation: {full_citation}")
        else:
            if claimed_title:
                parts.append(f"Title: {claimed_title}")
            if claimed_authors:
                parts.append(f"Authors: {claimed_authors}")
            if claimed_year:
                parts.append(f"Year: {claimed_year}")
            if claimed_url:
                parts.append(f"URL/DOI: {claimed_url}")
        
        return "\n".join(parts) if parts else ""

    def build_judgment_prompt(self, search_results: str, filtered_content: str, claim_text: str) -> str:
        return f"""Here is web search results that contains known, trustworthy information:
<WEB_SEARCH_RESULTS>
{search_results}

-- Start filtered content --
{filtered_content}
</WEB_SEARCH_RESULTS>

Here is the reference and user's snippet:
<START>
{claim_text}
<END>

Please output JSON with:
    - reference_name (as given in the text, or "Unknown" if none)
    - reference_grounding (Yes/No/N/A) followed by explanation
    - content_grounding (Yes/No/N/A) followed by explanation
    - hallucination (Yes/No)
    - abstention (Yes/No)
    - verification_error (Yes/No) - Yes ONLY if there is a technical error preventing judgment (e.g. nothing found in the websearch content tags). If the source is not found, set this to "No" and mark reference_grounding as "No".

Follow the system prompt rules."""

    def build_snippets_only_judgment_prompt(self, search_results: str, claim_text: str) -> str:
        return f"""Here are web search result snippets:
<WEB_SEARCH_RESULTS>
{search_results}
</WEB_SEARCH_RESULTS>

Here is the reference and user's snippet:
<START>
{claim_text}
<END>

Note: Only search snippets are available (full page content could not be retrieved).
Base your judgment on the available snippet information.

Please output JSON with:
    - reference_name (as given in the text, or "Unknown" if none)
    - reference_grounding (Yes/No/N/A) followed by explanation
    - content_grounding (Yes/No/N/A) followed by explanation
    - hallucination (Yes/No)
    - abstention (Yes/No)
    - verification_error (Yes/No) - Yes ONLY if there is a technical error preventing judgment (e.g. nothing found in the websearch content tags). If snippets are insufficient or the source is not found, set this to "No" and mark reference_grounding as "No".

Follow the system prompt rules."""

    def build_fallback_judgment_prompt(self, claim_text: str) -> str:
        return f"""Please use web search to verify the following reference and claim.

Here is the reference and user's snippet:
<START>
{claim_text}
<END>

Search the web to find information about this reference and verify whether the claimed content is accurate.

Please output JSON with:
    - reference_name (as given in the text, or "Unknown" if none)
    - reference_grounding (Yes/No/N/A) followed by explanation
    - content_grounding (Yes/No/N/A) followed by explanation
    - hallucination (Yes/No)
    - abstention (Yes/No)
    - verification_error (Yes/No) - Yes only if web search failed

Follow the system prompt rules."""

