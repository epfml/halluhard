from __future__ import annotations

from typing import Any, Dict

from ..core.domain_strategy import DomainStrategy
from ..models.work_items import ClaimItem


class MedicalGuidelinesStrategy(DomainStrategy):
    """Strategy for Medical Guidelines domain."""

    @property
    def task_name(self) -> str:
        return "medical_guidelines"

    def get_extraction_user_prompt(self, content: str) -> str:
        return f"""Here is a statement:
<START_STATEMENT>
{content}
<END_STATEMENT>

Extract all explicitly sourced claims following the system prompt rules."""

    def is_valid_claim(self, claim: dict) -> bool:
        # Must have claimed_content and authority
        has_content = bool(claim.get("claimed_content") and claim["claimed_content"] not in ("", "Unknown"))
        has_authority = bool(claim.get("authority") and claim["authority"] not in ("", "Unknown"))
        
        return has_content and has_authority

    def map_to_claim_item(self, data: dict, conversation_id: int, turn_number: int) -> ClaimItem:
        return ClaimItem.from_dict(data, conversation_id, turn_number)

    def build_textual_claim_for_websearch(self, claim: ClaimItem) -> str:
        authority = claim.data.get("authority", "")
        claimed_content = claim.data.get("claimed_content", "")
        full_citation = claim.data.get("full_citation", "")
        claimed_url = claim.data.get("claimed_url", "")
        
        textual_claim = f"Authority: {authority}\n"
        
        if full_citation:
            textual_claim += f"\nFull Citation: {full_citation}"
        else:
            textual_claim += f"\nClaimed Content: {claimed_content}"
            
        if claimed_url:
            textual_claim += f"\nURL/DOI: {claimed_url}"
        
        return textual_claim

    def build_textual_claim_for_judging(self, claim: ClaimItem) -> str:
        authority = claim.data.get("authority", "")
        claimed_content = claim.data.get("claimed_content", "")
        full_citation = claim.data.get("full_citation", "")
        claimed_url = claim.data.get("claimed_url", "")
        
        textual_claim = f"Authority: {authority}\n"

        if claimed_url:
            textual_claim += f"\nURL/DOI: {claimed_url}"
        
        if full_citation:
            textual_claim += f"\nFull Citation: {full_citation}"
        
        textual_claim += f"\nClaimed Content: {claimed_content}"
        
        return textual_claim

        

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
    - verification_error (Yes/No) - Yes ONLY if there is a technical error preventing judgment. If source not found, set "No".

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

Note: Only search snippets are available.
Base your judgment on the available snippet information.

Please output JSON with:
    - reference_name (as given in the text, or "Unknown" if none)
    - reference_grounding (Yes/No/N/A) followed by explanation
    - content_grounding (Yes/No/N/A) followed by explanation
    - hallucination (Yes/No)
    - abstention (Yes/No)
    - verification_error (Yes/No) - Yes ONLY if technical error. If source not found, set "No".

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

