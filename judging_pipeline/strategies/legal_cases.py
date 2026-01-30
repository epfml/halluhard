from __future__ import annotations

from typing import Any, Dict

from ..core.domain_strategy import DomainStrategy
from ..models.work_items import ClaimItem


class LegalCasesStrategy(DomainStrategy):
    """Strategy for Legal Cases domain."""

    @property
    def task_name(self) -> str:
        return "legal_cases"

    def get_extraction_user_prompt(self, content: str) -> str:
        return f"""Here is a statement:
<START_STATEMENT>
{content}
<END_STATEMENT>

Please output JSON with:
    - type (<case|constitutional_provision|statute|article|other>)
    - content (faithful snippet from user text)
    - reference_name (name/citation of the legal authority)
    - holding_or_description (holding/description from text, or empty string if not provided)

Follow the system prompt rules."""

    def is_valid_claim(self, claim: dict) -> bool:
        # Must have content and a reference name
        has_content = bool(claim.get("content") and claim["content"] not in ("", "Unknown"))
        has_ref = bool(claim.get("reference_name") and claim["reference_name"] not in ("", "Unknown"))
        return has_content and has_ref

    def map_to_claim_item(self, data: dict, conversation_id: int, turn_number: int) -> ClaimItem:
        # Just pass the raw extracted data - no field mapping needed
        return ClaimItem.from_dict(data, conversation_id, turn_number)

    def build_textual_claim_for_websearch(self, claim: ClaimItem) -> str:
        # Access legal-specific fields directly from claim.data
        type_val = claim.data.get("type", "unknown")
        ref_name = claim.data.get("reference_name", "Unknown")
        content = claim.data.get("content", "")
        holding = claim.data.get("holding_or_description", "")
        
        textual_claim = f"Type: {type_val}\n"
        textual_claim += f"Reference: {ref_name}\n"
            
        if holding:
            textual_claim += f"Holding/Description: {holding}\n"
        else:
            textual_claim += "Holding/Description: (Mentioned without description)\n"
            
        textual_claim += f"\nUser's Full Statement:\n{content}"
        
        return textual_claim

    def build_textual_claim_for_judging(self, claim: ClaimItem) -> str:
        # Access legal-specific fields directly from claim.data
        type_val = claim.data.get("type", "unknown")
        ref_name = claim.data.get("reference_name", "Unknown")
        content = claim.data.get("content", "")
        holding = claim.data.get("holding_or_description", "")
        
        textual_claim = f"Type: {type_val}\n"
        textual_claim += f"Reference: {ref_name}\n"
            
        if holding:
            textual_claim += f"Holding/Description: {holding}\n"
        else:
            textual_claim += "Holding/Description: (Mentioned without description)\n"
            
        textual_claim += f"\nUser's Full Statement:\n{content}"
        
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

