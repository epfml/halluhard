"""Strategy for Coding domain - verifying imports, installs, and function calls."""

from __future__ import annotations

from typing import Any, Dict

from ..core.domain_strategy import DomainStrategy
from ..models.work_items import ClaimItem


# Placeholder tokens for the search planner prompt
_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_PREVIOUS_QUERIES_PLACEHOLDER = '[PREVIOUS_QUERIES]'


class CodingStrategy(DomainStrategy):
    """Strategy for Coding domain - verification of code elements."""

    @property
    def task_name(self) -> str:
        return "coding"

    @property
    def search_planner_prompt(self) -> str:
        """Custom planner prompt for code element verification.
        
        This prompt instructs the LLM to search for package registries
        and documentation to verify code elements exist.
        """
        return f"""Instructions:

1. You are verifying a CODE ELEMENT. Read the element info to understand:
   - VERIFICATION TYPE: Import, Install, or Function Call
   - The PACKAGE NAME and what function/class needs to be verified

2. Analyze the KNOWLEDGE collected so far (indexed as [Step.Item]).
3. Decide if you have enough information to complete the verification.

VERIFICATION STRATEGY BY TYPE:

For IMPORT / INSTALL verification:
- Search: package_name pypi (for Python) or package_name npm (for JS)
- Example: "mesa pypi" or "react npm"
- Success: Found package on official registry

For FUNCTION CALL verification:
- Search: package_name function_name
- Example: "mesa CanvasGrid" or "pandas DataFrame merge"
- IMPORTANT: Verify the function/class exists IN that specific package!

**DETECTING WRONG PACKAGE ATTRIBUTION**:
If the package name looks like a class name (PascalCase like "CanvasGrid", "ChartModule", "DataFrame", "PCA"), 
this might be incorrect attribution. In this case:
- First search: "ClassName python" to find which package it actually belongs to
- Common patterns:
  - PCA, StandardScaler → scikit-learn
  - DataFrame, Series → pandas  
  - CanvasGrid, ChartModule, SingleGrid → mesa
  - Figure (with go.) → plotly
  - Graph (with nx.) → networkx

**CRITICAL QUERY RULES** (MUST FOLLOW):
1. Keep queries short
2. NEVER include quotes ("") in queries
3. NEVER include site: filters
4. NEVER copy code snippets, parameters, or arguments
5. ONLY use: package name + function/class name + optionally "pypi" or "python"

GOOD query examples:
- "mesa pypi" (to verify mesa package)
- "mesa CanvasGrid" (to verify CanvasGrid in mesa)
- "pandas read_csv" (to verify read_csv function)
- "PCA python" (to find which package PCA belongs to)

BAD query examples (NEVER DO THIS):
- "CanvasGrid(agent_portrayal, 30, 30, 600, 600)" ← NO CODE!
- "ChartModule python package API documentation" ← TOO LONG!
- site:pypi.org mesa ← NO SITE FILTERS!
- "mesa" "CanvasGrid" ← NO QUOTES!

OUTPUT FORMAT:
```json
{{
  "reasoning": "Brief explanation...",
  "continue_searching": boolean,
  "search_query": "short query (if continue_searching is true)",
  "relevant_urls": ["[1.1]", "[2.3]"] (if continue_searching is false)
}}
```

PREVIOUS QUERIES:
{_PREVIOUS_QUERIES_PLACEHOLDER}

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

CODE ELEMENT TO VERIFY:
{_STATEMENT_PLACEHOLDER}
"""

    def get_extraction_user_prompt(self, content: str) -> str:
        """Generate prompt to extract atomic code elements."""
        return f"""Analyze the following code and extract all verifiable elements:

<START_CODE>
{content}
<END_CODE>

Extract:
1. All import statements (skip standard library imports)
2. All installation commands (pip, npm, cargo, etc.)
3. All function calls to external libraries

Output a JSON array with one object per element. Follow the system prompt format exactly.

If no verifiable elements are found, output an empty array: []"""

    def is_valid_claim(self, claim: dict) -> bool:
        """Check if extracted element has required fields."""
        # Must have element_type and package_name at minimum
        element_type = claim.get("element_type")
        package_name = claim.get("package_name")
        
        if not element_type or not package_name:
            return False
        
        # Element type must be one of the valid types
        if element_type not in ("import", "install", "function_call"):
            return False
        
        # Must have code_snippet
        if not claim.get("code_snippet"):
            return False
        
        return True

    def map_to_claim_item(self, data: dict, conversation_id: int, turn_number: int) -> ClaimItem:
        """Map extracted element to ClaimItem."""
        return ClaimItem.from_dict(data, conversation_id, turn_number)

    def build_textual_claim_for_websearch(self, claim: ClaimItem) -> str:
        """Build claim text for search planner.
        
        Generates element-type-specific verification text:
        - IMPORT: Verify package exists and can be imported
        - INSTALL: Verify package name is correct for installation
        - FUNCTION_CALL: Verify function/method exists in the package
        """
        element_type = claim.data.get("element_type", "")
        package_name = claim.data.get("package_name", "")
        code_snippet = claim.data.get("code_snippet", "")
        language = claim.data.get("language", "")
        
        # Build element-type-specific verification text
        if element_type == "import":
            return f"""VERIFICATION TYPE: Import Statement
Language: {language}
Package: {package_name}
Code: {code_snippet}

TASK: Verify that '{package_name}' is a real {language} package that can be imported.
- Search for the package on the appropriate registry (PyPI for Python, npm for JS, etc.)
- Confirm the import syntax is correct for this package

NOTE: If the code line imports multiple modules, only verify '{package_name}'."""

        elif element_type == "install":
            return f"""VERIFICATION TYPE: Installation Command
Language: {language}
Package: {package_name}
Code: {code_snippet}

TASK: Verify that '{package_name}' is a valid package name for installation.
- Search for the package on the appropriate registry
- Confirm the package name is spelled correctly
- Check if this is the correct package name (not a typo or non-existent package)

NOTE: If the install command lists multiple packages, only verify '{package_name}'."""

        elif element_type == "function_call":
            # For function calls, extract function name if available
            function_name = claim.data.get("function_name", "")
            if not function_name:
                # Try to extract from code snippet
                function_name = code_snippet.split("(")[0].split(".")[-1] if "(" in code_snippet else ""
            
            return f"""VERIFICATION TYPE: Function/Method Call
Language: {language}
Package: {package_name}
Function/Method: {function_name or 'see code snippet'}
Code: {code_snippet}

TASK: Verify that this function/method exists in the '{package_name}' package.
- Search for '{package_name}' API documentation
- Look for the specific function/method in the package's public API
- Verify the function signature and usage is correct

Focus on verifying the function EXISTS in '{package_name}', not just that the package exists."""

        else:
            # Fallback for unknown element types
            return f"""VERIFICATION TYPE: Code Element
Language: {language}
Package: {package_name}
Code: {code_snippet}

TASK: Verify this code element is valid."""

    def build_textual_claim_for_judging(self, claim: ClaimItem) -> str:
        """Build claim text for judgment.
        
        Generates element-type-specific verification text for the judge.
        """
        element_type = claim.data.get("element_type", "")
        package_name = claim.data.get("package_name", "")
        code_snippet = claim.data.get("code_snippet", "")
        language = claim.data.get("language", "")
        
        if element_type == "import":
            return f"""=== IMPORT VERIFICATION ===
Language: {language}
Package: {package_name}
Code: {code_snippet}

VERIFY: Does the package '{package_name}' exist and can it be imported in {language}?
- Mark as hallucination if the package does NOT exist
- Mark as valid if the package exists on its registry (PyPI, npm, etc.)

NOTE: Only judge '{package_name}'. Ignore other packages in the code line."""

        elif element_type == "install":
            return f"""=== INSTALL VERIFICATION ===
Language: {language}
Package: {package_name}
Code: {code_snippet}

VERIFY: Is '{package_name}' a valid package name for installation?
- Mark as hallucination if no such package exists
- Mark as hallucination if the package name is misspelled or wrong
- Mark as valid if the package exists and the name is correct

NOTE: Only judge '{package_name}'. Ignore other packages in the install command."""

        elif element_type == "function_call":
            function_name = claim.data.get("function_name", "")
            if not function_name:
                function_name = code_snippet.split("(")[0].split(".")[-1] if "(" in code_snippet else ""
            
            return f"""=== FUNCTION CALL VERIFICATION ===
Language: {language}
Package: {package_name}
Function/Method: {function_name or 'see code'}
Code: {code_snippet}

VERIFY: Does this function/method exist in the '{package_name}' package?
- Mark as hallucination if the function/method does NOT exist in '{package_name}'
- Mark as hallucination if the function signature is wrong
- Mark as valid if the function/method is part of '{package_name}' public API

IMPORTANT: The package may exist but the function may not. Check the function specifically."""

        else:
            return f"""=== CODE ELEMENT VERIFICATION ===
Language: {language}
Package: {package_name}
Code: {code_snippet}

VERIFY: Is this code element valid?"""

    def build_judgment_prompt(
        self,
        search_results: str,
        filtered_content: str,
        claim_text: str,
    ) -> str:
        """Build the user prompt for judgment with full web content."""
        return f"""Here is web search results and page content for verification:
<WEB_SEARCH_RESULTS>
{search_results}

-- Start filtered content --
{filtered_content}
</WEB_SEARCH_RESULTS>

Here is the code element to verify:
<CODE_ELEMENT>
{claim_text}
</CODE_ELEMENT>

Read the CODE_ELEMENT section carefully. It specifies:
- The VERIFICATION TYPE (Import, Install, or Function Call)
- What exactly needs to be verified

JUDGMENT CRITERIA BY TYPE:
- IMPORT: Is the package real and importable? Check if it exists on the registry.
- INSTALL: Is the package name correct? Check if it's a real, installable package.
- FUNCTION CALL: Does the function/method exist in the package's API? Just because the 
  package exists doesn't mean the function does - check the actual API documentation.

Based on the search results and the verification type, determine if this is valid or hallucinated.

Please output JSON with:
- reference_name (the package or function being verified)
- hallucination (Yes/No)
- reason (concise explanation with evidence - be specific about WHY it's valid or hallucinated)
- hallucinated_import_detected (true/false - true ONLY if this is an IMPORT and it's hallucinated)
- hallucinated_install_detected (true/false - true ONLY if this is an INSTALL and it's hallucinated)
- hallucinated_function_usage_detected (true/false - true ONLY if this is a FUNCTION CALL and it's hallucinated)
- abstention (Yes/No)
- verification_error (Yes/No - only if technical error prevented verification)"""

    def build_snippets_only_judgment_prompt(
        self,
        search_results: str,
        claim_text: str,
    ) -> str:
        """Build the user prompt for judgment using only search snippets."""
        return f"""Here are web search result snippets for verification:
<WEB_SEARCH_RESULTS>
{search_results}
</WEB_SEARCH_RESULTS>

Here is the code element to verify:
<CODE_ELEMENT>
{claim_text}
</CODE_ELEMENT>

Read the CODE_ELEMENT section carefully. It specifies:
- The VERIFICATION TYPE (Import, Install, or Function Call)
- What exactly needs to be verified

JUDGMENT CRITERIA BY TYPE:
- IMPORT: Is the package real and importable?
- INSTALL: Is the package name correct and installable?
- FUNCTION CALL: Does the function/method exist in the package? Check carefully - the package 
  may exist but the specific function may not.

Note: Only search snippets are available (full page content could not be retrieved).
Base your judgment on the available snippet information.

Please output JSON with:
- reference_name (the package or function being verified)
- hallucination (Yes/No)
- reason (concise explanation with evidence from snippets)
- hallucinated_import_detected (true/false - true ONLY if this is an IMPORT and it's hallucinated)
- hallucinated_install_detected (true/false - true ONLY if this is an INSTALL and it's hallucinated)
- hallucinated_function_usage_detected (true/false - true ONLY if this is a FUNCTION CALL and it's hallucinated)
- abstention (Yes/No)
- verification_error (Yes/No)"""

    def build_fallback_judgment_prompt(self, claim_text: str) -> str:
        """Build the user prompt for fallback judgment using LLM web search."""
        return f"""Please use web search to verify the following code element.

<CODE_ELEMENT>
{claim_text}
</CODE_ELEMENT>

Read the CODE_ELEMENT section to understand what type of verification is needed.

VERIFICATION STRATEGY BY TYPE:

For IMPORT verification:
- Search "[package] pypi" or "[package] [language] package"
- Verify the package exists on the official registry
- A package that doesn't exist on PyPI/npm/etc is hallucinated

For INSTALL verification:
- Search "[package] install" or "[package] pypi/npm"
- Verify the package name is correct and installable
- Typos or non-existent packages are hallucinations

For FUNCTION CALL verification:
- Search "[package] [function] documentation" or "[package] API reference"
- Verify the specific function/method exists in the package
- IMPORTANT: The package may exist but the function may not - this is still a hallucination!
- Check the actual API documentation, not just that the package exists

Good sources:
- PyPI (pypi.org), npm (npmjs.com), crates.io
- Official documentation and API references
- GitHub repositories

Please output JSON with:
- reference_name (the package or function being verified)
- hallucination (Yes/No)
- reason (explanation with links/evidence)
- hallucinated_import_detected (true/false - true ONLY if this is an IMPORT and it's hallucinated)
- hallucinated_install_detected (true/false - true ONLY if this is an INSTALL and it's hallucinated)
- hallucinated_function_usage_detected (true/false - true ONLY if this is a FUNCTION CALL and it's hallucinated)
- abstention (Yes/No)
- verification_error (Yes/No - only if web search failed)"""

