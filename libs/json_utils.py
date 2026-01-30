"""JSON utility functions for handling LLM-generated JSON responses."""

import json
import re


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON content from LLM response that may be wrapped in markdown code blocks.
    
    Handles responses like:
        ```json
        {"key": "value"}
        ```
    or:
        ```
        {"key": "value"}
        ```
    or plain JSON.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Extracted JSON string (without markdown code block markers)
    """
    text = response_text.strip()
    
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    return text


def sanitize_json_string(text: str) -> str:
    r"""Sanitize JSON string by handling invalid escape sequences.
    
    LLM responses often contain invalid JSON escape sequences like:
    - LaTeX notation: \(, \), \[, \], \alpha, \beta, \LaTeX
    - Windows paths: \Users, \Documents
    - Other: \s, \d (regex patterns), etc.
    
    Valid JSON escape sequences are: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    
    This function escapes backslashes that are NOT part of valid JSON escapes.
    
    Args:
        text: JSON string that may contain invalid escape sequences
        
    Returns:
        Sanitized JSON string that can be parsed by json.loads()
        
    Example:
        >>> text = '{"content": "Represent \\(t\\) as variable"}'
        >>> sanitized = sanitize_json_string(text)
        >>> result = json.loads(sanitized)
    """
    # First, try to decode as-is
    try:
        json.loads(text)
        return text  # If it works, return as-is
    except json.JSONDecodeError:
        pass
    
    # Fix invalid escape sequences by escaping the backslash
    # Valid JSON escapes: ", \, /, b, f, n, r, t, and uXXXX
    # Pattern matches backslash followed by a character that's NOT a valid escape
    def fix_escapes(match):
        char = match.group(1)
        # Check if this is a valid escape sequence
        if char in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't'):
            return match.group(0)  # Keep valid escapes as-is
        if char == 'u':
            # Check if it's a valid unicode escape \uXXXX
            # We can't easily check here, so keep it as-is
            return match.group(0)
        # Invalid escape - add an extra backslash
        return '\\\\' + char
    
    # Match backslash followed by any character
    fixed_text = re.sub(r'\\(.)', fix_escapes, text)
    
    return fixed_text

