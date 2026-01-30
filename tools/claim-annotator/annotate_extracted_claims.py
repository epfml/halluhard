"""
Claim Annotation Web Interface

A web-based tool to annotate claims from HTML evaluation reports or JSONL extraction files.
Allows users to mark reference grounding and content grounding for each fact.

For JSONL files:
    - Preserves the exact original structure
    - Adds human_reference_grounding and human_content_grounding fields to each extracted_claim
    - Output format: {"_type": "extraction_result", "conversation_id": 0, "turn_number": 1, 
                      "extracted_claims": [{"human_reference_grounding": "yes", 
                                            "human_content_grounding": "no", 
                                            "inferred_source_type": "website", ...}]}
    - By default, saves to the same input file (in-place annotation)

For HTML files:
    - Saves annotations progressively to a separate JSONL file

Usage:
    python annotate_responses.py <input_file> [--output <output.jsonl>] [--port 5000]
    
Supported input formats:
    - HTML: Evaluation report files (*.html)
    - JSONL: Extraction files with extracted_claims (*.jsonl)
"""

import argparse
import json
import os
import re
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import html as html_module

# Global state
STATE = {
    "facts": [],
    "current_index": 0,
    "annotations": {},
    "output_path": "",
    "input_file": "",
    "input_type": "html",  # "html" or "jsonl"
    "original_jsonl_entries": [],  # Store original JSONL structure for preservation
    "claim_mapping": {}  # Map fact_id -> (entry_idx, claim_idx) for JSONL
}

def parse_html_report(file_path):
    """Parse HTML report and extract facts."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    facts = []
    
    # Use regex to find result-item divs
    result_pattern = re.compile(
        r'<div class="result-item">(.*?)</div>\s*</div>\s*</div>',
        re.DOTALL
    )
    
    # More robust pattern for the entire result item
    item_pattern = re.compile(
        r'<div class="result-item">.*?<h4>(.*?)</h4>.*?<div class="claim-text">(.*?)</div>.*?<div class="status-badges">(.*?)</div>.*?<div class="reasoning">(.*?)</div>',
        re.DOTALL
    )
    
    for match in item_pattern.finditer(content):
        fact_id = match.group(1).strip()
        claim_text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        
        # Extract badges
        badges_html = match.group(3)
        badge_pattern = re.compile(r'<span class="badge[^"]*">(.*?)</span>', re.DOTALL)
        badges = [re.sub(r'<[^>]+>', '', b).strip() for b in badge_pattern.findall(badges_html)]
        
        reasoning = re.sub(r'<[^>]+>', '', match.group(4)).strip()
        
        facts.append({
            "fact_id": fact_id,
            "claim_text": claim_text,
            "original_badges": badges,
            "original_reasoning": reasoning
        })
    
    return facts


def parse_jsonl_extractions(file_path):
    """Parse JSONL extraction file and extract claims.
    
    Also stores original entries and creates mapping for preserving structure.
    """
    facts = []
    claim_counter = 0
    original_entries = []
    claim_mapping = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            entry_idx = len(original_entries)
            original_entries.append(data)
            
            conversation_id = data.get('conversation_id', f'conv_{line_num}')
            turn_number = data.get('turn_number', 0)
            claims = data.get('extracted_claims', [])
            
            for claim_idx, claim in enumerate(claims):
                claim_counter += 1
                
                # Build a readable claim text from the claim data
                claimed_content = claim.get('claimed_content', '')
                claimed_title = claim.get('claimed_title', '')
                claimed_authors = claim.get('claimed_authors', '')
                claimed_year = claim.get('claimed_year', '')
                claimed_institution = claim.get('claimed_institution', '')
                claimed_url = claim.get('claimed_url', '')
                original_statement = claim.get('original_statement', '')
                source_type = claim.get('inferred_source_type', '')
                
                # Create a unique fact ID
                fact_id = f"Claim #{claim_counter} - Conv {conversation_id} Turn {turn_number}"
                
                # Store mapping for this fact to its position in original structure
                claim_mapping[fact_id] = (entry_idx, claim_idx)
                
                facts.append({
                    "fact_id": fact_id,
                    "claim_text": claimed_content,
                    "original_badges": [],
                    "original_reasoning": "",
                    # Additional JSONL-specific fields
                    "source_type": source_type,
                    "claimed_title": claimed_title,
                    "claimed_authors": claimed_authors,
                    "claimed_year": claimed_year,
                    "claimed_institution": claimed_institution,
                    "claimed_url": claimed_url,
                    "original_statement": original_statement,
                    "conversation_id": conversation_id,
                    "turn_number": turn_number,
                    # Pre-loaded annotations if they exist in the original
                    "human_reference_grounding": claim.get('human_reference_grounding', ''),
                    "human_content_grounding": claim.get('human_content_grounding', ''),
                    "human_comment": claim.get('human_comment', ''),
                    "not_a_claim": claim.get('not_a_claim', False)
                })
    
    # Store in global state
    STATE["original_jsonl_entries"] = original_entries
    STATE["claim_mapping"] = claim_mapping
    
    return facts


def parse_input_file(file_path):
    """Parse input file based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.jsonl':
        STATE["input_type"] = "jsonl"
        return parse_jsonl_extractions(file_path)
    else:
        STATE["input_type"] = "html"
        return parse_html_report(file_path)

def load_existing_annotations(output_path):
    """Load existing annotations from JSONL file or from original JSONL entries."""
    annotations = {}
    
    # For JSONL input, load annotations from the original entries (already parsed)
    if STATE["input_type"] == "jsonl":
        for fact in STATE["facts"]:
            fact_id = fact["fact_id"]
            ref = fact.get("human_reference_grounding", "")
            content = fact.get("human_content_grounding", "")
            comment = fact.get("human_comment", "")
            not_a_claim = fact.get("not_a_claim", False)
            # Consider annotated if has ref+content OR marked as not_a_claim
            if (ref and content) or not_a_claim:
                annotations[fact_id] = {
                    "fact_id": fact_id,
                    "human_reference_grounding": ref,
                    "human_content_grounding": content,
                    "human_comment": comment,
                    "not_a_claim": not_a_claim
                }
        return annotations
    
    # For HTML input, load from output file
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'fact_id' in data:
                        annotations[data['fact_id']] = data
                except json.JSONDecodeError:
                    continue
    return annotations

def save_annotation(annotation):
    """Save a single annotation to JSONL file."""
    with open(STATE["output_path"], 'a', encoding='utf-8') as f:
        f.write(json.dumps(annotation) + "\n")

def get_html_page():
    """Generate the main annotation page HTML."""
    facts = STATE["facts"]
    current = STATE["current_index"]
    annotations = STATE["annotations"]
    
    if not facts:
        return "<html><body><h1>No facts found in the report.</h1></body></html>"
    
    # Calculate progress
    annotated_count = len(annotations)
    total_count = len(facts)
    progress_pct = (annotated_count / total_count * 100) if total_count > 0 else 0
    
    # Current fact
    fact = facts[current]
    fact_id = fact["fact_id"]
    
    # Check if already annotated
    existing = annotations.get(fact_id, {})
    # For JSONL, use human_reference_grounding/human_content_grounding; for HTML use reference_grounding/content_grounding
    if STATE["input_type"] == "jsonl":
        ref_value = existing.get("human_reference_grounding", "")
        content_value = existing.get("human_content_grounding", "")
        comment_value = existing.get("human_comment", "")
        not_a_claim_value = existing.get("not_a_claim", False)
    else:
        ref_value = existing.get("reference_grounding", "")
        content_value = existing.get("content_grounding", "")
        comment_value = existing.get("comment", "")
        not_a_claim_value = existing.get("not_a_claim", False)
    
    claim_escaped = html_module.escape(fact["claim_text"])
    
    # JSONL-specific fields (for extraction files)
    is_jsonl = STATE["input_type"] == "jsonl"
    source_type = html_module.escape(fact.get("source_type", "")) if is_jsonl else ""
    claimed_title = html_module.escape(fact.get("claimed_title", "")) if is_jsonl else ""
    claimed_authors = html_module.escape(fact.get("claimed_authors", "")) if is_jsonl else ""
    claimed_year = html_module.escape(str(fact.get("claimed_year", ""))) if is_jsonl else ""
    claimed_institution = html_module.escape(fact.get("claimed_institution", "")) if is_jsonl else ""
    claimed_url = html_module.escape(fact.get("claimed_url", "")) if is_jsonl else ""
    original_statement = html_module.escape(fact.get("original_statement", "")) if is_jsonl else ""
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claim Annotation Tool</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0d1117;
            --bg-card: #161b22;
            --bg-elevated: #21262d;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-purple: #a371f7;
            --accent-orange: #db6d28;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Space Grotesk', -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            padding: 24px;
        }}
        
        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }}
        
        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .file-info {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-secondary);
            background: var(--bg-elevated);
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }}
        
        /* Progress Bar */
        .progress-section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
        }}
        
        .progress-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .progress-text {{
            font-size: 0.875rem;
            color: var(--text-secondary);
        }}
        
        .progress-text strong {{
            color: var(--accent-blue);
        }}
        
        .progress-bar {{
            height: 8px;
            background: var(--bg-elevated);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        /* Navigation */
        .nav-row {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        
        .nav-btn {{
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.875rem;
            transition: all 0.15s ease;
        }}
        
        .nav-btn:hover:not(:disabled) {{
            background: var(--border);
            border-color: var(--accent-blue);
        }}
        
        .nav-btn:disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}
        
        .nav-input {{
            width: 70px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            text-align: center;
        }}
        
        .nav-input:focus {{
            outline: none;
            border-color: var(--accent-blue);
        }}
        
        /* Fact Card */
        .fact-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        
        .fact-header {{
            background: var(--bg-elevated);
            padding: 14px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .fact-id {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.875rem;
            color: var(--accent-purple);
        }}
        
        .fact-status {{
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 500;
        }}
        
        .fact-status.annotated {{
            background: rgba(63, 185, 80, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(63, 185, 80, 0.3);
        }}
        
        .fact-status.pending {{
            background: rgba(210, 153, 34, 0.15);
            color: var(--accent-yellow);
            border: 1px solid rgba(210, 153, 34, 0.3);
        }}
        
        .fact-body {{
            padding: 20px;
        }}
        
        .claim-section {{
            margin-bottom: 20px;
        }}
        
        .section-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            margin-bottom: 8px;
            font-weight: 600;
        }}
        
        .claim-text {{
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent-blue);
            padding: 16px;
            border-radius: 8px;
            font-size: 0.95rem;
            line-height: 1.7;
            white-space: pre-wrap;
        }}
        
        /* Metadata section for JSONL files */
        .metadata-section {{
            margin-top: 16px;
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        
        .meta-item {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .meta-item.full-width {{
            grid-column: 1 / -1;
        }}
        
        .meta-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        .meta-value {{
            font-size: 0.9rem;
            color: var(--text-primary);
            word-break: break-word;
        }}
        
        .meta-link {{
            color: var(--accent-blue);
            text-decoration: none;
        }}
        
        .meta-link:hover {{
            text-decoration: underline;
        }}
        
        .context-section {{
            margin-top: 16px;
        }}
        
        .context-toggle {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-secondary);
            font-weight: 600;
            cursor: pointer;
            padding: 8px 12px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            border-radius: 6px;
            transition: all 0.15s ease;
            list-style: none;
        }}
        
        .context-toggle::-webkit-details-marker {{
            display: none;
        }}
        
        .context-toggle:hover {{
            background: var(--border);
            color: var(--text-primary);
        }}
        
        .toggle-icon {{
            font-size: 0.6rem;
            transition: transform 0.2s ease;
        }}
        
        details.context-section[open] .toggle-icon {{
            transform: rotate(90deg);
        }}
        
        .context-text {{
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent-purple);
            padding: 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            height: 150px;
            min-height: 80px;
            max-height: 600px;
            overflow-y: auto;
            white-space: pre-wrap;
            line-height: 1.6;
            margin-top: 10px;
            resize: vertical;
        }}
        
        /* Badges */
        .badges-section {{
            margin-bottom: 20px;
        }}
        
        .badges-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        
        .badge {{
            font-size: 0.75rem;
            padding: 8px 12px;
            border-radius: 6px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            max-width: 100%;
            word-break: break-word;
        }}
        
        .badge-ref-yes {{
            background: rgba(63, 185, 80, 0.1);
            border-color: rgba(63, 185, 80, 0.3);
            color: var(--accent-green);
        }}
        
        .badge-ref-no {{
            background: rgba(248, 81, 73, 0.1);
            border-color: rgba(248, 81, 73, 0.3);
            color: var(--accent-red);
        }}
        
        .badge-content-yes {{
            background: rgba(88, 166, 255, 0.1);
            border-color: rgba(88, 166, 255, 0.3);
            color: var(--accent-blue);
        }}
        
        .badge-content-no {{
            background: rgba(219, 109, 40, 0.1);
            border-color: rgba(219, 109, 40, 0.3);
            color: var(--accent-orange);
        }}
        
        .badge-valid {{
            background: rgba(63, 185, 80, 0.15);
            border-color: rgba(63, 185, 80, 0.4);
            color: var(--accent-green);
            font-weight: 600;
        }}
        
        .badge-hallucination {{
            background: rgba(248, 81, 73, 0.15);
            border-color: rgba(248, 81, 73, 0.4);
            color: var(--accent-red);
            font-weight: 600;
        }}
        
        /* Reasoning */
        .reasoning-section {{
            margin-bottom: 20px;
        }}
        
        .reasoning-text {{
            background: var(--bg-dark);
            border: 1px solid var(--border);
            padding: 14px;
            border-radius: 8px;
            font-size: 0.85rem;
            color: var(--text-secondary);
            max-height: 150px;
            overflow-y: auto;
        }}
        
        /* Annotation Form */
        .annotation-form {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
        }}
        
        .form-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: var(--accent-blue);
        }}
        
        .form-group {{
            margin-bottom: 24px;
        }}
        
        .form-label {{
            display: block;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 12px;
            color: var(--text-primary);
        }}
        
        .form-label small {{
            font-weight: 400;
            color: var(--text-secondary);
            display: block;
            margin-top: 4px;
        }}
        
        .options-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
        }}
        
        .option-btn {{
            background: var(--bg-elevated);
            border: 2px solid var(--border);
            color: var(--text-secondary);
            padding: 12px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.875rem;
            font-weight: 500;
            transition: all 0.15s ease;
            text-align: center;
        }}
        
        .option-btn:hover {{
            border-color: var(--accent-blue);
            color: var(--text-primary);
        }}
        
        .option-btn.selected {{
            border-color: var(--accent-blue);
            background: rgba(88, 166, 255, 0.15);
            color: var(--accent-blue);
        }}
        
        .option-btn.yes.selected {{
            border-color: var(--accent-green);
            background: rgba(63, 185, 80, 0.15);
            color: var(--accent-green);
        }}
        
        .option-btn.no.selected {{
            border-color: var(--accent-red);
            background: rgba(248, 81, 73, 0.15);
            color: var(--accent-red);
        }}
        
        .option-btn.unsure.selected {{
            border-color: var(--accent-purple);
            background: rgba(163, 113, 247, 0.15);
            color: var(--accent-purple);
        }}
        
        /* Submit Button */
        .submit-row {{
            display: flex;
            gap: 12px;
            margin-top: 24px;
        }}
        
        .submit-btn {{
            flex: 1;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border: none;
            color: white;
            padding: 14px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        
        .submit-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 20px rgba(88, 166, 255, 0.3);
        }}
        
        .submit-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        
        .skip-btn {{
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 14px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            transition: all 0.15s ease;
        }}
        
        .skip-btn:hover {{
            border-color: var(--accent-yellow);
            color: var(--accent-yellow);
        }}
        
        .not-claim-btn {{
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 14px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            transition: all 0.15s ease;
        }}
        
        .not-claim-btn:hover {{
            border-color: var(--accent-orange);
            color: var(--accent-orange);
        }}
        
        .not-claim-btn.active {{
            border-color: var(--accent-orange);
            background: rgba(219, 109, 40, 0.15);
            color: var(--accent-orange);
        }}
        
        .comment-textarea {{
            width: 100%;
            min-height: 80px;
            background: var(--bg-elevated);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 12px;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.875rem;
            resize: vertical;
            line-height: 1.5;
        }}
        
        .comment-textarea:focus {{
            outline: none;
            border-color: var(--accent-blue);
        }}
        
        .comment-textarea::placeholder {{
            color: var(--text-secondary);
        }}
        
        .grounding-section {{
            transition: opacity 0.2s ease;
        }}
        
        .grounding-section.disabled {{
            opacity: 0.4;
            pointer-events: none;
        }}
        
        /* Keyboard shortcuts hint */
        .shortcuts {{
            margin-top: 20px;
            padding: 12px 16px;
            background: var(--bg-elevated);
            border-radius: 8px;
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .shortcuts kbd {{
            background: var(--bg-dark);
            border: 1px solid var(--border);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            margin: 0 2px;
        }}
        
        /* Toast notification */
        .toast {{
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: var(--accent-green);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 500;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.3s ease;
            z-index: 1000;
        }}
        
        .toast.show {{
            opacity: 1;
            transform: translateY(0);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📋 Claim Annotation Tool</h1>
            <div class="file-info">{html_module.escape(os.path.basename(STATE["input_file"]))}</div>
        </div>
        
        <div class="progress-section">
            <div class="progress-header">
                <span class="progress-text">
                    <strong>{annotated_count}</strong> of <strong>{total_count}</strong> facts annotated ({progress_pct:.1f}%)
                </span>
                <div class="nav-row">
                    <button class="nav-btn" onclick="navigate('first')" {"disabled" if current == 0 else ""}>⏮ First</button>
                    <button class="nav-btn" onclick="navigate('prev')" {"disabled" if current == 0 else ""}>← Prev</button>
                    <input type="number" class="nav-input" id="jumpInput" value="{current + 1}" min="1" max="{total_count}" onchange="jumpTo(this.value)">
                    <span class="progress-text">/ {total_count}</span>
                    <button class="nav-btn" onclick="navigate('next')" {"disabled" if current >= total_count - 1 else ""}>Next →</button>
                    <button class="nav-btn" onclick="navigate('last')" {"disabled" if current >= total_count - 1 else ""}>Last ⏭</button>
                    <button class="nav-btn" onclick="navigate('next_unannotated')">Next Empty ⏩</button>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_pct}%"></div>
            </div>
        </div>
        
        <div class="fact-card">
            <div class="fact-header">
                <span class="fact-id">{html_module.escape(fact_id)}</span>
                <span class="fact-status {"annotated" if fact_id in annotations else "pending"}">
                    {"✓ Annotated" if fact_id in annotations else "○ Pending"}
                </span>
            </div>
            <div class="fact-body">
                <div class="claim-section">
                    <div class="section-label">Claim</div>
                    <div class="claim-text">{claim_escaped}</div>
                </div>
                
                {"" if not is_jsonl else f"""
                <div class="metadata-section">
                    <div class="section-label">Reference Metadata</div>
                    <div class="metadata-grid">
                        {f'<div class="meta-item"><span class="meta-label">Type</span><span class="meta-value">{source_type}</span></div>' if source_type else ''}
                        {f'<div class="meta-item"><span class="meta-label">Title</span><span class="meta-value">{claimed_title}</span></div>' if claimed_title else ''}
                        {f'<div class="meta-item"><span class="meta-label">Authors</span><span class="meta-value">{claimed_authors}</span></div>' if claimed_authors else ''}
                        {f'<div class="meta-item"><span class="meta-label">Year</span><span class="meta-value">{claimed_year}</span></div>' if claimed_year else ''}
                        {f'<div class="meta-item"><span class="meta-label">Institution</span><span class="meta-value">{claimed_institution}</span></div>' if claimed_institution else ''}
                        {f'<div class="meta-item full-width"><span class="meta-label">URL</span><a class="meta-value meta-link" href="{claimed_url}" target="_blank">{claimed_url}</a></div>' if claimed_url else ''}
                    </div>
                </div>
                
                {f'''<details class="context-section">
                    <summary class="context-toggle">Original Context <span class="toggle-icon">▶</span></summary>
                    <div class="context-text">{original_statement}</div>
                </details>''' if original_statement else ''}
                """}
            </div>
        </div>
        
        <div class="annotation-form">
            <div class="form-title">Your Annotation</div>
            
            <div class="form-group">
                <button class="not-claim-btn {"active" if not_a_claim_value else ""}" id="notClaimBtn" onclick="toggleNotAClaim()">
                    ⊘ Not a Claim
                </button>
            </div>
            
            <div class="grounding-section {"disabled" if not_a_claim_value else ""}" id="groundingSection">
                <div class="form-group">
                    <label class="form-label">
                        Reference Grounding
                        <small>Is the reference real and retrievable?</small>
                    </label>
                    <div class="options-grid" id="refOptions">
                        <button class="option-btn yes {"selected" if ref_value == "yes" else ""}" data-value="yes" onclick="selectOption('ref', 'yes', this)">Yes</button>
                        <button class="option-btn no {"selected" if ref_value == "no" else ""}" data-value="no" onclick="selectOption('ref', 'no', this)">No</button>
                        <button class="option-btn unsure {"selected" if ref_value == "unsure" else ""}" data-value="unsure" onclick="selectOption('ref', 'unsure', this)">Unsure</button>
                    </div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">
                        Content Grounding
                        <small>Does the content accurately match the source?</small>
                    </label>
                    <div class="options-grid" id="contentOptions">
                        <button class="option-btn yes {"selected" if content_value == "yes" else ""}" data-value="yes" onclick="selectOption('content', 'yes', this)">Yes</button>
                        <button class="option-btn no {"selected" if content_value == "no" else ""}" data-value="no" onclick="selectOption('content', 'no', this)">No</button>
                        <button class="option-btn unsure {"selected" if content_value == "unsure" else ""}" data-value="unsure" onclick="selectOption('content', 'unsure', this)">Unsure</button>
                    </div>
                </div>
            </div>
            
            <div class="form-group">
                <label class="form-label">
                    Comment
                    <small>Optional justification or notes</small>
                </label>
                <textarea class="comment-textarea" id="commentInput" placeholder="Add any notes or justification here...">{html_module.escape(comment_value)}</textarea>
            </div>
            
            <div class="submit-row">
                <button class="skip-btn" onclick="navigate('next')">Skip →</button>
                <button class="submit-btn" id="submitBtn" onclick="submitAnnotation()">Save & Next →</button>
            </div>
            
            <div class="shortcuts">
                <strong>Keyboard shortcuts:</strong>
                <kbd>←</kbd> Previous &nbsp;|&nbsp;
                <kbd>→</kbd> Next &nbsp;|&nbsp;
                <kbd>1-3</kbd> Reference (Yes/No/Unsure) &nbsp;|&nbsp;
                <kbd>Q/W/E</kbd> Content (Yes/No/Unsure) &nbsp;|&nbsp;
                <kbd>N</kbd> Not a Claim &nbsp;|&nbsp;
                <kbd>Enter</kbd> Save & Next
            </div>
        </div>
    </div>
    
    <div class="toast" id="toast">Saved!</div>
    
    <script>
        let refValue = "{ref_value}";
        let contentValue = "{content_value}";
        let notAClaim = {"true" if not_a_claim_value else "false"};
        
        function selectOption(type, value, btn) {{
            const container = type === 'ref' ? document.getElementById('refOptions') : document.getElementById('contentOptions');
            container.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            
            if (type === 'ref') {{
                refValue = value;
            }} else {{
                contentValue = value;
            }}
        }}
        
        function toggleNotAClaim() {{
            notAClaim = !notAClaim;
            const btn = document.getElementById('notClaimBtn');
            const section = document.getElementById('groundingSection');
            
            if (notAClaim) {{
                btn.classList.add('active');
                section.classList.add('disabled');
            }} else {{
                btn.classList.remove('active');
                section.classList.remove('disabled');
            }}
        }}
        
        function navigate(action) {{
            window.location.href = '/navigate?action=' + action;
        }}
        
        function jumpTo(num) {{
            const index = parseInt(num) - 1;
            if (index >= 0 && index < {total_count}) {{
                window.location.href = '/navigate?action=jump&index=' + index;
            }}
        }}
        
        function submitAnnotation() {{
            // If not marked as "not a claim", require grounding selections
            if (!notAClaim && (!refValue || !contentValue)) {{
                alert('Please select both Reference Grounding and Content Grounding options, or mark as "Not a Claim".');
                return;
            }}
            
            const comment = document.getElementById('commentInput').value;
            
            const form = new URLSearchParams();
            form.append('ref', notAClaim ? '' : refValue);
            form.append('content', notAClaim ? '' : contentValue);
            form.append('comment', comment);
            form.append('not_a_claim', notAClaim ? 'true' : 'false');
            
            fetch('/save', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
                body: form.toString()
            }}).then(response => {{
                if (response.ok) {{
                    showToast();
                    setTimeout(() => navigate('next'), 300);
                }}
            }});
        }}
        
        function showToast() {{
            const toast = document.getElementById('toast');
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            // Don't trigger if typing in input or textarea
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            switch(e.key) {{
                case 'ArrowLeft':
                    if ({current} > 0) navigate('prev');
                    break;
                case 'ArrowRight':
                    if ({current} < {total_count - 1}) navigate('next');
                    break;
                case '1':
                    document.querySelector('#refOptions .yes').click();
                    break;
                case '2':
                    document.querySelector('#refOptions .no').click();
                    break;
                case '3':
                    document.querySelector('#refOptions .unsure').click();
                    break;
                case 'q':
                case 'Q':
                    document.querySelector('#contentOptions .yes').click();
                    break;
                case 'w':
                case 'W':
                    document.querySelector('#contentOptions .no').click();
                    break;
                case 'e':
                case 'E':
                    document.querySelector('#contentOptions .unsure').click();
                    break;
                case 'n':
                case 'N':
                    toggleNotAClaim();
                    break;
                case 'Enter':
                    e.preventDefault();
                    submitAnnotation();
                    break;
            }}
        }});
    </script>
</body>
</html>'''


class AnnotationHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the annotation interface."""
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/' or parsed.path == '':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(get_html_page().encode('utf-8'))
            
        elif parsed.path == '/navigate':
            params = parse_qs(parsed.query)
            action = params.get('action', [''])[0]
            
            if action == 'first':
                STATE["current_index"] = 0
            elif action == 'last':
                STATE["current_index"] = len(STATE["facts"]) - 1
            elif action == 'prev':
                STATE["current_index"] = max(0, STATE["current_index"] - 1)
            elif action == 'next':
                STATE["current_index"] = min(len(STATE["facts"]) - 1, STATE["current_index"] + 1)
            elif action == 'jump':
                index = int(params.get('index', [0])[0])
                STATE["current_index"] = max(0, min(len(STATE["facts"]) - 1, index))
            elif action == 'next_unannotated':
                # Find next unannotated fact
                for i in range(STATE["current_index"] + 1, len(STATE["facts"])):
                    if STATE["facts"][i]["fact_id"] not in STATE["annotations"]:
                        STATE["current_index"] = i
                        break
                else:
                    # Wrap around to beginning
                    for i in range(0, STATE["current_index"]):
                        if STATE["facts"][i]["fact_id"] not in STATE["annotations"]:
                            STATE["current_index"] = i
                            break
            
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/save':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            params = parse_qs(post_data)
            
            ref_value = params.get('ref', [''])[0]
            content_value = params.get('content', [''])[0]
            comment_value = params.get('comment', [''])[0]
            not_a_claim = params.get('not_a_claim', ['false'])[0] == 'true'
            
            # Valid if has grounding values OR marked as not a claim
            is_valid = (ref_value and content_value) or not_a_claim
            
            if is_valid:
                fact = STATE["facts"][STATE["current_index"]]
                fact_id = fact["fact_id"]
                
                if STATE["input_type"] == "jsonl":
                    # For JSONL: Update the original entry structure and preserve it
                    entry_idx, claim_idx = STATE["claim_mapping"][fact_id]
                    
                    # Get the original claim and add annotations at the beginning
                    original_claim = STATE["original_jsonl_entries"][entry_idx]["extracted_claims"][claim_idx]
                    
                    # Create new claim dict with human annotations first, then rest of fields
                    annotated_claim = {}
                    if not_a_claim:
                        annotated_claim["not_a_claim"] = True
                    if ref_value:
                        annotated_claim["human_reference_grounding"] = ref_value
                    if content_value:
                        annotated_claim["human_content_grounding"] = content_value
                    if comment_value:
                        annotated_claim["human_comment"] = comment_value
                    annotated_claim.update(original_claim)
                    
                    # Update the original entry
                    STATE["original_jsonl_entries"][entry_idx]["extracted_claims"][claim_idx] = annotated_claim
                    
                    # Update in-memory annotations tracking
                    STATE["annotations"][fact_id] = {
                        "fact_id": fact_id,
                        "human_reference_grounding": ref_value,
                        "human_content_grounding": content_value,
                        "human_comment": comment_value,
                        "not_a_claim": not_a_claim
                    }
                    
                    # Also update the fact's stored values
                    STATE["facts"][STATE["current_index"]]["human_reference_grounding"] = ref_value
                    STATE["facts"][STATE["current_index"]]["human_content_grounding"] = content_value
                    STATE["facts"][STATE["current_index"]]["human_comment"] = comment_value
                    STATE["facts"][STATE["current_index"]]["not_a_claim"] = not_a_claim
                    
                    # Save only entries that have at least one annotated claim
                    with open(STATE["output_path"], 'w', encoding='utf-8') as f:
                        for entry in STATE["original_jsonl_entries"]:
                            claims = entry.get("extracted_claims", [])
                            if any(c.get("human_reference_grounding") or c.get("not_a_claim") for c in claims):
                                f.write(json.dumps(entry) + "\n")
                else:
                    # For HTML: Use the original annotation format
                    annotation = {
                        "fact_id": fact_id,
                        "claim_text": fact["claim_text"],
                        "reference_grounding": ref_value,
                        "content_grounding": content_value,
                        "comment": comment_value,
                        "not_a_claim": not_a_claim,
                        "original_context": {
                            "badges": fact["original_badges"],
                            "reasoning": fact["original_reasoning"]
                        }
                    }
                    
                    # Update in-memory state
                    STATE["annotations"][fact_id] = annotation
                    
                    # Save to file (rewrite entire file to handle updates)
                    with open(STATE["output_path"], 'w', encoding='utf-8') as f:
                        for ann in STATE["annotations"].values():
                            f.write(json.dumps(ann) + "\n")
            
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Web-based claim annotation tool.")
    parser.add_argument("input_file", help="Path to the input file (HTML report or JSONL extractions)")
    parser.add_argument("--output", help="Path to output JSONL file (default: <input_name>_annotations.jsonl)")
    parser.add_argument("--port", type=int, default=5050, help="Port to run the server on (default: 5050)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found.")
        return
    
    # Set up paths
    STATE["input_file"] = args.input_file
    
    # Detect input type
    ext = os.path.splitext(args.input_file)[1].lower()
    STATE["input_type"] = "jsonl" if ext == ".jsonl" else "html"
    input_type = "JSONL extractions" if ext == ".jsonl" else "HTML report"
    
    # Set default output path
    output_path = args.output
    if not output_path:
        if ext == ".jsonl":
            # For JSONL, default to same file (in-place annotation)
            output_path = args.input_file
        else:
            base_name = os.path.splitext(args.input_file)[0]
            output_path = f"{base_name}_annotations.jsonl"
    STATE["output_path"] = output_path
    
    print(f"Loading {input_type}: {args.input_file}")
    print(f"Output file: {output_path}")
    
    # Parse input file and load existing annotations
    try:
        STATE["facts"] = parse_input_file(args.input_file)
        print(f"Found {len(STATE['facts'])} claims.")
    except Exception as e:
        print(f"Failed to parse input file: {e}")
        return
    
    STATE["annotations"] = load_existing_annotations(output_path)
    print(f"Loaded {len(STATE['annotations'])} existing annotations.")
    
    # Find first unannotated fact
    for i, fact in enumerate(STATE["facts"]):
        if fact["fact_id"] not in STATE["annotations"]:
            STATE["current_index"] = i
            break
    
    # Start server
    server = ThreadingHTTPServer(('localhost', args.port), AnnotationHandler)
    url = f"http://localhost:{args.port}"
    
    print(f"\n{'='*50}")
    print(f"  Annotation server running at: {url}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*50}\n")
    
    # Open browser
    webbrowser.open(url)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        print(f"Annotations saved to: {output_path}")


if __name__ == "__main__":
    main()
