#!/usr/bin/env python3
"""
Claim Extraction and Annotation Web Interface

A web-based tool to manually extract claims from model responses in conversation files
and annotate them with human grounding judgments.

Input: Conversation JSONL file (like conversations_gpt-5_20convs.jsonl)
Output: Extraction results JSONL file with human annotations

Output format:
    {"_type": "extraction_result", "conversation_id": 0, "turn_number": 1,
     "extracted_claims": [{"human_reference_grounding": "yes", 
                           "human_content_grounding": "no",
                           "human_comment": "...",
                           "inferred_source_type": "paper", ...}],
     "metadata": {...}}

Usage:
    # Local mode (default) - opens browser automatically
    python annotate_responses.py <conversations.jsonl> [--output <output.jsonl>] [--port 5050]
    
    # Server mode with file picker (for EC2 deployment)
    python annotate_responses.py --server-mode --data-dir /path/to/data [--port 8080]
    
    # Server mode with pre-selected file
    python annotate_responses.py <conversations.jsonl> --server-mode [--port 5050]
    
    # Server mode with HTTPS
    python annotate_responses.py --server-mode --data-dir /data --ssl-cert cert.pem --ssl-key key.pem
"""

from claim_annotator import main

if __name__ == "__main__":
    main()
