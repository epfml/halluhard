"""
Claim Extraction and Annotation Web Interface

A web-based tool to manually extract claims from model responses in conversation files
and annotate them with human grounding judgments.
"""

from .cli import main

__all__ = ["main"]

