#!/usr/bin/env python3
"""Convert old multi-claim-per-line extraction format to new one-claim-per-line format.

Old format (extraction_result with embedded claims array):
{
    "_type": "extraction_result",
    "conv_idx": 0,
    "conversation_id": 0,
    "turn_number": 1,
    "original_statement": "...",
    "extracted_claims": [
        {
            "inferred_source_type": "paper",
            "claimed_content": "...",
            "claimed_title": "...",
            "claimed_authors": "...",
            "claimed_year": "...",
            "human_reference_grounding": "yes",
            "human_content_grounding": "no",
            "human_comment": "..."
        },
        ...
    ],
    "metadata": {...}
}

New format (one claim per line, ClaimItem-style):
{
    "claim_id": "a984ed87",
    "conversation_id": 1,
    "turn_number": 1,
    "data": {
        "inferred_source_type": "paper",
        "claimed_content": "...",
        "claimed_title": "...",
        "claimed_authors": "...",
        "claimed_year": "...",
        "human_reference_grounding": "yes",
        "human_content_grounding": "no",
        "human_comment": "...",
        "original_statement": "..."
    },
    "metadata": {...}
}

Usage:
    python backwards_compatibility_converter.py input.jsonl output.jsonl
"""

import argparse
import json
import uuid
from pathlib import Path


def generate_claim_id() -> str:
    """Generate a short unique claim ID."""
    return str(uuid.uuid4())[:8]


def convert_old_to_new(input_path: Path, output_path: Path) -> dict:
    """Convert old format to new format.
    
    Args:
        input_path: Path to old format JSONL file
        output_path: Path to write new format JSONL file
        
    Returns:
        Statistics dict with conversion counts
    """
    stats = {
        "input_lines": 0,
        "extraction_results": 0,
        "claims_converted": 0,
        "skipped_lines": 0,
        "human_annotations_preserved": 0,
    }
    
    converted_claims = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            stats["input_lines"] += 1
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Line {line_num}: JSON parse error: {e}")
                stats["skipped_lines"] += 1
                continue
            
            # Check if this is an extraction_result with embedded claims
            if data.get("_type") == "extraction_result" and "extracted_claims" in data:
                stats["extraction_results"] += 1
                
                conversation_id = data.get("conversation_id", data.get("conv_idx", 0))
                turn_number = data.get("turn_number", 1)
                original_statement = data.get("original_statement", "")
                metadata = data.get("metadata", {})
                
                extracted_claims = data.get("extracted_claims", [])
                
                for claim in extracted_claims:
                    # Build the new claim data dict
                    claim_data = {
                        "inferred_source_type": claim.get("inferred_source_type", ""),
                        "claimed_content": claim.get("claimed_content", ""),
                        "full_citation": claim.get("full_citation", ""),
                        "claimed_title": claim.get("claimed_title", ""),
                        "claimed_authors": claim.get("claimed_authors", ""),
                        "claimed_year": claim.get("claimed_year", ""),
                        "claimed_institution": claim.get("claimed_institution", ""),
                        "claimed_url": claim.get("claimed_url", ""),
                        "original_statement": original_statement,
                    }
                    
                    # Preserve human annotations if present
                    human_fields = [
                        "human_reference_grounding",
                        "human_content_grounding", 
                        "human_comment",
                    ]
                    has_human_annotation = False
                    for field in human_fields:
                        if field in claim:
                            claim_data[field] = claim[field]
                            has_human_annotation = True
                    
                    if has_human_annotation:
                        stats["human_annotations_preserved"] += 1
                    
                    # Also preserve any other fields that might be present
                    skip_fields = {
                        "inferred_source_type", "claimed_content", "full_citation",
                        "claimed_title", "claimed_authors", "claimed_year",
                        "claimed_institution", "claimed_url",
                        "human_reference_grounding", "human_content_grounding", "human_comment",
                    }
                    for key, value in claim.items():
                        if key not in skip_fields and key not in claim_data:
                            claim_data[key] = value
                    
                    # Build the new format claim
                    new_claim = {
                        "claim_id": generate_claim_id(),
                        "conversation_id": conversation_id,
                        "turn_number": turn_number,
                        "data": claim_data,
                        "metadata": metadata,
                    }
                    
                    converted_claims.append(new_claim)
                    stats["claims_converted"] += 1
            
            # If it's already in new format (has claim_id and data), pass through
            elif "claim_id" in data and "data" in data:
                converted_claims.append(data)
                stats["claims_converted"] += 1
                # Check for human annotations in data
                if any(k.startswith("human_") for k in data.get("data", {}).keys()):
                    stats["human_annotations_preserved"] += 1
            
            else:
                print(f"[WARNING] Line {line_num}: Unknown format, skipping")
                stats["skipped_lines"] += 1
    
    # Sort by conversation_id, then turn_number for consistent ordering
    converted_claims.sort(key=lambda c: (c["conversation_id"], c["turn_number"]))
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for claim in converted_claims:
            f.write(json.dumps(claim, ensure_ascii=False) + '\n')
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert old multi-claim extraction format to new one-claim-per-line format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input JSONL file in old format",
    )
    
    parser.add_argument(
        "output",
        type=str,
        help="Output JSONL file in new format",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without writing output",
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1
    
    print("=" * 60)
    print("BACKWARDS COMPATIBILITY CONVERTER")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    if args.dry_run:
        print("[DRY RUN] No files will be written\n")
    
    stats = convert_old_to_new(input_path, output_path if not args.dry_run else Path("/dev/null"))
    
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Input lines read:           {stats['input_lines']}")
    print(f"Extraction results found:   {stats['extraction_results']}")
    print(f"Claims converted:           {stats['claims_converted']}")
    print(f"Human annotations kept:     {stats['human_annotations_preserved']}")
    print(f"Skipped lines:              {stats['skipped_lines']}")
    print("=" * 60)
    
    if not args.dry_run:
        print(f"\n[OK] Output written to: {output_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

