"""Generate HTML reports from evaluation results JSONL files.

This script reads evaluation results from JSONL files and generates
HTML reports using the HTMLReporter class. It supports all task types:
- Grounding tasks (research_questions, legal_cases, medical_guidelines)
- Paper authors task
- Coding task

Usage:
    python report.py --task <task_name> --input <eval_results.jsonl> [--output <output_dir>]

Examples:
    # Generate report for research questions
    python report.py --task research_questions --input research_questions/results/conversations_eval.jsonl
    
    # Generate report for paper authors
    python report.py --task paper_authors --input paper_authors/results/conversations_eval.jsonl
    
    # Generate report for coding task
    python report.py --task coding --input coding/results/conversations_eval.jsonl --output reports/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict

from libs.html_reporter import HTMLReporter, HTMLReportConfig
from libs.schemas import Conversation, ConversationTurn
from libs.storage import load_conversations


def load_evaluation_results(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load evaluation results from JSONL file.
    
    Args:
        jsonl_path: Path to evaluation results JSONL file
        
    Returns:
        List of evaluation result dictionaries
    """
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Skip metadata lines
                if data.get("_type") == "evaluation_result":
                    results.append(data)
    return results


def aggregate_results_for_grounding_task(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results for grounding-based tasks (research_questions, legal_cases, medical_guidelines).
    
    Args:
        results: List of evaluation results
        
    Returns:
        Aggregated result dictionary with hallucination_rate, total_facts, per_turn_stats, etc.
    """
    all_facts = []
    hallucinated_count = 0
    verification_errors = 0
    # Pipeline fallback/debug flags (primarily for judging_pipeline outputs)
    input_use_fallback_count = 0
    judge_used_websearch_fallback_count = 0
    snippets_only_count = 0
    
    # Per-turn statistics
    per_turn_stats = {}
    
    # Citation-based statistics (for research_questions task)
    # Categories: low_citations (<50), high_citations (>=1000)
    citation_stats = {
        "low_citations": {  # cited_by_count < 50
            "total_facts": 0,
            "hallucinated_facts": 0,
            "verification_errors": 0,
            "reference_failures": 0,
            "content_failures": 0,
            "abstentions": 0,
            "paper_count": 0,
            "papers_seen": set(),  # Track unique papers
        },
        "high_citations": {  # cited_by_count >= 1000
            "total_facts": 0,
            "hallucinated_facts": 0,
            "verification_errors": 0,
            "reference_failures": 0,
            "content_failures": 0,
            "abstentions": 0,
            "paper_count": 0,
            "papers_seen": set(),  # Track unique papers
        },
    }
    
    for result in results:
        details = result.get("details", {})
        metadata = result.get("metadata", {})  # Get conversation-level metadata
        # Try both possible keys for facts/claims
        facts = details.get("facts", details.get("claim_evaluations", []))
        
        # Get citation count for this paper (from metadata)
        cited_by_count = metadata.get("cited_by_count")
        conversation_id = result.get("conversation_id", "")
        
        # Determine citation category
        citation_category = None
        if cited_by_count is not None:
            if cited_by_count < 50:
                citation_category = "low_citations"
            elif cited_by_count >= 1000:
                citation_category = "high_citations"
            
            # Track unique papers per category
            if citation_category:
                paper_key = metadata.get("paper_title", conversation_id)
                if paper_key not in citation_stats[citation_category]["papers_seen"]:
                    citation_stats[citation_category]["papers_seen"].add(paper_key)
                    citation_stats[citation_category]["paper_count"] += 1
        
        for fact in facts:
            # Enrich fact with metadata from the conversation (for research_questions)
            enriched_fact = {
                **fact,
                "_metadata": metadata,  # Store full metadata dict
                "_conversation_id": result.get("conversation_id", ""),
            }
            all_facts.append(enriched_fact)
            
            # Check for verification error first - these should be excluded from stats
            # verification_error can be: "Yes", "Unknown" (both are errors), or "No" (success)
            verification_error = fact.get("verification_error", "No")
            is_verification_error = verification_error if isinstance(verification_error, bool) else verification_error.lower() in ["yes", "true", "unknown"]
            
            if is_verification_error:
                verification_errors += 1
                # Also track verification errors in citation stats
                if citation_category:
                    citation_stats[citation_category]["verification_errors"] += 1
                continue  # Skip this fact from all calculations
            
            # Track judging pipeline fallback/debug flags (exclude verification errors)
            def _as_bool(v: Any) -> bool:
                if isinstance(v, bool):
                    return v
                if v is None:
                    return False
                return str(v).strip().lower() in ["yes", "true", "1"]

            if _as_bool(fact.get("input_use_fallback", False)):
                input_use_fallback_count += 1
            if _as_bool(fact.get("judge_used_websearch_fallback", False)):
                judge_used_websearch_fallback_count += 1
            if _as_bool(fact.get("snippets_only", False)):
                snippets_only_count += 1

            # Check if hallucination is Yes/No string or boolean
            halluc_value = fact.get("hallucination", False)
            is_halluc = halluc_value if isinstance(halluc_value, bool) else halluc_value.lower() in ["yes", "true"]
            if is_halluc:
                hallucinated_count += 1
            
            # Check abstention
            abstention_value = fact.get("abstention", "No")
            is_abstention = abstention_value if isinstance(abstention_value, bool) else abstention_value.lower() in ["yes", "true"]
            
            # Track per-turn statistics (excluding verification errors)
            turn_idx = fact.get("turn_idx", 0)
            if turn_idx not in per_turn_stats:
                per_turn_stats[turn_idx] = {
                    "total_facts": 0,
                    "hallucinated_facts": 0,
                    "reference_failures": 0,
                    "content_failures": 0,
                    "abstentions": 0,
                }
            
            per_turn_stats[turn_idx]["total_facts"] += 1
            if is_halluc:
                per_turn_stats[turn_idx]["hallucinated_facts"] += 1
            if is_abstention:
                per_turn_stats[turn_idx]["abstentions"] += 1
            
            # Check reference and content grounding
            ref_grounding = fact.get("reference_grounding", "").lower()
            content_grounding = fact.get("content_grounding", "").lower()
            
            if ref_grounding.startswith("no") or ref_grounding in ["unknown", "not found", "false"]:
                per_turn_stats[turn_idx]["reference_failures"] += 1
            
            if content_grounding.startswith("no") or content_grounding in ["unknown", "not grounded", "false"]:
                per_turn_stats[turn_idx]["content_failures"] += 1
            
            # Update citation-based statistics
            if citation_category:
                citation_stats[citation_category]["total_facts"] += 1
                if is_halluc:
                    citation_stats[citation_category]["hallucinated_facts"] += 1
                if is_abstention:
                    citation_stats[citation_category]["abstentions"] += 1
                if ref_grounding.startswith("no") or ref_grounding in ["unknown", "not found", "false"]:
                    citation_stats[citation_category]["reference_failures"] += 1
                if content_grounding.startswith("no") or content_grounding in ["unknown", "not grounded", "false"]:
                    citation_stats[citation_category]["content_failures"] += 1
    
    # Compute rates for each turn
    for turn_idx, stats in per_turn_stats.items():
        total = stats["total_facts"]
        if total > 0:
            stats["hallucination_rate"] = stats["hallucinated_facts"] / total
            stats["reference_failure_rate"] = stats["reference_failures"] / total
            stats["content_failure_rate"] = stats["content_failures"] / total
            stats["abstention_rate"] = stats["abstentions"] / total
        else:
            stats["hallucination_rate"] = 0.0
            stats["reference_failure_rate"] = 0.0
            stats["content_failure_rate"] = 0.0
            stats["abstention_rate"] = 0.0
    
    # Compute rates for each citation category
    for category, stats in citation_stats.items():
        # Remove the temporary set used for tracking unique papers
        del stats["papers_seen"]
        
        total = stats["total_facts"]
        if total > 0:
            stats["hallucination_rate"] = stats["hallucinated_facts"] / total
            stats["reference_failure_rate"] = stats["reference_failures"] / total
            stats["content_failure_rate"] = stats["content_failures"] / total
            stats["abstention_rate"] = stats["abstentions"] / total
        else:
            stats["hallucination_rate"] = 0.0
            stats["reference_failure_rate"] = 0.0
            stats["content_failure_rate"] = 0.0
            stats["abstention_rate"] = 0.0
    
    total_facts = len(all_facts)
    valid_facts = total_facts - verification_errors  # Facts that were successfully evaluated
    hallucination_rate = hallucinated_count / valid_facts if valid_facts > 0 else 0.0
    
    return {
        "hallucination_rate": hallucination_rate,
        "total_facts": total_facts,
        "valid_facts": valid_facts,
        "verification_errors": verification_errors,
        "hallucinated_facts": hallucinated_count,
        # Fallback/debug flags (counts exclude verification errors)
        "input_use_fallback_count": input_use_fallback_count,
        "judge_used_websearch_fallback_count": judge_used_websearch_fallback_count,
        "snippets_only_count": snippets_only_count,
        "input_use_fallback_rate": (input_use_fallback_count / valid_facts) if valid_facts > 0 else 0.0,
        "judge_used_websearch_fallback_rate": (judge_used_websearch_fallback_count / valid_facts) if valid_facts > 0 else 0.0,
        "snippets_only_rate": (snippets_only_count / valid_facts) if valid_facts > 0 else 0.0,
        "facts": all_facts,
        "per_turn_stats": per_turn_stats,
        "citation_stats": citation_stats,
    }


def aggregate_results_for_paper_authors(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results for paper_authors task.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Aggregated result dictionary with average score and aggregated details
    """
    total_score = 0.0
    total_predicted = 0
    total_matches = 0
    all_predicted_names = []
    all_matched_names = []
    categories = []
    
    for result in results:
        total_score += result.get("score", 0.0)
        details = result.get("details", {})
        
        predicted_names = details.get("predicted_names", [])
        matched_names = details.get("matched_names", [])
        matches = details.get("matches", len(matched_names))
        predicted_count = details.get("predicted_count", len(predicted_names))
        
        total_predicted += predicted_count
        total_matches += matches
        all_predicted_names.extend(predicted_names)
        all_matched_names.extend(matched_names)
        categories.append(details.get("category", "Unknown"))
    
    avg_score = total_score / len(results) if results else 0.0
    
    # Count category distribution
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return {
        "score": avg_score,
        "reasoning": f"Average across {len(results)} conversations",
        "details": {
            "predicted_names": all_predicted_names,
            "matched_names": all_matched_names,
            "matches": total_matches,
            "predicted_count": total_predicted,
            "category_distribution": category_counts,
        }
    }


def aggregate_results_for_coding(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results for coding task with language-specific per-turn evaluation.
    
    Supports both old format (turn_evaluations) and new format (claim_evaluations from judging pipeline).
    
    Args:
        results: List of evaluation results
        
    Returns:
        Aggregated result dictionary with:
        - Overall hallucination rate (hallucinated claims / total claims)
        - Language-specific statistics
        - Per-element-type statistics
    """
    total_score = 0.0
    
    # Conversation-level hallucination counts
    conv_import_halluc_count = 0
    conv_install_halluc_count = 0
    conv_function_halluc_count = 0
    
    # All claim evaluations (enriched with metadata)
    all_claim_evaluations = []
    
    # Total claims and hallucinated claims for overall rate
    total_claims = 0
    hallucinated_claims = 0
    
    # Response-level (turn-level) aggregation - sum across all conversations
    total_responses = 0
    import_hallucinated_responses = 0
    install_hallucinated_responses = 0
    function_hallucinated_responses = 0
    overall_hallucinated_responses = 0
    
    # Per-element-type statistics
    element_type_stats = {
        "import": {"total": 0, "hallucinated": 0},
        "install": {"total": 0, "hallucinated": 0},
        "function_call": {"total": 0, "hallucinated": 0},
    }
    
    # Language-specific statistics
    language_stats = {}
    
    # Per-turn statistics (aggregated across all languages)
    per_turn_stats = {}
    
    for result in results:
        total_score += result.get("score", 0.0)
        details = result.get("details", {})
        metadata = result.get("metadata", {})
        language = metadata.get("language", "Unknown")
        
        # Track conversation-level hallucinations
        if details.get("hallucinated_import_detected", False):
            conv_import_halluc_count += 1
        if details.get("hallucinated_install_detected", False):
            conv_install_halluc_count += 1
        if details.get("hallucinated_function_usage_detected", False):
            conv_function_halluc_count += 1
        
        # Aggregate response-level (turn-level) stats from each conversation
        total_responses += details.get("total_responses", 0)
        import_hallucinated_responses += details.get("import_hallucinated_responses", 0)
        install_hallucinated_responses += details.get("install_hallucinated_responses", 0)
        function_hallucinated_responses += details.get("function_hallucinated_responses", 0)
        overall_hallucinated_responses += details.get("overall_hallucinated_responses", 0)
        
        # Support both old format (turn_evaluations) and new format (claim_evaluations)
        claim_evals = details.get("claim_evaluations", details.get("turn_evaluations", []))
        
        # Initialize language stats if not exists (response-level, not claim-level)
        if language not in language_stats:
            language_stats[language] = {
                "total_responses": 0,
                "hallucinated_responses": 0,
                "import_hallucinated_responses": 0,
                "install_hallucinated_responses": 0,
                "function_hallucinated_responses": 0,
            }
        
        # Aggregate response-level stats per language
        language_stats[language]["total_responses"] += details.get("total_responses", 0)
        language_stats[language]["hallucinated_responses"] += details.get("overall_hallucinated_responses", 0)
        language_stats[language]["import_hallucinated_responses"] += details.get("import_hallucinated_responses", 0)
        language_stats[language]["install_hallucinated_responses"] += details.get("install_hallucinated_responses", 0)
        language_stats[language]["function_hallucinated_responses"] += details.get("function_hallucinated_responses", 0)
        
        # Process each claim evaluation
        for claim_eval in claim_evals:
            # Determine element type from claim data or flags
            claim_data = claim_eval.get("claim", {})
            element_type = claim_data.get("element_type", "unknown")
            
            # If element_type not in claim, infer from flags
            if element_type == "unknown":
                if claim_eval.get("hallucinated_import_detected", False):
                    element_type = "import"
                elif claim_eval.get("hallucinated_install_detected", False):
                    element_type = "install"
                elif claim_eval.get("hallucinated_function_usage_detected", False):
                    element_type = "function_call"
            
            # Check if this claim is hallucinated
            # Primary check: use the hallucination field from the judge
            hallucination_value = claim_eval.get("hallucination", "Unknown")
            is_hallucinated = (
                hallucination_value if isinstance(hallucination_value, bool) 
                else str(hallucination_value).lower() in ["yes", "true"]
            )
            
            # Only use individual flags as fallback if hallucination field is unknown/missing
            # Don't override explicit "No" from the judge
            if str(hallucination_value).lower() in ["unknown", ""]:
                is_hallucinated = any([
                    claim_eval.get("hallucinated_import_detected", False),
                    claim_eval.get("hallucinated_install_detected", False),
                    claim_eval.get("hallucinated_function_usage_detected", False)
                ])
            
            total_claims += 1
            if is_hallucinated:
                hallucinated_claims += 1
            
            # Update element type stats
            if element_type in element_type_stats:
                element_type_stats[element_type]["total"] += 1
                if is_hallucinated:
                    element_type_stats[element_type]["hallucinated"] += 1
            
            # Get turn number for per-turn stats
            turn_number = claim_eval.get("turn_idx", claim_eval.get("turn_number", 0))
            
            # Enrich claim evaluation with metadata
            enriched_claim = {
                **claim_eval,
                "conversation_id": result.get("conversation_id"),
                "language": language,
                # Use prompt from metadata as the task description
                "task": metadata.get("prompt", metadata.get("task", "Unknown")),
                "prompt": metadata.get("prompt", ""),
                "element_type": element_type,
                "is_hallucinated": is_hallucinated,
                "package_name": claim_data.get("package_name", ""),
                "code_snippet": claim_data.get("code_snippet", ""),
                # Ensure turn_number is available (from turn_idx in claim_eval)
                "turn_number": turn_number,
                # Ensure reason is available
                "reason": claim_eval.get("reason", ""),
                # Include search queries for debugging/analysis
                "search_queries": claim_eval.get("search_queries", []),
            }
            all_claim_evaluations.append(enriched_claim)
            
            # Track per-turn statistics (response-level, unique per conversation+turn)
            # We need to track unique (conv_id, turn) pairs and their hallucination status
            conv_id = result.get("conversation_id", 0)
            turn_key = (conv_id, turn_number)
            
            if turn_number not in per_turn_stats:
                per_turn_stats[turn_number] = {
                    "turn_keys_seen": set(),  # Track unique (conv_id, turn) pairs
                    "turn_halluc_flags": {},  # Track hallucination flags per turn key
                }
            
            # Initialize flags for this turn key if not seen
            if turn_key not in per_turn_stats[turn_number]["turn_keys_seen"]:
                per_turn_stats[turn_number]["turn_keys_seen"].add(turn_key)
                per_turn_stats[turn_number]["turn_halluc_flags"][turn_key] = {
                    "import": False,
                    "install": False,
                    "function": False,
                    "any": False,
                }
            
            # Update flags for this turn key based on current claim
            if is_hallucinated:
                per_turn_stats[turn_number]["turn_halluc_flags"][turn_key]["any"] = True
                if element_type == "import":
                    per_turn_stats[turn_number]["turn_halluc_flags"][turn_key]["import"] = True
                elif element_type == "install":
                    per_turn_stats[turn_number]["turn_halluc_flags"][turn_key]["install"] = True
                elif element_type == "function_call":
                    per_turn_stats[turn_number]["turn_halluc_flags"][turn_key]["function"] = True
    
    # Calculate rates for element types
    for etype, stats in element_type_stats.items():
        if stats["total"] > 0:
            stats["hallucination_rate"] = stats["hallucinated"] / stats["total"]
        else:
            stats["hallucination_rate"] = 0.0
    
    # Calculate per-turn statistics (response-level)
    per_turn_stats_final = {}
    for turn_number, turn_data in per_turn_stats.items():
        flags_dict = turn_data["turn_halluc_flags"]
        total_turns = len(flags_dict)
        
        halluc_turns = sum(1 for flags in flags_dict.values() if flags["any"])
        import_halluc_turns = sum(1 for flags in flags_dict.values() if flags["import"])
        install_halluc_turns = sum(1 for flags in flags_dict.values() if flags["install"])
        function_halluc_turns = sum(1 for flags in flags_dict.values() if flags["function"])
        
        per_turn_stats_final[turn_number] = {
            "total_turns": total_turns,
            "hallucinated_turns": halluc_turns,
            "import_hallucinations": import_halluc_turns,
            "install_hallucinations": install_halluc_turns,
            "function_hallucinations": function_halluc_turns,
            "hallucination_rate": halluc_turns / total_turns if total_turns > 0 else 0.0,
            "import_hallucination_rate": import_halluc_turns / total_turns if total_turns > 0 else 0.0,
            "install_hallucination_rate": install_halluc_turns / total_turns if total_turns > 0 else 0.0,
            "function_hallucination_rate": function_halluc_turns / total_turns if total_turns > 0 else 0.0,
        }
    
    # Calculate rates for each language (response-level)
    for lang, stats in language_stats.items():
        total_resp = stats.get("total_responses", 0)
        if total_resp > 0:
            stats["hallucination_rate"] = stats["hallucinated_responses"] / total_resp
            stats["import_hallucination_rate"] = stats["import_hallucinated_responses"] / total_resp
            stats["install_hallucination_rate"] = stats["install_hallucinated_responses"] / total_resp
            stats["function_hallucination_rate"] = stats["function_hallucinated_responses"] / total_resp
        else:
            stats["hallucination_rate"] = 0.0
            stats["import_hallucination_rate"] = 0.0
            stats["install_hallucination_rate"] = 0.0
            stats["function_hallucination_rate"] = 0.0
    
    # Calculate overall hallucination rate (claim-level)
    overall_hallucination_rate = hallucinated_claims / total_claims if total_claims > 0 else 0.0
    
    # Calculate response-level hallucination rates
    import_hallucination_rate = import_hallucinated_responses / total_responses if total_responses > 0 else 0.0
    install_hallucination_rate = install_hallucinated_responses / total_responses if total_responses > 0 else 0.0
    function_hallucination_rate = function_hallucinated_responses / total_responses if total_responses > 0 else 0.0
    overall_response_hallucination_rate = overall_hallucinated_responses / total_responses if total_responses > 0 else 0.0
    
    avg_score = total_score / len(results) if results else 0.0
    total_convs = len(results)
    
    # Extract hallucinated claims for easy access
    hallucinated_segments = [
        claim for claim in all_claim_evaluations 
        if claim.get("is_hallucinated", False)
    ]
    
    # Group hallucinated segments by language
    hallucinated_segments_by_language = {}
    for segment in hallucinated_segments:
        lang = segment.get("language", "Unknown")
        if lang not in hallucinated_segments_by_language:
            hallucinated_segments_by_language[lang] = []
        hallucinated_segments_by_language[lang].append(segment)
    
    # Group hallucinated segments by element type
    hallucinated_segments_by_type = {}
    for segment in hallucinated_segments:
        etype = segment.get("element_type", "unknown")
        if etype not in hallucinated_segments_by_type:
            hallucinated_segments_by_type[etype] = []
        hallucinated_segments_by_type[etype].append(segment)
    
    return {
        "score": 1.0 - overall_response_hallucination_rate,  # Score based on response-level rate
        "reasoning": f"Response hallucination rate: {overall_response_hallucination_rate:.1%} ({overall_hallucinated_responses}/{total_responses} responses)",
        "details": {
            # Response-level stats (primary metric for coding tasks)
            "total_responses": total_responses,
            "import_hallucinated_responses": import_hallucinated_responses,
            "install_hallucinated_responses": install_hallucinated_responses,
            "function_hallucinated_responses": function_hallucinated_responses,
            "overall_hallucinated_responses": overall_hallucinated_responses,
            "import_hallucination_rate": import_hallucination_rate,
            "install_hallucination_rate": install_hallucination_rate,
            "function_hallucination_rate": function_hallucination_rate,
            "overall_hallucination_rate": overall_response_hallucination_rate,
            # Claim-level stats (for reference)
            "total_claims": total_claims,
            "hallucinated_claims": hallucinated_claims,
            "claim_level_hallucination_rate": overall_hallucination_rate,
            # Conversation-level stats
            "conversations_with_import_halluc": conv_import_halluc_count,
            "conversations_with_install_halluc": conv_install_halluc_count,
            "conversations_with_function_halluc": conv_function_halluc_count,
            "total_conversations": total_convs,
            # Per-type stats (claim-level)
            "element_type_stats": element_type_stats,
            "claim_evaluations": all_claim_evaluations,
            "hallucinated_segments": hallucinated_segments,
            "hallucinated_segments_by_language": hallucinated_segments_by_language,
            "hallucinated_segments_by_type": hallucinated_segments_by_type,
            "language_stats": language_stats,
            "per_turn_stats": per_turn_stats_final,
        }
    }


def generate_reports(
    task: str,
    eval_results: List[Dict[str, Any]],
    output_dir: Path,
    conversations_path: Path = None,
    input_path: Path = None,
) -> List[Path]:
    """Generate HTML reports from evaluation results.
    
    Args:
        task: Task name (research_questions, paper_authors, coding, etc.)
        eval_results: List of evaluation results
        output_dir: Directory to save reports
        conversations_path: Optional path to original conversations file
        
    Returns:
        List of generated report file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine task type for reporter
    if task in ["research_questions", "legal_cases", "medical_guidelines"]:
        task_type = "grounding"
        aggregator = aggregate_results_for_grounding_task
    elif task == "paper_authors":
        task_type = "paper_authors"
        aggregator = aggregate_results_for_paper_authors
    elif task == "coding":
        task_type = "coding"
        aggregator = aggregate_results_for_coding
    else:
        # Try to auto-detect from first result
        if eval_results:
            first_result = eval_results[0]
            details = first_result.get("details", {})
            
            if "facts" in first_result or "facts" in details or "claim_evaluations" in details:
                # Check if it's coding (claim_evaluations with element_type) or grounding (facts)
                claim_evals = details.get("claim_evaluations", [])
                if claim_evals and "claim" in claim_evals[0] and "element_type" in claim_evals[0].get("claim", {}):
                    task_type = "coding"
                    aggregator = aggregate_results_for_coding
                elif "hallucinated_import_detected" in details:
                    task_type = "coding"
                    aggregator = aggregate_results_for_coding
                else:
                    task_type = "grounding"
                    aggregator = aggregate_results_for_grounding_task
            elif "predicted_names" in details:
                task_type = "paper_authors"
                aggregator = aggregate_results_for_paper_authors
            elif "hallucinated_import_detected" in details or "turn_evaluations" in details:
                task_type = "coding"
                aggregator = aggregate_results_for_coding
            else:
                raise ValueError(f"Unknown task type: {task}")
        else:
            raise ValueError("No evaluation results found")
    
    # Load conversations if available
    conversations = {}
    if conversations_path and conversations_path.exists():
        convs, metas = load_conversations(conversations_path)
        for conv, meta in zip(convs, metas):
            conv_id = meta.get("conversation_id", 0)
            conversations[conv_id] = conv
    
    # Extract model name for report title
    model_name = "unknown"
    if eval_results and eval_results[0].get("metadata"):
        model_name = eval_results[0]["metadata"].get("model_name", "unknown")
    
    # Detect if this is from coding_direct pipeline
    is_coding_direct = False
    if input_path and "_eval_coding_direct" in input_path.name:
        is_coding_direct = True
    elif task_type == "coding" and eval_results:
        # Also check details structure as fallback
        first_details = eval_results[0].get("details", {})
        # coding_direct has both turn_evaluations and claim_evaluations with specific structure
        if "turn_evaluations" in first_details and "claim_evaluations" in first_details:
            turn_evals = first_details.get("turn_evaluations", [])
            if turn_evals and "has_hallucination" in turn_evals[0]:
                is_coding_direct = True
    
    # Create reporter
    config = HTMLReportConfig(
        title=f"{task.replace('_', ' ').title()} - Evaluation Report",
        include_conversation=True,
        include_detailed_results=True,
        include_statistics=True,
    )
    reporter = HTMLReporter(config)
    
    generated_reports = []
    
    # Generate aggregate report
    print(f"Generating aggregate report for {len(eval_results)} conversations...")
    aggregated = aggregator(eval_results)
    
    # Add model name to aggregated results for display in executive summary
    aggregated["model_name"] = model_name
    
    # Get fact count based on task type
    if task_type == "grounding":
        num_facts = aggregated.get("total_facts", 0)
    elif task_type == "paper_authors":
        num_facts = aggregated["details"].get("predicted_count", 0)
    elif task_type == "coding":
        num_facts = aggregated["details"].get("total_claims", 0)
    else:
        num_facts = 0
    
    # Add suffix for coding_direct reports
    suffix = "_coding_direct" if is_coding_direct else ""
    output_path = output_dir / f"{task}_{model_name}_results{suffix}_aggregate_report.html"
    reporter.generate_report(
        result=aggregated,
        conversation=None,
        conversation_id="Aggregate",
        output_path=str(output_path),
        task_type=task_type,
    )
    generated_reports.append(output_path)
    print(f"[OK] Aggregate report: {output_path}")
    
    # Generate per-conversation reports (optional, for detailed analysis)
    # Uncomment if needed:
    # for result in eval_results[:10]:  # Limit to first 10 for performance
    #     conv_id = result.get("conversation_id", 0)
    #     conv = conversations.get(conv_id)
    #     
    #     output_path = output_dir / f"{task}_conversation_{conv_id}_report.html"
    #     reporter.generate_report(
    #         result=result,
    #         conversation=conv,
    #         conversation_id=str(conv_id),
    #         output_path=str(output_path),
    #         task_type=task_type,
    #     )
    #     generated_reports.append(output_path)
    
    return generated_reports


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate HTML reports from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["research_questions", "legal_cases", "medical_guidelines", "paper_authors", "coding"],
        help="Task name",
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to evaluation results JSONL file",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reports (default: task_name/reports/)",
    )
    
    parser.add_argument(
        "--conversations",
        type=str,
        default=None,
        help="Optional path to original conversations JSONL file (for including conversation history)",
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return 1
    
    # Set default output directory
    if args.output is None:
        output_dir = input_path.parent / "reports"
    else:
        output_dir = Path(args.output)
    
    # Load conversations path if not specified but default exists
    conversations_path = None
    if args.conversations:
        conversations_path = Path(args.conversations)
    else:
        # Try to find conversations file in same directory
        # First try exact name "conversations.jsonl"
        default_conv_path = input_path.parent / "conversations.jsonl"
        if default_conv_path.exists():
            conversations_path = default_conv_path
        else:
            # Look for files like conversations_{model}_{n}convs.jsonl
            conv_files = list(input_path.parent.glob("conversations_*.jsonl"))
            # Filter out eval, extraction, and cache files
            conv_files = [f for f in conv_files if "_eval" not in f.name and "_extraction" not in f.name and "_cache" not in f.name]
            if conv_files:
                # Use the first match (or most recent if multiple)
                conversations_path = conv_files[0]
    
    print("=" * 80)
    print("HTML REPORT GENERATION")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Input: {input_path}")
    print(f"Output Directory: {output_dir}")
    if conversations_path:
        print(f"Conversations: {conversations_path}")
    print("=" * 80 + "\n")
    
    # Load evaluation results
    print(f"Loading evaluation results from: {input_path}")
    eval_results = load_evaluation_results(input_path)
    print(f"[OK] Loaded {len(eval_results)} evaluation results\n")
    
    if not eval_results:
        print("[ERROR] No evaluation results found in input file")
        return 1
    
    # Generate reports
    try:
        reports = generate_reports(
            task=args.task,
            eval_results=eval_results,
            output_dir=output_dir,
            conversations_path=conversations_path,
            input_path=input_path,
        )
        
        print("\n" + "=" * 80)
        print("[OK] REPORT GENERATION COMPLETE")
        print("=" * 80)
        print(f"Generated {len(reports)} report(s):")
        for report_path in reports:
            print(f"  - {report_path}")
        print("=" * 80)
        
        return 0
    
    except Exception as e:
        print(f"\n[ERROR] Error generating reports: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

