#!/usr/bin/env python3
"""
Judge vs Judge Performance Evaluation

Compares two LLM judge evaluations against each other, treating one as the baseline.
Outputs an HTML report with metrics, confusion matrices, and disagreement analysis.

Usage:
    python evaluate_judge_vs_judge_performance.py baseline.jsonl compare.jsonl
    python evaluate_judge_vs_judge_performance.py baseline.jsonl compare.jsonl --baseline-label "GPT-4" --compare-label "Claude"
    python evaluate_judge_vs_judge_performance.py baseline.jsonl compare.jsonl --output-dir ./results
"""

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# =============================================================================
# Data Loading & Parsing
# =============================================================================

def load_evaluation_results(path: Path) -> list[dict[str, Any]]:
    """Load evaluation results from JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def parse_grounding(grounding_text: str) -> str:
    """Parse judge grounding response to yes/no/na/unknown."""
    text_lower = grounding_text.lower().strip()
    
    if text_lower.startswith("yes"):
        return "yes"
    elif text_lower.startswith("no"):
        return "no"
    elif text_lower.startswith("n/a") or "could not" in text_lower or "cannot verify" in text_lower:
        return "na"
    else:
        if "yes" in text_lower[:50]:
            return "yes"
        elif "no" in text_lower[:50]:
            return "no"
        return "unknown"


def detect_task_type(eval_results: list[dict]) -> str:
    """Detect if results are from coding or grounding task."""
    if not eval_results:
        return "grounding"
    
    # Check first result's claim evaluations for coding-specific fields
    for result in eval_results[:3]:  # Check first few results
        claim_evals = result.get("details", {}).get("claim_evaluations", [])
        for eval_item in claim_evals:
            claim_data = eval_item.get("claim", {})
            # Coding tasks have element_type in claim data
            if claim_data.get("element_type") in ["import", "install", "function_call"]:
                return "coding"
            # Also check for coding-specific flags
            if any([
                eval_item.get("hallucinated_import_detected") is not None,
                eval_item.get("hallucinated_install_detected") is not None,
                eval_item.get("hallucinated_function_usage_detected") is not None,
            ]) and claim_data.get("package_name"):
                return "coding"
    
    return "grounding"


def extract_claims(eval_results: list[dict], source_label: str, task_type: str = "grounding") -> list[dict]:
    """Extract all claims with judge annotations from evaluation results."""
    claims = []
    
    for result in eval_results:
        conversation_id = result.get("conversation_id", 0)
        metadata = result.get("metadata", {})
        claim_evaluations = result.get("details", {}).get("claim_evaluations", [])
        
        for eval_item in claim_evaluations:
            claim_data = eval_item.get("claim", {})
            
            ref_grounding_raw = eval_item.get("reference_grounding", "")
            content_grounding_raw = eval_item.get("content_grounding", "")
            hallucination = eval_item.get("hallucination", "").lower().strip()
            abstention = eval_item.get("abstention", "").lower().strip()
            verification_error = eval_item.get("verification_error", "").lower().strip()
            
            # Use claim_id as primary key since it's guaranteed unique
            claim_id = eval_item.get("claim_id", "")
            if claim_id:
                claim_key = (str(conversation_id), str(claim_id))
            else:
                claim_key = (
                    str(conversation_id),
                    str(claim_data.get("turn_number", eval_item.get("turn_idx", 0))),
                    str(claim_data.get("package_name", "") or claim_data.get("claimed_authors", "")),
                    str(claim_data.get("code_snippet", "") or claim_data.get("claimed_year", "")),
                )
            
            if task_type == "coding":
                # Coding task format
                element_type = claim_data.get("element_type", "unknown")
                package_name = claim_data.get("package_name", "")
                code_snippet = claim_data.get("code_snippet", "")
                language = claim_data.get("language", metadata.get("language", "unknown"))
                reason = eval_item.get("reason", "")
                
                display_text = package_name or eval_item.get("reference_name", "Unknown")
                display_detail = element_type.replace("_", " ").title()
                
                # Coding-specific hallucination flags
                import_halluc = eval_item.get("hallucinated_import_detected", False)
                install_halluc = eval_item.get("hallucinated_install_detected", False)
                function_halluc = eval_item.get("hallucinated_function_usage_detected", False)
                
                claims.append({
                    "claim_key": claim_key,
                    "conversation_id": conversation_id,
                    "turn_number": claim_data.get("turn_number", eval_item.get("turn_idx", 0)),
                    "claim_identifier": package_name,
                    "display_text": display_text,
                    "display_detail": display_detail,
                    "element_type": element_type,
                    "package_name": package_name,
                    "code_snippet": code_snippet,
                    "language": language,
                    "reason": reason,
                    "claimed_content": code_snippet,
                    "ref_grounding_raw": ref_grounding_raw,
                    "content_grounding_raw": content_grounding_raw,
                    "ref_grounding": parse_grounding(ref_grounding_raw),
                    "content_grounding": parse_grounding(content_grounding_raw),
                    "hallucination": hallucination,
                    "abstention": abstention,
                    "verification_error": verification_error,
                    "import_halluc": import_halluc,
                    "install_halluc": install_halluc,
                    "function_halluc": function_halluc,
                    "source": source_label,
                })
            else:
                # Grounding task format (research papers, legal cases, medical guidelines)
                claim_identifier = (
                    claim_data.get("claimed_authors") or 
                    claim_data.get("reference_name") or 
                    claim_data.get("content") or 
                    eval_item.get("reference_name") or
                    ""
                )
                claim_year = claim_data.get("claimed_year", "")
                claim_type = claim_data.get("type", "")
                
                # Build display text for claim
                if claim_data.get("claimed_authors"):
                    # Research paper format
                    display_text = claim_data.get("claimed_authors", "")
                    display_detail = claim_data.get("claimed_year", "")
                else:
                    # Legal case format
                    display_text = (
                        claim_data.get("reference_name") or 
                        claim_data.get("content") or 
                        eval_item.get("reference_name") or 
                        "Unknown"
                    )
                    display_detail = claim_data.get("type", "")
                
                # Get the best content description available
                claimed_content = (
                    claim_data.get("holding_or_description") or  # Legal cases
                    claim_data.get("claimed_content") or  # Research papers
                    claim_data.get("original_statement") or  # Fallback
                    ""
                )
                if len(claimed_content) > 2000:
                    claimed_content = claimed_content[:2000]
                
                claims.append({
                    "claim_key": claim_key,
                    "conversation_id": conversation_id,
                    "turn_number": claim_data.get("turn_number", eval_item.get("turn_idx", 0)),
                    "claim_identifier": claim_identifier,
                    "display_text": display_text,
                    "display_detail": display_detail,
                    "claimed_authors": claim_data.get("claimed_authors", ""),
                    "claimed_year": claim_data.get("claimed_year", ""),
                    "claimed_title": claim_data.get("claimed_title", ""),
                    "reference_name": claim_data.get("reference_name") or eval_item.get("reference_name", ""),
                    "claim_type": claim_type,
                    "claimed_content": claimed_content,
                    "ref_grounding_raw": ref_grounding_raw,
                    "content_grounding_raw": content_grounding_raw,
                    "ref_grounding": parse_grounding(ref_grounding_raw),
                    "content_grounding": parse_grounding(content_grounding_raw),
                    "hallucination": hallucination,
                    "abstention": abstention,
                    "verification_error": verification_error,
                    "source": source_label,
                })
    
    return claims


def merge_claims(baseline_claims: list[dict], compare_claims: list[dict], task_type: str = "grounding") -> pd.DataFrame:
    """Merge baseline and comparison claims on claim_key."""
    baseline_df = pd.DataFrame(baseline_claims)
    compare_df = pd.DataFrame(compare_claims)
    
    # Columns to rename with baseline/compare prefix
    baseline_cols = {
        "ref_grounding": "baseline_ref_grounding",
        "content_grounding": "baseline_content_grounding",
        "hallucination": "baseline_hallucination",
        "abstention": "baseline_abstention",
        "verification_error": "baseline_verification_error",
        "ref_grounding_raw": "baseline_ref_grounding_raw",
        "content_grounding_raw": "baseline_content_grounding_raw",
    }
    
    compare_cols = {
        "ref_grounding": "compare_ref_grounding",
        "content_grounding": "compare_content_grounding",
        "hallucination": "compare_hallucination",
        "abstention": "compare_abstention",
        "verification_error": "compare_verification_error",
        "ref_grounding_raw": "compare_ref_grounding_raw",
        "content_grounding_raw": "compare_content_grounding_raw",
    }
    
    # Add coding-specific columns if present
    if task_type == "coding":
        baseline_cols.update({
            "import_halluc": "baseline_import_halluc",
            "install_halluc": "baseline_install_halluc",
            "function_halluc": "baseline_function_halluc",
            "reason": "baseline_reason",
        })
        compare_cols.update({
            "import_halluc": "compare_import_halluc",
            "install_halluc": "compare_install_halluc",
            "function_halluc": "compare_function_halluc",
            "reason": "compare_reason",
        })
    
    baseline_df = baseline_df.rename(columns=baseline_cols)
    compare_df = compare_df.rename(columns=compare_cols)
    
    # Only keep necessary columns from compare_df for merge
    compare_merge_cols = ["claim_key"] + [c for c in compare_cols.values() if c in compare_df.columns]
    compare_df_merge = compare_df[compare_merge_cols]
    
    merged_df = baseline_df.merge(compare_df_merge, on="claim_key", how="inner")
    return merged_df


def filter_valid_claims(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to claims without verification errors in both judges."""
    return merged_df[
        (merged_df["baseline_verification_error"].str.lower().str.startswith("no", na=True)) &
        (merged_df["compare_verification_error"].str.lower().str.startswith("no", na=True))
    ].copy()


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(y_true: list, y_pred: list, positive_label: str = "yes") -> dict:
    """Compute classification metrics treating y_true as ground truth."""
    valid_mask = [(t in ["yes", "no"]) and (p in ["yes", "no"]) for t, p in zip(y_true, y_pred)]
    y_true_valid = [y_true[i] for i in range(len(y_true)) if valid_mask[i]]
    y_pred_valid = [y_pred[i] for i in range(len(y_pred)) if valid_mask[i]]
    
    if len(y_true_valid) == 0:
        return {"error": "No valid predictions to evaluate"}
    
    y_true_bin = [1 if y == positive_label else 0 for y in y_true_valid]
    y_pred_bin = [1 if y == positive_label else 0 for y in y_pred_valid]
    
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    
    return {
        "n_samples": len(y_true_valid),
        "n_excluded": len(y_true) - len(y_true_valid),
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }


def compute_agreement(y1: list, y2: list) -> dict:
    """Compute inter-rater agreement metrics."""
    valid_mask = [(a in ["yes", "no"]) and (b in ["yes", "no"]) for a, b in zip(y1, y2)]
    y1_valid = [y1[i] for i in range(len(y1)) if valid_mask[i]]
    y2_valid = [y2[i] for i in range(len(y2)) if valid_mask[i]]
    
    if len(y1_valid) < 2:
        return {"error": "Not enough valid samples for agreement metrics"}
    
    agreement = sum(a == b for a, b in zip(y1_valid, y2_valid)) / len(y1_valid)
    
    try:
        kappa = cohen_kappa_score(y1_valid, y2_valid)
    except Exception:
        kappa = float('nan')
    
    return {
        "n_samples": len(y1_valid),
        "n_excluded": len(y1) - len(y1_valid),
        "agreement": agreement,
        "cohen_kappa": kappa,
    }


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value."""
    if kappa < 0:
        return "Less than chance agreement"
    elif kappa < 0.20:
        return "Slight agreement"
    elif kappa < 0.40:
        return "Fair agreement"
    elif kappa < 0.60:
        return "Moderate agreement"
    elif kappa < 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


# =============================================================================
# HTML Report Generation
# =============================================================================

def get_html_css() -> str:
    """Return CSS styles for the HTML report."""
    return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        
        header {
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #007bff;
            font-size: 2.2em;
            margin-bottom: 10px;
        }
        
        h2 {
            color: #495057;
            font-size: 1.5em;
            margin-top: 30px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e9ecef;
        }
        
        h3 {
            color: #6c757d;
            font-size: 1.2em;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        .metadata {
            color: #666;
            font-size: 0.9em;
        }
        
        .summary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .summary h2 {
            color: white;
            border-bottom: none;
            margin-top: 0;
        }
        
        .summary p {
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stat-card.good {
            border-color: #28a745;
            background-color: #d4edda;
        }
        
        .stat-card.warning {
            border-color: #ffc107;
            background-color: #fff3cd;
        }
        
        .stat-card.danger {
            border-color: #dc3545;
            background-color: #f8d7da;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.85em;
            margin-top: 5px;
        }
        
        .metrics-section {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .metric-box {
            background: white;
            border-radius: 6px;
            padding: 12px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #007bff;
        }
        
        .metric-label {
            font-size: 0.8em;
            color: #6c757d;
        }
        
        .confusion-matrix {
            margin: 15px 0;
        }
        
        .confusion-matrix table {
            border-collapse: collapse;
            margin: 0 auto;
        }
        
        .confusion-matrix th, .confusion-matrix td {
            padding: 12px 20px;
            text-align: center;
            border: 1px solid #dee2e6;
        }
        
        .confusion-matrix th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        .confusion-matrix .header-cell {
            background-color: #e9ecef;
            font-weight: bold;
        }
        
        .confusion-matrix .diagonal {
            background-color: #d4edda;
            font-weight: bold;
        }
        
        .confusion-matrix .off-diagonal {
            background-color: #f8d7da;
        }
        
        .confusion-matrix .label-cell {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        .agreement-box {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            border: 1px solid #dee2e6;
        }
        
        .agreement-box .kappa-value {
            font-size: 1.3em;
            font-weight: bold;
        }
        
        .agreement-box .kappa-interp {
            color: #6c757d;
            font-style: italic;
        }
        
        .disagreements-section {
            margin-top: 20px;
        }
        
        .disagreement-item {
            background: white;
            border: 1px solid #e9ecef;
            border-left: 4px solid #dc3545;
            border-radius: 4px;
            padding: 12px 15px;
            margin-bottom: 10px;
        }
        
        .disagreement-item .claim-ref {
            font-weight: 600;
            color: #495057;
        }
        
        .disagreement-item .comparison {
            margin-top: 5px;
            font-size: 0.95em;
        }
        
        .disagreement-item .baseline {
            color: #007bff;
        }
        
        .disagreement-item .compare {
            color: #28a745;
        }
        
        .disagreement-item .detail {
            color: #6c757d;
            font-size: 0.85em;
            margin-top: 3px;
        }
        
        .disagreement-item .claim-content {
            background: #f8f9fa;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 4px;
            font-size: 0.9em;
            color: #495057;
            font-style: italic;
            border-left: 3px solid #dee2e6;
        }
        
        .justifications {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #e9ecef;
        }
        
        .justification {
            background: #f8f9fa;
            border-radius: 6px;
            padding: 10px 12px;
            font-size: 0.85em;
        }
        
        .justification-label {
            font-weight: 600;
            margin-bottom: 6px;
            font-size: 0.9em;
        }
        
        .baseline-justification {
            border-left: 3px solid #007bff;
        }
        
        .baseline-justification .justification-label {
            color: #007bff;
        }
        
        .compare-justification {
            border-left: 3px solid #28a745;
        }
        
        .compare-justification .justification-label {
            color: #28a745;
        }
        
        .justification-text {
            color: #495057;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }
        
        .outcome-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .outcome-box {
            background: white;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #dee2e6;
        }
        
        .outcome-box h4 {
            margin-bottom: 10px;
            color: #495057;
        }
        
        .outcome-list {
            list-style: none;
        }
        
        .outcome-list li {
            padding: 5px 0;
            border-bottom: 1px solid #f1f1f1;
        }
        
        .outcome-list li:last-child {
            border-bottom: none;
        }
        
        .match-summary {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
        }
        
        .match-summary h3 {
            color: white;
            margin-top: 0;
        }
        
        .match-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-top: 15px;
        }
        
        .match-item {
            text-align: center;
        }
        
        .match-value {
            font-size: 1.8em;
            font-weight: bold;
        }
        
        .match-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .collapsible {
            cursor: pointer;
            padding: 10px;
            background: #f8f9fa;
            border: none;
            width: 100%;
            text-align: left;
            font-size: 1em;
            border-radius: 4px;
            margin-top: 10px;
        }
        
        .collapsible:hover {
            background: #e9ecef;
        }
        
        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        
        .collapsible-content.show {
            max-height: 2000px;
        }
    """


def generate_confusion_matrix_html(cm: list, baseline_label: str, compare_label: str) -> str:
    """Generate HTML for a confusion matrix table."""
    return f"""
    <div class="confusion-matrix">
        <table>
            <tr>
                <th class="header-cell" rowspan="2" colspan="2">Confusion Matrix</th>
                <th class="header-cell" colspan="2">Predicted ({compare_label})</th>
            </tr>
            <tr>
                <th>No</th>
                <th>Yes</th>
            </tr>
            <tr>
                <th class="label-cell" rowspan="2">Actual<br>({baseline_label})</th>
                <th class="label-cell">No</th>
                <td class="diagonal">{cm[0][0]}</td>
                <td class="off-diagonal">{cm[0][1]}</td>
            </tr>
            <tr>
                <th class="label-cell">Yes</th>
                <td class="off-diagonal">{cm[1][0]}</td>
                <td class="diagonal">{cm[1][1]}</td>
            </tr>
        </table>
    </div>
    """


def generate_metrics_section_html(title: str, metrics: dict, agreement: dict, 
                                   baseline_label: str, compare_label: str,
                                   disagreements_html: str = "") -> str:
    """Generate HTML for a metrics section."""
    if "error" in metrics:
        return f"""
        <div class="metrics-section">
            <h3>{title}</h3>
            <p>Error: {metrics['error']}</p>
        </div>
        """
    
    cm = metrics["confusion_matrix"]
    kappa = agreement.get("cohen_kappa", float('nan'))
    kappa_interp = interpret_kappa(kappa) if not pd.isna(kappa) else "N/A"
    
    return f"""
    <div class="metrics-section">
        <h3>{title}</h3>
        <p style="color: #6c757d; margin-bottom: 15px;">
            Samples evaluated: {metrics['n_samples']} (excluded: {metrics['n_excluded']})
        </p>
        
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value">{metrics['accuracy']:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics['precision']:.1%}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics['recall']:.1%}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{metrics['f1']:.1%}</div>
                <div class="metric-label">F1 Score</div>
            </div>
        </div>
        
        {generate_confusion_matrix_html(cm, baseline_label, compare_label)}
        
        <div class="agreement-box">
            <strong>Inter-Judge Agreement:</strong>
            <span class="kappa-value">{agreement.get('agreement', 0):.1%}</span> simple agreement,
            Cohen's κ = <span class="kappa-value">{kappa:.3f}</span>
            <span class="kappa-interp">({kappa_interp})</span>
        </div>
        
        {disagreements_html}
    </div>
    """


def generate_disagreements_html(df: pd.DataFrame, col1: str, col2: str, 
                                 baseline_label: str, compare_label: str,
                                 max_show: int = 20, include_detail: bool = False,
                                 task_type: str = "grounding") -> str:
    """Generate HTML for disagreement cases."""
    disagreements = df[df[col1] != df[col2]]
    if len(disagreements) == 0:
        return ""
    
    items_html = ""
    for _, row in disagreements.head(max_show).iterrows():
        detail_html = ""
        if include_detail and task_type == "grounding":
            detail_html = f"""
            <div class="detail">
                Ref: {baseline_label}={row.get('baseline_ref_grounding', 'N/A')}/{compare_label}={row.get('compare_ref_grounding', 'N/A')},
                Content: {baseline_label}={row.get('baseline_content_grounding', 'N/A')}/{compare_label}={row.get('compare_content_grounding', 'N/A')}
            </div>
            """
        elif include_detail and task_type == "coding":
            element_type = row.get('element_type', 'unknown')
            language = row.get('language', 'unknown')
            detail_html = f"""
            <div class="detail">
                Element: {element_type.replace('_', ' ').title()} | Language: {language}
            </div>
            """
        
        if task_type == "coding":
            # Coding-specific display
            package_name = row.get('package_name', '') or row.get('display_text', '') or 'Unknown'
            element_type = row.get('element_type', 'unknown')
            code_snippet = row.get('code_snippet', '') or row.get('claimed_content', '')
            language = row.get('language', 'unknown')
            
            claim_display = f"{html.escape(str(package_name))} ({element_type.replace('_', ' ').title()})"
            
            content_html = ""
            if code_snippet:
                snippet_escaped = html.escape(str(code_snippet)[:500])
                content_html = f'<div class="claim-content"><code>{snippet_escaped}</code></div>'
            
            # Get reasons from both judges
            baseline_reason = str(row.get('baseline_reason', '') or '')
            compare_reason = str(row.get('compare_reason', '') or '')
            
            justification_html = ""
            if baseline_reason or compare_reason:
                baseline_display = html.escape(baseline_reason[:1000] + "..." if len(baseline_reason) > 1000 else baseline_reason)
                compare_display = html.escape(compare_reason[:1000] + "..." if len(compare_reason) > 1000 else compare_reason)
                baseline_display = baseline_display.replace('\n', '<br>')
                compare_display = compare_display.replace('\n', '<br>')
                
                justification_html = f"""
                <div class="justifications">
                    <div class="justification baseline-justification">
                        <div class="justification-label">{baseline_label} Reason:</div>
                        <div class="justification-text">{baseline_display or 'N/A'}</div>
                    </div>
                    <div class="justification compare-justification">
                        <div class="justification-label">{compare_label} Reason:</div>
                        <div class="justification-text">{compare_display or 'N/A'}</div>
                    </div>
                </div>
                """
        else:
            # Grounding task display
            display_text = row.get('display_text', '') or row.get('claimed_authors', '') or row.get('reference_name', '') or 'Unknown'
            display_detail = row.get('display_detail', '') or row.get('claimed_year', '') or row.get('claim_type', '')
            
            if display_detail:
                claim_display = f"{html.escape(str(display_text))} ({html.escape(str(display_detail))})"
            else:
                claim_display = html.escape(str(display_text))
            
            claim_content = row.get('claimed_content', '') or ''
            claim_content = str(claim_content).strip()
            content_html = ""
            if claim_content:
                if len(claim_content) > 2000:
                    claim_content = claim_content[:2000] + "..."
                content_html = f'<div class="claim-content">{html.escape(claim_content)}</div>'
            
            # Get justifications from both judges
            baseline_justification = ""
            compare_justification = ""
            
            if "ref_grounding" in col1:
                baseline_justification = str(row.get('baseline_ref_grounding_raw', '') or '')
                compare_justification = str(row.get('compare_ref_grounding_raw', '') or '')
            elif "content_grounding" in col1:
                baseline_justification = str(row.get('baseline_content_grounding_raw', '') or '')
                compare_justification = str(row.get('compare_content_grounding_raw', '') or '')
            elif "hallucination" in col1:
                baseline_ref = str(row.get('baseline_ref_grounding_raw', '') or '')
                baseline_content = str(row.get('baseline_content_grounding_raw', '') or '')
                compare_ref = str(row.get('compare_ref_grounding_raw', '') or '')
                compare_content = str(row.get('compare_content_grounding_raw', '') or '')
                
                baseline_justification = ""
                if baseline_ref:
                    baseline_justification += f"Reference: {baseline_ref}"
                if baseline_content:
                    baseline_justification += f"\n\nContent: {baseline_content}"
                
                compare_justification = ""
                if compare_ref:
                    compare_justification += f"Reference: {compare_ref}"
                if compare_content:
                    compare_justification += f"\n\nContent: {compare_content}"
            
            justification_html = ""
            if baseline_justification or compare_justification:
                baseline_just_display = html.escape(baseline_justification[:1000] + "..." if len(baseline_justification) > 1000 else baseline_justification)
                compare_just_display = html.escape(compare_justification[:1000] + "..." if len(compare_justification) > 1000 else compare_justification)
                baseline_just_display = baseline_just_display.replace('\n', '<br>')
                compare_just_display = compare_just_display.replace('\n', '<br>')
                
                justification_html = f"""
                <div class="justifications">
                    <div class="justification baseline-justification">
                        <div class="justification-label">{baseline_label} Justification:</div>
                        <div class="justification-text">{baseline_just_display}</div>
                    </div>
                    <div class="justification compare-justification">
                        <div class="justification-label">{compare_label} Justification:</div>
                        <div class="justification-text">{compare_just_display}</div>
                    </div>
                </div>
                """
        
        items_html += f"""
        <div class="disagreement-item">
            <div class="claim-ref">{claim_display}</div>
            {content_html}
            <div class="comparison">
                <span class="baseline">{baseline_label}: {row[col1]}</span> vs 
                <span class="compare">{compare_label}: {row[col2]}</span>
            </div>
            {detail_html}
            {justification_html}
        </div>
        """
    
    more_text = f" (showing first {max_show})" if len(disagreements) > max_show else ""
    
    return f"""
    <div class="disagreements-section">
        <button class="collapsible" onclick="this.nextElementSibling.classList.toggle('show')">
            ▶ Show Disagreements ({len(disagreements)} cases{more_text})
        </button>
        <div class="collapsible-content">
            {items_html}
        </div>
    </div>
    """


def generate_coding_html_report(claims_df: pd.DataFrame, merged_count: int,
                                baseline_label: str, compare_label: str,
                                baseline_path: Path, compare_path: Path,
                                hall_metrics: dict, hall_agreement: dict,
                                hall_disagreements_html: str, hall_match: float,
                                max_disagreements: int = 20) -> str:
    """Generate HTML report for coding task comparisons."""
    
    # Compute per-element-type metrics
    element_types = ["import", "install", "function_call"]
    element_metrics_html = ""
    
    for etype in element_types:
        etype_df = claims_df[claims_df.get("element_type", pd.Series()) == etype]
        if len(etype_df) == 0:
            continue
        
        etype_valid = etype_df[
            (etype_df["baseline_hallucination"].isin(["yes", "no"])) &
            (etype_df["compare_hallucination"].isin(["yes", "no"]))
        ]
        
        if len(etype_valid) == 0:
            continue
        
        etype_metrics = compute_metrics(
            etype_valid["baseline_hallucination"].tolist(),
            etype_valid["compare_hallucination"].tolist(),
        )
        etype_agreement = compute_agreement(
            etype_valid["baseline_hallucination"].tolist(),
            etype_valid["compare_hallucination"].tolist(),
        )
        
        etype_display = etype.replace('_', ' ').title()
        baseline_yes = (etype_valid['baseline_hallucination'] == 'yes').sum()
        compare_yes = (etype_valid['compare_hallucination'] == 'yes').sum()
        
        if "error" not in etype_metrics:
            element_metrics_html += f"""
            <div class="metrics-section">
                <h3>{etype_display} Claims ({len(etype_valid)} claims)</h3>
                <p style="color: #6c757d; margin-bottom: 10px;">
                    {baseline_label} found {baseline_yes} hallucinations | {compare_label} found {compare_yes} hallucinations
                </p>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-value">{etype_metrics['accuracy']:.1%}</div>
                        <div class="metric-label">Agreement</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{etype_agreement.get('cohen_kappa', 0):.3f}</div>
                        <div class="metric-label">Cohen's κ</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{etype_metrics['f1']:.1%}</div>
                        <div class="metric-label">F1 Score</div>
                    </div>
                </div>
            </div>
            """
    
    # Compute per-language metrics if available
    language_metrics_html = ""
    if "language" in claims_df.columns:
        languages = claims_df["language"].dropna().unique()
        for lang in languages:
            lang_df = claims_df[claims_df["language"] == lang]
            lang_valid = lang_df[
                (lang_df["baseline_hallucination"].isin(["yes", "no"])) &
                (lang_df["compare_hallucination"].isin(["yes", "no"]))
            ]
            
            if len(lang_valid) < 5:  # Skip languages with too few samples
                continue
            
            lang_agreement = compute_agreement(
                lang_valid["baseline_hallucination"].tolist(),
                lang_valid["compare_hallucination"].tolist(),
            )
            
            if "error" not in lang_agreement:
                language_metrics_html += f"""
                <div class="stat-card {'good' if lang_agreement.get('agreement', 0) > 0.8 else 'warning' if lang_agreement.get('agreement', 0) > 0.6 else 'danger'}">
                    <div class="stat-number">{lang_agreement.get('agreement', 0):.1%}</div>
                    <div class="stat-label">{lang.title()} ({len(lang_valid)})</div>
                </div>
                """
    
    language_section = ""
    if language_metrics_html:
        language_section = f"""
        <h2>Agreement by Language</h2>
        <div class="stats-grid">
            {language_metrics_html}
        </div>
        """
    
    # Count hallucinations by each judge
    baseline_yes = (claims_df['baseline_hallucination'] == 'yes').sum()
    baseline_no = (claims_df['baseline_hallucination'] == 'no').sum()
    compare_yes = (claims_df['compare_hallucination'] == 'yes').sum()
    compare_no = (claims_df['compare_hallucination'] == 'no').sum()
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coding Judge Comparison - {baseline_label} vs {compare_label}</title>
    <style>
        {get_html_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Coding Judge Comparison</h1>
            <p class="metadata">
                <strong>Baseline:</strong> {baseline_label} ({baseline_path.name})<br>
                <strong>Comparison:</strong> {compare_label} ({compare_path.name})<br>
                <strong>Task Type:</strong> Coding (imports, installs, function calls)<br>
                <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </p>
        </header>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Comparing <strong>{compare_label}</strong> against <strong>{baseline_label}</strong> (baseline) on {len(claims_df)} coding claims.</p>
            <p>Overall hallucination detection agreement: <strong>{hall_agreement.get('agreement', 0):.1%}</strong></p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{merged_count}</div>
                <div class="stat-label">Total Matched Claims</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(claims_df)}</div>
                <div class="stat-label">Valid Claims</div>
            </div>
            <div class="stat-card {'good' if hall_agreement.get('agreement', 0) > 0.8 else 'warning' if hall_agreement.get('agreement', 0) > 0.6 else 'danger'}">
                <div class="stat-number">{hall_agreement.get('agreement', 0):.1%}</div>
                <div class="stat-label">Hallucination Agreement</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{hall_agreement.get('cohen_kappa', 0):.3f}</div>
                <div class="stat-label">Cohen's Kappa</div>
            </div>
        </div>
        
        <h2>Hallucination Detection by Judge</h2>
        <div class="outcome-grid">
            <div class="outcome-box">
                <h4>{baseline_label}</h4>
                <ul class="outcome-list">
                    <li>Hallucination=Yes: <strong>{baseline_yes}</strong> ({baseline_yes/len(claims_df)*100:.1f}%)</li>
                    <li>Hallucination=No: <strong>{baseline_no}</strong> ({baseline_no/len(claims_df)*100:.1f}%)</li>
                </ul>
            </div>
            <div class="outcome-box">
                <h4>{compare_label}</h4>
                <ul class="outcome-list">
                    <li>Hallucination=Yes: <strong>{compare_yes}</strong> ({compare_yes/len(claims_df)*100:.1f}%)</li>
                    <li>Hallucination=No: <strong>{compare_no}</strong> ({compare_no/len(claims_df)*100:.1f}%)</li>
                </ul>
            </div>
        </div>
        
        <h2>Overall Hallucination Detection Metrics</h2>
        {generate_metrics_section_html(
            "Hallucination Detection (Did the judges agree on hallucinations?)",
            hall_metrics, hall_agreement, baseline_label, compare_label, hall_disagreements_html
        )}
        
        <h2>Metrics by Element Type</h2>
        {element_metrics_html if element_metrics_html else '<p>No element type breakdown available.</p>'}
        
        {language_section}
        
        <div class="match-summary">
            <h3>Overall Match Summary</h3>
            <div class="match-grid">
                <div class="match-item">
                    <div class="match-value">{hall_match:.1%}</div>
                    <div class="match-label">Hallucination Agreement</div>
                </div>
                <div class="match-item">
                    <div class="match-value">{hall_agreement.get('cohen_kappa', 0):.3f}</div>
                    <div class="match-label">Cohen's Kappa</div>
                </div>
                <div class="match-item">
                    <div class="match-value">{interpret_kappa(hall_agreement.get('cohen_kappa', 0))}</div>
                    <div class="match-label">Agreement Level</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


def generate_html_report(claims_df: pd.DataFrame, merged_count: int,
                         baseline_label: str, compare_label: str,
                         baseline_path: Path, compare_path: Path,
                         max_disagreements: int = 20,
                         task_type: str = "grounding") -> str:
    """Generate the full HTML report."""
    
    # Filter for valid hallucination comparisons
    hall_valid = claims_df[
        (claims_df["baseline_hallucination"].isin(["yes", "no"])) &
        (claims_df["compare_hallucination"].isin(["yes", "no"]))
    ]
    
    # Hallucination detection metrics
    hall_metrics = compute_metrics(
        hall_valid["baseline_hallucination"].tolist(),
        hall_valid["compare_hallucination"].tolist(),
    ) if len(hall_valid) > 0 else {"error": "No data"}
    hall_agreement = compute_agreement(
        hall_valid["baseline_hallucination"].tolist(),
        hall_valid["compare_hallucination"].tolist(),
    ) if len(hall_valid) > 0 else {"error": "No data"}
    hall_disagreements_html = generate_disagreements_html(
        hall_valid, "baseline_hallucination", "compare_hallucination",
        baseline_label, compare_label, max_disagreements, include_detail=True,
        task_type=task_type
    ) if len(hall_valid) > 0 else ""
    
    hall_match = (claims_df["baseline_hallucination"] == claims_df["compare_hallucination"]).mean()
    
    if task_type == "coding":
        # Coding-specific report
        return generate_coding_html_report(
            claims_df, merged_count, baseline_label, compare_label,
            baseline_path, compare_path, hall_metrics, hall_agreement,
            hall_disagreements_html, hall_match, max_disagreements
        )
    
    # Grounding task report
    ref_valid = claims_df[
        (claims_df["baseline_ref_grounding"].isin(["yes", "no"])) |
        (claims_df["compare_ref_grounding"].isin(["yes", "no"]))
    ]
    content_valid = claims_df[
        (claims_df["baseline_content_grounding"].isin(["yes", "no"])) |
        (claims_df["compare_content_grounding"].isin(["yes", "no"]))
    ]
    
    # Reference grounding
    ref_metrics = compute_metrics(
        ref_valid["baseline_ref_grounding"].tolist(),
        ref_valid["compare_ref_grounding"].tolist(),
    ) if len(ref_valid) > 0 else {"error": "No data"}
    ref_agreement = compute_agreement(
        ref_valid["baseline_ref_grounding"].tolist(),
        ref_valid["compare_ref_grounding"].tolist(),
    ) if len(ref_valid) > 0 else {"error": "No data"}
    ref_disagreements_html = generate_disagreements_html(
        ref_valid, "baseline_ref_grounding", "compare_ref_grounding",
        baseline_label, compare_label, max_disagreements,
        task_type=task_type
    ) if len(ref_valid) > 0 else ""
    
    # Content grounding
    content_metrics = compute_metrics(
        content_valid["baseline_content_grounding"].tolist(),
        content_valid["compare_content_grounding"].tolist(),
    ) if len(content_valid) > 0 else {"error": "No data"}
    content_agreement = compute_agreement(
        content_valid["baseline_content_grounding"].tolist(),
        content_valid["compare_content_grounding"].tolist(),
    ) if len(content_valid) > 0 else {"error": "No data"}
    content_disagreements_html = generate_disagreements_html(
        content_valid, "baseline_content_grounding", "compare_content_grounding",
        baseline_label, compare_label, max_disagreements,
        task_type=task_type
    ) if len(content_valid) > 0 else ""
    
    # Match summary
    ref_match = (claims_df["baseline_ref_grounding"] == claims_df["compare_ref_grounding"]).mean()
    content_match = (claims_df["baseline_content_grounding"] == claims_df["compare_content_grounding"]).mean()
    
    # Generate HTML
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Judge vs Judge Comparison - {baseline_label} vs {compare_label}</title>
    <style>
        {get_html_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Judge vs Judge Comparison</h1>
            <p class="metadata">
                <strong>Baseline:</strong> {baseline_label} ({baseline_path.name})<br>
                <strong>Comparison:</strong> {compare_label} ({compare_path.name})<br>
                <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </p>
        </header>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Comparing <strong>{compare_label}</strong> against <strong>{baseline_label}</strong> (baseline) on {len(claims_df)} matched claims.</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{merged_count}</div>
                <div class="stat-label">Total Matched Claims</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(claims_df)}</div>
                <div class="stat-label">Valid Claims (no errors)</div>
            </div>
            <div class="stat-card {'good' if hall_agreement.get('agreement', 0) > 0.8 else 'warning' if hall_agreement.get('agreement', 0) > 0.6 else 'danger'}">
                <div class="stat-number">{hall_agreement.get('agreement', 0):.1%}</div>
                <div class="stat-label">Hallucination Agreement</div>
            </div>
        </div>
        
        <h2>Outcome Distribution</h2>
        <div class="outcome-grid">
            <div class="outcome-box">
                <h4>{baseline_label}</h4>
                <ul class="outcome-list">
                    <li>Hallucination=Yes: <strong>{(claims_df['baseline_hallucination'] == 'yes').sum()}</strong></li>
                    <li>Hallucination=No: <strong>{(claims_df['baseline_hallucination'] == 'no').sum()}</strong></li>
                    <li>Abstention=Yes: <strong>{(claims_df['baseline_abstention'] == 'yes').sum()}</strong></li>
                </ul>
            </div>
            <div class="outcome-box">
                <h4>{compare_label}</h4>
                <ul class="outcome-list">
                    <li>Hallucination=Yes: <strong>{(claims_df['compare_hallucination'] == 'yes').sum()}</strong></li>
                    <li>Hallucination=No: <strong>{(claims_df['compare_hallucination'] == 'no').sum()}</strong></li>
                    <li>Abstention=Yes: <strong>{(claims_df['compare_abstention'] == 'yes').sum()}</strong></li>
                </ul>
            </div>
        </div>
        
        <h2>Detailed Metrics</h2>
        
        {generate_metrics_section_html(
            "Reference Grounding (Does the cited reference exist?)",
            ref_metrics, ref_agreement, baseline_label, compare_label, ref_disagreements_html
        )}
        
        {generate_metrics_section_html(
            "Content Grounding (Does the content match the source?)",
            content_metrics, content_agreement, baseline_label, compare_label, content_disagreements_html
        )}
        
        {generate_metrics_section_html(
            "Hallucination Detection (Overall)",
            hall_metrics, hall_agreement, baseline_label, compare_label, hall_disagreements_html
        )}
        
        <div class="match-summary">
            <h3>Overall Match Summary</h3>
            <div class="match-grid">
                <div class="match-item">
                    <div class="match-value">{ref_match:.1%}</div>
                    <div class="match-label">Reference Grounding</div>
                </div>
                <div class="match-item">
                    <div class="match-value">{content_match:.1%}</div>
                    <div class="match-label">Content Grounding</div>
                </div>
                <div class="match-item">
                    <div class="match-value">{hall_match:.1%}</div>
                    <div class="match-label">Hallucination Detection</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


# =============================================================================
# Console Output Functions (kept for CLI feedback)
# =============================================================================

def print_progress(message: str):
    """Print progress message to console."""
    print(f"  {message}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare two LLM judge evaluations and generate an HTML report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_judge_vs_judge_performance.py baseline.jsonl compare.jsonl
  python evaluate_judge_vs_judge_performance.py baseline.jsonl compare.jsonl --baseline-label "GPT-4" --compare-label "Claude"
  python evaluate_judge_vs_judge_performance.py baseline.jsonl compare.jsonl --output-dir ./results
        """
    )
    parser.add_argument("baseline", type=Path, help="Path to baseline evaluation JSONL file")
    parser.add_argument("compare", type=Path, help="Path to comparison evaluation JSONL file")
    parser.add_argument("--baseline-label", default="Baseline", help="Label for baseline judge (default: Baseline)")
    parser.add_argument("--compare-label", default="Compare", help="Label for comparison judge (default: Compare)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: same as baseline)")
    parser.add_argument("--max-disagreements", type=int, default=20, help="Max disagreements to show per category (default: 20)")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.baseline.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📊 Judge vs Judge Comparison")
    print(f"{'='*50}")
    print(f"Baseline: {args.baseline_label} ({args.baseline})")
    print(f"Compare:  {args.compare_label} ({args.compare})")
    print()
    
    print("Loading data...")
    baseline_results = load_evaluation_results(args.baseline)
    compare_results = load_evaluation_results(args.compare)
    print_progress(f"Loaded {len(baseline_results)} baseline results")
    print_progress(f"Loaded {len(compare_results)} comparison results")
    
    # Detect task type
    task_type = detect_task_type(baseline_results)
    print_progress(f"Detected task type: {task_type}")
    
    # Extract and merge claims
    print("\nProcessing claims...")
    baseline_claims = extract_claims(baseline_results, args.baseline_label, task_type)
    compare_claims = extract_claims(compare_results, args.compare_label, task_type)
    print_progress(f"Extracted {len(baseline_claims)} baseline claims")
    print_progress(f"Extracted {len(compare_claims)} comparison claims")
    
    merged_df = merge_claims(baseline_claims, compare_claims, task_type)
    print_progress(f"Matched {len(merged_df)} claims")
    
    claims_df = filter_valid_claims(merged_df)
    print_progress(f"Valid claims (no errors): {len(claims_df)}")
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    html_content = generate_html_report(
        claims_df=claims_df,
        merged_count=len(merged_df),
        baseline_label=args.baseline_label,
        compare_label=args.compare_label,
        baseline_path=args.baseline,
        compare_path=args.compare,
        max_disagreements=args.max_disagreements,
        task_type=task_type,
    )
    
    # Save HTML report
    output_filename = f"judge_comparison_{args.baseline_label.lower()}_{args.compare_label.lower()}.html"
    output_path = output_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n✅ Report saved to: {output_path}")
    
    # Quick summary
    hall_valid = claims_df[
        (claims_df["baseline_hallucination"].isin(["yes", "no"])) &
        (claims_df["compare_hallucination"].isin(["yes", "no"]))
    ]
    if len(hall_valid) > 0:
        agreement = compute_agreement(
            hall_valid["baseline_hallucination"].tolist(),
            hall_valid["compare_hallucination"].tolist(),
        )
        print(f"\n📈 Quick Summary:")
        print(f"   Hallucination Agreement: {agreement.get('agreement', 0):.1%}")
        print(f"   Cohen's Kappa: {agreement.get('cohen_kappa', 0):.3f} ({interpret_kappa(agreement.get('cohen_kappa', 0))})")


if __name__ == "__main__":
    main()
