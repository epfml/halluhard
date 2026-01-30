#!/usr/bin/env python3
"""
Judge vs Human Performance Evaluation

Compares automated judge annotations against human ground truth annotations.
Outputs an HTML report with metrics, confusion matrices, and detailed comparisons.

Usage:
    python evaluate_judge_vs_human_performance.py eval_results.jsonl
    python evaluate_judge_vs_human_performance.py eval_results.jsonl --output-dir ./results
    python evaluate_judge_vs_human_performance.py eval_results.jsonl --max-examples 50
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


def extract_claims_with_annotations(eval_results: list[dict]) -> list[dict]:
    """Extract all claims with both human and judge annotations."""
    claims = []
    
    for result in eval_results:
        conversation_id = result.get("conversation_id", 0)
        claim_evaluations = result.get("details", {}).get("claim_evaluations", [])
        
        for eval_item in claim_evaluations:
            claim_data = eval_item.get("claim", {})
            
            # Extract human annotations
            human_ref_grounding = claim_data.get("human_reference_grounding", "").lower().strip()
            human_content_grounding = claim_data.get("human_content_grounding", "").lower().strip()
            human_comment = claim_data.get("human_comment", "")
            
            # Extract judge annotations
            judge_ref_grounding_raw = eval_item.get("reference_grounding", "")
            judge_content_grounding_raw = eval_item.get("content_grounding", "")
            judge_hallucination = eval_item.get("hallucination", "").lower().strip()
            judge_abstention = eval_item.get("abstention", "").lower().strip()
            judge_verification_error = eval_item.get("verification_error", "").lower().strip()
            
            # Get claim identifier - support multiple formats
            claim_identifier = (
                claim_data.get("claimed_authors") or 
                claim_data.get("reference_name") or 
                claim_data.get("content") or 
                eval_item.get("reference_name") or
                ""
            )
            claim_year = claim_data.get("claimed_year", "")
            claim_type = claim_data.get("type", "")
            
            # Get claim content
            claimed_content = (
                claim_data.get("holding_or_description") or
                claim_data.get("claimed_content") or
                claim_data.get("original_statement") or
                ""
            )
            if len(claimed_content) > 500:
                claimed_content = claimed_content[:500]
            
            claims.append({
                "conversation_id": conversation_id,
                "turn_number": claim_data.get("turn_number", eval_item.get("turn_idx", 0)),
                "claim_identifier": claim_identifier,
                "claimed_authors": claim_data.get("claimed_authors", ""),
                "claimed_year": claim_year,
                "claimed_title": claim_data.get("claimed_title", ""),
                "claim_type": claim_type,
                "claimed_content": claimed_content,
                # Human annotations
                "human_ref_grounding": human_ref_grounding,
                "human_content_grounding": human_content_grounding,
                "human_comment": human_comment,
                # Judge annotations (raw)
                "judge_ref_grounding_raw": judge_ref_grounding_raw,
                "judge_content_grounding_raw": judge_content_grounding_raw,
                # Judge annotations (parsed)
                "judge_ref_grounding": parse_grounding(judge_ref_grounding_raw),
                "judge_content_grounding": parse_grounding(judge_content_grounding_raw),
                "judge_hallucination": judge_hallucination,
                "judge_abstention": judge_abstention,
                "judge_verification_error": judge_verification_error,
            })
    
    return claims


def compute_human_hallucination(row: pd.Series) -> str:
    """Compute hallucination label from human reference and content grounding."""
    ref = row.get("human_ref_grounding", "")
    content = row.get("human_content_grounding", "")
    
    # If reference doesn't exist -> hallucination
    if ref == "no":
        return "yes"
    # If content doesn't match -> hallucination  
    if content == "no":
        return "yes"
    # If both are yes -> not hallucination
    if ref == "yes" and content == "yes":
        return "no"
    # If reference exists but content unknown -> unknown
    if ref == "yes" and content in ["", "na", "unknown"]:
        return "unknown"
    # If reference unknown but content is known
    if ref in ["", "na", "unknown"] and content == "yes":
        return "no"
    if ref in ["", "na", "unknown"] and content == "no":
        return "yes"
    
    return "unknown"


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
        
        .comparison-item {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 12px 15px;
            margin-bottom: 10px;
        }
        
        .comparison-item.match {
            border-left: 4px solid #28a745;
        }
        
        .comparison-item.mismatch {
            border-left: 4px solid #dc3545;
        }
        
        .comparison-item .claim-ref {
            font-weight: 600;
            color: #495057;
        }
        
        .comparison-item .claim-content {
            background: #f8f9fa;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 4px;
            font-size: 0.9em;
            color: #495057;
            font-style: italic;
            border-left: 3px solid #dee2e6;
        }
        
        .comparison-item .comparison {
            margin-top: 5px;
            font-size: 0.95em;
        }
        
        .comparison-item .human {
            color: #007bff;
        }
        
        .comparison-item .judge {
            color: #28a745;
        }
        
        .comparison-item .comment {
            color: #6c757d;
            font-size: 0.85em;
            margin-top: 5px;
            font-style: italic;
        }
        
        .comparison-item .justification {
            background: #f8f9fa;
            padding: 8px 12px;
            margin-top: 8px;
            border-radius: 4px;
            font-size: 0.85em;
            color: #495057;
            border-left: 3px solid #28a745;
        }
        
        .comparison-item .justification-label {
            font-weight: 600;
            color: #28a745;
            margin-bottom: 4px;
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
            max-height: 5000px;
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
    """


def generate_confusion_matrix_html(cm: list) -> str:
    """Generate HTML for a confusion matrix table."""
    return f"""
    <div class="confusion-matrix">
        <table>
            <tr>
                <th class="header-cell" rowspan="2" colspan="2">Confusion Matrix</th>
                <th class="header-cell" colspan="2">Predicted (Judge)</th>
            </tr>
            <tr>
                <th>No</th>
                <th>Yes</th>
            </tr>
            <tr>
                <th class="label-cell" rowspan="2">Actual<br>(Human)</th>
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
                                   comparisons_html: str = "") -> str:
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
        
        {generate_confusion_matrix_html(cm)}
        
        <div class="agreement-box">
            <strong>Human-Judge Agreement:</strong>
            <span class="kappa-value">{agreement.get('agreement', 0):.1%}</span> simple agreement,
            Cohen's κ = <span class="kappa-value">{kappa:.3f}</span>
            <span class="kappa-interp">({kappa_interp})</span>
        </div>
        
        {comparisons_html}
    </div>
    """


def generate_comparisons_html(df: pd.DataFrame, human_col: str, judge_col: str,
                               judge_raw_col: str, max_show: int = 30) -> str:
    """Generate HTML for detailed comparisons."""
    if len(df) == 0:
        return ""
    
    items_html = ""
    for _, row in df.head(max_show).iterrows():
        is_match = row[human_col] == row[judge_col]
        match_class = "match" if is_match else "mismatch"
        match_icon = "✓" if is_match else "✗"
        
        # Get display text
        display_text = row.get('claimed_authors') or row.get('claim_identifier') or 'Unknown'
        display_detail = row.get('claimed_year') or row.get('claim_type') or ''
        
        if display_detail:
            claim_display = f"{html.escape(str(display_text))} ({html.escape(str(display_detail))})"
        else:
            claim_display = html.escape(str(display_text))
        
        # Get claim content
        claim_content = str(row.get('claimed_content', '') or '')
        content_html = ""
        if claim_content:
            if len(claim_content) > 300:
                claim_content = claim_content[:300] + "..."
            content_html = f'<div class="claim-content">{html.escape(claim_content)}</div>'
        
        # Get human comment
        comment_html = ""
        if row.get('human_comment'):
            comment_html = f'<div class="comment">Human comment: {html.escape(str(row["human_comment"]))}</div>'
        
        # Get judge justification
        justification = str(row.get(judge_raw_col, '') or '')
        justification_html = ""
        if justification:
            if len(justification) > 500:
                justification = justification[:500] + "..."
            justification_html = f"""
            <div class="justification">
                <div class="justification-label">Judge Justification:</div>
                {html.escape(justification)}
            </div>
            """
        
        items_html += f"""
        <div class="comparison-item {match_class}">
            <div class="claim-ref">{match_icon} {claim_display}</div>
            {content_html}
            <div class="comparison">
                <span class="human">Human: {row[human_col]}</span> vs 
                <span class="judge">Judge: {row[judge_col]}</span>
            </div>
            {comment_html}
            {justification_html}
        </div>
        """
    
    more_text = f" (showing first {max_show})" if len(df) > max_show else ""
    
    return f"""
    <div class="comparisons-section">
        <button class="collapsible" onclick="this.nextElementSibling.classList.toggle('show')">
            ▶ Show Detailed Comparisons ({len(df)} claims{more_text})
        </button>
        <div class="collapsible-content">
            {items_html}
        </div>
    </div>
    """


def generate_html_report(claims_df: pd.DataFrame, annotated_claims: pd.DataFrame,
                         eval_path: Path, max_examples: int = 30) -> str:
    """Generate the full HTML report."""
    
    # Compute human hallucination labels
    annotated_claims = annotated_claims.copy()
    annotated_claims["human_hallucination"] = annotated_claims.apply(compute_human_hallucination, axis=1)
    
    # Get subsets for each metric type
    ref_annotated = annotated_claims[annotated_claims["human_ref_grounding"].isin(["yes", "no"])]
    content_annotated = annotated_claims[annotated_claims["human_content_grounding"].isin(["yes", "no"])]
    hall_annotated = annotated_claims[annotated_claims["human_hallucination"].isin(["yes", "no"])]
    
    # Reference grounding metrics
    ref_metrics = compute_metrics(
        ref_annotated["human_ref_grounding"].tolist(),
        ref_annotated["judge_ref_grounding"].tolist(),
    ) if len(ref_annotated) > 0 else {"error": "No data"}
    ref_agreement = compute_agreement(
        ref_annotated["human_ref_grounding"].tolist(),
        ref_annotated["judge_ref_grounding"].tolist(),
    ) if len(ref_annotated) > 0 else {"error": "No data"}
    ref_comparisons_html = generate_comparisons_html(
        ref_annotated, "human_ref_grounding", "judge_ref_grounding",
        "judge_ref_grounding_raw", max_examples
    ) if len(ref_annotated) > 0 else ""
    
    # Content grounding metrics
    content_metrics = compute_metrics(
        content_annotated["human_content_grounding"].tolist(),
        content_annotated["judge_content_grounding"].tolist(),
    ) if len(content_annotated) > 0 else {"error": "No data"}
    content_agreement = compute_agreement(
        content_annotated["human_content_grounding"].tolist(),
        content_annotated["judge_content_grounding"].tolist(),
    ) if len(content_annotated) > 0 else {"error": "No data"}
    content_comparisons_html = generate_comparisons_html(
        content_annotated, "human_content_grounding", "judge_content_grounding",
        "judge_content_grounding_raw", max_examples
    ) if len(content_annotated) > 0 else ""
    
    # Hallucination detection metrics
    hall_metrics = compute_metrics(
        hall_annotated["human_hallucination"].tolist(),
        hall_annotated["judge_hallucination"].tolist(),
    ) if len(hall_annotated) > 0 else {"error": "No data"}
    hall_agreement = compute_agreement(
        hall_annotated["human_hallucination"].tolist(),
        hall_annotated["judge_hallucination"].tolist(),
    ) if len(hall_annotated) > 0 else {"error": "No data"}
    hall_comparisons_html = generate_comparisons_html(
        hall_annotated, "human_hallucination", "judge_hallucination",
        "judge_ref_grounding_raw", max_examples
    ) if len(hall_annotated) > 0 else ""
    
    # Match summary
    ref_match = (ref_annotated["human_ref_grounding"] == ref_annotated["judge_ref_grounding"]).mean() if len(ref_annotated) > 0 else 0
    content_match = (content_annotated["human_content_grounding"] == content_annotated["judge_content_grounding"]).mean() if len(content_annotated) > 0 else 0
    hall_match = (hall_annotated["human_hallucination"] == hall_annotated["judge_hallucination"]).mean() if len(hall_annotated) > 0 else 0
    
    # Generate HTML
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Judge vs Human Evaluation - {eval_path.name}</title>
    <style>
        {get_html_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Judge vs Human Evaluation</h1>
            <p class="metadata">
                <strong>Evaluation File:</strong> {eval_path.name}<br>
                <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </p>
        </header>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Comparing automated judge annotations against human ground truth on {len(annotated_claims)} annotated claims.</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(claims_df)}</div>
                <div class="stat-label">Total Claims</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(annotated_claims)}</div>
                <div class="stat-label">Human Annotated</div>
            </div>
            <div class="stat-card {'good' if hall_agreement.get('agreement', 0) > 0.8 else 'warning' if hall_agreement.get('agreement', 0) > 0.6 else 'danger'}">
                <div class="stat-number">{hall_agreement.get('agreement', 0):.1%}</div>
                <div class="stat-label">Hallucination Agreement</div>
            </div>
        </div>
        
        <h2>Judge Outcomes (All Claims)</h2>
        <div class="outcome-grid">
            <div class="outcome-box">
                <h4>Hallucination Detection</h4>
                <ul class="outcome-list">
                    <li>Hallucination=Yes: <strong>{(claims_df['judge_hallucination'] == 'yes').sum()}</strong></li>
                    <li>Hallucination=No: <strong>{(claims_df['judge_hallucination'] == 'no').sum()}</strong></li>
                    <li>Abstention=Yes: <strong>{(claims_df['judge_abstention'] == 'yes').sum()}</strong></li>
                    <li>Verification Error: <strong>{(claims_df['judge_verification_error'] == 'yes').sum()}</strong></li>
                </ul>
            </div>
            <div class="outcome-box">
                <h4>Grounding Decisions</h4>
                <ul class="outcome-list">
                    <li>Reference Yes: <strong>{(claims_df['judge_ref_grounding'] == 'yes').sum()}</strong>, No: <strong>{(claims_df['judge_ref_grounding'] == 'no').sum()}</strong></li>
                    <li>Content Yes: <strong>{(claims_df['judge_content_grounding'] == 'yes').sum()}</strong>, No: <strong>{(claims_df['judge_content_grounding'] == 'no').sum()}</strong></li>
                </ul>
            </div>
        </div>
        
        <h2>Detailed Metrics (Annotated Claims Only)</h2>
        
        {generate_metrics_section_html(
            "Reference Grounding (Does the cited reference exist?)",
            ref_metrics, ref_agreement, ref_comparisons_html
        )}
        
        {generate_metrics_section_html(
            "Content Grounding (Does the content match the source?)",
            content_metrics, content_agreement, content_comparisons_html
        )}
        
        {generate_metrics_section_html(
            "Hallucination Detection (Overall)",
            hall_metrics, hall_agreement, hall_comparisons_html
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
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare automated judge annotations against human ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_judge_vs_human_performance.py eval_results.jsonl
  python evaluate_judge_vs_human_performance.py eval_results.jsonl --output-dir ./results
  python evaluate_judge_vs_human_performance.py eval_results.jsonl --max-examples 50
        """
    )
    parser.add_argument("eval_file", type=Path, help="Path to evaluation results JSONL file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: same as input)")
    parser.add_argument("--max-examples", type=int, default=30, help="Max examples to show per category (default: 30)")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.eval_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📊 Judge vs Human Evaluation")
    print(f"{'='*50}")
    print(f"Evaluation file: {args.eval_file}")
    print()
    
    print("Loading data...")
    eval_results = load_evaluation_results(args.eval_file)
    print(f"  Loaded {len(eval_results)} evaluation results")
    
    # Extract claims
    print("\nProcessing claims...")
    claims = extract_claims_with_annotations(eval_results)
    claims_df = pd.DataFrame(claims)
    print(f"  Extracted {len(claims_df)} total claims")
    
    # Filter to valid claims (no verification errors)
    claims_df = claims_df[claims_df["judge_verification_error"].str.lower().str.startswith("no", na=True)]
    print(f"  Valid claims (no errors): {len(claims_df)}")
    
    # Filter to claims with human annotations
    annotated_claims = claims_df[
        (claims_df["human_ref_grounding"].isin(["yes", "no"])) | 
        (claims_df["human_content_grounding"].isin(["yes", "no"]))
    ].copy()
    print(f"  Claims with human annotations: {len(annotated_claims)}")
    
    if len(annotated_claims) == 0:
        print("\n⚠️  No claims with human annotations found!")
        return
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    html_content = generate_html_report(
        claims_df=claims_df,
        annotated_claims=annotated_claims,
        eval_path=args.eval_file,
        max_examples=args.max_examples,
    )
    
    # Save HTML report
    output_filename = f"judge_vs_human_{args.eval_file.stem}.html"
    output_path = output_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\n✅ Report saved to: {output_path}")
    
    # Quick summary
    annotated_claims["human_hallucination"] = annotated_claims.apply(compute_human_hallucination, axis=1)
    hall_annotated = annotated_claims[annotated_claims["human_hallucination"].isin(["yes", "no"])]
    
    if len(hall_annotated) > 0:
        agreement = compute_agreement(
            hall_annotated["human_hallucination"].tolist(),
            hall_annotated["judge_hallucination"].tolist(),
        )
        print(f"\n📈 Quick Summary:")
        print(f"   Hallucination Agreement: {agreement.get('agreement', 0):.1%}")
        print(f"   Cohen's Kappa: {agreement.get('cohen_kappa', 0):.3f} ({interpret_kappa(agreement.get('cohen_kappa', 0))})")


if __name__ == "__main__":
    main()
