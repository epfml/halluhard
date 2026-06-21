#!/usr/bin/env python3
"""
Fallback & Hallucination-Type Usage Evaluation

Auto-detects evaluation format (webscraper vs coding) and analyses per-claim flags:

Webscraper format (medical / legal / research):
  - input_use_fallback, judge_used_websearch_fallback, snippets_only

Coding format:
  - hallucinated_import_detected, hallucinated_install_detected,
    hallucinated_function_usage_detected

Outputs an HTML report with aggregate stats, per-conversation breakdowns,
and hallucination correlation analysis.

Usage:
    python evaluate_fallback_usage.py eval_results.jsonl
    python evaluate_fallback_usage.py file1.jsonl file2.jsonl file3.jsonl
    python evaluate_fallback_usage.py eval_results.jsonl --output-dir ./reports
"""

import argparse
import html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# Format definitions
# =============================================================================

WEBSCRAPER_FLAGS = ["input_use_fallback", "judge_used_websearch_fallback", "snippets_only"]
WEBSCRAPER_LABELS = {
    "input_use_fallback": "Input Use Fallback",
    "judge_used_websearch_fallback": "Websearch Fallback",
    "snippets_only": "Snippets Only",
}
WEBSCRAPER_CONV_COUNT_KEYS = {
    "input_use_fallback": "input_use_fallback_count",
    "judge_used_websearch_fallback": "judge_used_websearch_fallback_count",
    "snippets_only": "snippets_only_count",
}

CODING_FLAGS = [
    "hallucinated_import_detected",
    "hallucinated_install_detected",
    "hallucinated_function_usage_detected",
]
CODING_LABELS = {
    "hallucinated_import_detected": "Import Hallucination",
    "hallucinated_install_detected": "Install Hallucination",
    "hallucinated_function_usage_detected": "Function Hallucination",
}
CODING_CONV_COUNT_KEYS = {
    "hallucinated_import_detected": "import_hallucinations",
    "hallucinated_install_detected": "install_hallucinations",
    "hallucinated_function_usage_detected": "function_hallucinations",
}


# =============================================================================
# Data Loading & Format Detection
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


def detect_format(results: list[dict[str, Any]]) -> str:
    """Return 'coding' or 'webscraper' based on the structure of the first record."""
    if not results:
        return "webscraper"
    details = results[0].get("details", {})
    if "turn_evaluations" in details and "import_hallucinations" in details:
        return "coding"
    return "webscraper"


def get_format_config(fmt: str) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return (flags, labels, conv_count_keys) for the detected format."""
    if fmt == "coding":
        return CODING_FLAGS, CODING_LABELS, CODING_CONV_COUNT_KEYS
    return WEBSCRAPER_FLAGS, WEBSCRAPER_LABELS, WEBSCRAPER_CONV_COUNT_KEYS


# =============================================================================
# Data Extraction
# =============================================================================

def extract_claim_level_stats(
    results: list[dict[str, Any]],
    flags: list[str],
) -> list[dict[str, Any]]:
    """Extract per-claim flag and hallucination data from evaluation results."""
    claims = []
    for record in results:
        conv_id = record.get("conversation_id", "?")
        details = record.get("details", {})
        for claim_eval in details.get("claim_evaluations", []):
            row: dict[str, Any] = {
                "conversation_id": conv_id,
                "claim_id": claim_eval.get("claim_id", ""),
                "turn_idx": claim_eval.get("turn_idx", claim_eval.get("turn_number", "")),
                "hallucination": claim_eval.get("hallucination", "").lower().strip(),
                "abstention": claim_eval.get("abstention", "").lower().strip(),
                "verification_error": claim_eval.get("verification_error", "").lower().strip(),
            }
            for flag in flags:
                row[flag] = bool(claim_eval.get(flag, False))
            claims.append(row)
    return claims


def extract_conversation_level_stats(
    results: list[dict[str, Any]],
    flags: list[str],
    conv_count_keys: dict[str, str],
    fmt: str,
) -> list[dict[str, Any]]:
    """Extract per-conversation aggregated flag counts."""
    conversations = []
    for record in results:
        details = record.get("details", {})
        if fmt == "coding":
            total_hall = details.get("overall_hallucinated_responses", 0)
        else:
            total_hall = details.get("hallucinations", 0)
        row: dict[str, Any] = {
            "conversation_id": record.get("conversation_id", "?"),
            "score": record.get("score"),
            "total_claims": details.get("total_claims", 0),
            "hallucinations": total_hall,
        }
        for flag in flags:
            row[conv_count_keys[flag]] = details.get(conv_count_keys[flag], 0)
        conversations.append(row)
    return conversations


# =============================================================================
# Statistics Computation
# =============================================================================

def compute_aggregate_stats(
    claims: list[dict[str, Any]], flags: list[str],
) -> dict[str, Any]:
    """Compute aggregate flag statistics across all claims."""
    total = len(claims)
    if total == 0:
        return {"total_claims": 0}

    stats: dict[str, Any] = {"total_claims": total}
    for flag in flags:
        true_count = sum(1 for c in claims if c[flag])
        stats[flag] = {
            "true": true_count,
            "false": total - true_count,
            "rate": true_count / total,
        }

    any_flag = sum(
        1 for c in claims if any(c[f] for f in flags)
    )
    stats["any_flag"] = {
        "true": any_flag,
        "false": total - any_flag,
        "rate": any_flag / total,
    }

    return stats


def compute_co_occurrence(
    claims: list[dict[str, Any]], flags: list[str], labels: dict[str, str],
) -> dict[str, int]:
    """Count co-occurrences of flag combinations."""
    combos: dict[str, int] = defaultdict(int)
    for c in claims:
        active = tuple(sorted(f for f in flags if c[f]))
        key = " + ".join(labels[f] for f in active) if active else "None"
        combos[key] += 1
    return dict(sorted(combos.items(), key=lambda x: -x[1]))


def compute_hallucination_correlation(
    claims: list[dict[str, Any]], flags: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute hallucination rates stratified by each flag."""
    hall_claims = [
        c for c in claims if c["hallucination"] in ("yes", "no")
    ]
    if not hall_claims:
        return {}

    results = {}
    for flag in flags + ["any_flag"]:
        if flag == "any_flag":
            with_fb = [c for c in hall_claims if any(c[f] for f in flags)]
            without_fb = [c for c in hall_claims if not any(c[f] for f in flags)]
        else:
            with_fb = [c for c in hall_claims if c[flag]]
            without_fb = [c for c in hall_claims if not c[flag]]

        def _hall_rate(subset: list[dict]) -> tuple[int, int, float]:
            total = len(subset)
            if total == 0:
                return 0, 0, 0.0
            h = sum(1 for c in subset if c["hallucination"] == "yes")
            return h, total, h / total

        h_with, n_with, rate_with = _hall_rate(with_fb)
        h_without, n_without, rate_without = _hall_rate(without_fb)

        results[flag] = {
            "with_flag": {"hallucinations": h_with, "total": n_with, "rate": rate_with},
            "without_flag": {"hallucinations": h_without, "total": n_without, "rate": rate_without},
        }

    return results


# =============================================================================
# Console Summary
# =============================================================================

def print_summary(
    agg_stats: dict[str, Any],
    co_occurrence: dict[str, int],
    hall_corr: dict[str, dict[str, Any]],
    conv_stats: list[dict[str, Any]],
    file_label: str,
    flags: list[str],
    labels: dict[str, str],
    conv_count_keys: dict[str, str],
    fmt: str,
) -> None:
    """Print a concise console summary."""
    total = agg_stats["total_claims"]
    kind = "Coding Hallucination-Type" if fmt == "coding" else "Fallback Usage"
    any_label = "Any Hallucination Type" if fmt == "coding" else "Any Fallback"

    print(f"\n{'=' * 60}")
    print(f"  {kind} Summary — {file_label}")
    print(f"{'=' * 60}")
    print(f"  Total claims: {total}")
    print()

    for flag in flags:
        info = agg_stats[flag]
        print(f"  {labels[flag]:.<35s} {info['true']:>5d} / {total}  ({info['rate']:.1%})")
    info = agg_stats["any_flag"]
    print(f"  {any_label:.<35s} {info['true']:>5d} / {total}  ({info['rate']:.1%})")

    print(f"\n  Conversations: {len(conv_stats)}")
    convs_with_any = sum(
        1 for c in conv_stats
        if sum(c[conv_count_keys[f]] for f in flags) > 0
    )
    print(f"  Conversations with {any_label.lower()}: {convs_with_any} ({convs_with_any / len(conv_stats):.1%})")

    if hall_corr:
        print(f"\n  Hallucination rate by flag status:")
        for flag in flags + ["any_flag"]:
            label = labels.get(flag, any_label)
            info = hall_corr.get(flag, {})
            w = info.get("with_flag", {})
            wo = info.get("without_flag", {})
            if w.get("total", 0) > 0 and wo.get("total", 0) > 0:
                print(f"    {label}: with={w['rate']:.1%} ({w['hallucinations']}/{w['total']})  "
                      f"without={wo['rate']:.1%} ({wo['hallucinations']}/{wo['total']})")


# =============================================================================
# HTML Report
# =============================================================================

def _esc(text: Any) -> str:
    return html.escape(str(text))


def generate_html_report(
    agg_stats: dict[str, Any],
    co_occurrence: dict[str, int],
    hall_corr: dict[str, dict[str, Any]],
    conv_stats: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    file_paths: list[Path],
    flags: list[str],
    labels: dict[str, str],
    conv_count_keys: dict[str, str],
    fmt: str,
) -> str:
    """Generate a self-contained HTML report."""
    total = agg_stats["total_claims"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    files_str = ", ".join(p.name for p in file_paths)
    is_coding = fmt == "coding"
    report_title = "Coding Hallucination-Type Report" if is_coding else "Fallback Usage Report"
    any_label = "Any Hallucination Type" if is_coding else "Any Fallback"
    section_agg = "Aggregate Hallucination-Type Counts" if is_coding else "Aggregate Fallback Counts"
    section_co = "Hallucination-Type Co-occurrence" if is_coding else "Fallback Combination Co-occurrence"
    total_col = "Total HT" if is_coding else "Total FB"

    sections: list[str] = []

    # --- Aggregate table ---
    rows = ""
    for flag in flags:
        info = agg_stats[flag]
        rows += f"""
        <tr>
          <td>{_esc(labels[flag])}</td>
          <td class="num">{info['true']}</td>
          <td class="num">{info['false']}</td>
          <td class="num">{total}</td>
          <td class="num">{info['rate']:.1%}</td>
        </tr>"""
    info = agg_stats["any_flag"]
    rows += f"""
        <tr class="highlight">
          <td><strong>{_esc(any_label)}</strong></td>
          <td class="num"><strong>{info['true']}</strong></td>
          <td class="num"><strong>{info['false']}</strong></td>
          <td class="num"><strong>{total}</strong></td>
          <td class="num"><strong>{info['rate']:.1%}</strong></td>
        </tr>"""

    sections.append(f"""
    <h2>{section_agg}</h2>
    <table>
      <thead>
        <tr><th>Flag</th><th>True</th><th>False</th><th>Total</th><th>Rate</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>""")

    # --- Co-occurrence ---
    co_rows = ""
    for combo, count in co_occurrence.items():
        pct = count / total if total else 0
        co_rows += f"<tr><td>{_esc(combo)}</td><td class='num'>{count}</td><td class='num'>{pct:.1%}</td></tr>"
    sections.append(f"""
    <h2>{section_co}</h2>
    <table>
      <thead><tr><th>Active Flags</th><th>Count</th><th>%</th></tr></thead>
      <tbody>{co_rows}</tbody>
    </table>""")

    # --- Hallucination correlation ---
    if hall_corr:
        hall_rows = ""
        for flag in flags + ["any_flag"]:
            label = labels.get(flag, any_label)
            info = hall_corr.get(flag, {})
            w = info.get("with_flag", {})
            wo = info.get("without_flag", {})
            delta = ""
            if w.get("total", 0) > 0 and wo.get("total", 0) > 0:
                d = w["rate"] - wo["rate"]
                sign = "+" if d >= 0 else ""
                delta = f"{sign}{d:.1%}"
            hall_rows += f"""
            <tr>
              <td>{_esc(label)}</td>
              <td class="num">{w.get('hallucinations', 0)}/{w.get('total', 0)} ({w.get('rate', 0):.1%})</td>
              <td class="num">{wo.get('hallucinations', 0)}/{wo.get('total', 0)} ({wo.get('rate', 0):.1%})</td>
              <td class="num">{delta}</td>
            </tr>"""
        sections.append(f"""
        <h2>Hallucination Rate by Flag Status</h2>
        <p>Compares hallucination rate for claims <em>with</em> vs. <em>without</em> each flag
           (restricted to claims with a definitive yes/no hallucination judgment).</p>
        <table>
          <thead>
            <tr><th>Flag</th><th>With Flag</th><th>Without Flag</th><th>Delta</th></tr>
          </thead>
          <tbody>{hall_rows}</tbody>
        </table>""")

    # --- Per-conversation table ---
    conv_stats_sorted = sorted(conv_stats, key=lambda c: c["conversation_id"])
    flag_headers = "".join(f"<th>{_esc(labels[f])}</th>" for f in flags)
    conv_rows = ""
    for c in conv_stats_sorted:
        total_count = sum(c[conv_count_keys[f]] for f in flags)
        cls = ' class="warn"' if total_count > 0 else ""
        score_str = f"{c['score']:.2%}" if c["score"] is not None else "N/A"
        flag_cells = "".join(
            f'<td class="num">{c[conv_count_keys[f]]}</td>' for f in flags
        )
        conv_rows += f"""
        <tr{cls}>
          <td class="num">{c['conversation_id']}</td>
          <td class="num">{c['total_claims']}</td>
          <td class="num">{c['hallucinations']}</td>
          <td class="num">{score_str}</td>
          {flag_cells}
          <td class="num">{total_count}</td>
        </tr>"""

    sections.append(f"""
    <h2>Per-Conversation Breakdown</h2>
    <table id="conv-table">
      <thead>
        <tr>
          <th>Conv ID</th><th>Claims</th><th>Hallucinations</th><th>Score</th>
          {flag_headers}
          <th>{total_col}</th>
        </tr>
      </thead>
      <tbody>{conv_rows}</tbody>
    </table>""")

    # --- Assemble full HTML ---
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{report_title}</title>
<style>
  :root {{ --bg: #fafafa; --card: #fff; --border: #e0e0e0; --accent: #1a73e8;
           --warn-bg: #fff3e0; --text: #222; --muted: #666; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
          background: var(--bg); color: var(--text); padding: 2rem; line-height: 1.5; }}
  h1 {{ color: var(--accent); margin-bottom: .25rem; }}
  .meta {{ color: var(--muted); font-size: .85rem; margin-bottom: 1.5rem; }}
  h2 {{ margin-top: 2rem; margin-bottom: .75rem; color: #333; border-bottom: 2px solid var(--accent);
        padding-bottom: .25rem; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
           border: 1px solid var(--border); margin-bottom: 1rem; font-size: .9rem; }}
  th, td {{ padding: .5rem .75rem; border: 1px solid var(--border); text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; position: sticky; top: 0; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.highlight {{ background: #e8f0fe; }}
  tr.warn {{ background: var(--warn-bg); }}
  p {{ margin: .5rem 0 1rem; color: var(--muted); }}
</style>
</head>
<body>
  <h1>{report_title}</h1>
  <div class="meta">
    Generated: {timestamp}<br/>
    Files: {_esc(files_str)}<br/>
    Total claims: {total} &middot; Conversations: {len(conv_stats)}
  </div>
  {body}
</body>
</html>"""


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze fallback usage across evaluation JSONL files.",
    )
    parser.add_argument(
        "eval_files",
        nargs="+",
        type=Path,
        help="One or more evaluation JSONL files to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the HTML report (default: same dir as first input file).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation, print console summary only.",
    )
    args = parser.parse_args()

    all_results: list[dict[str, Any]] = []
    for fp in args.eval_files:
        if not fp.exists():
            print(f"WARNING: {fp} not found, skipping.")
            continue
        records = load_evaluation_results(fp)
        print(f"Loaded {len(records)} conversations from {fp.name}")
        all_results.extend(records)

    if not all_results:
        print("No data loaded. Exiting.")
        return

    fmt = detect_format(all_results)
    flags, labels, conv_count_keys = get_format_config(fmt)
    print(f"Detected format: {fmt} (flags: {', '.join(labels.values())})")

    claims = extract_claim_level_stats(all_results, flags)
    conv_stats = extract_conversation_level_stats(all_results, flags, conv_count_keys, fmt)
    agg_stats = compute_aggregate_stats(claims, flags)
    co_occurrence = compute_co_occurrence(claims, flags, labels)
    hall_corr = compute_hallucination_correlation(claims, flags)

    file_label = ", ".join(p.name for p in args.eval_files)
    print_summary(
        agg_stats, co_occurrence, hall_corr, conv_stats, file_label,
        flags, labels, conv_count_keys, fmt,
    )

    if not args.no_html:
        output_dir = args.output_dir or args.eval_files[0].parent
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = args.eval_files[0].stem if len(args.eval_files) == 1 else "combined"
        output_path = output_dir / f"fallback_usage_{stem}.html"
        html_content = generate_html_report(
            agg_stats, co_occurrence, hall_corr, conv_stats, claims,
            args.eval_files, flags, labels, conv_count_keys, fmt,
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"\n  HTML report saved to: {output_path}")


if __name__ == "__main__":
    main()
