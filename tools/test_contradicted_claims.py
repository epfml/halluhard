#!/usr/bin/env python3
"""Re-run contradicted claims through the real pipeline to test judging fixes.

Picks claims where a previous run gave the SAME cited URL opposite
reference_grounding verdicts - the clearest evidence of non-deterministic
judging - and runs each one through the live path (search -> fetch -> filter
-> judge), so the fixes can be checked against the verdicts they produced.

Reports, per claim: what the old run said, what the new run says, and which
mechanisms fired (identifier lookup, cited-source identity block, evidence
escalation).

Usage:
    python -m tools.test_contradicted_claims \
        --input legal_cases/results/conversations_gpt-6-astra-websearch_100convs_eval_webscraper.jsonl \
        --task legal_cases --pairs 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from judging_pipeline.core.queue import MonitoredQueue
from judging_pipeline.models.work_items import ClaimItem
from judging_pipeline.strategies import get_strategy

META = ('claimed_title', 'claimed_authors', 'claimed_year', 'full_citation', 'claimed_institution')


def find_contradicted(path: Path, max_pairs: int):
    """Return [(url, [claim_eval, ...])] where verdicts on the same URL disagree."""
    by_url = defaultdict(list)
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record.get('_type') != 'evaluation_result':
            continue
        for claim_eval in record.get('details', {}).get('claim_evaluations', []):
            if str(claim_eval.get('verification_error', 'No')).lower() in ('yes', 'true', 'unknown'):
                continue
            claim = claim_eval.get('claim', {})
            url = (claim.get('claimed_url') or '').strip().split('?')[0]
            if not url or any((claim.get(k) or '').strip() for k in META):
                continue
            grounding = (claim_eval.get('reference_grounding') or '').strip()
            if grounding:
                by_url[url].append(claim_eval)

    pairs = []
    for url, evals in by_url.items():
        verdicts = {(e['reference_grounding'] or '').strip().lower().startswith('yes') for e in evals}
        if len(verdicts) > 1:
            yes = next(e for e in evals if e['reference_grounding'].strip().lower().startswith('yes'))
            no = next(e for e in evals if not e['reference_grounding'].strip().lower().startswith('yes'))
            pairs.append((url, [no, yes]))  # disagreeing claim first
        if len(pairs) >= max_pairs:
            break
    return pairs


class _CollectingQueue:
    """Stands in for the PDF queue so this harness can convert PDFs inline."""

    def __init__(self):
        self.items = []

    async def put(self, item, **kwargs):
        self.items.append(item)


async def judge_one(claim_eval, strategy, workers) -> dict:
    """Run a single claim through search -> fetch -> (pdf) -> filter -> judge."""
    searcher, fetcher, filterer, judge, pdf_converter = workers
    claim = ClaimItem(
        claim_id=claim_eval.get('claim_id', ''),
        conversation_id=claim_eval.get('conversation_id', 0),
        turn_number=claim_eval.get('turn_idx', claim_eval.get('turn_number', 0)),
        data=claim_eval.get('claim', {}),
        metadata={},
    )
    search_task = await searcher.process(claim, None)
    fetcher.pdf_queue = _CollectingQueue()
    content = await fetcher.process(search_task, None)

    # The real pipeline converts queued PDFs in a separate worker and merges them
    # back via the aggregator; do that inline so PDF-cited claims are testable.
    for pdf_task in fetcher.pdf_queue.items:
        result = await pdf_converter.process(pdf_task, None)
        if getattr(result, "success", False) and result.content:
            content.pdf_contents.append({
                "title": result.title, "url": result.url, "snippet": "", "content": result.content,
            })

    filtered = await filterer.process(content, None)
    verdict = await judge.process(filtered, None)
    return {
        'verdict': verdict,
        'queries': search_task.queries_executed,
        'identity_block': '[CITED SOURCE]' in (filtered.filtered_content or ''),
        'fetched': len(content.contents) + len(content.pdf_contents),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', '-i', required=True, help='Existing *_eval_*.jsonl file')
    parser.add_argument('--task', required=True)
    parser.add_argument('--base_path', default=None)
    parser.add_argument('--pairs', type=int, default=5, help='How many contradicted URLs to test')
    parser.add_argument('--judge-model', default='gpt-5-mini-medium')
    parser.add_argument('--judge-fallback-model', default='gpt-5-mini-medium-websearch')
    parser.add_argument('--aux-model', default='gpt-5-mini-minimal', help='Search-query planner')
    parser.add_argument('--escalation-rounds', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true', help='Show the selected claims, call nothing')
    args = parser.parse_args()

    pairs = find_contradicted(Path(args.input), args.pairs)
    print(f'Found {len(pairs)} contradicted URLs in {Path(args.input).name}\n')
    for url, (no_side, yes_side) in pairs:
        print(f'  {url[:88]}')
        print(f'     claim {no_side.get("claim_id")}: NO  - {(no_side.get("reference_grounding") or "")[:110]}')
        print(f'     claim {yes_side.get("claim_id")}: YES - {(yes_side.get("reference_grounding") or "")[:110]}')
    if args.dry_run:
        print('\n[dry run] no API calls made')
        return

    from judging_pipeline.workers import (
        WebSearcherWorker, WebFetcherWorker, ContentFilterWorker, JudgeWorker, EvidenceEscalator,
        PDFConverterWorker,
    )
    from libs.models import get_sampler
    from libs.serper.client import SerperSearchClient

    strategy = get_strategy(args.task, Path(args.base_path or args.task))
    q = lambda name: MonitoredQueue(name)
    workers = (
        WebSearcherWorker(input_queue=q('c'), output_queue=q('s'),
                          search_sampler=get_sampler(args.aux_model),
                          claim_text_builder=strategy.build_textual_claim_for_websearch,
                          strategy=strategy),
        WebFetcherWorker(input_queue=q('s'), output_queue=q('f'), pdf_queue=None),
        ContentFilterWorker(input_queue=q('f'), output_queue=q('t'),
                            claim_text_builder=strategy.build_textual_claim_for_websearch),
        JudgeWorker(input_queue=q('t2'), output_queue=q('o'),
                    sampler=get_sampler(args.judge_model), strategy=strategy,
                    sampler_fallback=get_sampler(args.judge_fallback_model),
                    evidence_escalator=EvidenceEscalator(SerperSearchClient()),
                    max_escalation_rounds=args.escalation_rounds),
        PDFConverterWorker(input_queue=q('p1'), output_queue=q('p2')),
    )

    # Workers create their HTTP/Serper clients in setup(); the queue runtime
    # normally calls it, so driving process() directly means doing it by hand.
    for worker in workers:
        await worker.setup()

    print(f'\nRe-running {len(pairs)*2} claims through the fixed pipeline '
          f'(judge={args.judge_model}, escalation={args.escalation_rounds})\n')
    agree = 0
    for url, evals in pairs:
        print(f'{url[:88]}')
        outcomes = []
        for claim_eval in evals:
            was = 'YES' if (claim_eval.get('reference_grounding') or '').strip().lower().startswith('yes') else 'NO'
            try:
                out = await judge_one(claim_eval, strategy, workers)
            except Exception as e:  # keep going; one bad claim should not end the run
                print(f'   claim {claim_eval.get("claim_id")}: ERROR {type(e).__name__}: {e}')
                continue
            v = out['verdict']
            now = 'YES' if (v.reference_grounding or '').strip().lower().startswith('yes') else 'NO'
            excluded = str(v.verification_error).strip().lower() in ('yes', 'true')
            outcomes.append('UNVERIFIABLE' if excluded else now)
            print(f'   claim {claim_eval.get("claim_id")}: was {was:3} -> now {now:3}'
                  f'{" (insufficient -> excluded)" if excluded else ""}'
                  f'  | halluc={v.hallucination} escalations={v.escalation_rounds}'
                  f' identity_block={out["identity_block"]} docs={out["fetched"]}')
            print(f'      queries: {out["queries"]}')
            print(f'      reason : {(v.reference_grounding or "")[:150]}')
        if len(outcomes) < 2:
            # A claim that errored produced no verdict; that is a failed test, not
            # a disagreement, and must not be reported as one.
            print(f'   => NO RESULT ({2 - len(outcomes)} of 2 claims failed to produce a verdict)')
        elif len(set(outcomes)) == 1:
            agree += 1
            print(f'   => consistent now (both {outcomes[0]})')
        else:
            print(f'   => STILL INCONSISTENT ({" vs ".join(outcomes)})')
        print()
    for worker in workers:
        await worker.teardown()

    print(f'Consistent after fixes: {agree}/{len(pairs)} previously-contradicted URLs '
          f'(pairs with a failed claim are excluded from this count)')


if __name__ == '__main__':
    asyncio.run(main())
