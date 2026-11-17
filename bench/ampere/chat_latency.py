#!/usr/bin/env python3
"""Measure local Strata Chat Completions with synthetic prompts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

TIMING_FIELDS = ('prompt_n', 'prompt_ms', 'predicted_n', 'predicted_ms',
                 'predicted_per_token_ms', 'predicted_per_second', 'prompt_per_token_ms',
                 'prompt_per_second', 'draft_n', 'draft_n_accepted', 'cache_n')


def workloads():
    faults = ('checksum mismatch', 'stale index', 'clock skew', 'orphaned segment')
    records = '\n'.join(
        f'Record {10000 + i}, shard {(i * 223) % 81817}: detected {faults[i % len(faults)]}; '
        f'restored journal offset {(i * 3817) % 2750089}; '
        f'operator {i % 17} verified page {i * 9 % 6101} and reopened the shard.'
        for i in range(600)
    )
    return (
        ('warmup', 'Explain idempotent log replay in one paragraph.', 64),
        ('long_output', 'Write Python source for an extensive pytest suite for a journaled key-value store. '
         'Include fixtures, helper implementations, and at least sixty distinct fully implemented '
         'tests for crash recovery, concurrency, truncated records, corrupt checksums, model-based '
         'sequences, and idempotent replays. Write out every test body and assertion. '
         'Keep going through the full suite. Do not summarize or abbreviate.', 3072),
        ('long_context', 'Use these database incident records as the basis for your answer.\n' + records + '\n\n'
         'Write a numbered incident response procedure and a decision table.\n'
         'Write out a complete operational handbook with at least 30 numbered procedures. '
         'For each procedure give a trigger, diagnosis, detailed actions, recovery verification, '
         'and a failure example. Keep the procedures fully written out.', 1536),
        ('second_output', 'Write a complete Rust integration test suite for a message queue that is durable '
         'across process restarts. Include mocks, fixtures and at least 40 implemented tests for delayed delivery, '
         'requeue, backpressure and recovery. Give full code bodies without placeholders.', 3072),
    )


def content_gaps(times):
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    result = {'content_events': len(times)}
    if gaps:
        for percent in (50, 90, 95, 99):
            result[f'gap_p{percent}_ms'] = round(gaps[math.ceil(len(gaps) * percent / 100) - 1] * 1000, 3)
        result['gap_max_ms'] = round(gaps[-1] * 1000, 3)
    return result


def measure(upstream, name, prompt, limit, timeout):
    body = {
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': limit, 'temperature': 0.0, 'seed': 123, 'stream': True,
        'stream_options': {'include_usage': True}, 'reasoning_effort': 'none',
        'chat_template_kwargs': {'enable_thinking': False},
    }
    req = urllib.request.Request(upstream + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    content_times = []
    timings = {}
    finish = None
    done = False
    output_hash = hashlib.sha256()
    with opener.open(req, timeout=timeout) as response:
        if response.headers.get_content_type() != 'text/event-stream':
            raise RuntimeError(f'{name}: backend did not stream SSE')
        for line in response:
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                raise TimeoutError(f'{name}: benchmark request exceeded {timeout}s')
            if not line.startswith(b'data:'):
                continue
            payload = line[5:].strip()
            if payload == b'[DONE]':
                done = True
                break
            item = json.loads(payload)
            if item.get('error'):
                raise RuntimeError(f'{name}: benchmark request returned an error')
            if item.get('timings'):
                timings = item['timings']
            has_content = False
            for choice in item.get('choices') or []:
                delta = choice.get('delta') or {}
                if delta.get('content'):
                    output_hash.update(delta['content'].encode())
                has_content = has_content or bool(delta.get('content') or delta.get('reasoning_content'))
                finish = choice.get('finish_reason') or finish
            if has_content:
                content_times.append(elapsed)
    if not done or not content_times or finish not in ('stop', 'length'):
        raise RuntimeError(f'{name}: benchmark stream did not finish normally')
    if not isinstance(timings, dict) or any(
        not isinstance(timings.get(key), (int, float)) or not math.isfinite(timings[key]) or timings[key] <= 0
        for key in ('prompt_ms', 'predicted_ms', 'predicted_per_second', 'predicted_n')
    ) or timings['predicted_n'] < 2:
        raise RuntimeError(f'{name}: too few tokens or invalid engine timings')
    return {'case': name, 'target_tokens': limit, 'first_content_ms': round(content_times[0] * 1000, 3),
            'wall_ms': round((time.monotonic() - started) * 1000, 3), 'finish': finish,
            'stream': content_gaps(content_times),
            'timings': {key: timings[key] for key in TIMING_FIELDS if type(timings.get(key)) in (int, float)},
            'output_sha256': output_hash.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8095)
    parser.add_argument('--mode', choices=('single', 'pair'), required=True, help='label for the running engine')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    upstream = f'http://127.0.0.1:{args.port}'
    cases = [measure(upstream, name, prompt, limit, 210) for name, prompt, limit in workloads()]
    args.output.write_text(json.dumps({'mode': args.mode, 'cases': cases}, indent=2) + '\n')


if __name__ == '__main__':
    main()
