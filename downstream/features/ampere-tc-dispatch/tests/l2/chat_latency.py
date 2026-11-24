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


def measure(upstream, name, prompt, limit, timeout, *, capture_output=False,
            tools=None, tool_choice=None, messages=None):
    body = {
        'messages': messages if messages is not None else [{'role': 'user', 'content': prompt}],
        'max_tokens': limit, 'temperature': 0.0, 'seed': 123, 'stream': True,
        'stream_options': {'include_usage': True}, 'reasoning_effort': 'none',
        'chat_template_kwargs': {'enable_thinking': False},
    }
    if tools:
        body['tools'] = tools
        if tool_choice is not None:
            body['tool_choice'] = tool_choice
    request_hash = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    req = urllib.request.Request(upstream + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    output_times = []
    timings = {}
    usage = {}
    finish = None
    done = False
    output_hash = hashlib.sha256()
    reasoning_hash = hashlib.sha256()
    content_chunks = []
    reasoning_chunks = []
    tool_calls = {}
    generated_token_ids = []
    generation_failure_count = 0
    generation_recovery_count = 0
    recovery_reasons = []
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
            failure = item.get('strata_generation_failure')
            recovery = item.get('strata_generation_recovery')
            if failure or recovery:
                generation_failure_count += bool(failure)
                generation_recovery_count += bool(recovery)
                if isinstance(recovery, dict):
                    reason = recovery.get('reason') or recovery.get('type')
                    if reason:
                        recovery_reasons.append(str(reason)[:160])
                if not capture_output:
                    raise RuntimeError(f'{name}: repetition failure/recovery is not a valid performance baseline')
            if item.get('timings'):
                timings = item['timings']
            if item.get('usage'):
                usage = item['usage']
            for choice in item.get('choices') or []:
                delta = choice.get('delta') or {}
                content = delta.get('content')
                reasoning = delta.get('reasoning_content')
                if isinstance(content, str):
                    output_hash.update(content.encode())
                    if capture_output:
                        content_chunks.append(content)
                if isinstance(reasoning, str):
                    reasoning_hash.update(reasoning.encode())
                    if capture_output:
                        reasoning_chunks.append(reasoning)
                for call in delta.get('tool_calls') or []:
                    index = call.get('index', 0)
                    target = tool_calls.setdefault(index, {
                        'id': None, 'type': 'function',
                        'function': {'name': '', 'arguments': ''},
                    })
                    if call.get('id'):
                        target['id'] = call['id']
                    if call.get('type'):
                        target['type'] = call['type']
                    fn = call.get('function') or {}
                    if fn.get('name'):
                        target['function']['name'] += fn['name']
                    if fn.get('arguments'):
                        target['function']['arguments'] += fn['arguments']
                ids = delta.get('generated_token_ids') or delta.get('token_ids') or []
                if isinstance(ids, list):
                    generated_token_ids.extend(i for i in ids if isinstance(i, int) and not isinstance(i, bool))
                finish = choice.get('finish_reason') or finish
                if content or reasoning or delta.get('tool_calls'):
                    output_times.append(elapsed)
    if not done or not output_times or finish not in ('stop', 'length', 'tool_calls'):
        raise RuntimeError(f'{name}: benchmark stream did not finish normally')
    if not isinstance(timings, dict) or any(
        not isinstance(timings.get(key), (int, float)) or not math.isfinite(timings[key]) or timings[key] <= 0
        for key in ('prompt_ms', 'predicted_ms', 'predicted_per_second', 'predicted_n')
    ) or timings['predicted_n'] < 2:
        raise RuntimeError(f'{name}: too few tokens or invalid engine timings')
    calls = [tool_calls[key] for key in sorted(tool_calls)]
    output = content_gaps(output_times)
    output['output_events'] = len(output_times)
    result = {
        'case': name, 'target_tokens': limit,
        'prompt_sha256': hashlib.sha256(
            prompt.encode() if messages is None else
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        'request_sha256': request_hash,
        'first_content_ms': round(output_times[0] * 1000, 3),
        'first_output_ms': round(output_times[0] * 1000, 3),
        'wall_ms': round((time.monotonic() - started) * 1000, 3),
        'finish': finish, 'stream': output,
        'timings': {key: timings[key] for key in TIMING_FIELDS if type(timings.get(key)) in (int, float)},
        'usage': usage or None, 'output_sha256': output_hash.hexdigest(),
    }
    if capture_output:
        result.update(
            content=''.join(content_chunks), reasoning_content=''.join(reasoning_chunks),
            tool_calls=calls, reasoning_sha256=reasoning_hash.hexdigest(),
            tool_calls_sha256=hashlib.sha256(json.dumps(calls,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            generated_token_ids=generated_token_ids or None,
            token_id_source='server SSE token ID field' if generated_token_ids else
                            'not exposed; no output retokenization performed',
            generation_failure_count=int(generation_failure_count),
            generation_recovery_count=int(generation_recovery_count),
            generation_recovery_reasons=recovery_reasons,
        )
    return result


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
