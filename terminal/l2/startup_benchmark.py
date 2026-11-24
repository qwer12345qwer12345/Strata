"""Run a fixed local Chat Completions suite before publishing the service."""
import datetime
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.request
from pathlib import Path


SUITE = 'gyro-s-startup-v1'
CASES = (
    ('warmup', 'Explain a database transaction with one short example.', 48),
    ('decode', 'Implement a Python function that merges overlapping closed intervals. Include a docstring, '
               'type annotations, edge cases, time complexity, and six runnable examples.', 256),
    ('context', 'Reference notes for the storage-engine handbook:\n' +
                ('A write-ahead log records changes before commit. Checksums identify torn writes. '
                 'Recovery replays committed changes after the last durable checkpoint. '
                 'A half-written record is ignored and reported to the operator.\n' * 180) +
                'Write a handbook section with a numbered recovery procedure, three edge cases, and two tests.', 192),
)
ROUNDS = re.compile(rb'strata draft: request \d+ drafter=\S+ window<=\d+ tokens=(\d+) rounds=(\d+)')


def profile_seed(cfg):
    """Record the profile present before the engine has a chance to update it."""
    args = cfg['args']
    if '--expert-profile' not in args:
        return None
    path = Path(args[args.index('--expert-profile') + 1])
    try:
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    except FileNotFoundError:
        return None


def run_case(upstream, name, prompt, limit, log, timeout, *, open_request=urllib.request.urlopen):
    offset = log.stat().st_size if log and log.exists() else 0
    body = {
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': limit, 'temperature': 0.0, 'seed': 123, 'stream': True,
        'stream_options': {'include_usage': True}, 'reasoning_effort': 'none',
        'chat_template_kwargs': {'enable_thinking': False},
    }
    req = urllib.request.Request(upstream + '/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first = None
    timings = None
    finish = None
    done = False
    output_hash = hashlib.sha256()
    with open_request(req, timeout=timeout) as response:
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
            if item.get('strata_generation_failure'):
                raise RuntimeError(f'{name}: generation was stopped or the retry did not finish')
            if item.get('timings'):
                timings = item['timings']
            for choice in item.get('choices') or []:
                delta = choice.get('delta') or {}
                if delta.get('content'):
                    output_hash.update(delta['content'].encode())
                if first is None and (delta.get('content') or delta.get('reasoning_content')):
                    first = elapsed
                finish = choice.get('finish_reason') or finish
    if not done or first is None or finish not in ('stop', 'length'):
        raise RuntimeError(f'{name}: benchmark stream did not finish normally')
    if not isinstance(timings, dict) or any(
        not isinstance(timings.get(k), (int, float)) or not math.isfinite(timings[k]) or timings[k] <= 0
        for k in ('prompt_ms', 'predicted_ms', 'predicted_per_second', 'predicted_n')
    ):
        raise RuntimeError(f'{name}: engine did not report valid timings')
    if timings['predicted_n'] < 2:
        raise RuntimeError(f'{name}: too few generated tokens to measure decode')

    row = {'case': name, 'first_content_ms': round(first * 1000, 1),
           'wall_ms': round((time.monotonic() - started) * 1000, 1), 'finish': finish,
           'timings': timings, 'output_sha256': output_hash.hexdigest()}
    if log and log.is_file():
        with log.open('rb') as f:
            f.seek(offset)
            matches = ROUNDS.findall(f.read())
        if matches and int(matches[-1][0]) == timings['predicted_n'] and int(matches[-1][1]) > 0:
            row['verify_rounds'] = int(matches[-1][1])
            row['tokens_per_round'] = round(timings['predicted_n'] / row['verify_rounds'], 3)
    accepted = timings.get('draft_n_accepted')
    if isinstance(accepted, int) and 0 <= accepted < timings['predicted_n']:
        row['estimated_tokens_per_round'] = round(timings['predicted_n'] / (timings['predicted_n'] - accepted), 3)
    return row


def save_report(directory, report):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    run = directory / (report['run'] + '.json')
    latest = directory / 'latest.json'
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=directory, delete=False) as f:
        tmp = Path(f.name)
        f.write(encoded)
    try:
        os.replace(tmp, run)
        os.link(run, tmp)
        os.replace(tmp, latest)
        if report['status'] == 'passed' and report['drafting'] == 'dflash2':
            try:
                os.link(run, directory / 'baseline-dflash2.json')
            except FileExistsError:
                pass
    finally:
        tmp.unlink(missing_ok=True)
    return run


def run_suite(upstream, cfg, settings, seed):
    now = datetime.datetime.now().astimezone()
    report = {
        'suite': SUITE, 'run': now.strftime('%Y%m%d-%H%M%S-%f'), 'started_at': now.isoformat(),
        'drafting': cfg.get('drafting'), 'engine': cfg.get('exe'), 'engine_args': cfg.get('args'),
        'prior': cfg.get('env', {}).get('STRATA_ROUTE_PRIOR'), 'expert_profile_sha256': seed,
        'status': 'failed', 'cases': [],
    }
    directory = Path(settings['output_dir'])
    log = Path(cfg['log']) if cfg.get('log') else None
    if log and log.is_file():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - 65536))
            tail = f.read().decode(errors='replace').rsplit('engine started:', 1)
        if len(tail) == 2:
            launch, *lines = tail[-1].splitlines()
            report['effective_engine_command'] = launch.strip()
            report['startup_resources'] = [line for line in lines if any(marker in line for marker in (
                'expert cache auto:', 'resident RAM mode:', 'lendable slots', 'PCIe probe:',
                'VRAM free with everything loaded', 'DFlash2 drafter on', 'MTP drafter on',
            ))]
    print('Strata 正在执行开机基准，外部端口尚未开放。', flush=True)
    try:
        for name, prompt, limit in CASES:
            row = run_case(upstream, name, prompt, limit, log, timeout=90)
            report['cases'].append(row)
            t = row['timings']
            print(f"Strata 开机基准 {name}：首字 {row['first_content_ms']:.0f} ms，"
                  f"预填充 {t['prompt_ms']:.0f} ms，解码 {t['predicted_per_second']:.1f} token/s"
                  f"（输出 {t['predicted_n']} token）。", flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report['error'] = str(exc)
        save_report(directory, report)
        raise RuntimeError(f'Strata 开机基准失败，外部端口未开放：{exc}') from exc
    path = save_report(directory, report)
    print(f'Strata 开机基准完成，报告：{path}。即将开放外部端口。', flush=True)
    return report
