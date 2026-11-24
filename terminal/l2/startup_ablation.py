"""Flip each PR switch from its production value before binding the public API."""
import copy
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from l2.runtime_config import set_options
from l2.startup_benchmark import run_case


SWITCHES = (
    ('pr6-k8v4-tc', 'STRATA_ATTN_DECODE_TC', '1'),
    ('pr7-tq-f16', 'STRATA_PF_FUSED_TQ_F16', '1'),
    ('pr8-adapt-fetch', 'STRATA_ADAPT_FETCH', '2'),
    ('pr9-hc-persist', 'STRATA_HC_PERSIST', '1'),
)
OUTPUT_TOKENS = 320
PROMPT = '\n'.join(
    f'Incident {i}: service {service} received duplicate abort messages during restart. '
    f'The restored checkpoint was {i * 17} events behind the durable log.'
    for i, service in enumerate(('billing', 'search', 'queue', 'storage') * 12, 1)
) + ('\nWrite a detailed incident review and propose a Python replay algorithm. '
     'Include tradeoffs, failure modes, pseudocode, and a numbered test plan with 20 distinct cases.')
WINDOW = re.compile(r'strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window')


def headless_boot(cmdline=Path('/proc/cmdline')):
    return 'systemd.unit=strata.target' in cmdline.read_text().split()


def meminfo():
    return {key: int(value) // 1024 for key, value in
            re.findall(r'^(\w+):\s+(\d+) kB', Path('/proc/meminfo').read_text(), re.M)}


def trial_config(cfg, folder, seed, switch, trial_value):
    trial = copy.deepcopy(cfg)
    for k in ('slot_save_path', 'startup_benchmark', 'l2_backend_port', 'l2_disk_budget_mib', 'startup_pr_ablation'):
        trial.pop(k, None)
    for k in ('args', 'user_args'):
        args = trial.get(k, []).copy()
        while '--expert-profile-save' in args:
            pos = args.index('--expert-profile-save')
            del args[pos:pos + 2]
        trial[k] = set_options(args, {'--expert-profile': seed})
    trial.update(host='127.0.0.1', port=0, offline=True, api_key='', log=str(folder / 'engine.log'), engine_ready_s=85)
    trial['env'].update(STRATA_CONV_DISK_DIR=str(folder / 'parked'), STRATA_DRAFT_LOG='1', STRATA_DECODE_TIMING='1')
    if switch:
        trial['env'][switch] = trial_value
    return trial


def stop_trial(proc):
    if proc.poll() is not None:
        return
    # The HTTP server gives the engine its own process group.
    children = []
    try:
        for path in Path(f'/proc/{proc.pid}/task').glob('*/children'):
            for child in path.read_text().split():
                try:
                    children.append(os.pidfd_open(int(child)))
                except ProcessLookupError:
                    continue
    except FileNotFoundError:
        pass
    try:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=4)
        for handle in children:
            try:
                signal.pidfd_send_signal(handle, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        for handle in children:
            os.close(handle)


def serve_trial(root, trial, folder, end, result):
    profile = folder / 'profile.json'
    profile.write_text(json.dumps(trial))
    ready = folder / 'ready.json'
    log = Path(trial['log'])
    (folder / 'parked').mkdir()
    env = {k: v for k, v in os.environ.items() if k != 'STRATA_API_KEY'}
    env.update(STRATA_NO_ACQUIRE='1', STRATA_RUNTIME_DIR=str(folder / 'runtime'))
    mem = meminfo()
    initial_swap = mem['SwapTotal'] - mem['SwapFree']
    result['memory'] = {'min_available_mib': mem['MemAvailable'], 'max_new_swap_mib': 0}
    stop = threading.Event()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (folder / 'server.log').open('wb') as server_log:
        proc = subprocess.Popen([sys.executable, str(root / 'strata.py'), 'serve', '--profile', str(profile),
                                 '--ready-file', str(ready), '--shutdown-timeout', '6'],
                                cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=server_log,
                                stderr=subprocess.STDOUT, start_new_session=True)

        def watch():
            while not stop.wait(0.5):
                sample = meminfo()
                memory = result['memory']
                memory['min_available_mib'] = min(memory['min_available_mib'], sample['MemAvailable'])
                memory['max_new_swap_mib'] = max(memory['max_new_swap_mib'],
                                                 sample['SwapTotal'] - sample['SwapFree'] - initial_swap)
                if time.monotonic() >= end or memory['min_available_mib'] < 1024 or memory['max_new_swap_mib'] > 512:
                    result['limit_hit'] = 'time' if time.monotonic() >= end else 'memory'
                    proc.send_signal(signal.SIGINT)
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            while not ready.exists():
                if proc.poll() is not None:
                    raise RuntimeError(f'engine exited before readiness ({proc.returncode})')
                if time.monotonic() >= end or result.get('limit_hit'):
                    raise TimeoutError('trial did not reach readiness within its budget')
                time.sleep(0.2)
            data = json.loads(ready.read_text())
            port = data.get('bound_port')
            if (data.get('pid') != proc.pid or data.get('api_state') != 'ready' or data.get('model_state') != 'ready' or
                    data.get('bound_host') != '127.0.0.1' or type(port) is not int or not 1 <= port <= 65535):
                raise RuntimeError('invalid loopback readiness record')
            startup = log.read_text(errors='replace')
            if not re.search(r'resident RAM mode:[^\n]*page-locked', startup) or 'DFlash2 drafter on' not in startup:
                raise RuntimeError('trial did not reach resident RAM + DFlash2 mode')
            result['resources'] = [line for line in startup.splitlines() if any(
                marker in line for marker in ('expert cache auto:', 'resident RAM mode:', 'STRATA_ADAPT_FETCH=2:',
                                              'Q8_0 persistent read residency checked', 'prompt chunk auto:'))]
            endpoint = f'http://127.0.0.1:{port}'
            for name, prompt, tokens in (
                ('warmup', 'Explain database transaction isolation in a short paragraph.', 40),
                ('timed', PROMPT, OUTPUT_TOKENS),
            ):
                remaining = end - time.monotonic()
                if remaining <= 0 or result.get('limit_hit'):
                    raise TimeoutError('trial exceeded its budget')
                offset = log.stat().st_size
                row = run_case(endpoint, name, prompt, tokens, log, min(45, remaining), open_request=opener.open)
                wait_until = min(end, time.monotonic() + 1.0)
                while True:
                    with log.open('rb') as f:
                        f.seek(offset)
                        detail = f.read().decode(errors='replace')
                    reads = re.findall(r'(\d+) blob reads from the file', detail)
                    if reads or time.monotonic() >= wait_until:
                        break
                    time.sleep(0.03)
                match = WINDOW.search(detail)
                if match:
                    row.update(verify_rounds=int(match[1]), verify_width_mean=float(match[2]),
                               tokens_per_round=float(match[3]), ms_per_round=float(match[4]))
                    # Keep the engine's existing breakdown. "GPU-reach wait" includes scheduling
                    # and synchronization, so it must not be reported as the isolated kernel time.
                    timing_line = detail[match.start():].splitlines()[0]
                    row['decode_timing'] = timing_line
                    cpu = re.search(r'CPU ([\d.]+)\]', timing_line)
                    if cpu:
                        row['cpu_expert_ms_per_round'] = float(cpu[1])
                hit = re.search(r'decode expert cache hit rate: ([\d.]+)%', detail)
                if hit:
                    row['vram_hit_pct'] = float(hit[1])
                row['expert_file_reads'] = max(map(int, reads)) if reads else None
                result[name] = row
            if result.get('limit_hit'):
                raise RuntimeError('trial reached its time or memory guard')
            if row['expert_file_reads'] != 0 or row['timings']['predicted_n'] < OUTPUT_TOKENS * 0.75:
                raise RuntimeError('timed output was too short or zero expert file reads could not be confirmed')
            result['status'] = 'measured'
        finally:
            stop.set()
            watcher.join(timeout=1)
            stop_trial(proc)


def run_at_boot(root, cfg, directory, settings, scratch_parent):
    """A diagnostic failure is recorded before the normal service starts."""
    if not headless_boot():
        return
    report = {'suite': 'startup-pr-ablation-v2', 'status': 'incomplete', 'runs': []}
    destination = directory / 'startup-pr-ablation-latest.json'
    try:
        deadline = time.monotonic() + int(settings.get('budget_s', 330))
        args = cfg['args']
        profile = Path(args[args.index('--expert-profile') + 1])
        with Path(cfg['exe']).open('rb') as f:
            report['engine_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
        report['fixed_environment'] = {key: cfg['env'].get(key) for key in
                                       ('STRATA_ROUTE_TAIL_SKIP', 'STRATA_ROUTE_PRIOR', 'STRATA_TQ_AMPERE_ROWS')}
        report['baseline_switches'] = {key: cfg['env'].get(key) for _, key, _ in SWITCHES}
        print('Strata 开机 PR 消融开始，正式 API 将在测试后开放。', flush=True)
        with tempfile.TemporaryDirectory(prefix='pr-ablation-', dir=scratch_parent) as temp:
            scratch = Path(temp)
            seed = scratch / 'profile.bin'
            seed.write_bytes(profile.read_bytes())
            report['expert_profile_sha256'] = hashlib.sha256(seed.read_bytes()).hexdigest()
            for name, switch, enabled_value in (('baseline', None, None), *SWITCHES):
                if time.monotonic() >= deadline:
                    break
                result = {'arm': name, 'switch': switch, 'status': 'failed'}
                report['runs'].append(result)
                trial_value = None
                label = '生产基线'
                if switch:
                    production_value = report['baseline_switches'][switch]
                    if production_value not in ('0', enabled_value):
                        result.update(status='skipped', error='production must explicitly select 0 or the tested enabled value')
                        continue
                    trial_value = enabled_value if production_value == '0' else '0'
                    result.update(production_value=production_value, trial_value=trial_value)
                    label = f'{name}（{switch}: {production_value} → {trial_value}）'
                print(f'Strata PR 消融 {label} 开始。', flush=True)
                folder = scratch / name
                folder.mkdir()
                try:
                    serve_trial(root, trial_config(cfg, folder, seed, switch, trial_value), folder,
                                min(deadline, time.monotonic() + 85), result)
                    row = result['timed']
                    timing = row['timings']
                    if switch:
                        baseline = report['runs'][0]['timed']['timings']
                        result['delta_vs_production'] = {
                            k: round(timing[k] - baseline[k], 3) for k in ('prompt_ms', 'predicted_per_second')
                        }
                    rounds = ''
                    if 'ms_per_round' in row:
                        rounds = f"（每窗口 {row['ms_per_round']:.2f} ms，产出 {row['tokens_per_round']:.2f} token）"
                    print(f"Strata PR 消融 {label}：预填充 {timing['prompt_ms']:.0f} ms，"
                          f"解码 {timing['predicted_per_second']:.1f} token/s{rounds}。", flush=True)
                except Exception as exc:
                    result['error'] = str(exc)
                    print(f'Strata PR 消融 {label}：{exc}', flush=True)
                    if name == 'baseline' or result.get('limit_hit') == 'memory':
                        break
        if len(report['runs']) == 1 + len(SWITCHES) and all(r['status'] == 'measured' for r in report['runs']):
            report['status'] = 'measured'
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        raise
    except Exception as exc:
        report['error'] = str(exc)
        print(f'Strata 开机 PR 消融中止：{exc}', flush=True)
    finally:
        try:
            with tempfile.NamedTemporaryFile('w', dir=directory, encoding='utf-8', delete=False) as f:
                temp = Path(f.name)
                try:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                    f.close()
                    os.replace(temp, destination)
                finally:
                    temp.unlink(missing_ok=True)
        except OSError as exc:
            print(f'Strata PR 消融报告写入失败：{exc}', flush=True)
        continuation = '，继续启动正式服务' if report['status'] != 'interrupted' else ''
        print(f"Strata 开机 PR 消融结束（{report['status']}）{continuation}。报告：{destination}", flush=True)
