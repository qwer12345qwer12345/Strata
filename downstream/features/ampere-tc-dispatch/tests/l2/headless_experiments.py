"""One-shot headless spec, SM86 shape and wide lookup comparisons. Never select a winner automatically."""
import datetime
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from l2.chat_latency import measure, workloads
from l2.runtime_config import DAILY, PREVIOUS, REFERENCE, set_options
WINDOW = re.compile(r'strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window')

def gpu_cache_slots_from_log(log_text):
    match = re.search(r'expert cache (\d+) slots,', log_text)
    return int(match.group(1)) if match else None

def validate_ampere_dispatch(log_text, env):
    requested = env.get('STRATA_TQ_AMPERE_ROWS') == '2'
    observed = 'TQ Ampere paired-row GU enabled' in log_text
    if observed != requested:
        raise RuntimeError(
            f'actual Ampere paired-row dispatch marker={observed}, '
            f'requested STRATA_TQ_AMPERE_ROWS={env.get("STRATA_TQ_AMPERE_ROWS")!r}'
        )
    return observed

def long_fixture_validity_failures(cases):
    required = {'long_output', 'long_context', 'second_output'}
    return [row.get('case') for row in cases
            if row.get('case') in required and row.get('target_tokens', 0) >= 1024
            and row.get('timings', {}).get('predicted_n', 0) < row['target_tokens'] * .75]

def meminfo():
    lines = Path('/proc/meminfo').read_text()
    return {k: int(v) // 1024 for k, v in re.findall(r'^(\w+):\s+(\d+) kB', lines, re.M)}

def check_environment(*, allow_strata=False):
    desktop = subprocess.check_output(
        ['systemctl', 'show', '-p', 'ActiveState', '--value', 'sddm.service'], text=True
    ).strip()
    if desktop != 'inactive':
        raise RuntimeError(f'SDDM state is {desktop}')
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            name = path.read_text().strip()
        except OSError:
            continue
        is_strata = name == 'strata' or name.startswith('strata-pr5500-')
        if (is_strata and not allow_strata) or name in {'ninfer-serve', 'ininfer', 'ninja', 'make', 'cc1plus', 'nvcc', 'ptxas', 'cicc'}:
            raise RuntimeError(f'{name} is already running')

def stop_trial(proc):
    # The upstream server puts its engine in a separate process group.
    # Keep PID handles in case the server cannot reap them after SIGINT.
    handles = []
    for path in Path(f'/proc/{proc.pid}/task').glob('*/children'):
        try:
            children = path.read_text().split()
        except OSError:
            continue
        for child in children:
            try:
                handles.append(os.pidfd_open(int(child)))
            except ProcessLookupError:
                pass
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for handle in handles:
            try:
                signal.pidfd_send_signal(handle, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        for handle in handles:
            os.close(handle)

def run_phase(root, cfg, phase, folder, seed, port, deadline, candidate=None, *,
              workloads=None, launch_profile=None, validate_ampere=False, capture_output=False):
    report = {'phase': phase, 'status': 'failed', 'cases': []}
    if time.monotonic() >= deadline:
        report['error'] = 'startup comparison time budget was exhausted'
        return report
    folder.mkdir()
    ready, profile, log = (folder / 'ready.json', folder / 'profile.json', folder / 'engine.log')
    if launch_profile is None:
        raise ValueError('explicit experiment profile required')
    launch_cfg = dict(launch_profile, host='127.0.0.1', port=port, log=str(log), engine_ready_s=180)
    residency_trial = phase == 'gyro-m-32k'
    (folder / 'parked').mkdir()
    profile.write_text(json.dumps(launch_cfg, indent=2) + '\n')
    cmd = [sys.executable, str(root / 'strata.py'), 'serve', '--profile', str(profile),
           '--ready-file', str(ready), '--shutdown-timeout', '8']
    stop = threading.Event()
    mem = meminfo()
    start_swap = mem['SwapTotal'] - mem['SwapFree']
    resources = {'min_available_mib': mem['MemAvailable'], 'max_new_swap_mib': 0}
    report['memory'] = resources
    watcher = None
    proc = None
    try:
        check_environment()
        with (folder / 'server.log').open('wb') as journal:
            proc = subprocess.Popen(cmd, cwd=root, stdin=subprocess.DEVNULL, stdout=journal,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, 'STRATA_NO_ACQUIRE': '1'})
        def watch():
            while not stop.wait(.4):
                sample = meminfo()
                resources['min_available_mib'] = min(resources['min_available_mib'], sample['MemAvailable'])
                resources['max_new_swap_mib'] = max(
                    resources['max_new_swap_mib'], sample['SwapTotal'] - sample['SwapFree'] - start_swap
                )
                if time.monotonic() >= deadline or sample['MemAvailable'] < 1024 or resources['max_new_swap_mib'] > 1024:
                    report['error'] = 'startup comparison reached its time or memory limit'
                    if proc.poll() is None:
                        proc.send_signal(signal.SIGINT)
                    break
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        ready_limit = min(deadline, time.monotonic() + 180)
        while not ready.exists():
            if proc.poll() is not None:
                lines = log.read_text(errors='replace').strip().splitlines() if log.is_file() else []
                reason = lines[-1] if lines else 'engine log is empty'
                raise RuntimeError(f'trial exited before ready (status {proc.returncode}): {reason}')
            if time.monotonic() >= ready_limit:
                raise RuntimeError('trial readiness timed out')
            time.sleep(.2)
        health = json.loads(ready.read_text())
        if health.get('api_state') != 'ready' or health.get('pid') != proc.pid:
            raise RuntimeError('trial readiness belongs to another process')
        startup = log.read_text(errors='replace')
        report['effective_engine_command'] = next(
            (line.split('engine started:', 1)[1].strip() for line in startup.splitlines() if 'engine started:' in line), None
        )
        report['resources'] = [line for line in startup.splitlines() if any(s in line for s in (
            'expert cache auto:', 'resident RAM mode:', 'lendable slots', 'DFlash2 drafter on',
            'draft layer loaded', 'PCIe probe:', 'VRAM free with everything loaded',
            'fused', 'prompt chunk auto', 'whole resident RAM mode does not fit',
            'room for the complement:', 'file tier', 'pinned RAM', 'expert cache ',
        ))]
        report['gpu_cache_slots'] = gpu_cache_slots_from_log(startup)
        if not re.search(r'resident RAM mode:[^\n]*page-locked', startup):
            raise RuntimeError('trial did not page-lock the RAM expert complement')
        if residency_trial:
            if any(flag in report['effective_engine_command'].split() for flag in ('--mtp', '--draft')):
                raise RuntimeError('Gyro-M residency trial unexpectedly loaded a drafter')
            if 'whole resident RAM mode does not fit' in startup or 'FileExpertSource: RAM budget ' in startup:
                raise RuntimeError('the full Gyro-M decode expert complement did not fit')
        else:
            expected = 'strata mtp: draft layer loaded' if phase == 'mtp4' else 'DFlash2 drafter on'
            if expected not in startup:
                raise RuntimeError('trial did not enable the requested drafter')
        if candidate is not None and not any(
            'strata: prompt kernels:' in line and 'fused' in line for line in startup.splitlines()
        ):
            raise RuntimeError('candidate did not report fused prompt kernels')
        for workload in (workloads or ()):
            label, prompt, limit = workload[:3]
            options = workload[3] if len(workload) > 3 else {}
            offset = log.stat().st_size
            remaining = deadline - time.monotonic()
            if remaining <= 0 or 'error' in report:
                raise RuntimeError('trial interrupted')
            endpoint = f'http://127.0.0.1:{port}'
            timeout = min(300, remaining)
            row = measure(endpoint, label, prompt, limit, timeout,
                          capture_output=capture_output, **options)
            # The engine flushes DONE before appending these expert counters.
            log_deadline = min(deadline, time.monotonic() + 1.0)
            while True:
                with log.open('rb') as source:
                    source.seek(offset)
                    detail = source.read().decode(errors='replace')
                reads = re.findall(r'(\d+) blob reads from the file', detail)
                expert_mb = re.findall(r'([\d.]+) MB of expert reads', detail)
                if (reads and (not residency_trial or expert_mb)) or time.monotonic() >= log_deadline:
                    break
                time.sleep(.02)
            m = WINDOW.search(detail)
            if m:
                row.update(verify_rounds=int(m[1]), verify_width_mean=float(m[2]),
                           tokens_per_round=float(m[3]), ms_per_round=float(m[4]))
            if reads:
                row['expert_file_reads'] = max(map(int, reads))
            gpu_line = next((line for line in detail.splitlines()
                             if 'strata decode GPU stages (ms/window):' in line), '')
            if gpu_line:
                row['gpu_resident_expert_ms_per_round'] = round(
                    sum(map(float, re.findall(r'VRAM hits ([\d.]+)', gpu_line))), 3
                )
            if expert_mb:
                row['expert_mb_accounted'] = float(expert_mb[-1])
            if capture_output:
                from l2.quality_diagnostics import analyze_case
                row['quality'] = analyze_case(row)
            report['cases'].append(row)
            t = row['timings']
            print(f"Strata 无头对照 {phase}/{label} 预填充 {t['prompt_ms']:.0f} ms，"
                  f"解码 {t['predicted_per_second']:.1f} token/s，输出 {t['predicted_n']} token。", flush=True)
        if 'error' not in report:
            if validate_ampere:
                validate_ampere_dispatch(log.read_text(errors='replace'), launch_cfg['env'])
            if not all(row.get('expert_file_reads') == 0 for row in report['cases']):
                raise RuntimeError('zero decode expert file reads could not be confirmed')
            if residency_trial and not all(row.get('expert_mb_accounted') == 0
                                           for row in report['cases'] if row['case'] != 'warmup'):
                raise RuntimeError('zero request-time expert file MB could not be confirmed')
            report['status'] = 'passed'
    except Exception as exc:
        report.setdefault('error', str(exc))
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=1)
        if proc is not None:
            stop_trial(proc)
        (folder / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report

def arms(groups=None):
    def arm(group, label, changes):
        return {**DAILY, **changes, 'group':group, 'name':label}
    def mirror(group, fields, cases):
        return [arm(group, group+'-'+('-'.join(map(str, values))),dict(zip(fields,values)))
                for values in list(cases)+list(reversed(cases))]
    result = [arm('combined','combined-'+label,cfg) for label,cfg in
              [('previous',PREVIOUS),('daily',DAILY),('daily',DAILY),('previous',PREVIOUS)]]
    # Cross the interacting choices instead of assuming each local winner composes.
    spec_lookup = [(spec,lookup) for i,spec in enumerate((6,4,3))
                   for lookup in ((8,12,16) if i%2==0 else (16,12,8))]
    result += mirror('spec-lookup',('spec','lookup'),spec_lookup)
    result += mirror('shape-qkv',('shapes','tc'),[(1,2),(1,0),(2,0),(2,2)])
    result += mirror('dense-lookup',('dense','lookup'),[(0,8),(1,8),(1,16),(0,16)])
    result += mirror('skip-lookup',('skip','lookup'),[(0,8),(8,8),(8,16),(0,16)])
    result += mirror('rows',('rows',),[(2,),(0,)])
    result += mirror('adapt-tail',('nowait','tail'),[(0,0),(1,0),(1,7),(0,7)])
    result += mirror('tail-spec',('tail','spec'),[(0,6),(7,6),(7,4),(0,4),(0,3),(7,3)])
    # Numerical differences are not an exclusion: compare interacting choices
    # under the same residency and long-output requirements as the existing arms.
    result += mirror('emb-gdn',('emb','gdn_np'),[(0,3),(1,3),(1,2),(0,2)])
    result += mirror('gdn-precision',('gdn_np',),[(2,),(1,)])
    result += mirror('hc-upmix',('upmix',),[(0,),(1,)])
    result += mirror('gdn-execution',('gdn_tiles','gdn_fuse'),[(1,1),(2,1),(2,0),(1,0)])
    result += mirror('prefill-cap',('prefill_cap',),[(8192,),(16384,),(32768,)])
    result += mirror('cpu-prefill',('cpu_share','cpu_nt','cpu_max'),[(0,0,0),(1,32,3072),(1,32,6144),(1,8,6144)])
    result += mirror('adapt-mode',('nowait','lag','overlap','fetch'),[(1,2,0,2),(1,2,0,0),(0,1,0,0),(0,1,1,0)])
    for engine in (0,1,1,0):
        result.append(arm('engine-reference','engine-reference-'+str(engine),{**REFERENCE,'engine':engine}))
    official=dict(decode_tc=0,tq_f16=0,fetch=0,hc_persist=0)
    for label,changes in [('official',official),('daily',{}),('daily',{}),('official',official)]:
        result.append(arm('update-bundle','update-bundle-'+label,changes))
    result += mirror('decode-tc-spec',('decode_tc','spec'),[(0,4),(1,4),(1,6),(0,6)])
    result += mirror('tq-prefill',('tq_f16','prefill_cap'),[(0,8192),(1,8192),(1,16384),(0,16384)])
    result += mirror('hc-qfuse',('hc_persist','qfuse'),[(0,0),(1,0),(1,1),(0,1)])
    result += mirror('fetch-pin',('fetch','pin'),[(0,0),(2,0),(2,0.25),(0,0.25)])
    result += mirror('fetch-mode',('fetch',),[(0,),(1,),(2,)])
    result += mirror('upstream-adapt',('adapt_every','adapt_swaps','adapt_decay'),[(4,96,0.7),(2,192,0.92)])
    result += mirror('qfuse-l3',('qfuse','l3'),[(0,0),(1,0),(1,1),(0,1)])
    result += mirror('dense-min',('dense_min',),[(9,),(6,),(3,)])
    if groups:
        # Explicit diagnostics stay out of the default 180-arm --all sequence.
        if 'ampere-path' in groups:
            result += [arm('ampere-path','ampere-path-off',{'rows':0}),
                       arm('ampere-path','ampere-path-paired',{'rows':2})]
        if 'tail-fetch' in groups:
            result += mirror('tail-fetch',('tail','fetch'),[(0,0),(7,0),(7,2),(0,2)])
        if 'updates' in groups:
            groups=[g for g in groups if g!='updates']+['engine-reference','update-bundle','decode-tc-spec','tq-prefill','hc-qfuse','fetch-pin','fetch-mode','upstream-adapt','qfuse-l3','dense-min']
        if 'numeric' in groups:
            groups=[g for g in groups if g!='numeric']+['emb-gdn','gdn-precision','hc-upmix','gdn-execution','prefill-cap','cpu-prefill','adapt-mode']
        aliases = {'spec':'spec-lookup','lookup':'spec-lookup','shape':'shape-qkv',
                   'qkv-tc':'shape-qkv','dense':'dense-lookup','lookup-skip':'skip-lookup',
                   'nowait':'adapt-tail','tail-skip':'adapt-tail','embedding':'emb-gdn','gdn':'emb-gdn'}
        wanted = {aliases.get(group,group) for group in groups}
        unknown = wanted - {r['group'] for r in result}
        if unknown:raise ValueError('Unknown comparison groups: '+', '.join(sorted(unknown)))
        result = [r for r in result if r['group'] in wanted]
    return result


def comparison_workloads(group):
    cases=list(workloads()[:3])
    if group=='ampere-path':
        return [cases[0]]
    if group=='tail-fetch':
        from l2.quality_diagnostics import objective_workloads
        return cases + objective_workloads()
    if group=='cpu-prefill':
        # The long prompt can exceed the share's range; include a follow-up-sized
        # prompt so extending that range is actually exercised, not just requested.
        records=[line for line in cases[2][1].splitlines() if line.startswith('Record ')][:60]
        prompt='Use these incident records to write a detailed recovery handbook.\n'+'\n'.join(records)
        prompt+='\nGive at least 30 fully written procedures, with diagnosis, actions, verification and failure examples. Do not abbreviate.'
        cases.append(('medium_context',prompt,1536))
    return cases


def configure(base, arm, folder, seed, port):
    spec, shape, lookup = arm['spec'], arm['shapes'], arm['lookup']
    cfg = json.loads(json.dumps(base))
    args = cfg['args']
    for flag in ('--expert-profile-save', '--expert-profile', '--conversation-cache-mib',
                 '--conversation-cache-slots', '--spec', '--draft-conf', '--lookup-max-t',
                 '--lookup-sizes', '--spec-min-p', '--prefill', '--adapt-every', '--adapt-swaps', '--adapt-decay'):
        while flag in args:
            at = args.index(flag)
            del args[at:at + 2]
    sizes = list(range(2, 9)) + ([10, 12] if lookup >= 12 else []) + ([16] if lookup == 16 else [])
    options = ['--spec', str(spec), '--draft-conf', '0.8', '--lookup-max-t', str(lookup),
               '--lookup-sizes', ','.join(map(str, sizes)), '--prefill', f'auto:{arm["prefill_cap"]}', '--adapt-every', str(arm['adapt_every']), '--adapt-swaps', str(arm['adapt_swaps']), '--adapt-decay', str(arm['adapt_decay'])]
    args += ['--expert-profile', str(seed), '--conversation-cache-mib', '0',
             '--conversation-cache-slots', '0'] + options
    for key in ('startup_benchmark', 'slot_save_path', 'l2_backend_port', 'l2_disk_budget_mib'):
        cfg.pop(key, None)
    cfg.update(host='127.0.0.1', port=port, log=str(folder / 'engine.log'),
               drafting='dflash2', mtp=False, user_args=options, engine_ready_s=180)
    if str(cfg['env'].get('STRATA_ROUTE_PRIOR')) != '0.5':
        raise ValueError('prior=0.5 is required')
    cfg['env'].update(STRATA_ADAPT_NOWAIT=str(arm['nowait']), STRATA_ADAPT_LAG=str(arm['lag']),
                      STRATA_ADAPT_OVERLAP=str(arm['overlap']), STRATA_ADAPT_OVERLAP_RESIDENT=str(arm['overlap']),
                      STRATA_ROUTE_TAIL_SKIP=str(arm['tail']), STRATA_ROUTE_TAIL_RENORM='0',
                      STRATA_TQ_AMPERE_ROWS=str(arm['rows']), STRATA_DRAFT_LOG='1', STRATA_DECODE_TIMING='1',
                      STRATA_DF2_AMPERE_TUNING=str(shape), STRATA_DENSE_I8_AMPERE_SHAPES=str(arm['dense']),
                      STRATA_DF2_GEMM_TC=str(arm['tc']), STRATA_DF2_LOOKUP_SKIP=str(arm['skip']), STRATA_DF2_WIDE_CONTEXT='1' if lookup > 8 else '0',
                      STRATA_SPEC_TRACE=str(folder / 'spec.trace'),
                      STRATA_CONV_DISK_DIR=str(folder / 'parked'))
    cfg['env'].update(STRATA_EMB_REUSE_ACCOUNT=str(arm['emb']), STRATA_GDN_TC='1',
                      STRATA_GDN_TC_NP=str(arm['gdn_np']), STRATA_HC_UPMIX=str(arm['upmix']),
                      STRATA_GDN_TC_TILES=str(arm['gdn_tiles']), STRATA_GDN_TC_FUSE=str(arm['gdn_fuse']),
                      STRATA_PREFILL_TIMING='1')
    cfg['env']['STRATA_PREFILL_CPU_SHARE']='auto' if arm['cpu_share'] else '0'
    cfg['env']['STRATA_POOL_STATIC']='0'
    for key, field in (('STRATA_PREFILL_CPU_SHARE_NT','cpu_nt'),('STRATA_PREFILL_CPU_SHARE_MAX','cpu_max')):
        cfg['env'].pop(key,None)
        if arm[field]:cfg['env'][key]=str(arm[field])
    cfg['env'].update(STRATA_ATTN_DECODE_TC=str(arm['decode_tc']), STRATA_PF_FUSED_TQ_F16=str(arm['tq_f16']),
                      STRATA_PF_BATCH='256', STRATA_ADAPT_FETCH=str(arm['fetch']), STRATA_HC_PERSIST=str(arm['hc_persist']),
                      STRATA_QFUSE=str(arm['qfuse']), STRATA_L3_PF=str(arm['l3']), STRATA_CACHE_PIN_FRAC=str(arm['pin']),
                      STRATA_DENSE_I8_TC_MIN_T=str(arm['dense_min']))
    return cfg


def summarize(runs):
    grouped = defaultdict(list)
    for run in runs:
        if run['status'] == 'passed':
            for row in run['cases']:
                if row['case'] != 'warmup':
                    grouped[run['phase'], row['case']].append(row)
    result = {}
    for (phase, case), rows in grouped.items():
        result.setdefault(phase, {})[case] = {
            'runs': len(rows),
            'median_decode_tps': statistics.median(r['timings']['predicted_per_second'] for r in rows),
            'median_prefill_ms': statistics.median(r['timings']['prompt_ms'] for r in rows),
            'median_first_content_ms': statistics.median(r['first_content_ms'] for r in rows),
            'median_gap_p99_ms': statistics.median(r['stream'].get('gap_p99_ms', 0) for r in rows),
            'outputs_sha256': sorted(set(r['output_sha256'] for r in rows)),
        }
    return result


def optimization_log_summary(text):
    tail = re.findall(r'route tail skip: (\d+) missed experts skipped \((\d+) entries\)',text)
    wait = re.findall(r'strata decode timing:[^\n]*?adapt wait ([\d.]+)',text)
    tail_settings = [line.strip() for line in text.splitlines()
                     if 'STRATA_ROUTE_TAIL_SKIP=' in line and ' is ON:' in line]
    fetch_settings = [line.strip() for line in text.splitlines()
                      if 'STRATA_ADAPT_FETCH' in line and
                      ('strata serve:' in line or 'strata generate:' in line)]
    return {'tail_skip_reported': bool(tail_settings),
            'tail_skip_setting_lines': tail_settings,
            'fetch_effective_setting_lines': fetch_settings,
            'tail_skip_cumulative_samples': [{'experts':int(e),'entries':int(n)} for e,n in tail],
            'adapt_wait_ms_per_window_samples':list(map(float,wait))}


def trace_summary(path):
    result = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        if line.startswith('# request'):
            result.append({'neural': {'rounds': 0, 'offered': 0, 'accepted': 0, '_ms': []},
                           'lookup': {'rounds': 0, 'offered': 0, 'accepted': 0, '_ms': []},
                           'widths': {}, '_width_ms': {}})
        elif line.startswith('R ') and result:
            fields = line.split('|', 1)[0].split()
            width, accepted, lookup = map(int, fields[2:5])
            elapsed = float(fields[5])
            row = result[-1]
            source = row['lookup' if lookup else 'neural']
            source['rounds'] += 1
            source['offered'] += width - 1
            source['accepted'] += accepted
            source['_ms'].append(elapsed)
            row['widths'][str(width)] = row['widths'].get(str(width), 0) + 1
            row['_width_ms'].setdefault(str(width), []).append(elapsed)
    def percentiles(samples):
        values = sorted(samples)
        return {'samples': len(values), **{
            f'p{p}': values[math.ceil(len(values) * p / 100) - 1] for p in (50, 95, 99)
        }} if values else {'samples': 0}
    for row in result:
        for source in ('neural', 'lookup'):
            row[source]['round_latency_ms'] = percentiles(row[source].pop('_ms'))
        row['round_latency_ms_by_width'] = {width: percentiles(samples)
                                           for width, samples in row.pop('_width_ms').items()}
    return result


def run_comparison(root, cfg, directory, port, request):
    sequence = arms(request.get("groups"))
    groups = request.get('groups') or []
    label = '-'.join(sorted(groups)) if groups else 'all'
    work = directory / (label + '-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    work.mkdir(mode=0o700, parents=True)
    (work / 'request.json').write_text(json.dumps(request, indent=2)+'\n')
    report = {'status': 'failed', 'runs': [], 'sequence': sequence,
              'suite': ('tail-fetch-quality-v1' if 'tail-fetch' in groups else
                        'ampere-path-diagnostic-v1' if 'ampere-path' in groups else
                        'manual-theory-v4')}
    if 'tail-fetch' in groups:
        report['agent_tool_probe_plan'] = {
            'per_fresh_start': 16, 'cold_single_turn': 8, 'hot_history_replay': 8,
            'probes_per_arm_across_two_fresh_starts': 32,
            'hot_mode_limitation': 'synthetic prior assistant tool call plus fixture-only tool result; engine conversation-cache slots remain disabled',
            'external_tools_executed': False,
        }
    try:
        check_environment()
        exe = Path(cfg['exe'])
        digest = hashlib.sha256(exe.read_bytes()).hexdigest()
        if request['engine_sha256'] != digest:
            raise ValueError('requested engine hash differs from the selected baseline')
        report['engine_sha256'] = digest
        args = cfg['args']
        seed = work / 'expert-profile.bin'
        seed.write_bytes(Path(args[args.index('--expert-profile') + 1]).read_bytes())
        report['seed_sha256'] = hashlib.sha256(seed.read_bytes()).hexdigest()
        deadline = time.monotonic() + max(60, int(request.get('budget_s', 14400)))
        # Warmup plus 3072-token output and 1536-token long-context output.
        print(f'Strata diagnostic {label} starts with {len(sequence)} fresh runs. Report: {work}', flush=True)
        for i, arm in enumerate(sequence, 1):
            folder = work / f'{i:02d}-{arm["name"]}'
            trial_root, trial_base = root, cfg
            if not arm['engine']:
                reference=cfg.get('comparison_reference')
                if not reference:raise ValueError('the frozen previous-engine comparison reference is missing')
                trial_root=Path(reference['root']);trial_base=reference['config']
                if hashlib.sha256(Path(trial_base['exe']).read_bytes()).hexdigest()!=reference['engine_sha256']:
                    raise ValueError('previous-engine comparison binary changed')
            trial_cfg = configure(trial_base, arm, folder, seed, port)
            cases = comparison_workloads(arm['group'])
            # run_phase owns every child and closes it before the next arm.
            result = run_phase(trial_root, trial_cfg, arm['name'], folder, seed, port, deadline,
                               workloads=cases, launch_profile=trial_cfg, validate_ampere=True,
                               capture_output=arm['group']=='tail-fetch')
            result['settings'] = dict(arm)
            result['run_index'] = i
            result['arm_name'] = arm['name']
            result['trial_engine_sha256']=hashlib.sha256(Path(trial_cfg['exe']).read_bytes()).hexdigest()
            if arm['group'] in ('ampere-path','tail-fetch') and result['status'] == 'passed':
                if result.get('gpu_cache_slots') != 5316:
                    result.update(status='invalid', error=(
                        f'actual GPU expert cache slots {result.get("gpu_cache_slots")!r}; expected 5316'))
            short_long_fixtures = long_fixture_validity_failures(result['cases'])
            if result['status'] == 'passed' and short_long_fixtures:
                result.update(status='invalid', error='long-output fixture ended before 75% of the requested length',
                              short_long_fixture_cases=short_long_fixtures)
            # Never mistake enabling wide lookup for evidence that it actually ran.
            log_path = folder / 'engine.log'
            text = log_path.read_text(errors='replace') if log_path.is_file() else ''
            result['wide_bridge_exercised'] = 'wide context bridge active' in text
            result['trace'] = trace_summary(folder / 'spec.trace')
            result['optimization_log'] = optimization_log_summary(text)
            result['effective_optimization_environment'] = {name:trial_cfg['env'][name] for name in
                ('STRATA_ADAPT_NOWAIT','STRATA_ADAPT_LAG','STRATA_ADAPT_OVERLAP','STRATA_ADAPT_OVERLAP_RESIDENT','STRATA_ROUTE_PRIOR','STRATA_ROUTE_TAIL_SKIP','STRATA_ROUTE_TAIL_RENORM',
                 'STRATA_EMB_REUSE_ACCOUNT','STRATA_GDN_TC','STRATA_GDN_TC_NP','STRATA_GDN_TC_TILES',
                 'STRATA_GDN_TC_FUSE','STRATA_HC_UPMIX','STRATA_PREFILL_CPU_SHARE_NT','STRATA_PREFILL_CPU_SHARE_MAX',
                 'STRATA_ATTN_DECODE_TC','STRATA_PF_FUSED_TQ_F16','STRATA_ADAPT_FETCH','STRATA_HC_PERSIST',
                 'STRATA_QFUSE','STRATA_L3_PF','STRATA_CACHE_PIN_FRAC','STRATA_DENSE_I8_TC_MIN_T')
                if name in trial_cfg['env']}
            if arm['group']=='tail-fetch' and result['status']=='passed':
                bad = [r['case'] for r in result['cases']
                       if r.get('quality',{}).get('task_evaluation',{}).get('passed') is False]
                retried = [r['case'] for r in result['cases']
                           if r.get('generation_failure_count',0) or r.get('generation_recovery_count',0)]
                if bad or retried:
                    result.update(status='invalid', quality_failure_cases=bad,
                                  generation_recovery_cases=retried,
                                  error='objective quality failed or repetition recovery occurred')
            report['runs'].append(result)
            (work / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'Strata 对比进度 {i}/{len(sequence)}: {arm["name"]} {result["status"]}', flush=True)
            if (result['status'] == 'failed' and not request.get('continue_on_failure')) or time.monotonic() >= deadline:
                break
        report['summary'] = summarize(report['runs'])
        if 'tail-fetch' in groups:
            from l2.quality_diagnostics import compare_runs
            report['tail_fetch_summary'] = compare_runs(report['runs'])
        report['status'] = 'passed' if len(report['runs']) == len(sequence) and all(
            r['status'] == 'passed' for r in report['runs']) else 'incomplete'
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        (work / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f'Strata 手动无头对比 {report["status"]}。报告 {work / "results.json"}', flush=True)
    return report
