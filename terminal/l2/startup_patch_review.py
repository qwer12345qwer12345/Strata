"""Run the full headless patch review before serving, without production writes."""

import base64
import hashlib
import json
import os
import struct
import tempfile
import time
import zlib
from datetime import datetime
from pathlib import Path

from l2.runtime_config import set_options
from l2.startup_ablation import serve_trial, trial_config


# Each arm is a named change to the actual production profile. Values are
# explicit when the architecture's auto default would otherwise be ambiguous.
def arm(name, env=None, args=None):
    return name, env or {}, args or {}


GROUPS = (
    ('prompt-attention', (
        arm('prefill-attention-int8-imma-off', {'STRATA_K8V4_QK_IMMA': '0'}),
    )),
    ('expert-kernels', (
        arm('gpu-single-row', {'STRATA_TQ_AMPERE_ROWS': '0'}),
        arm('cpu-avxvnni-off', {'STRATA_TQ_AVXVNNI': '0'}),
    )),
    ('ple-bf16-upload', (
        arm('bf16-upload-off', {'STRATA_PLE_BF16_COMPACT': '0'}),
        arm('bf16-independent-copy-off', {'STRATA_PLE_COPY_STREAM': '0'}),
    )),
    ('dflash-lookup', (
        arm('dflash-lookup-eight', {'STRATA_DF2_WIDE_CONTEXT': '0'},
            {'--lookup-max-t': '8', '--lookup-sizes': '2,3,4,5,6,7,8'}),
        arm('dflash-context-batch-off', {'STRATA_DF2_CTX_BATCH': '0'}),
    )),
    ('dflash-spec', (
        arm('spec-3', args={'--spec': '3'}),
    )),
    ('dense-and-hc', (
        arm('dense-shape-dispatch-off', {'STRATA_DENSE_I8_AMPERE_SHAPES': '0'}),
        arm('hc-q8-off', {'STRATA_HC_Q8': '0'}),
    )),
    ('adaptive-fetch', (
        arm('fetch-1', {'STRATA_ADAPT_FETCH': '1'}),
    )),
    ('adaptive-scheduling', (
        arm('nowait-off', {'STRATA_ADAPT_NOWAIT': '0'}),
    )),
    ('adaptive-overlap', (
        arm('lag-one', {'STRATA_ADAPT_LAG': '1'}),
        # Overlap requires no lag, NOWAIT or fetch. Its result is a bundle comparison.
        arm('resident-overlap-bundle', {'STRATA_ADAPT_FETCH': '0', 'STRATA_ADAPT_LAG': '1',
            'STRATA_ADAPT_NOWAIT': '0', 'STRATA_ADAPT_OVERLAP': '1',
            'STRATA_ADAPT_OVERLAP_RESIDENT': '1'}),
    )),
    ('prompt-capacity', (
        arm('embedding-accounting-off', {'STRATA_EMB_REUSE_ACCOUNT': '0'}),
        arm('prefill-hc-region-off', {'STRATA_PF_HC_REGION': '0'}),
    )),
    ('prompt-scheduling', (
        arm('kv-prefetch-off', {'STRATA_KV_PREFETCH': '0'}),
        arm('group-scratch-on', {'STRATA_PF_GROUP_SCRATCH': '1'}),
    )),
    ('prompt-cpu-and-batch', (
        arm('cpu-prefill-share', {'STRATA_PREFILL_CPU_SHARE': '1',
            'STRATA_PREFILL_CPU_SHARE_NT': '32', 'STRATA_PREFILL_CPU_SHARE_MAX': '1024'}),
        arm('default-expert-prefill-batch', {'STRATA_PF_BATCH': '0'}),
    )),
    ('gdn-prefill', (
        arm('gdn-tc-off', {'STRATA_GDN_TC': '0'}),
        arm('gdn-np3', {'STRATA_GDN_TC_NP': '3'}),
    )),
    ('gdn-variants', (
        arm('gdn-tiles2', {'STRATA_GDN_TC_TILES': '2'}),
        arm('gdn-fuse-off', {'STRATA_GDN_TC_FUSE': '0'}),
    )),
    ('prompt-cpu-share-variants', (
        arm('cpu-prefill-share-3072', {'STRATA_PREFILL_CPU_SHARE': '1',
            'STRATA_PREFILL_CPU_SHARE_NT': '32', 'STRATA_PREFILL_CPU_SHARE_MAX': '3072'}),
        arm('prefill-chunk-4096', args={'--prefill': 'auto:4096'}),
    )),
    ('fused-activations', (
        arm('qfuse-on', {'STRATA_QFUSE': '1'}),
        arm('hc-upmix-on', {'STRATA_HC_UPMIX': '1'}),
    )),
    ('vision-lease', (arm('upstream-elastic-vision', {'STRATA_VISION_EXPERT_LEASE': '0'}),)),
)

# A long cold request and a shared-prefix continuation in addition to a long
# decode. Avoid relying on a tiny prompt to exercise a prefill-only change.
WARMUP = ('warmup', 'Explain database transaction isolation in a short paragraph.', 40)
DECODE = ('decode', 'Write 80 fully implemented Python tests for a journaled storage engine. '
          'Give every body, fixture and assertion. Keep implementing tests through the output limit.', 1536)
RECORDS = '\n'.join(
    f'Record {10000+i}: shard {(i*223)%81817}, checkpoint {(i*3817)%2750089}, '
    f'worker {i%17} compared sequence {i*53} with the durable offset and recorded diagnostic {i%137}.'
    for i in range(600)
)
LONG_SYSTEM = 'Review the following synthetic storage records as reference material.\n' + RECORDS
COLD = ('cold_long', [{'role': 'system', 'content': LONG_SYSTEM},
                      {'role': 'user', 'content': 'Write 60 detailed recovery runbook steps. Keep going.'}], 768)
HOT = ('hot_prefix', [{'role': 'system', 'content': LONG_SYSTEM},
                       {'role': 'user', 'content': 'Write 40 detailed test plans for aborted transactions. Keep going.'}], 384)
WORKLOADS = (WARMUP, DECODE, COLD, HOT)
CPU_PREFILL_GROUPS = {'prompt-cpu-and-batch', 'prompt-cpu-share-variants'}
CPU_PREFILL_CASES = tuple(
    (name, '\\n'.join(RECORDS.splitlines()[start:start + count])
     + '\\nDescribe how to verify these storage records. Explain several options.', 64)
    for name, start, count in (('prefill_under_1k', 0, 16), ('prefill_1k_to_3k', 40, 42))
)


def vision_workloads():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    line = b'\x00' + bytes((240, 32, 32)) * 64 + bytes((32, 240, 32)) * 64 + bytes((32, 32, 240)) * 64
    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 192, 96, 8, 2, 0, 0, 0)) +
           chunk(b'IDAT', zlib.compress(line * 96)) + chunk(b'IEND', b''))
    image = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe the three colored panels in this image. Write at least '
                                 'six detailed sentences about their left-to-right positions and appearance.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}},
    ]}]
    return (('image_cold', image, 40), ('image_cached', image, 40), ('post_vision_decode', *DECODE[1:]))


def publish(path, document):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(document, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.close()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def digest_file(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run_group(root, cfg, directory, seed, references, report, group, variants, workloads, arm_timeout):
    batch = {'suite': report['suite'], 'group': group, 'status': 'incomplete', 'runs': [],
             'batch': len(report['groups']) + 1, 'batches': report['batches'],
             'pass_number': report['pass_number'], 'series': report['series'],
             'engine_sha256': report['engine_sha256'],
             'expert_profile_sha256': report['expert_profile_sha256'],
             'workloads': [x[0] for x in workloads]}
    if group.startswith('baseline-'):
        print(f'Strata 开机补丁对照：{group}', flush=True)
        batch.pop('batch')
    else:
        print(f"Strata 开机补丁对照 {batch['batch']}/{batch['batches']}：{group}", flush=True)
    # Use disk scratch, not tmpfs. Remove each group's scratch before starting the next.
    with tempfile.TemporaryDirectory(prefix='patch-review-', dir=directory) as tmp:
        for name, environment, arguments in variants:
            row = {'arm': name, 'status': 'failed', 'engine_sha256': batch['engine_sha256'],
                   'environment_overrides': environment, 'argument_overrides': arguments}
            batch['runs'].append(row)
            folder = Path(tmp) / name
            folder.mkdir()
            try:
                print(f'Strata 对照 {group}/{name} 开始。', flush=True)
                trial = trial_config(cfg, folder, seed, None, None)
                trial['env'].update(environment)
                trial['env']['STRATA_PREFILL_TIMING'] = '1'
                # Never let a trial use or clean the production spill directory.
                storage = {**arguments, '--conversation-spill-dir': str(folder/'spill'),
                           '--conversation-spill-mib': '0'}
                for k in ('args', 'user_args'):
                    trial[k] = set_options(trial[k], storage)
                if trial.get('vision'):
                    trial['vision']['cache_parent'] = str(folder/'vision')
                    if name == 'upstream-elastic-vision':
                        trial['vision'].update(expert_lease=False, elastic=True)
                trial_root = root
                if name in references:
                    reference = references[name]
                    trial['exe'] = reference['exe']
                    if digest_file(trial['exe']) != reference['sha256']:
                        raise ValueError(f'{name} engine fingerprint mismatch')
                    row['engine_sha256'] = reference['sha256']
                    if reference.get('root'):
                        trial_root = Path(reference['root'])
                        trial['cwd'] = str(trial_root)
                serve_trial(trial_root, trial, folder, time.monotonic() + arm_timeout, row, workloads=workloads)
                text = row.get('decode') or row.get('post_vision_decode', {})
                cold, hot = row.get('cold_long'), row.get('hot_prefix')
                if 'image_cold' in row and 'image_cached' in row:
                    extra = (f"冷图 {row['image_cold']['first_content_ms']:.0f} ms，"
                             f"缓存图 {row['image_cached']['first_content_ms']:.0f} ms")
                elif cold and hot:
                    extra = (f"冷首字 {cold['first_content_ms']:.0f} ms，冷预填充 {cold['timings']['prompt_ms']:.0f} ms，"
                             f"热首字 {hot['first_content_ms']:.0f} ms / 复用 {hot['timings'].get('cache_n', 0)} token")
                else:
                    extra = '参见单项结果'
                stream = text.get('stream_gap_p99_ms')
                window = text.get('ms_per_round')
                if stream is not None:
                    extra += f'，流式 p99 {stream:.1f} ms'
                if window is not None:
                    extra += f"，平均验证窗口 {window:.1f} ms / {text['tokens_per_round']:.2f} token"
                if text and not text.get('decode_comparable'):
                    extra += '（短输出，不参与长解码比较）'
                print(f"Strata 对照 {name}（{row['status']}）：{extra}。", flush=True)
            except Exception as exc:
                row.update(status='failed', error=str(exc))
                print(f'Strata 对照 {name} 未完成：{exc}', flush=True)
    if all(row['status'] == 'measured' for row in batch['runs']):
        batch['status'] = 'measured'
    return batch


def run_suite(root, cfg, directory, settings):
    """Attempt every group in one boot and retain results as groups finish."""
    now = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    progress = directory / 'startup-patch-review-progress.json'
    latest = directory / 'startup-patch-review-latest.json'
    results = directory / 'startup-patch-review'
    report = {'suite': 'startup-patch-review-v4', 'status': 'running', 'groups': [], 'baselines': []}
    attempted = False
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        results.mkdir(exist_ok=True, mode=0o700)
        baseline_args = cfg['args']
        seed_path = Path(baseline_args[baseline_args.index('--expert-profile') + 1])
        engine_sha = digest_file(cfg['exe'])
        references = {}
        requested = settings.get('groups')
        groups = GROUPS if requested is None else tuple(group for group in GROUPS if group[0] in requested)
        if requested is not None and len(groups) != len(requested):
            raise ValueError('Unknown or duplicate Strata patch-review group')
        if not (cfg.get('vision') or {}).get('gpu'):
            groups = tuple(g for g in groups if g[0] != 'vision-lease')
        if settings.get('lean_candidate'):
            references['lean-candidate'] = settings['lean_candidate']
            groups = (('lean-production-bundle', (arm('lean-candidate', {
                'STRATA_TQ_AMPERE_ROWS': '0', 'STRATA_TQ_AVXVNNI': '0',
                'STRATA_K8V4_QK_IMMA': '0', 'STRATA_PLE_BF16_COMPACT': '0',
                'STRATA_PLE_COPY_STREAM': '0', 'STRATA_DENSE_I8_AMPERE_SHAPES': '0',
                'STRATA_DF2_WIDE_CONTEXT': '0', 'STRATA_PF_BATCH': '0',
            }, {'--lookup-max-t': '8', '--lookup-sizes': '2,3,4,5,6,7,8'}),)),) + groups
        if settings.get('upstream_candidate'):
            references['upstream-candidate'] = settings['upstream_candidate']
            references['upstream-cublas-warm-off'] = settings['upstream_candidate']
            references['upstream-reserve-384'] = settings['upstream_candidate']
            groups = (('upstream-engine', (
                arm('upstream-candidate'),
                arm('upstream-cublas-warm-off', {'STRATA_PREFILL_WARM': '0'}),
                arm('upstream-reserve-384', args={'--vram-reserve-mib': '384'}),
            )),) + groups
        vision_cases = vision_workloads()
        passes = int(settings.get('passes', 1))
        if passes not in (1, 2):
            raise ValueError('patch review supports one or two passes')
        arm_timeout = int(settings.get('arm_timeout_s', 125))
        if arm_timeout <= 0:
            raise ValueError('patch review needs a positive timeout per arm')
        series = settings.get('series', 1)
        final_baseline = settings.get('final_baseline', True)
        fingerprint = hashlib.sha256(json.dumps({
            'suite': report['suite'], 'engine': engine_sha,
            'references': {name: item['sha256'] for name, item in references.items()}, 'groups': groups,
            'args': baseline_args, 'env': cfg['env'], 'series': series,
            'passes': passes, 'final_baseline': final_baseline, 'arm_timeout_s': arm_timeout,
            'workloads': (WORKLOADS, CPU_PREFILL_CASES, vision_cases),
        }, sort_keys=True).encode()).hexdigest()
        try:
            state = json.loads(progress.read_text())
        except (OSError, ValueError):
            state = {}
        if state.get('fingerprint') == fingerprint and state.get('all_groups_attempted'):
            print(f'Strata 开机补丁对照已跑过 {len(groups) * passes} 组，直接启动正式服务。', flush=True)
            return

        plan = [(name, variants if pass_no == 1 else variants[::-1], pass_no)
                for pass_no in range(1, passes + 1)
                for name, variants in (groups if pass_no == 1 else groups[::-1])]
        arms = 1 + int(final_baseline) + sum(len(variants) for _, variants, _ in plan)
        vision_suffix = vision_cases if any(name == 'vision-lease' for name, _ in groups) else ()
        # Run vision in the same place for its candidate and both baseline starts.
        cpu_cases = WORKLOADS + CPU_PREFILL_CASES
        baseline_cases = cpu_cases + vision_suffix
        report.update(series=series, engine_sha256=engine_sha, batches=len(plan), arms=arms,
                      arm_timeout_s=arm_timeout)
        attempted = True
        started = time.monotonic()
        print(f'Strata 开机补丁对照：本次串行跑完 {len(plan)} 组、{arms} 次引擎启动，再开放正式 API。', flush=True)
        with tempfile.TemporaryDirectory(prefix='patch-review-seed-', dir=directory) as tmp:
            seed = Path(tmp) / 'expert-profile.bin'
            seed.write_bytes(seed_path.read_bytes())
            report['expert_profile_sha256'] = digest_file(seed)
            publish(latest, report)

            def baseline(label, number):
                report['pass_number'] = number
                # Baselines include the matched extra vision and CPU-prefill cases.
                timeout = arm_timeout + 80
                batch = run_group(root, cfg, directory, seed, references, report, label, (arm(label),),
                                  baseline_cases, timeout)
                name = f'{now}-{label}.json'
                publish(results / name, batch)
                report['baselines'].append({'status': batch['status'], 'report': name})
                publish(latest, report)
            baseline('baseline-first', 1)
            for group, variants, pass_no in plan:
                report['pass_number'] = pass_no
                workloads = WORKLOADS
                timeout = arm_timeout
                if group == 'vision-lease':
                    workloads = baseline_cases
                    timeout += 80
                elif group in CPU_PREFILL_GROUPS:
                    workloads = cpu_cases
                    timeout += 30
                batch = run_group(root, cfg, directory, seed, references, report, group, variants, workloads, timeout)
                name = f"{now}-{batch['batch']:02d}-{group}.json"
                publish(results / name, batch)
                report['groups'].append({'group': group, 'pass_number': pass_no,
                                         'status': batch['status'], 'report': name})
                report['elapsed_s'] = round(time.monotonic() - started, 1)
                publish(latest, report)
            if final_baseline:
                baseline('baseline-last', passes)
        report['elapsed_s'] = round(time.monotonic() - started, 1)
        report['status'] = ('measured' if all(x['status'] == 'measured'
                                             for x in report['groups'] + report['baselines']) else 'incomplete')
        publish(latest, report)
        publish(progress, {'fingerprint': fingerprint, 'all_groups_attempted': True,
                           'status': report['status'], 'groups': report['groups']})
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        raise
    except Exception as exc:
        report.update(status='incomplete', error=str(exc))
        print(f'Strata 开机补丁对照中止：{exc}', flush=True)
    finally:
        if attempted:
            try:
                publish(latest, report)
            except OSError as exc:
                print(f'Strata 开机补丁对照总报告写入失败：{exc}', flush=True)
            continuation = '，继续启动正式服务' if report['status'] != 'interrupted' else ''
            print(f"Strata 开机补丁对照结束（{report['status']}）{continuation}。报告：{latest}", flush=True)
