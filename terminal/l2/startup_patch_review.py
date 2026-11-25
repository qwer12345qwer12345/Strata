"""Bounded headless review. One group per boot, no production writes or promotion."""

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
    ('attention-and-trellis', (
        arm('decode-attention-int8-tc', {'STRATA_ATTN_DECODE_TC': '1'}),
        arm('prefill-attention-int8-imma-off', {'STRATA_K8V4_QK_IMMA': '0'}),
    )),
    ('trellis-prefill', (
        arm('trellis-f16', {'STRATA_PF_FUSED_TQ_F16': '1'}),
        arm('trellis-fused-off', {'STRATA_PF_FUSED_TQ': '0'}),
    )),
    ('expert-kernels', (
        arm('gpu-single-row', {'STRATA_TQ_AMPERE_ROWS': '0'}),
        arm('cpu-avxvnni-off', {'STRATA_TQ_AVXVNNI': '0'}),
    )),
    ('ple-bf16-upload', (
        arm('bf16-upload-off', {'STRATA_PLE_BF16_COMPACT': '0'}),
        arm('bf16-independent-copy-off', {'STRATA_PLE_COPY_STREAM': '0'}),
    )),
    ('dflash-projection', (
        arm('dflash-upstream-shapes', {'STRATA_DF2_AMPERE_TUNING': '0'}),
        arm('dflash-qkv-tensor-core', {'STRATA_DF2_GEMM_TC': '2'}),
    )),
    ('dflash-lookup', (
        arm('dflash-lookup-eight', {'STRATA_DF2_WIDE_CONTEXT': '0'},
            {'--lookup-max-t': '8', '--lookup-sizes': '2,3,4,5,6,7,8'}),
        arm('dflash-context-batch-off', {'STRATA_DF2_CTX_BATCH': '0'}),
    )),
    ('dflash-spec', (
        arm('spec-3', args={'--spec': '3'}),
        arm('spec-6', args={'--spec': '6'}),
    )),
    ('dense-and-hc', (
        arm('dense-shape-dispatch-off', {'STRATA_DENSE_I8_AMPERE_SHAPES': '0'}),
        arm('hc-q8-off', {'STRATA_HC_Q8': '0'}),
    )),
    ('adaptive-fetch', (
        arm('fetch-0', {'STRATA_ADAPT_FETCH': '0'}),
        arm('fetch-1', {'STRATA_ADAPT_FETCH': '1'}),
    )),
    ('adaptive-scheduling', (
        arm('nowait-off', {'STRATA_ADAPT_NOWAIT': '0'}),
        arm('profile-pin-off', {'STRATA_CACHE_PIN_FRAC': '0'}),
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
    return (WARMUP, ('image_cold', image, 40), ('image_cached', image, 40), DECODE)


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


def run_rotation(root, cfg, directory, settings):
    """Run a baseline at each end; rotate even if one arm could not be measured."""
    now = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    progress = directory / 'startup-patch-review-progress.json'
    latest = directory / 'startup-patch-review-latest.json'
    results = directory / 'startup-patch-review'
    report = {'suite': 'startup-patch-review-v1', 'status': 'incomplete', 'runs': []}
    attempted = False
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        results.mkdir(exist_ok=True, mode=0o700)
        baseline_args = cfg['args']
        seed_path = Path(baseline_args[baseline_args.index('--expert-profile') + 1])
        engine_sha = digest_file(cfg['exe'])
        reference = settings.get('lean_candidate')
        groups = GROUPS
        if not (cfg.get('vision') or {}).get('gpu'):
            groups = tuple(g for g in groups if g[0] != 'vision-lease')
        if reference:
            groups = (('lean-production-bundle', (arm('lean-candidate', {
                'STRATA_TQ_AMPERE_ROWS': '0', 'STRATA_TQ_AVXVNNI': '0',
                'STRATA_K8V4_QK_IMMA': '0', 'STRATA_PLE_BF16_COMPACT': '0',
                'STRATA_PLE_COPY_STREAM': '0', 'STRATA_DENSE_I8_AMPERE_SHAPES': '0',
                'STRATA_DF2_WIDE_CONTEXT': '0', 'STRATA_PF_BATCH': '0',
            }, {'--lookup-max-t': '8', '--lookup-sizes': '2,3,4,5,6,7,8'}),)),) + groups
        vision_cases = vision_workloads()
        fingerprint = hashlib.sha256(json.dumps({
            'engine': engine_sha, 'reference': reference and reference['sha256'],
            'groups': groups, 'args': baseline_args, 'env': cfg['env'],
            'series': settings.get('series', 1), 'passes': settings.get('passes', 1),
            'workloads': (WORKLOADS, vision_cases),
        }, sort_keys=True).encode()).hexdigest()
        try:
            state = json.loads(progress.read_text())
        except (OSError, ValueError):
            state = {}
        if state.get('fingerprint') != fingerprint:
            state = {'fingerprint': fingerprint, 'next': 0, 'completed': {}}
        passes = int(settings.get('passes', 1))
        if passes not in (1, 2):
            raise ValueError('patch review supports one or two passes')
        order = [(i, bool(pass_no)) for pass_no in range(passes)
                 for i in (range(len(groups) - 1, -1, -1) if pass_no else range(len(groups)))]
        idx = state['next']
        if idx >= len(order):
            print(f'Strata 开机补丁对照已完成 {len(order)} 批，直接启动正式服务。', flush=True)
            return
        group_idx, reverse = order[idx]
        group, variants = groups[group_idx]
        if reverse:
            variants = variants[::-1]
        workloads = vision_cases if group == 'vision-lease' else WORKLOADS
        report.update(group=group, batch=idx + 1, batches=len(order), engine_sha256=engine_sha,
                      series=settings.get('series', 1), pass_number=2 if reverse else 1)
        attempted = True
        deadline = time.monotonic() + int(settings.get('budget_s', 480))
        print(f'Strata 开机补丁对照 {idx+1}/{len(order)}：{group}，结束后启动正式服务。', flush=True)
        # Keep scratch files out of tmpfs on a 32 GiB resident-expert host.
        with tempfile.TemporaryDirectory(prefix='patch-review-', dir=directory) as tmp:
            scratch = Path(tmp)
            seed = scratch / 'expert-profile.bin'
            seed.write_bytes(seed_path.read_bytes())
            report['expert_profile_sha256'] = digest_file(seed)
            # Do not let a trial clean or write the production SSD-spill directory.
            report['workloads'] = [x[0] for x in workloads]
            expected = len(variants) + 2
            for name, environment, arguments in (arm('baseline-first'), *variants, arm('baseline-last')):
                if time.monotonic() >= deadline:
                    report['error'] = 'boot budget exhausted'
                    break
                row = {'arm': name, 'status': 'failed', 'engine_sha256': engine_sha,
                       'environment_overrides': environment, 'argument_overrides': arguments}
                report['runs'].append(row)
                folder = scratch / name
                folder.mkdir()
                trial = trial_config(cfg, folder, seed, None, None)
                trial['env'].update(environment)
                trial['env']['STRATA_PREFILL_TIMING'] = '1'
                storage = dict(arguments, **{'--conversation-spill-dir': str(folder/'spill'), '--conversation-spill-mib': '0'})
                for k in ('args','user_args'):
                    trial[k] = set_options(trial[k], storage)
                if trial.get('vision'):
                    trial['vision']['cache_parent'] = str(folder/'vision')
                    if name == 'upstream-elastic-vision':
                        trial['vision'].update(expert_lease=False, elastic=True)
                try:
                    if name == 'lean-candidate':
                        trial['exe'] = reference['exe']
                        if digest_file(trial['exe']) != reference['sha256']:
                            raise ValueError('lean candidate fingerprint mismatch')
                        row['engine_sha256'] = reference['sha256']
                    serve_trial(root, trial, folder, min(deadline, time.monotonic() + 125), row,
                                workloads=workloads)
                    if group == 'vision-lease':
                        extra = (f"冷图 {row['image_cold']['first_content_ms']:.0f} ms，"
                                 f"缓存图 {row['image_cached']['first_content_ms']:.0f} ms")
                    else:
                        extra = (f"长预填充 {row['cold_long']['timings']['prompt_ms']:.0f} ms，"
                                 f"热前缀 {row['hot_prefix']['timings'].get('cache_n', 0)} token")
                    print(f"Strata 对照 {name}：解码 {row['decode']['timings']['predicted_per_second']:.1f} token/s，"
                          f"{extra}。", flush=True)
                except Exception as exc:
                    row['error'] = str(exc)
                    print(f'Strata 对照 {name} 未完成：{exc}', flush=True)
                    if row.get('limit_hit') == 'memory':
                        break
            if len(report['runs']) == expected and all(row['status'] == 'measured' for row in report['runs']):
                report['status'] = 'measured'
        report_path = results / (now + '-' + group + '.json')
        publish(report_path, report)
        publish(latest, report)
        state['next'] = idx + 1
        if report['status'] == 'measured':
            state['completed'][f'{group}:{report["pass_number"]}'] = report_path.name
        state.setdefault('attempts', {})[f'{group}:{report["pass_number"]}'] = {'status': report['status'], 'report': report_path.name}
        publish(progress, state)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f'Strata 开机补丁对照中止：{exc}', flush=True)
    finally:
        if attempted:
            print(f"Strata 开机补丁对照结束（{report['status']}），继续启动正式服务。", flush=True)
