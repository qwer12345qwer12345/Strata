"""Apply the one user-owned runtime selection. Reading it never runs a benchmark."""
import copy
import hashlib
import json
from pathlib import Path

DAILY = dict(spec=4, shapes=2, lookup=16, tc=0, dense=1, skip=8, rows=2, nowait=1, tail=7, emb=1, gdn_np=2, upmix=0, gdn_tiles=1, gdn_fuse=1, prefill_cap=8192, cpu_nt=0, cpu_max=0, lag=2, overlap=0)
PREVIOUS = dict(spec=6, shapes=1, lookup=8, tc=2, dense=0, skip=0, rows=2, nowait=0, tail=0, emb=0, gdn_np=3)
STATUS = '理论优先配置；组合尚未完成无头性能与质量验证（NOT TESTED）'


def set_options(args, overrides):
    args = list(args)
    for flag, value in overrides.items():
        while flag in args:
            at = args.index(flag)
            del args[at:at + 2]
        args.extend([flag, str(value)])
    return args


def set_flags(args, overrides):
    args = list(args)
    for flag, enabled in overrides.items():
        if not isinstance(enabled, bool):
            raise ValueError('Boolean engine flag expected: ' + flag)
        args = [item for item in args if item != flag]
        if enabled:
            args.append(flag)
    return args


def select_boot_model(cfg, cmdline):
    cfg = copy.deepcopy(cfg)
    boot = cfg.pop('boot_models')
    selections = [arg.partition('=')[2] for arg in cmdline.split()
                  if arg.startswith('strata.model=')]
    if len(selections) > 1:
        raise ValueError('Only one strata.model= boot selection is allowed')
    model = selections[0] if selections else boot['default']
    if model not in boot['profiles']:
        raise ValueError(f'Unknown Strata boot model: {model}')
    selected = copy.deepcopy(boot['profiles'][model])
    overrides = selected.pop('argument_overrides', {})
    if selected.get('slot_save_path'):
        overrides.setdefault('--conversation-spill-dir', str(Path(selected['slot_save_path']) / '.spill'))
    cfg['args'] = set_options(cfg['args'], overrides)
    cfg['user_args'] = set_options(cfg.get('user_args', []), overrides)
    cfg.setdefault('env', {}).update(selected.pop('env', {}))
    cfg.update(selected)
    return cfg


def select(root, cfg, directory):
    if 'boot_models' in cfg:
        cfg = select_boot_model(cfg, Path('/proc/cmdline').read_text())
        return Path(cfg['cwd']), cfg
    path = directory / 'production-baseline.json'
    if not path.is_file():
        return root, copy.deepcopy(cfg)
    data = json.loads(path.read_text())
    if data.get('base_exe') and data['base_exe'] != cfg['exe']:
        return root, copy.deepcopy(cfg)
    root = Path(data['root'])
    exe = root / 'build/strata'
    with exe.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != data['engine_sha256']:
            raise ValueError('Selected Strata engine changed on disk')
    if not (root / 'strata.py').is_file():
        raise ValueError('Selected Strata server source is missing')
    result = copy.deepcopy(cfg)
    args = result['args']
    overrides = dict(data.get('argument_overrides', {}))
    if data.get('ple_gguf'):
        if not Path(data['ple_gguf']).is_file():
            raise ValueError('Selected BF16 PLE table is missing')
        overrides['--ple-gguf'] = data['ple_gguf']
    result['args'] = set_options(args, overrides)
    # The official drafting layer only preserves explicit user_args. Legacy
    # profiles may have neither user_args nor args_written, so changing args
    # alone lets drafting.apply replace our selected --spec with its default.
    result['user_args'] = set_options(result.get('user_args', []), overrides)
    for key in ('args', 'user_args'):
        result[key] = set_flags(result[key], data.get('flag_overrides', {}))
    rows = data.get('ampere_rows', '2')
    if rows not in ('0', '2'):
        raise ValueError('Ampere expert rows must be 0 or 2')
    result.setdefault('env', {}).update(STRATA_TQ_AMPERE_ROWS=rows,
                                      STRATA_PF_FUSED_TQ='1', STRATA_PF_GROUP_SCRATCH='0')
    result['env'].update(data.get('environment', {}))
    for key in ('vision', 'vram_elastic', 'repeat_failure_retry', 'startup_pr_ablation'):
        if key in data:result[key]=copy.deepcopy(data[key])
    result.update(exe=str(exe), cwd=str(root), log=data['log'], slot_save_path=data['archive_dir'],
                  configuration_status=data.get('configuration_status', STATUS))
    return root, result
