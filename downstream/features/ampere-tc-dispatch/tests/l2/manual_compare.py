"""Explicit headless comparison; normal startup never imports this module."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import resource
import signal
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from l2 import headless_experiments as experiments
from l2 import runtime_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=Path('/etc/strata/server.json'))
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--group', action='append', help='combined/spec-lookup/shape-qkv/dense-lookup/skip-lookup/rows/adapt-tail/tail-spec/emb-gdn/gdn-precision/hc-upmix/gdn-execution/prefill-cap/cpu-prefill/adapt-mode; ampere-path validates real dispatch; tail-fetch runs the 2x2 quality study; numeric selects numerical groups; updates selects 0.2.20 and PR groups; default all')
    parser.add_argument('--budget-s', type=int, help='total seconds; default 72 hours for --all, 4 hours for one suite')
    parser.add_argument('--all', dest='full', action='store_true', help='all kernels, 180 comparison runs and vision; detached systemd job by default')
    parser.add_argument('--foreground', action='store_true', help='with --all, wait in this terminal instead of detaching')
    parser.add_argument('--kernel-check',action='store_true',help='prebuilt synthetic GPU checks and Dense sweep only; no model requests')
    parser.add_argument('--vision-check',action='store_true',help='run the optional vision lease correctness suite only')
    parser.add_argument('--plan', action='store_true', help='show the comparison plan without stopping or starting anything')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        payload = json.loads(args.worker.read_text())
        if payload['request'].get('suite')=='all':
            from l2.full_suite import run
        elif payload['request'].get('suite')=='kernels':
            from l2.kernel_check import run
        elif payload['request'].get('suite')=='vision':
            from l2.vision_check import run
        else:run=experiments.run_comparison
        report = run(Path(payload['root']),payload['cfg'],Path(payload['directory']),payload['port'],payload['request'])
        return 0 if report['status'] == 'passed' else 1
    cfg = json.loads(args.profile.read_text())
    directory = args.state_dir or Path(cfg.get('startup_benchmark', {}).get(
        'output_dir', str(Path.home()/'.local/state/strata/startup-benchmark')))
    root, cfg = runtime_config.select(Path(cfg['cwd']), cfg, directory)
    if sum([args.full,args.vision_check,args.kernel_check,bool(args.group)])>1:parser.error('--all, --vision-check, --kernel-check and --group are separate selections')
    if args.foreground and not args.full:parser.error('--foreground requires --all')
    if args.budget_s is None:args.budget_s=259200 if args.full else 14400
    if args.budget_s < (300 if args.full else 60):parser.error('time budget is too short')
    sequence = experiments.arms(args.group)
    request = {'engine_sha256': json.loads((directory/'production-baseline.json').read_text())['engine_sha256'],
               'budget_s': args.budget_s, 'groups': args.group}
    if args.group and any(g in args.group for g in ('ampere-path','tail-fetch')):
        request['continue_on_failure']=True
    if args.full:
        from l2.full_suite import plan
        full_plan=plan(cfg);request['suite']='all'
        if args.plan:
            full_plan.update(budget_s=args.budget_s,detached=not args.foreground)
            print(json.dumps(full_plan,indent=2,ensure_ascii=False));return 0
    if args.kernel_check:
        from l2.kernel_check import plan
        kernel_plan=plan(cfg);request['suite']='kernels'
        if args.plan:
            print(json.dumps(kernel_plan,indent=2,ensure_ascii=False));return 0
    if args.vision_check:
        from l2.vision_check import plan
        vision_plan=plan(cfg)
        request['suite']='vision'
        if args.plan:
            print(json.dumps(vision_plan,indent=2,ensure_ascii=False))
            return 0
    if args.plan:
        print(json.dumps({'status':runtime_config.STATUS, 'automatic_tests':False, 'daily':runtime_config.DAILY,
                          'groups':list(dict.fromkeys(r['group'] for r in sequence)), 'arms':len(sequence),
                          'output_tokens':[3072,1536], 'budget_s':args.budget_s,
                          'sequence':sequence}, indent=2, ensure_ascii=False))
        return 0
    desktop = subprocess.check_output(['systemctl','show','-p','ActiveState','--value','sddm.service'], text=True).strip()
    if desktop != 'inactive':
        parser.error('只在无头环境运行；当前桌面仍在使用，不会停止任何服务。')
    if os.geteuid() != 0:
        parser.error('完整专家驻留需要无限memlock并需临时停止/恢复服务，请用 sudo ~/.local/bin/strata-headless-compare；--plan无需sudo。')
    owner = pwd.getpwuid(directory.stat().st_uid)
    if owner.pw_uid == 0:
        parser.error('comparison state directory must belong to the model user')
    # Reject competing inference or compilation before stopping the daily service.
    try:experiments.check_environment(allow_strata=True)
    except RuntimeError as exc:parser.error(str(exc))
    if args.full and not args.foreground:
        log=directory/'full-suite.log'
        log.touch(mode=0o600,exist_ok=True)
        os.chown(log,owner.pw_uid,owner.pw_gid)
        command=['systemd-run','--unit=strata-headless-test','--collect',
                 '--description=Strata manual headless full suite',
                 '--property=Type=exec','--property=LimitMEMLOCK=infinity',
                 '--property=KillMode=mixed','--property=TimeoutStopSec=120',
                 '--property=UMask=0077','--property=StandardOutput=append:'+str(log),
                 '--property=StandardError=inherit',
                 sys.executable,str(Path(__file__).resolve()),'--all','--foreground',
                 '--profile',str(args.profile.resolve()),'--state-dir',str(directory.resolve()),
                 '--budget-s',str(args.budget_s)]
        subprocess.run(command,check=True)
        print(f'Strata 长程测试已交给后台，退出终端后继续。日志 {log}；总报告 {directory/"full-suite-latest.json"}。',flush=True)
        return 0
    lock=(directory/'manual-compare.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();parser.error('另一项手动测试正在运行；不会停止服务。')
    try:
        return run_worker(root,cfg,directory,owner,request)
    finally:lock.close()


def run_worker(root,cfg,directory,owner,request):
    # The worker keeps the service user and RAM locking limit of the system service.
    # All process-affecting operations occur after the explicit command and preflight.
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    with tempfile.NamedTemporaryFile('w',dir=directory,prefix='manual-compare-',suffix='.json',delete=False) as f:
        payload_path = Path(f.name)
        json.dump({'root':str(root),'cfg':cfg,'directory':str(directory),
                   'port':int(cfg.get('l2_backend_port',8096)),'request':request}, f)
    os.chown(payload_path,owner.pw_uid,owner.pw_gid)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    stopped = False
    worker = None
    try:
        subprocess.run(['systemctl','stop','strata.service'],check=True)
        stopped = True
        worker = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(payload_path)],
                                  user=owner.pw_uid,group=owner.pw_gid,
                                  extra_groups=os.getgrouplist(owner.pw_name,owner.pw_gid),
                                  start_new_session=True,
                                  env={**os.environ,'HOME':owner.pw_dir,'USER':owner.pw_name,'LOGNAME':owner.pw_name})
        try:
            return worker.wait()
        except KeyboardInterrupt:
            os.killpg(worker.pid,signal.SIGINT)
            worker.wait(timeout=60)
            return 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        payload_path.unlink(missing_ok=True)
        desktop = subprocess.check_output(['systemctl','show','-p','ActiveState','--value','sddm.service'],text=True).strip()
        if stopped and desktop == 'inactive':
            subprocess.run(['systemctl','start','strata.service'],check=True)
            print('Strata 手动对比已结束，已启动日常服务。',flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
