#!/usr/bin/env python3
"""Supervise the single-slot upstream and streaming L2 adapter together."""
import argparse,json,os,signal,subprocess,sys,tempfile,threading,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from l2 import reference_controller as controller
from l2 import runtime_config

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',required=True);ap.add_argument('--ready-file');ap.add_argument('--shutdown-timeout',default='30');a=ap.parse_args()
 cfg=json.loads(Path(a.profile).read_text());port=int(cfg.get('port',8095));backend_port=int(cfg.get('l2_backend_port',8096))
 if backend_port==port:raise SystemExit('L2 public and backend ports must differ')
 # PR's async captures can use 2 GiB by default; this 32 GiB host uses synchronous incremental saves.
 cfg.setdefault('env',{})['STRATA_L2_WRITER_BUDGET_MIB']='0'
 settings=cfg.get('startup_benchmark') or {}
 output=Path(settings.get('output_dir',str(Path.home()/'.local/state/strata/startup-benchmark')))
 ready=Path(a.ready_file) if a.ready_file else None
 if ready:ready.unlink(missing_ok=True)
 root,cfg=runtime_config.select(Path(__file__).resolve().parents[1],cfg,output)
 cfg.pop('startup_benchmark',None)
 archive = Path(cfg.get('slot_save_path') or Path.home()/'.local/share/strata/sessions-pr5500-0.1.65').expanduser().resolve()
 archive.mkdir(mode=0o700, parents=True, exist_ok=True)
 cfg['slot_save_path'] = str(archive)
 cfg['env'].setdefault('STRATA_SESSION_V2', '1')
 cfg['env'].setdefault('STRATA_SESSION_V3', '1')
 cfg['env'].setdefault('STRATA_CONV_DISK', '1')
 cfg['env'].setdefault('STRATA_CONV_DISK_DIR', str(archive / '.parked'))
 Path(cfg['env']['STRATA_CONV_DISK_DIR']).mkdir(mode=0o700, parents=True, exist_ok=True)
 def stop(signum,frame):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 if cfg.get('configuration_status'):print('Strata '+cfg['configuration_status'],flush=True)
 ablation=cfg.pop('startup_pr_ablation',None)
 if ablation and ablation.get('enabled'):
  from l2.startup_ablation import run_at_boot
  try:run_at_boot(root,cfg,output,ablation,ready.parent if ready else None)
  except KeyboardInterrupt:return
 backend=dict(cfg,host='127.0.0.1',port=backend_port)
 parent=ready.parent if ready else Path(tempfile.gettempdir());parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile('w',dir=parent,prefix='strata-l2-backend-',suffix='.json',delete=False) as f:json.dump(backend,f);runtime=Path(f.name)
 cmd=[sys.executable,str(root/'strata.py'),'serve','--profile',str(runtime),'--shutdown-timeout',a.shutdown_timeout]
 backend_ready=runtime.with_suffix('.ready.json')
 cmd+=['--ready-file',str(backend_ready)]
 proc=None;server=None
 try:
  proc=subprocess.Popen(cmd,cwd=root)
  upstream=f'http://127.0.0.1:{backend_port}'
  deadline=time.monotonic()+float(cfg.get('engine_ready_s',1076))+60
  while time.monotonic()<deadline:
   if proc.poll() is not None:raise RuntimeError(f'Upstream exited: {proc.returncode}')
   try:
    with urllib.request.urlopen(upstream+'/health',timeout=2) as r:
     if r.status==200 and backend_ready.exists():
      health=json.load(r)
      try:backend_record=json.loads(backend_ready.read_text())
      except (OSError,ValueError):backend_record={}
      if backend_record.get('pid')==proc.pid and backend_record.get('api_state')=='ready':break
   except (OSError,ValueError):pass
   time.sleep(0.25)
  else:raise RuntimeError('Upstream readiness timeout')
  if proc.poll() is not None:raise RuntimeError(f'Upstream exited: {proc.returncode}')
  controller.ENGINE_URL=upstream;controller.LISTEN_HOST=cfg.get('host','0.0.0.0');controller.LISTEN_PORT=port
  controller.SESSION_DIR=cfg['slot_save_path'];controller.INDEX_FILE=str(Path(controller.SESSION_DIR)/'index.json')
  controller.CACHE_BUDGET=int(cfg.get('l2_disk_budget_mib',32768))*1024**2
  server=controller.ThreadingHTTPServer((controller.LISTEN_HOST,port),controller.ControllerHandler);server.daemon_threads=True
  images=health.get('images') if isinstance(health,dict) else None
  if type(images) is not bool:images=None
  vision=cfg.get('vision') or {}
  on_demand=images is True and (vision.get('expert_lease') is True or vision.get('elastic') is True)
  if images is True:vision_status='已启用，首张图片时加载' if on_demand else '已启用'
  elif images is False:vision_status='后端未启用' if vision else '未配置'
  else:vision_status='后端未报告状态'
  if ready:
   record=dict(backend_record)
   record.update(backend_api_url=record.get('api_url'),backend_pid=record.get('pid'),pid=os.getpid(),bound_host=controller.LISTEN_HOST,bound_port=port,api_url=f'http://127.0.0.1:{port}')
   record['vision']={'enabled':images,'on_demand':on_demand}
   if cfg.get('configuration_status'):record['configuration_status']=cfg['configuration_status']
   with tempfile.NamedTemporaryFile('w',dir=ready.parent,delete=False) as f:json.dump(record,f);tmp=Path(f.name)
   os.replace(tmp,ready)
  def watch():
   proc.wait();server.shutdown()
  threading.Thread(target=watch,daemon=True).start()
  print(json.dumps({'l2':'ready','host':controller.LISTEN_HOST,'port':port,'backend_port':backend_port,'prior':cfg.get('env',{}).get('STRATA_ROUTE_PRIOR'),'session_v2':cfg.get('env',{}).get('STRATA_SESSION_V2'),'session_v3':cfg.get('env',{}).get('STRATA_SESSION_V3'),'disk_budget_mib':controller.CACHE_BUDGET//1024**2,'writer_budget_mib':0}),flush=True)
  print(f'Strata 正式 API 已就绪（端口 {port}）。视觉：{vision_status}。',flush=True)
  server.serve_forever()
  if proc.poll() not in (None,0):raise RuntimeError(f'Upstream exited: {proc.returncode}')
 except KeyboardInterrupt:pass
 finally:
  if server:server.server_close()
  drained = controller.request_lock.acquire(timeout=float(a.shutdown_timeout))
  if not drained:
   print('[l2] shutdown: the active turn exceeded the drain deadline; retaining its last committed archive', flush=True)
  if proc and proc.poll() is None:
   proc.send_signal(signal.SIGINT)
   try:proc.wait(timeout=float(a.shutdown_timeout)+15)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
  if drained:controller.request_lock.release()
  runtime.unlink(missing_ok=True);backend_ready.unlink(missing_ok=True)
  if ready:
   try:
    if json.loads(ready.read_text()).get('pid')==os.getpid():ready.unlink(missing_ok=True)
   except (OSError,ValueError):pass
if __name__=='__main__':main()
