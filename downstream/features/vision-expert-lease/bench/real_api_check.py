import base64,copy,hashlib,json,os,signal,socket,subprocess,sys,threading,time,urllib.request,urllib.error
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Desktop bounded-RAM API vision lifecycle check; stop other model processes first.')
parser.add_argument('--terminal-root',type=Path,required=True)
parser.add_argument('--profile',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--reader',type=Path,required=True)
parser.add_argument('--mmproj',type=Path,required=True)
parser.add_argument('--state-dir',type=Path,default=Path.home()/'.local/state/strata/startup-benchmark')
parser.add_argument('--image-max-tokens',type=int,default=300)
cli=parser.parse_args()
SOURCE=cli.terminal_root.resolve();OUT=cli.output.resolve();OUT.mkdir(parents=True,mode=0o700,exist_ok=True)
sys.path.insert(0,str(SOURCE));from l2.runtime_config import select,set_options
from l2.headless_experiments import stop_trial
PYTHON=sys.executable;state=cli.state_dir
_,base=select(SOURCE,json.loads(cli.profile.read_text()),state)
args=base['args']
for flag in ['--resident-experts','--expert-profile-save']:
 while flag in args:
  i=args.index(flag);del args[i:i+(2 if flag=='--expert-profile-save' else 1)]
args=set_options(args,{'--max-context':8192,'--kv-resident':2048,'--conversation-cache-mib':0,'--conversation-cache-slots':0,'--prefill':512,'--expert-cache':5600,'--vram-segment-mib':128})
args+=['--resident-budget-gib','2']
base.update(args=args,host='127.0.0.1',port=18295)
for key in ['startup_benchmark','l2_backend_port','l2_disk_budget_mib','slot_save_path']:base.pop(key,None)
base['vision']={'exe':str(cli.reader.resolve()),'mmproj':str(cli.mmproj.resolve()),'model':args[args.index('--native')+1],'gpu':True,'expert_lease':True,'max_tokens':cli.image_max_tokens,'threads':8}
base['env'].update(STRATA_CONV_DISK_DIR=str(OUT/'parked'),STRATA_L2_WRITER_BUDGET_MIB='0',STRATA_DECODE_TIMING='1')
report={'scope':'Desktop API vision/expert-lease correctness with 2GiB partial expert RAM, 5600 GPU slots, 8K context; not full-resident headless speed','cases':[],'status':'running'};active=None;stop=threading.Event()
def save(): (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
def mem():
 d={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines() if ':' in l and l.split()[1].isdigit()};return d
start_swap=mem()['SwapTotal']-mem()['SwapFree']
def watch():
 while not stop.wait(.5):
  d=mem();a=d['MemAvailable'];report['min_mem_available_kib']=min(a,report.get('min_mem_available_kib',a));new=d['SwapTotal']-d['SwapFree']-start_swap;report['max_new_swap_kib']=max(new,report.get('max_new_swap_kib',0))
  owned_swap=0
  if active:
   pending=[active.pid];seen=set()
   while pending:
    pid=pending.pop()
    if pid in seen:continue
    seen.add(pid)
    try:
     status=Path(f'/proc/{pid}/status').read_text().splitlines()
     owned_swap+=next((int(l.split()[1]) for l in status if l.startswith('VmSwap:')),0)
     pending+=list(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
    except (OSError,ValueError):pass
  report['max_owned_process_swap_kib']=max(owned_swap,report.get('max_owned_process_swap_kib',0))
  # This host has vm.swappiness=150: file-tier I/O can swap unrelated idle apps
  # despite 17GiB MemAvailable. Keep the guard on model processes and low RAM;
  # report global swap separately instead of calling every eviction an OOM.
  if a<1536*1024 or owned_swap>256*1024 or (new>1024*1024 and a<4096*1024):
   report['guard']='memory or swap limit';save()
   if active and active.poll() is None:active.send_signal(signal.SIGINT)
   return
threading.Thread(target=watch,daemon=True).start()
def start(label,cfg):
 global active,log,protocol_log
 folder=OUT/label;folder.mkdir(mode=0o700,exist_ok=True);cfg=copy.deepcopy(cfg);log=folder/'engine.log';cfg['log']=str(log)
 profile=folder/'profile.json';profile.write_text(json.dumps(cfg));profile.chmod(0o600);ready=folder/'ready.json';ready.unlink(missing_ok=True)
 protocol_log=folder/'console.log';console=protocol_log.open('w');active=subprocess.Popen([PYTHON,str(SOURCE/'strata.py'),'serve','--profile',str(profile),'--ready-file',str(ready),'--shutdown-timeout','15'],cwd=SOURCE,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
 limit=time.monotonic()+300
 while not ready.exists():
  if active.poll() is not None:raise RuntimeError('backend exited before ready: '+label)
  if time.monotonic()>limit:raise TimeoutError('startup '+label)
  time.sleep(.25)
 return console
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def request(label,content,expected=200):
 body={'messages':[{'role':'user','content':content}],'max_tokens':96,'temperature':0,'seed':123,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
 before=protocol_log.stat().st_size;started=time.monotonic();req=urllib.request.Request('http://127.0.0.1:18295/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with opener.open(req,timeout=240) as r:status=r.status;data=json.load(r)
 except urllib.error.HTTPError as e:status=e.code;data=json.loads(e.read())
 text=(data.get('choices') or [{}])[0].get('message',{}).get('content') or ''
 detail=protocol_log.read_bytes()[before:].decode(errors='replace');row={'case':label,'status':status,'seconds':time.monotonic()-started,'text':text,'timings':data.get('timings'),'error':data.get('error'),'lease_begins':detail.count('VISION_LEASE state=held'),'lease_ends':detail.count('VISION_LEASE state=idle')}
 report['cases'].append(row);save();print(json.dumps(row),flush=True)
 if status!=expected:raise AssertionError(f'{label}: expected {expected}, got {status}')
 if status==200 and not text:raise AssertionError(f'{label}: empty model response')
 return row
try:
 # A deterministic local test picture, no user media.
 from PIL import Image,ImageDraw,ImageFont
 im=Image.new('RGB',(384,144),'white');draw=ImageDraw.Draw(im)
 for i,(color,name) in enumerate([('red','RED'),('green','GREEN'),('blue','BLUE')]):
  draw.rectangle((i*128+8,16,i*128+120,104),fill=color);draw.text((i*128+23,112),name,fill='black')
 path=OUT/'colors.png';im.save(path);url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
 image=lambda u:{'type':'image_url','image_url':{'url':u}}
 prompt=[{'type':'text','text':'Name the three colored rectangles from left to right. Answer briefly in English.'},image(url)]
 console=start('real-reader',base)
 request('text_before','Explain a database transaction in two short sentences.')
 cold=request('image_cold',prompt)
 report['image_answer_contains_expected_colors']=all(c in cold['text'].lower() for c in ['red','green','blue']);save()
 if not report['image_answer_contains_expected_colors']:raise AssertionError('image did not identify expected colors')
 if (cold['lease_begins'],cold['lease_ends'])!=(1,1):raise AssertionError('cold image must borrow and restore once')
 cached=request('same_image_cached',prompt)
 if (cached['lease_begins'],cached['lease_ends'])!=(0,0):raise AssertionError('cached image borrowed again')
 other=Image.new('RGB',(160,160),'yellow');other_path=OUT/'yellow.png';other.save(other_path)
 other_url='data:image/png;base64,'+base64.b64encode(other_path.read_bytes()).decode()
 request('multiple_images',[{'type':'text','text':'Briefly describe both pictures.'},image(url),image(other_url)])
 # Use a fresh image so this is not satisfied by the image cache.
 other=Image.new('RGB',(160,160),'purple');other_path=OUT/'purple.png';other.save(other_path)
 cancel_url='data:image/png;base64,'+base64.b64encode(other_path.read_bytes()).decode()
 before=protocol_log.stat().st_size
 body=json.dumps({'messages':[{'role':'user','content':[{'type':'text','text':'Describe.'},image(cancel_url)]}],'max_tokens':96,'stream':True}).encode()
 wire=socket.create_connection(('127.0.0.1',18295),timeout=10)
 wire.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body)
 until=time.monotonic()+30
 while b'VISION_LEASE state=held' not in protocol_log.read_bytes()[before:]:
  if time.monotonic()>until:raise TimeoutError('cancel did not reach active lease')
  time.sleep(.01)
 wire.close()
 until=time.monotonic()+90
 while b'VISION_LEASE state=idle' not in protocol_log.read_bytes()[before:]:
  if time.monotonic()>until:raise TimeoutError('cancelled lease did not restore')
  time.sleep(.05)
 detail=protocol_log.read_bytes()[before:]
 report['cases'].append({'case':'client_disconnected_during_lease','lease_begins':detail.count(b'VISION_LEASE state=held'),'lease_ends':detail.count(b'VISION_LEASE state=idle'),'restored_before_next_request':True});save()
 request('text_after_cancel','Give the result of 2 + 3 in one short sentence.')

 # Valid container/header admission but actual decoder failure after BEGIN.
 bad='data:image/png;base64,'+base64.b64encode(b'\x89PNG\r\n\x1a\n'+b'bad-png-payload'*5).decode()
 request('bad_image_restore',[{'type':'text','text':'describe'},image(bad)],expected=400)
 request('text_after_image','Write a Python function that adds two integers. Include a tiny example.')
 stop_trial(active);active=None;console.close()
 broken=OUT/'failed-reader.sh';broken.write_text("#!/bin/sh\nprintf 'ERR injected encoder-start failure\\n'\nexit 23\n");broken.chmod(0o700)
 failed=copy.deepcopy(base);failed['vision']['exe']=str(broken)
 console=start('failed-reader',failed)
 request('reader_start_failure',prompt,expected=503)
 request('text_after_start_failure','Name two prime numbers.')
 stop_trial(active);active=None;console.close()
 report['status']='passed';save()
except BaseException as exc:
 report['status']='failed';report['failure']=str(exc);save();raise
finally:
 stop.set()
 if active:stop_trial(active)
