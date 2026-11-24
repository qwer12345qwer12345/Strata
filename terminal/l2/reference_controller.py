#!/usr/bin/env python3
"""Single-slot Chat Completions adapter for PR #5500's saveinc/preload APIs.
Streams SSE immediately. Stores message hashes, never message text, in its index.
"""
import hashlib,http.client,json,os,re,select,shutil,socket,threading,time,urllib.error,urllib.parse,urllib.request,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
ENGINE_URL=os.environ.get('STRATA_ENGINE_URL','http://127.0.0.1:8096')
LISTEN_HOST=os.environ.get('STRATA_CONTROLLER_BIND','127.0.0.1')
LISTEN_PORT=int(os.environ.get('STRATA_CONTROLLER_PORT','8095'))
SESSION_DIR=os.environ.get('STRATA_SESSION_DIR','./sessions')
INDEX_FILE=os.path.join(SESSION_DIR,'index.json')
CACHE_BUDGET=int(os.environ.get('STRATA_L2_DISK_BUDGET_MIB','32768'))*1024**2
MAX_ARCHIVES=16
state={'current_family':None,'current_chain':None,'current_file':None,'dirty':False}
state_lock=threading.RLock()
request_lock=threading.Lock()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def family(body):
 head=[]
 for m in body.get('messages') or []:
  head.append(m)
  if m.get('role')=='user':break
 return digest({'model':body.get('model'),'tools':body.get('tools'),'messages':head,'template':body.get('chat_template_kwargs')})[:32]
def chain_hashes(messages):return [digest(m) for m in messages or []]
def chain_id(hashes):return digest(hashes)[:32]
def load_index():
 try:
  with open(INDEX_FILE) as f:d=json.load(f)
  return d if isinstance(d,dict) else {}
 except (OSError,ValueError):return {}
def save_index(idx):
 os.makedirs(SESSION_DIR,mode=0o700,exist_ok=True)
 tmp=INDEX_FILE+'.tmp'
 with open(tmp,'w') as f:json.dump(idx,f)
 os.chmod(tmp,0o600);os.replace(tmp,INDEX_FILE)
def common(a,b):
 n=0
 for x,y in zip(a,b):
  if x!=y:break
  n+=1
 return n
def find_entry(idx,fp,hs):
 entries=[r for r in idx.values() if isinstance(r,dict) and r.get('family')==fp and common(r.get('hashes',[]),hs)>0]
 return max(entries,key=lambda r:(common(r['hashes'],hs),r.get('updated',0)),default=None)
def engine_request(method,path,body=None,timeout=180):
 data=json.dumps(body).encode() if body is not None else None
 req=urllib.request.Request(ENGINE_URL+path,method=method,data=data,headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,json.loads(r.read())
 except urllib.error.HTTPError as e:
  try:return e.code,json.loads(e.read())
  except (ValueError,OSError):return e.code,{}
 except (OSError,ValueError) as e:return 0,{'error':str(e)}
def cache_body(body):return {k:body[k] for k in ('messages','tools','chat_template_kwargs') if k in body}
def handle_turn(body):
 hs=chain_hashes(body.get('messages'));row=find_entry(load_index(),family(body),hs)
 if not row:return None
 status,probe=engine_request('POST','/v1/cache/probe',cache_body(body))
 if status==200 and probe.get('reusable_tokens',0)>=max(1,probe.get('prompt_tokens',0)*0.98):return None
 t=time.monotonic()
 status,result=engine_request('POST','/v1/cache/preload',dict(cache_body(body),filename=row['filename'],pin=True))
 print(json.dumps({'l2':'preload','status':status,'ms':round((time.monotonic()-t)*1000,1),'tokens':result.get('tokens'),'partial':result.get('partial')}),flush=True)
 return result if status==200 else None
def disk_size(path):
 if not path.exists():return 0
 if path.is_file():return path.stat().st_size
 return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
def after_turn(body,response_ok):
 if not response_ok:return
 idx=load_index();fp=family(body);hs=chain_hashes(body.get('messages'));row=find_entry(idx,fp,hs)
 # An identical prompt already has a useful archive. A restored short tail can
 # have no earlier checkpoint, so re-saving it would discard the archived prefix.
 if row and row['hashes']==hs:
  row['updated']=time.time();save_index(idx);return
 # Append only along the same branch; a divergent branch gets its own archive.
 if row and hs[:len(row['hashes'])]==row['hashes']:name=row['filename']
 else:name=uuid.uuid4().hex+'.bin'
 t=time.monotonic();status,result=engine_request('POST','/slots/0?action=saveinc',{'filename':name})
 print(json.dumps({'l2':'saveinc','status':status,'ms':round((time.monotonic()-t)*1000,1),'incremental':result.get('incremental')}),flush=True)
 if status!=200:return
 idx[name]={'family':fp,'hashes':hs,'filename':name,'updated':time.time()}
 # Saves are synchronous in managed_server, so deleting old archives cannot race a writer.
 total=sum(disk_size(Path(SESSION_DIR)/n) for n in idx if re.fullmatch(r'[a-f0-9]{32}\.bin',n))
 for n,r in sorted(list(idx.items()),key=lambda kv:kv[1].get('updated',0)):
  if total<=CACHE_BUDGET and len(idx)<=MAX_ARCHIVES:break
  if n==name or not re.fullmatch(r'[a-f0-9]{32}\.bin',n):continue
  p=Path(SESSION_DIR)/n;total-=disk_size(p)
  if p.is_dir():shutil.rmtree(p)
  else:p.unlink(missing_ok=True)
  del idx[n]
 save_index(idx)
def unpin_serial(serial):
 if serial:engine_request('POST','/v1/cache/unpin',{'serial':int(serial)},timeout=10)

class ControllerHandler(BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def setup(self):
  super().setup();self.connection.settimeout(30)
 def gone(self):
  try:
   return bool(select.select([self.connection],[],[],0)[0]) and self.connection.recv(1,socket.MSG_PEEK|socket.MSG_DONTWAIT)==b''
  except (OSError,ValueError):return True
 def log_message(self,*a):pass
 def reply(self,status,obj):
  data=json.dumps(obj).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def do_GET(self):
  if self.path.split('?')[0] not in ('/health','/v1/models'):return self.reply(404,{'error':{'message':'Use /v1/chat/completions'}})
  status,data=engine_request('GET',self.path);self.reply(status or 502,data)
 def do_POST(self):
  if self.path.split('?')[0]!='/v1/chat/completions':return self.reply(404,{'error':{'message':'Use /v1/chat/completions'}})
  try:
   size=int(self.headers.get('Content-Length','0'))
   if not 0<size<=32*1024**2:return self.reply(413,{'error':{'message':'Request body too large or empty'}})
   raw=self.rfile.read(size);body=json.loads(raw)
   if not isinstance(body,dict) or not isinstance(body.get('messages'),list):raise ValueError('messages must be a list')
  except (ValueError,TypeError):return self.reply(400,{'error':{'message':'Invalid Chat Completions request'}})
  queued=time.monotonic()
  while not request_lock.acquire(timeout=.1):
   if self.gone():return
  try:
   if self.gone():return
   print(json.dumps({'l2':'queue','ms':round((time.monotonic()-queued)*1000,1)}),flush=True)
   preload=None;sent=False;conn=None;watch_stop=threading.Event()
   try:
    preload=handle_turn(body)
    headers={k:v for k,v in self.headers.items() if k.lower() not in ('host','content-length','connection','transfer-encoding','accept-encoding')}
    if self.gone():return
    upstream=urllib.parse.urlsplit(ENGINE_URL)
    conn=http.client.HTTPConnection(upstream.hostname,upstream.port,timeout=600)
    conn.request('POST',self.path,body=raw,headers=headers)
    upstream_sock=conn.sock
    def watch_disconnect():
     while not watch_stop.wait(.1):
      if self.gone():
       try:upstream_sock.shutdown(socket.SHUT_RDWR)
       except OSError:pass
       return
    threading.Thread(target=watch_disconnect,daemon=True).start()
    with conn.getresponse() as r:
     if body.get('stream') and r.headers.get_content_type()=='text/event-stream':
      self.send_response(r.status);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers();sent=True;self.close_connection=True
      complete=False;failed=False
      for line in r:
       self.wfile.write(line);self.wfile.flush()
       if line.startswith(b'data:'):
        payload=line[5:].strip()
        if payload==b'[DONE]':
         self.wfile.write(b'\n');self.wfile.flush();complete=True;break
        try:failed=failed or 'error' in json.loads(payload)
        except (ValueError,TypeError):pass
      watch_stop.set()
      if complete and not failed:after_turn(body,True)
     else:
      data=r.read();watch_stop.set()
      if r.status==200:after_turn(body,True)
      self.send_response(r.status);self.send_header('Content-Type',r.headers.get('Content-Type','application/json'));self.send_header('Content-Length',str(len(data)));self.end_headers();sent=True;self.wfile.write(data)
   except urllib.error.HTTPError as e:
    if not sent:
     data=e.read();self.send_response(e.code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
   except (BrokenPipeError,ConnectionResetError):pass
   except TimeoutError:
    if not sent and not self.gone():self.reply(504,{'error':{'message':'Upstream timed out'}})
   except Exception as e:
    print(json.dumps({'l2':'error','type':type(e).__name__,'message':str(e)}),flush=True)
    if not sent:self.reply(502,{'error':{'message':'L2 upstream request failed; inspect service log'}})
   finally:
    watch_stop.set()
    if conn:conn.close()
    if preload:unpin_serial(preload.get('serial'))
  finally:request_lock.release()

def main():
 os.makedirs(SESSION_DIR,mode=0o700,exist_ok=True)
 server=ThreadingHTTPServer((LISTEN_HOST,LISTEN_PORT),ControllerHandler);server.daemon_threads=True
 print(f'[l2] streaming Chat Completions {LISTEN_HOST}:{LISTEN_PORT} -> {ENGINE_URL}',flush=True)
 server.serve_forever()
if __name__=='__main__':main()
