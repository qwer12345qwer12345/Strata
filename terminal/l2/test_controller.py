import json,tempfile,threading,unittest,urllib.request,socket
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from l2 import reference_controller as c

class ControllerTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.calls=[];self.release=threading.Event();owner=self
  class Backend(BaseHTTPRequestHandler):
   def log_message(self,*a):pass
   def do_GET(self):self.reply({'ok':True})
   def reply(self,obj):
    b=json.dumps(obj).encode();self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))));owner.calls.append((self.path,body))
    if self.path=='/v1/chat/completions' and body.get('messages',[{}])[-1].get('content')=='slow-prefill':owner.release.wait(3)
    if self.path=='/v1/chat/completions' and body.get('stream'):
     self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Connection','close');self.end_headers()
     self.wfile.write(b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n');self.wfile.flush()
     owner.release.wait(3)
     if body['messages'][-1].get('content')=='repeat-failure':
      self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"strata_generation_failure":{"reason":"repetition"}}\n\n');self.wfile.flush()
     self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush();self.close_connection=True
    elif self.path=='/v1/chat/completions':self.reply({'choices':[{'message':{'role':'assistant','content':'ok'}}],**({'strata_generation_failure':{'reason':'repetition'}} if body['messages'][-1].get('content')=='repeat-failure' else {})})
    elif self.path=='/v1/cache/probe':self.reply({'prompt_tokens':10000,'reusable_tokens':0})
    elif self.path=='/v1/cache/preload':self.reply({'serial':1,'tokens':9000})
    else:self.reply({'ok':True,'tokens':9000})
  self.backend=ThreadingHTTPServer(('127.0.0.1',0),Backend);self.backend.daemon_threads=True
  c.ENGINE_URL=f'http://127.0.0.1:{self.backend.server_port}';c.SESSION_DIR=self.tmp.name;c.INDEX_FILE=str(Path(self.tmp.name)/'index.json')
  c.state.update(current_family=None,current_chain=None,current_file=None,dirty=False)
  self.front=ThreadingHTTPServer(('127.0.0.1',0),c.ControllerHandler);self.front.daemon_threads=True
  for srv in [self.backend,self.front]:threading.Thread(target=srv.serve_forever,daemon=True).start()
 def tearDown(self):
  self.release.set()
  for s in [self.front,self.backend]:s.shutdown();s.server_close()
  self.tmp.cleanup()
 def body(self,text):return {'messages':[{'role':'system','content':'same system'},{'role':'user','content':text}],'max_tokens':8}
 def request(self,b,timeout=10):return urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{self.front.server_port}/v1/chat/completions',data=json.dumps(b).encode(),headers={'Content-Type':'application/json'}),timeout=timeout)
 def test_first_user_distinguishes_same_tool_conversations(self):self.assertNotEqual(c.family(self.body('A')),c.family(self.body('B')))
 def test_stream_arrives_before_backend_finishes(self):
  b=self.body('stream');b['stream']=True;received=False
  try:
   with self.request(b,timeout=1) as r:
    received=r.headers.get_content_type()=='text/event-stream' and b'first' in r.readline()
    self.release.set();r.read()
  except (socket.timeout,TimeoutError):pass
  finally:self.release.set()
  self.assertTrue(received,'first SSE event must arrive before upstream completion')
 def test_completed_turn_is_saved_and_returning_session_is_preloaded(self):
  for text in ['A','B','A']:
   with self.request(self.body(text)) as r:r.read()
  self.assertGreaterEqual(sum('action=saveinc' in p for p,b in self.calls),2)
  self.assertTrue(any(p=='/v1/cache/preload' for p,b in self.calls),'returning A must use PR preload')
 def test_stream_done_has_complete_sse_delimiter(self):
  self.release.set();b=self.body('complete');b['stream']=True
  with self.request(b) as r:data=r.read()
  self.assertTrue(data.endswith(b'data: [DONE]\n\n'))
 def test_same_family_divergent_branch_uses_distinct_archive(self):
  a=self.body('root');b=self.body('root');a['messages'] += [{'role':'assistant','content':'one'},{'role':'user','content':'A'}];b['messages'] += [{'role':'assistant','content':'two'},{'role':'user','content':'B'}]
  for body in (a,b):
   with self.request(body) as r:r.read()
  names=[body['filename'] for path,body in self.calls if 'action=saveinc' in path]
  self.assertEqual(len(set(names)),2)
 def test_index_does_not_retain_message_text(self):
  with self.request(self.body('secret marker 174239')) as r:r.read()
  self.assertNotIn('secret marker 174239',Path(c.INDEX_FILE).read_text())
 def test_repeated_identical_request_preserves_existing_archive(self):
  for text in ['A','B','A']:
   with self.request(self.body(text)) as r:r.read()
  self.assertEqual(sum('action=saveinc' in p for p,b in self.calls),2)
 def test_failed_generation_notice_does_not_overwrite_good_archive(self):
  with self.request(self.body('root')) as r:r.read()
  before=c.load_index();self.release.set()
  for stream in (False,True):
   bad=self.body('root');bad.update(stream=stream)
   bad['messages'] += [{'role':'assistant','content':'previous good reply'},{'role':'user','content':'repeat-failure'}]
   with self.request(bad) as r:r.read()
   self.assertEqual(c.load_index(),before)
  self.assertEqual(sum('action=saveinc' in p for p,b in self.calls),1)
 def test_return_to_ancestor_does_not_overwrite_descendant(self):
  root=self.body('root');child=self.body('root');child['messages'] += [{'role':'assistant','content':'one'},{'role':'user','content':'child'}]
  for body in [root,child,root]:
   with self.request(body) as r:r.read()
  idx=c.load_index();self.assertEqual(len(idx),2)
 def test_disconnected_queued_request_is_not_generated(self):
  import time
  c.request_lock.acquire()
  sock=socket.create_connection(('127.0.0.1',self.front.server_port))
  raw=json.dumps(self.body('abandoned')).encode();sock.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '+str(len(raw)).encode()+b'\r\n\r\n'+raw)
  sock.close();time.sleep(.1);c.request_lock.release();time.sleep(.2)
  self.assertFalse(any(p=='/v1/chat/completions' for p,b in self.calls))

 def test_connected_upstream_timeout_returns_http_error(self):
  from unittest.mock import patch
  real=c.http.client.HTTPConnection
  with patch.object(c.http.client,'HTTPConnection',lambda host,port,timeout:real(host,port,timeout=.05)):
   sock=socket.create_connection(('127.0.0.1',self.front.server_port));sock.settimeout(2)
   raw=json.dumps(self.body('slow-prefill')).encode();sock.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: '+str(len(raw)).encode()+b'\r\n\r\n'+raw)
   try:
    try:data=sock.recv(4096)
    except TimeoutError:data=b''
   finally:sock.close();self.release.set()
   self.assertIn(b'504',data.split(b'\r\n')[0])
