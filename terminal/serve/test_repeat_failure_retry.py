import contextlib
import io
import json
import threading
import unittest
import urllib.request
import tempfile
import itertools
from pathlib import Path
from unittest.mock import patch
from serve.server import ByteTokenizer,ChatTemplate,MockEngine,Service,serve
from serve import test_server as fixtures
ROOT=fixtures.ROOT

PARTIAL=fixtures.UnfinishedToolCall.CUT.removeprefix('</think>\n\n')
GOOD=fixtures.UnfinishedToolCall.WHOLE.removeprefix('</think>\n\n')
BAD=PARTIAL+'x'*300
TOOLS=[{'type':'function','function':{'name':'write','parameters':{'type':'object','properties':fixtures.UnfinishedToolCall.PROPS}}}]


class RecordingEngine(MockEngine):
    def __init__(self,tok,scripts):
        super().__init__(tok,scripts)
        self.calls=[]
    def generate(self,ids,max_new,sampling,cancel,embeddings=None):
        self.calls.append((list(ids),max_new,dict(sampling or {})))
        yield from super().generate(ids,max_new,sampling,cancel,embeddings)


class RepeatFailureRetry(unittest.TestCase):
    def service(self,scripts):
        tok=ByteTokenizer();engine=RecordingEngine(tok,scripts)
        svc=Service(engine,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
        svc.chatml_tool_guard=True;svc.repeat_failure_retry=True
        return svc,engine

    def events(self,scripts,limit=1000,thinking=False,sampling=None):
        svc,engine=self.service(scripts)
        request=sampling or {}
        with contextlib.redirect_stdout(io.StringIO()):
            events=list(svc.run([1,2,3],thinking,[t["function"] for t in TOOLS],limit,request,threading.Event()))
        return svc,engine,events

    def test_unpublished_tool_retries_original_history_once_without_leaking_draft(self):
        request={'temperature':0,'presence_penalty':0,'top_p':.7,'top_k':9,'seed':7}
        svc,engine,events=self.events([BAD,GOOD],sampling=request)
        self.assertEqual(len(engine.calls),2)
        self.assertEqual(engine.calls[0][0],engine.calls[1][0])
        self.assertEqual(engine.calls[1][2],{**request,'temperature':1.0,'presence_penalty':1.5,'seed':8})
        self.assertEqual(request['temperature'],0)
        calls=[e for kind,e in events if kind=='event' and e.kind=='tool_call']
        self.assertEqual(len(calls),1);self.assertEqual(calls[0].call.arguments['content'],'all of it')
        self.assertEqual(sum(e.kind=='tool_start' for kind,e in events if kind=='event'),1)
        self.assertEqual(events[-1][1]['generation_recovery']['retries'],1)
        self.assertNotIn('generation_failure',events[-1][1])
        self.assertEqual(events[-1][1]['completion_tokens'],1000-engine.calls[1][1]+len(engine.script))

    def test_repeat_twice_returns_normal_notice_and_no_partial_tool(self):
        _,engine,events=self.events([BAD,BAD])
        self.assertEqual(len(engine.calls),2)
        self.assertFalse(any(e.kind.startswith('tool_') for kind,e in events if kind=='event'))
        notice=''.join(e.text or '' for kind,e in events if kind=='event' and e.kind=='content')
        self.assertIn('[Strata]',notice);self.assertNotIn('xxxx',notice)
        self.assertEqual(events[-1][1]['finish'],'stop')
        self.assertTrue(events[-1][1]['generation_failure']['retryable'])

    def test_retry_shares_output_limit_even_when_the_second_attempt_is_cut_short(self):
        _,engine,events=self.events([BAD,GOOD],limit=400)
        self.assertEqual(len(engine.calls),2)
        self.assertLess(engine.calls[1][1],100)
        self.assertLessEqual(events[-1][1]['completion_tokens'],400)
        self.assertEqual(events[-1][1]['finish'],'stop')
        self.assertEqual(events[-1][1]['generation_failure']['reason'],'retry_incomplete')

    def test_published_content_or_complete_tool_is_never_replayed(self):
        for script in ('hello '+'x'*300,GOOD+'\n'+BAD):
            with self.subTest(script=script[:8]):
                _,engine,events=self.events([script,GOOD])
                self.assertEqual(len(engine.calls),1)
                self.assertEqual(events[-1][1]['generation_failure']['retries'],0)

    def test_reasoning_only_failure_can_retry_but_user_cancellation_cannot(self):
        _,engine,events=self.events(['thinking '+'x'*300,'</think>\nAnswer.'],thinking=True)
        self.assertEqual(len(engine.calls),2)
        self.assertTrue(any(e.kind=='content' and e.text for kind,e in events if kind=='event'))
        svc,engine=self.service(['thinking '+'x'*300,'unused'])
        cancel=threading.Event();events=[]
        with contextlib.redirect_stdout(io.StringIO()):
            for item in svc.run([1,2,3],True,[t["function"] for t in TOOLS],1000,{},cancel):
                events.append(item)
                if item[0]=='event':cancel.set()
        self.assertEqual(len(engine.calls),1)
        self.assertEqual(events[-1][1]['finish'],'cancel')
        self.assertNotIn('generation_failure',events[-1][1])

    def test_retry_preserves_higher_effective_config_sampler(self):
        svc,engine=self.service([BAD,GOOD]);svc.sampling_defaults={'temperature':1.7,'presence_penalty':2.0,'seed':5}
        with contextlib.redirect_stdout(io.StringIO()):list(svc.run([1],False,[t["function"] for t in TOOLS],1000,{},threading.Event()))
        self.assertEqual(engine.calls[1][2]['temperature'],1.7)
        self.assertEqual(engine.calls[1][2]['presence_penalty'],2.0)

    def test_image_request_does_not_reuse_consumed_embeddings(self):
        svc,engine=self.service([BAD,GOOD])
        with tempfile.TemporaryDirectory() as t:
            emb=Path(t)/'embedding';emb.write_bytes(b'fixture')
            svc.embeddings.path=str(emb)
            with contextlib.redirect_stdout(io.StringIO()):
                events=list(svc.run([1],False,[t['function'] for t in TOOLS],1000,{},threading.Event()))
            self.assertFalse(emb.exists())
        self.assertEqual(len(engine.calls),1)
        self.assertEqual(events[-1][1]['generation_failure']['retries'],0)

    def test_unrelated_engine_error_is_not_converted_to_a_success_notice(self):
        svc,_=self.service([GOOD])
        with patch.object(svc,'_run',side_effect=RuntimeError('engine failed')):
            with self.assertRaisesRegex(RuntimeError,'engine failed'):
                list(svc.run([1],False,[],1000,{},threading.Event()))

    def test_buffered_tool_arguments_keep_connection_alive(self):
        svc,_=self.service([GOOD])
        with patch('serve.repeat_recovery.time.monotonic',side_effect=itertools.count(0,11).__next__),contextlib.redirect_stdout(io.StringIO()):
            events=list(svc.run([1],False,[t['function'] for t in TOOLS],1000,{},threading.Event()))
        ping=next(i for i,(kind,_) in enumerate(events) if kind=='ping')
        start=next(i for i,(kind,e) in enumerate(events) if kind=='event' and e.kind=='tool_start')
        self.assertLess(ping,start)

    def test_stream_and_nonstream_failure_stay_usable_on_the_next_request(self):
        for stream in (False,True):
            with self.subTest(stream=stream):
                svc,engine=self.service([BAD,BAD,GOOD]);httpd=serve(svc,port=0)
                try:
                    results=[]
                    for _ in range(2):
                        body={'model':'fixture','messages':[{'role':'user','content':'Save notes'}],
                              'tools':TOOLS,'stream':stream,'max_tokens':1000,
                              'chat_template_kwargs':{'enable_thinking':False}}
                        req=urllib.request.Request(f'http://127.0.0.1:{httpd.server_port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                        with contextlib.redirect_stdout(io.StringIO()),urllib.request.urlopen(req,timeout=5) as response:
                            self.assertEqual(response.status,200);raw=response.read()
                        if stream:
                            self.assertTrue(raw.endswith(b'data: [DONE]\n\n'))
                            frames=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith(b'data: {')]
                            self.assertFalse(any('error' in frame for frame in frames))
                            results.append(frames[-1])
                        else:results.append(json.loads(raw))
                    self.assertEqual(results[0]['choices'][0]['finish_reason'],'stop')
                    self.assertTrue(results[0]['strata_generation_failure']['retryable'])
                    self.assertEqual(results[1]['choices'][0]['finish_reason'],'tool_calls')
                    self.assertNotIn('strata_generation_failure',results[1])
                    self.assertEqual(len(engine.calls),3)
                finally:httpd.shutdown();httpd.server_close()


if __name__=='__main__':unittest.main()
