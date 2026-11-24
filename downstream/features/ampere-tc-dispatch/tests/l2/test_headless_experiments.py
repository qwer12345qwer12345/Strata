import copy
import contextlib
import io
import hashlib
import json
from pathlib import Path
import tempfile
import socket
import time
import unittest
from unittest.mock import patch
from l2 import headless_experiments as h
from l2 import quality_diagnostics as q

FAKE_ENGINE = r'''import argparse,json,os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('--profile');p.add_argument('--ready-file');p.add_argument('--shutdown-timeout')
a=p.parse_args();c=json.loads(Path(a.profile).read_text());log=Path(c['log'])
log.write_text('resident RAM mode: fake page-locked\nDFlash2 drafter on\nTQ Ampere paired-row GU enabled\n')
class API(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_POST(self):
  self.rfile.read(int(self.headers['Content-Length']))
  self.send_response(200);self.send_header('content-type','text/event-stream');self.end_headers()
  for i in range(4):self.wfile.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n');self.wfile.flush()
  self.wfile.write(b'data: {"choices":[{"finish_reason":"length"}],"timings":{"prompt_ms":10,"predicted_ms":100,"predicted_n":8,"predicted_per_second":80}}\n\n')
  with log.open('a') as f:f.write('0 blob reads from the file\nstrata decode timing: 4 windows, avg T 2.0, 2.0 tokens/window, 25.0 ms/window\n')
  self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
s=ThreadingHTTPServer(('127.0.0.1',c['port']),API)
Path(a.ready_file).write_text(json.dumps({'api_state':'ready','pid':os.getpid()}))
try:s.serve_forever()
except KeyboardInterrupt:pass
finally:s.server_close()
'''

class Experiments(unittest.TestCase):
 def base(self, root):
  (root/'engine').write_bytes(b'engine')
  (root/'seed').write_bytes(b'fixed-seed')
  return {'exe':str(root/'engine'),'args':['--expert-profile',str(root/'seed'),
          '--expert-profile-save',str(root/'seed'),'--spec','6','--resident-experts',
          '--max-context','262144','--pool-workers','15'],
          'env':{'STRATA_ROUTE_PRIOR':'0.5'}, 'startup_benchmark':{'x':1}}
 def test_one_variable_and_immutable_profile(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);base=self.base(root);old=copy.deepcopy(base)
   for arm in h.arms():
    cfg=h.configure(base,arm,root/'arm',root/'seed',18001)
    args=cfg['args'];val=lambda k:args[args.index(k)+1]
    self.assertNotIn('--expert-profile-save',args)
    self.assertEqual(val('--spec'),str(arm['spec']))
    self.assertEqual(val('--lookup-max-t'),str(arm['lookup']))
    self.assertEqual(val('--draft-conf'),'0.8')
    self.assertEqual(val('--prefill'),f'auto:{arm["prefill_cap"]}')
    for name,key in [('STRATA_EMB_REUSE_ACCOUNT','emb'),('STRATA_GDN_TC_NP','gdn_np'),('STRATA_HC_UPMIX','upmix'),('STRATA_GDN_TC_TILES','gdn_tiles'),('STRATA_GDN_TC_FUSE','gdn_fuse')]:
     self.assertEqual(cfg['env'][name],str(arm[key]))
    for name,key in [('STRATA_PREFILL_CPU_SHARE_NT','cpu_nt'),('STRATA_PREFILL_CPU_SHARE_MAX','cpu_max')]:
     if arm[key]:self.assertEqual(cfg['env'][name],str(arm[key]))
     else:self.assertNotIn(name,cfg['env'])
    self.assertEqual(cfg['env']['STRATA_PREFILL_CPU_SHARE'],'auto' if arm['cpu_share'] else '0')
    self.assertEqual(val('--pool-workers'),'15')
    self.assertEqual(val('--max-context'),'262144')
    self.assertIn('--resident-experts',args)
    self.assertEqual(cfg['env']['STRATA_TQ_AMPERE_ROWS'],str(arm['rows']))
    self.assertEqual(cfg['env']['STRATA_ADAPT_NOWAIT'],str(arm['nowait']))
    self.assertEqual(cfg['env']['STRATA_ROUTE_TAIL_SKIP'],str(arm['tail']))
    self.assertEqual(cfg['env']['STRATA_ROUTE_TAIL_RENORM'],'0')
    self.assertEqual(cfg['env']['STRATA_ADAPT_LAG'],str(arm['lag']))
    self.assertEqual(cfg['env']['STRATA_ADAPT_OVERLAP'],str(arm['overlap']))
    if arm['overlap']:self.assertEqual((arm['nowait'],arm['lag'],arm['fetch']),(0,1,0))
    self.assertEqual(cfg['env']['STRATA_DF2_WIDE_CONTEXT'],'1' if arm['lookup']>8 else '0')
    if arm['group']=='spec-lookup':
     expected='2,3,4,5,6,7,8'+(',10,12' if arm['lookup']>=12 else '')+(',16' if arm['lookup']==16 else '')
     self.assertEqual(val('--lookup-sizes'),expected)
   self.assertEqual(base,old)
 def test_cpu_share_has_a_medium_prompt_inside_the_extended_range(self):
  cases=h.comparison_workloads('cpu-prefill')
  self.assertEqual(cases[-1][0],'medium_context')
  self.assertEqual(cases[-1][2],1536)
  self.assertEqual(sum(line.startswith('Record ') for line in cases[-1][1].splitlines()),60)
  self.assertEqual(len(h.comparison_workloads('emb-gdn')),3)
 def test_hash_mismatch_never_starts_a_trial(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);base=self.base(root)
   with patch.object(h,'check_environment'),patch.object(h,'run_phase') as run:
    report=h.run_comparison(root,base,root,18001,{'engine_sha256':'wrong'})
   run.assert_not_called();self.assertIn('hash differs',report['error'])
   self.assertEqual(list(root.glob('*.request.json')),[])
 def test_interrupt_saves_report_without_queue(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);base=self.base(root)
   with patch.object(h,'check_environment'),patch.object(h,'run_phase',side_effect=KeyboardInterrupt):
    with self.assertRaises(KeyboardInterrupt):h.run_comparison(root,base,root,18001,{'engine_sha256':hashlib.sha256(b'engine').hexdigest(),'groups':['shape']})
   self.assertTrue(next(root.glob('shape-*/results.json')).exists())
   self.assertEqual(list(root.glob('*.request.json')),[])
 def test_long_budget_and_continue_on_failure_do_not_stop_at_four_hours(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);base=self.base(root)
   with patch.object(h,'check_environment'),patch.object(h.time,'monotonic',side_effect=[0]+[20000]*50),patch.object(h,'run_phase',return_value={'status':'failed','cases':[]}) as run,contextlib.redirect_stdout(io.StringIO()):
    result=h.run_comparison(root,base,root,18001,{'engine_sha256':hashlib.sha256(b'engine').hexdigest(),'groups':['shape'],'budget_s':86400,'continue_on_failure':True})
   self.assertEqual(len(result['runs']),8)
   self.assertEqual(result['status'],'incomplete')
   self.assertEqual(run.call_args.args[6],86400)
 def test_trace_sources_widths_and_no_tokens(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'trace';p.write_text('# request prompt 3 df2 1 spec 6\nR 3 4 2 0 1.2 4 0 | 0.9 | 789 | 123 | 456\nR 6 12 11 1 2.5 12 32 | 0.9 | 789 | 123 | 456\n')
   data=h.trace_summary(p)
   self.assertEqual(data[0]['neural']['accepted'],2)
   self.assertEqual(data[0]['lookup']['accepted'],11)
   self.assertEqual(data[0]['lookup']['round_latency_ms']['p99'],2.5)
   self.assertEqual(data[0]['widths'],{'4':1,'12':1})
   self.assertNotIn('789',json.dumps(data))
 def test_benchmark_rejects_service_retry_metadata(self):
  from email.message import Message
  for field in ('strata_generation_failure','strata_generation_recovery'):
   response=io.BytesIO(b'data: '+json.dumps({field:{'reason':'repetition'}}).encode()+b'\n\ndata: [DONE]\n\n')
   response.headers=Message();response.headers['Content-Type']='text/event-stream'
   with patch('l2.chat_latency.urllib.request.build_opener') as opener:
    opener.return_value.open.return_value=response
    with self.assertRaisesRegex(RuntimeError,'not a valid performance baseline'):
     h.measure('http://127.0.0.1:1','fixture','prompt',64,10)
 def test_skip_and_adapt_metrics_are_observed_not_inferred_from_flags(self):
  text=('STRATA_ROUTE_TAIL_SKIP=7 is ON: missed route\n'
        'strata serve: route tail skip: 15 missed experts skipped (21 entries) since the start\n'
        'strata decode timing: 4 windows, avg T 2.0, 2.0 tokens/window, 25.0 ms/window; adapt wait 1.25 + verify 20\n')
  result=h.optimization_log_summary(text)
  self.assertTrue(result['tail_skip_reported'])
  self.assertEqual(result['tail_skip_cumulative_samples'],[{'experts':15,'entries':21}])
  self.assertEqual(result['adapt_wait_ms_per_window_samples'],[1.25])
  self.assertEqual(result['tail_skip_setting_lines'],['STRATA_ROUTE_TAIL_SKIP=7 is ON: missed route'])
  fetch=h.optimization_log_summary('strata serve: STRATA_ADAPT_FETCH=2: active')
  self.assertEqual(fetch['fetch_effective_setting_lines'],['strata serve: STRATA_ADAPT_FETCH=2: active'])
  empty=h.optimization_log_summary('')
  self.assertEqual(empty['tail_skip_cumulative_samples'],[])
  self.assertFalse(empty['tail_skip_reported'])
 def test_trial_stream_metrics_and_cleanup(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);base=self.base(root)
   with socket.socket() as sock:
    sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
   (root/'strata.py').write_text(FAKE_ENGINE)
   folder=root/'trial';cfg=h.configure(base,h.arms(['shape'])[1],folder,root/'seed',port)
   with patch.object(h,'check_environment'),patch.object(h,'meminfo',return_value={'MemAvailable':12000,'SwapTotal':1000,'SwapFree':1000}):
    result=h.run_phase(root,cfg,'shape2',folder,root/'seed',port,time.monotonic()+15,
                       workloads=[('fixture','fixed test prompt',8)],launch_profile=cfg,validate_ampere=True)
   self.assertEqual(result['status'],'passed',result)
   row=result['cases'][0]
   self.assertEqual(row['expert_file_reads'],0)
   self.assertEqual(row['timings']['predicted_n'],8)
   self.assertEqual(row['stream']['content_events'],4)
   self.assertIn('gap_p99_ms',row['stream'])
   # The trial's listening port must be reusable before the next arm begins.
   with socket.socket() as probe:self.assertNotEqual(probe.connect_ex(('127.0.0.1',port)),0)

 def test_groups_balanced_and_each_ablation_changes_one_field(self):
  seq=h.arms();self.assertEqual(len(seq),180)
  for group in {r['group'] for r in seq}:
   rows=h.arms([group]);self.assertEqual(rows,rows[::-1])
   if group not in ('combined','engine-reference'):
    fields={k for k in h.DAILY if len({r[k] for r in rows})>1}
    expected={'spec-lookup':{'spec','lookup'},'shape-qkv':{'shapes','tc'},'dense-lookup':{'dense','lookup'},'skip-lookup':{'skip','lookup'},'rows':{'rows'},'adapt-tail':{'nowait','tail'},'tail-spec':{'tail','spec'},'emb-gdn':{'emb','gdn_np'},'gdn-precision':{'gdn_np'},'hc-upmix':{'upmix'},'gdn-execution':{'gdn_tiles','gdn_fuse'},'prefill-cap':{'prefill_cap'},'cpu-prefill':{'cpu_share','cpu_nt','cpu_max'},'adapt-mode':{'nowait','lag','overlap','fetch'},'update-bundle':{'decode_tc','tq_f16','fetch','hc_persist'},'decode-tc-spec':{'decode_tc','spec'},'tq-prefill':{'tq_f16','prefill_cap'},'hc-qfuse':{'hc_persist','qfuse'},'fetch-pin':{'fetch','pin'},'fetch-mode':{'fetch'},'upstream-adapt':{'adapt_every','adapt_swaps','adapt_decay'},'qfuse-l3':{'qfuse','l3'},'dense-min':{'dense_min'}}
    self.assertEqual(fields,expected[group],group)
  self.assertEqual(len(h.arms(['numeric'])),46)
  self.assertEqual(len(h.arms(['updates'])),64)
  self.assertEqual({(a['emb'],a['gdn_np']) for a in h.arms(['emb-gdn'])},{(0,3),(1,3),(0,2),(1,2)})
  self.assertEqual(len({(a['spec'],a['lookup']) for a in h.arms(['spec-lookup'])}),9)
  self.assertEqual(len({(a['shapes'],a['tc']) for a in h.arms(['shape-qkv'])}),4)
  self.assertEqual({(a['nowait'],a['tail']) for a in h.arms(['adapt-tail'])},{(0,0),(1,0),(0,7),(1,7)})
  self.assertEqual(len({(a['tail'],a['spec']) for a in h.arms(['tail-spec'])}),6)
  self.assertEqual(h.arms(['combined'])[1],{**h.DAILY,'group':'combined','name':'combined-daily'})


 def test_explicit_diagnostic_groups_are_outside_all(self):
  self.assertEqual(len(h.arms()),180)
  path=h.arms(['ampere-path'])
  self.assertEqual([a['rows'] for a in path],[0,2])
  self.assertNotIn('ampere-path',{a['group'] for a in h.arms()})
  arms=h.arms(['tail-fetch'])
  first=[(a['tail'],a['fetch']) for a in arms[:4]]
  self.assertEqual(first,[(0,0),(7,0),(7,2),(0,2)])
  self.assertEqual([(a['tail'],a['fetch']) for a in arms[4:]],list(reversed(first)))
  self.assertEqual(len(arms),8)
  workloads=h.comparison_workloads('tail-fetch')
  self.assertEqual([x[0] for x in workloads[:6]],
   ['warmup','long_output','long_context','kv_store','math_logic','long_needle'])
  tool_probes=[x for x in workloads if x[0].startswith('agent_tool_')]
  self.assertEqual(len(tool_probes),16)
  self.assertEqual(sum(x[0].startswith('agent_tool_cold_') for x in tool_probes),8)
  self.assertEqual(sum(x[0].startswith('agent_tool_hot_') for x in tool_probes),8)
  self.assertIn('tool', [m['role'] for m in next(x for x in tool_probes if x[0].startswith('agent_tool_hot_'))[3]['messages']])
  self.assertNotIn('tail-fetch',{a['group'] for a in h.arms()})
  self.assertNotIn('tail-fetch',{a['group'] for a in h.arms()})

 def test_short_objective_code_is_not_misclassified_as_a_truncated_long_fixture(self):
  cases=[{'case':'kv_store','target_tokens':6144,'timings':{'predicted_n':300}},
         {'case':'math_logic','target_tokens':192,'timings':{'predicted_n':20}}]
  self.assertEqual(h.long_fixture_validity_failures(cases),[])
  cases.append({'case':'long_output','target_tokens':3072,'timings':{'predicted_n':1000}})
  self.assertEqual(h.long_fixture_validity_failures(cases),['long_output'])

 def test_gpu_cache_slots_require_the_actual_startup_line(self):
  self.assertEqual(h.gpu_cache_slots_from_log('expert cache 5316 slots, 5.20 GiB'),5316)
  self.assertIsNone(h.gpu_cache_slots_from_log('expert cache auto: 5316 slots'))

 def test_ampere_dispatch_validation_observes_the_real_marker(self):
  marker='TQ Ampere paired-row GU enabled (type=145, n_embd=2560, n_ff=640)'
  self.assertFalse(h.validate_ampere_dispatch('',{'STRATA_TQ_AMPERE_ROWS':'0'}))
  self.assertTrue(h.validate_ampere_dispatch(marker,{'STRATA_TQ_AMPERE_ROWS':'2'}))
  with self.assertRaisesRegex(RuntimeError,'actual Ampere paired-row dispatch marker=False'):
   h.validate_ampere_dispatch('',{'STRATA_TQ_AMPERE_ROWS':'2'})
  with self.assertRaisesRegex(RuntimeError,'actual Ampere paired-row dispatch marker=True'):
   h.validate_ampere_dispatch(marker,{'STRATA_TQ_AMPERE_ROWS':'0'})

 def test_objective_quality_checks_allow_equivalent_math_and_unambiguous_needle(self):
  math_result=q.analyze_case({'case':'math_logic','content':
   '{"liters":15.0,"equation":"18 - (2.5 * 4) + 7 = 15"}'})
  self.assertTrue(math_result['task_evaluation']['passed'])
  wrong_math=q.analyze_case({'case':'math_logic','content':
   '{"liters":15,"equation":"18 - 2.5 + 4 + 7 = 26.5"}'})
  self.assertFalse(wrong_math['task_evaluation']['passed'])
  workloads=q.objective_workloads()
  self.assertEqual(workloads[0][0:3],('kv_store',q.CODE_PROMPT,6144))
  self.assertIn('fault=stale index',workloads[2][1])
  self.assertNotIn('detected stale index',workloads[2][1])
  needle=q.analyze_case({'case':'long_needle','content':json.dumps({
   'record_id':20117,'shard':3638,'fault':'stale index',
   'journal_offset':386234,'operator':10,'page':1989})})
  self.assertTrue(needle['task_evaluation']['passed'])

 def test_quality_diagnostics_keep_repetition_as_evidence_not_failure(self):
  repeated=q.repetition_metrics('alpha beta '*8)
  self.assertEqual(repeated['text_token_count'],16)
  self.assertEqual(repeated['longest_repeat']['period_tokens'],2)
  self.assertGreater(repeated['repeat_coverage_ratio'],0)
  result=q.analyze_case({'case':'math_logic','content':'{"liters":15,"equation":"18 - 2.5 * 4 + 7"}'})
  self.assertTrue(result['task_evaluation']['passed'])
  cold=q.analyze_case({'case':'agent_tool_cold_01','tool_calls':[{'function':{
   'name':'create_reservation','arguments':'{"city":"Kyoto","check_in":"2027-05-18","nights":3,"cancellation":"flexible"}'}}]})
  hot=q.analyze_case({'case':'agent_tool_hot_01','tool_calls':[{'function':{
   'name':'create_reservation','arguments':'{"city":"Lima","check_in":"2027-05-18","nights":4,"cancellation":"flexible"}'}}]})
  self.assertTrue(cold['task_evaluation']['passed'])
  self.assertTrue(hot['task_evaluation']['passed'])
  self.assertFalse(cold['task_evaluation']['executed'])
  self.assertFalse(hot['task_evaluation']['executed'])

 def test_quality_capture_preserves_content_and_retry_counts(self):
  from email.message import Message
  events=[
   {'strata_generation_recovery':{'reason':'synthetic repeat recovery'}},
   {'choices':[{'delta':{'content':'fixed output'}}]},
   {'choices':[{'finish_reason':'stop'}],
    'timings':{'prompt_ms':10,'predicted_ms':100,'predicted_n':8,'predicted_per_second':80}},
  ]
  response=io.BytesIO(b''.join(b'data: '+json.dumps(e).encode()+b'\n\n' for e in events)+b'data: [DONE]\n\n')
  response.headers=Message();response.headers['Content-Type']='text/event-stream'
  with patch('l2.chat_latency.urllib.request.build_opener') as opener:
   opener.return_value.open.return_value=response
   row=h.measure('http://127.0.0.1:1','math_logic','synthetic prompt',192,10,capture_output=True)
  self.assertEqual(row['content'],'fixed output')
  self.assertEqual(row['generation_recovery_count'],1)
  self.assertEqual(row['token_id_source'],'not exposed; no output retokenization performed')

if __name__=='__main__':unittest.main()
