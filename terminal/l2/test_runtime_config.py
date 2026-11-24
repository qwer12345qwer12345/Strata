import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from l2 import runtime_config as runtime

class RuntimeConfiguration(unittest.TestCase):
 def fixture(self, root):
  engine=root/'runtime';(engine/'build').mkdir(parents=True)
  (engine/'strata.py').write_text('')
  (engine/'build/strata').write_bytes(b'engine')
  ple=root/'bf16.gguf';ple.write_bytes(b'bf16')
  state=root/'state';state.mkdir()
  profile={'exe':'previous-engine','cwd':str(engine),'args':['--spec','6','--spec','6','--ple-gguf','old'],
           'user_args':['--spec','6'],'env':{'STRATA_ROUTE_PRIOR':'0.5'},
           'startup_benchmark':{'output_dir':str(state)}}
  selection={'base_exe':'previous-engine','root':str(engine),
   'engine_sha256':hashlib.sha256(b'engine').hexdigest(),'log':str(root/'engine.log'),
   'archive_dir':str(root/'archives'),'ple_gguf':str(ple),'ampere_rows':'2',
   'environment':{'STRATA_DF2_AMPERE_TUNING':'2','STRATA_DF2_GEMM_TC':'0','STRATA_ADAPT_NOWAIT':'1','STRATA_ROUTE_TAIL_SKIP':'7'},
   'argument_overrides':{'--spec':'4','--lookup-max-t':'16'},'configuration_status':'NOT TESTED',
   'vision':{'gpu':True,'expert_lease':True},'vram_elastic':True,'repeat_failure_retry':True,
   'flag_overrides':{'--session-save-reclaim':True}}
  (state/'production-baseline.json').write_text(json.dumps(selection))
  file=root/'profile.json';file.write_text(json.dumps(profile))
  return profile,file,state
 def test_single_selection_overrides_duplicate_old_values(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);profile,_,state=self.fixture(root);before=copy.deepcopy(profile)
   _,cfg=runtime.select(root,profile,state)
   self.assertEqual(cfg['args'].count('--spec'),1)
   self.assertEqual(cfg['args'][cfg['args'].index('--spec')+1],'4')
   self.assertEqual(cfg['user_args'][cfg['user_args'].index('--spec')+1],'4')
   self.assertEqual(cfg['env']['STRATA_DF2_GEMM_TC'],'0')
   self.assertEqual(cfg['env']['STRATA_ADAPT_NOWAIT'],'1')
   self.assertEqual(cfg['env']['STRATA_ROUTE_TAIL_SKIP'],'7')
   self.assertEqual(cfg['configuration_status'],'NOT TESTED')
   self.assertTrue(cfg['vision']['expert_lease']);self.assertTrue(cfg['vram_elastic'])
   self.assertTrue(cfg['repeat_failure_retry'])
   for key in ('args', 'user_args'):
    self.assertEqual(cfg[key].count('--session-save-reclaim'), 1)
   self.assertEqual(profile,before)
 def test_legacy_profile_pins_selection_for_official_drafting(self):
  from strata_control import user_args
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);profile,_,state=self.fixture(root)
   profile.pop('user_args')
   _,cfg=runtime.select(root,profile,state)
   mine=user_args.capture(cfg)
   self.assertEqual(mine[mine.index('--spec')+1],'4')
   self.assertEqual(mine[mine.index('--lookup-max-t')+1],'16')
if __name__=='__main__':unittest.main()
