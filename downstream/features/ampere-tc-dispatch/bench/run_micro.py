#!/usr/bin/env python3
"""Build first, then run SM86 microbenchmarks sequentially. Uses an existing CUDA build."""
import argparse,json,os,shlex,subprocess
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',required=True);p.add_argument('--build',required=True);p.add_argument('--output',required=True)
p.add_argument('--columns',default='2,4,6,8');p.add_argument('--variants',action='store_true')
a=p.parse_args();source=Path(a.source).resolve();build=Path(a.build).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
here=Path(__file__).resolve().parent
# Finish all compile/link steps before executing any GPU measurement.
subprocess.run(['cmake','--build',str(build),'--target','dflash2_gemm_bench','dense_i8_tc_test','-j2'],check=True)
for target,name in [('dflash2_gemm_bench','df2_compare'),('dense_i8_tc_test','dense_compare'),('dflash2_gemm_bench','df2_variants')]:
    line=subprocess.check_output(['ninja','-t','commands',target],cwd=build,text=True).splitlines()[-1]
    commands=[shlex.split(x) for x in line.split(' && ')]
    cmd=next(c for c in commands if c and c[0].endswith('c++'))
    objects=[i for i,x in enumerate(cmd) if x.endswith('.o')]
    if len(objects)!=1:raise RuntimeError('Unsupported link command; expected one benchmark object')
    cmd[objects[0]]=str(here/(name+'.cpp'));cmd[cmd.index('-o')+1]=str(out/name)
    toolkit=os.environ.get('CUDA_HOME','/opt/cuda')
    cmd[1:1]=['-std=c++20','-I'+str(source/'engine/include'),'-I'+str(Path(toolkit)/'include')]
    subprocess.run(cmd,cwd=build,check=True)
runs=[]
for col in map(int,a.columns.split(',')):
    for arm in ('off','on','on','off'):
        name=f'df2-{col}-{len(runs)}-{arm}.log'
        with (out/name).open('w') as f:
            subprocess.run([str(out/'df2_compare'),'320','q4',str(col)],stdout=f,stderr=subprocess.STDOUT,
                           env=dict(os.environ,STRATA_DF2_GEMM_TC='1' if arm=='on' else '0',STRATA_DF2_AMPERE_TUNING='0'),check=True)
        runs.append(dict(cols=col,arm=arm,log=name))
        (out/'runs.json').write_text(json.dumps(runs,indent=2)+'\n')
    if a.variants:
        for repeat in range(2):
            with (out/f'variants-{col}-{repeat}.log').open('w') as f:
                subprocess.run([str(out/'df2_variants'),'320','q4',str(col)],stdout=f,stderr=subprocess.STDOUT,
                               env=dict(os.environ,STRATA_DF2_GEMM_TC='0',STRATA_DF2_AMPERE_TUNING='0'),check=True)
with (out/'dense.jsonl').open('w') as f:
    subprocess.run([str(out/'dense_compare')],stdout=f,env=dict(os.environ,STRATA_DENSE_I8_AMPERE_SHAPES='0'),check=True)
