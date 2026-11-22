"""Link a shape microbenchmark against a completed CUDA engine build.
Usage: python compile_shape_bench.py SOURCE_ROOT BUILD_DIR OUTPUT_DIR [CUDA_INCLUDE]
All engine compilation must finish before this script or any measurements run.
"""
import shlex
import subprocess
import sys
from pathlib import Path
source,build,out=map(lambda x:Path(x).resolve(),sys.argv[1:4])
include=sys.argv[4] if len(sys.argv)>4 else '/opt/cuda/targets/x86_64-linux/include'
out.mkdir(parents=True,exist_ok=True)
line=subprocess.check_output(['ninja','-t','commands','dflash2_gemm_bench'],cwd=build,text=True).splitlines()[-1]
base=next(shlex.split(x) for x in line.split(' && ') if shlex.split(x)[0].endswith('c++'))
for name in ['shape_bench','shape_abba']:
 cmd=base.copy();objects=[i for i,x in enumerate(cmd) if x.endswith('.o')]
 assert len(objects)==1
 cmd[objects[0]]=str(Path(__file__).with_name(name+'.cpp'))
 cmd[cmd.index('-o')+1]=str(out/name)
 cmd[1:1]=['-std=c++20','-I'+str(source/'engine/include'),'-I'+include]
 subprocess.run(cmd,cwd=build,check=True)
