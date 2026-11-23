#!/usr/bin/env python3
"""Build the disposable SM86 layout sweep; measure only when --run is supplied."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def run(command: list[str], cwd: Path | None = None, **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="disposable source created by downstream/scripts/prepare.py")
    parser.add_argument("--build", required=True, help="dedicated CUDA build directory")
    parser.add_argument("--output", required=True, help="directory for environment and optional JSONL results")
    parser.add_argument("--run", action="store_true", help="run GPU measurements after the build; requires SM86")
    parser.add_argument("--iters", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    source, build, output = (Path(x).expanduser().resolve() for x in (args.source, args.build, args.output))
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        raise RuntimeError(f"prepared source does not exist: {source}")
    missing = [tool for tool in ("cmake", "nvcc") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"required build tools are not on PATH: {', '.join(missing)}")

    test = ROOT / "downstream" / "scripts" / "test.py"
    command = [sys.executable, str(test), "--feature", "ampere-tc-dispatch", "--source", str(source),
               "--build", str(build), "--with-cuda", "--build-only"]
    for value in ("-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=86", "-DSTRATA_WINDOW_MAX_T=16",
                  "-DSTRATA_BUILD_TESTS=ON"):
        command.extend(["--cmake-arg", value])
    run(command, cwd=ROOT)

    environment = {
        "source_revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "feature_patch_sha256": hashlib.sha256((ROOT / "downstream" / "features" / "ampere-tc-dispatch" /
                                                 "patches" / "0001-ampere-tc-dispatch.patch").read_bytes()).hexdigest(),
        "target_architecture": "SM86",
        "window_max_t": 16,
        "build_type": "Release",
        "measurement_requested": bool(args.run),
        "gpu": None,
        "cuda_compiler": None,
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run([nvidia_smi, "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader"],
                                text=True, capture_output=True)
        environment["gpu"] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    nvcc = shutil.which("nvcc")
    if nvcc:
        result = subprocess.run([nvcc, "--version"], text=True, capture_output=True)
        environment["cuda_compiler"] = (result.stdout or result.stderr).strip()
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(f"Build ready. Environment recorded in {output / 'environment.json'}.")
    if not args.run:
        print("No GPU measurements were run. Re-run with --run on an RTX 3080 / SM86 host.")
        return 0

    names = ["dense_layout_sweep.exe", "dense_layout_sweep"]
    binaries = [p for name in names for p in build.rglob(name) if p.is_file()]
    if len(binaries) != 1:
        raise RuntimeError(f"expected one dense_layout_sweep executable under {build}, found {binaries}")
    env = dict(os.environ, STRATA_DENSE_I8_TC="0", STRATA_DENSE_I8_AMPERE_SHAPES="0", STRATA_Q6_PACKED="1")
    with (output / "dense-layout.jsonl").open("w", encoding="utf-8", newline="\n") as out, \
         (output / "dense-layout.stderr.log").open("w", encoding="utf-8", newline="\n") as err:
        run([str(binaries[0]), str(args.iters), str(args.pairs)], cwd=source, env=env, stdout=out, stderr=err)
    print(f"Measurements saved to {output / 'dense-layout.jsonl'}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"dense layout sweep failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
