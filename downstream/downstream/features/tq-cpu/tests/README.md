# TQ CPU tests

This package reuses the pinned upstream `src/kernels/cpu/tq_cpu_parity.cpp` parity program and registers two CTest invocations:

- `tq_cpu_parity`: baseline AVX2/default dispatch.
- `tq_cpu_parity_avxvnni`: `STRATA_TQ_AVXVNNI=1`.

Run from the downstream repository root:

```bash
python downstream/scripts/test.py --feature tq-cpu \
  --source ../Strata-prepared --build ../Strata-build-tq-cpu
```

Configure with an x86 CPU build and `STRATA_NATIVE_EXPERTS=ON`. The AVX-VNNI invocation must be run on a CPU that supports AVX-VNNI to qualify the optimized path; unsupported CPUs retain the runtime fallback.
