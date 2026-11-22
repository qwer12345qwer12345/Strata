# TQ CPU tests

This package reuses the pinned upstream `engine/src/kernels/cpu/tqp_cpu_test.cpp` synthetic parity program. It compares TQ2_T, TQK6, TQK7 and the TQP formats against the scalar path across groups of 1 through 8 tokens, including gate/up. The two CTest invocations are

- `tqp_cpu_test` exercises upstream ISA dispatch.
- `tqp_cpu_test_avxvnni` enables `STRATA_TQ_AVXVNNI=1` with AVX2 dispatch.

Run from the downstream repository root:

```bash
python downstream/scripts/test.py --feature tq-cpu \
  --source ../Strata-prepared --build ../Strata-build-tq-cpu
```

Configure with an x86 CPU build and `STRATA_NATIVE_EXPERTS=ON`. The AVX-VNNI invocation must be run on a CPU that supports AVX-VNNI to qualify the optimized path; unsupported CPUs retain the runtime fallback.
