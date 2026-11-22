# TQ Ampere tests

The package uses the upstream `tq_pascal_bench` target. CTest runs its synthetic fixtures at the default row mode and with `STRATA_TQ_AMPERE_ROWS=2`. On the SM86 qualification machine the two runs also had identical checksums for all 30 `SUM` records. These fixtures include grouped entries, empty groups, varied token counts and fallback dimensions.

The package registers two tests on the existing `native_expert_parity` executable:

- `native_expert_parity_tqk6_tqk7_ampere_pair`
- `native_expert_parity_tq2_t_ampere_pair`

Both run with `STRATA_TQ_AMPERE_ROWS=2`.

```bash
python downstream/scripts/test.py --feature tq-ampere \
  --source ../Strata-prepared --build ../Strata-build-tq-ampere
```

Requires a CUDA build with native experts enabled. Run on SM86 hardware for meaningful qualification; other architectures exercise the fallback route rather than the Ampere specialization.
