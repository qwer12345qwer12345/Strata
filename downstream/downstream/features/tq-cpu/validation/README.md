# Validation record — TQ CPU

- Source: PR #1, CPU AVX-VNNI portion.
- Baseline: `c6f6427d9de9016fd92d13e29740f65625261df7`.
- Patch package migration: created from the PR diff; generated Settings outputs and shared PR documentation were excluded.
- Test coverage: existing TQ CPU parity program, baseline and opt-in CTest registrations.
- Status: not executed during package migration.
- Required qualification: build and run on an AVX-VNNI-capable x86 CPU; record compiler, CPU model, parity result and latency separately.
