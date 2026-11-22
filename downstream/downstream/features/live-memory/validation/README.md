# Validation record — Live Memory

- Source: PR #3, then PR #4 (Phase 2 RAM blocks, Phase 3 engine protocol, Phase 4 server policy).
- Baseline: `c6f6427d9de9016fd92d13e29740f65625261df7`.
- Test sources: CPU policy, CPU RAM-blocks, CUDA segmented ExpertCache and Settings scanner tests preserved
  under this package's test overlay; the server policy and protocol tests ride `terminal/serve`'s suites.
- Status: the CPU tests pass on the migration machine (Ryzen laptop, Windows, Python 3.14), and the CUDA
  segmented-cache test passes on its RTX 4060 Laptop (sm_89, CUDA 12.5, driver VMM): the 6-patch stack
  configures, builds and runs on a real VMM device. The full-engine safe-point paths (MEMORY during a live
  conversation, Vision turns beside the policy thread) are still awaiting a Linux qualification run; the
  package's CI runs the maintenance scripts, hashes and tree fingerprints only.
- Required qualification:
  - CPU: `live_memory_policy_test` (grammar, ACK/CAPACITY lines, pending supersede), `live_ram_blocks_test`
    (grow publish-or-rollback, allocation and copy failure injection, release order, pinned fallback),
    Settings scanner regression, `test_live_memory_policy.py`, `test_live_memory_protocol.py`.
  - CUDA: `expert_cache_segmented_test` on a supported VMM device/driver; verify the initial commit size,
    `resize_transactional` grow/shrink behavior and failure handling.
  - Engine-on-hardware: MEMORY requests at a real safe point (loans out, windows in flight), ACK truth under
    a refused grow, CAPACITY cadence, a Vision turn through the shared owner while the policy thread runs.
- Scope note: PR #4 implements the Phase 2 RAM blocks, the Phase 3 MEMORY/ACK/CAPACITY protocol with the
  engine as final admission authority, and the Phase 4 opt-in server policy with the unified Vision owner.
  The whole stack stays opt-in (`STRATA_LIVE_MEMORY=1` / config `"live_memory"`; the policy additionally
  `STRATA_LIVE_MEMORY_POLICY=1` / config `"live_memory_policy"`).
