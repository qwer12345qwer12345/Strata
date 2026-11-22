# Validation — Runtime KV Storage

Status: production patch and pure-host test overlay are written. Compilation and test execution are still pending.

## Verified while preparing the package

- The patch is based on the pinned upstream commit 6e38f588a646c0ee6948a74afbe670eb75896106.
- The production change is limited to the SessionFile header/implementation and optional pure-host CMake test registration.
- The standalone source tree was constructed from the exact modified source blobs: aa9fe3e95f15d80ab077914fc580be67c32f4228.
- The full feature-stack tree including this patch was constructed from the existing stack tree plus the same source changes: 929c0a798b10ef83d4da33c7f1dccb8b821799e7.
- Patch SHA256 is recorded in feature.json.

These tree objects are metadata calculations; they are not a substitute for running downstream/scripts/verify.py in a local clone.

## Pending

1. Run python downstream/scripts/verify.py and confirm all standalone and full-stack patch applications.
2. Run the package prepare.py and test.py commands on a CPU build.
3. Run the full conversation/session CTest suite to check the existing STRSESS v1 behavior remains unchanged.
4. Add Snapshot provenance tests for every officially supported KV representation and residency mode.
5. Only after those gates, integrate non-destructive spill Restore, lineage leases and Delta commits into ConversationSpill.
6. Run multi-session churn/soak tests and measure SSD bytes written, read bytes, restore latency and RAM peak.

No runtime write-amplification improvement is claimed by Patch 1 alone.
