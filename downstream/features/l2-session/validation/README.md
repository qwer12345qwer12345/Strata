# Validation record — Persistent L2

- Source: PR #2.
- Baseline: `6e38f588a646c0ee6948a74afbe670eb75896106`.
- Test source: updated conversation spill regression preserved under this package's test overlay.
- Status: not executed during package migration.
- Required qualification: build and run conversation spill tests on supported POSIX and Windows filesystems; add fault-injection coverage for payload/sidecar publication, ENOSPC and restart recovery.
