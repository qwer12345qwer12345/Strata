# Validation record — Live Memory

- Source: PR #3.
- Baseline: `6e38f588a646c0ee6948a74afbe670eb75896106`.
- Test sources: CPU policy, CUDA segmented ExpertCache and Settings scanner tests preserved under this package's test overlay.
- Status: not executed during package migration.
- Required qualification: CPU policy and Settings tests; CUDA VMM tests on a supported device/driver; verify initial commit size, grow/shrink behavior and failure handling.
- Scope note: no claim is made that the MEMORY/ACK/CAPACITY runtime protocol or dynamic RAM/VRAM policy exists.
