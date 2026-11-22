# Validation

The standalone source was prepared from pinned upstream `c6f6427d9de9016fd92d13e29740f65625261df7` using the package scripts. A second source was prepared with the complete downstream stack, including Live Memory's server changes.

Both sources passed the ten `ChatMLToolGuard` regressions and the eight existing `UnfinishedToolCall` regressions, 18 tests per source. Disabled-mode regressions cover the previous role output and incremental tool argument streaming. Package SHA256, standalone tree and full stack tree verification passed.

The checks used Python 3.12 on Linux and the real local API handler with synthetic engine output. Real-model tool error frequency and long tool argument latency have yet to be measured on the new engine.
