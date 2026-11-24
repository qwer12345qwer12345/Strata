# Strata Tail Skip x ADAPT_FETCH quality diagnostic (2026-11-24)

## Scope and conclusion

This PR is separate from PR #12 and contains only the diagnostic harness snapshot plus sanitized evidence. It does not change the engine, kernel, daily runtime settings, or select a winning parameter set.

The final 2x2 matrix completed all eight fresh starts in the requested order A/B/C/D then D/C/B/A. The suite status is **incomplete/invalid**, not passed: every arm failed the independent KV-store implementation gate; both C starts (TailSkip=7, ADAPT_FETCH=2) also failed the arithmetic task. All eight long-needle checks and all 128 synthetic tool-call probes passed. No external tool was executed.

## Runtime and provenance

- Peer device: NVIDIA RTX 3080 12 GiB (SM86); test binary SHA-256 49c4d277e65c782e239172ec40dab3ec73053d4bd4ea4fbcc67dc7757d877980.
- Runtime source checkout HEAD: 6a6db7ff58c283cf28d2b913404b5fa31a767eaa. The runtime worktree was dirty (66 dirty/untracked status entries). The compatible test binary was assembled via a temporary object/archive link from that dirty tree; the paired-row kernel source was not included here, and the original build artifact was restored afterward. Therefore the exact binary is not reproducible from the checked-in source commit alone.
- One fresh server process per arm/start; same model and frozen expert-profile seed; 64-token warmup; 3072-token long output; 1536-token long context; synthetic KV task max budget 6144; synthetic math, needle, and cold/hot tool fixtures.
- Common requested settings: spec=4, draft confidence=0.8, lookup max T=16, 15 pool workers, expert cache measured at 5316 slots, ADAPT every 2 / swaps 192 / decay 0.92, ADAPT_NOWAIT=1, ADAPT_LAG=2, cache pin fraction=0.25; only TailSkip (0/7) and ADAPT_FETCH (0/2) vary.
- The launch environment records the requested TailSkip/ADAPT_FETCH values. The engine explicitly logs TailSkip=7 when enabled; absence of a positive line is not treated as an engine-reported disable for zero. ADAPT_FETCH is recorded from the effective launch environment; do not infer a runtime log marker that the engine does not emit.
- Every request reported 0 expert blob-file reads. The 64-token warmup reports 5580.7 MB of mmap-accounted expert pages per fresh start; each timed case reports 0.0 MB. This is an expert-read metric, not a claim of zero OS disk I/O for all files.

## Long-output observations

| TailSkip | ADAPT_FETCH | runs (tok/s) | repeat coverage / longest repeated span | output SHA-256 (8-char prefix) |
|---:|---:|---:|---|---|
| 0 | 0 | 85.3, 85.7 | 19.75% / 380 tok (period 64, 5 cycles), 19.75% / 380 tok (period 64, 5 cycles) | 77d65453, 77d65453 |
| 7 | 0 | 145.8, 145.8 | 93.55% / 1884 tok (period 64, 29 cycles), 93.55% / 1884 tok (period 64, 29 cycles) | f36fd5d1, f36fd5d1 |
| 7 | 2 | 100.5, 100.0 | 46.81% / 1047 tok (period 64, 16 cycles), 46.81% / 1047 tok (period 64, 16 cycles) | 8a736886, 8a736886 |
| 0 | 2 | 86.3, 86.7 | 5.90% / 166 tok (period 64, 2 cycles), 5.90% / 166 tok (period 64, 2 cycles) | b786b935, b786b935 |

The high-throughput TailSkip=7/ADAPT_FETCH=0 output reproduces the same hash in both starts and is dominated by a long repeated span; it is not a valid performance winner. The C arm is also degraded and returns 15.5 instead of 15 in both math trials. The D arm has the lowest observed long-output repetition, but the KV-store task still fails. Therefore no parameter winner is supported.


## Preparation and superseded attempts

The following attempts were not counted in the final matrix: an older installed engine exited before serving because it lacked the expected model block-count metadata; a system-Python launch lacked the bundled runtime dependencies and never reached inference; the first compatible-binary attempt exposed a cache-slot log parser mismatch, which was corrected; and the earlier exploratory TailFetch run used an incorrect quality evaluator. Those runs are not presented as valid comparisons. The authoritative final run is the 8-start report and raw artifact bundle referenced below.


## Files

- ../../tests/l2/: six-file snapshot of the modified Python test harness. Run from that tests/ directory with python3 -m unittest -v l2.test_headless_experiments.
- results-final.json.gz: complete sanitized aggregate, including all synthetic prompt/response content, reasoning, tool-call payloads, finish reasons, hashes, token IDs only where exposed, effective settings and per-window traces.
- tail-fetch-raw-artifacts.tar.gz: sanitized per-start server.log, engine.log, and raw spec.trace for all eight starts; per-start result values are in the complete aggregate. Private profiles/seeds are excluded.
- ampere-path-artifacts.tar.gz: the two-arm Ampere requested-vs-actual dispatch check with result and engine logs/traces; zero rows has no paired-row marker and two rows has the actual paired-row marker.
- runtime-evidence.json, unit-tests.log, and SHA256SUMS.txt: compact machine-readable evidence, reproducible test output, and artifact hashes.

Peer-local paths and the private Tailscale address were replaced; authorization material and private profiles/seeds were not included. All request fixtures are synthetic and tool execution is disabled.

## Validation

- py_compile passed for all six staged files.
- The staged snapshot passed all 17 l2.test_headless_experiments unit tests.
- The independent Ampere dispatch check passed for both the off and paired-row arms.
- Daily strata.service was active when artifacts were packaged; no daily settings were changed.
- The next, conditional phase was not run. This report is diagnostic evidence, not a standard benchmark or a claim that the engine issue is repaired.
