# Validation — Shutdown Persistent Cache

Status: production patch and pure-host test overlay are written; the host test was compiled and run locally.
The engine wiring (serve-thread restore, shutdown flush) is not yet exercised on a real model.

## Verified while preparing the package

- The patch is based on the pinned upstream commit `c6f6427d9de9016fd92d13e29740f65625261df7`.
- The production change adds one header-only host module (`engine/include/strata/core/conversation_persist.hpp`),
  a read-only entry accessor on `ConversationCache`, the serve wiring in `generate.cpp` (opt-in env, lazy open,
  restore hook after the runtime SSD tier, shutdown flush after the request loop), and an optional guarded CTest
  registration. No upstream behavior changes when the feature is off.
- The host test compiled and passed locally (MSVC x64, C++20): 54 checks, 0 failed, against the exact patched
  sources of this package (`conversation_file.cpp` + the overlay), covering the acceptance criteria of the patch
  plan: metadata-only index rebuild, matched restore, miss fallback, namespace isolation, records surviving
  restores, replace/budget eviction, and fail-safe behavior on missing/corrupt sidecars and a full write.
- `generate.cpp` was compiled to an object file with the same toolchain (syntax and name check of the wiring); a
  full engine link and run were not performed here.
- The standalone source tree and the full feature-stack tree are recorded in `feature.json` and `stack.json`; the
  patch SHA256 is recorded in `feature.json`.

## Pending

1. Run `python downstream/scripts/verify.py` in a clean clone (patch hashes, standalone tree, full stack tree).
2. Run the package's prepare.py and test.py commands on a CPU build (the CTest path inside the standard runner).
3. A serve run with a real model and `STRATA_SHUTDOWN_CACHE_MIB` set: park a conversation, end the engine
   orderly, restart, confirm the index is rebuilt and the matched prompt restores without reading the prompt
   again; confirm a miss (different model/config namespace, changed `--max-context`, corrupt record) falls back
   to a normal prefill.
4. A serve run with the runtime SSD tier also on, to confirm the two tiers compose (spill wins on its longer
   prefix; persist answers when the spill tier has none or did not win).
5. A disk-full / read-only-folder drill: confirm only the affected record is lost and the engine still starts.
