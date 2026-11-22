# QSA IMMA tests

The test project contains the PR's complete synthetic parity program under `overlay/engine/src/kernels/qsa_prompt_attn_parity.cpp`. The central test runner temporarily overlays it at the source path expected by the CMake target, then restores the prepared source tree after the test.

```bash
python downstream/scripts/test.py --feature qsa-imma \
  --source ../Strata-prepared --build ../Strata-build-qsa-imma
```

Requires a CUDA build. The test explicitly skips unless the active device is SM86. A skip is not evidence that the IMMA implementation passed.
