# Shutdown Persistent Cache tests

The `conversation_persist_test` is a pure-host CTest target (no GPU, no model). The test runner copies its source
overlay into the prepared source tree before configuring CMake; the production patch registers the target only when
that overlay exists.

From the repository root, using an already prepared source tree:

```bash
python downstream/scripts/test.py --feature shutdown-persistent-cache \
  --source ../Strata-prepared --build ../Strata-build-shutdown-persist
```

The test covers: a shutdown flush of several parked conversations; the next start's scan rebuilding the same index
from sidecars alone (serials, sizes, steering, live/checkpoint stubs); a restore that reads the payload back byte
for byte and does not consume its record; a restore that re-reads the same record (the next restart); a payload
that no longer reads being dropped with its sidecar, alone; identity namespace isolation (two configurations never
share records); a new snapshot of the same conversation replacing the older record only once it is complete, with a
sibling that shares a prefix keeping its record; budget refusals (an image larger than the whole budget, a third
record pushing the oldest out whole, a lowered budget retiring at the scan); a damaged namespace (a cut sidecar, a
foreign file and a missing payload are ignored without deleting anything else); refused images (an empty live
state, a layer split) leaving nothing on the disk; and the namespace lock (a second engine's flush writing nothing
while another engine holds it).

The host test was compiled and run locally with the MSVC toolchain against the pinned base commit; the CUDA build
is not exercised by this package and the engine wiring (serve-thread restore and the shutdown flush) still needs a
serve run with a real model to qualify (see `validation/README.md`).
