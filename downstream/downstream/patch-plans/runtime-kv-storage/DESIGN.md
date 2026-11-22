# Runtime KV Storage — design

Status: implementation in progress  
Feature package: `runtime-kv-storage`  
Patch plan: Patch A only  
Upstream base: `Niko1221/Strata:main` at `c6f6427d9de9016fd92d13e29740f65625261df7`

## 1. Problem and objective

The official runtime SSD tier is already the correct place to park evicted conversations. The write amplification comes from its lifecycle: `ConversationSpill::put()` writes a complete `SavedConversation`, and `take_serial()` consumes and deletes that file. If a conversation is restored and later evicted again, the complete KV image is written again even when most of its prefix is unchanged.

The objective is to retain the useful disk image across runtime restores and write only the changed KV suffix on subsequent parks. This is not a second cache tier and not a persistent-across-restarts feature.

## 2. Chosen storage model

A logical spill entry is:

```text
Entry
 ├── immutable Base (one complete STRSESS v1 file)
 ├── Delta generation 1 (independent STRSDLT file)
 ├── Delta generation 2 (independent STRSDLT file)
 └── ...
```

- A Base is always a complete official session file and is never modified in place.
- Each Delta is a separate immutable file. It binds to one chain ID and exactly one parent generation.
- The first parent generation is 0 (the Base); every accepted Delta advances by exactly one generation.
- A failed write never advances the in-memory entry generation.
- A new Base starts a new chain ID and generation 0. Old files remain until the new Base is fully published and the entry switches.
- A chain ID is a per-entry opaque nonzero identifier. It is not a model identity; model/config compatibility remains the responsibility of `SessionFileIdentity`.

## 3. Delta semantics

Each Delta describes the complete next logical conversation state, but stores only changed KV bytes:

1. Full non-KV state for the new generation: geometry, layer range, cvec flag, live checkpoint and saved checkpoints.
2. For each KV layer, the official layout metadata for the target generation.
3. For each of the five official KV parts — K, V, K scale, V scale, pooled/indexer — a `keep_bytes` prefix and the changed tail bytes.

The materialized part is:

```text
new_part = parent_part[0 : keep_bytes] + delta_tail
```

A `keep_bytes = 0` part is a full replacement. This supports rewind, truncation, or a part whose prefix is not reusable without introducing a second format. If the layout/format itself changes, the runtime should write a new Base rather than append a Delta.

The Delta core treats part bytes as opaque. It does not assume FP16, INT8, Q4, K8V4, scale layout, page geometry, ring slots, or pooled-row arithmetic. The Snapshot adapter obtains exact part sizes and keep boundaries from official Snapshot functions. It must not infer byte counts from token counts.

## 4. Format and validation

STRSESS v1 remains unchanged. Delta uses a separate versioned STRSDLT header and its own end marker.

The header binds:
- file format version and header size;
- model/config identity;
- chain ID;
- parent generation and new generation;
- payload byte length;
- KV layer count;
- header checksum.

The payload is hashed in full and followed by the payload hash and Delta end marker. The payload begins with five keep lengths per layer, followed by the same complete non-KV metadata and five tail buffers per layer in the existing SessionFile serialization order.

Reader checks, before applying any bytes:
- regular file, no symlink/reparse point or hard-link alias, bounded file size;
- magic/version/header size/reserved fields/header hash;
- model/config identity, chain ID and exact parent generation;
- strictly consecutive generation;
- layer count and all array lengths against caller limits and remaining payload;
- payload hash and end marker;
- base/delta layer count and static KV layout compatibility;
- every keep length is within its parent part, and keep + tail has no integer overflow.

The Delta core cannot prove that a byte offset is a valid token/page boundary. That is a caller contract: only the Snapshot adapter may construct keep lengths, using the official per-part size API. Any uncertainty forces a new Base.

Hashes detect accidental corruption; they are not authentication. Restore only trusted files.

## 5. Restore model

Restore always performs a full reconstruction:

1. Read the complete immutable Base with `session_file_read()` and existing `SessionReadLimits`.
2. For each Delta in chain order, require its parent generation to equal the current generation.
3. Parse and validate the entire Delta before mutating the in-progress image.
4. Apply each part as parent prefix + Delta tail; replace the non-KV state with the Delta's full state.
5. After the last Delta, pass the complete `SavedConversation` to the existing `conversation_snapshot_validate()` / `conversation_snapshot_restore()` path.

A partial materialization is never exposed to inference. If a Delta is missing, corrupt, out of order, incompatible, or fails during application, discard the in-progress image and treat the chain as a cache miss. Do not continue with a partially modified image.

The Base read is non-destructive. Runtime entry ownership and a lease prevent compaction/eviction from deleting files while a restore or active session still refers to that chain.

## 6. Official KV coverage

The target is the full set of KV representations that the current whole-session Snapshot contract can authoritatively capture and restore:

- FP16 K/V;
- INT8 K/V and scale buffers, including supported rotated-INT8 format;
- Q4_0 K/V;
- K8V4 hybrid K/V;
- resident and streamed QSA modes (the authoritative host pool is the source for streamed modes);
- ring-backed draft KV where the Snapshot layer captures its authoritative logical image;
- indexer pooled rows;
- MTP draft KV supported by the current Snapshot contract.

The five-part representation is already the common persistence boundary in `SessionKvSource` and `ConversationKv`. Delta does not persist transient VRAM slot tags, ring slot mappings, or residency pointers. Restore reconstructs the logical bytes and lets the existing Snapshot restore path rebuild runtime residency/ring state.

The current official whole-session file path rejects layer-split `stage_parts`; this feature does not silently claim to add that unsupported topology.

## 7. Lineage and dirty-range source

Token equality is necessary for conversation matching but is not sufficient proof that KV bytes are reusable. A Delta is allowed only when all of the following hold:

- the restored conversation carries the same spill chain ID and generation;
- the retained KV provenance belongs to that exact chain;
- the official reuse/dirty tracking says the prefix before the first dirty token is unchanged;
- model/config identity, geometry, layer range, format and per-layer layout still match;
- exact per-part keep sizes are supplied by the official Snapshot API.

Append-only turns normally produce a nonzero keep prefix. Rewrites, compaction of conversation text, branch changes, image changes that invalidate the prefix, ring/layout changes, unknown provenance, or generation mismatch use a new Base. If no KV bytes changed and non-KV state is unchanged, no Delta should be written.

The implementation must not scan or compare multi-gigabyte KV images to discover a prefix. That would exchange SSD write amplification for a large CPU/memory-bandwidth scan.

## 8. Runtime ownership and concurrency

- One writer may advance a given chain at a time; writes to different chains may remain independent.
- A restore acquires a lease on the chain. A leased chain cannot be evicted or compacted.
- The entry generation advances only after a Delta file is completely written, flushed and atomically published.
- The existing BackgroundWriter remains the only runtime spill writer.
- Spill budget counts Base bytes + all Delta bytes + in-flight temporary bytes.
- LRU eviction drops a whole unleased chain, never just its Base or an arbitrary Delta.
- A restore does not consume/delete a chain. The chain is retained until a new Base supersedes it or the whole entry is evicted.

## 9. Compaction

Compaction is deliberately simple and deterministic. Initial thresholds:

- 8 Deltas in one chain; or
- total Delta bytes >= 50% of Base bytes; or
- one Delta >= Base bytes.

Compaction writes a complete new Base using the normal session-file writer. Only after its publication succeeds does the entry switch to the new Base/generation 0 chain. Old Base and Deltas are then removed. If free-space admission or the write fails, the old chain remains valid and usable.

Thresholds are tunable after measuring real workloads; they are not correctness assumptions.

## 10. Patch sequence

1. **Session Delta Core:** independent Delta file format, writer/reader, validation and Base+Delta materialization. Pure Host tests. No runtime behavior change.
2. **Snapshot adapter:** carry lineage and dirty-range provenance; compute exact five-part keep lengths using official Snapshot APIs; add mode coverage tests.
3. **ConversationSpill integration:** immutable Base refs, ordered Delta refs, non-destructive Restore, lease, per-chain writer serialization and whole-chain budget/eviction.
4. **Compaction and soak:** atomic new-Base switch, failure recovery, multi-session churn tests and write-amplification metrics.

Each patch is independently reviewable and remains inside the Runtime KV Storage Feature Package. Patch B (shutdown persistent cache) remains independent and untouched.
