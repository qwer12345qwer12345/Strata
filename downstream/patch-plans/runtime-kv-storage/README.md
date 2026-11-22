# Patch A — Runtime KV Storage

状态：**Patch 1 代码已写入活动 Feature Package；编译和测试验证待执行。运行时接入尚未开始。**本计划对应独立的 Runtime KV Storage 功能包；它不会并入 Patch B，也不修改跨重启缓存。

## 已确定的设计

- **Immutable Base + 独立 Delta 链**：每条会话链由一个完整、不可原地修改的 Base 和零个或多个独立 Delta 文件组成。
- **完整恢复**：Restore 始终先读完整 Base，再严格按 generation 顺序应用 Delta，得到完整 SavedConversation；最后交给现有 Snapshot 验证和 Restore。运行时不接受半恢复状态。
- **尽可能覆盖官方 KV 模式**：Delta 将 K/V、scale、pooled/indexer 等五个 part 视为不透明字节范围；布局和 keep 边界由官方 Snapshot API 提供，不在补丁中重写格式计算。
- **保守回退**：无法证明同一会话 lineage、Dirty Range、KV layout 或父 generation 时，写新 Base；绝不猜测可复用前缀。
- **复用官方设施**：继续使用 SessionFileIdentity、session-file 原子发布/校验、ConversationSpill 预算和 BackgroundWriter、ConversationSnapshot 的 capture/validate/restore。
- **只优化运行期**：不承诺跨 Engine 重启保留；不依赖 Patch B。

详细格式、生命周期和阶段见 [DESIGN.md](DESIGN.md)。

## 当前官方路径与写放大来源

`ConversationCache::set_evict_sink()` → `ConversationSpill::put()` → `session_file_write()` → 官方 `BackgroundWriter`。

当前每次进入 SSD tier 都写完整 SavedConversation；`take_serial()` 恢复后消费并删除文件。相同会话再次被淘汰时，即使大部分 KV 与上次相同，也会重新写入完整 KV。BackgroundWriter 只把 I/O 移出请求线程，不减少写入量。

## 实现顺序

### Patch 1 — Session Delta Core（代码已落地，验证待执行）

- 在现有 SessionFile 序列化层增加独立版本化 Delta 文件格式，不改 STRSESS v1。
- Delta header 绑定 model/config identity、chain ID、parent generation、generation、payload 长度和校验值。
- Delta payload 保存当前完整的非 KV 状态，以及每层五个 KV part 的 keep 长度和变化尾部。
- 提供 Delta 写入、严格读取/校验、将 Delta 应用到 Base image 的接口。
- keep 长度只作为字节边界处理；调用方必须使用官方 Snapshot 的 part-size/layout 结果计算，禁止按 token 数自行估算。
- 纯 Host 测试覆盖 Base + 多 Delta、五个 part、追加、keep=0 全量替换、generation/identity 错误、截断和哈希错误。
- 生产补丁：[0001-session-delta-core.patch](../../features/runtime-kv-storage/patches/0001-session-delta-core.patch)。测试源码仅通过 Feature Package overlay 注入。
- 当前只增加 Delta 文件核心 API，不改变 ConversationSpill 的 put/take 行为；SSD 写入量尚未因此减少。

**验收门槛：** Base + Delta materialize 的 SavedConversation 与对应完整快照逐字段、逐字节一致。此阶段不接入运行时淘汰路径。

### Patch 2 — Snapshot provenance adapter

- 把现有 KV reuse 的 lineage/generation/unchanged-token 信息传递到下一次 park。
- 通过官方 Snapshot API 计算各层 K/V、scale、pooled 的准确 keep 字节数。
- 覆盖 FP16、INT8（含 scale/rotation）、Q4_0、K8V4 hybrid、streamed host pool、ring/draft 和 indexer pooled state。
- 对无法提供权威布局或 Dirty Range 的状态，明确退回完整 Base。

**验收门槛：** 每个官方支持模式的 capture → Base/Delta → materialize → Snapshot validate/restore 往返一致。

### Patch 3 — ConversationSpill integration

- Spill entry 从单个 session 文件扩展为 Base + Delta refs。
- Restore 改为非破坏性读取；通过 lease 固定链，避免恢复期间被淘汰或删除。
- 再次 park 时按 lineage 和 generation 追加 Delta；同一链的写入串行化。
- 预算统计 Base、全部 Delta 和写入中的临时文件；淘汰整个未 lease 链。
- 继续复用官方 BackgroundWriter，不引入第二套 Writer。

**验收门槛：** 多会话反复 park/restore 不丢 Base、不跨会话复用 KV，RAM 淘汰行为不变。

### Patch 4 — Compaction and qualification

- Delta 数量达到 8、Delta 总量达到 Base 的 50%，或单个 Delta 不小于 Base 时，生成新完整 Base。
- 新 Base 原子发布并切换索引后，才删除旧 Base/Delta；失败时旧链保持可恢复。
- 用多会话轮换 soak 测试比较 Host Bytes Written、读回字节、恢复延迟、RAM 峰值和 Base fallback 比例。

## 不做的事

- 不重写 KV kernels、attention、allocator 或 GPU residency。
- 不扫描/比较 2 GB KV 来猜增量；Dirty Range 必须来自真实更新/reuse provenance。
- 不做任意前缀随机读取、数据库、通用 WAL、复杂调度器、多进程共享或跨重启持久化。
- 不把测试 overlay 混进生产 patch。
- 不支持官方 SessionFile/Snapshot 当前本身拒绝的状态（例如当前 whole-session session-file 路径拒绝的 layer-split stage_parts）；若未来要支持，先扩展官方 Snapshot contract 并单独验收。

## 旧参考

上一个完整 KV 硬盘方案和已移除的 `l2-session` 仅作历史参考。当前实现以固定上游基线的官方 SessionFile / ConversationSpill / Snapshot 为出发点，不恢复旧的大范围改造。
