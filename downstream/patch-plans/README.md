# L2 patch plans

这里记录 L2 的下一轮实现拆分。Patch A 与 Patch B 都已建立正常的 Feature Package 并进入活动功能栈（Patch A 的运行时接入与 Patch B 的真实模型 serve 验收仍待完成）；patch-plans 目录中的 .patch 仅作占位说明，真实补丁位于对应 Feature Package。

## 两个独立补丁

- [Patch A — Runtime KV Storage](runtime-kv-storage/README.md)：从官方 ConversationSpill / SessionFile 出发，以 Immutable Base + 独立 Delta 链减少多会话轮换时反复写入完整 KV。当前已落地 Session Delta Core；Snapshot provenance、Spill 生命周期接入和 compaction 仍待完成。真实补丁位于 [features/runtime-kv-storage](../features/runtime-kv-storage/)。
- [Patch B — Shutdown Persistent Cache](shutdown-persistent-cache/README.md)：在 Engine 正常关闭时保存 RAM 中的会话，并在下次启动时仅凭 metadata 重建索引。计划已实现为 Feature Package，真实补丁位于 [features/shutdown-persistent-cache](../features/shutdown-persistent-cache/)。

两者分别开发、分别启用。Patch A 不负责跨重启持久化；Patch B 不改造运行时 Spill，也不依赖 Patch A。

## 活动栈与验证状态

Patch A 的 Session Delta Core 与 Patch B 都已加入 stack.json，独立/完整 stack tree 指纹和 patch SHA256 已记录并通过 downstream/scripts/verify.py；Patch B 的纯 Host 测试已在本地编译运行通过，Patch A 的编译和 CTest 仍待执行。后续 Snapshot provenance、ConversationSpill chain 和 compaction 仍按独立 patch 提交，并在加入活动栈前完成 patch 应用验证。计划目录中的占位 .patch 不参与 stack.json。
