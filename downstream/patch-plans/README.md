# L2 patch plans

这里记录 L2 的下一轮实现拆分。Patch A 已建立正常的 Feature Package 并进入活动功能栈；Patch B 仍是计划阶段，目录中的 .patch 仅作占位说明，不是可应用的 unified diff。

## 两个独立补丁

- [Patch A — Runtime KV Storage](runtime-kv-storage/README.md)：从官方 ConversationSpill / SessionFile 出发，以 Immutable Base + 独立 Delta 链减少多会话轮换时反复写入完整 KV。当前已落地 Session Delta Core；Snapshot provenance、Spill 生命周期接入和 compaction 仍待完成。真实补丁位于 [features/runtime-kv-storage](../features/runtime-kv-storage/)。
- [Patch B — Shutdown Persistent Cache](shutdown-persistent-cache/README.md)：在 Engine 正常关闭时保存 RAM 中值得保留的会话，并在下次启动时重建索引；仍为占位计划。

两者分别开发、分别启用。Patch A 不负责跨重启持久化；Patch B 不改造运行时 Spill，也不依赖 Patch A。

## 进入活动功能栈的条件

计划目录中的占位 .patch 不参与 stack.json。实现进入活动栈时，必须建立正常 Feature Package，补齐真实生产 patch、测试、验证记录、patch SHA256 和独立/完整 stack tree 指纹。只有真实 unified diff 通过 downstream/scripts/verify.py 后，才加入 stack.json。
