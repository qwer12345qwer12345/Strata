# L2 patch plans

这里记录 L2 的下一轮实现拆分。目录中的 .patch 文件目前只是占位说明，不是可应用的 unified diff；两个计划均未实现，也不属于 downstream/stack.json 的活动 Feature Stack。

## 两个独立补丁

- [Patch A — Runtime KV Storage](runtime-kv-storage/README.md)：直接改造官方运行时 ConversationSpill，重点减少多会话轮换时反复写入完整 KV。
- [Patch B — Shutdown Persistent Cache](shutdown-persistent-cache/README.md)：在 Engine 正常关闭时保存 RAM 中值得保留的会话，并在下次启动时重建索引。

两者应能分别开发、分别启用。Patch A 不负责跨重启持久化；Patch B 不改造运行时 Spill，也不依赖 Patch A。

## 进入活动功能栈的条件

实现开始后，分别建立正常的 Feature Package，补齐真实生产 patch、测试、验证记录、patch SHA256 和 tree 指纹。只有真实 unified diff 通过 downstream/scripts/verify.py 后，才加入 stack.json。不要把当前占位 .patch 文件交给 git apply。
