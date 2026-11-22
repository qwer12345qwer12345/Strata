# Patch B — Shutdown Persistent Cache

状态：**规划占位，尚未实现**。对应的 0001-shutdown-persistent-cache.patch 不是可应用补丁。

## 目标

只解决 Engine **正常关闭**时 RAM ConversationCache 中会话的跨重启保留。它与运行期间的 SSD Spill 分离：不接管 RAM 淘汰，不修改官方 ConversationSpill::put()/take_serial()，也不依赖 Patch A。

## 需要实现

1. **关闭时采集**：在 Engine 停止接收新请求、正常关闭且 ConversationCache 尚未销毁时，读取 RAM 中已停车的 SavedConversation。第一版只处理 RAM cache entries，不扫描或搬运运行时临时 Spill 文件。
2. **选择与预算**：复用现有会话身份和基本缓存准入/预算概念，保存可复用且能放入预算的条目；不新增复杂优先级系统。
3. **稳定命名空间**：按现有 SessionFileIdentity（model/config）隔离持久化目录，避免不同模型或配置互相命中。
4. **持久化格式**：使用官方 session-file 保存完整 SavedConversation；使用 metadata sidecar 保存重建 Prefix Index 所需的轻量信息，不在启动时读取全部 KV payload。
5. **启动重建**：启动时扫描有效 metadata，重建 prefix index；只有匹配命中时才读取对应 payload。无效或不兼容记录直接忽略并回退正常 Prefill。
6. **持久记录生命周期**：Restore 不应把它当作一次性 Runtime Spill 消费并立即删除；记录保留到被新快照替换或被预算淘汰。
7. **失败行为**：写入失败、文件缺失或身份不匹配只导致缓存未命中，不影响正常 Engine 启动和推理。

## 设计边界

- 只保证**正常关闭**时保存；不承诺进程崩溃或断电时保存最后一批 RAM 会话。
- 不改变运行期间官方 Spill 的写入/读取/淘汰行为。
- 不保存 GPU 正在活动的会话状态；第一版只持久化 RAM ConversationCache 已有的 SavedConversation。
- 不搬运 Patch A 的临时文件；两个补丁可独立实现和启用。
- 继续使用官方 session-file 格式与 SessionFileIdentity；不引入数据库、通用事务日志或额外存储抽象。
- 如果关闭时没有值得保存的 RAM 条目，就不做额外写入。

## 验收标准

- 正常关闭后，下一次启动能仅凭 metadata 重建索引。
- 匹配会话时能恢复完整 SavedConversation；未命中时正常 Prefill。
- 不同 model/config namespace 互相隔离。
- Restore 后持久记录仍可用于后续重启，直到替换或淘汰。
- 磁盘满、sidecar 缺失或损坏时只丢弃对应缓存，不阻断启动。
