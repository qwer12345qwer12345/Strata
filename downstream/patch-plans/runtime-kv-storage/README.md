# Patch A — Runtime KV Storage

状态：**规划占位，尚未实现**。对应的 0001-runtime-kv-storage.patch 不是可应用补丁。

## 目标

直接改造官方运行时 SSD Spill，而不是在官方机制外再包一层临时缓存。针对低 RAM、多会话频繁切换的场景，减少同一会话反复淘汰时完整写入 SavedConversation/KV 的写放大。

## 当前官方路径与问题

当前运行时大致经过：

ConversationCache::set_evict_sink() → ConversationSpill::put() → session_file_write() → BackgroundWriter。

put() 保存完整 SavedConversation；take_serial() 读取后消费并删除对应文件。会话恢复到 RAM、再次被淘汰时，可能重新写入完整 KV。BackgroundWriter 将写入移出请求线程，但不会减少写入字节数。

主要参考：

- 上游 engine/include/strata/core/conversation_spill.hpp
- 上游 engine/include/strata/core/conversation_cache.hpp
- 上游 engine/src/program/generate.cpp

## 需要实现

1. **Base 复用**：同一会话首次 Spill 写完整 Base；恢复时采用非破坏性读取，不因一次 Restore 就删除可复用 Base。
2. **会话身份与版本**：基于现有 SessionFileIdentity 和会话状态，确定可稳定识别同一会话的 key/generation，避免把不同会话或不同 KV 布局误认为同一个 Base。
3. **增量写入**：利用现有 token/KV 状态识别追加式 Dirty Range；只把新增 KV 写入 Delta。没有可靠 Dirty Range、发生截断/覆盖、身份或布局变化时，退回写新 Base。
4. **Delta Restore**：按顺序将 Base 与 Delta 恢复为当前 SavedConversation；校验前置 generation，避免重复或错序应用。
5. **简单 Compaction**：Delta 数量或累计大小达到固定阈值时合并为新 Base。阈值先用实际写入/恢复数据确定，不引入复杂调度器。
6. **成本估计**：让现有 SpillCost 使用预计的 Base/Delta 实际写入量，而不是总把完整快照大小当成写入量；保留官方成本模型和准入逻辑。
7. **观测与测试**：记录 Base 写入字节、Delta 写入字节、恢复读取字节、Base 复用次数和重新生成 Base 次数；覆盖无变化往返、追加写入、截断/覆盖回退、连续多会话切换。

## 设计边界

- 这是**运行期间**的临时 SSD 后备；不负责 Engine 重启后的持久化。
- 继续复用官方 BackgroundWriter、SessionFileIdentity、Prefix Matching 和 Restore Limits。
- 不另建数据库、事务日志或通用存储框架。
- 第一版不做任意前缀随机读取、复杂 LRU/TTL、多进程共享和自定义 Writer。
- Dirty Tracking 必须来自真实 KV 更新边界；如果只能通过扫描/复制整份 KV 才能计算增量，就不值得作为默认路径，应先保留完整 Base fallback。

## 验收标准

- 同一会话恢复后未变化，再次 Spill 不产生第二份完整 KV 写入。
- 追加式更新只写入对应 Delta；不兼容修改安全地生成新 Base。
- Base + Delta 恢复结果与当前完整 session-file 恢复一致。
- RAM 淘汰后实际释放内存；SSD 预算和失败回退行为仍正确。
- 用真实多会话轮换负载比较总 Host Bytes Written，而不只比较写入次数。
