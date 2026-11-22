# 分支审计与保留策略

本审计基于当前仓库分支列表、各分支相对 `main` 的 commit compare 和关键文件内容。审计对象是功能分支和历史备份分支；`main` 与 `downstream/main` 属于长期分支。

## 必须保留

### `feature/live-memory-cuda`

保留。该分支建立在旧的 `0.2.15 / Engine 0.1.60` 基线上，包含早期 Live Memory 研究计划、容量策略 helper 和 CPU-only policy test。它记录了旧实现的参考范围、RTX 3080/32 GB 的验收约束、VMM/RAM block 设计、MEMORY/ACK/CAPACITY 协议及 Vision reserve 协调要求。当前 `live-memory` 包只迁移了后续实现的初始 VMM commit 和策略 helper，不等于完整 runtime，因此旧分支仍有研究和兼容性参考价值。

### `ampere-gyro-latency`

保留。该分支包含独立的 Ampere benchmark 工具和测量数据（kernel latency、chat latency、profile、历史结果），并包含旧 L2 archive/V2/V3 相关实现、控制器、测试和设计文档。这些内容不是当前五个功能包的完整迁移范围，尤其旧 L2 增量归档和基准数据不能因 PR #2 的 Persistent Conversation Spill 包而视为已迁移。

## 已迁移，可列为清理候选

以下分支的当前生产改动已迁移到 Feature Packages。清理前，本次已把尚有参考价值的设计说明归档到 `downstream/main`：

| 分支 | 已迁移内容 | 归档位置 |
|---|---|---|
| `feature/ampere-gyro-reimplementation` | TQ CPU AVX-VNNI、Ampere TQ 配对 Kernel、K8V4 Q*K IMMA、相关构建接线和 QSA parity test | `features/tq-cpu`、`features/tq-ampere`、`features/qsa-imma`；`design/AMPERE_GYRO_REIMPLEMENTATION.md` |
| `feature/l2-session-v3-reimplementation` | 当前上游 API 上的 Persistent Conversation Spill 实现和回归测试 | `features/l2-session`；`features/l2-session/design/L2_SESSION_V3_PORT.md` |
| `feature/live-memory-reimplementation` | 初始 VMM commit、策略 helper、Settings 接线和 CPU/CUDA/Settings 测试 | `features/live-memory`；`features/live-memory/design/PORT_PLAN.md` |

特别说明：`feature/l2-session-v3-reimplementation` 的当前实现是使用新上游 session-file API 的持久化 spill 基础，并没有实现旧 L2 分支的 immutable base/delta V3 archive、SAVEINC/RESTOREPP 或增量 K/V 写入。旧的 L2 V2/V3 代码仍在必须保留的 `ampere-gyro-latency` 参考分支中。

## 冗余备份候选

### `backup/pre-upstream-sync-main`

建议清理。该分支相对当前 `main` 的 compare 结果为 `ahead_by=0`、`behind_by=1229`：没有比当前 main 多出的独立提交，是上游同步前的旧基线指针。它的基线 commit `c77572806996349b49995e4f98d41b5294d36a21` 仍是保留分支 `feature/live-memory-cuda` 的祖先，因此删除这个备份 ref 不会丢失该旧基线对象或保留分支中的工作。

## 当前状态与安全边界

- `main` 和 `downstream/main` 保留。
- `feature/live-memory-cuda` 和 `ampere-gyro-latency` 保留。
- 上述三个 reimplementation 分支和一个 backup 分支是可清理候选，但本次执行环境中的 GitHub 工具没有提供删除 branch/ref 的操作，因此没有伪称它们已经删除。
- 删除候选前，先确认 `downstream/main` 上的设计归档和功能包均已提交；随后可在本地执行：

```bash
git push origin --delete \
  backup/pre-upstream-sync-main \
  feature/ampere-gyro-reimplementation \
  feature/l2-session-v3-reimplementation \
  feature/live-memory-reimplementation
```

不要删除本文件“必须保留”一节列出的两个分支。
