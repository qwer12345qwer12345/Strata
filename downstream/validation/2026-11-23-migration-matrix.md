# 旧优化到新功能包的迁移核对

本次对照旧 `ampere-gyro-latency` 的 `bcaf61f6` 源码、已部署的 0.1.60 选择配置，以及 0.2.19 / Engine 0.1.65 的功能包。旧补丁的全部 hunk 和实际开关均作为调查依据。功能包编译和单元测试通过之后，还需要核对旧部署的功能覆盖。

| 原能力 | 新栈的现状 | 迁移结论 |
| --- | --- | --- |
| 固定成对行 `STRATA_TQ_AMPERE_ROWS=2` | `tq-ampere` 已迁入，30 条输出校验一致 | 已保留 |
| TQ CPU AVX-VNNI | `tq-cpu` 已迁入，原生 TQ 数值测试通过 | 已保留 |
| K8V4 Q*K IMMA | `qsa-imma` 已迁入，SM86 数值测试通过 | 已保留 |
| `STRATA_KV_PREFETCH`、`STRATA_PF_HC_REGION`、`STRATA_PIN_BY_NEED` | 上游仍有相应实现 | 部署时核对开关即可 |
| `STRATA_PF_FUSED_TQ=1` | 上游仍支持显式启用，原生无 MMQ 构建会自动启用 | 新配置保留显式选择可准确复现此前配置，实际路径以构建和启动日志核对 |
| `STRATA_PF_GROUP_SCRATCH` | 上游仍保留 MMQ 专用实现 | 当前原生融合 TQ 基线取 0，符合此前 0.1.60 基线选择 |
| ChatML 工具边界保护 | 已加入独立 `chatml-tool-guard` 功能包 | 原遗漏已补，独立及完整栈各 18 项测试通过 |
| `STRATA_CONV_DISK` 文件后备快照及驻留页释放 | 已迁入完整存档包，保留新版 SSD fallback | 文件后备 CUDA 往返通过；256K 峰值仍待无头验收 |
| PR5500 V2/V3 兼容存档与增量恢复 | 单一 `legacy-session-archive` 大补丁已迁入 | 两次真实模型跨进程恢复与追加增量写入通过 |
| 正常退出后的跨重启缓存 | 每轮完成后自动保存，退出等待正在进行的请求/存档 | 活动会话、续聊及 SIGINT 等待保存均有验证；强杀前未完成轮次不保证 |
| 细粒度 CUDA 延迟测试、长输出 SSE 对照、无头开机测试与监督器 | 存档包已带入监督器与开机基准，完整对照工具仍在旧分支 | 监督器/基准接线有模拟验收；长输出正式对照保留旧入口 |
| `prior=0.5`、DFlash2 spec 6/confidence 0.8、15 workers、K8V4 与驻留预算 | 属于部署配置 | `prepare.py` 应用代码补丁，feature.json 的 environment 用于声明，不会自动把这些值写入服务配置 |
| `STRATA_UNDECLARED_TOOL_CALLS=call` | 上游支持这项配置 | 与 ChatML 角色边界保护的职责分别核对 |
| `STRATA_TQ_AMPERE_ROWS=auto` 动态形状分派及共享输入候选 | 新内核包保留固定成对行 | 旧实测中候选较慢，历史记录保留在旧分支，生产收益迁移以固定成对行为准 |

## 工具调用回归的实际复现

将旧分支的 `serve/test_chatml_tool_guard.py` 复制到独立测试目录，使用新版 `terminal` 作为 Python 导入路径，回放以下三个用例。测试使用合成模型输出和本机回环 API。

- `test_real_role_token_does_not_escape_in_agent_reply`
- `test_role_token_in_truncated_tool_does_not_publish_a_tool`
- `test_agent_token_limit_does_not_publish_partial_json`

旧实现三个用例全部通过。新版三个用例均失败，其中流式和非流式子用例分别计数，共四条失败断言。新版会把真实生成的 `<|im_start|>` 继续输出，流式结果也可能提前发布未完成的工具调用。现有结束 token 集合包含 `im_end` 和 `endoftext`，因此仍需迁入角色边界处理和不完整工具调用的发布控制。

这组回归检验此前已选择的服务行为。它没有加载真实模型，也没有测量随机工具错误的发生率。

## 当前选择与剩余验收

用户选择旧方案作为一个大补丁直接使用，已完成迁移。完整存档包取代 Live Memory、实验性 Runtime KV Storage 和旧 Shutdown 包的活动地位。工具保护保持独立。

[存档包的实际验收](../features/legacy-session-archive/validation/README.md)覆盖 Host、GPU 及真实模型跨重启。文件后备也保留旧 RAM 准入估算；256K 下的捕获峰值、写回脏页和专家驻留需要无头验收，不能由当前 6K 提示的结果推断。

## 新增且已采用的路径

独立BF16 PLE已接入实际启动选择器；`ple-bf16-compact` 提供 Direct raw row、GPU逐位展开与独立流。`ampere-tc-dispatch` 借用其他GPU测得的整数点积变体，在SM86重新按形状验证；当前只启用DFlash2分派，主模型Dense短窗口候选保留但关闭。原有prior=.5、spec6/conf=.8、成对行不变。
