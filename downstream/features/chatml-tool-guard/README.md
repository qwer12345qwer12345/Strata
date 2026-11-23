# ChatML tool boundary guard

This package ports the deployed Chat Completions guard to Strata 0.2.19 / Engine 0.1.65, pinned at `c6f6427d9de9016fd92d13e29740f65625261df7`.

Enable it in the server profile using `"chatml_tool_guard": true`. The default is false. The value must be a JSON boolean.

For a request with tools, a generated ChatML role control token ends the assistant turn before that token enters the text parser. The parser first finalizes any complete tool call held for lookahead. A call interrupted while incomplete ends as `length`. Chat Completions streams publish each tool call after its complete arguments have been parsed. This delays the tool's arguments until completion. Completed undeclared calls still follow the existing `undeclared_tool_calls` policy.

A literal marker inside file contents is preserved when it is ordinary text tokens. Requests with no tools keep their existing text behavior. The role-token boundary applies to ChatML and excludes the DeepSeek family. The default tool streaming path retains incremental argument chunks.

The production patch changes `terminal/serve/server.py`. Tests live in the package overlay. This package adds no engine code or model assets.

```sh
python downstream/scripts/prepare.py --feature chatml-tool-guard --target ../Strata-tool-guard
python downstream/scripts/test.py --feature chatml-tool-guard --source ../Strata-tool-guard
```

Use a Python environment with the Terminal's runtime dependencies. The test runner runs the Python suite directly for this package. See [validation](validation/README.md) for results.


## 重复生成的有限恢复（2026-11-23）

配置`repeat_failure_retry=true`（默认false，本机已启用）。保留`repeat_stop_tokens=256`，连续重复只终止当前生成，不关闭服务。对尚未发布正文或完整工具调用的草稿，从原始prompt重试一次；不拼入失败草稿，不自动执行残缺JSON。工具开始/参数事件在完整调用前暂存，成功后只发布一次；暂存期间每10秒发SSE心跳，避免长工具调用被客户端空闲超时打断。

重试与原尝试共用`max_tokens`预算。仅本次重试将temperature至少提高到1.0、presence penalty至少提高到1.5，保留更高的有效设置、原top_p/top_k；显式seed递增一次，客户端参数对象不修改。已经发布正文或完整调用时不自动重放。图片请求的编码文件已由首次生成接管，暂不自动重试图片，避免复用被删除的embedding。用户取消、引擎错误和非重复的普通输出上限保留原语义。

第二次重复，或重试后仍无可用正文/完整调用，返回带`[Strata]`前缀的普通提示，`finish_reason=stop`，并带`strata_generation_failure`元数据。已发布的完整调用不撤回、不重放。客户端可继续下一轮；不伪造工具结果或自动补全破损参数。成功重试带`strata_generation_recovery`，不带失败标记。

L2控制器识别失败标记，不执行saveinc、不覆盖原可用存档。成功重试照常存档；失败后的下次输入按原始历史重建/复用匹配前缀，不要求重启模型。性能测试拒绝带恢复/失败标记的回复，避免把失败草稿计成有效长输出。

已通过模拟流式/非流式、再次失败后的后续请求、取消、预算、图片文件归属、不得重放已发布调用以及保留旧存档的回归。Pi客户端真实模型联调未运行。验收见[记录](../../validation/2026-11-23-repeat-recovery.json)。最终93项协议、存档和运行器回归通过。`--all`的内核阶段会先跑这93项CPU回归，再跑GPU检查和模型矩阵；普通启动不测试。

独立准备测试目录时，需把本包`tests/overlay/terminal/serve/test_repeat_failure_retry.py`安装到准备目录的`terminal/serve/`，与预编译的GPU测试目标一起保留；本机运行目录已经安装。
