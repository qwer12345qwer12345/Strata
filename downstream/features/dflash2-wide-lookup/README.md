# DFlash2 宽 lookup 与无头对比

在 c6f6427 / engine 0.1.65 上提供可关闭的单卡实验。DFlash2 神经草稿仍是 anchor + 7 masks、固定 8 行；扩展的是主模型验证的 prompt lookup 窗口，不是增加神经草稿长度。

## 接入方式

`STRATA_DF2_WIDE_CONTEXT=1` 与 `--lookup-max-t 12` 或 `16` 一起使用。显式指定 `--lookup-sizes` 控制候选宽度。未启用时，1..8 行仍调用原有 `draft()`；默认配置没有开启宽窗口。

`draft_extended()` 把长窗口前面的已提交行按最多8行送入现有 `context()`，把最后一行在 drafter 自己的流上复制到已有 landing buffer，再重放独立的一行上下文图。草稿块、RoPE位置、ring标签、日志与 selector 逻辑保持原有关系。同步返回后恢复原来的图槽和 taps 指针；修改 spec 时同时释放这张额外图。

不增加模型权重副本或专家驻留要求，也不为短 lookup 新建大批量 prompt context arena。额外 CUDA graph 有驱动元数据开销。不能把“没有新增显式张量 arena”解读成绝对零字节显存开销。

本次接入用于单卡 CUDA serve / Chat Completions，CLI 同步草稿调用也接入了桥接。没有扩展多卡 taps landing、batch slot 的异步 launch/finish 或 DFlash2 主体行数；没有验证其他后端。

## 无头队列

配合 legacy-session-archive 的 supervisor hook，在正常 backend 与对外 listener 启动前读取 startup benchmark 输出目录内的 `spec-shape-lookup.request.json`。请求包含已选引擎的 SHA256 和 `budget_s`；一次领取后移入私有结果目录，失败或中断不会在每次重启重复测试。测试失败会记录原因并继续日常基线；用户中断则退出。

16组按回文顺序执行，每组全新引擎、固定 expert profile 副本，禁用 profile 保存与会话缓存：

| 对比 | 顺序 | 固定项 |
|---|---|---|
| spec | 6/4/3/3/4/6 | 旧形状表，lookup上限8，sizes=2..8 |
| SM86形状 | 旧/新/新/旧 | spec6、lookup上限8 |
| lookup上限 | 8/12/16/16/12/8 | spec6、新形状表 |

prior=.5、confidence=.8、成对行、BF16 PLE、15 workers、正式专家预算与上下文设置保持不变。每组64-token热身，3072-token代码输出与1536-token长上下文输出。长输出提前结束到不足75%会标无效；报告不自动选择获胜配置。

报告包括首个内容到达时间、引擎吞吐、HTTP内容块间隔p50/p95/p99、完整窗口耗时按来源/宽度的p50/p95/p99、实际宽度分布、分别的神经/lookup接受数及输出散列。HTTP内容块不是单个token；窗口时间包含这一轮的验证与草稿，不是纯GPU kernel时间。原始 trace 含令牌，留在本机私有目录，发布时只导出聚合计数。

只在 SDDM inactive、无其他引擎或编译任务时运行。可用内存低于1GiB、新增swap超过1GiB或达到预算会结束试验。要求完整RAM专家补集page-lock，且每个请求确认零专家文件blob读取，才计入有效性能组。每组关闭所有引擎子进程后才启动下一组。

## 验证及边界

- `dflash2_wide_test PACK`：9/12/16宽行与1/4/8窄行交替，跨2064行ring边界，修改spec后重建图，检查草稿、概率、ring与标签逐位一致。参考是显式分块context加最后一行原有draft。拒绝未启用/非法宽度。
- 真实Q4 pack + 固定合成head/taps，旧/新/新/旧四进程的8行最终hidden state、草稿、概率散列全部一致，见 [真实pack记录](validation/real-pack-parity.json)。这不是全模型质量基准。
- 真实Chat Completions完成四组、每组1024-token代码与384-token说明输出。12上限实际触发10行窗口；16上限本次没有使用>8行，不能称为API的T16覆盖。C++桥接测试单独覆盖16行。见 [模型记录](validation/model-check.json)。
- 桌面模型检查有专家文件读取，不能据此判定内核或无头吞吐优劣。shape8与wide12输出发生分叉，来源尚未隔离；wide16与参考的长输出散列相同。不能把分叉直接叫作质量降低，也不能声称已完成质量等价验收。
- ring恢复附加检查发现：开启上游批量context时，关闭新形状表也不逐位一致；`STRATA_DF2_CTX_BATCH=0` 则通过。批量context的TC与窄行重建不是同一累加路径。桥接只使用窄行context，未更改生产prompt批处理默认，也未声称解决上游所有ring精度问题。原始对照保存在 validation。

本机保留旧形状表、spec6与默认lookup8作为日常基线；新表和宽窗口等待完整无头16组结果。引擎二进制SHA256：`85a8b3084262d90b671ba5442bab6a0885bfb7faf34cc76e775b0f12ab4f7569`。编译和GPU测试依次执行，未切换桌面或网络。
