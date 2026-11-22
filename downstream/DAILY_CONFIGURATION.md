# 日常采用理论优先组合，验证改为手动

用户选择先采用当前有理论依据的组合，再手动验证。这里的“理论最优”是待检验的选型，不是已经证明的全局最优；组合状态明确标为 **未完成无头性能与质量测试（NOT TESTED）**。已有独立GPU数据继续有效，但不能代替完整模型结果。用户允许为有效性能收益接受数值/输出差异，不把逐位复现单独作为关闭优化的理由；影响大小未知时仍写未测。

## 本次统一的差异

| 项目 | 此前日常 | 现在日常 | 依据与未测部分 | 手动组 |
|---|---|---|---|---|
| DFlash2形状表 | tuning=1 | tuning=2 | GPU扫描与固定输入真实drafter检查通过；完整组合未测 | shape |
| ≤8列QKV | GEMM_TC=2 | GEMM_TC=0，新DP4A表 | 新表比旧QKV TC快；整段吞吐/接受率未测 | qkv-tc |
| spec | 6 | 4 | 观察到长神经提案较少；减少head/selector工作；主体仍算8行。4与3/6的取舍未测 | spec |
| lookup上限 | 8 | 16，候选2..8/10/12/16 | 成本策略可选较窄窗口；跨宽行桥接检查通过，API实测触发10行；完整16行收益及驻留峰值未测 | lookup |
| Dense短窗TC | AMPERE_SHAPES=0 | =1 | 只开启已测形状：Q4_K/Q5_K的T6..8、Q6_K的T8，K2560/N10240；有舍入/输出变化，组合未测 | dense |
| lookup草稿延后 | LOOKUP_SKIP=0 | =8 | 长匹配时可避免lookup已获选后无用的神经草稿计算；验证仍照常执行；本项仅逻辑分析、未做性能实测 | lookup-skip |
| 专家搬运不等待 | ADAPT_NOWAIT默认关闭 | =1，ADAPT_LAG=2 | 上游保留的opt-in；等待更少但CPU/GPU执行时序与舍入不固定；本机未测 | adapt-tail |
| CPU尾部专家跳过 | ROUTE_TAIL_SKIP=0 | =7 | 只跳过GPU未命中且整窗口均排在尾部的专家；近似计算，本机收益/质量未测 | adapt-tail / tail-spec |

QKV需要随新表重新选型：已发布的修正后数据中，2/4/8列的新DP4A分别约12.26/13.28/16.29µs，旧QKV TC约15.82/16.04/18.43µs。新表采用后继续固定QKV TC已不是当前数据支持的优先选项。这里关闭的是DFlash2的≤8列GEMM TC开关；批量context、主模型Dense等其他TC路径不因此全部关闭。

新旧表固定输入检查相同，不表示完整模型每次输出都相同。之前桌面模型检查已出现输出分叉，原因未完全隔离；Dense TC本身也有数值差异。按用户选择先用于日常，保持未测标注，后续同时比较输出、接受率、延迟和内存。

## 已经统一并保留的配置

prior固定0.5、DFlash2/confidence0.8、专家成对行、15 workers、Gyro-S、BF16 PLE + compact + 独立复制流、fused TQ prompt、group scratch关闭、QSA K8V4 IMMA、CPU AVX-VNNI路径均保留。既有PR5500 V2/V3增量磁盘存档、同步写入预算0、32GiB磁盘缓存预算和工具边界保护也保留。

上下文262144、KV resident20480、KV k8v4、专家auto/完整resident、显存余量448MiB保持现有设置。没有因为磁盘充足就把活跃KV或专家内存预算放大。DFlash2继续优先于先前较慢的MTP；没有重启Live Memory，没有为触发TC补无用行，没有全局把Dense的9行门槛降到更小。

`STRATA_DF2_CTX_BATCH` 保持上游默认：关闭它虽可让旧ring测试逐位一致，却没有性能收益依据，长prompt会回到更多窄批调用，故不当成性能优化打开。新宽窗口桥接自身按≤8行录入。单行/成对行有未决细微延迟对比，日常按用户选定的成对行，另列rows组。

## 启动与手动命令

正常 `sudo systemctl start strata` 只等模型加载和内部服务ready，随后开放服务，不发送基准prompt、不执行旧队列。旧的启动基准和一次性请求已取消。已完成请求的存档与退出drain关系不变。

本机手动入口：

```sh
strata-headless-compare --plan
sudo ~/.local/bin/strata-headless-compare
```

第一条只看计划，不停止服务。第二条须在无头环境执行；先正常停止Strata，测试时继承无限memlock并以模型用户运行，结束（含试验失败）后恢复日常服务。桌面仍active时直接退出，不切换桌面。需要sudo是为了完整专家锁页的memlock和systemd停止/恢复服务，不用于模型进程提权。

可只选一组，例如 `sudo ~/.local/bin/strata-headless-compare --group spec`；默认全部114组、预算14400秒。可用 `--budget-s` 调整，运行器上限14400秒。源码入口为 `terminal/l2/manual_compare.py`，命令本身没有开机队列副作用。

## 原有70组：关键交互矩阵

| 分组 | 参数矩阵 | 组数（正序+反序） |
|---|---|---|
| combined | 旧组合 / 理论日常组合 | 4 |
| spec-lookup | spec=6/4/3 × lookup=8/12/16 | 18 |
| shape-qkv | tuning=1/2 × QKV TC=2/0 | 8 |
| dense-lookup | Dense=0/1 × lookup=8/16 | 8 |
| skip-lookup | 草稿延后=0/8 × lookup=8/16 | 8 |
| rows | 成对行 / 单行 | 4 |
| adapt-tail | NOWAIT=0/1 × tail skip=0/7，lag固定2 | 8 |
| tail-spec | tail skip=0/7 × spec=3/4/6 | 12 |

关键交互采用完整二维组合，其余参数以理论日常组合固定。相比只做逐项消融，这能直接发现“旧表下TC赢，新表下DP4A赢”这类交叉结果。每个组合测两遍，顺序互为反向；combined也单独比较整个旧/新组合。它仍不穷举所有三维以上组合，不宣称能数学证明全局最优。`--group spec` 等旧名称映射到对应交互组，推荐使用表中组名。

每组相同profile副本、64-token热身、3072-token长代码和1536-token长上下文输出。基准会话不入日常存档，profile不保存回原文件。报告记录首内容延迟、SSE内容块间隔、窗口按宽度/来源的p50/p95/p99、实际宽度、神经与lookup分开的接受数、文本散列及内存/swap，另记录实测的adapt wait耗时与tail skip累计计数。必须确认请求中零专家文件blob读取，才计入有效性能组。

尚未完成整个矩阵，也没有用“配置检查通过”替代性能测试。日常参数不会被测试程序自动选优或修改。


HC Q8并非遗漏的关闭项：当前Gyro-S GGUF包含Q8_0 HC投影，上游自动启用，已有真实模型日志记录0.62GiB Q8_0 HC。本次显式固定 `STRATA_HC_Q8=1` 保持这一既有行为。Tail skip不混入额外的tail renorm实验，保持 `STRATA_ROUTE_TAIL_RENORM=0` 的原始近似方式。

## 无头按需识图

已编译部署 `vision-expert-lease`：GPU编码前借专家显存，编码后退出reader并恢复原专家，再开始语言生成。同图磁盘缓存命中不借显存。配置 `vision.gpu=true`、`expert_lease=true`、`max_tokens=1024`、`threads=8`，`vram_elastic=true`、`STRATA_VISION_EXPERT_LEASE=1`、`--vram-segment-mib=128`。不为视觉静态保留2GiB专家容量；128MiB分段相对原分段的吞吐影响仍未测。

下一次无头正常启动生效，无需桌面快捷方式。真实桌面受限预算API已通过，完整驻留和恢复后零文件层读取尚待以下显式检查；它与114组参数矩阵分开运行：

```sh
strata-headless-compare --vision-check --plan
sudo ~/.local/bin/strata-headless-compare --vision-check
```

该检查保留日常完整驻留预算，执行冷图、缓存图和恢复后的3072-token文本。桌面active时退出；无开机自动测试。实现、测试边界见 [功能说明](features/vision-expert-lease/README.md)。

2026-11-23接线复查：此前旧profile缺少`user_args`，官方drafting层会将选择器写入的spec4改回6。已修正为显式用户参数，最终命令干跑确认spec4/confidence0.8/lookup16；没有重新跑完整组合。详见 [数值与性能审计](NUMERICAL_PERFORMANCE_AUDIT.md)。

## 新增44组：数值与执行策略（未测）

用户明确选择日常直接启用 `STRATA_EMB_REUSE_ACCOUNT=1`、`STRATA_GDN_TC_NP=2`，并显式固定 `STRATA_GDN_TC=1`。已写入实际无头启动选择器，下一次启动生效；无须编译。NP2简化直接输出侧乘法，递归状态相关乘法保留补偿；输出变化仍会影响后续层，不保证一般质量损失小。prior保持0.5。

| 组名 | 对比 | 组数（正反序） |
|---|---|---|
| emb-gdn | Embedding实际计费0/1 × GDN NP3/2 | 8 |
| gdn-precision | NP2/1；NP1为更强近似 | 4 |
| hc-upmix | HC上投影+mix融合0/1，确认权重视图是否命中 | 4 |
| gdn-execution | tiles1/2 × fused norm0/1；70SM下两tile可能受第二wave拖慢，仍作为实测候选 | 8 |
| prefill-cap | auto:8192/16384/32768；这是上限，记录实际可分配chunk | 6 |
| cpu-prefill | 当前auto范围、NT32/MAX3072、NT32/MAX6144、NT8/MAX6144 | 8 |
| adapt-mode | NOWAIT1/lag2、resident overlap1/NOWAIT0/lag1、等待模式NOWAIT0/lag1 | 6 |

每组以最新日常为固定背景；combined的旧组合同时保留Embedding0/NP3，新组合采用Embedding1/NP2。CPU auto组删除NT/MAX显式覆盖，以保留官方当前规则。resident overlap使用官方 `STRATA_ADAPT_OVERLAP_RESIDENT=1`，与NOWAIT互斥；不是叠加。此前审计把overlap仅描述为替代关系，此处补全已有resident opt-in。

```sh
strata-headless-compare --group numeric --plan
sudo ~/.local/bin/strata-headless-compare --group numeric
```

`numeric`选择新增44组；默认无group为全部114组。也可选 `--group emb-gdn`、`--group hc-upmix` 等单组。仍为显式无头命令，普通启动无基准、无队列，测试结束恢复日常服务。默认/最大预算4小时，未完成应报告incomplete，不假称114组全部通过。

每组保留64-token热身、3072-token长输出、600条记录的长prompt后1536-token输出，至少达到目标长度75%才有效；记录实际prompt token数，不能仅凭cap32768宣称测到32K单块。CPU share组另加60条记录的中等prompt及1536-token输出，以覆盖扩展的分担范围，不能只测超出范围的大块。新增记录各arm的完整请求参数、精度与调度环境；启用参数不等于已观察到内核命中。显式打开prefill分项计时，所有arm一致。一般任务质量、工具可靠性仍需单独判断，输出hash不作为淘汰条件。

需要修改推理代码的候选见[下一轮计划](NEXT_OPTIMIZATION_PLAN.md)。本次不实现这些新内核。
