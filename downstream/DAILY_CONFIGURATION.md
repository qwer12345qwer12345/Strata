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
strata-headless-compare --all --plan
sudo ~/.local/bin/strata-headless-compare --all
```

第一条只看计划，不停止服务。第二条须在无头环境执行；先正常停止Strata，测试时继承无限memlock并以模型用户运行，结束（含试验失败）后恢复日常服务。桌面仍active时直接退出，不切换桌面。需要sudo是为了完整专家锁页的memlock和systemd停止/恢复服务，不用于模型进程提权。

可只选一组，例如 `sudo ~/.local/bin/strata-headless-compare --group spec`；单独矩阵默认180组、预算14400秒，可用 `--budget-s` 调整，已取消4小时硬上限。推荐 `--all` 一次运行内核、完整矩阵和识图，默认后台执行、总预算72小时。源码入口为 `terminal/l2/manual_compare.py`，命令本身没有开机队列副作用。

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

下一次无头正常启动生效，无需桌面快捷方式。真实桌面受限预算API已通过，完整驻留和恢复后零文件层读取尚待以下显式检查；它与180组参数矩阵分开运行：

```sh
strata-headless-compare --vision-check --plan
sudo ~/.local/bin/strata-headless-compare --vision-check
```

该检查保留日常完整驻留预算，执行冷图、缓存图和恢复后的3072-token文本。桌面active时退出；无开机自动测试。实现、测试边界见 [功能说明](features/vision-expert-lease/README.md)。

2026-11-23接线复查：此前旧profile缺少`user_args`，官方drafting层会将选择器写入的spec4改回6。已修正为显式用户参数，最终命令干跑确认spec4/confidence0.8/lookup16；没有重新跑完整组合。详见 [数值与性能审计](NUMERICAL_PERFORMANCE_AUDIT.md)。

## 新增46组：数值与执行策略（未测）

用户明确选择日常直接启用 `STRATA_EMB_REUSE_ACCOUNT=1`、`STRATA_GDN_TC_NP=2`，并显式固定 `STRATA_GDN_TC=1`。已写入实际无头启动选择器，下一次启动生效；无须编译。NP2简化直接输出侧乘法，递归状态相关乘法保留补偿；输出变化仍会影响后续层，不保证一般质量损失小。prior保持0.5。

| 组名 | 对比 | 组数（正反序） |
|---|---|---|
| emb-gdn | Embedding实际计费0/1 × GDN NP3/2 | 8 |
| gdn-precision | NP2/1；NP1为更强近似 | 4 |
| hc-upmix | HC上投影+mix融合0/1，确认权重视图是否命中 | 4 |
| gdn-execution | tiles1/2 × fused norm0/1；70SM下两tile可能受第二wave拖慢，仍作为实测候选 | 8 |
| prefill-cap | auto:8192/16384/32768；这是上限，记录实际可分配chunk | 6 |
| cpu-prefill | 当前auto范围、NT32/MAX3072、NT32/MAX6144、NT8/MAX6144 | 8 |
| adapt-mode | NOWAIT/lag2下fetch2或0；等待/lag1/fetch0；overlap/lag1/fetch0 | 8 |

每组以最新日常为固定背景；combined的旧组合同时保留Embedding0/NP3，新组合采用Embedding1/NP2。CPU auto组删除NT/MAX显式覆盖，以保留官方当前规则。resident overlap使用官方 `STRATA_ADAPT_OVERLAP_RESIDENT=1`，与NOWAIT互斥；不是叠加。此前审计把overlap仅描述为替代关系，此处补全已有resident opt-in。

```sh
strata-headless-compare --group numeric --plan
sudo ~/.local/bin/strata-headless-compare --group numeric
```

`numeric`选择新增46组；默认无group为全部180组。也可选 `--group emb-gdn`、`--group hc-upmix` 等单组。仍为显式无头命令，普通启动无基准、无队列，测试结束恢复日常服务。单独矩阵默认预算4小时，可显式增加；`--all`默认72小时。未完成应报告incomplete，不假称180组全部通过。

每组保留64-token热身、3072-token长输出、600条记录的长prompt后1536-token输出，至少达到目标长度75%才有效；记录实际prompt token数，不能仅凭cap32768宣称测到32K单块。CPU share组另加60条记录的中等prompt及1536-token输出，以覆盖扩展的分担范围，不能只测超出范围的大块。新增记录各arm的完整请求参数、精度与调度环境；启用参数不等于已观察到内核命中。显式打开prefill分项计时，所有arm一致。一般任务质量、工具可靠性仍需单独判断，输出hash不作为淘汰条件。

需要修改推理代码的候选见[下一轮计划](NEXT_OPTIMIZATION_PLAN.md)。这些候选已有PR的验收和当前状态见下一节及更新后的计划。

## 0.2.20 / engine 0.1.66 与PR生产选择（未完成模型验证）

固定上游 `6a6db7ff58c283cf28d2b913404b5fa31a767eaa`。旧9包及V2/V3存档、工具边界、BF16 PLE、识图租约保留，新增4包。日常按用户要求采用理论优先选择，不能据此宣称比旧版更快。

| 选择 | 生产值 | 原因与限制 |
|---|---|---|
| K8V4解码TC | `STRATA_ATTN_DECODE_TC=1` | 3部分INT8 query score，V仍Q4；目标卡收益待测 |
| TQ F16预填充 | `STRATA_PF_FUSED_TQ_F16=1`、`STRATA_PF_BATCH=256` | TQ独立路径；精度、带宽、暂存及任务质量待验收 |
| resident抓取入GPU | `STRATA_ADAPT_FETCH=2` | 已有fetch代替重复tier拷贝；RAM/GPU归属批量交换；与NOWAIT和vision组合待全模型验证 |
| Q8 HC持久内核 | `STRATA_HC_PERSIST=1` | 每设备驻留探测，不能驻留则回退；新QFUSE能力保留供测 |
| QFUSE | `STRATA_QFUSE=0` | 新上游取消SM86默认；此前覆盖更合适的up路径，在部分驻留下有回退数据 |
| HC upmix / CPU share | `STRATA_HC_UPMIX=1`，NT32/MAX3072 | 当前上游候选；覆盖Gyro的实际分派仍需实测 |
| CPU专家L3预取 | `STRATA_L3_PF=1` | 重叠GPU attention与CPU预取；约120MiB BF16 router副本成本，观察总RAM |
| profile固定 / 自适应 | pin0.25；every2/swaps192/decay0.92 | 当前约22.5%专家在GPU，取新上游15–65%覆盖策略；非套用旧注释的<15%小缓存策略 |
| Dense全局门槛 | minT9 | #10只有测量准备，没有新的生产分派赢家；保留旧已测局部形状表 |

保留prior0.5、spec4/confidence0.8、lookup16、成对行、15workers、262144上下文、20480 KV resident、448MiB显存余量；不因升级把活跃内存预算扩大。新版本使用独立存档目录，旧存档保留；跨二进制恢复兼容性尚未验证。旧0.1.65二进制和冻结配置用于对照。

新增 `updates` 共64次正反序对照：engine-reference4、update-bundle4、decode-tc-spec8、tq-prefill8、hc-qfuse8、fetch-pin8、fetch-mode6、upstream-adapt4、qfuse-l3 8、dense-min6。参考版本组对齐旧日常开关，并记录实际二进制SHA；其他组使用同一新二进制。原70 + numeric46 + updates64 = 180；单独矩阵4小时可能不足，推荐使用72小时预算的`--all`；全矩阵未完成必须标incomplete。

```sh
strata-headless-compare --group updates --plan
sudo ~/.local/bin/strata-headless-compare --kernel-check
sudo ~/.local/bin/strata-headless-compare --group updates
```

`--kernel-check`只使用预编译目标：GPU正确性、两种TQ预填充数值检查及128MiB常驻权重Dense延迟扫描；无模型请求、无编译。缺测试或关键GPU项跳过即不算通过。正常启动无测试、无队列。完整模型和实际识图验收仍分别通过`--group`和`--vision-check`显式执行。

## 一条命令挂完整长程测试

```sh
sudo ~/.local/bin/strata-headless-compare --all
```

只在无头环境运行。该命令将任务交给系统级临时服务`strata-headless-test.service`，退出终端后继续；不会开机自动执行。只停止一次日常Strata，依次执行预编译内核正确性和常驻权重Dense扫描、180次完整参数对照、真实识图借还验收，全部阶段结束后恢复日常服务。没有编译与测试并行。

总预算默认72小时，`--budget-s 345600`可改为96小时，已经取消旧4小时硬上限。`--foreground`可让同一完整任务留在当前终端。参数组失败或输出过短会保留记录并继续其余组，最终报告incomplete；内核前置正确性失败则不启动模型阶段。中断保留已完成结果；不承诺断电后自动续跑。相互排斥的候选只在对照中切换，不自动改变日常配置。

日志：`~/.local/state/strata/startup-benchmark/full-suite.log`。总报告：同目录`full-suite-latest.json`，内含各阶段报告与独立结果目录。每个参数组仍即时保存独立结果。单独检查命令继续保留，但不必逐条执行。

```sh
strata-headless-compare --all --plan
systemctl status strata-headless-test.service
sudo systemctl stop strata-headless-test.service
```

计划命令无服务副作用。停止命令会请求正常中断并恢复日常Strata；若此时桌面已启用，则不强行切回无头。重复启动会被服务名和运行锁拒绝，不同时争用GPU。识图验收每次用独立合成图缓存目录，确保冷图真实借显存、第二次缓存图不借，恢复后文本须达到3072-token目标的75%。

本轮再次核对生产最终命令和环境，全部DAILY参数一致；prior0.5、spec4/confidence0.8、lookup16、成对行、NOWAIT1/lag2、tail skip7、Embedding计费1、GDN NP2、BF16 PLE及新PR开关均已选择。完整组合仍属未完成模型性能/质量验收的理论选择；新Dense扫描还没有产生新的生产赢家，保留原形状表。


## 重复生成失败重试

已启用`repeat_failure_retry=true`。256次连续token重复保护保留；尚未发布正文/完整工具调用时内部重试一次，重试共用原输出预算。再次失败返回普通提示，保留会话和旧磁盘存档，不再返回空的截断工具草稿。已发布的工具调用不自动重放；图片请求暂不重试，取消语义保留。

`--all`命令不变，内核阶段增加工具恢复/存档回归；性能矩阵中的重复失败或重试结果记为无效，继续后续组。该策略的模拟检查已通过，真实Pi模型联调仍待进行。细节见[工具保护包](features/chatml-tool-guard/README.md#重复生成的有限恢复2026-11-23)。
