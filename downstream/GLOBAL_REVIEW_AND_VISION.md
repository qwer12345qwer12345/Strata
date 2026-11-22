# 跨模块配置复核与识图状态

对象：上游固定版本 c6f6427 / engine 0.1.65 + 本仓库活动补丁；RTX3080 12GiB、32GiB RAM、Gyro-S。以下是本地代码核对和已有GPU数据复用，没有执行新的真实模型或GPU性能测试。

## 结论的边界

日常已按用户选择采用理论优先组合并标注NOT TESTED。不能把已有单核最低延迟叫作整机“全局最优”：总耗时还由窗口分布、接受率、CPU专家、PCIe、文件读取和显存分配共同决定。本次复核增加关键交互矩阵，保留整套新旧配置的直接比较；没有虚构这些比较的结果。

## 已确认或需要交叉验证的相互作用

| 关系 | 核对结果 | 目前选择和后续 |
|---|---|---|
| 新形状表 × QKV TC | 已有修正后的微测试显示：新DP4A的2/4/8列约12.26/13.28/16.29µs，旧TC约15.82/16.04/18.43µs。旧表的选择不能直接继承到新表 | 日常新表+关闭≤8列TC；shape-qkv组测完整2×2，避免把TC优劣误归给形状表 |
| spec × lookup | `generate.cpp` 将神经窗口上限保留在mtp_max_t，再单独放宽lookup。只改spec而不固定lookup范围会同时改变两个变量。DFlash2主体固定8行，spec只缩短head/selector和验证提案，不会同比缩短整个drafter | 日常spec4、lookup cap16；spec-lookup测3×3完整矩阵 |
| Dense TC × lookup | Dense短窗的舍入会改变文本和路由，继而改变可匹配的lookup，kernel快并不保证少跑总轮次。spec4下Dense T6..8优化主要由lookup窗口触发 | 日常开启已测形状，Dense与lookup cap8/16测2×2；分别记录神经/lookup接受数及输出hash |
| 草稿延后 × lookup | 满足长匹配时延后drafter，成本策略若仍选lookup便省一次无用神经草稿。但宽提交窗口的 `context()` 可走>8行batch TC，而直接 `draft_extended()` 按≤8行录入，两条累加路径不同 | 日常skip8为未测理论选择；skip-lookup测2×2。不能宣称它只有时间变化、绝无数值或后续提案变化 |
| fused prompt × group scratch | group scratch要求每层MMQ且没有fused/FP16路径；`prefill.cpp` 对不满足条件的路径拒绝或忽略。fused TQ下打开两者不是叠加收益 | 保持fused TQ=1、group scratch=0，不以“更多开关”为优化 |
| 宽窗口 × 显存/专家 | 宽窗口图按需捕获；显存不足时上游禁用该宽度并回退。额外预留显存会减少GPU专家，RAM补集增加。prior又与GPU驻留相关，变化可能影响输出和命中 | 保持448MiB预留；对比记录实际宽度、内存、文件读取，不把cap16开启当成16行有效收益 |
| KV常驻 × 专家 | 增大KV常驻可减少长上下文传输，却占GPU专家空间；减小则可能把瓶颈移到PCIe。CPU/GPU可同时快也不代表两者分配最优 | 保持KV resident20480；不能凭静态推演定量断言更大或更小更优 |
| warm-up × prior | 上游 `STRATA_WINDOW_WARM` 默认遇到prior时关闭，明确为避免dummy token填入的专家影响首次请求路由。强行预热或开机基准会改变冷启动的缓存状态 | 正常启动无模型测试；不强行打开window warm，允许首次请求捕获图的成本存在 |
| 异步交换 × 可重复性 | 上游SM86默认已移除ADAPT_NOWAIT=1，并解释其CPU/GPU执行副本时序可能使输出不重复。用户接受这类差异，不能据此替用户把性能选项一律关闭 | 已显式开启NOWAIT=1、lag=2，加入adapt-tail矩阵，性能与质量标记未测 |
| BF16 compact × 内存 | compact把H2D减半且不丢BF16精度，但raw GPU scratch、BF16行缓存仍占资源；复制流改善重叠并不凭空省掉专家容量 | 保留已有compact/独立流；不以这项传输优化支持增加视觉预算 |

源码入口：`engine/src/kernels/cuda/dflash2.cu`（表与TC优先级）；`engine/src/core/dflash2.cpp`（context/bridge）；`engine/include/strata/spec/window_rule.hpp`（延后）；`engine/src/program/generate.cpp`（窗口上限、graph_fits、prior warm-up、VRAM命令）；`engine/src/prefill/prefill.cpp`（group scratch资格）；`engine/src/core/nv_defaults.cpp`（SM86默认）。

对比已扩成70组关键交互矩阵，默认预算4小时，含ADAPT_NOWAIT与tail skip及spec的交叉。详见 [日常配置与命令](DAILY_CONFIGURATION.md)。这仍是有限实验设计，不是所有开关的高维穷举。

## 识图现在的准确状态

已按 [显存借还设计](design/GYRO_VISION_VRAM_LEASE.md) 实现独立 [vision-expert-lease补丁](features/vision-expert-lease/README.md)，并配置为下一次无头启动启用。该节更新取代此前“未实现、保持关闭”的状态；前面的计算内核分析仍有效。

单卡CUDA下，服务在FIFO内暂停语言引擎，释放专家缓存尾部物理段，GPU编码图片后立即退出reader，再从原始专家来源恢复原槽位。借出期间命令门禁禁止生成，归还失败保留门禁供重试。普通VRAM命令原有resident拒绝条件未删除；不需要增添一份完整RAM专家副本，也不复用Live Memory。

本机0.1.58 reader和BF16 mmproj已与本次编译的0.1.65引擎完成真实API兼容检查；图片上限1024 token。冷图、多图、缓存、取消、坏图和reader启动失败的归还均检查通过。测试采用桌面2GiB局部专家RAM预算和8K上下文，不能证明日常完整驻留、256K峰值或一般识图质量。无头正式检查入口为 `sudo ~/.local/bin/strata-headless-compare --vision-check`；普通启动不会执行它。

CPU静默回退在该opt-in模式中关闭，避免GPU reader启动失败时引入未预算的CPU权重和工作区。图片编码缓存使用磁盘缓存目录；专家恢复只用单blob暂存。完整结果及失败记录见 [validation](features/vision-expert-lease/validation/README.md)。

补充：ADAPT_NOWAIT与官方tail skip的源码核对、文档矛盾和用户性能取舍见 [NOWAIT_AND_TAIL_SKIP.md](NOWAIT_AND_TAIL_SKIP.md)。此前以严格重复性关闭NOWAIT的本机决策已纠正；上游默认不做修改，通过其保留的环境开关显式选入。
