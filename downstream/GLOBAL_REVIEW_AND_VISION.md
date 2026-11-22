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
| 异步交换 × 可重复性 | 上游SM86默认已移除ADAPT_NOWAIT=1，并解释其CPU/GPU执行副本时序可能使输出不重复。追求低等待不能直接把旧快速开关全部恢复 | 继承当前上游默认，不强开未验证异步交换 |
| BF16 compact × 内存 | compact把H2D减半且不丢BF16精度，但raw GPU scratch、BF16行缓存仍占资源；复制流改善重叠并不凭空省掉专家容量 | 保留已有compact/独立流；不以这项传输优化支持增加视觉预算 |

源码入口：`engine/src/kernels/cuda/dflash2.cu`（表与TC优先级）；`engine/src/core/dflash2.cpp`（context/bridge）；`engine/include/strata/spec/window_rule.hpp`（延后）；`engine/src/program/generate.cpp`（窗口上限、graph_fits、prior warm-up、VRAM命令）；`engine/src/prefill/prefill.cpp`（group scratch资格）；`engine/src/core/nv_defaults.cpp`（SM86默认）。

对比从36组逐项消融扩成50组关键交互矩阵，默认预算3小时。详见 [日常配置与命令](DAILY_CONFIGURATION.md)。这仍是有限实验设计，不是所有开关的高维穷举。

## 识图现在的准确状态

**当前日常没有开启识图。** 原服务profile没有vision段或`--vision`，统一选择器也没有添加它。

本机有对应的约866MiB BF16 mmproj文件，有一份来自engine0.1.58组件的`strata-vision`；这证明文件存在，不证明它与当前0.1.65、模型和低内存预算已经完成兼容性验收。此轮没有加载编码器。

### 上游现成GPU方案为什么还不满足我们的目标

1. `serve/server.py` 的 `_vision_up()` 请求引擎腾显存；`_vision_idle_down()` 在闲置后关闭编码器并归还。默认闲置时长约120秒，常见实现会让编码器与后续语言生成共存一段时间，并非图片编码完成后立刻归还。
2. 当前低内存模式仅在RAM保存GPU专家的补集。`generate.cpp` 的 `vram_command` 在 `src.complement_ready()` 时明确拒绝伸缩：GPU中的一些专家没有RAM副本，直接让出槽位会失去完整可用覆盖。
3. 上游 `reader_room.hpp` 为识图+elastic+该驻留模式至少保留2048MiB。相对现有448MiB，需再让出约1600MiB GPU专家空间；对应专家转入RAM补集，可能使32GiB内存越界或触发文件层，不能只比较编码器自身快慢。
4. 原先“编码前借专家显存、编码后按原槽恢复再生成”的方案仍只是 [设计文档](design/GYRO_VISION_VRAM_LEASE.md)，活动补丁栈没有实现或验收该事务。新DP4A优化速度，不改变这个专家所有权关系。

因此，**不能宣称GPU按需借还识图已经可直接开启**。启用上游静态保留方案也许能运行，但会改变专家容量，尚无本机驻留证据；本次没有悄悄开它。

### CPU路径

上游支持 `vision.gpu=false`，可绕开GPU腾显存问题；仍需在RAM容纳编码器权重和工作区，读取图片也有CPU耗时。当前32GiB与0.25GiB专家headroom并不足以静态证明额外内存一定能装下，旧reader兼容性也未验证。它是比GPU借还更接近现成实现的候选，并不是已验收功能。若开启，宜按需加载、限制图片token，单独核对内存与图片输出，再决定是否日常启用。

本次保持识图关闭，保留当前文本服务容量；未启用Live Memory、未重开ninfer、未下载或加载视觉模型。
