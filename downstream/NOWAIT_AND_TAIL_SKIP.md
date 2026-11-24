# ADAPT_NOWAIT 与官方 tail skip=7 核对

初始源码核对基于 c6f6427 / engine0.1.65，GPU RTX3080 12GiB，Gyro-S，prior固定0.5。下面的tail=7说明记录其原理与旧选择；2026-11-24本机受控输出质量测试后，当前日用配置已改为tail=0、ADAPT_FETCH=2。

用户允许为性能收益接受数值/输出差异。此前以严格可重复性为默认目标关闭NOWAIT的本机决策不符合这一取舍，现已纠正。两项都用上游仍保留的开关选入，不把上游默认全局改回去，也不需要新内核。

## 已经部署的选择

```json
{
  "STRATA_ADAPT_NOWAIT": "1",
  "STRATA_ADAPT_LAG": "2",
  "STRATA_ROUTE_TAIL_SKIP": "0",
  "STRATA_ROUTE_TAIL_RENORM": "0",
  "STRATA_ADAPT_FETCH": "2"
}
```

下次无头启动生效。完整组合的性能/质量仍未通过验收；不为配置切换中断当前使用，不恢复开机基准或自动队列。

## ADAPT_NOWAIT

`engine/src/core/nv_defaults.cpp` 的SM86表明确移除了默认NOWAIT。理由是交换专家的复制何时完成，决定该专家此轮用CPU还是GPU计算，舍入不同会导致同一请求多次启动输出不一致。该注释给出的RTX A6000对照是开启时24个请求仅7个相同，关闭后3轮均24/24相同；不是在报告内存损坏，也不是功能已删除。

同一注释记录关掉时decode比值0.988、区间[0.969,1.002]（3对），以及早期A4000 drop-one的约6.3个百分点影响。这些都是上游其他硬件/模型条件的记录；不能据此保证本机大幅提速，已有lag2也会减少原本要等的时间。用户要求仍选入并手动测量。

`generate.cpp` 的 `adapt_nowait()` 读环境开关，verify前 `apply_pending(false)` 只查询已经完成的复制；未完成的swap并不被当作已就绪的GPU权重使用。既有eviction table同步保留，未关闭 `STRATA_ADAPT_EVICT_SYNC`。它与 `ADAPT_OVERLAP` 的资格相斥，所以不会为了“全开”再叠不兼容模式。

注意文档矛盾：固定版本的 `docs/DETAILS.md` SM86表格和settings register的部分note仍写NOWAIT默认1，而当前 `nv_defaults.cpp` 真正执行的SM86表未包含它。以代码分支为准；本机现在通过显式环境项1消除歧义。

## ROUTE_TAIL_SKIP=7（原选项，现已关闭）

它与 `STRATA_DF2_LOOKUP_SKIP=8` 是两个独立开关。前者省CPU/PCIe专家工作；后者延后可避免的神经草稿。

当前源码中，某专家只有在**不在GPU缓存**，且verify窗口内所有用到它的token都把它排在索引7及以后时才跳过。Top-10使用0起索引，即尾部最多三个位置；不是每层固定只算7个专家，显存命中的尾部专家照常计算，只要有一个token把该专家排到更前，也不能把它整项跳过。CPU中已有权重也会因计算被跳过而省时间。

跳过的贡献清零，原有其余权重不重新归一化。本机显式维持 `STRATA_ROUTE_TAIL_RENORM=0`，不把未验证的另一个实验混入官方rank7方案。共享专家、神经草稿上限、prior0.5均未因此修改。

`generate.cpp` 和 `tail_skip_default.hpp` 明确：0.1.63起默认关闭；代码仍支持1..9，官方主要测过7。关闭与用户质量反馈有关，不能保证所有任务影响都小。`expert_source.cpp` 的旧行内注释仍写“default7”，但解析器实际未设值为0；已核对实际控制流。

上游 `docs/DETAILS.md` 的UD-Q4_K_XL/RTX5090数据记录decode约×1.24–1.29，KL约0.0126、top-1对齐97.3%；历史3060/A4000也有KL约0.003–0.006记录。这些不是本机Gyro-S数据，不能照搬成速度或质量承诺。缓存命中更高、CPU更强、窗口更长都可能缩小收益；prior0.5与tail skip也可能减少同一批miss工作，收益不一定相加。

该版本退休的 `STRATA_TAIL_SKIP_MAX_CACHED` 不再读取，不添加这个已失效开关。有效性按miss/rank和当前单卡模式判断，不能据旧文档的90%缓存线判定它一定不工作。

## 已有近似优化复核

- 新DP4A表、主模型形状限定Dense TC、lookup16/spec4/草稿延后继续按用户选择开启。
- HC Q8已有：`NativeDense::gguf_carries_q8_hc()` 发现文件的Q8_0 HC投影便自动启用；此前真实模型日志已经记录0.62GiB Q8_0 HC。本次显式固定 `STRATA_HC_Q8=1`，并非再加一轮新量化。
- 不把“会改变末位/输出”本身当作停用理由；仍区分有依据的优化、模式不兼容、已知损坏和无法量化的质量变化。没有为了全开叠加互斥的ADAPT_OVERLAP、fused TQ下不适用的group scratch或退休设置。

## 手动对比

```sh
strata-headless-compare --plan --group adapt-tail
sudo ~/.local/bin/strata-headless-compare --group adapt-tail
sudo ~/.local/bin/strata-headless-compare --group tail-spec
```

第一组是NOWAIT=0/1 × tail=0/7的完整2×2，正反各一次共8组，lag固定2；因此NOWAIT有独立开关对照和同时开启tail的交互对照。第二组tail0/7 × spec3/4/6，共12组，区分短窗口更易整项跳过和长窗口复用更多专家的效果。默认完整计划现为70组、4小时预算，仍只在用户显式命令时运行。

每组长输出3072/1536 token，继续记录首字、吞吐、p95/p99窗口、分来源接受率、输出hash、内存和专家文件读取。新增原始日志聚合：实际adapt wait ms/window、实际tail skip累计专家/entry计数，不能仅因开关为1就宣称已提速或已跳过专家。没有日志样本时明确留空，不编造零等待/零跳过。

完整RAM专家补集和零专家文件读取仍是有效性能组条件。不会借skip掩盖原本的内存不足；磁盘存档与工具边界保护保留。
