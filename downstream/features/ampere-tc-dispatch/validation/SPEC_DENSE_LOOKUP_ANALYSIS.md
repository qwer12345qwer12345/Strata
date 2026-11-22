# Spec、Dense TC与宽lookup的进一步核对

本次只读取已有真实逐轮记录及当前源码，未新跑吞吐测试、未修改日常spec=6配置。计数来自已公开的1024-token代码和6027-token上下文测试。更短轨迹的裁切不是反事实吞吐测试，后续状态、提议和窗口策略会变。[结构化核算](spec-and-dense-causality.json)

## 降低spec

值得先对照6与4，3作为更激进的下一档。当前DFlash2调优代码样本：57个神经轮次中3轮超过4行；按原轨迹截到4行少验证5行，没有截掉这些轮次原已提交的token。对compact说明文本样本，112个神经轮次中11轮超过4行，原轨迹会截掉18个已提交token。短平均值不能排除少数高收益长窗口。

`--draft-conf .8` 已用草稿概率乘积缩短验证窗口。降低spec不会把DFlash2的五层Transformer主体从8行缩为4行；它仍计算anchor+7 masks。主要减少head、top-k、selector的行数，以及原先可能被验证的尾部行。5个head输出缩为3个也不表示整个drafter成本减少40%，权重读取和主体计算仍在。

直接把spec从6改为4，还会把默认suffix lookup上限从8改为6，混入另一个变化。公平对照应三组都显式保持 `--lookup-max-t 8 --lookup-sizes 2,3,4,5,6,7,8`，分别spec6/4/3，固定prior=.5、conf=.8与其他参数。两种测量分开：同token序列/同窗口的计算成本，以及自由生成的端到端结果。后者必须同时报告来源占比、有效token/round、verify/commit/draft分项，不能只报告accepted/offered。

## Dense TC接受率的根因边界

应比较相同DFlash2 scalar+QKV TC调优下，Dense短窗口开关的开/关：

| 1024-token代码 | Dense关闭 | Dense开启 |
|---|---:|---:|
| 神经草稿 accepted/offered | 89/101=88.12% | 695/792=87.75% |
| lookup accepted/offered | 699/700=99.86% | 59/76=77.63% |
| lookup占所有提议 | 87.39% | 8.76% |
| 合计accepted/offered | 98.38% | 86.87% |

保持旧的两类来源接受率，只换为新来源占比，合计已经降到89.15%。这是算术分解，不是同前缀质量/因果估计。已确认的大头是新文本和窗口策略下lookup提议变少；不能把合计下降全部归到神经drafter变差。

已有GPU微测试表明Dense TC与普通native MMVQ的相对RMS差异约1e-7，非逐位相同。该decode路径继续使用传入的相同Q8_1激活和相同Q5_K权重，不是新增了FP4/FP8量化。INT32点积后的缩放、FMA和浮点累加/归约顺序不同，提供了数值漂移来源；微小漂移如何在真实投影、再量化、注意力/递归状态和MoE路由中放大，仍需逐层测量。

相同drafter设置的短请求，trace位置22开始的27个已观察输出token相同，首个输出分歧在位置49；此前对应起点的验证窗口大小没有分歧，drafter概率已不同。这支持隐藏状态/概率先漂移、文本随后分叉，不支持简单归因为先选了不同lookup窗口。但没有逐层真实权重、同前缀logits/top2 margin核查，不能声称已确定首个异常层，或已证明模型质量下降。

进一步定位：固定prompt、continuation、验证T和专家分配，只切Dense开关；上游serve已有STRATA_FOLLOW_DIR与STRATA_FOLLOW_LOGPOS，可保持同一token前缀，另记录argmax、forced-token logp及差异。逐层比较Q5_K投影前后残差、再量化代码、路由top-k与首次argmax翻转的top2 margin。最后另做自由生成和质量评估，避免混淆数值一致性、接受率和质量。

## 形状变体与特调空间

上一轮只选择已有的17/26/31等变体，尚未系统搜索所有tile、warp、unroll或为3080写新的数据管线。应优先优化真实分布：drafter主体固定8行的gate/up、down和QKV；context update多为1..4行；selector/head则随spec变化。目标按“调用频率×可节省GPU时间”排序。

测量继续以固定resident weights、超过L2的轮换权重和CUDA event/graph为依据，避免专家磁盘读取污染。先搜索已有kernel变体，再考虑协同向量加载、寄存器/共享内存占用、权重块复用、双缓冲及小投影融合。仅改变线程和加载分配时优先保留原累加顺序；改变算术顺序的候选独立标记，并用真实权重与近似/位级边界分别验证。

## ≥9行探索

优先探索更长的lookup，保留神经spec4/6。参考代码151/208有效轮次使用lookup，其中149轮全部接受、历史匹配最长64token，已经有数据动机；说明文本lookup少，不能统一放宽。

上游有独立最高16行lookup，窄8行构建也支持；默认明确排除DFlash2。DFlash2的draft_launch与context_async仍限制每次新上下文1..8行，但同步context已能分块处理更长输入，且已经分配8行landing缓冲。因此不需要修改drafter的训练结构来制造第8/9个神经草稿。

可做的桥接补丁：对长lookup已接受的token前缀，用现有context按≤8行分块更新ring/log；保留最后1..8行拷入稳定landing后，再启动一次正常draft。需同步调整prepare绑定、常规/延迟draft、请求结束更新，保持taps、position、epoch、rollback和log完整，避免CUDA graph的输入指针变化。仅增加“允许n_new>8”的判断不够。

验收先比较8、12、16行的分块状态、ring/tag、draft输出、拒绝后下一轮、请求退出/重启恢复，再做同输入的ABBA。评判每提交token耗时与额外显存、专家命中，不能为了触发Tensor Core而填充无用行。语义正确之后由lookup成本策略只选择实际获益的长度。

当前没有启用宽lookup，也没有把以上桥接描述当成已通过测试的实现。
