# 下一轮需要代码的优化计划

2026-11-23。当前执行范围为配置和手动测试计划；以下尚未实现，不混入本次配置交付。目标仍为RTX3080 12GiB、32GiB RAM、Gyro-S、prior0.5、DFlash2、K8V4；不要求跨硬件逐位相同。

| 顺序 | 改动 | 复用与预期收益来源 | 验证要点 |
|---|---|---|---|
| 1 | K8V4解码Tensor Core分块 | 复用已有query三部分INT8 IMMA、Q4 V gather，以及现有decode scratch/log-sum-exp merge；保留K8V4存储 | 当前decode已有chunk/merge，需比较实际选中cells和query宽度下的CTA、TC吞吐与merge成本。短/长attention、paged/streamed KV、尾块和空split；显存/内存预算不增加一份K8V8 |
| 2 | Dense非exact布局的按形状分派 | 将官方保留的generic reduction布局加入候选，按格式/K/N/T比较当前B6、IL表、packed和TC路径；只替换赢家 | 现有setter无生产接线，不能全局false绕过所有新优化。固定resident权重与窗口测细微GPU延迟，再测完整模型接受率、lookup和长输出 |
| 3 | Q8 HC融合/持久内核 | 现有HC_PERSIST直接拒绝Q8 HC和QFUSE，须复用INT8点积、现有scale与epilogue实现兼容版本 | CTA必须同时驻留，避免grid barrier死锁；比较现有Q8 TC路径，包含QFUSE和≥9行分块，不以减少launch数直接推导收益 |
| 4 | Gyro TQ的F16/更宽预填充变体 | 参考官方IQ格式F16/NT256候选，为TQ另写合法分派，保留其码本/旋转和权重存储 | 现有全局F16/NT256会令trellis融合失去资格，不能直接叠加。比较INT8 DP4A/TC与F16的真实形状、暂存和带宽，检查range/NaN及任务质量 |
| 5 | resident complement下ADAPT_FETCH | 复用已抓取的热门miss直接录入GPU缓存，减少重复PCIe读取，提高命中；需要与RAM补集所有权交换接线 | 逐层归属、被驱逐专家的RAM可用性、异步复制完成、NOWAIT可见性和vision lease冻结一致；不删除拒绝条件来冒充支持 |

优先先取得本轮配置的手动无头数据，再按结果调整以上顺序。每项先做不受专家文件读取干扰的微测试，再做实际任务；数值差异本身不淘汰。性能候选与已证实优化分别记录，失败或无收益的结果也保留。

不需要代码的候选已加入[日常配置与手动矩阵](DAILY_CONFIGURATION.md)：Embedding/GDN、HC_UPMIX、GDN执行分块、预填充上限/CPU share以及resident overlap替代调度。旧GPU的SELECT_SIMT只对其他设备有意义，不计入3080测试。GR_V3在当前Q8 HC分支前已返回，HC_PERSIST也拒绝该格式；不把无效开关占满测试队列。STAGE_PIN已有真实损坏记录，不属于本轮数值取舍候选。
