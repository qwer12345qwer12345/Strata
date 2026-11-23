# 后续优化计划

2026-11-23，0.2.20集成后。prior0.5、RTX3080 12GiB、32GiB RAM、Gyro-S、K8V4；数值差异本身不淘汰候选。

| 原计划 | 当前状态 | 下一步 |
|---|---|---|
| K8V4 decode TC | #6已移植为独立包 | 同一输入下检查短/长、paged/masked，再比较spec4/6的实际端到端延迟 |
| Dense布局分派 | #10测量工具已接线，没有新赢家 | 先跑常驻权重形状扫描，覆盖T2..16；按胜出形状写下一补丁，不全局覆盖旧表 |
| Q8 HC persistent | #9已移植，修复编译和共享内存读取问题 | 与现有Q8 INT8 TC及QFUSE交叉比较，确认实际是否命中persistent或回退 |
| TQ F16预填充 | #7独立候选已移植，修复geometry变量引用 | INT8/F16数值及真实prompt吞吐、内存峰值和任务质量 |
| resident ADAPT_FETCH | #8已移植，增加批量交换原子性回归 | fetch0/1/2、pin、NOWAIT，完整驻留且专家文件读取为零；再验vision借还 |

配置与64次新增对照见[日常配置](DAILY_CONFIGURATION.md)。先执行GPU正确性与Dense扫描，再手动运行updates及识图完整模型对照；不在正常启动时自动执行。性能/任务结果出来后才能形成新的实测基线。
