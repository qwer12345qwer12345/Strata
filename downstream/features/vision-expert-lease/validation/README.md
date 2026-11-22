# 验证记录（2026-11-23）

此记录仅含自行生成的测试图片和提示，不含用户会话、地址、设备序列号或本地绝对路径。

- [test-summary.json](test-summary.json)：SM86构建、3项C++/GPU测试、64项视觉/手动入口Python测试、25项控制回归通过。
- [gpu-tests.txt](gpu-tests.txt)：真实GPU分段释放/分配、原地址及数据恢复、故障重试。
- [model-api.json](model-api.json)：300图片token预算下10例真实API序列，包括多图、取消、坏图和启动失败后文本恢复。
- [model-api-1024.json](model-api-1024.json)：最终引擎下默认1024图片token预算，文本、冷图、缓存图、后续文本4例通过；reader实际预热1024 tokens。
- [lease-protocol.json](lease-protocol.json)：释放1126–1382MiB、单专家1049600-byte暂存、同槽恢复及耗时。不同请求空闲显存变化会影响释放量。
- [earlier-attempts.json](earlier-attempts.json)：保留两次swap guard中断及一次启动失败错误映射500的发现。最终修正为503，重新跑完整序列通过。
- [environment.json](environment.json)、[deployment.json](deployment.json)：硬件、工具链和部署指纹。

真实API使用桌面受限预算：2GiB专家RAM、5600GPU槽、8K上下文，存在专家文件层读取。其速度不代表完整驻留无头性能。300-token全套最低可用RAM约17GiB，1024-token检查约16.4GiB；全系统swap分别增加约3.4/3.1GiB，所测进程最大swap约204/163MiB。系统swappiness=150，不能把这份结果描述为零交换或32GiB完整驻留已验收。详细KiB值保留在JSON中；没有修改系统swap或网络配置。

最初严格的全系统新增1GiB swap guard在仍有10/17GiB可用RAM时中断。最终受限预算功能检查同时监测进程swap与可用RAM，保留全系统swap记录；见复现脚本。普通系统Python缺少regex，功能包测试改用实际服务Python3.12环境通过；没有为此安装全局依赖。

未测：HIP/SYCL/Metal、一般识图/OCR质量、满256K上下文峰值、完整驻留下归还后零专家文件blob读取。手动 `strata-headless-compare --vision-check` 才做后一项检查；正常服务启动不生成测试请求。该命令还输出恢复后3072-token长文本，防止仅用短回答掩盖文件层退化。
