# 0.1.65 迁移验收

硬件：RTX 3080 12 GiB SM86、Core Ultra 7 270K Plus、32 GB DDR5。CachyOS 7.3.3、NVIDIA 620.70.06、NVCC 13.4.92、GCC 16.2.1。编译与模型测试依次执行。

- 一个补丁可独立应用；完整活动栈也通过 patch SHA256 与 Git tree 校验，prepare.py 的 Settings 生成/检查通过。
- 独立存档包及完整栈的四项 Host CTest 均通过：conversation_file_test、conversation_cache_test、conversation_cache_l2_test、conversation_spill_test。
- 实际 SM86 上 conversation_snapshot_test、异步 checkpoint、VRAM slot 三种测试在文件后备打开时各重复 10 次通过。普通快照用例覆盖 4349 个检查，含所有 KV 格式与 draft/no-draft。
- 控制器和启动基准共 14 项 Python 测试通过；新增测试在完成 SSE 后阻塞保存，SIGINT 后监督器必须等到保存结束才退出。
- 完整栈工具边界/不完整工具发布的 18 项测试通过。系统 Python 缺 regex，使用已经安装服务依赖的 Python 3.12 运行。
- 已编译完整活动栈的 strata，版本为 0.1.65。

## 真实模型与跨进程恢复

[去隐私结构化结果](model-smoke.json)。保持桌面和连接，独立回环端口，32K 上下文上限、专家 RAM 限制 6 GiB、缓存 1024 MiB / 2 slots。该限额用于验证状态正确性，刻意允许剩余专家从文件读取，不代表无头生产吞吐。

1. 6025-token 提示完成后保存约 304.8 MB V3 档案，正常退出。
2. 全新后端进程通过控制器自动载入，复用 6018/6025 token；相同提示不重写旧档案。
3. 追加对话，复用 6056/6081 token，saveinc 报告 incremental=true，档案增长约 9.81 MB，而非重写完整快照。
4. 再次退出及重启，追加后的提示复用 6074/6081 token。

初次保存 470.2 ms，追加保存 215.6 ms，最后一次 preload 282.3 ms。最低系统可用 RAM 约 11.3 GiB，未触发 1 GiB 的测试停止线。所有启动、请求及正常退出通过。

## 未证明的范围

- 生产 256K 上下文的峰值 RAM、swap 与专家文件读取仍需无头运行实测。
- 0.1.60 存档未宣称可直接恢复到 0.1.65；新部署使用独立目录，原档案保留。模型、配置或状态布局不兼容时重新预填充。
- 没有把受限专家预算的短输出速率当作性能对比。长输出、长上下文正式吞吐由既有无头开机基准另行记录。
- Snapshot 测试原先偶发不一致是 fixture 的 pageable H2D staging 与非阻塞流读取竞态；在 fixture 填充后同步，未改变生产状态复制算法。

## 本机启用

已将本机生产选择器切换到本次完整栈构建，并通过同一选择函数验证版本摘要、源码入口及部署参数。旧二进制与旧档案保留；新档案使用 0.1.65 独立命名空间。原有开机基准及先基准后公开服务的顺序保留。当前桌面保持运行，未在桌面下启动生产 256K 驻留配置。

新引擎构建 SHA256：`da1d4cfc5d039147434662f98f527b5491442d6ee96014feedb46838e32cc405`。完整补丁树指纹见仓库 stack.json；Git tree 不含本地编译产物。
