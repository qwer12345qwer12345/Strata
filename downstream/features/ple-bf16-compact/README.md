# BF16 PLE 原始行上传

适配上游 c6f6427d（0.2.19 / engine 0.1.65）。`STRATA_PLE_BF16_COMPACT=1` 时，BF16 PLE 的原始 320-byte 行直接进入现有双缓冲，再上传 GPU，以位移无损展开成 FP32。原有 Direct reader 的去重、排序、有界行缓存和异步 I/O 保留。无需把整张约 95.37 GiB 的 BF16 表放入 RAM。

`STRATA_PLE_COPY_STREAM=1` 复用上游 IQ4_NL compact 的独立上传流与事件屏障；上一块读取完原始 GPU 缓冲才允许覆盖，本块数据上传完才允许计算读取。输入为每 token 16 行、共 2560 个 BF16：H2D 从 10240 降至 5120 bytes/token，CPU 不再形成这份 FP32 展开副本。没有重数量化，signed zero、subnormal、NaN payload 全部保留。

缺少 pinned host buffer 或 raw GPU buffer 时仍走原来的 FP32 上传。额外 raw GPU scratch 为每块 token 数 ×5120 bytes（2048 token 约10 MiB）；相对上游 IQ4_NL raw scratch 增量为每 token 3680 bytes。Pinned host buffer 的现有容量未增加。BF16 行缓存同样的行数会比 IQ4_NL 多占 RAM，不能以传输减少推断总内存下降。

该开关只改变 BF16 的 prompt 路径，decode 的表读取与原有格式均保留。开关默认关闭。完整栈 metadata 仅声明开关；部署须显式配置 `STRATA_PLE_BF16_COMPACT=1` 和 `STRATA_PLE_COPY_STREAM=1`。

本包沿用上游 IQ4_NL 的“原始行上传、GPU 展开、独立流”结构，扩展到 Direct BF16，未修改上游推测解码算法。

## 验证

`ple_reader_selftest` 增加 Direct/mmap、threaded/non-threaded、有界缓存、越界行、重复行、预读与 pending-token 交互的 raw BF16 往返。`ple_bf16_widen_test` 穷举所有 65536 个 BF16 位模式，5 次跨流事件往返均逐位相等。真实 BF16 模型验证范围与结果见 validation。
