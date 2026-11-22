# 完整旧版 V2/V3 存档包

按用户选择，把旧部署使用的 PR5500 存储、低内存停放、缓存控制协议和自动 Chat Completions 控制器合为 **一个补丁**，移植到上游 `c6f6427d`（终端 0.2.19 / 引擎 0.1.65）。旧实现依据 `ampere-gyro-latency` 中的 PR5500 核心及文件后备改动，控制器依据实际部署的流式版本。

## 行为

- `/v1/chat/completions` 每次成功完成后同步 `saveinc`；SSE 逐行转发，结束后保存。索引仅保存消息散列，存档本身包含可恢复的模型状态。
- 下次相同会话先探测活跃缓存，必要时按消息前缀载入磁盘存档；分支使用独立存档，相同提示不覆盖已有的较完整 checkpoint。
- V3 使用 base/delta 段、checkpoint 和原子 manifest 发布；保留 V1/V2 读写与兼容检查。当前 V2/V3 为原有 stored blocks 格式，不需要 LZ4 链接库。
- `STRATA_CONV_DISK=1` 与 `STRATA_CONV_DISK_DIR` 让 Linux 停放快照使用临时文件后备；不改变活跃 KV 所需的 RAM/VRAM。保留新版内存诊断、SSD spill fallback、P2P 与草稿状态布局。
- 控制器正常收到 SIGINT/SIGTERM 后关闭监听、等待正在处理的请求与保存结束，再停止后端。默认等待 30 秒，超时保留最后已提交存档；强杀和断电无法保证保存尚未完成的一轮。
- 默认磁盘预算 32768 MiB、最多 16 个会话档案。当前正在保存的单个档案不被预算清理删除，因此预算不是单档案的硬上限。

## 使用

```bash
python downstream/scripts/prepare.py --all --target ../Strata-prepared
cmake -S ../Strata-prepared/engine -B ../Strata-prepared/terminal/build \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build ../Strata-prepared/terminal/build --target strata -j2
python ../Strata-prepared/terminal/l2/managed_server.py --profile /path/to/profile.json
```

profile 的 `exe` 应指向上述可执行文件，`cwd` 指向 prepared 的 `terminal`；保留用户的模型/量化/采样设置。`slot_save_path` 应使用新版本专用目录。wrapper 默认启用 V2/V3、文件后备和同步写入（`STRATA_L2_WRITER_BUDGET_MIB=0`），显式 profile 环境项除 writer budget 外优先。可通过 `l2_disk_budget_mib` 改变磁盘预算。模型/配置 identity 不符时会拒绝恢复并重新预填充。

正常启动不发送基准请求、不执行旧队列；内部服务ready后直接对外提供服务。显式手动性能测试的流量不进入日常会话档案。统一runtime selection负责环境项、BF16表和参数覆盖，未测试组合会写入状态提示。独立使用本包不强制安装内核或工具保护；完整活动栈另含这两类补丁。未使用已退役的 Live Memory 方案。

## 验证边界

Host 文件/缓存测试、CUDA 状态往返和真实模型跨进程恢复分别验证不同边界；详见 validation。测试中修正了上游 fixture 的异步 H2D 写入竞态：填充结束后先同步，再允许非阻塞流读取，未改变运行时算法。

32 GB RAM 下 256K 的极限峰值仍须无头实测。文件后备可减轻停放副本压力，但活跃 KV、专家驻留、checkpoint 与写回脏页仍占资源，磁盘容量不能直接换算为可支持的活跃上下文。
