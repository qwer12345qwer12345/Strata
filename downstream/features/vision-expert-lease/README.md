# Gyro识图的专家显存借还

c6f6427 / engine0.1.65上的独立opt-in包。单卡CUDA、FileExpertSource驻留补集、单请求serve。让语言引擎在图片编码期间暂停，把专家缓存的尾部物理显存段借给编码器；编码器结束后从原始模型文件/已有RAM副本逐个恢复原专家、原槽位，再开始语言推理。

不增加一整份RAM专家镜像，不降低prior，不复用已退役Live Memory。普通VRAM命令原来的resident拒绝条件保持；这是单独的事务，不允许缩容后继续用不完整专家集合生成。

## 使用

本机已经配置，下一次正常无头启动生效。上游source默认仍关闭，外部使用需明确配置：

```json
{
  "vision": {
    "exe": "<strata-vision>",
    "mmproj": "<mmproj-Qwen3.8-Flash-Next-BF16.gguf>",
    "model": "<native-GGUF>",
    "gpu": true,
    "expert_lease": true,
    "max_tokens": 1024,
    "threads": 8
  },
  "vram_elastic": true,
  "env": {"STRATA_VISION_EXPERT_LEASE": "1"}
}
```

`apply_vision_defaults()` 在`expert_lease=true`时补齐elastic与引擎环境开关；engine_args沿用已有`--vision`/`--vram-elastic`接线。本机分段设128MiB以减少让出容量的舍入，尚无128对512MiB的吞吐A/B。图像预算使用上游计算，默认至少2048MiB，不能将显存空闲量不足仍当成功。

1024是图片token上限，不是语言输出或总上下文上限。图片质量不是这次小型功能测试的结论。

## 生命周期与失败关系

1. Service在现有FIFO内确认图片缓存；全命中则不启动编码器、不借显存。
2. `VISION_LEASE BEGIN reserve_mib`排空verify commit、异步adaptive工作和CUDA流；要求prefill借槽均已归还。元数据先分配，再释放缓存尾部整段。
3. 原host/device专家归属表保持不变，snapshot记录其所有权和原映射容量。此时命令分派器拒绝模型、存档和普通VRAM工作，后台pool upkeep也暂停；仅允许归还事务或安全退出。
4. 编码本次所有图片；任何编码失败、启动失败或取消都进入finally。先关闭并确认编码器子进程已回收，再发送`VISION_LEASE END`。
5. 映射回原地址，以单个最大专家大小的pinned缓冲，复用FileExpertSource `copy_blob()`和原startup的blocking slot fill。每次上传完成才重用缓冲。整批恢复与residency publication成功前，不清除推理门禁。
6. 部分unmap失败会按实际剩余段回滚；映射/读取/发布失败保留事务，可重试END。恢复失败返回503，后续请求先尝试恢复，不能悄悄带缺失专家继续生成。重复END是幂等的。

源码分为 `vision_expert_lease.hpp` 状态事务、generate.cpp的安全点/分派接线、server.py的编码器作用域。`ExpertCache`稳定虚拟地址和source布局逻辑直接复用，没有再实现另一套量化或缓存格式。

本机实际最大专家暂存1049600 bytes，snapshot为少量索引；不是将约1.2–1.4GiB让出专家复制进RAM。分段映射的驱动元数据、文件页缓存和编码器自身内存仍存在。

图片编码缓存放到`vision.cache_parent`；lease模式未指定时使用XDG磁盘缓存目录下`strata/vision`，不默认放在可能为tmpfs的`/tmp`。延用上游64图缓存和unload保留编码结果的行为，服务shutdown清理自己的临时子目录。合并embedding的临时缓冲仍可用上游共享内存路径。GPU reader启动失败时，本模式不静默切CPU以免额外RAM冲掉专家预算，返回503并恢复专家。

## 测试

- C++状态事务：成功借还、重复BEGIN/END、部分shrink失败回滚、grow/read失败后重试、所有权变化门禁。
- 真GPU VMM：均匀/不同slot尺寸，释放物理段后另一个模拟编码器实际分配显存；按原地址/原槽位逐字节校验数据；注入部分unmap和读取失败后恢复。
- Python新旧视觉/HTTP回归64项、既有控制回归25项通过，包括FIFO并发等待、取消、多图、缓存、不可回收子进程、失败后文本重试、手动计划零副作用。
- 真实Chat Completions：自行生成的三色矩形，正确回答“Red, Green, Blue”；冷图借还各一次，同图再次请求零借还；多图、借出后断开客户端、损坏图片400、注入编码器启动失败503及后续文本均通过。300 token上限下完整10例，1024上限另做冷图/缓存图/后续文本检查，reader确实预热到1024。见 [记录](validation/README.md)。

桌面测试使用2GiB局部专家RAM、5600 GPU槽、8K上下文，存在专家文件层I/O。不能据此宣布正式256K满驻留、零文件读取或一般识图质量验收完成。新借还发生时临时读原始文件是预期的；正式无头验收要求归还后的生成不继续依赖专家文件层。

## 手动无头验收

```sh
strata-headless-compare --vision-check --plan
sudo ~/.local/bin/strata-headless-compare --vision-check
```

显式命令才暂停服务、测冷图/缓存图/恢复后的3072-token文本，并在结束后恢复日常服务。它要求完整补集page-lock并确认请求的decode专家blob文件读取为零；检查到RAM budget降级或未完整驻留不算通过。正常启动不会自行运行测试。

桌面受限预算实验的参数化脚本为 `bench/real_api_check.py`，使用服务的Python环境（含Pillow/regex），显式传terminal-root、profile、output、reader、mmproj。它只开本地测试端口18295，不修改systemd/网络；运行前须避免其他模型进程。此脚本是受限预算功能检查，不能替代上述无头验收。

## 兼容边界

本次只构建并验证SM86 CUDA/Linux。拒绝layer split、batch、pipeline、kv-grow、peer/helper缓存与sealed nibble缓存。没有HIP/SYCL/Metal结果。已使用本机0.1.58组件的reader与BF16 mmproj完成API兼容检查；没有发布或替换该二进制。推理引擎是本次重新编译的0.1.65+补丁。
