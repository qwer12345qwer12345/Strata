# SM86 验证

GPU RTX 3080 12 GiB / SM86，CPU Core Ultra 7 270K Plus，32 GB DDR5。NVCC13.4.92 / GCC16.2.1 / NVIDIA620.70.06，Release、CUDA arch86。所有编译结束后再做GPU测量。

- `ple_reader_selftest` 通过，含新 raw BF16 Direct/mmap、I/O threaded/caller、重复/越界/缓存/预读/pending 的往返。
- `ple_bf16_widen_test` 通过，所有65536编码、5次跨流事件顺序检查逐位相等，包括NaN位。
- 实际BF16 PLE表只有 `per_layer_token_embd.weight`，GGUF type30，shape[160,320001536]。服务日志确认走Direct、lossless BF16 raw upload与independent copy stream。
- 完整生产源树与功能包精确拼接的树一致，指纹见 stack.json。prepare.py 的 Settings生成/检查通过。
- 对照模型保持5600个GPU专家槽、6 GiB专家RAM、16K上下文、2048 prefill，关闭自适应换入和PCIe分担，prior=.5、DFlash2 spec6/conf=.8。桌面保持运行，模型会读磁盘专家，不能当无头性能验收。
- BF16普通与compact的48-token热身、1024-token代码输出散列相同。6027-token上下文后256-token输出散列不同；代码检查和纯数据测试证明BF16位转换无损，但这组模型运行不构成整引擎所有输出逐位相同的证明。

模型结果和逐轮统计见 [另包的共享记录](../../ampere-tc-dispatch/validation/model-check.json)。GPU位转换和读取器的通过不能证明256K活跃上下文的内存峰值。

HIP与SYCL未构建验证。本次仅部署SM86 CUDA二进制。

## 本机生效路径

系统服务仍读取原root profile；其实际生产选择器现覆盖 `--ple-gguf` 到已核实的BF16表。没有修改量化专家/原生dense来源。使用独立BF16存档目录，保留原IQ4_NL档案，避免把旧PLE状态误恢复到新表。已在调用服务相同的选择函数中验证最终参数与二进制摘要。
