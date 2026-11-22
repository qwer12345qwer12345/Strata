# 验证

一个补丁包含旧方案的原有 Host 回归测试与实际部署控制器测试，保留为同一交付单元。

```bash
python downstream/scripts/test.py --feature legacy-session-archive \
  --source ../Strata-prepared --build ../Strata-archive-host
python downstream/scripts/test.py --feature legacy-session-archive \
  --source ../Strata-prepared --build ../Strata-archive-cuda --with-cuda \
  --cmake-arg=-DCMAKE_CUDA_ARCHITECTURES=86
```

在 Linux 上另设置 `STRATA_CONV_DISK=1`、`STRATA_CONV_DISK_DIR=<已存在的测试目录>` 重跑快照测试，覆盖文件后备分配。默认测试验证文件边界、原子提交、增量、缓存控制、流式取消、分支与启动基准。真实模型重启恢复和 32 GB 长上下文峰值需要独立验收。
