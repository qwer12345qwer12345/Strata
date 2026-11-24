# RTX 3080 / SM86 无头测试记录 — 2026-11-24

本目录记录针对 `ampere-tc-dispatch` 功能分支执行的全套手动无头诊断测试。它不是仓库的 `strata bench run` 标准基准，不应被解读为性能排名、质量结论或完整模型验收。

## 环境与执行

- 命令：`sudo ~/.local/bin/strata-headless-compare --all`
- Run ID：`full-suite-20261124-041837-964068`；本地时间约 04:18–08:29（Asia/Shanghai，约 4 小时 11 分）。
- CachyOS Linux 7.3.3-1；NVIDIA GeForce RTX 3080 12 GiB（SM86）；驱动 620.70.06；CUDA Toolkit 13.4。
- Strata Terminal 0.2.12（build 0.1.58），Python 3.12.15；本次 engine 二进制 SHA-256：`1bd22aee89eb99f7635089c0b04d6265507b13f3c92acd7c01e1d39726a371bf`。
- Runtime 基于 `6a6db7ff58c283cf28d2b913404b5fa31a767eaa`，但当时工作树有 66 个已修改/未跟踪路径；该完整构建不能仅由该提交复现。

## 测试代码修正

Dense sweep 原先以 `native_q8_1_bytes(c.k, 16)` 请求 16 列 Q8_1 缓冲区；helper 的单次列数上限为 8，而后续量化本来就按两批、每批最多 8 列执行，因此请求本身超出 helper 契约。现改为按每列字节数乘 16 分配：`size_t(16) * native_q8_1_bytes(c.k, 1)`。这让总容量与实际两批写入相符，不再让测试在进入候选内核前失败。

修改后的完整测试源码位于 [dense_layout_sweep.cpp](../../tests/overlay/engine/tests/cuda/dense_layout_sweep.cpp)；单行补丁见 [dense-layout-allocation-fix.patch](dense-layout-allocation-fix.patch)。修正文件 SHA-256：`64165749772178a6d2cca47c3689250bdd2343a793bd062ac5dad8ff95fd91ca`。

## 结果

| 阶段 | 状态 | 观测 |
|---|---|---|
| Kernels | **passed** | repeat-recovery 回归通过；6 个必需 CUDA CTest 用例全部通过、0 跳过；TQ-F16 mode 0/1 通过；Dense layout sweep 通过。 |
| Comparison | **incomplete** | 180/180 项均完成记录：2 passed、178 failed；170 项因记录到的 Ampere CUDA 路径与请求模式不一致而失败，8 项达到比较阶段时间/内存限额。 |
| Vision | **failed** | warmup、冷图像、缓存图像、专家恢复后文本共 4 条诊断记录；阶段以相同的 Ampere 路径不一致错误结束。 |

通过的 6 个 CTest：`qsa_decode_tc_parity`、`qsa_decode_tc_parity_env`、`file_expert_source_rotation_gpu`、`vision_expert_lease_gpu_test`、`hc_persist_parity`、`hc_persist_default_off`。CTest XML 和各阶段记录见下方原始证据链接。

Dense sweep 使用 128 次迭代、4 组交错计时，覆盖 20 个 `(type,K,N)` 形状和 `T={2,3,4,5,6,8,9,12,16}`；JSONL 共 566 行（含元信息行），最大观测相对 RMS 误差 `1.96995491e-7`。完整逐项数据在 `dense-layout.jsonl`。

**结论边界：**内核检查阶段通过不代表 `--all` 全部通过。比较阶段大多数运行未满足模式路径校验，视觉阶段也因该校验失败；因此这批数据不能支持性能赢家、完整比较或端到端质量结论。没有将其标作标准基准。

## 完整机器可读记录

- [full-suite-envelope.json](full-suite-envelope.json)：运行计划、各阶段完整摘要/记录；180 条 comparison 运行明细按序放在下列 12 个 shard 中（每份 15 条），所有配置、指标和失败原因均保留。
- Comparison：
  - [comparison-runs-001.json](comparison-runs-001.json)
  - [comparison-runs-002.json](comparison-runs-002.json)
  - [comparison-runs-003.json](comparison-runs-003.json)
  - [comparison-runs-004.json](comparison-runs-004.json)
  - [comparison-runs-005.json](comparison-runs-005.json)
  - [comparison-runs-006.json](comparison-runs-006.json)
  - [comparison-runs-007.json](comparison-runs-007.json)
  - [comparison-runs-008.json](comparison-runs-008.json)
  - [comparison-runs-009.json](comparison-runs-009.json)
  - [comparison-runs-010.json](comparison-runs-010.json)
  - [comparison-runs-011.json](comparison-runs-011.json)
  - [comparison-runs-012.json](comparison-runs-012.json)
- [comparison-request.json](comparison-request.json)：本次比较阶段请求参数。
- Kernel： [kernel-results.json](kernel-results.json)、[ctest.xml](ctest.xml)、[repeat-recovery.log](repeat-recovery.log)、[tq-f16-0.log](tq-f16-0.log)、[tq-f16-1.log](tq-f16-1.log)、[dense-layout.stderr.log](dense-layout.stderr.log)、[dense-layout.jsonl](dense-layout.jsonl)。
- Vision： [vision-results.json](vision-results.json)。
- [provenance.json](provenance.json)：原始收集文件的 SHA-256/大小、运行环境和脱敏说明；[artifact-hashes.json](artifact-hashes.json) 校验本目录内发布的文件及测试源码。

发布前已去除本机绝对路径、主机名和私有网络地址；指标、参数、状态、哈希及失败原因保留。报告只含生成输出的 SHA-256 指纹，不含生成文本或访问凭据。
