# SM86 的 DFlash2 与 Dense 分派

针对 RTX 3080 / SM86 的可选优化；保留上游默认。依据真实 GPU 上、权重循环超过 L2 容量的微测试，不以专家落盘或系统 RAM 竞争作为 kernel 性能依据。

## DFlash2

- `STRATA_DF2_AMPERE_TUNING=1`：借用上游 AMD Q4_0 的精确整数点积 kernel 变体，按已测矩阵形状使用 unroll=1 的 variant 26/31。未知形状仍使用原 auto 表，显式 `STRATA_DF2_GEMM4` 和 benchmark forced variant 保留原优先级。
- `STRATA_DF2_GEMM_TC=1`：保持上游的全形状 TC opt-in。
- `STRATA_DF2_GEMM_TC=2`：新增本机测量模式。与 Ampere tuning 一起，只对 2560 输入、[4096,1024,1024] 三矩阵 QKV 合并投影使用 TC。其他矩阵使用标量/DP4A 的形状选择，避免 KV、output、FC 上全面 TC 的回退。
- 两项新选择均限制 SM86，Q4_0；未对 Q8 drafter 宣称收益。Tensor Core 结果与原浮点累加不逐位相同，数值误差和真实接受率分开验收；scalar variant 的检查也独立保留。

## Dense

`STRATA_DENSE_I8_AMPERE_SHAPES=1` 仅把已有 native Dense INT8 TC 内核用于测得更快的短窗口：K=2560/N=10240，Q4_K/Q5_K 的 T=6..8，以及 Q6_K 的 T=8。其余短窗口保留原路径，全局 `STRATA_DENSE_I8_TC_MIN_T` 默认仍为9，`STRATA_DENSE_I8_TC=0` 仍可整体关闭。

这些是 dense projections，不是 Gyro TQ 专家；本机 Gyro-S 包的36个 `attn_qkv` 是 Q5_K / [2560,10240]，因此存在真实命中对象。微测试的误差约 1e-7 相对 RMS，不宣称逐位相同。

## 接受率与9行

当前 DFlash2 固定 anchor+7 masks，最多8行；其 Transformer 主体每轮仍计算8行。spec=6 主要限制送入主模型验证的窗口，以及 drafter head/selector 的输出数量；confidence=.8 再筛短验证窗口，并不会把 drafter 主体的8行计算同比缩短。本机 engine 构建也为8行。历史约90%的 accepted/offered 只计算已经过筛选的草稿；每轮实际提交远低于6，不能据此外推更长链的接受概率。

上游已有独立的最高16行 lookup 路径（narrow 构建也有），但默认明确排除 DFlash2；`Dflash2Drafter::draft_launch` 仍拒绝一次超过8个新上下文 token，直接设置 `--lookup-max-t 12` 并不构成正确接入。宽 lookup 值得研究，但需要让已接受的长片段分块进入 drafter context，核对 ring/taps/position 之后再启用。≥9行的 TC 也可供更宽的 MTP 或多请求合批研究；不能靠调大当前 DFlash2 的 spec 直接得到。优先优化现有宽度内的 GPU 时延，不为触发 TC 额外填充无效草稿。

## 复现与限制

bench 目录包含从上游 benchmark 适配的源文件与运行器；验证数据见 validation。所有 CUDA 编译结束后才开始测量。独立 kernel 测量不代表无头端到端的同比收益；桌面低专家预算的模型检查只判断接线、输出与接受率。

只在 SM86 CUDA 上构建和测试；HIP/SYCL 未验证，不发布它们的二进制或性能结论。

## 当前本机选择

已启用 `STRATA_DF2_AMPERE_TUNING=1` 与 `STRATA_DF2_GEMM_TC=2`；Dense候选保持 `STRATA_DENSE_I8_AMPERE_SHAPES=0`。该候选有GPU形状收益，但主模型舍入变化导致长输出和lookup分布改变，需更充分的同机质量/延迟对照后再作为默认。没有全面打开TC或全局降低9行门槛。
