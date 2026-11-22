# 数值一致性与性能审计

2026-11-23。审计包括我们的活动补丁、实际部署配置，以及 Strata 官方当前 main。远端 `git ls-remote` 当日返回 `c6f6427d9de9016fd92d13e29740f65625261df7`，与本地固定的 engine0.1.65 / Terminal0.2.19 相同。官方源码是本节判断的直接依据；其注释/文档中的性能数字属于官方或报告者的测量，不是本机复测。本轮没有运行模型或GPU性能测试。

## 采用的评价标准

跨硬件相同计算不保证相同浮点结果；与旧内核逐位相同也不等于更接近模型厂的实现。相同数学计算的不同求和顺序、融合、CPU/GPU选择，不能仅因文本hash不同而淘汰。FMA甚至可能比拆开乘加更准确，见 [NVIDIA浮点说明](https://docs.nvidia.com/cuda/floating-point/index.html)。

优化仍须实现其数据和生命周期关系：专家数据存在、复制完成后才使用、KV与位置一致、没有数据竞争/非有限值、API和工具格式有效。量化或省略专家是更强的近似，单独说明实际任务质量；它们与改变FP32求和顺序不能混为一谈。

固定输入/固定专家/固定窗口的测试用于解释内核成本；自由生成测试用于检查真实吞吐、首内容延迟、长输出、神经和lookup各自接受数、工具有效性与任务成功。输出hash保留为定位信息，不是通用通过条件。先解决专家文件层干扰，再判断GPU内核收益。

## 官方保留的机会

| 路径 | 实际代码与关闭原因 | 对本机的判断 |
|---|---|---|
| `STRATA_EMB_REUSE_ACCOUNT=1` | `prefill.cpp:827`附近明确：bo实际复用emb，规划默认仍重复计费，以保持旧chunk/借槽和输出。开关只令规划使用已经复用的空间。官方3060 IQ3_XXS记录6400→6656 chunk、prompt约+3.9% | 最明确的“为旧数值保守计费”例子。优先候选，无需新内核。Flash-Next每8192-token chunk重复计入80MiB（8192×2560×4），不等于新增80MiB永久专家驻留。需比较实际chunk、借槽、prefill和归还后的专家文件读取；本轮未改日常开关 |
| Dense多列的非exact布局 | `native_mmvq.cu:1233`、`:1566`；`native_mmvq.hpp:28`。官方保留generic布局，改变跨warp归约；生产路径无调用setter，注释说速度未测。源码里的“faster”不能替代当日实测 | 有代码可复用，适合按本机Dense形状和列数扫描。不能直接全局置false：会绕过B6、IL表、部分Q8/Q6 packed/融合路径，可能失去比旧generic更快的新优化。应做形状分派，比较当前完整分派的赢家 |
| `STRATA_GDN_TC_NP=2` | `gdn_tc.cuh:158`、`kernels.cu:3233`。split TC默认NP3；NP2仅将直接通往输出的q.S、D.P用单次F16 hi×hi乘法，递归状态相关乘法保留三次补偿；NP1全面单通道 | 优先研究已有NP2，不因数值变化排除。本机99KiB共享内存可用split计划；需确认实际命中。少做MMA/加载不表示整个GDN同比提速；NP2也不是“所有乘法从3次变2次”。检查长prompt后的状态/输出质量；NP1另列更强近似；NP2输出变化也会沿后续层传播，不能保证未来状态完全相同 |
| `STRATA_HC_UPMIX=1` | `prefill.cpp:975/5144`、`kernels.cu:2759`，融合HC上投影和mix；CUDA仅T≥33，官方解释短块cuBLAS换核会有舍入差异。sm89/120/121默认、sm86未选入；并无本机证明 | 减少一次中间写读和launch，适合核对本机实际权重视图是否满足资格后测。不要把其他卡的约1–5%模型收益套到3080。与HC_Q8、FP16/BF16视图、残差低位补偿的接线必须逐项确认 |
| `STRATA_SELECT_SIMT=1/2` | `qsa_select.cu:2289`：旧卡FP32 tiled scorer因求和次序不同而opt-in；注释报告2080Ti scorer约4.5–6.8倍 | 是官方另一个可直接确认的数值保守例子。3080已走TF32 TC scorer，开此项不会替换当前路径；适合Turing/Pascal设备，不是本机可直接获得的增益 |
| `STRATA_PREFILL_EQUAL` | `generate.cpp:10820`区分编译；有MMQ分支保留opt-in且写明chunk改变舍入，无MMQ+FUSED的单卡分支已有带门槛equal默认 | 本机编译没有STRATA_PREFILL_MMQ、已有FUSED，不是漏开的优化。强行要求全部等分可在官方样本慢14–34%，原因是更多专家流送与KV搬运，不是精度问题 |

源码链接：[预填充规划/HC](https://github.com/Niko1221/Strata/blob/c6f6427d9de9016fd92d13e29740f65625261df7/engine/src/prefill/prefill.cpp)、[Dense多列分派](https://github.com/Niko1221/Strata/blob/c6f6427d9de9016fd92d13e29740f65625261df7/engine/src/kernels/cuda/native_mmvq.cu)、[GDN拆分计算](https://github.com/Niko1221/Strata/blob/c6f6427d9de9016fd92d13e29740f65625261df7/engine/src/prefill/gdn_tc.cuh)、[官方speed settings](https://github.com/Niko1221/Strata/blob/c6f6427d9de9016fd92d13e29740f65625261df7/docs/DETAILS.md#speed-settings)。

## K8V4必须沿我们的补丁继续研究

已有 `qsa-imma` 不是缺失项。`STRATA_K8V4_QK_IMMA=1` 已启用，SM86预填充走 `launch<3,false,true>`：query拆成三部分INT8，Q×K用IMMA，V保持Q4_0存储、加载解量化，其后继续现有P×V路径。这里的split-INT8表示数值表示拆分，不表示将KV序列分给多个CTA。迁移时数值CTest已通过，尚无这一项的独立延迟结论。

官方 `QSA_SPLIT_ATTN` 的decode入口是另一套内核，`qsa_decode_attn_split`仍要求K/V均INT8；现有补丁改的是prompt入口，`qsa_decode_attn_batch`仍直接调用chunk/merge实现，未接入我们的IMMA。这说明缺少解码接线，不说明K8V4不能做TC或应换成K8V8。

下一项有价值的代码候选：复用我们的三部分query IMMA与Q4 V加载，把decode的选中KV切分给多个CTA，各自产生max/sum/加权V，再用已有log-sum-exp合并；分派按真实选中cells、query数和70SM的占用测量，不只看名义总上下文。当前decode本来已有chunk/merge，并非从完全串行变并行；收益须来自TC吞吐、分块/复用和启动代价的改进。较短attention可能被额外merge抵消。维持K8V4存储和既有scratch上限，不为这项实验增加一份K8V8。

这是需要新增实现和测量的方案，尚未打出该解码扩展补丁；不能把已有prefill IMMA通过写成decode split通过。

## 看起来类似但不应当盲开

- `GDN_TC`已由官方Linux CUDA默认启用，并有适合99KiB卡的split fallback；不能仅看旧文档“122KiB”就认为3080用不了。反而非TC `GDN_CHUNKED=2`强行绕过128SM门槛，官方预期多wave更慢；不是为一致性关闭。
- `PF_FUSED_F16`/`PF_FUSED_NT=256`只覆盖原有部分IQ格式。启用后`d.occ_nt>0`会使当前Gyro的trellis fused资格失败（`moe_fused_iq.cu:3444`）。F16与WS表中的其他设备收益不适用于本机Gyro；WS还超过SM86共享内存。应针对TQ另写变体，不能拿全局开关冒充叠加收益。
- `GR_V3`在A4000/3060有负收益记录，且我们的Q8 HC路径还有独立资格；未作为本机优先项不是因为逐位比较失败。
- `HC_PERSIST`现有实现明确拒绝Q8 HC、HC_REQ8及QFUSE输出；本机已有Q8 HC，不是打开变量就生效。需要另做INT8版本才值得排期。
- `ADAPT_FETCH=2`可提高小缓存命中，官方甚至已在小显存Blackwell默认选用非逐位一致版本；但当前resident complement模式资格明确拒绝。须解决专家所有权交换，不能绕过检查直接开。普通ADAPT_OVERLAP与现有NOWAIT/lag互斥，属于替代调度方案。
- `STAGE_PIN`官方记录过实际解码损坏；eviction同步、恢复门禁和缓存有效性检查保护数据关系，不属于为统一数值付出的成本。
- CUDA关键路径已使用fast_math；`CPU_IQ_ALL_EXACT1`虽叫EXACT，本身在SM86性能表中是更快实现，不应按名字关闭。`BF16X2`是多做计算增加激活精度，当前没有强开。

官方并非一律拒绝数值差异。默认的混合CPU/GPU prefill、部分架构的GDN/selector、路由分担已容许变化；需逐项识别关闭原因。其他模型的SYCL oneMKL固定序内核等也在官方历史中出现，但不属于当前Flash-Next CUDA执行路径，其几乎无速度差异的报告不能推导本机有大收益。

## 我们的配置修正与现有选择

NOWAIT1/lag2、tail skip7、Dense短窗TC、prior0.5均已选入。新DP4A表下关闭DFlash2短窗QKV TC是本机测得该形状更快，不能归为数值保守。DF2_CTX_BATCH保留默认，未为通过逐位ring检查而关闭。

本轮发现并修正一个实际接线问题：旧profile没有`user_args`/`args_written`时，runtime selection只改`args`，官方drafting.apply会把选定spec4改回默认6。现在将选择器的明确覆盖项写入`user_args`，沿用官方参数优先级。新增回归覆盖旧profile；最终engine_args只生成命令、不启动模型，确认spec4、draft-conf0.8、lookup16。之前受限预算识图日志中确实为spec6；其图像生命周期结果仍有效，不视为spec4组合性能结果。

本轮没有为未测候选更改运行开关、执行模型基准或修改正常启动行为。建议研究次序：embedding实际计费 → GDN NP2 → K8V4 decode TC/分块扩展 → Dense按形状比较非exact布局与当前分派 → HC_UPMIX资格和延迟。使用既有显式无头测试方式，不能把这份源码审计当作性能提升已验证。

后续执行更新：用户已选择启用Embedding实际计费和GDN NP2。无需推理代码改动的候选已加入114组手动矩阵，其中新增44组可用`--group numeric`单独选择；待代码项见[NEXT_OPTIMIZATION_PLAN.md](NEXT_OPTIMIZATION_PLAN.md)。这更新了前文“本轮未改开关”的历史状态，仍无这些候选的真实性能结果。resident overlap已有`STRATA_ADAPT_OVERLAP_RESIDENT=1`，本次作为NOWAIT替代方案列测，不需新增所有权实现；ADAPT_FETCH仍需代码。
