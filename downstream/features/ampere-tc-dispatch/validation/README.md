# SM86 计算与接受率核对

硬件与工具链同 BF16 PLE 包。编译与GPU测试顺序执行，保持桌面和网络。所有源路径、主机名、客户端消息与令牌已从发布数据中排除。

## 独立GPU内核

- [稳定时钟ABBA原始记录](df2-settled-raw.json)与[汇总](df2-settled-summary.json)：32个独立进程，列数1..8，每列off/on/on/off。每形状先预热1秒，循环约128 MiB不同权重，超过3080 L2。没有CPU专家和系统RAM换页。
- 全开 `STRATA_DF2_GEMM_TC=1` 有得有失；6列时 QKV 22.65→18.70μs，KV 9.20→13.10μs，output17.00→20.15μs。最终mode2只取QKV的TC。
- [Dense108组数据](dense-compare.jsonl)与[汇总](dense-summary.json)：4个相同形状/列数的交错计时对，超过L2的权重循环。T4没有>5%收益的组，T6有2/12组，T8有3/12组；不能全面降低9行门槛。Q5_K / [2560,10240] 的T6为59.57→52.57μs，T8为82.99→53.22μs。
- [真实GGUF dense形状](model-dense-shapes.json)确认目标Q5_K attn_qkv投影存在36个。Dense TC误差为约1e-7相对RMS，与原路径不逐位相同。
- `dflash2_q4_parity_ampere` 通过CPU double参考检查，覆盖新shape模式及scalar选择。`dense_i8_tc_test` 的decode与prefill数值校验通过。不是模型质量评测。

## 真实模型与接受率

[三组模型记录](model-check.json)，[按来源拆分的窗口与分派记录](acceptance-and-dispatch.json)。prior=.5、DFlash2 spec6/conf=.8，5600个GPU专家槽、6 GiB专家RAM、16K上下文，冻结自适应与PCIe分担。代码输出1024 token，上下文6027 token后输出256 token。磁盘专家读入存在，整段速度不用于证明GPU kernel优劣。

- 参考/compact代码输出相同。神经草稿均89/102（87.25%）；lookup分别726/730和729/734（约99%）。合计接受率约97.9%，大量来自lookup，不是7个连续神经草稿都可靠。
- 同时启用全部新分派后，神经草稿695/792（87.75%），lookup59/76。代码输出不同，lookup占比大降，合计接受率86.87%。因此不能把合计下降说成DFlash2神经草稿全面变差，也不能把该次低吞吐解释为GPU kernel变慢。
- TC capture计数确认新Dense路径命中T6/7/8、36个Q5_K投影；上游默认参考组没有这条capture。
- 参考代码的151/208有效轮次使用lookup，其中149轮全收；存在更宽lookup研究价值。当前DFlash2的draft_launch仍限8个新上下文token，上游默认排除DFlash2宽lookup。没有直接打开或把强制补齐行当优化。

所有配置继续使用prior=.5。历史先前同机接受率见[历史汇总](historical-summary.json)；它使用IQ4_NL PLE/不同运行条件，不能据此单独归因BF16或新kernel的变化。

## 已知边界

这是有限的代码/说明文本检查，不是质量基准。没有宣称无头256K峰值、吞吐或一般模型质量已经通过。对输出内容不同的两组，性能数字同时受专家路由、lookup和文件读取影响。新增开关独立可关闭，使用详情及实际启用选择见包README与最终部署记录。

## 拆分后的选择

[只改drafter的补充记录](model-split.json)与[来源拆分](acceptance-split.json)：scalar-only 与 scalar+QKV TC 两组的48-token热身、1024-token代码输出散列均与参考一致。代码神经草稿89/102→89/101（87.25%→88.12%），lookup729/734→699/700（99.32%→99.86%）。lookup窗口分布随成本策略改变，仍不能用单次整段速度给kernel定胜负。

本机采用 BF16 compact、独立复制流、DFlash2 Ampere scalar形状表与QKV限定TC。新增的主模型Dense短窗口分派代码保留为候选，本机 `STRATA_DENSE_I8_AMPERE_SHAPES=0`。它在独立GPU上有收益并实际命中，但同时改变主模型舍入和后续lookup使用，当前样本不足以作为日常默认。全局9行门槛未降低。

服务仍在下次用户选择无头模式时启动，先跑既有开机基准；没有停止桌面或变更网络。引擎SHA256为 `926250931e754820796112e3b5627342b6aba2c99974414d92fbd27a421549a2`。正式256K驻留配置的吞吐与内存峰值仍待无头验收。

## 从其他设备借用的scalar变体

[稳定时钟的variant数据](df2-row-variants.json)与[原始输出](df2-row-variants-raw.json)：8列主干中，variant17→26的gate/up为45.30→39.90μs，output17.45→15.05μs；variant17→31的down为37.80→27.65μs、FC60.80→46.55μs、selector7.95→4.95μs。它们是同一整数点积的线程/展开分配差异，未增加权重副本或要求更多专家驻留。完整drafter主体固定计算8行，selector/context部分列数另行变化。
