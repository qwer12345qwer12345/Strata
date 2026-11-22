# DFlash2 宽 lookup 与无头对比

在 c6f6427 / engine 0.1.65 上提供可关闭的单卡实验。DFlash2 神经草稿仍是 anchor + 7 masks、固定 8 行；扩展的是主模型验证的 prompt lookup 窗口，不是增加神经草稿长度。

## 接入方式

`STRATA_DF2_WIDE_CONTEXT=1` 与 `--lookup-max-t 12` 或 `16` 一起使用。显式指定 `--lookup-sizes` 控制候选宽度。未启用时，1..8 行仍调用原有 `draft()`；默认配置没有开启宽窗口。

`draft_extended()` 把长窗口前面的已提交行按最多8行送入现有 `context()`，把最后一行在 drafter 自己的流上复制到已有 landing buffer，再重放独立的一行上下文图。草稿块、RoPE位置、ring标签、日志与 selector 逻辑保持原有关系。同步返回后恢复原来的图槽和 taps 指针；修改 spec 时同时释放这张额外图。

不增加模型权重副本或专家驻留要求，也不为短 lookup 新建大批量 prompt context arena。额外 CUDA graph 有驱动元数据开销。不能把“没有新增显式张量 arena”解读成绝对零字节显存开销。

本次接入用于单卡 CUDA serve / Chat Completions，CLI 同步草稿调用也接入了桥接。没有扩展多卡 taps landing、batch slot 的异步 launch/finish 或 DFlash2 主体行数；没有验证其他后端。

## 手动无头对比

正常supervisor不执行基准、不读取旧请求队列。显式命令 `sudo ~/.local/bin/strata-headless-compare` 在无头环境暂停服务，运行比较后恢复日常服务；`strata-headless-compare --plan` 只显示计划。源码入口为 `terminal/l2/manual_compare.py`，运行时以模型用户和无限memlock执行，不以root运行模型。

默认50组，包含旧/新日常组合、新旧形状、QKV TC与DP4A、spec6/4/3、lookup上限8/12/16、Dense短窗TC、lookup草稿延后与单行/成对行。关键交互用二维参数矩阵，正反顺序平衡冷热与先后；完整定义见 [日常配置与对比计划](../../DAILY_CONFIGURATION.md)。每组3072-token代码和1536-token长上下文输出，默认三小时预算，测试不会自动修改日常配置。

保留完整驻留、零专家文件读取确认、内存/swap限制、流式与窗口延迟、输出散列及接受率报告。报告记录实际是否使用>8行，不能将“打开开关”当作已测到宽行收益。依赖legacy-session-archive提供统一runtime selection与正常supervisor，engine桥接本身仍是独立opt-in补丁。

## 验证及边界

- `dflash2_wide_test PACK`：9/12/16宽行与1/4/8窄行交替，跨2064行ring边界，修改spec后重建图，检查草稿、概率、ring与标签逐位一致。参考是显式分块context加最后一行原有draft。拒绝未启用/非法宽度。
- 真实Q4 pack + 固定合成head/taps，旧/新/新/旧四进程的8行最终hidden state、草稿、概率散列全部一致，见 [真实pack记录](validation/real-pack-parity.json)。这不是全模型质量基准。
- 真实Chat Completions完成四组、每组1024-token代码与384-token说明输出。12上限实际触发10行窗口；16上限本次没有使用>8行，不能称为API的T16覆盖。C++桥接测试单独覆盖16行。见 [模型记录](validation/model-check.json)。
- 桌面模型检查有专家文件读取，不能据此判定内核或无头吞吐优劣。shape8与wide12输出发生分叉，来源尚未隔离；wide16与参考的长输出散列相同。不能把分叉直接叫作质量降低，也不能声称已完成质量等价验收。
- ring恢复附加检查发现：开启上游批量context时，关闭新形状表也不逐位一致；`STRATA_DF2_CTX_BATCH=0` 则通过。批量context的TC与窄行重建不是同一累加路径。桥接只使用窄行context，未更改生产prompt批处理默认，也未声称解决上游所有ring精度问题。原始对照保存在 validation。

本机按用户选择采用理论优先日常组合，明确标注未经无头组合验证；测试改为显式命令触发，详见统一配置文档。引擎二进制SHA256：`85a8b3084262d90b671ba5442bab6a0885bfb7faf34cc76e775b0f12ab4f7569`。编译和GPU测试依次执行，未切换桌面或网络。
