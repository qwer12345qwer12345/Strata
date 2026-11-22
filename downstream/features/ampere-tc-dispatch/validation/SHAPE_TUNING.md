# 新形状表的验收范围

本次独立GPU扫描和ABBA复测见 shape-summary.json、shape-sweep.jsonl、shape-abba.jsonl。8列主体kernel的下降约为gate/up20.7%、down32.4%、output20.4%；1列FC下降49.3%。这些数字是冷权重CUDA graph kernel时间，不是整段模型吞吐。

8256条scalar比较均逐位一致。新增真实Q4 pack固定输入检查，在旧/新/新/旧四进程中，混合宽窄行、跨ring边界、修改spec后的最终normalized hidden rows、token和概率散列一致。原有QKV TC两种表都保留，TC本身不逐位等于scalar；CPU参考40例通过。

桌面有限内存模型检查的shape8与wide12发生输出分叉；wide16长代码和说明输出与旧参考相同。新旧表固定输入相同，不足以证明全模型每次请求完全确定或已排除其他数值/时序问题。没有把这轮带专家文件读取的吞吐做为kernel优劣依据，未自动替换日常旧表。下一次无头启动跑完整驻留、零文件读取的长输出回文对比。

复核修正了QKV合成权重的分段尺度偏移，并独立重跑全部QKV的1..8列、43变体和ABBA。发布数据已替换这部分无效旧样本，其余7种形状没有这个布局问题。
