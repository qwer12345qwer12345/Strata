# 实测记录

- wide-final.txt：真实Q4 drafter，9/12/16行与1/4/8行交替，环绕、图失效、非法宽度和禁用开关；通过。
- real-pack-parity.json：旧/新/新/旧的固定输入最终hidden state、token/probability散列一致。
- ctest-columns.txt：旧表与新表的CPU参考各40例，两个CTest通过。
- ring-control.json 及对应日志：批量context打开时旧表/关闭优化也不逐位一致；narrow context模式恢复测试通过。保留失败记录，未当作新桥接的通过项。
- model-check.json：桌面受限专家预算的Chat Completions正确性/覆盖；包含专家文件读取、接受数、实际宽度、输出散列与窗口延迟聚合。原始令牌和文本未发布。T12 cap触发T10；T16 cap未触发>8，不夸大覆盖。

额外Python验收共20例（队列/流式与子进程回收6例、既有startup/controller14例）通过；改进窗口延迟聚合后重跑相关6例通过。完整8包可独立重放，SHA256/独立树/完整树验证通过。仅在SM86 CUDA构建；HIP、SYCL、跨重启的宽窗口恢复和正式256K无头峰值未实测。
