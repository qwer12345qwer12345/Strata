# 实测记录

- wide-final.txt：真实Q4 drafter，9/12/16行与1/4/8行交替，环绕、图失效、非法宽度和禁用开关；通过。
- real-pack-parity.json：旧/新/新/旧的固定输入最终hidden state、token/probability散列一致。
- ctest-columns.txt：旧表与新表的CPU参考各40例，两个CTest通过。
- ring-control.json 及对应日志：批量context打开时旧表/关闭优化也不逐位一致；narrow context模式恢复测试通过。保留失败记录，未当作新桥接的通过项。
- model-check.json：桌面受限专家预算的Chat Completions正确性/覆盖；包含专家文件读取、接受数、实际宽度、输出散列与窗口延迟聚合。原始令牌和文本未发布。T12 cap触发T10；T16 cap未触发>8，不夸大覆盖。

额外Python验收共20例（队列/流式与子进程回收6例、既有startup/controller14例）通过；改进窗口延迟聚合后重跑相关6例通过。完整8包可独立重放，SHA256/独立树/完整树验证通过。仅在SM86 CUDA构建；HIP、SYCL、跨重启的宽窗口恢复和正式256K无头峰值未实测。

## 当前配置与控制流程复核

正常启动不再运行固定基准或消费请求队列。手动入口与50组交互矩阵已安装，日常按用户选择采用理论优先组合并标注NOT TESTED。24个控制流程测试用实际服务使用的Python3.12通过，覆盖正常启动零generation请求、手动plan零服务操作、桌面active拒绝、模拟试验失败后恢复服务、流式与存档drain等。没有以模型或GPU性能测试占用本次使用时段，没有重新编译引擎。当前部署状态见deployment.json；原始GPU/模型记录仍为此前测得的数据，不能当作新日常组合的验收结果。
