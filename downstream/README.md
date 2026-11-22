# Strata 下游补丁仓库（Downstream Feature Packages）

本轮版本、补丁及硬件验证见[0.2.19 核查记录](validation/2026-11-23-review.md)。识图功能的新结构采用[临时借用专家显存](design/GYRO_VISION_VRAM_LEASE.md)，已实现为[vision-expert-lease](features/vision-expert-lease/README.md)，完成真实图片受限预算API检查；完整驻留无头验收待手动执行。

本分支是 Strata 的**下游功能包管理层**。它不维护一份长期漂移的完整源码分叉，而是把每项下游改动拆成可单独应用、验证、测试、启停和退役的补丁包。

- 上游源码基线：`Niko1221/Strata:main`
- 原下游管理分支：`downstream/main`
- 当前已验证补丁与旧存档移植：`audit/0219-patch-validation`
- 当前固定基线：`c6f6427d9de9016fd92d13e29740f65625261df7`
- 当前功能栈指纹：见 `stack.json` 的 `patched_tree`

## 1. 项目补丁思路

### 1.1 不把下游实现直接堆进上游主分支

`main` 用来跟踪上游基线；`downstream/main` 只维护功能包、补丁清单、测试项目、设计说明和维护脚本。准备开发或测试时，脚本会从固定的上游 commit 创建一个独立 worktree，再按顺序应用所选补丁。

这样可以避免：

- 下游功能长期混在一条完整分叉里，难以判断每项改动的来源；
- 上游同步时把互不相关的功能绑在一起解决冲突；
- 测试代码、生成文件和生产实现混成一个无法复用的大补丁；
- 为了启用一个功能，不得不同时引入其他尚未准备好的功能。

### 1.2 一个功能一个 Feature Package

每个目录是一个独立交付单元：

```text
downstream/features/<feature-id>/
  README.md              # 行为、开关、兼容范围和限制
  feature.json           # 基线、依赖、patch SHA256、独立 tree SHA
  patches/               # 对上游源码的有序 unified diff
  tests/
    test.json            # 构建目标、CTest、Python 测试和 overlay 配置
    README.md
    overlay/             # 仅供测试使用的源码，不进入生产补丁
  validation/README.md   # 已有证据、未执行的测试和硬件验收要求
  design/                # 设计/迁移计划（如有）
```

当前功能包：

| Feature ID | 内容 | 默认状态 |
|---|---|---|
| `tq-cpu` | TQ CPU AVX-VNNI pair-dot | 关闭，环境变量 opt-in |
| `tq-ampere` | Ampere TQ Gate/Up 配对 CUDA Kernel | 关闭，环境变量 opt-in |
| `qsa-imma` | K8V4 Q*K split-INT8 IMMA | 关闭，环境变量 opt-in |
| `chatml-tool-guard` | Chat Completions 角色边界与完整工具调用发布 | 关闭，配置 `chatml_tool_guard=true` |
| `legacy-session-archive` | 完整旧 PR5500 V2/V3、Linux 文件后备停放、自动 Chat Completions 保存与正常退出等待（单一大补丁） | 活动；由 managed_server 启用 V2/V3 与同步保存 |
| `ple-bf16-compact` | Direct BF16 PLE 原始行上传、GPU 无损展开、独立复制流 | opt-in，5120 bytes/token |
| `ampere-tc-dispatch` | SM86 DFlash2 按形状选 scalar/TC，部分 Dense 短窗口 INT8 TC | opt-in，保留9行全局门槛 |
| `dflash2-wide-lookup` | 10/12/16行lookup与宽窗口桥接、手动对比入口 | opt-in |
| `vision-expert-lease` | 图像编码借出专家显存、结束后按原槽恢复 | opt-in，本机无头启用 |

`live-memory`、`runtime-kv-storage`、`shutdown-persistent-cache` 已从默认活动栈移除，原目录保留作参考，不能与完整存档包叠加使用。

### 1.3 Patch 只保存生产改动

每个 patch 应尽可能只包含该功能必需的生产源码和构建接线：

- 运行时代码；
- 必需的头文件；
- 必需的 CMake target / test registration；
- 必需的设置项解析入口。

以下内容通常不进入生产 patch。`legacy-session-archive` 按用户要求整体交付，包含其原有回归测试、控制器和说明，作为例外：

- 测试专用源文件：放在本功能的 `tests/overlay/`；
- 生成的 Settings 文档和注册表：由上游 `tools/settings_register.py` 生成；
- 设计说明、验收记录和迁移计划：放在 Feature Package 文档目录；
- 与功能无关的格式化、重命名或顺手重构。

测试脚本会在临时阶段把 overlay 文件复制到准备好的源码 worktree，测试完成后恢复原文件。不会把测试专用实现永久写入生产源码。

### 1.4 可复现和失败即停止

每个功能的 `feature.json` 记录：

- 精确的 `base_commit`；
- 按顺序应用的 patch 路径和 SHA256；
- 单独应用该功能后的 `patched_tree`；
- 功能依赖和启用方式。

`stack.json` 记录功能应用顺序、完整补丁栈的 `patched_tree` 和上游地址。顺序是有意义的：多个功能可能修改同一份 CMake 文件，但每个 patch 必须在它声明的基线和顺序下精确应用。

校验不只检查 patch 能否 apply，还检查 patch 哈希、独立功能树和完整功能栈树。任一检查不一致就失败，不允许静默跳过冲突或继续生成一个不确定的工作区。

## 2. 分支职责

| 分支 | 职责 |
|---|---|
| `main` | 上游源码基线，不放下游功能包 |
| `downstream/main` | 原始功能包管理层，当前核查修复见下一行 |
| `audit/0219-patch-validation` | 已核查的补丁、完整旧存档移植、测试与验收记录 |
| `feature/live-memory-cuda` | 保留的早期 Live Memory 研究/实现参考 |
| `ampere-gyro-latency` | 保留的 Ampere 基准、测量记录和早期 L2 实现参考 |

分支审计和可清理候选见 [BRANCH_AUDIT.md](BRANCH_AUDIT.md)。没有迁移的历史实现不能仅因新包已建立就直接删除。

## 3. 快速使用

所有命令从仓库根目录执行。需要 Git、Python 3.11、CMake、C++ 工具链；CUDA 功能还需要匹配的 CUDA Toolkit、驱动和目标 GPU。

### 3.1 获取管理分支

```bash
git fetch origin
git switch --track origin/audit/0219-patch-validation
```

如果本地已有同名分支：

```bash
git switch audit/0219-patch-validation
git pull --ff-only
```

### 3.2 校验补丁包

```bash
python downstream/scripts/verify.py
```

该命令校验所有功能包的元数据、patch SHA256、独立 tree SHA 和完整 stack tree SHA。它不编译代码，也不代表硬件测试通过。

### 3.3 准备源码工作区

准备全部功能：

```bash
python downstream/scripts/prepare.py --all
```

只准备一个功能：

```bash
python downstream/scripts/prepare.py --feature legacy-session-archive
```

同时准备多个功能：

```bash
python downstream/scripts/prepare.py --features tq-cpu,tq-ampere,qsa-imma
```

指定工作区位置：

```bash
```

默认目标是仓库同级的 `Strata-prepared`。如果目标目录已经存在，脚本会停止；不会自动删除、重置或覆盖用户目录。准备前会先验证完整补丁栈，再创建固定上游 commit 的 detached worktree、应用 patch，并按需重新生成 Settings 文件。

### 3.4 运行功能测试

先准备对应功能，再运行测试。示例：

```bash
# TQ CPU：x86 native experts
python downstream/scripts/test.py --feature tq-cpu \
  --source ../Strata-prepared --build ../Strata-build-tq-cpu

# Ampere TQ：CUDA / SM86
python downstream/scripts/test.py --feature tq-ampere \
  --source ../Strata-prepared --build ../Strata-build-tq-ampere

# K8V4 Q*K IMMA：CUDA / SM86
python downstream/scripts/test.py --feature qsa-imma \
  --source ../Strata-prepared --build ../Strata-build-qsa-imma


# 完整存档包
python downstream/scripts/test.py --feature legacy-session-archive \
  --source ../Strata-prepared --build ../Strata-build-archive

```

`test.py` 会确认源码位于固定基线、确认该功能 patch 已应用，再临时覆盖测试源码、执行 Settings 生成/检查、Python 测试、CMake build 和 CTest，最后恢复 overlay 和生成文件。每个功能使用独立 build 目录，避免不同 CMake 配置互相污染。

硬件约束：

- `tq-cpu` 的 AVX-VNNI 快速路径需要支持 AVX-VNNI 的 x86 CPU；其他 CPU 只能验证 fallback。
- `tq-ampere` 和 `qsa-imma` 的优化路径需要 CUDA SM86 设备；不符合条件的设备不能作为优化路径通过的证据。
- `legacy-session-archive --with-cuda` 运行 CUDA 状态往返测试；Linux 文件后备需要可写且有空间的本地目录。
- CPU-only policy 测试通过，不代表 CUDA 路径、真实模型或性能目标通过。

### 3.5 手动应用补丁

如果已有一个干净的上游 worktree：

```bash
python downstream/scripts/apply.py \
  --source ../Strata-clean \
  --features tq-cpu,tq-ampere
```

源码必须位于 `stack.json` 指定的精确 `base_commit`，并且 worktree 必须干净。脚本会检查每个 patch 的 SHA256、先执行 `git apply --check`，然后才应用。

## 4. 如何新增或维护一个功能包

### 4.1 从固定基线开始

```bash
BASE=c6f6427d9de9016fd92d13e29740f65625261df7
git worktree add --detach ../Strata-feature-work "$BASE"
```

在独立 worktree 中开发功能，不要直接在 `main` 或 `downstream/main` 上修改上游生产源码。

### 4.2 分离生产 patch 和测试 overlay

- 生产 patch 只选择该功能需要的源码/构建文件；
- 新增或修改的测试源码放入 `features/<id>/tests/overlay/<原源码相对路径>`；
- 在 `tests/test.json` 中声明 overlay、CMake 参数、构建目标和 CTest regex；
- 生成文件不进入 patch；
- 功能开关默认关闭，除非已有明确的兼容性和验收依据。

示例（从管理仓库根目录执行，路径按实际改动调整）：

```bash
git -C ../Strata-feature-work diff --binary "$BASE" -- \
  engine/src/core/example.cpp engine/include/strata/core/example.hpp \
  > downstream/features/example/patches/0001-example.patch
sha256sum downstream/features/example/patches/0001-example.patch
```

将实际 SHA256 写入 `feature.json`。如果一个功能需要多个 patch，按依赖顺序列入 `patches` 数组。

### 4.3 更新 tree 指纹

独立 tree 指纹是从指定 base commit 的临时 Git index 开始，按顺序执行所有 patch 的 `git apply --cached`，再执行 `git write-tree` 得到的 Git tree SHA。完整 stack 指纹则按 `stack.json` 的 `feature_order` 顺序应用所有功能的 patch 后计算。

将结果写回功能的 `feature.json.patched_tree` 和 `stack.json.patched_tree`。不要手工猜测或复制其他基线的 tree SHA。

### 4.4 完成前的检查

```bash
python -m compileall -q downstream/scripts
python downstream/scripts/verify.py
python downstream/scripts/prepare.py --feature example --target ../Strata-example-prepared
python downstream/scripts/test.py --feature example \
  --source ../Strata-example-prepared --build ../Strata-build-example
```

完成后提交 Feature Package 的补丁、测试、说明、验证记录和元数据。不要把准备好的临时 worktree 或 build 目录提交到管理分支。

## 5. 上游升级流程

上游升级不是简单地把新 `main` merge 进下游分支。推荐顺序：

1. 选定并记录新的上游 commit，更新 `stack.json.base_commit`；
2. 从新基线建立干净 worktree；
3. 按 `feature_order` 逐个重放功能 patch；冲突要按功能单独解决，不能用整分支 merge 掩盖；
4. 对每个功能重新生成 patch、SHA256 和独立 tree SHA；
5. 更新完整 stack tree SHA；
6. 对每个包运行测试；CUDA/VMM/AVX-VNNI 路径必须在相应硬件上验收；
7. 运行 CI，审阅所有功能差异后再提交新的管理版本。

某个功能暂时无法迁移时，应将它标记为待移植/不兼容并保留原始资料，不得悄悄从 stack 中移除或假装已迁移。

## 6. 当前验证边界

活动栈目前包含三个原有计算优化包、一个完整存档包、一个工具边界保护包，以及 BF16 PLE、SM86 分派、opt-in DFlash2 宽 lookup 和视觉专家显存借还四项新包，共九项。独立包与完整栈的 Git tree、SHA256 均校验；硬件与真实模型验收范围见各包 validation。存档迁移依据用户选定的旧方案，未启用 Live Memory 或未完成的 Runtime KV Storage。CI 的补丁校验不能替代真实 GPU、跨重启恢复与无头长上下文峰值验收。

当前本机日常配置与手动无头对比命令见 [DAILY_CONFIGURATION.md](DAILY_CONFIGURATION.md)。正常启动不再执行基准或请求队列。

关键参数交互与识图当前边界见 [GLOBAL_REVIEW_AND_VISION.md](GLOBAL_REVIEW_AND_VISION.md)。

当前显式开启ADAPT_NOWAIT与官方tail skip7，依据和手动测试组见 [NOWAIT_AND_TAIL_SKIP.md](NOWAIT_AND_TAIL_SKIP.md)。
