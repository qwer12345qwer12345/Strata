# Strata Downstream Feature Packages

本目录是 Strata 下游功能补丁包仓库。上游源码本身不承载下游实现；每项功能的说明、补丁、测试和验证记录都放在自己的目录中。

## 分支约定

- `main`：只跟踪 `Niko1221/Strata:main`，不提交下游改动。
- `downstream/main`：维护本目录中的 Feature Packages、补丁清单和自动化脚本。
- 历史功能分支保留为迁移参考；不要因为补丁包已建立就删除尚未迁移的旧分支。

## 当前基线

上游基线：`6e38f588a646c0ee6948a74afbe670eb75896106`

当前迁移的功能包：

| Feature | 内容 |
|---|---|
| `tq-cpu` | TQ AVX-VNNI CPU 路径 |
| `tq-ampere` | Ampere TQ Gate/Up 配对 Kernel |
| `qsa-imma` | K8V4 Q*K split-INT8 IMMA |
| `l2-session` | Persistent Conversation Spill |
| `live-memory` | Live Memory 初始 VMM commit 与容量策略 |

## 使用

在仓库根目录执行：

```bash
# 验证所有 Feature Package、Patch SHA256 和源码树指纹
python downstream/scripts/verify.py

# 创建独立的上游源码 worktree，并应用所有功能补丁
python downstream/scripts/prepare.py --all

# 只准备一个功能
python downstream/scripts/prepare.py --feature live-memory

# 在准备好的源码工作区运行功能测试
python downstream/scripts/test.py --feature live-memory \
  --source ../Strata-prepared --build ../Strata-build-live-memory

# 需要 CUDA/VMM 测试时
python downstream/scripts/test.py --feature live-memory \
  --source ../Strata-prepared --build ../Strata-build-live-memory --with-cuda
```

`prepare.py` 默认创建仓库同级的 `Strata-prepared` worktree。如果目标目录已存在，脚本会停止，不会自动删除或重置它。

## 设计规则

1. 每个可独立启停、测试和退役的功能对应一个目录。
2. 功能补丁只修改上游源码；说明和测试项目留在功能目录内。
3. 测试源文件通过临时 overlay 放入准备好的 worktree，测试结束后恢复。
4. Settings 注册表和文档是生成文件，不进入功能补丁；准备/测试时由上游工具重新生成。
5. 每个功能包绑定精确的上游 commit、补丁 SHA256 和独立应用后的 Git tree SHA。
6. 任一补丁无法应用、哈希不匹配或 tree 不匹配时必须失败，不做静默跳过。
7. 本次迁移只处理 PR #1～#3；更早的分支保留，不视为已迁移。

## 验证状态

这些功能包保存的是 PR #1～#3 中已有的实现和测试。此次结构迁移不等于完成了编译或硬件验收。各功能的验证限制见对应的 `validation/README.md`。
