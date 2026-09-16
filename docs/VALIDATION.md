# 代码交付与验证记录

交付日期：2026-09-16。项目目录 `F:/DetectFirst`，远端 `xingxiaoyang-888/DetectFirst`。以下记录来自本地实际执行，不代表 GPU 结果或研究精度。

用户最后明确要求把具体测试放到服务器。本地随后只完成代码整理和发布，不继续增加实验测试。

## 2026-09-16 接手后 R1–R3 修复

`codex/server-audit-evidence` 分支修复环境反传失败漏报、模型审核未执行配置精度、并行日志覆盖及下载收据历史覆盖。新增故障回归用例在 `test_server_audits.py` 和 `test_evidence_capture.py`。本轮仅做静态检查，未运行这些用例、pytest、smoke、真实模型或 GPU 实验；历史 69 项检查不能作为本次修复的通过证据。

最新执行与判据见 [S00/S01 修订安排](S00_S01_RUNBOOK.md)，本地静态日志与提交交接保存在 `reports/fix_R1_R3_20260916/`。服务器应对修复提交运行完整 `verify_release.py`，并实际执行 DINO FP32/BF16 精度审核；Linux/Bash 日志回归不可作为跳过项处理。

## 已执行

| 检查 | 实际结果与范围 |
|---|---|
| 干净环境安装 | Python 3.10、CPU PyTorch 2.5.1，基础、开发与生成依赖均安装成功；pip check 无依赖冲突 |
| 静态检查 | Ruff、格式检查、关键模块 mypy、Python 编译通过 |
| 单元与集成 | 最近一次完整执行 **69 passed**；6 条 warning 均为官方 DINO 提示 xFormers 被禁用／不可用 |
| 数学与指标 | 角度已知答案、FP64 gradcheck、空区域、面积与条件平均、InfoNCE/MMD 参考、AP/AU-PRO 边界、校准 ties 与按图权重 |
| 训练与数据 | 内部全部方法反传；P/DRO/memory 连续与恢复一致；拒绝测试角色与未选支持；原图预测可重算 |
| 机制实验 | 冻结表示重训头只改变分类器；正确与打乱配对输入及曝光相同、各自使用自身 mask |
| 六视图 | 输入不变量通过；保存的 alpha 与原语可逐像素重建全部 PNG |
| 官方 DINO 架构 | 固定官方源码、随机初始化权重、CPU 56×56，前六层冻结及后六层／decoder 辅助梯度通过；不是预训练效果验证 |
| FLUX 接口 | Diffusers 0.35.1 + Transformers 4.56.2 可导入 FluxFillPipeline，代码使用的参数存在；没有加载 FLUX 权重或实际生图 |
| 命令行闭环 | 人工夹具上的训练→校准→评估→保存预测重算通过；不作为论文结果 |

完整输出随仓库保存在 [验证材料目录](validation/20260916/)。其中 argv 中的 F 盘路径属于原始执行记录，不是服务器配置依赖。实际测试覆盖的源码指纹见 `verification.json`。

最后整理补充了原生／最终交叉图的编辑足迹分类及准入记录一致性，并修正根目录数据的 Git 忽略范围；遵照用户要求，没有在这些整理后重新执行完整回归。服务器应对最终 commit 运行下面的验证命令，不把先前的测试记录当成对后续任何改动的自动担保。

## 服务器仍需完成

- 真实预训练 DINO 的 GPU／BF16 前向、梯度、显存、实际 200 步与恢复验证。
- FLUX 权重加载、生成方向、图像质量、20 次以上测速、人工 ROI/M/G 复核。
- 真实数据全部角色、近重复、接受池与固定协议审查。
- E_SSN/E_AVFM/E_SEAS/E_O2MAG 的具体官方适配器接通及公平复现。当前只提供执行与监督预算准入接口，未冒充完整复现。
- 独立真实测试、研究效果、置信区间和论文性能主张。

## 重跑入口

```bash
python scripts/verify_release.py --output-dir reports/server_release_check --include-dino --include-generation
```

需要先下载固定 DINO 源码，并安装 generation 依赖；该命令的 DINO 部分仍使用随机初始化架构夹具。实际预训练 GPU 检查使用 `audit_model.py`，后续步骤见 [服务器执行文档](SERVER_START.md)。每次选新输出目录，保留原记录。

数据、模型、虚拟环境、生成图和 checkpoint 未纳入 Git。CI 提供 Linux CPU 检查流程，实际 CI 是否通过以 GitHub 的运行结果为准。
