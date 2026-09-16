# 2026-09-16 服务器阶段交接

批次：`server_20260916T100542Z`。本记录区分工程检查与研究结果；当前无研究性能结论。

## 结论

- `server_validation_status=FAIL`：真实 DINO 的单张/批量数值差异仍超过既定绝对容差 `atol=1e-5, rtol=0`。
- `research_observation=NOT_EVALUATED`：未执行正式生成、人工 M/G 审核、P/B3 或 200 步训练。
- 环境、两张 L40 的 CUDA 反传、独立冻结策略及有限梯度检查已通过。最近定向回归 27 项通过，不能替代完整回归全部通过。
- 三个 GPU 作业均已结束，交接时队列为空。全程未在本地运行项目测试、训练或下载模型/数据。

## 代码与验证范围

分支 `codex/server-audit-evidence`；main 保持 `631bce25a9922f933c80bd06d41cbce06ba48f27`，未合并。

| 服务器提交 | 修改 | 验证 |
| --- | --- | --- |
| `193b3aa30fb211e2ba7e95e56006db13f19ba0a1` | 独立比较预期与实际冻结策略；补错误冻结/解冻用例及 norm、三路投影梯度要求 | 完整测试 97 通过、1 失败；其他发布检查通过，但发布总状态 FAIL |
| `e6819cb07faf32eb3e65598ab7f1bfa7f46ab514` | 通过 Conv/Linear 输出核实 autocast；接受 GroupNorm 合法 FP32 输出；审核沿用训练 seed_all(11) 和数值设置 | 27 项定向测试通过；真实 FP32/BF16 审核仍 FAIL |

第一轮失败测试为 `test_official_dino_freezing_and_gradients`：CPU 随机官方骨干样例的 logits 单张/批量误差约 `1.28597e-5`。该轮 Ruff、格式、mypy、compile、pip check、FLUX API 导入、smoke 与指标重算检查通过。新提交没有重跑整个 98 项测试集合。

最近定向测试覆盖 `test_server_audits.py`、`test_freeze_policy.py` 和 `test_dino_architecture.py`。模型审核调用已有训练初始化函数，开启 deterministic algorithms，关闭 CUDA matmul 和 cuDNN TF32；不改变网络、批大小或容差。

## 真实 GPU 审核

| 精度 | features 单张/批量最大绝对误差 | logits 单张/批量最大绝对误差 | 状态 |
| --- | ---: | ---: | --- |
| FP32 | 0.00001642853021621704 | 0.00003159046173095703 | FAIL |
| BF16 autocast | 0.006845727562904358 | 0.02864217758178711 | FAIL |

两种精度的同形状排列、替换同伴样本和重复计算误差均为 0；冻结策略通过，所需梯度存在且有限。BF16 检查中实际 Conv/Linear 输出确为 bfloat16，raw features 因后续算子为 float32，归一化特征和分类头仍为 FP32。

这些观测不能证明跨图泄漏，也不能把严格容差失败改记为通过。研究主任务应决定后续计算策略及验收解释；不得仅在审核中逐图循环、改变批大小、暗中回退 FP32 或放宽阈值以制造通过。

第一轮 C 作业脚本未汇总每个检查的失败码，必须逐项读取 record 日志与报告，不能依据批作业结束或 release 一项推断全部通过。C2 已汇总必要模型检查失败；数据不完整单独记录。

## 资产和数据

- Python 3.11.7，Torch 2.5.1+cu124；服务器环境已安装，65 个 wheel 的来源/哈希验证通过。镜像补充包与官方 PyPI 元数据哈希比对，CUDA 包复用官方源字节。
- DINO 源 revision：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
- DINO 权重：346378731 bytes，SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`。这是本地一致性摘要，没有官方公布哈希验证。
- MVTec 仅 grid 完成下载与服务器解压，train=264、test=78；carpet/cable 保留断点，hazelnut 未完成。状态 `DEV_PARTIAL`，未生成完整四产品任务清单或 ROI。
- FLUX 目标 revision：`358293da0354175698b67ec8299acf928313a78a`；23 文件共 33915988848 bytes。网页访问许可已获准，服务器认证待完成，权重未下载。
- 原代理反向隧道已断开。自动审批拒绝了隐藏 SSH 恢复操作，仅返回 blocked by policy；未绕过。官方 PyPI/PyTorch 直连完成了环境准备，MVTec 直连仍超时。
- 数据恢复须保留本轮报告，使用新记录目录；现有解压脚本对已存在 grid 目录/符号链接需要显式续跑适配。

## 资源与授权

| 作业 | 内容 | L40 数量 | 秒 | Slurm 状态 |
| --- | --- | ---: | ---: | --- |
| 2710581 | 硬件盘点 | 2 | 1 | COMPLETED |
| 2711654 | 环境、完整回归和首轮模型审核 | 2 | 176 | FAILED |
| 2711669 | 定向回归、模型复验和已有数据盘点 | 2 | 25 | FAILED |

总计 `(1+176+25)*2/3600 = 0.1122222222` L40 卡时；按 2 元/卡时估算 0.2244444444 元，未计收费取整或其他费用。只统计作业主行，不重复统计 batch/extern。累计预算仍为 4 卡时/8 元，最多同时两张 L40，不申请 A800。

用户已撤销全局两小时截止；原始 window.json 保留，authorization_amendment.json 记录变更。已有作业各自时限未延长。等待下载、认证和人工审核时不占 GPU。

C1/C2 范围、4 个开发产品、K_train=5/C_cal=5/K_ref=0、seed 11/22/33、24 个正式单元、每单元 200 cores + original + 2 sham、完整训练 4000 步（200 仅 stop-after）、生成 50 步/guidance 30 均保持。未扩展到 C3、正式全量实验或四个 E_* 适配器。

## 证据定位与交接

服务器根目录：`/ssdfs/datahome/u15016/XXY_CVPR/DetectFirst`。本轮 `reports/server_20260916T100542Z/` 和 `logs/server_20260916T100542Z/` 保存逐项 argv、时间、退出码、日志与报告。

本地小型包：`F:/DetectFirst/reports/server_20260916T100542Z/C2_checkpoint_v2.tar.gz`；展开到同级 `C2_checkpoint_v2/`。只同步审核记录、代码 bundle 等小文件，未同步模型/数据。

包 SHA256：`8eda16e9b96ddfab0fd0027ed1f17cc9f2ff64b2a50413f1cda1a1a58dbdcc36`，服务器与本地一致。首次 C2_collect 归档因在输入目录内生成而 tar 返回 1，原记录保留；v2 改在输入目录外创建，成功退出。Windows Python 不支持 tarfile filter 参数，第一次本地解压未展开文件；随后逐项验证路径和文件类型后成功展开。

主要证据：

- `C2/model_summary.json`、`C2/sacct.txt`、`C2/targeted_tests.xml`。
- `S05_fp32_v2/model_audit.json`、`S05_bf16_v2/model_audit.json`。
- `release/verification.json`、`S00/`、`B/wheel_provenance.json`。
- `D/development_status.json` 及各下载批次独立 receipts。
- `C2/audit_precision.bundle`：以 4599abc 为前提，包含两次服务器修改；已验证并快进同步本地。

尚需研究主任务决定数值检查后续方案，接续服务器 HF 认证与剩余数据连接问题。当前记录不授权新 GPU 实验。
