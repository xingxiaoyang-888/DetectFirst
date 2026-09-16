# 2026-09-16 服务器阶段交接

批次：`server_20260916T100542Z`。本记录区分工程检查与研究结果；当前无研究性能结论。

下文先保留首轮交接快照；新增授权诊断、认证和下载进展见末节。当前原严格审核仍 FAIL。

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

## 后续授权：单卡诊断与下载准备

主任务随后转达用户授权：启动 FLUX 下载、恢复剩余三类开发数据，并执行一次最多 1 张 L40、10 分钟的前向输入梯度诊断。严格验收定义、网络和精度保持。

### 单卡诊断已完成

作业 `2713054`，批次 `input_isolation_20260916T115802Z`，gpu4002，1 张 L40，实际 16 秒，Slurm COMPLETED/退出 0。语法记录和诊断记录均退出 0，GPU 已释放，最终队列为空。

实际代码提交仍为 `dc514def64bf7aedda74f003b4a8956c625c948f`，跟踪工作树干净；项目源码 fingerprint 为 `4c9f92784f3b11ab871cc1a98eff5c23e6948decc8e0f37a3042ccf638f7f7d0`。诊断脚本 SHA256 为 `606c517f672955f1703421bc32714ef4a67710f1ae2858068a9e90770d04c122`。脚本、配置、官方 DINO Python 文件哈希清单、关键源码副本、模块清单和实际算子 dtype/shape 均保存在本批报告。

输入为相同 seed 17 的 4 张合成探针，归一化并填充后形状 `[4,3,532,532]`；模型使用原 train 前向、seed 11、关闭两种 TF32、确定性设置。没有逐图循环替换原批量前向、精度回退、配对损失、参数更新、训练或生成。FP32/BF16 的实际 Conv/Linear dtype 均符合请求。

分别对 anchor 0 的 features 和 logits 做固定随机线性投影，再求其对完整输入 batch 的梯度：

| 精度/输出 | 自身输入梯度 max abs | 其他三图梯度 max abs | 故意混合输入正对照的他图梯度 max abs |
| --- | ---: | ---: | ---: |
| FP32/features | 3.245813331886893e-6 | 0 | 1.2654434158321237e-6 |
| FP32/logits | 2.4254908203147352e-4 | 0 | 7.432675920426846e-5 |
| BF16/features | 3.56137752532959e-6 | 0 | 1.5497207641601562e-6 |
| BF16/logits | 2.613067626953125e-4 | 0 | 9.298324584960938e-5 |

自身梯度均有限且非零，其他图梯度严格零。同 shape 排列、替换同伴和重复运行误差仍全为 0。正对照仅在诊断夹具中令 `mixed[i] = input[i] + 0.25 * input[(i-1) mod B]`，原模型不变；检查正确检出 anchor 0 对 image 3 的依赖。

`supplemental_evidence_status=PASS` 只指这些补充检查通过。单张/批量误差完整复现首轮 FP32/BF16 数值，原 `atol=1e-5, rtol=0` 审核继续 FAIL，`acceptance_definition_changed=false`，`research_observation=NOT_EVALUATED`。有限探针和两个标量投影不能证明所有输入下的完整 Jacobian 隔离，也不证明分割性能或 BF16 精度无影响。

累计四个作业主行：420 GPU 秒，`0.1166666667` L40 卡时，按 2 元/卡时估算 `0.2333333333` 元，未计取整/其他费用。总预算仍为 4 卡时/8 元。

### 认证通过，下载因连接阻塞

主任务已完成官方 OAuth 设备登录并用原下载 CLI 独立确认，权限为 `openid profile gated-repos`，约 30 天有效、支持刷新。凭据由官方库写入默认缓存，文件权限 600，未同步或加入版本控制。隔离认证环境位于 `/ssdfs/datahome/u15016/XXY_CVPR/hpc/.venv-hf-auth`，训练和下载环境未改变。认证复核记录 `SUPERVISOR_AUTH_REVIEW.md` 已加入交接包，SHA256 `f809096dc5885c6de65d99a4e2e71f670362bcbf57a8e7f4d6c27b8a33a254f0`。

首次 FLUX 启动批次 `flux_only_20260916T112949Z`，PID 2966960，在只读访问预检返回 ProxyError/退出 2；文件数和字节数均为 0。随后只读探测明确 `127.0.0.1:17890` ConnectionRefusedError，主任务也确认端口无监听。认证复核成功与后续连接断开分别记录，不重新认证。

- FLUX 新批次 `flux_only_20260916T115833Z` 准备完成，尚未启动。固定 revision 与 23 个精确文件名，共 33915988848 bytes；排除根目录重复权重。配置 SHA256 `89e400230992a2883e315df3de44fae1adeaf5ddda2607c24c74abc892011d3a`。下载后核对 23 文件字节数，并将已有下载器从实际字节计算的 SHA256 与官方 LFS 哈希比较；无官方 LFS 哈希的小文件仅记录一致性摘要。
- MVTec 新批次 `mvtec_resume_20260916T115610Z` 只含 carpet/cable/hazelnut，顺序单文件传输；分别保留 175112192/14680064/0 bytes 断点。FLUX 两文件与 MVTec 一文件合计最多三个大文件传输。
- `D_resume.py` 已适配验证并复用原 grid 解压目录，不删除、覆盖或重复解压；尚未执行。解压及完整清单/ROI 准备仍需资产就绪和服务器计算分配。
- 停止重复下载/认证，没有创建或恢复隧道，也没有绕过此前自动审批拒绝。等待主任务接续持久连接安排。

### 新证据包

本地：`F:/DetectFirst/reports/server_20260916T100542Z/download_diagnostic_checkpoint.tar.gz`，展开目录同级 `download_diagnostic_checkpoint/`。SHA256 `7bc414c9e134ac66f39f34070410c8d78584c92994c194d485c15d6cf79318cb`，服务器与本地一致。

包含诊断完整 JSON/summary、逐项原始命令和退出记录、Slurm 精确记账、认证复核、首次失败下载与两个新准备批次；没有模型/数据或凭据。核心报告是 `reports/input_isolation_20260916T115802Z/results/input_isolation.json`，交接汇总为同批次 `handoff_status.json`。本地仍未执行项目测试。
