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

## 连接恢复与四产品数据准备：13:52:40 UTC 更新快照

主任务按用户后续基础连接修复授权恢复了标准反向隧道。此前自动审批拒绝及各次连接失败仍作为历史记录保留；本快照的代理连接已经恢复，不再以旧拒绝作为当前阻塞。

### 下载路径与有限恢复

首次恢复后，`flux_only_20260916T115833Z` 通过访问预检并开始下载。官方 `us.aws.cdn.hf.co` 的 8 MiB 实际载荷比较为：直连 HTTP 206，10.64884 秒、0.7512555 MiB/s；经代理 HTTP 206，18.98816 秒、0.4213151 MiB/s。该短测量不保证全程速度。只将此已验证官方 CDN 加入 NO_PROXY，HF API/认证继续走 17890。旧下载有意终止，wrapper 退出 241，确认原写入进程和所有锁释放后，使用新批次 `flux_direct_20260916T121206Z` 复用断点；原收据与切换记录保留。

该批次随后因 ChunkedEncodingError/不完整响应退出 2；MVTec 旧恢复批次的 cable 同时记录不完整 HTTP 响应，hazelnut 记录 ConnectionRefused。只读检查确认 17890 再次无监听。主任务在北京时间 21:23:22 启动标准隐藏 SSH PID 51996，先确认无重复隧道，随后核实服务器端口监听和官方 FLUX API HTTP 200。凭据未重新登录或导出。

新批次 `flux_resume_20260916T132054Z`、`mvtec_retry_20260916T132054Z` 在下载锁全部可用后启动。cable 对已有 455408409 字节断点的官方 Range/If-Range 探针返回匹配的 HTTP 206，总大小 504731120 字节，之后安全续传完成。FLUX 也观察到分片继续增长。

用户要求每 10 分钟检查一次，旧 45 秒监控已单独停止，下载进程保持后台运行。恢复 helper 对临时连接/超时/不完整响应使用新独立尝试目录，每项最多 3 次恢复，两次启动至少相隔 600 秒；已有匹配进程就跳过。当前两项恢复计数各 1/3。服务器语法和幂等操作通过，未重复启动或增加计数。源码 SHA256 `f8ce76807c6c0aeef2c5b5152ee53ab5f727de470dc21f935e1521ab7e06bff5`，首版源码保留。代理由主任务管理，等待不分配 GPU。

### 四产品开发数据已准备

CPU 作业 `2715380`，账户 `p_p15016`，qos `cpu-500_core-l40-8_card-a800-8_card`，intel 分区，2 CPU/16G/10 分钟上限，0 GPU，实际 106 秒，Slurm COMPLETED/退出 0。实际运行代码提交为 `b11dd97955a6acabae72d1d7394a528fe4245a89`；语法和数据准备必要记录均退出 0。

| 产品 | 归档 bytes | train | test | 解压处理 |
| --- | ---: | ---: | ---: | --- |
| carpet | 740285760 | 280 | 117 | 新独立目录 |
| grid | 160763852 | 264 | 78 | 验证并复用旧目录 |
| cable | 504731120 | 224 | 150 | 新独立目录 |
| hazelnut | 617098680 | 391 | 110 | 新独立目录 |

服务器计算节点重新核对实际归档 SHA 与 READY 收据一致。归档哈希均为本地一致性摘要，没有官方公布哈希验证。未删除、覆盖或重复解压 grid。

- 开发清单 1614 条；manifest SHA256 `bc8cf47e0b5ed5b35bf526251255cd3836b34f842a76b53272653f3b36119324`。
- support_ids SHA256 `66860401ed92c43aa66f1a2b891269e2d4c57706c266bcec8549390e7ec8644c`。元数据文件的实际字节已与准备收据重新比对。
- 声明划分 audit 无冲突；1614 条 entity ID 全部 unknown，因此该 audit 不证明物理实体间的独立划分。
- 40 个正常 parent 的待审 ROI，每产品 10 个，全部 WAITING_HUMAN；reviewer 和审核 sha256 全为 null。四张联系表和单个预览只保留在服务器，未传至本地，也没有代替人工审核。
- 状态 `DEV_READY_WAITING_ROI_REVIEW`，范围仅四个既定开发产品。未修改完整数据协议、正式单元或 K_train/C_cal/K_ref、种子、生成/训练设置，未训练或生成。

### FLUX 仍在下载

13:52:40 UTC 快照：17/23 个文件已完成，9775640480 bytes；现有 incomplete 文件共 2369781760 bytes，写入并发仍按原下载器最多两项，保留的 incomplete 文件数不代表当前传输数。目标仍为固定 revision `358293da0354175698b67ec8299acf928313a78a` 的 23 个 Diffusers 文件，总 33915988848 bytes。下载收据和 `verified_files.json` 尚未 READY，不能确认模型下载完成；之后需核对精确文件集/大小及 8 个官方 LFS SHA。

原严格 FP32/BF16 审核仍 FAIL；补充有限输入梯度证据 PASS 单独保留，研究结果仍 NOT_EVALUATED。L40 累计仍 420 GPU 秒、0.1166666667 卡时，按 2 元/卡时估算 0.2333333333 元；本次 CPU 分配不追加 L40 卡时，CPU 费用未估算。

### 开发阶段纯记录证据包

本地 `F:/DetectFirst/reports/server_20260916T100542Z/development_checkpoint_20260916T135230Z.tar.gz`，375324 bytes，SHA256 `5acb15c76e45907aedf6e205ad111ca82a85d64c501c27e5659cca4db2f25b0e`，服务器与本地一致。同级同名目录已按成员路径/类型检查后展开，53 个来源记录文件，78 个 tar 成员。包含归档收据、独立命令/退出记录、CPU 记账、清单/划分/support 元数据、待审 ROI JSONL 和下载恢复记录；没有模型、原始图像或凭据。本地仍仅用于同步、记录与版本管理，没有运行项目测试。

核心汇总为包内 `handoff_status.json`，原服务器结果在 `reports/mvtec_retry_20260916T132054Z/D/`。数据准备不再重复提交；10 分钟监控继续等待 FLUX 下载校验。全部下载和记录交接完成后删除该监控，不启动新的 GPU 实验。

## FLUX 下载校验完成：北京时间 2026-09-17 00:11:33

本节追加最终结果，保留前面的失败记录和阶段快照。最终批次 `flux_resume_20260916T151523Z_r3` 于 2026-09-16 15:24:49 UTC 启动，16:11:33 UTC 结束，launcher 退出 0。16:22:04 UTC 的单次监控确认下载收据与 `verified_files.json` 均为 READY，下载进程已退出。

### 固定资产验证

- 仓库 `black-forest-labs/FLUX.1-Fill-dev`，revision `358293da0354175698b67ec8299acf928313a78a`。
- 实际文件集、下载收据及原始 expected 文件集精确一致：23 个选定的 Diffusers 文件，总 33915988848 bytes，无 incomplete 文件；排除根目录重复权重。
- 8 个官方 LFS SHA256 全部匹配现有下载器对实际下载字节计算的 SHA256。其他小文件仅核对收据一致性摘要，最终记录整理时重新核对了这些小文件的实际字节；不宣称它们通过官方哈希验证。
- 收据 SHA256 `d4db348707865761e784ffcbc415ade6c1f74296236d3d17700b1d4d4cbbebd0`。
- 原始 expected 清单 SHA256 `f8294d6602cb4232c8eb8dfd58b005401c1e118f10ea727fd21bbb1909b273a1`，与首次固定清单字节一致。
- 原 `verify_flux_receipt.py` 已由批次启动脚本执行并退出 0，验证文件的 revision、总量、哈希及失败列表已复核；文件大小及修改时间仍符合收据。未重复读取全部大权重计算第二轮哈希。

模型资产位于服务器 `models_cache/FLUX.1-Fill-dev`，最终原始收据和验证文件位于 `reports/flux_resume_20260916T151523Z_r3/`。READY 仅说明下载资产验证完成，没有模型加载、推理、生成、训练或研究性能结果。

### 恢复记录与连接

FLUX 有限恢复共使用 3/3 次，MVTec 使用 1/3 次；各次失败收据、命令和分片保留。第二次 FLUX 恢复完成 22 个文件后因 ChunkedEncodingError 退出。第三次准备期间代理端口 ConnectionRefused，未消耗启动次数；主任务在北京时间 23:22:27 恢复同一条标准隐藏 SSH 隧道（本地 PID 45144），核实服务器端口监听及官方 API HTTP 200 后，第三次恢复安全复用缓存并完成。

主任务还转达了未来同一条标准本地连接的基础修复授权，记录在 `reports/server_20260916T100542Z/connection_authorization_20260916T152227Z.json`。本轮收尾未启动新隧道。旧 PID 51996 消失原因仍未知；更早 PID 48104 的 Connection reset 只证明至少一次 SSH 传输重置，不作为 HF 授权过期证据。认证未重复登录，API/认证仍经 17890，只有既定官方 CDN 载荷直连。

### 最终纯记录证据包

服务器及本地同相对路径 `reports/server_20260916T100542Z/final_download_checkpoint_20260916T162204Z.tar.gz`，本地绝对路径 `F:/DetectFirst/reports/server_20260916T100542Z/final_download_checkpoint_20260916T162204Z.tar.gz`，86931 bytes，SHA256 `d1d95c954b9d4270c76d64c684ca48dfa7e759364faa0b6f7d7ffa8b75ae7ec3`。

包内 235 个来源记录文件、292 个 tar 成员，包含最终完整验证 JSON、固定清单、下载收据、认证访问结果、各次历史失败/恢复/切换记录、连接授权及冻结的 CPU/GPU 记账。开发阶段完整清单/ROI 元数据继续引用前一 `development_checkpoint_20260916T135230Z.tar.gz`，其原 SHA 与大小已复核。模型、原始数据、ROI 预览和凭据均未纳入。

交付副本按 URL 查询参数、HF token 与 Bearer 值进行脱敏检查，本批选定记录的替换数量为 0；原服务器日志保持原字节。所有来源和交付副本的 SHA/大小记入包内 `handoff_status.json`。本地仅传输此小型记录包，检查成员路径/类型、包 SHA 与全部记录文件 SHA 后展开，验证 PASS，没有执行本地项目测试。

最终包记录的服务器代码 HEAD 为 `927b0d733289272a10fcd39ac8b45ac392ee6098`，跟踪工作树干净。相对算法代码提交 `e6819cb07faf32eb3e65598ab7f1bfa7f46ab514` 仅有 docs 变更；CPU 数据准备的实际运行提交仍为 `b11dd97955a6acabae72d1d7394a528fe4245a89`。

### 阶段边界与交接

四产品数据仍为 `DEV_READY_WAITING_ROI_REVIEW`：1614 条清单及原 manifest/support SHA 未改变，40 条 ROI 全部 WAITING_HUMAN，reviewer 和审核 SHA 全为 null；entity ID 全部 unknown 的限制保留。未重复提交 CPU 准备、解压或制造人工审核结果。

原严格 FP32/BF16 审计继续 FAIL；补充输入隔离仅为有限探针 PASS，验收定义保持 `atol=1e-5, rtol=0`，研究结果 NOT_EVALUATED。最终只读队列记录为空，累计仍为 420 GPU 秒、0.1166666667 L40 卡时，按 2 元/卡时估算 0.2333333333 元，不含取整及其他费用；CPU 费用未估算。等待和收尾未增加 GPU 作业。

下载校验、既定 CPU 数据准备与记录整理已完成，结果交接主任务后停止这项 10 分钟监控；人工 ROI 审核及研究验收由主任务接续。本轮仅在 `codex/server-audit-evidence` 同步文档，不合并 main，不启动新实验。

## 后续 S02 授权：首组生图与固定污渍对照

主任务随后转达用户的首组真实生图授权，以及对规则椭圆污斑的质疑。下载监控已停止，本节是新的有限生图阶段。没有训练、正式扩量或人工审核通过。

### 首组真实 FLUX 图片

服务器目录 `/ssdfs/datahome/u15016/XXY_CVPR/DetectFirst/outputs/s02_first_group_20260916163237Z/`。保存 `source.png`、`candidate_roi.png`、`edit_mask.png`、`roi_overlay.png`、`defect.png`、`normal_edit_1.png`、`normal_edit_2.png`、`contact_sheet.png` 以及 manifest/summary/README。前三张生成输出均为可读的 512×512 RGB，联系表 1536×1080。

底图为 MVTec AD carpet，parent `mvtec/carpet/0333c45d5f605d77c743`，原始 `carpet/train/good/074.png`，original_split=train、role=normal_train、label=0；原图 1024×1024，实际输入缩放为 512×512。它是正常图，尚未产生 defect.png 时已向主任务明确说明，未把 source 当缺陷输出。

固定 FLUX revision 不变，50 步/guidance 30/BF16，实际 transformer、两个 text encoder 和 VAE 的参数 dtype 均为 BF16。单 pipeline、`enable_model_cpu_offload(gpu_id=0)`，离线 `local_files_only=True`，没有量化或网络下载。白色/255 的 mask 为允许编辑的 R；黑色/0 为保留条件。保存原生输出，不合成 mask 外像素。候选 ROI 仍未审，R 不是缺陷真值。

CPU 预检 `2721472`：2CPU/16G/0GPU，15 秒，Ruff/语法/pip check/输入准备退出 0。GPU `2721474`：显式 `p_p15016`、`cpu-500_core-l40-8_card-a800-8_card`、L40 分区、`gpu:l40:1`，实际 NVIDIA L40，6CPU/96G/30 分钟上限，COMPLETED/0:0，实际 125 秒。入口运行提交 `4e37283cbf978fd27c060ed0195b9affcb8993df`，pipeline 加载 10.23861 秒。

| 调用 | pipeline 秒 | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: | ---: |
| defect | 54.19345 | 24179614208 | 24352129024 |
| normal_edit_1 | 26.31349 | 24179614208 | 24352129024 |
| normal_edit_2 | 25.34665 | 24179614208 | 24352129024 |

设备总显存 47576711168 bytes。首次调用有较大启动开销；稳态两次约 2.32 张/分钟，三次 pipeline 总计约 1.70 张/分钟，不含加载/脚本/保存成本。只记录这些有限样本，不构成 20 次测速结论，也不能由 allocated 峰值推断双实例安全并行。主任务与本任务的 AI 视觉初检均认为深色污斑边缘太规则、像椭圆贴片，正常编辑也有局部模糊/纹理变化；技术生成完成不代表缺陷真实性或正常语义合格，未写人工审核通过。

### 有限 2×2 污渍真实性诊断

用户优先要求诊断真实性，因此未提交吞吐试验 job。新的目录 `/ssdfs/datahome/u15016/XXY_CVPR/DetectFirst/outputs/s02_stain_diagnostic_20260916T164729Z/`，`comparison_grid.png` 为 1024×1080：上行旧 prompt、下行改进 prompt，左列原椭圆 R、右列不规则 R。左上 `old_prompt_ellipse_historical.png` 是原首图字节复制，没有重生成。三张新增 raw PNG 为 `old_prompt_irregular.png`、`new_prompt_ellipse.png`、`new_prompt_irregular.png`，全部可读的 512×512 RGB。

三次全部同正常训练底图、同首图 defect seed `6542678547070446254`，保持 50/guidance30/512/BF16/原 model CPU offload 单 pipeline 串行；没有批处理或 prompt embedding 缓存改动。历史参考来自另一个 GPU 作业/节点，解释保持有限。没有查看真实测试缺陷选 prompt，没有后处理粘贴噪声或污渍。

原 R 面积 13085，bbox xyxy（含端点）`[139,233,291,341]`；不规则 R 面积 13087（仅 +2 像素），bbox `[124,223,318,336]`，8 连通单组件，均在同候选 ROI 内。不规则 R 用原椭圆中心/长宽比的 128 顶点径向多谐波轮廓，经固定 24 步缩放二分匹配面积；公式、scale、mask SHA 保存于 manifest。它只是编辑许可区域，不是污渍真值。

改进普通 prompt 描述非对称灰褐液体吸附、浓淡不均、羽化、透明渗染及可见纤维/编织结构，并要求避免黑色几何贴片。该版本不支持 negative_prompt，未传入。日志保留 CLIP 108-token 文本按 77 上限截断末尾的 warning；T5 默认上限 512。未在作业途中缩短或替换冻结 prompt。

CPU `2721504`：2CPU/16G/0GPU，13 秒，Ruff/语法/输入与 mask 准备退出 0。GPU `2721505`：同显式账户/qos/实际 1×L40，6CPU/96G/30 分钟上限，COMPLETED/0:0，128 秒，运行提交 `7949af588bcb332f3ec372b1dcb5862df4bdd9d1`，加载 10.50629 秒，无 OOM。

| 新增对照 | pipeline 秒 | peak allocated bytes | peak reserved bytes |
| --- | ---: | ---: | ---: |
| old_prompt_irregular | 55.27777 | 24179614208 | 24352129024 |
| new_prompt_ellipse | 26.35551 | 24179614208 | 24352129024 |
| new_prompt_irregular | 26.03874 | 24179614208 | 24352129024 |

AI 初检目标仍未达到：新 prompt 配椭圆 R 仍形成偏平的规则灰色块，纤维未连续可见；不规则 R 减弱明显色块，却没有确认清晰可信的吸附污渍。此判断保存在独立 `ai_visual_check.json`，不是人工审核，也不是对 mask/prompt 的普遍因果证明。全部结果保留，未抽换 seed、无限调 prompt 或扩大旧 recipe。该版本原生 image/mask processor 与 generator 支持列表输入，源码证据保存于 `native_api_source.json`；本轮没有 batch2 性能实测。

### 记账与记录交接

本次共 6 次真实开发 diffusion 调用（首组 3＋新增对照 3）计入 120 次开发预算，全部尚未准入。两个 GPU 作业已结束释放，本项目累计 420＋125＋128＝673 GPU 秒，0.1869444444 L40 卡时，按 2 元/卡时估算 0.3738888889 元，不含取整/其他费用。两次 CPU 分配合计 28 秒、0 GPU，CPU 费未估算。其他 ACL 项目作业按 WorkDir 核实为独立任务，未操作或混入本项目预算。

最终纯记录包 `F:/DetectFirst/reports/s02_stain_diagnostic_20260916T164729Z/s02_generation_checkpoint_20260916T164729Z_v2.tar.gz`，服务器同相对路径，52183 bytes，SHA256 `7751afd52ca640cf198098bbabccf91f4a8a5275970e13e0859b83374e8ddc1c`。65 个来源记录＋1 个 checksum 索引，共 66 个 tar 成员，本地路径/类型/包及逐文件 SHA 检查 PASS；没有图片、模型、原始数据或凭据。本地仍仅同步代码/记录，没有本地项目测试。

首次记录包中仅 AI 侧 JSON 数字字段发生 64 位 seed 序列化舍入，v2 改用准确十进制字符串并包含 erratum，旧包保留；原 Python 运行 seed、原生生成 manifest 与图片均未改变或重生成。逐调用计时使用 perf_counter，费用 elapsed 用 Slurm，节点 UTC 时钟与登录节点可能有偏差，保留原始时间字段。

两个独立技术入口未改变正式 ROI 人审门槛、模型源码、严格 `atol=1e-5, rtol=0` 验收或数据划分。原 FP32/BF16 审计 FAIL 保留，ROI reviewer/审核 SHA 仍 null，研究 NOT_EVALUATED。有限诊断已完成，本轮不再新增生图、吞吐试验、训练或正式扩量，交由主任务与用户查看原始图片决定后续。
