# 服务器接手待办与本地暂停安排

更新：2026-09-16。代码修复提交：c6037353657ec556ad578e191f12305086ff1f39；分支：codex/server-audit-evidence。main 基线保持 631bce25a9922f933c80bd06d41cbce06ba48f27，不合并 main。

## 当前状态

- 用户要求本次复核结束后等待服务器，停止本地代码扩展、修复和测试，也不追加静态检查轮次。当前仅完成文档/台账收尾与分支推送。
- c603735 的原对话复核状态：review_status=STATIC_REVIEWED_WITH_SERVER_FOLLOWUP。
- server_validation=NOT_RUN；server_validation_status=NOT_RUN；research_observation=NOT_EVALUATED。不能将静态复核记为工程阶段 PASS。
- 原对话已核对 11 个修改文件与该提交一致、5 项静态日志哈希一致。此次收尾文档没有新增测试或静态检查证据；既有日志只对应当时的修复内容。
- 原始复核材料：F:/DetectFirst/reports/fix_R1_R3_20260916/SUPERVISOR_REVIEW.md。服务器执行安排见 [S00/S01 文档](S00_S01_RUNBOOK.md)。

## 接入后首先补齐的验收缺口

1. **冻结策略独立验收。**当前 model_audit.py 只根据参数实际 requires_grad 检查冻结参数是否有梯度，未独立验证它本来是否应被冻结。若前六层被错误解冻，当前检查可能误判。服务器阶段先添加预期/实际 trainable 对照：patch/token/position 与 blocks 1–6 应冻结；blocks 7–12、输出 LayerNorm、投影、decoder、classifier 应按既定模型设定可训练。检查所需表示路径的梯度，补错误解冻与错误冻结故障用例，再执行对应回归。当前骨干冻结实现本身仍符合设定；这是审核缺口，不在本机继续修复。
2. **对服务器实际提交执行完整回归。**保留 commit、实际 argv、退出码、原始日志和 XML。历史 69 项 CPU 检查及本地五项静态检查都不能替代行为验证。服务器若修复上述缺口，后续证据必须绑定新的实际 commit。
3. **真实 DINO FP32/BF16 审核。**先满足冻结策略验收，再按执行文档 G 节核对实际精度、输出/损失/梯度有限性、分类路径和独立性。保留失败原因；绝对 atol=1e-5、rtol=0 不因 BF16 差异自动放宽，FP32 PASS 不转移为 BF16 PASS。
4. **环境与记录行为。**验证逐设备初始化/反传失败汇总；在 Linux/Bash 验证 set -e 下失败退出码保留、重复 label 拒绝覆盖、并行阶段隔离；验证下载失败、恢复与逐批收据保留。不可跳过服务器 Bash 用例后声称 R3 已验收。

## 下载尚在进行时如何推进

download_batch.py 在全部下载结束后才发布可变 latest 目录。下载期间，相互独立的任务应使用 `reports/<本批次>/S01_download/receipts/<asset_id>.lock.json` 中已实际写出且 status=READY 的收据，不等待其余无关资产，也不读取上一批 latest 来冒充当前就绪。

- DINO 检查需要本批 DINO 源码与权重分别 READY；权重摘要从本批收据写入审核配置，并核对源码 revision。
- FLUX 相关检查/生成需要其本批收据 READY，且相应数据、任务清单和人工审核等前置条件已满足。运行配置中读取 FLUX 锁文件的路径应明确指向本批 receipts 的 flux_fill.lock.json；在运行记录中保留配置和收据摘要。
- 尚未出现或 status 不为 READY 的收据不能标就绪。download_status.json.complete 仅表示所有任务已返回，不代表全资产成功。
- latest 是工作索引；历史验收以本批配置、receipts、command.json 和原始日志为准。恢复使用新批次证据目录，不覆盖旧批次。

## 等待范围

本轮推送完成后等待服务器接入或用户明确后续任务；不自主触发本地迭代、自动任务、下载、模型检查或全量实验。C1/C2、数据角色、监督量、seed 和生成/训练预算保持不变。四个公开方法适配器仍待实现，不由本次收尾推进。
