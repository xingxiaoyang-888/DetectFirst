# DefectFirst 夜间生成晨报 — 2026-09-17

## 结论

本轮提前收尾：北京时间2026-09-17约04:43（UTC 2026-09-16T20:43Z）。A、B各完成最多两轮有具体假设的修订，仍未达到各自质量门槛，均暂停。GPU已释放，不继续盲重抽。完整合成数据集尚未完成；没有组合A/B、生成正式产品池或启动分割训练。

所有图片和失败记录保留在服务器。人工准入0，reviewer及review hash均为null；AI初检不能代替人工审核。A第一修订有2张带flags的候选，B原始批有1张疑似裂纹，B最后修订有2张疑似小孔；这些局部结果不构成两条路线通过，也不具备已验证的合成缺陷像素标签。

## 数量与质量

| 批次 | 新模型调用 | 技术完成 | CPU派生 | AI结果 | 人工准入 |
|---|---:|---:|---:|---|---:|
| A原始真实核心保护批（窗口前） | 8 | 8 | 0 | 路线未通过，新增外围异常/边界问题 | 0 |
| A第一修订：局部材质编辑 | 8 | 8 | 0 | 2/8带flags；6/8拒绝，路线未过 | 0 |
| A第二修订：source材质+低频照明 | 0 | 0新增 | 8 | 核心/材质保持8/8，但有效条件变化0/8 | 0 |
| B原定固定8张 | 8 | 8（1张恢复包装检查失败） | 0 | 整体0/8；1张疑似裂纹，标注候选失败 | 0 |
| B第一修订：局部R+cut/crack | 8 | 8 | 0 | 整体0/8，目标结构异常不清晰 | 0 |
| B第二修订：局部R+hole/init0.55 | 8 | 8 | 0 | 整体0/8；2张疑似小孔，边界候选不合格 | 0 |

本夜新增32次模型调用：A8+B24，32次技术完成；另有A8个CPU派生结果，不重复记为模型调用。当前A/B系列共有40次模型调用和8个派生结果（48个候选结果）；项目此前另有6次旧FLUX试验，因此累计开发生成46/120次。所有尝试均保留分母，未按合格数补抽。

## 最终配置与实际发现

### A：真实缺陷核心，修改正常外围

4个固定真实支持ID、每图seed字符串14291/22592。FLUX.1-Fill-dev固定revision `358293da0354175698b67ec8299acf928313a78a`，50步、guidance30、BF16/CPU offload。

第一修订把G膨胀24改64、外围编辑E半径96改48，Q羽化12；榛子再限制在source-only median5/Otsu最大连通分量填洞、腐蚀24的材质内部，ROI边缘羽化12。该候选ROI不是人工物体标注。8张final的G及E外像素误差均0；但4张地毯织纹带/相位衔接不自然，2张榛子冠部正常性未成立。仅hazelnut9ff两张带flags。

第二修订复用8张raw，不再调用模型：保留source纹理/几何，仅提取Gaussian sigma48的低频灰度照明比值，gain限制0.94–1.06并按E融合。8张保持原缺陷、材质及边界，但视觉几乎仍为原图，E内平均RGB变化0.40–2.66，未证明有用的正常条件变化。两轮修订已用尽，A暂停。

### B：正常父图，完整AnomalyAny生成结构异常

4个固定normal_train父图、label0、每图同两个seed，原图缩放到512。SD1.5固定revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`、原方法代码commit `585198d594582f41442a779dd2f91288066092e7`，SD全FP32，原内部autocast/CLIP精度及ViT-L/14、RN50双加载保留。所有原注意力/提示/潜变量梯度保留；未用普通img2img替代完整方法。

第二且最后修订 `B_local_structural_hole_r2`：200配置步、guidance12.5、scale50、max_iter25；init0.3改0.55，carpet/hazelnut均使用预先固定的small irregular hole概念。局部R与第一修订字节相同：地毯中心半轴72×40，榛子正常前景质心半轴64×28，轻微不规则边缘并与正常基底相交。R是允许编辑支持域，不能冒充真实缺陷M。属于记录清楚的输入/参数适配，不称完全复现作者notebook参数。

固定PNDM原生timesteps201、t_start90，实际111步/1110次attention。early refinement在5图触发，latent/prompt2246次；3图不触发，2240次，这是原生条件分支。8张全部技术完成。实际查看全部source/native全图、attention/候选overlay及原生R局部：4地毯无清晰孔洞；2榛子variant1仅浅色块/纹理变化；2榛子variant2有两处疑似小孔，物理结构及杂点歧义保留。候选attention∩R未解析真实孔洞边界，榛子候选覆盖R约96%或100%，不能作为像素标签。

原生E/R外也会受潜空间/VAE重建影响：最后B批outsideR平均RGB变化，地毯18.69–19.07、榛子2.33–2.54。B原生输出未声称source像素保持。B两轮修订已用尽并暂停。

## 图片位置与含义

服务器根：`/ssdfs/datahome/u15016/XXY_CVPR/DetectFirst`。以下路径均相对于该根；原始图片/模型留在服务器，本地只同步记录。

| 批次 | 输出目录 |
|---|---|
| A原始8张 | `outputs/round_real_core_anomalyany_20260916T170331Z/A_real_core` |
| A第一修订 | `outputs/overnight_20260917/A_local_material_r1` |
| A第二修订 | `outputs/overnight_20260917/A_source_texture_illumination_r2` |
| B首次包装检查失败，原件不覆盖 | `outputs/round_real_core_anomalyany_20260916T170331Z/B_anomalyany` |
| B固定8张含恢复首图 | `outputs/overnight_20260917/B_anomalyany_continue_v1` |
| B第一修订 | `outputs/overnight_20260917/B_local_edit_r1` |
| B第二修订 | `outputs/overnight_20260917/B_local_structural_hole_r2` |

每个父图目录保存source、call JSON、精确seed、输出SHA及运行测量。A的raw为模型直接输出；final复制source中G及E外像素并融合允许编辑区域，final不是质量合格标签。A第二修订的raw为复用第一修订输出。A的M来自已冻结的真实异常支持角色；B的R/attention及其交集均是候选，不能视为缺陷真值。

代表/失败图（完整相对路径，可在服务器根下直接查看）：

- A带flags：`outputs/overnight_20260917/A_local_material_r1/hazelnut_9ff8cc7816e85fee2912/final_1.png`，同目录`source.png`、`raw_1.png`、`boundary_zoom_1.png`。
- A边界失败：`outputs/overnight_20260917/A_local_material_r1/carpet_466ee643aadfca789cab/final_1.png`及`boundary_zoom_1.png`。
- A第二修订变化不足：`outputs/overnight_20260917/A_source_texture_illumination_r2/carpet_466ee643aadfca789cab/final_1.png`及`difference_x8_1.png`。
- B原始疑似裂纹：`outputs/overnight_20260917/B_anomalyany_continue_v1/hazelnut_a8225fba0ff8de93bf2d/generated_1.png`及`attention_candidate_1.png`。
- B第一修订目标cut缺失：`outputs/overnight_20260917/B_local_edit_r1/carpet_0333c45d5f605d77c743/generated_1.png`。
- B最后疑似小孔：`outputs/overnight_20260917/B_local_structural_hole_r2/hazelnut_a8225fba0ff8de93bf2d/generated_2.png`，另一父图`hazelnut_d51448dcc78a21c276e7/generated_2.png`。
- B最后失败地毯：`outputs/overnight_20260917/B_local_structural_hole_r2/carpet_0333c45d5f605d77c743/generated_1.png`。
- B最后候选覆盖过大：同最后批榛子目录的`annotation_candidate_in_R_2.png`；对应服务器QC局部为`reports/overnight_20260917/B_r2_QC/hazelnut_a8225fba0ff8de93bf2d/R_boundary_zoom_2.png`。

B三批CPU QC目录：`reports/overnight_20260917/B_fixed8_QC`、`B_r1_QC`、`B_r2_QC`。每批含`source_native_contact.png`、各父图的`RGB_difference_x4_*.png`、`R_boundary_zoom_*.png`、候选mask及全SHA收据。这些是检查材料，不是新增生成或标注真值。

## 资源账单及释放

| 夜间账本作业 | 用途 | Slurm状态 | 分配GPU秒 |
|---|---|---|---:|
| 2721694 | 导入窗口前待运行探针 | CANCELLED | 0 |
| 2721733 | gpu4020双环境探针，CUDA失败 | COMPLETED（探针记录FAIL） | 8 |
| 2721741 | gpu4020正确srun对照，CUDA仍失败 | COMPLETED（探针记录FAIL） | 5 |
| 2721744 | gpu4041正确srun对照，CUDA通过 | COMPLETED | 11 |
| 2721748 | B首张生成后旧包装器检查失败 | FAILED，图片独立技术恢复 | 185 |
| 2721755 | A第一修订8张 | COMPLETED | 270 |
| 2721762 | B剩余固定7张 | COMPLETED | 1,179 |
| 2721792 | B第一修订8张 | COMPLETED | 1,317 |
| 2721881 | B第二修订8张 | COMPLETED | 2,277 |
| **夜间新增合计** | **含失败探针/失败包装作业** | **均terminal** | **5,252** |

本夜新增1.4589 L40卡时/上限8，剩余23,548 GPU秒未使用。窗口前项目943 GPU秒（含原分割检查及旧生成/探针），项目累计6,195秒=1.7208卡时。开发调用46/120，剩余74；这些是上限，不要求用满。货币费用未查得，不将GPU时长冒充金额。

曾在两条独立已就绪流水间短时使用2卡，其余起步/最后1卡；未用4卡或A800。GPU脚本实际Slurm step入口与逐图检查带绝对截止、原子预留和全子进程组超时，授权截止01:33Z，实测时钟差保护提前至01:32Z。本轮质量条件不满足，04:43上海时间提前停止。

最终释放收据：`reports/overnight_20260917/final_resource_release_receipt.json`。按精确9个账本job ID查询sacct/squeue，所有GPU作业terminal、活动列表空、预留0、不确定提交0。仅核对uid/cwd/cmdline属于本项目后停止watchdog PID146866；初始与v2记录完整保留。未取消其他ACL作业或修改共享驱动。

## 核验、记录包与版本

- 服务器CPU2721879完成10项预算/截止/并发/异常收据/两线暂停/子孙进程超时检查，及固定PNDM111步、assets/合法角色/token预检。
- B最后QC由CPU2722062导出，新增GPU/模型调用0。导出后首次核验脚本误找`error.json`而退出1；真实原失败文件为`error.txt`。原日志/退出收据保留，CPU2722068仅修正核验路径，不重新导出QC或调用模型。
- 最终核验：`reports/overnight_20260917/final_evidence_verification.json`，包括8图AI检查hash、合法角色/冻结source/R、方法计数/梯度、人审null、QC全hash和原FAILED manifest/summary/error.txt不变。
- 最终紧凑状态：`reports/overnight_20260917/checkpoint.json`；完整实际账本：`overnight_state.json`；每批AI拒绝原因：`A_r1_ai_check.json`、`A_r2_ai_check.json`、`B_fixed8_ai_check.json`、`B_r1_ai_check.json`、`B_r2_ai_check.json`。
- 本轮生成/预算/QC代码版本：`811ad3e8a948bdf224e28e4f75ddc1a130633107`。收尾文档提交另见`reports/overnight_20260917/final_source_commit.txt`。本分支`codex/server-audit-evidence`，不合并主分支。
- 本地与服务器均保存纯记录包：`reports/overnight_20260917/overnight_final_records_v1.tar.gz`，对应`overnight_final_records_v1_receipt.json`及`overnight_final_records_v1_local_verification.json`记录归档SHA、成员逐一hash及排除项。无原图、生成图、`.npy`、权重、notebook或venv。旧`A_complete_B_pending_records_v1.tar.gz`仍保留。
- 本地报告：`F:/DetectFirst/docs/MORNING_REPORT.md`；服务器报告：`/ssdfs/datahome/u15016/XXY_CVPR/DetectFirst/docs/MORNING_REPORT.md`。

本轮只用既有`defectfirst`每10分钟检查，收尾后暂停，不创建重复监控。授权夜间窗口已收尾，科学/数据目标未标记完成。

## 保持的研究约束与后续事项

真实支持4个unique IDs不冒充K_ref=0；历史test路径只按此前冻结的anomaly_support_pool角色复用，未新读保留测试图调参。正常B父图来自冻结normal_train角色，未用异常图作B视觉参考；精确角色manifest及support ID hash写入各收据。

原严格分割器FP32/BF16数值审核在atol1e-5/rtol0仍FAIL；未放宽或启动训练。正式产品放量所需的完整组合管线与连续两批各至少10/12 AI通过均未成立，正式候选0。

后续需处理的具体问题：

1. 人工查看A的2个带flags结果、B的1个疑似裂纹与2个疑似小孔，保留当前拒绝/待审状态及真实review凭据。
2. A需要同时满足外围材质自然和有用条件变化；微小低频照明差不能替代这一目标。
3. B需要可靠、清晰的结构缺陷及独立可审核的像素边界；R/attention或重建残差不能直接当M。现有2轮修订额度已用尽，新研究方向应另作明确设计。
4. 两线可靠后再实现正常/异常反事实组合与真实预处理核对，处理严格模型数值审核，之后才讨论放量和训练。
