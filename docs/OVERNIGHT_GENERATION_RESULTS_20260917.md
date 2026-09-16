# 夜间生成执行记录

本文件只记录工程进展，所有候选保持 WAITING_HUMAN，reviewer及review hash为null。旧失败不覆盖，原严格分割模型审核FAIL不变。

## 硬截止与原子预算

服务器账本为 `reports/overnight_20260917/overnight_state.json`，固定截止 `2026-09-17T01:33:00Z`。项目窗口基线943 GPU秒、开发14次；本窗口新增GPU上限28,800秒、累计开发调用上限120次。queued/running最坏剩余预留计入上限，flock锁覆盖提交和调用登记。提交结果不确定时持久保留预留并阻止重复提交。

GPU作业在真实Slurm step中运行 `scripts/overnight_budget.py run`，实际启动时重新计算截止与作业预留，对全部生成子进程组限时；每张图调用前再检查。服务器截止watchdog只核对并取消本账本具体Job ID，PID/日志和命令保存在同目录。CPU2721736通过七项检查：过期拒绝、已耗+完整预留、卡数上限、异常收据持久性、不确定提交阻断、并发调用原子上限、子孙进程整体超时。测得计算节点入口UTC比登录提交收据早15秒，新增60秒提前量；实际入口、逐图检查和watchdog均以01:32Z收尾，用户授权截止仍为01:33Z。

## CUDA阻塞定位

原指定gpu4023探针2721694确认本项目WorkDir/名称且仍PENDING后取消，耗0秒。2721733在gpu4020两环境都初始化失败；2721741正常srun step复核仍失败。`CUDA_VISIBLE_DEVICES=0`和`SLURM_JOB_GPUS=4`的差异本身不证明绑定错误。

排除gpu4020后的独立探针2721744在gpu4041实际NVIDIA L40/driver550.54.14，两环境Torch2.5.1+cu124均CUDA可用，逻辑0对应作业GPU3而仍正常。未unset CUDA_VISIBLE_DEVICES、未猜测物理索引、未修改共享驱动或接触分配外显卡。证据定位为当前节点/运行层差异，具体底层原因未确定。新增三个短探针合计24 GPU秒，生成调用0；实际继续消耗以账本为准。

## B完整方法与A第一轮修订

B2721748在gpu4041生成首张后被包装检查器错误中止，耗185秒：原检查误以为200配置步/t_start140必然60实际步；固定PNDM原生timesteps有201个，因此实际61步、610次注意力更新，prompt update1次、latent/prompt梯度更新1246次、early refinement1次。CPU2721760从固定scheduler配置独立复核201−140=61，图像有限性/尺寸/hash通过。未修改官方方法或权重，修复动态原生步数检查和过早写SUCCESS的缺陷。原FAILED manifest/summary/error完整保留并登记SHA；在新continuation目录复用已生成首图，新增调用0，其余7个原定seed继续，整体仍最多8个固定调用。首图的技术恢复不是语义通过。

A第一轮固定 `A_local_material_r1`：同4合法真实支持、同2seed、同prompt及50步/guidance30/BF16/offload，仅把G24改为64、E96改为48；榛子限制在source灰度median5+Otsu最大连通分量填洞、腐蚀24像素的材质内部，ROI边缘额外羽化12像素。假设是增大复制保护消除原卫星异常，限制编辑范围保护轮廓和背景。该ROI为编辑候选，不是人审物体分割。

CPU2721750因zip严格参数lint失败，未生成；修复后2721751准备通过。逐行实际查看4个source/区域/ROI对照：E像素42,065、38,702、19,077、9,047，皆非空。榛子ROI偏向明亮上部、排除较暗下部且边缘波状，记录AI_EXECUTION_PASS_WITH_FLAGS。生成语义尚未检查。本轮产物在 `outputs/overnight_20260917/A_local_material_r1`，原A8及其AI拒绝保留。

A2721755完成270秒，8/8机械保持通过。实际查看全部8个最终/source全图和8组三列边界source/raw/final：榛子9ff两张不再新增黑洞，形状/背景保留，AI_PASS_WITH_FLAGS2张；4张地毯仍有软化织纹带或相位衔接问题，b2两张冠部纹理/暗沟细节的外围正常性未成立，AI_NOT_PASSED6张。A路线整体仍未通过，人工准入0。结果明细 `reports/overnight_20260917/A_r1_ai_check.json`。不因机械零误差或变色量较小改判。

两条独立流水就绪后只使用最多2卡。B其余7个固定调用在2721762继续，产物 `outputs/overnight_20260917/B_anomalyany_continue_v1`。两线完成后逐图检查核心保持、外围正常、缺陷真实可见、边界与标签；通过前不组合或放量。正式扩量仍要求完整管线和连续两批各至少10/12 AI通过。

## A第二轮具体假设与暂停

针对地毯织纹相位与榛子冠部重塑，第二轮 `A_source_texture_illumination_r2` 保留source高频材质和几何，只从第一轮8张raw提取灰度Gaussian sigma48的低频照明比值，限制gain0.94–1.06后按相同E羽化乘到source。新调用0、GPU0；原raw/A第一轮失败不改。此条件类明确缩窄为轻微局部照明，不能声称新的材质几何生成或学习到不变量。

CPU2721784完成8个派生候选。实际看全部8个source/final全图及8组source/reused_raw/final局部：核心、正常材质、缺陷可见性和边界保持8/8；但视觉几乎仍是原图，E内平均RGB改变量0.40–2.66，程序非零差异不证明外围变化有效。因此整体AI_NOT_PASSED_EFFECTIVE_EXTERIOR_CONDITION，AI整体准入0，人工准入0。明细 `reports/overnight_20260917/A_r2_ai_check.json`，产物 `outputs/overnight_20260917/A_source_texture_illumination_r2`。

A已用尽两轮具体假设修订，暂停该路线，继续独立B。两个seed派生的hash不同，但不按微小差异宣称有用条件多样性。不组合或正式扩量。
