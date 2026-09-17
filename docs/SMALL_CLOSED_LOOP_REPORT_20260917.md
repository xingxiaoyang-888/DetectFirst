# 2026-09-17 四核心真实网络小规模闭环结果

## 结论

**本轮工程闭环通过。** carpet/hazelnut 各 P、B3 共四个实际 DINOv2 ViT-B/14 FP32 运行全部完成 200 步；来源与标签、真实梯度隔离、参数更新、冻结策略、100 步保存退出再恢复、P/B3 公平曝光及预测文件核对通过。所有 GPU 作业已结束，资源预约为 0。

本轮是 `real_support_closed_loop_fp32` 独立诊断 variant，`test_only=false`。训练科学配置保持 steps=4000、warmup=200，以 stop-after=200 截断，因此原 train_report 状态仍为 `INTERRUPTED_AT_REQUESTED_STEP`。200 步不足以说明收敛或研究增益；AI 复核未替代正式人审，human_admitted=0。原严格单张/批量 atol=1e-5 数值审核 FAIL 保留。

## 数据与生成来源

固定 seed11、K_train=5 支持池中的四个真实异常核心，每产品两个；每核心为正常/异常 × 原条件/两个既有外围条件，共 **4 个独立核心、4 组六视图、24 张实际图像**。

| 产品 | 真实支持样本 ID | 最终正常核心来源 |
| --- | --- | --- |
| carpet | mvtec/carpet/466ee643aadfca789cab | 同源 GT 阴性材质区域 donor，固定平移 dy=-192 |
| carpet | mvtec/carpet/5965ac2cc93cb74b6b52 | FLUX 局部修复 |
| hazelnut | mvtec/hazelnut/9ff8cc7816e85fee2912 | 同源 GT 阴性材质区域 donor，固定平移 dx=-64 |
| hazelnut | mvtec/hazelnut/b2c6ac0e9359a16ddec9 | FLUX 局部修复 |

新增真实模型调用 **4 次**。其中 carpet466 的首次灰色椭圆补丁及 hazelnut9ff 的黑盘不能证明原孔洞已去除，失败记录保留；授权后的两次 donor 派生为 CPU 操作，新增模型调用 0 次。总修复记录 6 条，不计为 6 次模型生成。carpet donor 的先期 dy=-128 方案因覆盖 141 个 GT 阳性像素被机械检查拒绝，尚未产生 donor 图；随后使用已记录固定 dy=-192。

正常核心去除原目标缺陷，异常原条件严格保留原真实 source。修复仅在 R=M 膨胀8像素范围，M+4 内全强度，之后4像素过渡；R 外 source 精确不变。六视图在 G 内跨外围条件精确不变，同外围条件的正常/异常在 G 外精确一致。全四组原像素机械检查差异为 0，实际归一化/填充后的批次检查通过，形状为 **8×3×532×532**（6 张 crossed + 1 张真实正常 + 1 张真实异常支持）。

执行任务与主任务均实际检查四组24视图，接受普通纹理、轻微修补、冠部色块等 flags。正式审核路径仍拒绝 AI-only 池；五项真实来源/准入负向检查通过。来源 role 保持 `anomaly_support_pool`，没有伪装为 normal_train；修复足迹与标签相关性的混杂仍记录。generator_reference_ids 和 mask_training_ids 每产品真实两个，derived_normal_cores 单列；不声称 K_ref=0。

关键冻结记录（服务器项目根下）：

- groups：`data/crossed/pilot_real_support_small_20260917.jsonl`；SHA256 `06b3b9de42c3e3adccaa533614c3ba852ac06ee9cc84a1126cb48d5270785fde`。
- 数据 manifest SHA256：`bc8cf47e0b5ed5b35bf526251255cd3836b34f842a76b53272653f3b36119324`。
- 支持池 SHA256：`66860401ed92c43aa66f1a2b891269e2d4c57706c266bcec8549390e7ec8644c`。
- AI 准入：`reports/small_closed_loop_20260917/pilot_AI_screen.json`，SHA256 `cff2a1b525a08ac3a7f8f29f8161e01efa674f8f2ebc01400f7360ea6ec6979a`；human_admitted=0，reviewers=[null,null]。
- 图像与每组 review：`outputs/small_closed_loop_20260917/groups/`。

## 真实网络与训练核对

DINO 源 pin `7764ea0f912e53c92e82eb78a2a1631e92725fc8`；真实权重 SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`。四运行首真实步前源代码已提交/同步至 **3d86483**；src 在 100→200 恢复期间保持不变。源指纹 `3c8f2f63f3f37616cd0c5a5d61c09df2b0411172aff1a05fb7e6fc78c848396f`。

真实单 L40 预检作业 **2724314** 完成19秒：两个产品真实8张批次同形状替换同伴时 anchor features/logits maxdiff=0；真实输入梯度 anchor 非零有限、其他图梯度 max=0。分类器使用辅助损失同一份归一化 features。主 margin=1.0 的真实 inv/sep 及梯度非零，独立受控 sep 探针也通过。

| 产品 | 方法 | 实际训练 job | 实际步数 | 图像曝光 | 分配 GPU 秒 | 平均秒/步 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| carpet | P | 2724340 | 200 | 1600 | 207 | 0.785 |
| carpet | B3 | 2724341 | 200 | 1600 | 193 | 0.771 |
| hazelnut | P | 2724343 | 200 | 1600 | 205 | 0.786 |
| hazelnut | B3 | 2724344 | 200 | 1600 | 197 | 0.739 |

四运行相同初始模型 SHA256 `2edf1fb48d4e36f84be26e607079fcdf8651a5df2e39db249a0c08e947f114fc`，同产品 step0 概率文件逐字节相同。逐条200步采样/翻转/随机种子、学习率、曝光相同，配置仅 method 不同；P 增加 inv/sep 目标。每个运行 101 个可训练参数张量实际改变，89 个冻结参数张量精确未变，分类头及后6个骨干块实际更新。每步输出/损失/梯度有限，近零特征比例0；训练峰值 allocated 约6.8GiB。

carpet_P 在100步保存、进程退出后重新构造并恢复至200步。恢复收据 step=scheduler_last_epoch=100，模型/optimizer/scheduler/RNG 哈希核对全 true，恢复所用 checkpoint SHA 与保存的100步摘要一致。未进行连续运行的逐位对照，不作该等价声明。

## 损失与预测的可解释记录

以下为各运行首10步与末10步均值，采样固定且公平；不是研究效果结论。

| 产品/方法 | real 首→末 | cross 首→末 | inv 首→末 | sep 首→末 | total 首→末 |
| --- | --- | --- | --- | --- | --- |
| carpet/P | 1.0060→0.0992 | 1.0056→0.0161 | 0.01579→0.000767 | 0.05771→0.000114 | 2.0190→0.1154 |
| carpet/B3 | 1.0060→0.0975 | 1.0056→0.0157 | 未启用 | 未启用 | 2.0116→0.1132 |
| hazelnut/P | 1.0007→0.0662 | 0.9975→0.0311 | 0.01278→0.000606 | 0.02638→0.000270 | 2.0022→0.0974 |
| hazelnut/B3 | 1.0007→0.1052 | 0.9975→0.0325 | 未启用 | 未启用 | 1.9983→0.1377 |

**B3 步日志 inv/sep=0 表示该训练目标关闭，不能解释为特征相等或间隔损失测得0。** 两方法在0/100/200固定预测的同一次前向中另算 diagnostic_inv/diagnostic_sep，不增加训练目标。第200步实际诊断如下：

| 产品/方法 | diagnostic_inv | diagnostic_sep（margin1） |
| --- | ---: | ---: |
| carpet/P | 0.000616 | 0.000577 |
| carpet/B3 | 0.000682 | 0.000164 |
| hazelnut/P | 0.000542 | 0.000307 |
| hazelnut/B3 | 0.000502 | 0.000779 |

12套固定训练示例 RGB/GT/概率与原始概率数组均保留于各运行 `pilot_predictions/step_{000,100,200}/`，哈希/形状/有限值核对通过。执行任务实际查看两个产品共10套独立热图，B3 step0 与 P step0 的图像及概率文件相同，显示一次。主任务另独立查看四个 step200 对照图，观察一致：

- carpet：正常分支及真实正常低响应，原缺陷定位，三个外围条件表现相近；真实支持图存在部分欠覆盖。
- hazelnut：孔洞与细线有响应；P 第200步正常分支出现少量局部误报（局部最高约0.98，整图均值低于0.0008），真实正常也有局部响应，真实支持底行边界没有完全覆盖。
- 固定训练图响应和整体 loss 下降支持工程路径有效；不能说明泛化、可靠校准、收敛，也不能宣称 P 全面优于 B3。

仅按既定规则在200步做一次校准 AP 诊断：carpet P=0.7692、B3=0.7874；hazelnut P=0.3722、B3=0.3284。没有读取保留测试或据此调参/重训。

## 修复过的执行失败

两批启动失败均为0训练步，原日志与 GPU 用量保留：

1. 2724317/18/19/20 合计7 GPU 秒：`python -m defectfirst` 无对应 __main__。最终复用已有 `scripts/train.py`。
2. 2724324/25/26/27 合计17 GPU 秒：CLI 用 YAML 解析 JSON，`1e-05` 等数值变成字符串。`read_config` 的 .json 分支改用 json.loads，YAML 分支保留。CPU **2724334** 用实际四训练 argv 和 carpet_P resume argv 共5条 dry-run 验证数字类型/科学配置等值/参数解析，全部通过，无训练输出创建。

最终 CPU 审核首次2724352因控制脚本的导入路径缺少项目根退出；补充控制环境 PYTHONPATH 后 **2724356 完成13秒、exit0**，输出 `FOUR_REAL_DINO_200_STEP_CLOSED_LOOP_PASS`。该修正未改变 src 或训练记录，GPU用量0。

## 资源与交付位置

本轮新增 **996 GPU 秒 = 0.2767 L40 卡时**：修复151 + 预检19 + 两批0步失败24 + 四训练802。低于本轮7200秒上限。项目此前6195秒，累计 **7191秒 = 1.9975 L40 卡时**；development模型调用由46增至50。

释放记录含全部14个本轮GPU job的 sacct/AllocTRES（每作业单L40、6CPU），四实际训练均 COMPLETED/0:0，失败作业也已结束；全部14个 job 的 squeue 输出为空，reserved_GPU_seconds=0。精确核查 UTC 时间见 `release_receipt.json`。本轮 ledger 已标记 COMPLETE_SMALL_CLOSED_LOOP，没有自动接续训练。

- **本地结果报告**：`F:/DetectFirst/docs/SMALL_CLOSED_LOOP_REPORT_20260917.md`。
- **本地必要记录**：`F:/DetectFirst/reports/small_closed_loop_20260917/` 下 `closed_loop_audit.json`、`GPU_preflight.json`、`training_plan.json`、`entrypoint_prepare.json`、`prediction_AI_inspection.json`、`release_receipt.json` 等小型 JSON；不重复打包大图像或模型。
- **服务器项目根**：`/ssdfs/datahome/u15016/XXY_CVPR/DetectFirst`。
- **服务器四运行根**：`/ssdfs/datahome/u15016/XXY_CVPR/DetectFirst/reports/small_closed_loop_20260917/runs/`。
- **每运行**：config/contract/fairness、steps.jsonl、parameter_audit、initial_model_state、last/best checkpoint与SHA、0/100/200预测；carpet_P另有100步阶段记录与 pilot_resume_receipt。

本轮授权工作结束，主实验与扩量由主任务另行决定。
