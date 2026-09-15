# 实现与实验约定

## 1. 研究范围

保留 C1「缺陷保持的编辑对照与偏差诊断」和 C2「直接用于分割的缺陷表征学习」。不包含生成加速贡献。方法具备研究价值是研究判断，尚无本项目真实测试结果支持性能结论。

正式单元：VisA 12 类、MVTec AD 排除 carpet/grid/cable/hazelnut 后的 11 类、KSDD2 1 单元。四个排除产品仅用于开发。K_train=5，C_cal=5，K_ref=0；重复 11/22/33。K=0 而 C_cal=5 仍不属于完全无异常监督。

MVTec 采用明确登记的监督重划分，不能与官方仅正常训练结果混名。VisA 保留官方 fewshot CSV 测试集，KSDD2 保留官方测试集。角色交集、解码后像素重复、已知实体跨角色均会阻止训练。无法识别的实体记 unknown，程序不能证明未知实体绝无重叠。

正式 24×200=4,800 个核心尝试，最多 14,400 次扩散调用和 28,800 个视图记录；开发 40 核心、诊断 240 核心，全部预定 15,240 次调用。记录数不等于独立缺陷数。失败尝试保留，重试不能隐身。

## 2. 核心实现

组合：`x[d,e] = alpha * P[d] + (1-alpha) * b[e]`。G 内 alpha=1；P0/P1 在 G 外相同；过渡仅在 Q。所有几何在最终画布确定，合成组训练不再 resize，只做同组同步翻转。PNG 阶段精确不变量、实际归一化输入阶段不变量都检查。

实际缺陷 M 需要审核，不能拿生成区域 R 当真值。保护区初始膨胀 8 像素，过渡宽 8 像素，均需开发验证。未审、原始条件不合格、没有合法 sham 的组不能进入配对训练。当前选择统一拒绝不成组候选，不自动另建可能造成曝光差异的单图池。

DINOv2 ViT-B/14 使用官方代码固定 commit。冻结 patch/position/token 与 blocks 1–6，训练 blocks 7–12、输出 LayerNorm、三层投影和解码器。读取 6/9/12 层 token，经 1×1→256 通道和 stride-4 插值、拼接、两层 Conv/GN/GELU 得到 H。

`h = H / max(||H||₂, 1e-6)`，分类权重同样归一化，`z = 10 * w_normalized @ h`，无 bias；之后只有固定双线性上采样。模型 `forward` 只接受图片。配对、M/G、状态和条件 ID 全部留在损失侧。

L_inv：G 面积占比加权，同状态各无序条件对的平方距离；先位置、条件和状态，再有效组平均。L_sep：M 面积占比加权，异常与正常的所有有序条件对，含同条件，使用 `[m-distance]_+²`。空区域／单条件跳过并返回可反传零。m 默认 1，限定 0<m<2。单位特征平方距离与余弦几何一致；不意味着类间距离自动保证分类间隔，像素监督始终保留。

像素项为有效域 BCE＋仅在正例图启用的 soft Dice。真实正常／异常等权，交叉图按状态／条件均衡。真实训练 mask 用面积占比软标签，测试真值保持原始二值分辨率。默认 4,000 步、AdamW 1e-5/1e-4、warmup 200、cosine、BF16、clip=1。

关闭随机 DropPath/dropout 与 xFormers，使用 PyTorch 路径；CUDA 启动前设置确定性 matmul 工作区。仍须在实际 GPU 核验数值、速度和确定性，不把 CPU 位级恢复结论推广为跨硬件位级相同。

## 3. 内部对照覆盖

| 方法 | 实现 |
|---|---|
| B0 | 真实监督，保持真实曝光，不补齐合成槽 |
| B1/B2 | 原生合成异常不变，正常槽分别为原图／同源 sham |
| B3 | 同审核交叉池的均衡像素 ERM |
| B4/B7/B12 | 输出概率稳定、输出最坏条件间隔、两者联合 |
| B5 | 状态×原始/sham 四组 GroupDRO，log-space 更新权重 |
| B8 | 固定亮度／对比度扰动；同组共用变换参数 |
| B9 | 类均衡像素 SupCon，训练投影和 FIFO memory；保留 direct-h 开关 |
| B10 | 同核心对应位置多正例 InfoNCE；保留额外 L_inv 开关 |
| B11 | 状态条件原始/sham 加权 RBF-MMD；保留仅正常开关 |
| P | 同池 ERM＋共享分类特征上的 L_inv/L_sep |

B9 的采样、memory、投影是本项目公开定义的适配；不是复制原作者所有 hard-mining/region-memory 细节。B11 不是 PCIR 完整复现。比较名称和论文表格必须准确标注。

默认主矩阵 576 项内部主表＋288 项公开方法＋90 项内部诊断，共 954 项计划任务。额外 K、loss、head、InfoNCE+inv、direct-h、normal-only MMD、rho 配置由矩阵生成器列出，尚未执行。检查 `experiments.jsonl` 的实际状态，不把 PLANNED 当完成。

冻结表示重训头已接入训练器：绑定父 checkpoint，冻结所有表示参数和缓冲区，重新初始化并只训练同容量分类权重。编译器在父运行到达前标 WAITING_PARENT_CHECKPOINT。

跨组配对诊断使用 `pairing_mode=pooled_matched/pooled_shuffled`。先按同产品、同生成 recipe、同条件数和 mask 面积的 log2 档分层，在层内形成固定不相交交换对；没有伙伴的组从两版本共同排除。每步两组交叉图＋一份真实监督，三条件时共 14 张图，两版本输入、像素监督及曝光完全相同。每张图先按自身 M 面积权重池化 h，再归一化池化向量；只在分离目标中切换正常组的对应关系。G 内稳定项保留。它检验配对信息，不能把它与主逐像素方案的差异全部归因于配对正确性。

公开方法必须从官方代码接入所需数据格式和合理训练流程，登记 LoRA／额外训练阶段／生成参考来源。当前 `external.py` 是执行和准入接口，不能单凭其存在声称四项方法已复现。

## 4. 校准、恢复与结果

每 200 步及训练终点在固定校准集按 P-AP 选模，平分取更早。最后用 normal_cal 确定阈值：图像 top-0.1% 分数、经验图像 FPR≤5%、按图平均像素 FPR≤1%，严格 `score > threshold`、ties 成组。阈值绑定 checkpoint 和 manifest 哈希。

像素 AP 包含正常图。AU-PRO 使用 8 邻域、全部不同分数阈值、线性插值，分别积分并归一化到 0.05/0.30；没有正区域或背景返回 N/A。原图尺寸预测先去 padding/letterbox，保存连续预测后可独立重算。缺失主表不补零、不挑最佳 seed。诊断置信区间按父图／已知实体聚合，多个 sham 不作为独立样本。

checkpoint 同时保存模型、objective 的 DRO/memory 状态、优化器、scheduler、RNG、下一步位置、曝光计数和最佳校准记录。恢复检查源代码、配置、manifest、支持集和 schedule 指纹；不允许恢复时偷偷换 K 或数据。仅加载自己生成、带校验摘要的可信 checkpoint。

## 5. 主要实现来源

- [DINOv2 官方实现](https://github.com/facebookresearch/dinov2)
- [FLUX Fill 官方模型](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)、[Diffusers Flux API](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- [VisA 官方数据](https://github.com/amazon-science/spot-diff)、[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)、[KSDD2](https://www.vicos.si/resources/kolektorsdd2/)
- [Cross-Image Pixel Contrast](https://github.com/tfzhou/ContrastiveSeg)、[PCIR](https://github.com/JoaoCarv/invariant-anomaly-detection)
- [SuperSimpleNet](https://github.com/blaz-r/SuperSimpleNet)、[AnomalyVFM](https://github.com/MaticFuc/AnomalyVFM)、[SeaS](https://github.com/HUST-SLOW/SeaS)、[O2MAG](https://github.com/echrao/O2MAG)
