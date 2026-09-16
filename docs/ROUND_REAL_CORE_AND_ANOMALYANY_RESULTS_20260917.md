# 真实核心保护与 AnomalyAny：本轮进度和独立验收

本轮目录：`round_real_core_anomalyany_20260916T170331Z`。A 已完成 8 次调用；B 正在服务器登录节点准备权重和独立环境。两个输出目录独立。A 的构造检查通过，AI 语义检查未通过，人工准入为 0；本轮不自动组合、不进入训练。

## A：8/8 构造通过，外围正常性未通过

真实生成条件为四个固定支持 ID，repeat 11、carpet/hazelnut 各前两张，位于当次 K=5 范围。新增真实异常条件使用计数为 4，不能声称 K_ref=0。原路径 test 是历史官方划分；冻结 manifest 的实际角色为 anomaly_support_pool。

- carpet hole：`mvtec/carpet/466ee643aadfca789cab`。
- carpet cut：`mvtec/carpet/5965ac2cc93cb74b6b52`。
- hazelnut hole：`mvtec/hazelnut/9ff8cc7816e85fee2912`。
- hazelnut hole：`mvtec/hazelnut/b2c6ac0e9359a16ddec9`。

CPU job 2721553 成功，确认冻结角色/support 清单、repeat11/K5、源图/官方掩码 SHA、读取权限和几何。官方 M 最近邻变换至 512；G 候选上下文扩张 24 像素，E 为 G 外 96 像素内的局部环带，Q 为内侧 12 像素余弦过渡，外缘也渐变 12 像素。M⊆G、E∩G=∅、E 非空均通过。E 可含局部背景，不声称对象 ROI 或保护上下文已获人审。

FLUX Fill 固定 revision `358293da0354175698b67ec8299acf928313a78a`，50 步/guidance30/BF16/CPU offload，每父图 seed 精确字符串 `14291`、`22592`。GPU job 2721558 在实际 NVIDIA L40（gpu4023）完成，COMPLETED/0:0，252 卡秒。

| 父图后缀 | seed | 调用秒 | final G 最大误差 | raw G 最大漂移 | E 平均 RGB 绝对变化 | E 有变化比例 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| carpet/466ee… | 14291 | 55.57 | 0 | 76 | 20.41 | 97.48% |
| carpet/466ee… | 22592 | 25.51 | 0 | 72 | 20.20 | 97.47% |
| carpet/5965… | 14291 | 25.36 | 0 | 70 | 21.46 | 97.62% |
| carpet/5965… | 22592 | 24.67 | 0 | 72 | 21.20 | 97.55% |
| hazelnut/9ff8… | 14291 | 25.03 | 0 | 46 | 8.10 | 95.84% |
| hazelnut/9ff8… | 22592 | 24.96 | 0 | 54 | 8.03 | 95.91% |
| hazelnut/b2c6… | 14291 | 24.68 | 0 | 56 | 7.15 | 96.00% |
| hazelnut/b2c6… | 22592 | 25.38 | 0 | 61 | 7.97 | 96.35% |

8/8 final 在 G 和 E 外均逐像素等于源输入，E 内均发生实际变化。显存峰值 allocated 24,179,614,208 bytes、reserved 24,352,129,024 bytes。raw 图有 G 内漂移，已保存；final 零误差来自明确回填和局部混合，不是网络学会不变性。

AI 检查已对全部生成 final 做初看，主任务另独立对照本轮允许的源图和 contact grid：

- hazelnut 9ff、variant1（seed14291）在原孔下方新增黑孔样外围结构，AI 初检明确拒绝此候选。
- 同父 variant2 顶部新增浅色结构、壳面重构，语义未通过。
- hazelnut b2 的壳纹和顶部形态变化明显，需要评估外围是否仍为正常条件。
- carpet 局部织纹和边界痕迹发生改变；差分强度不能替代正常材质语义验收。

因此 8/8 机械不变量 PASS 与 AI 语义 NOT_PASSED 严格分开。所有人工 reviewer/hash 保持空、WAITING_HUMAN；已知 AI 拒绝不冒充人审意见。不重抽 A，也不把部分技术成功写成研究质量成功。

服务器 A：`/ssdfs/datahome/u15016/XXY_CVPR/DetectFirst/outputs/round_real_core_anomalyany_20260916T170331Z/A_real_core`。每父图含原尺寸源图/掩码、512 输入、M/G/Q/E/valid、权重、叠图、raw/final 两对、边界放大、逐调用 JSON；另有 manifest、summary、comparison_grid、README、A_ai_visual_check.json。

## B：固定输入已核对，准备后再申请 GPU

官方 [EPFL-IMOS/AnomalyAny](https://github.com/EPFL-IMOS/AnomalyAny) 冻结 commit `585198d594582f41442a779dd2f91288066092e7`，仅提取 notebook 代码，不使用官方示例图或异常参考。独立入口对应实际 MVTec notebook SD1.5/200 配置步/guidance12.5/初始化0.3/scale50，t_start140，实际执行60调度迭代；完整异常注意力优化、提示细化和每步内梯度循环均保留并计数。没有将 run.py 的 SD1.4/50/7.5 默认值当本轮方法。

CPU 输入 job2721576 成功。按冻结源路径次序固定四个 normal_train 父图，均历史 train、label0：

- carpet074：`mvtec/carpet/0333c45d5f605d77c743`。
- carpet000：`mvtec/carpet/b8c2c8b59c0a0ad9f397`。
- hazelnut000：`mvtec/hazelnut/a8225fba0ff8de93bf2d`。
- hazelnut001：`mvtec/hazelnut/d51448dcc78a21c276e7`。

每父图 seed 字符串14291/22592，最多8次调用。提示和一般缺陷概念在出图前冻结：carpet faded、hazelnut crack；不观察保留测试图定提示。完整 PNG 输入留在服务器。预检随后会验证 pinned tokenizer 索引/截断及 foreground mask。官方 masked blending 中白色保留生成 latent、黑色恢复加噪正常 latent；carpet 全白候选，hazelnut 使用作者 median5/threshold127 的 fg_extraction，均未获人审。

HF API 请求旧 `runwayml/stable-diffusion-v1-5` 实际返回 `stable-diffusion-v1-5/stable-diffusion-v1-5`；直接请求新命名返回同 commit `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` 和文件 metadata。该 [镜像模型卡](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 明示不隶属 Runway，故记录其命名解析事实，不冒称组织官方；只使用冻结 SD1.5 基础文件、逐项匹配公布的 LFS SHA，不换异常微调 checkpoint。

官方 [OpenAI CLIP](https://github.com/openai/CLIP) 代码也冻结，ViT-L/14 和 RN50 两个原生加载都预取、匹配作者 URL 中公布的 SHA。RN50 即使 texture loss 权重0也保留原生加载。GPU 调用只从本地路径读取，不能把联网等待记为 GPU 推理。

独立 `.venv-anomalyany` 使用 diffusers0.21.1、transformers4.29.2、huggingface-hub0.24.5；Python3.11/Torch2.5.1CUDA12.4 为运行时适配，环境完整 freeze 留存。原方法 Python 文件保持不改。外层仅固定权重路径、全局 RNG、资源/方法计数，另保存未作为真值的注意力候选图。真实输出和方法兼容性仍待 GPU 执行确认。

B 初次准备 CPU job2721563 因代理仅存在 logina04 而失败；初次输入 lint job2721573 因入口多语句格式失败，未调用模型，已修复并用新 CPU job验证。源代码最终检查 job2721580 成功。失败原日志保留。下载/环境准备改在登录节点，不另建服务器隧道、不等待资产而占 GPU。

服务器 B：`/ssdfs/datahome/u15016/XXY_CVPR/DetectFirst/outputs/round_real_core_anomalyany_20260916T170331Z/B_anomalyany`。当前状态 INPUTS_PREPARED，无实际 B 图，无质量通过主张。reports 同轮保存 B_prepare_login、B_clip_prefetch 进程/日志、资产 metadata/进度、预检及首次60分钟 Slurm入口。

## 资源和后续约束

A新增252 L40秒，先前673秒，累计925秒=0.2569444444卡时；距累计4卡时上限余13,475秒。开发调用旧6+A8=14/120；B本轮含失败最多8次，当前轮最多新增16次。B首次GPU作业1实际L40/6CPU/96G/60分钟，只有全部依赖和服务器CPU预检完成才提交；项目最多同时2实际L40，保护无关ACL作业。

自动检查 `defectfirst` 已绑定本任务，每10分钟检查并按授权继续；下载/排队/健康运行无变化保持安静，真实图片里程碑、完成、失败或需用户行动时报告。B有界执行和记录交付结束后删除检查。当前有外部资产等待，不声称本轮完整执行已结束。

原 FP32/BF16 分割模型的严格1e-5审计仍 FAIL_UNCHANGED；本轮无训练、无正式扩量、无阈值放宽、无自动组合。两线质量分别通过并完成真实人审之后才能另设计组合轮。

## B 准备完成及首次 CUDA 检查（后续更新）

B 的14个 SD 文件共4,266,680,108 bytes、两个 OpenAI CLIP 共1,188,595,637 bytes均已下载校验，独立环境完成、pip check通过。首次CPU预检发现 sparse source缺少MANIFEST.in，CLIP wheel因此漏包原版词表；纳入同commit原MANIFEST重装解决，不改算法。随后只读Python导入产生的__pycache__被clean守卫误判；仅本clone git/info/exclude忽略编译缓存并在wrapper禁写bytecode，原方法代码未改，失败日志保留。

作者fg_extraction threshold127在normal hazelnut000仅有237个亮像素，normal001为空。早期状态误写“000为空”，已在预检记录中更正；源图、父ID、提示和seed未换。任何GPU调用之前，两张正常榛子统一采用gray/median5/Otsu候选白编辑mask，方法名明确为**AnomalyAny＋正常输入前景适配**。原正常初始化、注意力梯度、提示梯度均保留，不能声称严格原样复现。

CPU job2721686通过，两个normal_train source/mask/overlay已实际AI查看，白区覆盖榛子主体，背景保持黑，非空非全图。000的Otsu55、面积86,081（32.84%）、9个8连通块、最大块98.88%；001的Otsu55、面积73,673（28.10%）、16块、最大块99.00%。阴影暗面/边缘有缺口、小块残留，因此仅有限技术执行AI预检PASS_WITH_FLAGS，不冒充完整物体分割、人工ROI批准或生成缺陷质量通过。精确mask/overlay哈希保存在B_foreground_ai_precheck.json，提交守卫核对该版本。入口CPU最终检查2721687通过，源码提交913e61f。

首次GPU job2721689分配gpu4020，在实际CUDA可用性/设备守卫处15秒退出；尚未加载pipeline、没有开始任何生成调用，8个call仍PLANNED。短设备probe2721692确认nvidia-smi实际NVIDIA L40/driver580.65.06，但独立环境CUDA初始化报No CUDA GPUs available，3卡秒后退出。仍保留实际L40严格守卫，不接受只凭Slurm Gres标签。

当前项目累计925+15+3=943 GPU秒；development仍14/120，B实际generation attempts=0、输出图=0。probe2721694已在曾成功完成A的gpu4023排队，2分钟封顶，6CPU/96G/ntasks1，比较独立环境与原FLUX环境的CUDA可见性/绑定/名称；不加载模型、不SSH未分配节点、不改ACL。需读该probe实际证据后用新的有界retry launcher继续，不能绕过已有首提交防重标记。探针也计入4卡时预算。最新状态以同轮round_status.json和原始sacct/日志为准。
