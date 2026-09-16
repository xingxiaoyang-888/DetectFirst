# 服务器执行顺序与回传

目标是先证明代码和对照构造正确，再使用真实数据评估 C1/C2。资源池可提供 8×L40S、8×A800；首先申请最多四张 L40S，按独立产品任务分配，避免环境安装和人工审核期间占卡。A800 显存和互联由实际预检确认。

首批操作以 [S00/S01 修订执行安排](S00_S01_RUNBOOK.md) 为准：公共记录只执行一次，阶段日志分目录且拒绝覆盖；下载用 `download_batch.py` 保存逐批原始收据。下文为阶段总览，命令执行时使用该文档的 `record` 包装。

## 0. 拉取与安装

```bash
git clone https://github.com/xingxiaoyang-888/DetectFirst.git
cd DetectFirst
git rev-parse HEAD
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==24.3.1
# 以下 wheel 对应 CUDA 12.4，先确认驱动支持；CPU wheel 不用于 GPU 实验。
python -m pip install torch==2.5.1 --no-deps --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/dev.lock.txt
python -m pip install -r requirements/generation.lock.txt
python -m pip install -e . --no-deps
python -m pip check
mkdir -p reports logs
python -m pip freeze > reports/server_environment.freeze.txt
```

代码的锁文件固定通用 Python 包和训练／生成直接依赖。GPU wheel 会引入与 CUDA 匹配的 NVIDIA 运行库，以该设备实际 `pip freeze` 补充锁定；不要把 Windows 的 CPU freeze 当成 Linux CUDA 环境。

先运行 `python scripts/smoke_test.py --output-dir reports/server_cpu_smoke`。它使用人工夹具和测试网络，不下载完整数据，不占 GPU；输出 `smoke_report.json`。随后执行 `python -m pytest tests -q --junitxml=reports/tests.xml`、Ruff 和 mypy。不要将 CPU 测试网络用于研究表格。

## 1. S00 与 S01 并行

先编辑 `configs/assets.yaml`，确认数据归档路径。VisA 与模型入口已配置；MVTec/KSDD2 从官方入口获取归档后，将 `manual` 项改为 `http`（确切官方文件 URL）或先在服务器手动存放并保留下载记录。FLUX 若有访问限制，在官方账号界面接受条款；令牌只放服务器环境，不写入配置、仓库或回传包。

两个终端分别运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_env.py --config configs/bootstrap.yaml --output-dir reports/S00
```

```bash
python scripts/download_batch.py --config configs/assets.yaml --output-dir reports/<新批次>/S01_download --latest-dir reports/S01_download
```

下载初始并发为 2，单个资产完成后即写 `<asset_id>.lock.json`，因此不必等所有下载结束才开始对应模型检查。`INCOMPLETE/WAITING_ACCESS` 不代表其他已完成资产不可用。不要让多个进程同时安装同一个虚拟环境。

模型 checksum 来自下载收据；没有官方摘要时仅是本地一致性校验，不称官方真实性证明。FLUX 使用完整 Diffusers 布局和固定 revision。生成配置读取 `reports/S01_download/flux_fill.lock.json`。

回传：Git commit、`env.json`、设备型号／显存、资产锁和 `download_status.json`、实际命令与退出码。无需回传整套模型。

环境总状态包含每卡反传和执行异常，失败保留逐设备原因。下载历史验收以新批次的 `receipts/` 和 `command.json` 为准；latest 只是可变索引，不能代替历史证据。

## 2. S01 数据核验与角色冻结

安全解压示例：建立 JSON 配置 `{"archive":"data/archives/VisA_20220922.tar"}`，执行 `python scripts/extract_archive.py --config <配置文件> --output-dir data/raw/VisA`。检查真实归档内部是否已有外层目录，再调整 `configs/protocol.yaml`；不要靠改文件计数“修好”数据。

```bash
python scripts/prepare_data.py --config configs/protocol.yaml --output-dir data/manifests
```

必须返回 `inventory.csv`、`manifest.jsonl`、`support_ids.json`、`split_audit.json`、`protocol.data.lock.json`。若存在跨角色重复内容或已知实体冲突，先处理来源与分组，重新生成一个有明确版本的新划分。固定数据后不覆盖 manifest。

训练层按角色白名单读取文件，校准集与测试集不能用作梯度训练或生成输入。主监督设定是 K=5、校准异常 5 张；不足 K=10 的产品必须在对应消融记录 N/A。

## 3. S02–S04 先完成开发 40 组

```bash
python scripts/prepare_rois.py --config configs/roi_development.yaml --output-dir annotations/development_rois
```

审核正常父图的可编辑表面，修订 ROI；合并为 `annotations/development_rois_reviewed.jsonl`，填写 reviewer 和最终 ROI 的 SHA256。ROI 仅基于正常图，可复用多次；它不是缺陷真值。详细格式见 [人工复核](ANNOTATION.md)。

```bash
python scripts/plan_generation.py --config configs/plan_generation_development.yaml --output-dir data/manifests/development_generation
```

将 `configs/generation.yaml` 的 tasks 指向刚生成的 `generation_tasks.jsonl`。先将正式任务清单中的第一组导出到单独清单运行；后续完整清单使用同一输出目录，已成功调用会校验后复用，失败调用保留且不自动重试。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_primitives.py --config configs/generation.yaml --output-dir data/primitives/development
```

初始 FLUX 部署启用 CPU offload，适合先在 48GB 卡上检查。每次调用记录完整 pipeline 耗时（包含编码／传输／去噪／解码）、allocated/reserved 峰值显存；PNG 保存耗时不混入 pipeline。单组跑通后，从开发 120 次调用中取至少 20 次记录测速。预热也属于生成预算，不另生重复样本。不要因速度慢而擅自缩减正式 200 核心预算。

使用 `python scripts/summarize_generation.py --config configs/generation_benchmark.yaml --output-dir reports/S02_timing` 汇总逐调用 JSONL、CSV 和核心／sham 的均值、P50/P95。预热调用 ID 在配置中显式列出；它们仍计入总成本。成功计时不足 20 次会明确返回未达到测速要求。

每组原语完成后：

1. `propose_annotation.py` 生成 **待审核** 的 M 初稿。低对比度缺陷可能需要人工修正；不直接以 R 当 M。
2. `build_crossed.py` 构造一组六视图及 M/G/Q、联系图。一个核心共享一份实际 M，正常状态标签为零。
3. 将 `candidate.json` 合并为 `groups_candidate.jsonl`，运行 `export_review.py`，使用导出的离线 `index.html` 独立填写审核。
4. 合并逐人 JSONL，运行 `audit_controls.py`。返回 accepted/rejected_or_pending、footprints、审核耗时。原始条件和至少一个 sham 必须合格。
5. 检查原生异常与 sham 足迹是否覆盖要研究的非缺陷编辑变化。数字不变量通过不等于研究假设已成立，仍需审阅正常语义和边界。

所有像素不变量在最终模型输入还会再次检查。生成规则、ROI/R、提示、保护区和准入规则只在开发产品修改，随后冻结。详细单组配置见 [数据格式](DATA_FORMATS.md)。

## 4. S05–S06 实际网络、200 步与开发对照

将 `dinov2_weights.lock.json` 中的 sha256 写入 `configs/train.yaml` 的 `model.weights_sha256`，确认 DINO 源码 commit 一致。

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/audit_model.py --config configs/train.yaml --output-dir reports/S05
```

返回实际分辨率的输入形状、单图独立性误差、逐参数冻结与梯度表。这个步骤在服务器验证真实预训练权重，本地的随机权重架构检查不能替代它。

审核实际遵循 `cfg.amp`。BF16 审核同时保留独立 FP32 对照，记录支持、实际输出精度、有限性、梯度以及各类独立性误差；归一化特征和分类头保持 FP32。原有绝对独立性阈值 1e-5 不自动放宽；低精度失败须诊断，FP32 PASS 不转移为 BF16 PASS。具体两种精度命令见修订执行安排 G 节。

训练正式产品前，先复制配置到四个开发产品，保持一致的支持与审核池；任务来源不跨产品。P、B3、B10、B12 是最先完成的强对照。GPU 检查与其他数据下载可以并行。

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config <开发配置> --output-dir runs/<开发运行ID> --stop-after 200
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config <同一配置> --output-dir runs/<同一运行ID> --resume
```

200 步是实际训练总计划的前 200 步，配置中的 steps 仍保留 4,000；不要把 steps 改成 200 再恢复到 4,000，否则学习率与计划已变。每步有原始耗时、loss、梯度范数、近零特征比例和曝光数。GPU 速度只来自真实记录。

对一个开发运行另做 N 步连续与 N/2 恢复对照。检查 model/objective/optimizer/scheduler/RNG 和逐步输入。CPU 测试已覆盖 P、DRO、memory，但 GPU 上也必须验证。

最多九组开发候选、同等搜索预算，在正式评估前冻结配置。负结果保留，不因 P 没胜出而删产品或换 seed。

## 5. S07 正式池与矩阵

将 ROI/生成计划 phase 分别改为 `training` 和 `diagnosis`。编译器自动使用 24×200、6×40 的固定范围；不要把整个产品池到齐前不断变化的接受清单喂给不同方法。可以按产品冻结，已就绪产品训练与其他产品生成并行。

```bash
python scripts/compile_experiments.py --config configs/compile.yaml --output-dir configs/compiled
```

运行 `experiments.jsonl` 中的命令，每个配置和运行目录独立。训练配置默认保留 4,000 步，K/seed 嵌套，K=5 主结果不重复计算。设置 `CUDA_VISIBLE_DEVICES=2` 后配置里的 `cuda:0` 指的是该进程可见的第一张卡。

四张 L40S 的起始安排：一张继续生成，其他卡按独立就绪产品／方法训练；训练测速时独占对应卡。使用服务器已有 Slurm/tmux，不额外启动长期常驻调度器。

公开方法 E_SSN/E_AVFM/E_SEAS/E_O2MAG 的代码适配任务会明确标 `WAITING_UPSTREAM_ADAPTER`。需要根据官方当前接口接入统一数据角色，逐项审核额外 mask/embedding/参考图来源。`run_external_baseline.py` 只负责审核配置和执行已复核的官方适配器；不把可运行命令当成方法复现证据。这些格子未完成前不能宣称主表齐全。

冻结表示重训头的任务等待对应主运行 best.pt；到达后重新编译即可生成绑定其 checksum 的 refit 配置。仅分类权重更新，表示及缓冲区保持冻结。

配对机制配置为 pooled_matched/pooled_shuffled。两版本在相同分层、相同样本和相同曝光上训练；没有足够可交换伙伴时明确停止该项，不自行跨产品借 mask。它们使用独立的 pooled schedule，不能与主逐像素运行混用计数。

## 6. S08–S10 评估、回算与交接

为每个运行修改 run 路径，分别执行：

```bash
python scripts/calibrate.py --config configs/calibrate.yaml --output-dir runs/<运行ID>/calibration
python scripts/evaluate.py --config configs/evaluate.yaml --output-dir runs/<运行ID>/evaluation
```

`recompute_metrics.py` 必须从保存的预测重算一致。`diagnose.py` 仅接收 normal_diag 独立父图及原始固定阈值。`measure_resources.py` 默认单图、预热 20、计时 100 次，分别报告网络与含前后处理的耗时。原始预测、阈值、输入与 checkpoint 的校验摘要一起保存。

`aggregate.py` 默认要求完整 24 单元和三种 seed；允许整理进行中结果时明确 `allow_incomplete: true`，缺失项仍保留。统计按产品宏平均，不能把数万个像素视为数万个独立实验。

每阶段回传至少包含：

| 字段 | 需要的实际内容 |
|---|---|
| 阶段／状态 | Sxx、批次、工程状态与研究观察分别填写 |
| 代码 | commit、源代码指纹、改动摘要 |
| 数据 | manifest、接受池、支持 ID、配置摘要 |
| 命令 | 完整 argv、开始／结束时间、退出码、原始日志 |
| 证据 | JSON/CSV、测试 XML、必要小图与异常样例 |
| 未完成项 | 具体依赖、失败原因、恢复入口 |

`validate_handoff.py` 检查每个 PASS 是否有实际命令、通过判据及证据文件，再打包指定的小型材料。不要回传 token、cookie、整套模型或完整数据。原始运行报告中的 `research_observation` 初始为 NOT_EVALUATED，不自动把程序成功转成论文正结果。
