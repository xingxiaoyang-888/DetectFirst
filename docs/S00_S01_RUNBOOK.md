# 首批服务器执行安排（R1–R3 修订；尚未执行）

适用 Linux/Bash；服务器系统待确认。仅在接入信息、工作目录和资源分配确认后运行。以下变量须填写实际值，不直接照搬占位符。已有仓库先检查 remote/status/HEAD，不覆盖修改或重新克隆替代仓库。

## A. 公共准备与命令记录

新服务器没有仓库时，在确认的父目录执行 `git clone https://github.com/xingxiaoyang-888/DetectFirst.git`。进入真实仓库，核对 HEAD 为本轮修复交接报告中的 commit（main 基线为 631bce25a9922f933c80bd06d41cbce06ba48f27）；不一致先保存实际差异。不要对已有工作树强制 reset。

本轮修复位于 `codex/server-audit-evidence` 分支。若远端尚无此分支，先通过交接的 Git bundle 或补丁导入；不能只 clone main 后便声称运行的是修复提交。部署来源和实际 commit 均记入公共日志。

公共准备只在一个终端执行一次。使用已有 Slurm/tmux 管理长任务，记录任务 ID。两个阶段使用相同 PROJECT_ROOT/BATCH、不同 STAGE。record 为每个 label 原子创建独立目录，重复 label 返回 73 且不执行命令；显式 if/else 保证 set -e 下先保存真实退出码再返回。重跑用新 BATCH，下载缓存保留。

```bash
export PROJECT_ROOT='/填写已确认的服务器仓库绝对路径'
export BATCH='S00_S01_填写UTC批次'
cd "$PROJECT_ROOT" || exit 1
# 父目录可复用，批次目录必须全新；失败时不要改为 -p 绕过。
mkdir -p reports logs
mkdir "reports/$BATCH" || exit 1
mkdir "logs/$BATCH" || exit 1
export STAGE=common
source scripts/record.sh
record git_head git rev-parse HEAD
record git_status git status --porcelain
record git_remote git remote -v
record hardware nvidia-smi
record gpu_inventory nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,driver_version --format=csv
record disk df -h .
record ram free -h
```

必须返回上述日志；确认 GPU 分配、当前占用、磁盘配额和写权限。只读查询可以显示整机设备，但计算只能使用分配卡。磁盘预算另含缓存、归档、解压、模型、生成图、checkpoint、临时文件，先从资产元数据估算。无 GPU 不阻塞已获准的 S01 下载。

## B. 终端一：S00 安装与检查

下面 cu124 安装仅在驱动兼容时执行，否则记录实际条件并调整兼容 wheel，保留版本差异。不要多个进程修改同一环境。已有环境先核查，示例用于新的 .venv-server。

```bash
# 在终端一设置已确认的 PROJECT_ROOT 和相同 BATCH，再执行：
cd "$PROJECT_ROOT" || exit 1
export STAGE=S00
source scripts/record.sh
record train_venv python3.10 -m venv .venv-server || exit 1
record pip_version .venv-server/bin/python -m pip install pip==24.3.1 || exit 1
record torch_install .venv-server/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || exit 1
record dev_install .venv-server/bin/python -m pip install -r requirements/dev.lock.txt || exit 1
record generation_install .venv-server/bin/python -m pip install -r requirements/generation.lock.txt || exit 1
record editable_install .venv-server/bin/python -m pip install -e . --no-deps || exit 1
record pip_check .venv-server/bin/python -m pip check || exit 1
record freeze .venv-server/bin/python -m pip freeze
export CUDA_VISIBLE_DEVICES='填写已分配的单卡UUID或编号'
record S00 .venv-server/bin/python scripts/check_env.py --config configs/bootstrap.yaml --output-dir "reports/$BATCH/S00"
```

安装 Torch 保留依赖解析，使 Linux CUDA 运行库一并安装；设备实际 freeze 是重建证据。S00 返回 `reports/$BATCH/S00/env.json`、安装日志、freeze、硬件记录和 GPU 分配表。判据：退出码 0、存在实际 CUDA GPU、每卡 backward_passed=true、所需包版本完整、磁盘预算可满足。总状态已纳入每卡反传与异常；FAIL 必须查看 failures、逐卡 error/traceback 和 nvidia_smi_check。无可见 GPU 且无执行错误时 WAITING_RESOURCE。报告仍不能替代 GPU 分配、软件完整性和空间预算的阶段复核。

独立 S00_check.json 由执行者汇总，不是 check_env.py 自动产物；填入所有真实命令与判据。S00 不等待完整 FLUX 权重。真实 DINO 审核属于资产到达后的 S05，实际模型审核按 cfg.amp 执行，见本页 G 节；目前仍待服务器执行。

## C. 终端二：S01 轻量下载，与 B 同时推进

下载 CLI 延迟导入，不需要 Torch。使用独立环境及 PYTHONPATH，避免安装整个项目依赖而等待 CUDA。

```bash
# 在终端二设置已确认的 PROJECT_ROOT 和相同 BATCH，再执行：
cd "$PROJECT_ROOT" || exit 1
export STAGE=S01
source scripts/record.sh
record download_venv python3.10 -m venv .venv-download || exit 1
record download_deps .venv-download/bin/python -m pip install PyYAML==6.0.2 filelock==3.19.1 huggingface-hub==0.35.1 || exit 1
record download_freeze .venv-download/bin/python -m pip freeze
export PYTHONPATH="$PROJECT_ROOT/src"
# 包装器先保存到不可覆盖的批次目录，完成后才更新 latest 索引。
# 显式 if 允许 INCOMPLETE/失败后继续读取证据，不让 set -e 跳过回传。
if record S01_download .venv-download/bin/python scripts/download_batch.py --config configs/assets.yaml --output-dir "reports/$BATCH/S01_download" --latest-dir reports/S01_download; then
  download_rc=0
else
  download_rc=$?
fi
printf 'Download command exited %s; inspect batch receipts.\n' "$download_rc"
```

download_batch.py 自动将实际配置、SHA256、argv、UTC、退出码和原始日志保存在 reports/$BATCH/S01_download；真实收据直接写入该目录的 receipts/，随后复制至可变 reports/S01_download。逐批目录拒绝复用，发布最新索引失败也保留批次证据并返回非零。最新 batch_index.json 只列本次收据；最新目录其他旧文件不能冒充本批输出。异常/中断时已返回的收据照实保留（强制杀进程可能来不及写结束记录，应标 INCOMPLETE）。确认官方访问条款、HF 权限、磁盘预算和已有缓存。token 仅在服务器安全配置，不放命令参数、仓库或回传日志。默认资产并发 2；不要多进程下载同一路径。

默认配置的 MVTec/KSDD2 是 manual，因此预计整体 INCOMPLETE、退出码 2；不能将此当全阶段 PASS，也不阻止 DINO/VisA 等已 READY 资产继续使用。读取 `reports/$BATCH/S01_download/receipts/<id>.lock.json` 可逐项推进；download_status.json 的 complete 仅表示任务均返回。真正完整要求各资产 READY/有独立审核通过的人工导入凭据，并完成数据划分。

MVTec/KSDD2：在官方网站确认确切归档 URL 后登记到新版配置的 http 条目，path 指向 data/archives 中的归档；登记可得官方摘要/字节数。现成文件无下载收据时脚本会拒绝自动认可，先核实来源与摘要；不伪造收据。此处不预填可能失效的下载 URL。

每项返回：固定来源/revision、文件清单、字节数、SHA256、官方摘要验证标记、下载记录和退出码；HF 返回完整组件文件清单，S02 再实际加载。无需回传整套模型。

## D. 最终提交回归（依赖到达即做）

训练环境与 dinov2_source READY 后运行；不必等待 FLUX 权重。只运行一轮整合检查，避免先后重复执行同一套 smoke/pytest。

```bash
record server_release .venv-server/bin/python scripts/verify_release.py --output-dir "reports/$BATCH/server_release" --include-dino --include-generation
```

返回 verification.json、tests.xml、各项原始日志及 smoke 报告；判据为退出码 0、报告 PASS、源码指纹不变、所有子命令退出 0。DINO 为随机架构夹具，FLUX 为接口导入，均不证明真实模型效果。

## E. 安全解压与 S01 划分

先查看归档顶层结构，避免目录多套一层。VisA 示例（已下载归档且目标目录尚空时）：

```bash
(set -o noclobber; printf '%s\n' '{"archive":"data/archives/VisA_20220922.tar"}' > "reports/$BATCH/extract_visa.json") || exit 1
record extract_visa .venv-download/bin/python scripts/extract_archive.py --config "reports/$BATCH/extract_visa.json" --output-dir data/raw/VisA
```

MVTec/KSDD2 按实际官方归档分别建配置并用同一 extract_archive.py，输出新的空目录。脚本拒绝覆盖、链接和路径越界；失败先检查归档，不强制解压覆盖。

三套数据完整后，按真实结构核对 configs/protocol.yaml，再执行：

```bash
record S01_prepare .venv-server/bin/python scripts/prepare_data.py --config configs/protocol.yaml --output-dir data/manifests
record config_hash sha256sum configs/assets.yaml configs/bootstrap.yaml configs/protocol.yaml
record manifest_hash sha256sum data/manifests/manifest.jsonl data/manifests/support_ids.json data/manifests/protocol.data.lock.json
```

返回 inventory.csv、manifest.jsonl、support_ids.json、split_audit.json、protocol.data.lock.json。判据：退出 0、VisA=10821/MVTec=5354/KSDD2=3335、角色和内容泄漏检查通过、官方测试保留、训练支持和校准各 5、24 正式与 4 开发分离、seed 11/22/33。逐产品计数从 inventory 复核。未知实体不能写成已证明无实体重叠。冻结后不覆盖 manifest；失败保留日志并使用有明确版本的新目录恢复。

不必等待全部数据才开展独立 DINO 检查；开发数据子集准备若需要单独配置，必须单独标 DEV_READY，保留正式完整计数配置。不得把减少 expected_counts 用作全量验收通过。

## F. 每步退出码与回传

安装/系统命令按实际退出码记录；CLI 异常一般 1，INCOMPLETE/WAITING_* 一般 2；0 仍需人工核对报告判据。verify_release 失败为 1。任何缺失日志、未实际执行或不满足前置条件均不能填 PASS。

每个 S00/S01_check.json 使用 DATA_FORMATS.md 的完整字段，并补 run_id、source_sha256、review_status=NOT_REVIEWED、accepted_pool_sha256=null（尚不存在）。commands 从 logs/$BATCH/$STAGE/$label/{argv,started,ended,exitcode,output.log} 读取实际信息；下载内部命令见批次 command.json；criteria 写实际值和证据路径。下载/安装日志回传前检查意外凭据，不回传认证信息。

阶段包包含源码快照（可用 git archive）、配置与 diff、日志、JSON/CSV/XML、checksums.sha256。原始环境与资产不足则保持 WAITING_ACCESS/WAITING_RESOURCE/INCOMPLETE；研究观察保持 NOT_EVALUATED。S00/S01 达到依赖条件后再安排 S02/S05，本文件不自动授权或触发全量实验。

## G. 真实 DINO 精度审核（S05；资产就绪后）

此项不需真实数据或 FLUX，但需要已核实的 DINO 源码和权重。在服务器用以下代码从下载收据创建两份配置；文件用 x 模式拒绝覆盖。示例使用开发 carpet，网络画布/结构保持既定配置；不是正式训练。

```bash
export STAGE=S05
source scripts/record.sh
# BATCH 必须已 export；该配置准备命令也留存原始记录。
record prepare_audit_configs .venv-server/bin/python - <<'PY'
import json, os
from pathlib import Path
import yaml
root = Path.cwd()
batch = root / "reports" / os.environ["BATCH"]
receipt = json.loads((batch / "S01_download/receipts/dinov2_weights.lock.json").read_text())
assert receipt["status"] == "READY"
config = yaml.safe_load((root / "configs/train.yaml").read_text())
config["product"] = "mvtec/carpet"
config["model"]["weights_sha256"] = receipt["sha256"]
for precision, amp in (("fp32", "none"), ("bf16", "bfloat16")):
    config["amp"] = amp
    with (batch / f"audit_{precision}.yaml").open("x") as stream:
        yaml.safe_dump(config, stream)
PY
export CUDA_VISIBLE_DEVICES='填写已分配的单卡UUID或编号'
record audit_fp32 .venv-server/bin/python scripts/audit_model.py --config "reports/$BATCH/audit_fp32.yaml" --output-dir "reports/$BATCH/S05_fp32"
record audit_bf16 .venv-server/bin/python scripts/audit_model.py --config "reports/$BATCH/audit_bf16.yaml" --output-dir "reports/$BATCH/S05_bf16"
```

若 shell 使用 set -e，任一失败会先保存报告和日志再退出；排查后在新批次恢复，不能覆盖旧输出。两种精度可分别调度，FP32 通过不自动批准 BF16。BF16 命令内部另保留同一模型/输入上的 FP32 对照，供数值诊断。

返回两份配置、原始日志、model_audit.json、权重收据及源码版本。必须检查 requested/effective precision、BF16 支持和实际算子探测、raw_features dtype、FP32 features/low_logits/logits、全部有限性、直接分类路径、两项辅助梯度的逐参数表，以及冻结层无梯度/后六层和 decoder 有有限非零梯度。

独立性保留绝对 atol=1e-5、rtol=0，FP32/BF16 相同且在报告显式登记。分别记录单图与批量、同形状置换、替换同批伙伴和重复执行差异。低精度超阈值返回 FAIL 和 LOW_PRECISION_FAILURE_REQUIRES_DIAGNOSIS；只能先诊断算子/形状相关误差与结构耦合，不能静默放宽阈值或宣布已证明只是舍入误差。跨精度单图差异仅作诊断，不用作独立性验收。

本轮新增的回归用例包含在 D 节整合回归中。若只需故障定位，可在服务器单独执行（不要与整合回归无理由重复）：

```bash
record regressions .venv-server/bin/python -m pytest tests/unit/test_server_audits.py tests/unit/test_evidence_capture.py -q --junitxml="reports/$BATCH/audit_regressions.xml"
```

服务器 Linux/Bash 回归不得跳过两项 shell 日志检查。所有新增用例本机均未运行；实际 DINO/BF16、完整回归和下载行为仍待服务器验证。
