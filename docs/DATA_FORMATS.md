# 文件格式与单步配置

所有输入 artifact 路径相对项目 root，CLI 用 `--root` 切换设备目录。配置不得写本机 F 盘路径。JSONL 每行一个对象，禁止 NaN/Inf；缺失指标使用 null。

## 原始数据

MVTec：`data/raw/mvtec/<product>/{train,test}/<type>/*.png`，缺陷 mask 在 `ground_truth/<type>/<stem>_mask.png`。

VisA：官方 `Data/Images` 和 `Data/Masks` 保持原布局，以固定 commit 的 `2cls_fewshot.csv` 为准。KSDD2：`data/raw/ksdd2/{train,test}/<id>.png` 与 `<id>_GT.png`。

`prepare_data.py` 写 `Sample` 字段：sample_id、dataset、product、source_path、mask_path、label、original_split、role、image_sha256、pixel_sha256、mask_sha256、defect_type、entity_id、entity_id_source。实际缺少物理实体标识时用 unknown。原始多类 mask 原样保留，读取时统一 >0 为异常。

## ROI 审核登记

`prepare_rois.py` 生成的每条记录含 parent_id、unit、canvas、normal_preview、normal_sha256、roi_path、roi_review。修改 ROI 后把 roi_review 填成：

```json
{"reviewer":"reviewer-01","sha256":"填写最终 ROI PNG 的 64 位 SHA256"}
```

只有单通道、最终画布内的 ROI 才可接受；正常父图也必须匹配冻结清单。生成计划明确填写 phase：development/training/diagnosis。

## 原语与单组构造

`generate_primitives.py` 保存每组 `primitive.json`，包括 normal、defect、两个 shams、region、valid、geometry、parent_id、unit、role、recipe_sha256 和 task_sha256。原始生成结果保留；最终交叉视图将无效 letterbox 区域恢复为同一正常留白。

掩码初稿配置：

```yaml
primitive: data/primitives/development/<group_id>/primitive.json
threshold: 20
```

构造配置：

```yaml
primitive: data/primitives/development/<group_id>/primitive.json
mask: annotations/<group_id>/M_reviewed.png
guard: 8
collar: 8
# protection: annotations/<group_id>/G_reviewed.png
```

命令：`python scripts/build_crossed.py --config <配置文件> --output-dir data/crossed/<group_id>/v1`。输出目录必须为新版本。

将 candidate.json 对象逐行收集为 groups_candidate.jsonl；审核表逐行合并为 reviews.jsonl。运行 `audit_controls.py`，训练只读取生成的 groups_accepted.jsonl。

## 审核 JSONL

```json
{"group_id":"group-001","content_sha256":"候选版本哈希","reviewer":"reviewer-01","seconds":45,"decision":"accept","conditions":{"0":{"mask_correct":true,"normal_valid":true,"defect_preserved":true,"boundary_valid":true,"editing_effective":true},"1":{"mask_correct":true,"normal_valid":true,"defect_preserved":true,"boundary_valid":true,"editing_effective":true}},"reason":""}
```

此行只解释字段，不能作为实际人工审核导入；真实 group ID、哈希、条件数和判断由离线网页输出。程序拒绝缺少判断、旧哈希、重复 reviewer、不足双审的记录。

## 重算指标与测速

重算配置：

```yaml
evaluation_dir: runs/<运行ID>/evaluation
manifest: data/manifests/manifest.jsonl
```

测速配置：

```yaml
run: runs/<运行ID>
device: cuda:0
image: data/raw/<一张允许用于测速的正常图路径>
warmup: 20
repeats: 100
```

诊断配置：

```yaml
run: runs/<运行ID>
device: cuda:0
thresholds: runs/<运行ID>/calibration/thresholds.json
groups: data/diagnosis/groups_accepted.jsonl
manifest: data/manifests/manifest.jsonl
```

## 阶段回传

validate_handoff 配置包含 `reports`（阶段 JSON 路径列表）与可选 `files`（打包文件白名单）。阶段报告必需字段为 stage、batch_id、engineering_status、research_observation、code_commit、config_sha256、manifest_sha256、commands、criteria、evidence_paths、failed_items、pending_human_review、running_jobs、next_action。

PASS 的 command 记录必须带 started_at、ended_at、exit_code=0、log_path；criterion 带 passed=true 和 evidence_path。原始大文件用哈希和固定来源定位，不塞进回传包。
