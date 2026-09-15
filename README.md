# DetectFirst

**DefectFirst: Learning Beyond Editing Shortcuts for Synthetic-to-Real Defect Segmentation**

仓库名称按项目要求使用 **DetectFirst**；论文方法名和 Python 包名分别为 **DefectFirst**、`defectfirst`。

本仓库实现两项研究贡献：

- **C1：缺陷保持的编辑对照与偏差诊断。** 固定缺陷及必要上下文，让原始外围和两个合格重建外围分别与正常／异常核心交叉组合。
- **C2：直接用于分割的缺陷表征学习。** 每张图独立编码，在分类器直接使用的归一化特征上同时优化编辑稳定性和缺陷可分性。

当前是服务器实验前的代码交付。CPU 测试、人工数据夹具、随机初始化官方 DINO 架构检查均不代表真实数据精度或 GPU 性能。完整验证记录见 [验证报告](docs/VALIDATION.md)。

## 从这里开始

- [服务器逐步启动与回传要求](docs/SERVER_START.md)
- [实现范围、数学约定与对照方法](docs/IMPLEMENTATION.md)
- [人工复核与六视图流程](docs/ANNOTATION.md)
- [实验数据格式](docs/DATA_FORMATS.md)

在 Python 3.10/3.11 的干净环境中安装。CPU 检查示例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==24.3.1
python -m pip install torch==2.5.1 --no-deps --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/dev.lock.txt
python -m pip install -e . --no-deps
python -m pip check
python -m pytest tests -q
python scripts/smoke_test.py --output-dir reports/cpu_smoke
```

Windows 将激活命令替换为 `.venv\Scripts\Activate.ps1`。GPU 安装见服务器文档；不要在服务器上沿用 CPU wheel。

## 代码入口

```text
src/defectfirst/
  assets.py               # 可恢复下载、模型/源码锁、校验
  planning.py             # 正常父图、生成预算、实验配置编译
  data/                   # 官方数据导入、角色隔离、支持集、几何映射
  controls/               # FLUX 原语、掩码初稿、固定核心组合、审核
  models/                 # 官方 DINOv2、融合解码器、固定尺度余弦头
  losses/                 # 主方法和内部强对照目标
  training/               # 固定采样、训练、校准选模、严格断点恢复
  evaluation/             # 原图预测、校准、指标、编辑诊断
  external.py             # 公开方法适配器启动与监督预算核查
scripts/                  # 各步骤命令行入口
tests/                    # 数值参考、行为、集成与恢复测试
configs/                  # 可迁移配置
requirements/             # 已指定版本的依赖；GPU wheel 单独选择
```

主方法与 B0/B1/B2/B3/B4/B5/B7/B8/B9/B10/B11/B12 使用同一训练入口。B9/B11 是明确标注的项目适配，不冒称完整复现原论文。四项公开方法的上游训练、生成和数据适配尚须在服务器阶段逐项接通并验证；编译器将其列为 `WAITING_UPSTREAM_ADAPTER`，不会用占位结果混入主表。

数据、模型、生成图、审核记录、checkpoint 和虚拟环境均不进入 Git。研究结果必须通过真实数据、GPU、人工审核及公平性验收后产生；仓库没有预填提升数值。
