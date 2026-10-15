# Lightweight_Gate_Face_Recognition_Liveness_Detection

Official implementation of "Lightweight Gate Face Recognition and Liveness Detection in Unconstrained Environments" (submitted to *Journal of Electronic Imaging*). **CD-MAGNet**: a lightweight multitask network with a shared DINOv2 ViT-S/14 encoder for face identity classification and presentation attack detection.

## 方法概述（Method Overview）

CD-MAGNet 面向机场门禁等非受限环境，单网络同时完成四类任务：

| 分支 | 任务 | 说明 |
|---|---|---|
| ID | 人脸识别 | ArcFace 度量学习头（s=30, m=0.35），512 维嵌入 |
| FAS | 活体检测（防伪） | 全局 + 局部纹理双分支，含 DPM 退化感知模块 |
| OCC | 口罩/遮挡检测 | 识别被遮挡样本，避免错误身份判定 |
| IDE | 环境感知 | 识别成像环境，配合 GRL 梯度反转做特征解耦 |

核心设计：

- **共享 DINOv2 ViT-S/14 编码器**（384 维，冻结/部分微调策略）
- **DPM（Degradation-aware Perception Module）**：显式建模非受限环境下的图像退化
- **FAS 局部纹理分支**：补足局部伪造痕迹（摩尔纹、屏幕边缘等）
- **特征解耦训练（disentangle）**：正交约束 + 梯度反转层（GRL），让身份特征与防伪/环境特征互不干扰
- **两阶段训练**：Stage 1 以身份识别为主；Stage 2 加入 FAS/OCC 并保护 ID 性能（分组学习率 + 冻结策略）

## 目录结构（Repository Layout）

```
.
├── resnet50-V0/            # 基线：手写 ResNet50 主干的 CD-MAGNet（不依赖 torchvision）
│   ├── config.py           # 数据路径、损失权重、评分权重
│   ├── models.py           # ResNet50 + 四分支头
│   ├── dataset.py          # 机场人脸数据集 + 非受限环境增广
│   ├── train.py            # 训练入口
│   ├── test.py             # 评估：生成学术报告（ACC/Precision/Recall/F1/ACER）
│   ├── analysis.py         # 误差与可视化分析
│   └── count.py            # 数据集统计
├── dinov2/                 # DINOv2 官方源码（本项目依赖，含其自身 LICENSE）
├── vit/                    # 主模型：DINOv2 ViT-S/14 主干的 CD-MAGNet
│   ├── config.py           # 两阶段训练 / 消融开关（环境变量 ABL_*）
│   ├── models.py           # CD_MAGNet（ArcFaceHead、DPM、FASLocalTextureBranch、GRL）
│   ├── train.py            # 两阶段训练 + 自动测试报告
│   ├── run_ablation.py     # 消融实验一键运行（full / wo_dpm / wo_local / wo_orth）
│   ├── test.py             # FLOPs / 参数量统计（thop）
│   └── ablation_summary.csv# 消融结果汇总
└── dataset_resplit_disjoint_manifest.csv  # 身份不相交的数据重划分清单
```

## 环境安装（Installation）

```bash
# 建议 Python >= 3.10
pip install torch torchvision
pip install numpy pandas scikit-learn matplotlib seaborn tqdm thop opencv-python
```

DINOv2 依赖本仓库内的 `dinov2/` 目录（[vit/models.py](vit/models.py) 中 `DINOV2_ROOT` 指向该目录，部署时按实际路径修改）。

预训练权重：将 `dinov2_vits14_pretrain.pth` 放置于 `vit/` 目录下（模型权重不随仓库分发，`.pth` 文件已在 .gitignore 中排除）。

## 数据准备（Data Preparation）

1. 原始数据放置于 `config.py` 中 `RAW_DATA_ROOT` 指定的路径
2. 人脸对齐（MTCNN）至 `data_mtcnn_aligned/`，训练输入尺寸 112×112（backbone 输入 224×224）
3. 训练/测试身份不相交划分见 `dataset_resplit_disjoint_manifest.csv`
4. 按需修改 `vit/config.py` / `resnet50-V0/config.py` 中的 `DATASET_ROOT` 等路径

## 训练与测试（Training & Evaluation）

```bash
cd vit

# 两阶段训练（Stage 1: ID → Stage 2: ID + FAS + OCC）
python train.py

# 一键消融实验（full / wo_dpm / wo_local / wo_orth，逐变体独立进程）
python run_ablation.py

# 统计 FLOPs 与参数量（先在 test.py 中填入实际 num_classes）
python test.py
```

基线版本：

```bash
cd resnet50-V0
python train.py
python test.py      # 输出完整学术评估报告
```

## 消融实验结果（Ablation Results）

| Variant | ID Acc (%) | OCC ID Acc (%) | Non-OCC ID Acc (%) | FAS Acc (%) | FAS F1 | OCC Det Acc (%) |
|---|---|---|---|---|---|---|
| **full（本文）** | **85.36** | **67.81** | **89.60** | **97.26** | **98.01** | **91.15** |
| w/o DPM | 84.36 | 65.41 | 88.93 | 97.72 | 98.34 | 91.22 |
| w/o Local branch | 84.50 | 65.41 | 89.10 | 86.29 | 89.42 | 90.82 |
| w/o Orthogonal | 85.30 | 67.81 | 89.51 | 97.40 | 98.11 | 91.08 |

- DPM 与正交解约束主要提升遮挡场景下的身份识别
- 局部纹理分支对活体检测至关重要（去掉后 FAS F1 下降约 8.6）

## License

代码遵循本仓库 [LICENSE](LICENSE)；`dinov2/` 目录遵循其自带的 LICENSE 条款（Meta AI / Facebook）。

## Citation

若本工作对您有帮助，请引用（正式发表后更新 BibTeX）：

```bibtex
@article{cd_magnet2026,
  title   = {Lightweight Gate Face Recognition and Liveness Detection in Unconstrained Environments},
  author  = {TODO},
  journal = {Journal of Electronic Imaging},
  year    = {2026},
  note    = {submitted}
}
```
