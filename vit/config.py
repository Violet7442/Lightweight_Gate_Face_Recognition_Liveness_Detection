# -*- coding: utf-8 -*-
# config.py

import os
import torch

class Config:
    def __init__(self):
        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ============================================================
        # 项目路径
        # ============================================================
        self.PROJECT_ROOT = "/root/autodl-tmp/CV/vit"
        self.DATASET_ROOT = "/root/autodl-tmp/CV/dataset"

        self.RAW_DATA_ROOT = "/root/autodl-tmp/CV/dataset/data"
        self.ALIGNED_REAL_ROOT = (
            "/root/autodl-tmp/CV/dataset/data_mtcnn_aligned"
        )
        self.ALIGNED_MASK_ROOT = (
            "/root/autodl-tmp/CV/dataset/data_mtcnn_aligned"
        )

        self.ALIGNED_DATA_ROOT = self.ALIGNED_REAL_ROOT
        self.DATA_ROOT = self.ALIGNED_REAL_ROOT

        if not os.path.exists(self.RAW_DATA_ROOT):
            raise FileNotFoundError(
                f"RAW_DATA_ROOT 路径不存在: {self.RAW_DATA_ROOT}"
            )

        self.FACE_MASK_12K_ROOT = os.path.join(
            self.DATASET_ROOT,
            "Face-Mask-12K-Images-Dataset",
            "Face Mask Dataset",
        )

        self.REAL_FAKE_ROOT = os.path.join(
            self.DATASET_ROOT,
            "Real-and-Fake-Face-Detection-Dataset",
        )

        self.IGNORE_DIR_NAMES = {
            "word2vec-pytorch-main",
            "word2vec",
            "__pycache__",
        }

        # ============================================================
        # 数据源开关
        # ============================================================
        self.USE_PRIVATE_DATA = True
        self.USE_PRIVATE_REAL = True
        self.USE_PRIVATE_MASK = True
        self.USE_PRIVATE_FAKE = True

        self.USE_FACE_MASK_12K = False
        self.USE_REAL_FAKE_PUBLIC = False

        self.USE_FAKE_FOR_ID = False
        self.PUBLIC_DATA_HAS_ID = False

        # ============================================================
        # 输出目录
        # Experiment A 独立目录，避免覆盖旧实验日志和可视化结果。
        # ============================================================
        self.VIS_DIR = "./vis_results_exp_a"
        self.LOG_DIR = "./logs_exp_a"

        os.makedirs(self.VIS_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

        # ============================================================
        # 图像尺寸
        # ============================================================
        self.INPUT_SIZE = (112, 112)
        self.ALIGN_IMAGE_SIZE = 112
        self.BACKBONE_INPUT_SIZE = 224

        self.REQUIRE_ALIGNED_REAL = True
        self.REQUIRE_ALIGNED_MASK = True

        # ============================================================
        # Backbone 配置
        # ============================================================
        self.BACKBONE_NAME = "dinov2_vits14"
        self.BACKBONE_DIM = 384
        self.USE_DINOV2_PRETRAINED = True

        # ============================================================
        # 基础训练配置
        # ============================================================
        self.EPOCHS = 100
        self.BATCH_SIZE = 64

        self.LR_BACKBONE = 1.5e-5
        self.LR_HEAD = 2.5e-4
        self.WEIGHT_DECAY = 8e-4

        self.NUM_WORKERS = 4
        self.WARMUP_EPOCHS = 5

        # ============================================================
        # ArcFace / Embedding
        # ============================================================
        self.EMBEDDING_DIM = 512
        self.DEG_EMBEDDING_DIM = 256

        self.ARCFACE_S = 30.0
        self.ARCFACE_M = 0.35
        self.LABEL_SMOOTHING = 0.02

        self.MIN_IMAGES_PER_ID = 5
        self.SEED = 42

        # ============================================================
        # 训练模式
        # ============================================================
        self.TRAIN_MODE = "disentangle"
        self.IGNORE_INDEX = -1

        # 此项会在 apply_stage_config() 中被 Stage 路径替换。
        self.BEST_MODEL_PATH = "exp_a_stage2_id_occ_fas_best.pth"

        # ============================================================
        # 冻结策略
        # ============================================================
        self.FREEZE_NON_ID_BRANCHES = False
        self.TRAIN_LAST_BLOCKS_ONLY = True
        self.TRAIN_LAST_BLOCKS_NUM = 2

        self.STAGE2_FREEZE_BACKBONE = True
        self.STAGE2_FREEZE_DPM = True

        # ============================================================
        # Stage 2 分组学习率
        # ============================================================
        self.STAGE2_LR_FAS = 3e-4
        self.STAGE2_LR_ID = 5e-5
        self.STAGE2_LR_OTHER_HEAD = 1e-5
        self.STAGE2_LR_SHARED = 1e-6
        self.STAGE2_WEIGHT_DECAY = 2e-4

        # ============================================================
        # 最佳模型策略
        # ============================================================
        self.ID_ACC_FLOOR = 50.0
        self.OCC_ID_ACC_FLOOR = 35.0

        self.ID_FAS_SCORE_ID_WEIGHT = 0.70
        self.ID_FAS_SCORE_FAS_WEIGHT = 0.30

        self.ID_OCC_FAS_SCORE_ID_WEIGHT = 0.50
        self.ID_OCC_FAS_SCORE_OCC_ID_WEIGHT = 0.30
        self.ID_OCC_FAS_SCORE_FAS_WEIGHT = 0.20

        # ============================================================
        # 学习率调度
        # ============================================================
        self.USE_COSINE_LR = True
        self.MIN_LR = 1e-6

        # ============================================================
        # 默认损失权重
        # 训练时实际参数由 Stage 1 / Stage 2 配置覆盖。
        # ============================================================
        self.LAMBDA_ID = 1.0
        self.LAMBDA_DPM_DEG = 0.08
        self.LAMBDA_DEG_EMB = 0.05
        self.LAMBDA_ADV_DEG = 0.00
        self.LAMBDA_ORTHO = 0.01

        self.USE_FAS_OCC_LOSS = True
        self.LAMBDA_FAS = 0.50
        self.LAMBDA_OCC = 0.05
        self.LAMBDA_ADV_FAS = 0.00
        self.LAMBDA_ADV_OCC = 0.00

        # ============================================================
        # FAS 配置
        # ============================================================
        self.USE_DPM = True          # 新增：是否使用 DPM 退化感知模块

        self.USE_FAS_LOCAL_BRANCH = True
        self.FAS_LOCAL_DIM = 256

        self.USE_FAS_SOBEL = False
        self.FAS_SOBEL_ALPHA = 0.35

        self.USE_FAS_BALANCED_LOSS = True
        self.FAS_POS_WEIGHT_OVERRIDE = 0.28
        self.USE_FAS_WEIGHTED_SAMPLER = False

        self.FAS_THRESHOLD = 0.60
        self.OCC_THRESHOLD = 0.50

        # ============================================================
        # 曲线输出
        # ============================================================
        self.DRAW_ALL_BRANCH_CURVES = True
        self.CURVE_FIG_NAME = "all_branch_training_curves.png"

        # ============================================================
        # 最佳模型保存策略
        # ============================================================
        self.BEST_MODEL_METRIC = "id_occ_fas"

        # ============================================================
        # 分阶段训练配置
        # ============================================================
        self.USE_STAGE_TRAINING = True

        self.STAGE1_NAME = "stage1_id"
        self.STAGE2_NAME = "stage2_id_fas"

        self.STAGE1_EPOCHS = 40
        self.STAGE2_EPOCHS = 80

        # Experiment A: 独立保存，不覆盖此前的 stage1_id_best.pth
        # 和 stage2_id_occ_fas_best.pth。
        self.STAGE1_BEST_MODEL_PATH = "exp_a_stage1_id_best.pth"
        self.STAGE2_BEST_MODEL_PATH = "exp_a_stage2_id_occ_fas_best.pth"

        self.STAGE1_BEST_MODEL_METRIC = "id_acc"
        self.STAGE2_BEST_MODEL_METRIC = "id_occ_fas"

        self.STAGE1_PATIENCE = 20
        self.STAGE2_PATIENCE = 30

        # ============================================================
        # Stage 1: 先训 ID
        # ============================================================
        self.STAGE1_TRAIN_MODE = "disentangle"

        self.STAGE1_LAMBDA_ID = 1.30
        self.STAGE1_LAMBDA_DPM_DEG = 0.05
        self.STAGE1_LAMBDA_DEG_EMB = 0.03
        self.STAGE1_LAMBDA_ADV_DEG = 0.00
        self.STAGE1_LAMBDA_ORTHO = 0.01

        self.STAGE1_LAMBDA_FAS = 0.00
        self.STAGE1_LAMBDA_OCC = 0.03
        self.STAGE1_LAMBDA_ADV_FAS = 0.00
        self.STAGE1_LAMBDA_ADV_OCC = 0.00

        self.STAGE1_USE_FAS_WEIGHTED_SAMPLER = False
        self.STAGE1_LABEL_SMOOTHING = 0.02
        self.STAGE1_ARCFACE_M = 0.35

        # ============================================================
        # Stage 2: 加入 FAS，但更保护 ID
        # ============================================================
        self.STAGE2_TRAIN_MODE = "disentangle"

        self.STAGE2_LAMBDA_ID = 1.50
        self.STAGE2_LAMBDA_DPM_DEG = 0.02
        self.STAGE2_LAMBDA_DEG_EMB = 0.01
        self.STAGE2_LAMBDA_ADV_DEG = 0.00
        self.STAGE2_LAMBDA_ORTHO = 0.005

        self.STAGE2_LAMBDA_FAS = 0.20

        # mask 子目录含有非严格口罩样本，保持当前低权重，不提高。
        self.STAGE2_LAMBDA_OCC = 0.02

        self.STAGE2_LAMBDA_ADV_FAS = 0.00
        self.STAGE2_LAMBDA_ADV_OCC = 0.00

        self.STAGE2_USE_FAS_WEIGHTED_SAMPLER = False
        self.STAGE2_LABEL_SMOOTHING = 0.02
        self.STAGE2_ARCFACE_M = 0.35

cfg = Config()

# ===== 消融实验覆盖（run_ablation.py 通过环境变量传入）=====
def _env_bool(name, default):
    v = os.environ.get("ABL_" + name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes")

tag = os.environ.get("ABLATION_TAG", "exp_a")
cfg.LOG_DIR  = f"./logs_ablation_{tag}"
cfg.VIS_DIR  = f"./vis_results_ablation_{tag}"
cfg.STAGE1_BEST_MODEL_PATH = f"{tag}_stage1_id_best.pth"
cfg.STAGE2_BEST_MODEL_PATH = f"{tag}_stage2_id_occ_fas_best.pth"
cfg.BEST_MODEL_PATH = cfg.STAGE2_BEST_MODEL_PATH

cfg.USE_DPM = _env_bool("USE_DPM", cfg.USE_DPM)
cfg.USE_FAS_LOCAL_BRANCH = _env_bool("USE_FAS_LOCAL_BRANCH", cfg.USE_FAS_LOCAL_BRANCH)
cfg.STAGE1_LAMBDA_ORTHO = float(os.environ.get("ABL_STAGE1_LAMBDA_ORTHO", cfg.STAGE1_LAMBDA_ORTHO))
cfg.STAGE2_LAMBDA_ORTHO = float(os.environ.get("ABL_STAGE2_LAMBDA_ORTHO", cfg.STAGE2_LAMBDA_ORTHO))