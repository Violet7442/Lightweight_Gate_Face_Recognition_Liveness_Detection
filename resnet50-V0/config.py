# config.py
import torch
import os

class Config:
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 你的私有根目录
        self.DATA_ROOT = "/root/autodl-tmp/CV/dataset/data_mtcnn_aligned"
        self.VIS_DIR = "./vis_results"
        os.makedirs(self.VIS_DIR, exist_ok=True)
        self.INPUT_SIZE = (112, 112)

        train_dir = os.path.join(self.DATA_ROOT, 'train')
        self.BASE_NUM_CLASSES = len(
            [name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))]
        ) if os.path.exists(train_dir) else 34

        self.NUM_CLASSES = None
        self.EMBEDDING_DIM = 512
        self.IGNORE_INDEX = -1

        self.EPOCHS = 60
        self.BATCH_SIZE = 64
        self.LR = 5e-5
        self.WEIGHT_DECAY = 5e-4
        self.NUM_WORKERS = 0
        self.WARMUP_EPOCHS = 10

        self.ARCFACE_S = 64.0
        self.ARCFACE_M = 0.5

        # 四分支权重分配 (用于 Loss 计算)
        self.WEIGHT_ID = 8.0
        self.WEIGHT_FAS = 5.0
        self.WEIGHT_OCC = 4.0
        self.WEIGHT_IDE = 4.0
        self.WEIGHT_ADV = 1.0
        self.NUM_IDE_CLASSES = 4

        self.BEST_MODEL_PATH = "cd_magnet_best.pth"

        # 综合评分权重
        # 权重总和为1.0，ACER是安全生命线，权重最高
        self.SCORE_WEIGHT_ID_ACC = 0.5   # 身份识别准确率
        self.SCORE_WEIGHT_FAS_ACER = 0.15 # 防伪ACER
        self.SCORE_WEIGHT_OCC_F1 = 0.15   # 遮挡检测F1
        self.SCORE_WEIGHT_IDE_ACC = 0.20  # 环境感知准确率

cfg = Config()