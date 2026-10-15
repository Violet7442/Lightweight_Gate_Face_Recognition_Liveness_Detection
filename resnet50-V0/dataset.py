import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from config import cfg


def get_base_transforms():
    """
    训练阶段增强：
    - Resize
    - 随机水平翻转
    - 随机旋转15度，平移10%，缩放90%-110%
    返回一个可调用的transform函数，替代torchvision Compose
    """
    def transform_fn(img: Image.Image):
        # Resize
        img = img.resize(cfg.INPUT_SIZE, Image.BILINEAR)
        # RandomHorizontalFlip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # RandomAffine: degrees=15, translate(0.1,0.1), scale(0.9,1.1)
        angle = random.uniform(-15, 15)
        w, h = img.size
        tx = random.uniform(-0.1, 0.1) * w
        ty = random.uniform(-0.1, 0.1) * h
        scale = random.uniform(0.9, 1.1)
        img = img.transform(
            (w, h),
            Image.AFFINE,
            (scale, 0, tx, 0, scale, ty),
            resample=Image.BILINEAR,
            fillcolor=(0, 0, 0)
        )
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0,0,0))
        return img
    return transform_fn


def get_test_base_transforms():
    """
    验证/测试阶段：仅Resize，无随机增强
    """
    def transform_fn(img: Image.Image):
        img = img.resize(cfg.INPUT_SIZE, Image.BILINEAR)
        return img
    return transform_fn


class UnconstrainedEnvWrapper(Dataset):
    """
    统一包装器：
    - 底层数据集给出真实标签
    - 负责 ToTensor + Normalize（原生实现，不依赖torchvision）
    """
    def __init__(self, dataset, is_train=True):
        self.dataset = dataset
        self.is_train = is_train
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, id_lbl, fas_lbl, occ_lbl, ide_lbl = self.dataset[idx]
        # ToTensor + Normalize
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        img_np = np.transpose(img_np, (2, 0, 1)) # HWC -> CHW
        img_tensor = torch.from_numpy(img_np)
        return (
            img_tensor,
            id_lbl,
            fas_lbl,
            occ_lbl,
            ide_lbl
        )


class AirportFaceDataset(Dataset):
    """
    私有机场人脸数据集。
    """
    def __init__(
            self,
            root_dir,
            split='train',
            transform=None,
            id_to_idx_map=None,
            allow_new_ids=True
    ):
        self.root_base = root_dir
        self.split = split
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.data_list = []
        if id_to_idx_map is not None:
            self.id_to_idx = id_to_idx_map
        else:
            self.id_to_idx = {}
        self.allow_new_ids = allow_new_ids
        if os.path.exists(self.root_dir):
            self._parse_dataset()
        else:
            print(f"⚠️ [私有数据集-{self.split}] 路径不存在: {self.root_dir}")

    @staticmethod
    def _parse_ide_label_from_filename(filename_no_ext):
        lower_name = filename_no_ext.lower()
        if lower_name.endswith("_origin"):
            return 0
        elif lower_name.endswith("_glare"):
            return 1
        elif lower_name.endswith("_blur"):
            return 2
        elif lower_name.endswith("_dark"):
            return 3
        else:
            return None

    def _parse_dataset(self):
        skipped_unknown_id = 0
        skipped_invalid_ide = 0
        for person_name in sorted(os.listdir(self.root_dir)):
            person_dir = os.path.join(self.root_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            if person_name not in self.id_to_idx:
                if self.allow_new_ids:
                    self.id_to_idx[person_name] = len(self.id_to_idx)
                else:
                    skipped_unknown_id += 1
                    continue
            person_id = self.id_to_idx[person_name]
            for status in ['real', 'fake', 'mask']:
                status_dir = os.path.join(person_dir, status)
                if not os.path.exists(status_dir):
                    continue
                for file in sorted(os.listdir(status_dir)):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        file_stem = os.path.splitext(file)[0]
                        ide_label = self._parse_ide_label_from_filename(file_stem)
                        if ide_label is None:
                            skipped_invalid_ide += 1
                            continue
                        img_path = os.path.join(status_dir, file)
                        # ✅ 核心修改点：如果是假脸，身份 ID 设为忽略（-1）
                        # 真人(real)和戴口罩的真人(mask)才保留身份 ID
                        actual_id_label = person_id if status in ['real', 'mask'] else cfg.IGNORE_INDEX
                        self.data_list.append({
                            'path': img_path,
                            'id_label': actual_id_label,  # 使用修改后的 ID 标签
                            'fas_label': 1.0 if status in ['real', 'mask'] else 0.0,
                            'occ_label': 1.0 if status == 'mask' else 0.0,
                            'ide_label': ide_label
                        })
        print(
            f"-> 🛫 [私有数据集-{self.split}] 挂载成功! "
            f"图片数: {len(self.data_list)} | "
            f"身份数: {len(self.id_to_idx)} | "
            f"跳过未知身份目录数: {skipped_unknown_id} | "
            f"跳过非法IDE命名图片数: {skipped_invalid_ide}"
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = Image.open(item['path']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return (
            image,
            torch.tensor(item['id_label'], dtype=torch.long),
            torch.tensor(item['fas_label'], dtype=torch.float32),
            torch.tensor(item['occ_label'], dtype=torch.float32),
            torch.tensor(item['ide_label'], dtype=torch.long)
        )
