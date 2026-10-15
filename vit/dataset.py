# -*- coding: utf-8 -*-
# dataset.py

import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import cfg

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_GLOBAL_ID_TO_IDX = None

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_image_file(name):
    return str(name).lower().endswith(IMAGE_EXTENSIONS)

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=cfg.INPUT_SIZE,
            scale=(0.90, 1.00),
            ratio=(0.96, 1.04),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.03,
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.03),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
        ], p=0.15),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.97, 1.03),
            shear=2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        transforms.RandomErasing(
            p=0.18,
            scale=(0.02, 0.10),
            ratio=(0.3, 3.3),
            value=0,
        ),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize(cfg.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])

def estimate_degradation_labels(pil_image):
    image = np.asarray(pil_image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    highlight_ratio = float((gray > 235).mean())

    return (
        1.0 if lap_var < 80.0 else 0.0,
        1.0 if highlight_ratio > 0.03 else 0.0,
        1.0 if brightness < 75.0 else 0.0,
    )

def build_global_id_to_idx(raw_private_root, min_images_per_id=5):
    global _GLOBAL_ID_TO_IDX
    if _GLOBAL_ID_TO_IDX is not None:
        return _GLOBAL_ID_TO_IDX

    train_dir = os.path.join(raw_private_root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"私有 train 目录不存在: {train_dir}")

    person_counts = defaultdict(int)
    for person_name in sorted(os.listdir(train_dir)):
        person_dir = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for subset in ("real", "mask"):
            subset_dir = os.path.join(person_dir, subset)
            if not os.path.isdir(subset_dir):
                continue
            person_counts[person_name] += sum(
                is_image_file(name) for name in os.listdir(subset_dir)
            )

    valid_people = sorted(
        name for name, count in person_counts.items()
        if count >= min_images_per_id
    )
    _GLOBAL_ID_TO_IDX = {name: index for index, name in enumerate(valid_people)}
    return _GLOBAL_ID_TO_IDX

def collect_images(root_dir):
    results = []
    if not os.path.isdir(root_dir):
        return results
    for current_root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in cfg.IGNORE_DIR_NAMES]
        for filename in files:
            if is_image_file(filename):
                results.append(os.path.join(current_root, filename))
    return sorted(results)

class MultiSourceFaceDataset(Dataset):
    def __init__(self, split="train", transform=None, seed=42, min_images_per_id=5):
        self.split = split
        self.transform = transform
        self.samples = []
        seed_everything(seed)

        self.id_to_idx = build_global_id_to_idx(
            cfg.RAW_DATA_ROOT,
            min_images_per_id=min_images_per_id,
        )

        if cfg.USE_PRIVATE_DATA:
            self._load_private_data()

        # 外部数据严格只加入训练集，绝不写入 val/test，避免数据泄漏。
        if split == "train" and cfg.USE_REAL_FAKE_PUBLIC:
            self._load_external_fas_crops()

        random.Random(seed).shuffle(self.samples)
        self._print_statistics()

    def _add_sample(
        self,
        image_path,
        id_label=-1,
        fas_label=-1.0,
        occ_label=-1.0,
        source="unknown",
        subset="unknown",
        person_name="unknown",
    ):
        self.samples.append({
            "image_path": image_path,
            "id": int(id_label),
            "fas": float(fas_label),
            "occ": float(occ_label),
            "source": str(source),
            "subset": str(subset),
            "person_name": str(person_name),
        })

    def _load_private_data(self):
        raw_split_dir = os.path.join(cfg.RAW_DATA_ROOT, self.split)
        aligned_real_split = os.path.join(cfg.ALIGNED_REAL_ROOT, self.split)
        aligned_mask_split = os.path.join(cfg.ALIGNED_MASK_ROOT, self.split)

        if not os.path.isdir(raw_split_dir):
            print(f"[Private] split 不存在，跳过: {raw_split_dir}")
            return

        if cfg.REQUIRE_ALIGNED_REAL and not os.path.isdir(aligned_real_split):
            raise FileNotFoundError(f"real 对齐目录不存在: {aligned_real_split}")
        if cfg.REQUIRE_ALIGNED_MASK and not os.path.isdir(aligned_mask_split):
            raise FileNotFoundError(f"mask 对齐目录不存在: {aligned_mask_split}")

        for person_name in sorted(os.listdir(raw_split_dir)):
            raw_person_dir = os.path.join(raw_split_dir, person_name)
            if not os.path.isdir(raw_person_dir) or person_name not in self.id_to_idx:
                continue

            person_id = self.id_to_idx[person_name]

            if cfg.USE_PRIVATE_REAL:
                aligned_dir = os.path.join(aligned_real_split, person_name, "real")
                raw_dir = os.path.join(raw_person_dir, "real")
                image_dir = aligned_dir if os.path.isdir(aligned_dir) else raw_dir
                source = "private_real_aligned" if image_dir == aligned_dir else "private_real_raw_fallback"
                for path in collect_images(image_dir):
                    self._add_sample(path, person_id, 1.0, 0.0, source, "real", person_name)

            if cfg.USE_PRIVATE_MASK:
                aligned_dir = os.path.join(aligned_mask_split, person_name, "mask")
                raw_dir = os.path.join(raw_person_dir, "mask")
                image_dir = aligned_dir if os.path.isdir(aligned_dir) else raw_dir
                source = "private_mask_aligned" if image_dir == aligned_dir else "private_mask_raw_fallback"
                for path in collect_images(image_dir):
                    self._add_sample(path, person_id, 1.0, 1.0, source, "mask", person_name)

            if cfg.USE_PRIVATE_FAKE:
                fake_dir = os.path.join(raw_person_dir, "fake")
                for path in collect_images(fake_dir):
                    fake_id = person_id if cfg.USE_FAKE_FOR_ID else -1
                    self._add_sample(path, fake_id, 0.0, -1.0, "private_fake_raw", "fake", person_name)

    def _load_external_fas_crops(self):
        real_paths = collect_images(cfg.EXTERNAL_FAS_CROP_REAL_DIR)
        fake_paths = collect_images(cfg.EXTERNAL_FAS_CROP_FAKE_DIR)

        if cfg.REQUIRE_EXTERNAL_FAS_CROPS and (not real_paths or not fake_paths):
            raise FileNotFoundError(
                "外部 FAS 裁剪数据不存在或为空。请先执行：\n"
                "python prepare_external_fas.py\n"
                f"期望 real: {cfg.EXTERNAL_FAS_CROP_REAL_DIR}\n"
                f"期望 fake: {cfg.EXTERNAL_FAS_CROP_FAKE_DIR}"
            )

        for path in real_paths:
            self._add_sample(
                path, -1, 1.0, -1.0,
                "external_fas_cropped", "external_real", "unknown",
            )
        for path in fake_paths:
            self._add_sample(
                path, -1, 0.0, -1.0,
                "external_fas_cropped", "external_fake", "unknown",
            )

        print(f"[External FAS] cropped real={len(real_paths)}, fake={len(fake_paths)}")

    def _print_statistics(self):
        source_counter = defaultdict(int)
        subset_counter = defaultdict(int)
        id_count = fas_count = occ_count = 0

        for sample in self.samples:
            source_counter[sample["source"]] += 1
            subset_counter[sample["subset"]] += 1
            id_count += int(sample["id"] >= 0)
            fas_count += int(sample["fas"] >= 0)
            occ_count += int(sample["occ"] >= 0)

        print(f"\n[{self.split.upper()}] 样本统计")
        print(f"总样本={len(self.samples)} | ID 类别={len(self.id_to_idx)} | "
              f"ID 标签={id_count} | FAS 标签={fas_count} | OCC 标签={occ_count}")
        print("子类型:")
        for key, value in sorted(subset_counter.items()):
            print(f"  {key:28s}: {value}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as image_file:
            image = image_file.convert("RGB")

        blur, glare, dark = estimate_degradation_labels(image)
        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            torch.tensor(sample["id"], dtype=torch.long),
            torch.tensor(sample["fas"], dtype=torch.float32),
            torch.tensor(sample["occ"], dtype=torch.float32),
            torch.tensor(blur, dtype=torch.float32),
            torch.tensor(glare, dtype=torch.float32),
            torch.tensor(dark, dtype=torch.float32),
        )