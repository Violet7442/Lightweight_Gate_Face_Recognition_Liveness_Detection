# -*- coding: utf-8 -*-
"""增量补齐 fake 的 MTCNN 对齐图。

只处理 raw 中存在、但 aligned 目录缺失的 fake 图片；
对齐逻辑与原 preprocess_align_mtcnn.py 完全一致（五点仿射 + 中心裁剪兜底）。
不修改任何已有文件，仅向 data_mtcnn_aligned/{split}/{person}/fake 写入缺失的 jpg。
本地 CPU 运行。
"""
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from facenet_pytorch import MTCNN

RAW_ROOT = Path(os.environ.get("DS_RAW", r"/root/autodl-tmp/CV/dataset/data"))
ALIGNED_ROOT = Path(os.environ.get("DS_ALIGNED", r"/root/autodl-tmp/CV/dataset/data_mtcnn_aligned"))
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
MIN_IMAGES_PER_ID = 5

REFERENCE_FACIAL_POINTS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def images(d):
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=lambda p: p.name.lower())


def center_resize_fallback(pil_image, image_size=112):
    image = pil_image.convert("RGB")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((image_size, image_size), Image.BILINEAR)


def choose_best_face(boxes, probs, landmarks):
    if boxes is None or landmarks is None or len(boxes) == 0:
        return None
    boxes = np.asarray(boxes)
    probs = np.asarray(probs) if probs is not None else np.ones((len(boxes),), np.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    scores = areas * np.maximum(probs, 0.0)
    return landmarks[int(np.argmax(scores))]


def align_face_by_five_points(pil_image, landmarks, image_size=112):
    src = np.asarray(landmarks, dtype=np.float32)
    dst = REFERENCE_FACIAL_POINTS_112.copy()
    if image_size != 112:
        dst *= float(image_size) / 112.0
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return center_resize_fallback(pil_image, image_size), "fallback"
    aligned = cv2.warpAffine(np.asarray(pil_image.convert("RGB")), matrix,
                             (image_size, image_size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return Image.fromarray(aligned), "mtcnn_five_points"


def main():
    # 身份树（与 dataset.py 一致）
    train_root = RAW_ROOT / "train"
    identity = {}
    for pd in sorted((p for p in train_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        c = len(images(pd / "real")) + len(images(pd / "mask"))
        if c >= MIN_IMAGES_PER_ID:
            identity[pd.name] = c

    # 找出缺失的 fake（按 stem 对比）
    missing = []
    for split in SPLITS:
        for person in identity:
            raw_fake = RAW_ROOT / split / person / "fake"
            alg_fake = ALIGNED_ROOT / split / person / "fake"
            raw_stems = {p.stem for p in images(raw_fake)}
            alg_stems = {p.stem for p in images(alg_fake)}
            for stem in sorted(raw_stems - alg_stems):
                src = next(p for p in images(raw_fake) if p.stem == stem)
                missing.append((split, person, src))
    print(f"TO_ALIGN={len(missing)}")

    if not missing:
        print("无缺失 fake，跳过。")
        return

    mtcnn = MTCNN(image_size=112, margin=0, keep_all=True, select_largest=False,
                  post_process=False, device="cpu")

    success = fallback = read_fail = save_fail = 0
    for split, person, src in missing:
        out_dir = ALIGNED_ROOT / split / person / "fake"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (src.stem + ".jpg")
        try:
            pil = Image.open(src).convert("RGB")
        except Exception:
            read_fail += 1
            continue
        try:
            boxes, probs, landmarks = mtcnn.detect(pil, landmarks=True)
            best = choose_best_face(boxes, probs, landmarks)
            if best is None:
                img, method = center_resize_fallback(pil, 112), "fallback"
                fallback += 1
            else:
                img, method = align_face_by_five_points(pil, best, 112)
                if method == "fallback":
                    fallback += 1
                else:
                    success += 1
        except Exception:
            img, method = center_resize_fallback(pil, 112), "fallback"
            fallback += 1
        try:
            img.save(out_path, quality=95)
        except Exception:
            save_fail += 1

    print(f"ALIGNED_SUCCESS={success} FALLBACK={fallback} READ_FAIL={read_fail} SAVE_FAIL={save_fail}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()