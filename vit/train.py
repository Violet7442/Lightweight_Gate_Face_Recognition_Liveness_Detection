# -*- coding: utf-8 -*-
# train.py

import os
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
)

from config import cfg
from dataset import MultiSourceFaceDataset, get_train_transforms, get_test_transforms
from models import CD_MAGNet

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PaperLogger:
    def __init__(self):
        self.train_history = []
        self.val_history = []
        self.test_history = []
        os.makedirs(cfg.LOG_DIR, exist_ok=True)

    def _save(self, history, filename):
        pd.DataFrame(history).to_csv(
            os.path.join(cfg.LOG_DIR, filename),
            index=False,
            encoding="utf-8-sig"
        )

    def update_train(self, epoch, metrics):
        row = copy.deepcopy(metrics)
        row["epoch"] = epoch
        self.train_history.append(row)
        self._save(self.train_history, "train_log.csv")

    def update_val(self, epoch, metrics):
        row = copy.deepcopy(metrics)
        row["epoch"] = epoch
        self.val_history.append(row)
        self._save(self.val_history, "val_log.csv")

    def update_test(self, metrics):
        self.test_history.append(copy.deepcopy(metrics))
        self._save(self.test_history, "final_test_report.csv")

    def draw_curves(self):
        if not self.train_history:
            return

        train_df = pd.DataFrame(self.train_history)
        val_df = pd.DataFrame(self.val_history) if self.val_history else None

        fig, axes = plt.subplots(2, 2, figsize=(18, 11))

        if "loss" in train_df.columns:
            axes[0, 0].plot(
                train_df["epoch"],
                train_df["loss"],
                label="Train Loss",
                color="blue",
                linewidth=2
            )

        if val_df is not None and "loss" in val_df.columns:
            axes[0, 0].plot(
                val_df["epoch"],
                val_df["loss"],
                label="Val Loss",
                color="red",
                linestyle="--",
                linewidth=2
            )

        axes[0, 0].set_title("Loss Curve", fontsize=16)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        if "id_acc" in train_df.columns:
            axes[0, 1].plot(
                train_df["epoch"],
                train_df["id_acc"],
                label="Train ID Acc",
                color="blue",
                linewidth=2
            )

        if val_df is not None and "id_acc" in val_df.columns:
            axes[0, 1].plot(
                val_df["epoch"],
                val_df["id_acc"],
                label="Val ID Acc",
                color="red",
                linestyle="--",
                linewidth=2
            )

        if "occ_id_acc" in train_df.columns:
            axes[0, 1].plot(
                train_df["epoch"],
                train_df["occ_id_acc"],
                label="Train Occ-ID Acc",
                color="purple",
                linewidth=1.5,
                alpha=0.8
            )

        if val_df is not None and "occ_id_acc" in val_df.columns:
            axes[0, 1].plot(
                val_df["epoch"],
                val_df["occ_id_acc"],
                label="Val Occ-ID Acc",
                color="brown",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8
            )

        axes[0, 1].set_title("ID Accuracy Curve", fontsize=16)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        if "fas_acc" in train_df.columns:
            axes[1, 0].plot(
                train_df["epoch"],
                train_df["fas_acc"],
                label="Train FAS Acc",
                color="green",
                linewidth=2
            )

        if val_df is not None and "fas_acc" in val_df.columns:
            axes[1, 0].plot(
                val_df["epoch"],
                val_df["fas_acc"],
                label="Val FAS Acc",
                color="orange",
                linestyle="--",
                linewidth=2
            )

        axes[1, 0].set_title("FAS Accuracy Curve", fontsize=16)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        if "occ_id_acc" in train_df.columns:
            axes[1, 1].plot(
                train_df["epoch"],
                train_df["occ_id_acc"],
                label="Train Occ-ID Acc",
                color="purple",
                linewidth=2
            )

        if val_df is not None and "occ_id_acc" in val_df.columns:
            axes[1, 1].plot(
                val_df["epoch"],
                val_df["occ_id_acc"],
                label="Val Occ-ID Acc",
                color="brown",
                linestyle="--",
                linewidth=2
            )

        axes[1, 1].set_title("Occluded ID Accuracy Curve", fontsize=16)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(cfg.LOG_DIR, cfg.CURVE_FIG_NAME)
        plt.savefig(save_path, dpi=300)
        plt.close()

def build_fas_weighted_sampler(train_set):
    labels = [float(sample.get("fas", -1.0)) for sample in train_set.samples]
    labels_np = np.array(labels)
    valid = labels_np >= 0

    pos_count = int((labels_np[valid] == 1).sum())
    neg_count = int((labels_np[valid] == 0).sum())

    if pos_count == 0 or neg_count == 0:
        print("[WeightedSampler] FAS 正负样本不完整，跳过 sampler。")
        return None

    pos_weight = 1.0 / pos_count
    neg_weight = 1.0 / neg_count

    sample_weights = []
    for fas in labels:
        if fas == 1.0:
            sample_weights.append(pos_weight)
        elif fas == 0.0:
            sample_weights.append(neg_weight)
        else:
            sample_weights.append(1.0 / len(labels))

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

def compute_fas_pos_weight(train_set):
    pos = 0
    neg = 0

    for sample in train_set.samples:
        fas = float(sample.get("fas", -1.0))
        if fas == 1.0:
            pos += 1
        elif fas == 0.0:
            neg += 1

    auto_weight = 1.0 if pos == 0 or neg == 0 else neg / pos
    override_weight = getattr(cfg, "FAS_POS_WEIGHT_OVERRIDE", None)
    weight = float(override_weight) if override_weight is not None else float(auto_weight)

    print("\n" + "=" * 80)
    print("[FAS Loss Weight]")
    print("=" * 80)
    print(f"FAS=1 样本数              : {pos}")
    print(f"FAS=0 样本数              : {neg}")
    print(f"自动 pos_weight=neg/pos   : {auto_weight:.6f}")
    print(f"最终使用 pos_weight       : {weight:.6f}")
    print("=" * 80 + "\n")

    return weight, pos, neg

def build_dataloaders():
    print("\n" + "=" * 80)
    print("正在加载多数据源数据集")
    print("=" * 80)
    print(f"原始私有数据集      : {cfg.RAW_DATA_ROOT}")
    print(f"real 五点对齐数据集 : {cfg.ALIGNED_REAL_ROOT}")
    print(f"mask 五点对齐数据集 : {cfg.ALIGNED_MASK_ROOT}")
    print(f"Face-Mask-12K       : {cfg.FACE_MASK_12K_ROOT}")
    print(f"Real/Fake Public    : {cfg.REAL_FAKE_ROOT}")
    print("=" * 80)

    if cfg.REQUIRE_ALIGNED_REAL and not os.path.exists(cfg.ALIGNED_REAL_ROOT):
        raise FileNotFoundError(f"real 五点对齐数据集不存在: {cfg.ALIGNED_REAL_ROOT}")

    if cfg.REQUIRE_ALIGNED_MASK and not os.path.exists(cfg.ALIGNED_MASK_ROOT):
        raise FileNotFoundError(f"mask 五点对齐数据集不存在: {cfg.ALIGNED_MASK_ROOT}")

    train_set = MultiSourceFaceDataset(
        split="train",
        transform=get_train_transforms(),
        seed=cfg.SEED,
        min_images_per_id=cfg.MIN_IMAGES_PER_ID
    )

    val_set = MultiSourceFaceDataset(
        split="val",
        transform=get_test_transforms(),
        seed=cfg.SEED,
        min_images_per_id=cfg.MIN_IMAGES_PER_ID
    )

    test_set = MultiSourceFaceDataset(
        split="test",
        transform=get_test_transforms(),
        seed=cfg.SEED,
        min_images_per_id=cfg.MIN_IMAGES_PER_ID
    )

    pin_memory = cfg.DEVICE.type == "cuda"
    sampler = None

    if cfg.USE_FAS_WEIGHTED_SAMPLER and cfg.TRAIN_MODE == "disentangle":
        sampler = build_fas_weighted_sampler(train_set)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=pin_memory
    )

    total_classes = len(train_set.id_to_idx)

    print("\n" + "=" * 80)
    print("数据加载完成")
    print("=" * 80)
    print(f"训练集样本数: {len(train_set)}")
    print(f"验证集样本数: {len(val_set)}")
    print(f"测试集样本数: {len(test_set)}")
    print(f"身份类别数  : {total_classes}")
    print("=" * 80 + "\n")

    return train_loader, val_loader, test_loader, train_set, train_set.id_to_idx, total_classes

def apply_freeze_strategy(model, stage_name=None):
    for parameter in model.parameters():
        parameter.requires_grad = True

    if cfg.FREEZE_NON_ID_BRANCHES:
        model.freeze_non_id_branches()

    if cfg.TRAIN_LAST_BLOCKS_ONLY:
        model.train_last_blocks_only(num_blocks=cfg.TRAIN_LAST_BLOCKS_NUM)

    if stage_name == "stage2":
        if getattr(cfg, "STAGE2_FREEZE_BACKBONE", True):
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False

        if getattr(cfg, "STAGE2_FREEZE_DPM", True) and model.dpm is not None:
            for parameter in model.dpm.parameters():
                parameter.requires_grad = False

def print_trainable_parameters(model):
    total = 0
    trainable = 0
    module_stats = {}

    for name, parameter in model.named_parameters():
        num_params = parameter.numel()
        total += num_params
        key = name.split(".")[0]

        if key not in module_stats:
            module_stats[key] = {"total": 0, "trainable": 0}

        module_stats[key]["total"] += num_params

        if parameter.requires_grad:
            trainable += num_params
            module_stats[key]["trainable"] += num_params

    print("\n" + "=" * 80)
    print("可训练参数统计")
    print("=" * 80)

    for key, stats in sorted(module_stats.items()):
        pct = stats["trainable"] / stats["total"] * 100 if stats["total"] else 0.0
        print(
            f"{key:20s}: {stats['total']:>12,} 参数 | "
            f"{stats['trainable']:>12,} 可训练 ({pct:>5.1f}%)"
        )

    print("-" * 80)
    total_pct = trainable / total * 100 if total else 0.0
    print(f"{'总计':20s}: {total:>12,} 参数 | {trainable:>12,} 可训练 ({total_pct:>5.1f}%)")
    print("=" * 80 + "\n")

def compute_metrics(y_true, y_pred):
    if not y_true:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "acc": round(float((y_true == y_pred).mean() * 100), 2),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100), 2),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100), 2),
        "f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100), 2),
    }

def compute_binary_metrics(y_true, y_pred, prefix="bin"):
    if len(y_true) == 0:
        return {
            f"{prefix}_acc": 0.0,
            f"{prefix}_precision": 0.0,
            f"{prefix}_recall": 0.0,
            f"{prefix}_f1": 0.0,
            f"{prefix}_samples": 0,
            f"{prefix}_pos_samples": 0,
            f"{prefix}_neg_samples": 0,
            f"{prefix}_tn": 0,
            f"{prefix}_fp": 0,
            f"{prefix}_fn": 0,
            f"{prefix}_tp": 0,
        }

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        f"{prefix}_acc": round(float((y_true == y_pred).mean() * 100), 2),
        f"{prefix}_precision": round(float(precision_score(y_true, y_pred, average="binary", zero_division=0) * 100), 2),
        f"{prefix}_recall": round(float(recall_score(y_true, y_pred, average="binary", zero_division=0) * 100), 2),
        f"{prefix}_f1": round(float(f1_score(y_true, y_pred, average="binary", zero_division=0) * 100), 2),
        f"{prefix}_samples": int(len(y_true)),
        f"{prefix}_pos_samples": int((y_true == 1).sum()),
        f"{prefix}_neg_samples": int((y_true == 0).sum()),
        f"{prefix}_tn": int(tn),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tp": int(tp),
    }

def compute_multilabel_metrics(y_true, y_pred, prefix="multi"):
    if len(y_true) == 0:
        return {
            f"{prefix}_micro_f1": 0.0,
            f"{prefix}_macro_f1": 0.0,
            f"{prefix}_micro_precision": 0.0,
            f"{prefix}_macro_precision": 0.0,
            f"{prefix}_micro_recall": 0.0,
            f"{prefix}_macro_recall": 0.0,
            f"{prefix}_exact_match": 0.0,
            f"{prefix}_samples": 0,
        }

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    exact_match = (y_true == y_pred).all(axis=1).mean() * 100

    return {
        f"{prefix}_micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0) * 100), 2),
        f"{prefix}_macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100), 2),
        f"{prefix}_micro_precision": round(float(precision_score(y_true, y_pred, average="micro", zero_division=0) * 100), 2),
        f"{prefix}_macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100), 2),
        f"{prefix}_micro_recall": round(float(recall_score(y_true, y_pred, average="micro", zero_division=0) * 100), 2),
        f"{prefix}_macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100), 2),
        f"{prefix}_exact_match": round(float(exact_match), 2),
        f"{prefix}_samples": int(y_true.shape[0]),
    }

def append_binary_from_logits(logits_tensor, label_tensor, y_true_list, y_pred_list, threshold=0.5):
    prob = torch.sigmoid(logits_tensor.detach()).view(-1)
    label = label_tensor.detach().view(-1)
    valid = label >= 0

    if valid.sum().item() == 0:
        return

    y_true_list.extend(label[valid].long().cpu().tolist())
    y_pred_list.extend((prob[valid] >= threshold).long().cpu().tolist())

def append_multilabel_from_logits(logits_tensor, label_tensor, y_true_list, y_pred_list, threshold=0.5):
    prob = torch.sigmoid(logits_tensor.detach())
    label = label_tensor.detach()
    valid = (label >= 0).all(dim=1)

    if valid.sum().item() == 0:
        return

    y_true_list.extend(label[valid].long().cpu().numpy().tolist())
    y_pred_list.extend((prob[valid] >= threshold).long().cpu().numpy().tolist())

def masked_bce_logits_loss(logits, target, pos_weight_value=None):
    if logits.dim() == 1:
        logits = logits.view(-1, 1)
    if target.dim() == 1:
        target = target.view(-1, 1)

    mask = target >= 0
    if mask.sum().item() == 0:
        return logits.sum() * 0.0

    logits_valid = logits[mask]
    target_valid = target[mask]

    loss = F.binary_cross_entropy_with_logits(
        logits_valid,
        target_valid,
        reduction="none"
    )

    if pos_weight_value is not None:
        sample_weight = torch.ones_like(target_valid)
        sample_weight = torch.where(
            target_valid >= 0.5,
            torch.full_like(target_valid, float(pos_weight_value)),
            sample_weight
        )
        loss = loss * sample_weight

    return loss.mean()

def orthogonality_loss(id_for_orth, deg_embedding):
    dot = torch.sum(
        F.normalize(id_for_orth, dim=1) * F.normalize(deg_embedding, dim=1),
        dim=1
    )
    return torch.mean(torch.abs(dot))

def safe_accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    return round(float((np.asarray(y_true) == np.asarray(y_pred)).mean() * 100), 2)

def append_id_predictions_by_occ(
    id_logits,
    id_labels,
    occ_labels,
    id_true_all,
    id_pred_all,
    occ_id_true_all,
    occ_id_pred_all,
    non_occ_id_true_all,
    non_occ_id_pred_all
):
    valid_id_mask = id_labels >= 0

    if valid_id_mask.sum().item() == 0:
        return

    preds = torch.argmax(id_logits[valid_id_mask], dim=1)
    true_ids = id_labels[valid_id_mask]
    occ_valid = occ_labels.view(-1)[valid_id_mask]

    id_true_all.extend(true_ids.detach().cpu().tolist())
    id_pred_all.extend(preds.detach().cpu().tolist())

    occ_mask = occ_valid == 1
    if occ_mask.sum().item() > 0:
        occ_id_true_all.extend(true_ids[occ_mask].detach().cpu().tolist())
        occ_id_pred_all.extend(preds[occ_mask].detach().cpu().tolist())

    non_occ_mask = occ_valid == 0
    if non_occ_mask.sum().item() > 0:
        non_occ_id_true_all.extend(true_ids[non_occ_mask].detach().cpu().tolist())
        non_occ_id_pred_all.extend(preds[non_occ_mask].detach().cpu().tolist())

def save_id_confusion_matrix(y_true, y_pred, id_to_idx, filename_prefix, title="ID Confusion Matrix"):
    if len(y_true) == 0:
        print(f"[Confusion Matrix] {filename_prefix} 没有可用样本，跳过保存。")
        return

    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    idx_to_id = {value: key for key, value in id_to_idx.items()}
    class_ids = sorted(set(y_true) | set(y_pred))
    class_names = [str(idx_to_id.get(class_id, class_id)) for class_id in class_ids]

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    csv_path = os.path.join(cfg.LOG_DIR, f"{filename_prefix}.csv")
    cm_df.to_csv(csv_path, encoding="utf-8-sig")

    try:
        cm_df.to_excel(os.path.join(cfg.LOG_DIR, f"{filename_prefix}.xlsx"))
    except Exception as error:
        print(f"[Confusion Matrix] 导出 xlsx 失败，已保留 csv: {error}")

    fig_size = max(8, min(24, len(class_names) * 0.45))
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90, fontsize=6)
    plt.yticks(ticks, class_names, fontsize=6)
    plt.ylabel("True ID")
    plt.xlabel("Predicted ID")
    plt.tight_layout()

    png_path = os.path.join(cfg.LOG_DIR, f"{filename_prefix}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Confusion Matrix] 已保存: {csv_path}")
    print(f"[Confusion Matrix] 已保存: {png_path}")

def run_one_epoch_disentangle(
    model,
    dataloader,
    optimizer,
    criterion_id,
    criterion_bce_logits,
    device,
    epoch,
    phase="train",
    fas_pos_weight_value=None,
    stage_name=None
):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    if stage_name == "stage2" and getattr(cfg, "STAGE2_FREEZE_BACKBONE", True):
        model.backbone.eval()

    if stage_name == "stage2" and getattr(cfg, "STAGE2_FREEZE_DPM", True) and model.dpm is not None:
        model.dpm.eval()

    running = {
        "loss": 0.0,
        "id": 0.0,
        "dpm": 0.0,
        "deg": 0.0,
        "adv": 0.0,
        "orth": 0.0,
        "fas": 0.0,
        "occ": 0.0,
        "adv_fas": 0.0,
        "adv_occ": 0.0,
    }

    id_true_all, id_pred_all = [], []
    occ_id_true_all, occ_id_pred_all = [], []
    non_occ_id_true_all, non_occ_id_pred_all = [], []
    fas_true_all, fas_pred_all = [], []
    occ_det_true_all, occ_det_pred_all = [], []
    adv_fas_true_all, adv_fas_pred_all = [], []
    adv_occ_true_all, adv_occ_pred_all = [], []
    blur_true_all, blur_pred_all = [], []
    glare_true_all, glare_pred_all = [], []
    dark_true_all, dark_pred_all = [], []
    deg_true_all, deg_pred_all = [], []
    adv_deg_true_all, adv_deg_pred_all = [], []

    loop = tqdm(dataloader, desc=f"Epoch {epoch:3d} [{phase.upper():5s}]", ncols=180)

    for batch in loop:
        images, id_labels, fas_labels, occ_labels, blur_labels, glare_labels, dark_labels = batch

        images = images.to(device, non_blocking=True)
        id_labels = id_labels.to(device, non_blocking=True).long()
        fas_labels = fas_labels.to(device, non_blocking=True).float().view(-1, 1)
        occ_labels = occ_labels.to(device, non_blocking=True).float().view(-1, 1)
        blur_labels = blur_labels.to(device, non_blocking=True).float().view(-1, 1)
        glare_labels = glare_labels.to(device, non_blocking=True).float().view(-1, 1)
        dark_labels = dark_labels.to(device, non_blocking=True).float().view(-1, 1)

        deg_labels = torch.cat([blur_labels, glare_labels, dark_labels], dim=1)
        valid_id_mask = id_labels >= 0

        with torch.set_grad_enabled(is_train):
            # =============================================================
            # ArcFace 正确逻辑：
            # - train：传入真实 ID，训练时施加 angular margin。
            # - val/test：传 None，推理时绝不能对真类额外施加 margin。
            # =============================================================
            arcface_labels = id_labels if is_train else None
            out = model(images, label_id=arcface_labels, id_only=False)
            logits_id = out["out_id"]

            if valid_id_mask.sum().item() > 0:
                loss_id = criterion_id(
                    logits_id[valid_id_mask],
                    id_labels[valid_id_mask]
                )
            else:
                loss_id = logits_id.sum() * 0.0
            if cfg.USE_DPM:
                loss_dpm_deg = (
                    criterion_bce_logits(out["logits_blur"], blur_labels) +
                    criterion_bce_logits(out["logits_glare"], glare_labels) +
                    criterion_bce_logits(out["logits_dark"], dark_labels)
                ) / 3.0
            else:
                loss_dpm_deg = logits_id.sum() * 0.0

            loss_deg_emb = criterion_bce_logits(out["deg_emb_logits"], deg_labels)
            loss_adv_deg = criterion_bce_logits(out["adv_deg_logits"], deg_labels)
            loss_orth = orthogonality_loss(out["id_for_orth"], out["deg_embedding"])

            if cfg.USE_FAS_OCC_LOSS:
                fas_pos_weight = fas_pos_weight_value if cfg.USE_FAS_BALANCED_LOSS else None
                loss_fas = masked_bce_logits_loss(out["out_fas"], fas_labels, fas_pos_weight)
                loss_occ = masked_bce_logits_loss(out["out_occ"], occ_labels)
                loss_adv_fas = masked_bce_logits_loss(out["adv_fas"], fas_labels, fas_pos_weight)
                loss_adv_occ = masked_bce_logits_loss(out["adv_occ"], occ_labels)
            else:
                loss_fas = logits_id.sum() * 0.0
                loss_occ = logits_id.sum() * 0.0
                loss_adv_fas = logits_id.sum() * 0.0
                loss_adv_occ = logits_id.sum() * 0.0

            loss = (
                cfg.LAMBDA_ID * loss_id +
                cfg.LAMBDA_DPM_DEG * loss_dpm_deg +
                cfg.LAMBDA_DEG_EMB * loss_deg_emb +
                cfg.LAMBDA_ADV_DEG * loss_adv_deg +
                cfg.LAMBDA_ORTHO * loss_orth +
                cfg.LAMBDA_FAS * loss_fas +
                cfg.LAMBDA_OCC * loss_occ +
                cfg.LAMBDA_ADV_FAS * loss_adv_fas +
                cfg.LAMBDA_ADV_OCC * loss_adv_occ
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=5.0
                )
                optimizer.step()

        append_id_predictions_by_occ(
            logits_id,
            id_labels,
            occ_labels,
            id_true_all,
            id_pred_all,
            occ_id_true_all,
            occ_id_pred_all,
            non_occ_id_true_all,
            non_occ_id_pred_all
        )

        append_binary_from_logits(out["out_fas"], fas_labels, fas_true_all, fas_pred_all, cfg.FAS_THRESHOLD)
        append_binary_from_logits(out["out_occ"], occ_labels, occ_det_true_all, occ_det_pred_all, cfg.OCC_THRESHOLD)
        append_binary_from_logits(out["adv_fas"], fas_labels, adv_fas_true_all, adv_fas_pred_all, cfg.FAS_THRESHOLD)
        append_binary_from_logits(out["adv_occ"], occ_labels, adv_occ_true_all, adv_occ_pred_all, cfg.OCC_THRESHOLD)
        if cfg.USE_DPM:
            append_binary_from_logits(out["logits_blur"], blur_labels, blur_true_all, blur_pred_all, 0.5)
            append_binary_from_logits(out["logits_glare"], glare_labels, glare_true_all, glare_pred_all, 0.5)
            append_binary_from_logits(out["logits_dark"], dark_labels, dark_true_all, dark_pred_all, 0.5)
        append_multilabel_from_logits(out["deg_emb_logits"], deg_labels, deg_true_all, deg_pred_all, 0.5)
        append_multilabel_from_logits(out["adv_deg_logits"], deg_labels, adv_deg_true_all, adv_deg_pred_all, 0.5)

        running["loss"] += loss.item()
        running["id"] += loss_id.item()
        running["dpm"] += loss_dpm_deg.item()
        running["deg"] += loss_deg_emb.item()
        running["adv"] += loss_adv_deg.item()
        running["orth"] += loss_orth.item()
        running["fas"] += loss_fas.item()
        running["occ"] += loss_occ.item()
        running["adv_fas"] += loss_adv_fas.item()
        running["adv_occ"] += loss_adv_occ.item()

        loop.set_postfix(
            Loss=f"{loss.item():.4f}",
            ID_Acc=f"{safe_accuracy(id_true_all, id_pred_all):.2f}%",
            OccID=f"{safe_accuracy(occ_id_true_all, occ_id_pred_all):.2f}%",
            FAS_Acc=f"{safe_accuracy(fas_true_all, fas_pred_all):.2f}%"
        )

    num_batches = max(len(dataloader), 1)
    id_metrics = compute_metrics(id_true_all, id_pred_all)
    occ_id_metrics = compute_metrics(occ_id_true_all, occ_id_pred_all)
    non_occ_id_metrics = compute_metrics(non_occ_id_true_all, non_occ_id_pred_all)

    metrics = {}
    metrics.update(id_metrics)
    metrics.update({
        "id_acc": id_metrics["acc"],
        "id_precision": id_metrics["precision"],
        "id_recall": id_metrics["recall"],
        "id_f1": id_metrics["f1"],
        "id_samples": len(id_true_all),
        "occ_id_acc": occ_id_metrics["acc"],
        "occ_id_precision": occ_id_metrics["precision"],
        "occ_id_recall": occ_id_metrics["recall"],
        "occ_id_f1": occ_id_metrics["f1"],
        "occ_id_samples": len(occ_id_true_all),
        "non_occ_id_acc": non_occ_id_metrics["acc"],
        "non_occ_id_precision": non_occ_id_metrics["precision"],
        "non_occ_id_recall": non_occ_id_metrics["recall"],
        "non_occ_id_f1": non_occ_id_metrics["f1"],
        "non_occ_id_samples": len(non_occ_id_true_all),
    })

    metrics.update(compute_binary_metrics(fas_true_all, fas_pred_all, prefix="fas"))
    metrics.update(compute_binary_metrics(occ_det_true_all, occ_det_pred_all, prefix="occ_det"))
    metrics.update(compute_binary_metrics(adv_fas_true_all, adv_fas_pred_all, prefix="adv_fas"))
    metrics.update(compute_binary_metrics(adv_occ_true_all, adv_occ_pred_all, prefix="adv_occ"))
    metrics.update(compute_binary_metrics(blur_true_all, blur_pred_all, prefix="dpm_blur"))
    metrics.update(compute_binary_metrics(glare_true_all, glare_pred_all, prefix="dpm_glare"))
    metrics.update(compute_binary_metrics(dark_true_all, dark_pred_all, prefix="dpm_dark"))
    metrics.update(compute_multilabel_metrics(deg_true_all, deg_pred_all, prefix="deg_emb"))
    metrics.update(compute_multilabel_metrics(adv_deg_true_all, adv_deg_pred_all, prefix="adv_deg"))

    metrics.update({
        "loss": round(running["loss"] / num_batches, 6),
        "loss_id": round(running["id"] / num_batches, 6),
        "loss_dpm_deg": round(running["dpm"] / num_batches, 6),
        "loss_deg_emb": round(running["deg"] / num_batches, 6),
        "loss_adv_deg": round(running["adv"] / num_batches, 6),
        "loss_orth": round(running["orth"] / num_batches, 6),
        "loss_fas": round(running["fas"] / num_batches, 6),
        "loss_occ": round(running["occ"] / num_batches, 6),
        "loss_adv_fas": round(running["adv_fas"] / num_batches, 6),
        "loss_adv_occ": round(running["adv_occ"] / num_batches, 6),
        "samples": len(id_true_all),
        "phase": phase,
        "occ_acc": occ_id_metrics["acc"],
        "occ_f1": occ_id_metrics["f1"],
    })

    return metrics

def print_report(metrics, epoch=None, title="Evaluation Report"):
    head = f"Epoch {epoch} {title}" if epoch is not None else title

    print(f"\n{'=' * 100}")
    print(head)
    print("=" * 100)

    print("\n[Loss]")
    for key in [
        "loss", "loss_id", "loss_dpm_deg", "loss_deg_emb", "loss_adv_deg",
        "loss_orth", "loss_fas", "loss_occ", "loss_adv_fas", "loss_adv_occ"
    ]:
        print(f"{key:24s}: {metrics.get(key, 0.0):.6f}")

    print("\n[ID Branch - 身份识别]")
    print(f"{'id_acc':24s}: {metrics.get('id_acc', 0.0):.2f}%")
    print(f"{'id_precision':24s}: {metrics.get('id_precision', 0.0):.2f}%")
    print(f"{'id_recall':24s}: {metrics.get('id_recall', 0.0):.2f}%")
    print(f"{'id_f1':24s}: {metrics.get('id_f1', 0.0):.2f}%")
    print(f"{'id_samples':24s}: {metrics.get('id_samples', 0)}")

    print("\n[Occluded ID - 遮挡条件下身份识别]")
    print(f"{'occ_id_acc':24s}: {metrics.get('occ_id_acc', 0.0):.2f}%")
    print(f"{'occ_id_precision':24s}: {metrics.get('occ_id_precision', 0.0):.2f}%")
    print(f"{'occ_id_recall':24s}: {metrics.get('occ_id_recall', 0.0):.2f}%")
    print(f"{'occ_id_f1':24s}: {metrics.get('occ_id_f1', 0.0):.2f}%")
    print(f"{'occ_id_samples':24s}: {metrics.get('occ_id_samples', 0)}")

    print("\n[Non-Occluded ID - 无遮挡条件下身份识别]")
    print(f"{'non_occ_id_acc':24s}: {metrics.get('non_occ_id_acc', 0.0):.2f}%")
    print(f"{'non_occ_id_precision':24s}: {metrics.get('non_occ_id_precision', 0.0):.2f}%")
    print(f"{'non_occ_id_recall':24s}: {metrics.get('non_occ_id_recall', 0.0):.2f}%")
    print(f"{'non_occ_id_f1':24s}: {metrics.get('non_occ_id_f1', 0.0):.2f}%")
    print(f"{'non_occ_id_samples':24s}: {metrics.get('non_occ_id_samples', 0)}")

    print("\n[FAS Branch - 活体/伪造检测]")
    print(f"{'fas_acc':24s}: {metrics.get('fas_acc', 0.0):.2f}%")
    print(f"{'fas_precision':24s}: {metrics.get('fas_precision', 0.0):.2f}%")
    print(f"{'fas_recall':24s}: {metrics.get('fas_recall', 0.0):.2f}%")
    print(f"{'fas_f1':24s}: {metrics.get('fas_f1', 0.0):.2f}%")

    print("\n[OCC Detection Branch - 是否遮挡二分类，仅辅助观察]")
    print(f"{'occ_det_acc':24s}: {metrics.get('occ_det_acc', 0.0):.2f}%")
    print(f"{'occ_det_precision':24s}: {metrics.get('occ_det_precision', 0.0):.2f}%")
    print(f"{'occ_det_recall':24s}: {metrics.get('occ_det_recall', 0.0):.2f}%")
    print(f"{'occ_det_f1':24s}: {metrics.get('occ_det_f1', 0.0):.2f}%")

    print("\n" + "=" * 100 + "\n")

def build_optimizer(model, stage_name=None):
    if stage_name == "stage2":
        fas_params = []
        id_params = []
        other_head_params = []
        shared_params = []

        fas_keywords = {"fas_head", "fas_global_proj", "fas_local_branch"}
        id_keywords = {"id_fc", "arcface"}
        other_head_keywords = {
            "occ_head", "adv_fas_head", "adv_occ_head", "deg_fc", "id_to_deg_dim",
            "deg_emb_head", "adv_deg_head", "dpm.head_blur", "dpm.head_glare", "dpm.head_dark"
        }

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            if any(keyword in name for keyword in fas_keywords):
                fas_params.append(parameter)
            elif any(keyword in name for keyword in id_keywords):
                id_params.append(parameter)
            elif any(keyword in name for keyword in other_head_keywords):
                other_head_params.append(parameter)
            else:
                shared_params.append(parameter)

        param_groups = []
        if fas_params:
            param_groups.append({"params": fas_params, "lr": cfg.STAGE2_LR_FAS, "name": "fas_params"})
        if id_params:
            param_groups.append({"params": id_params, "lr": cfg.STAGE2_LR_ID, "name": "id_params"})
        if other_head_params:
            param_groups.append({"params": other_head_params, "lr": cfg.STAGE2_LR_OTHER_HEAD, "name": "other_head_params"})
        if shared_params:
            param_groups.append({"params": shared_params, "lr": cfg.STAGE2_LR_SHARED, "name": "shared_params"})

        return torch.optim.AdamW(param_groups, weight_decay=cfg.STAGE2_WEIGHT_DECAY)

    head_keywords = {
        "id_fc", "arcface", "fas_head", "fas_local_branch", "fas_global_proj",
        "occ_head", "adv_fas_head", "adv_occ_head", "deg_fc", "id_to_deg_dim",
        "deg_emb_head", "adv_deg_head", "dpm.head_blur", "dpm.head_glare", "dpm.head_dark"
    }

    backbone_params = []
    head_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if any(keyword in name for keyword in head_keywords):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.LR_BACKBONE},
            {"params": head_params, "lr": cfg.LR_HEAD},
        ],
        weight_decay=cfg.WEIGHT_DECAY
    )

def get_best_score(metrics, metric_name=None):
    metric_name = metric_name or cfg.BEST_MODEL_METRIC

    id_acc = metrics.get("id_acc", metrics.get("acc", 0.0))
    occ_id_acc = metrics.get("occ_id_acc", 0.0)
    fas_f1 = metrics.get("fas_f1", 0.0)

    if metric_name == "id_acc":
        return id_acc
    if metric_name == "fas_f1":
        return fas_f1
    if metric_name == "occ_id_acc":
        return occ_id_acc

    if metric_name == "id_fas":
        if id_acc < float(getattr(cfg, "ID_ACC_FLOOR", 0.0)):
            return 0.0
        return (
            float(getattr(cfg, "ID_FAS_SCORE_ID_WEIGHT", 0.70)) * id_acc +
            float(getattr(cfg, "ID_FAS_SCORE_FAS_WEIGHT", 0.30)) * fas_f1
        )

    if metric_name == "id_occ_fas":
        if id_acc < float(getattr(cfg, "ID_ACC_FLOOR", 0.0)):
            return 0.0
        if occ_id_acc < float(getattr(cfg, "OCC_ID_ACC_FLOOR", 0.0)):
            return 0.0
        return (
            float(getattr(cfg, "ID_OCC_FAS_SCORE_ID_WEIGHT", 0.50)) * id_acc +
            float(getattr(cfg, "ID_OCC_FAS_SCORE_OCC_ID_WEIGHT", 0.30)) * occ_id_acc +
            float(getattr(cfg, "ID_OCC_FAS_SCORE_FAS_WEIGHT", 0.20)) * fas_f1
        )

    return id_acc

def apply_stage_config(stage_name):
    if stage_name == "stage1":
        cfg.TRAIN_MODE = cfg.STAGE1_TRAIN_MODE
        cfg.LAMBDA_ID = cfg.STAGE1_LAMBDA_ID
        cfg.LAMBDA_DPM_DEG = cfg.STAGE1_LAMBDA_DPM_DEG
        cfg.LAMBDA_DEG_EMB = cfg.STAGE1_LAMBDA_DEG_EMB
        cfg.LAMBDA_ADV_DEG = cfg.STAGE1_LAMBDA_ADV_DEG
        cfg.LAMBDA_ORTHO = cfg.STAGE1_LAMBDA_ORTHO
        cfg.LAMBDA_FAS = cfg.STAGE1_LAMBDA_FAS
        cfg.LAMBDA_OCC = cfg.STAGE1_LAMBDA_OCC
        cfg.LAMBDA_ADV_FAS = cfg.STAGE1_LAMBDA_ADV_FAS
        cfg.LAMBDA_ADV_OCC = cfg.STAGE1_LAMBDA_ADV_OCC
        cfg.USE_FAS_WEIGHTED_SAMPLER = cfg.STAGE1_USE_FAS_WEIGHTED_SAMPLER
        cfg.LABEL_SMOOTHING = cfg.STAGE1_LABEL_SMOOTHING
        cfg.ARCFACE_M = cfg.STAGE1_ARCFACE_M
        cfg.BEST_MODEL_PATH = cfg.STAGE1_BEST_MODEL_PATH
        cfg.BEST_MODEL_METRIC = cfg.STAGE1_BEST_MODEL_METRIC

    elif stage_name == "stage2":
        cfg.TRAIN_MODE = cfg.STAGE2_TRAIN_MODE
        cfg.LAMBDA_ID = cfg.STAGE2_LAMBDA_ID
        cfg.LAMBDA_DPM_DEG = cfg.STAGE2_LAMBDA_DPM_DEG
        cfg.LAMBDA_DEG_EMB = cfg.STAGE2_LAMBDA_DEG_EMB
        cfg.LAMBDA_ADV_DEG = cfg.STAGE2_LAMBDA_ADV_DEG
        cfg.LAMBDA_ORTHO = cfg.STAGE2_LAMBDA_ORTHO
        cfg.LAMBDA_FAS = cfg.STAGE2_LAMBDA_FAS
        cfg.LAMBDA_OCC = cfg.STAGE2_LAMBDA_OCC
        cfg.LAMBDA_ADV_FAS = cfg.STAGE2_LAMBDA_ADV_FAS
        cfg.LAMBDA_ADV_OCC = cfg.STAGE2_LAMBDA_ADV_OCC
        cfg.USE_FAS_WEIGHTED_SAMPLER = cfg.STAGE2_USE_FAS_WEIGHTED_SAMPLER
        cfg.LABEL_SMOOTHING = cfg.STAGE2_LABEL_SMOOTHING
        cfg.ARCFACE_M = cfg.STAGE2_ARCFACE_M
        cfg.BEST_MODEL_PATH = cfg.STAGE2_BEST_MODEL_PATH
        cfg.BEST_MODEL_METRIC = cfg.STAGE2_BEST_MODEL_METRIC

    else:
        raise ValueError(f"未知阶段: {stage_name}")

def build_model(total_classes):
    return CD_MAGNet(
        num_classes=total_classes,
        embedding_dim=cfg.EMBEDDING_DIM,
        deg_embedding_dim=cfg.DEG_EMBEDDING_DIM,
        adv_alpha=cfg.LAMBDA_ADV_DEG,
        arcface_s=cfg.ARCFACE_S,
        arcface_m=cfg.ARCFACE_M,
        backbone_name=cfg.BACKBONE_NAME,
        backbone_dim=cfg.BACKBONE_DIM,
        backbone_input_size=cfg.BACKBONE_INPUT_SIZE,
        use_dinov2_pretrained=cfg.USE_DINOV2_PRETRAINED,
        use_dpm=cfg.USE_DPM,                    # 新增
        use_fas_local_branch=cfg.USE_FAS_LOCAL_BRANCH,
        fas_local_dim=cfg.FAS_LOCAL_DIM,
        use_fas_sobel=cfg.USE_FAS_SOBEL,
        fas_sobel_alpha=cfg.FAS_SOBEL_ALPHA,
    ).to(cfg.DEVICE)

def train_one_stage(stage_name, init_state_dict=None):
    apply_stage_config(stage_name)
    seed_everything(cfg.SEED)

    train_loader, val_loader, _, train_set, _, total_classes = build_dataloaders()
    fas_pos_weight_value, _, _ = compute_fas_pos_weight(train_set)

    model = build_model(total_classes)

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict, strict=False)

    apply_freeze_strategy(model, stage_name=stage_name)
    print_trainable_parameters(model)

    criterion_id = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    criterion_bce_logits = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, stage_name=stage_name)

    total_epochs = cfg.STAGE1_EPOCHS if stage_name == "stage1" else cfg.STAGE2_EPOCHS
    patience = cfg.STAGE1_PATIENCE if stage_name == "stage1" else cfg.STAGE2_PATIENCE
    metric_name = cfg.STAGE1_BEST_MODEL_METRIC if stage_name == "stage1" else cfg.STAGE2_BEST_MODEL_METRIC

    if cfg.USE_COSINE_LR:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - cfg.WARMUP_EPOCHS),
            eta_min=cfg.MIN_LR
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (
            (epoch + 1) / max(1, cfg.WARMUP_EPOCHS)
            if epoch < cfg.WARMUP_EPOCHS else 1.0
        )
    )

    logger = PaperLogger()
    best_val_score = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'=' * 80}")
    print(
        f"开始训练 {stage_name} | 模式: {cfg.TRAIN_MODE} | "
        f"Epochs: {total_epochs} | BS: {cfg.BATCH_SIZE} | 身份类别数: {total_classes}"
    )
    print(f"Backbone                 : {cfg.BACKBONE_NAME}")
    print(f"最佳模型指标             : {metric_name}")
    print("=" * 80 + "\n")

    for epoch in range(1, total_epochs + 1):
        train_metrics = run_one_epoch_disentangle(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion_id=criterion_id,
            criterion_bce_logits=criterion_bce_logits,
            device=cfg.DEVICE,
            epoch=epoch,
            phase="train",
            fas_pos_weight_value=fas_pos_weight_value,
            stage_name=stage_name
        )

        val_metrics = run_one_epoch_disentangle(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            criterion_id=criterion_id,
            criterion_bce_logits=criterion_bce_logits,
            device=cfg.DEVICE,
            epoch=epoch,
            phase="val",
            fas_pos_weight_value=fas_pos_weight_value,
            stage_name=stage_name
        )

        print_report(train_metrics, epoch, f"{stage_name} 训练集")
        logger.update_train(epoch, train_metrics)

        print_report(val_metrics, epoch, f"{stage_name} 验证集")
        logger.update_val(epoch, val_metrics)

        if epoch <= cfg.WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()

        lr_info = []
        for index, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group_{index}")
            lr_info.append(f"{name}: {group['lr']:.2e}")
        print("LR " + " | ".join(lr_info) + "\n")

        current_score = get_best_score(val_metrics, metric_name=metric_name)

        if current_score > best_val_score:
            best_val_score = current_score
            best_epoch = epoch
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            patience_counter = 0

            print(
                f"✅ {stage_name} 最佳模型已保存 | 指标: {metric_name} | "
                f"Score: {best_val_score:.2f} | "
                f"Val ID Acc: {val_metrics.get('id_acc', 0.0):.2f}% | "
                f"Val Occ-ID Acc: {val_metrics.get('occ_id_acc', 0.0):.2f}% | "
                f"Val FAS F1: {val_metrics.get('fas_f1', 0.0):.2f}% | "
                f"Epoch {best_epoch}\n"
            )
        else:
            patience_counter += 1
            print(
                f"{stage_name} 验证指标未提升 ({patience_counter}/{patience}) | "
                f"当前 Score: {current_score:.2f} | 最佳 Score: {best_val_score:.2f}\n"
            )

            if patience_counter >= patience:
                print(
                    f"{stage_name} 早停触发！最佳指标: {best_val_score:.2f} "
                    f"(Epoch {best_epoch}, Metric={metric_name})"
                )
                break

        if cfg.DRAW_ALL_BRANCH_CURVES:
            logger.draw_curves()

    best_state = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)

    print(f"\n{'=' * 80}")
    print(f"{stage_name} 训练结束")
    print("=" * 80)
    print(f"最佳 Epoch     : {best_epoch}")
    print(f"最佳指标类型   : {metric_name}")
    print(f"最佳验证 Score : {best_val_score:.2f}")
    print(f"最佳模型路径   : {cfg.BEST_MODEL_PATH}")
    print("=" * 80 + "\n")

    return best_state, best_epoch, best_val_score

def export_final_id_confusion_matrices(model, test_loader, id_to_idx):
    model.eval()

    all_true, all_pred = [], []
    occ_true, occ_pred = [], []
    non_occ_true, non_occ_pred = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="导出 ID 混淆矩阵", ncols=140):
            images, id_labels, _, occ_labels, _, _, _ = batch

            images = images.to(cfg.DEVICE, non_blocking=True)
            id_labels = id_labels.to(cfg.DEVICE, non_blocking=True).long()
            occ_labels = occ_labels.to(cfg.DEVICE, non_blocking=True).float().view(-1)

            # =============================================================
            # 正确推理：测试混淆矩阵也绝不能传真实 ID 标签。
            # =============================================================
            out = model(images, label_id=None, id_only=False)
            logits_id = out["out_id"]

            valid_id_mask = id_labels >= 0
            if valid_id_mask.sum().item() == 0:
                continue

            preds = torch.argmax(logits_id[valid_id_mask], dim=1)
            true_ids = id_labels[valid_id_mask]
            occ_valid = occ_labels[valid_id_mask]

            all_true.extend(true_ids.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

            occ_mask = occ_valid == 1
            if occ_mask.sum().item() > 0:
                occ_true.extend(true_ids[occ_mask].detach().cpu().tolist())
                occ_pred.extend(preds[occ_mask].detach().cpu().tolist())

            non_occ_mask = occ_valid == 0
            if non_occ_mask.sum().item() > 0:
                non_occ_true.extend(true_ids[non_occ_mask].detach().cpu().tolist())
                non_occ_pred.extend(preds[non_occ_mask].detach().cpu().tolist())

    save_id_confusion_matrix(
        all_true,
        all_pred,
        id_to_idx,
        "id_confusion_matrix_all",
        "ID Confusion Matrix - All Valid ID Samples"
    )
    save_id_confusion_matrix(
        occ_true,
        occ_pred,
        id_to_idx,
        "id_confusion_matrix_occ_only",
        "ID Confusion Matrix - Occluded Samples Only"
    )
    save_id_confusion_matrix(
        non_occ_true,
        non_occ_pred,
        id_to_idx,
        "id_confusion_matrix_non_occ",
        "ID Confusion Matrix - Non-Occluded Samples Only"
    )

def export_test_sample_predictions(model, test_loader, test_set, id_to_idx):
    """导出测试集逐样本预测，以及可由逐样本结果复核的论文指标。"""
    model.eval()
    idx_to_id = {value: key for key, value in id_to_idx.items()}
    rows = []
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="导出测试集逐样本结果", ncols=140):
            images, id_labels, fas_labels, occ_labels, blur_labels, glare_labels, dark_labels = batch
            batch_size = images.size(0)
            sample_meta = test_set.samples[sample_offset:sample_offset + batch_size]
            sample_offset += batch_size

            images = images.to(cfg.DEVICE, non_blocking=True)
            out = model(images, label_id=None, id_only=False)

            id_prob = torch.softmax(out["out_id"], dim=1)
            id_score, id_pred = id_prob.max(dim=1)
            fas_score = torch.sigmoid(out["out_fas"]).view(-1)
            occ_score = torch.sigmoid(out["out_occ"]).view(-1)
            blur_score = torch.sigmoid(out["logits_blur"]).view(-1) if cfg.USE_DPM else None
            glare_score = torch.sigmoid(out["logits_glare"]).view(-1) if cfg.USE_DPM else None
            dark_score = torch.sigmoid(out["logits_dark"]).view(-1) if cfg.USE_DPM else None

            for i, meta in enumerate(sample_meta):
                true_id = int(id_labels[i].item())
                pred_id = int(id_pred[i].item())
                true_fas = float(fas_labels[i].item())
                true_occ = float(occ_labels[i].item())
                true_blur = float(blur_labels[i].item())
                true_glare = float(glare_labels[i].item())
                true_dark = float(dark_labels[i].item())
                fas_prob = float(fas_score[i].cpu().item())
                occ_prob = float(occ_score[i].cpu().item())

                # sample_id 只写相对匿名信息，避免在日志中暴露数据根目录。
                sample_id = os.path.join(
                    str(meta.get("source", "unknown")),
                    str(meta.get("subset", "unknown")),
                    os.path.basename(str(meta.get("image_path", i)))
                ).replace("\\", "/")

                rows.append({
                    "sample_id": sample_id,
                    "split": "test",
                    "source": meta.get("source", "unknown"),
                    "subset": meta.get("subset", "unknown"),
                    "person_name": meta.get("person_name", "unknown"),
                    "true_id_index": true_id,
                    "true_id_name": idx_to_id.get(true_id, "unlabeled") if true_id >= 0 else "unlabeled",
                    "pred_id_index": pred_id,
                    "pred_id_name": idx_to_id.get(pred_id, str(pred_id)),
                    "id_score": float(id_score[i].cpu().item()),
                    "id_correct": int(true_id >= 0 and pred_id == true_id),
                    "true_fas": int(true_fas) if true_fas >= 0 else -1,
                    "pred_fas": int(fas_prob >= cfg.FAS_THRESHOLD) if true_fas >= 0 else -1,
                    "fas_score_bona_fide": fas_prob,
                    "true_occ": int(true_occ) if true_occ >= 0 else -1,
                    "pred_occ": int(occ_prob >= cfg.OCC_THRESHOLD) if true_occ >= 0 else -1,
                    "occ_score": occ_prob,
                    "true_blur": int(true_blur),
                    "pred_blur": int(float(blur_score[i].cpu().item()) >= 0.5) if blur_score is not None else -1,
                    "blur_score": float(blur_score[i].cpu().item()) if blur_score is not None else np.nan,
                    "true_glare": int(true_glare),
                    "pred_glare": int(float(glare_score[i].cpu().item()) >= 0.5) if glare_score is not None else -1,
                    "glare_score": float(glare_score[i].cpu().item()) if glare_score is not None else np.nan,
                    "true_dark": int(true_dark),
                    "pred_dark": int(float(dark_score[i].cpu().item()) >= 0.5) if dark_score is not None else -1,
                    "dark_score": float(dark_score[i].cpu().item()) if dark_score is not None else np.nan,
                })

    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    prediction_path = os.path.join(cfg.LOG_DIR, "test_sample_predictions.csv")
    df = pd.DataFrame(rows)
    df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    # 约定：FAS 标签 1=bona fide（real/mask），0=attack（fake）。
    fas_df = df[df["true_fas"].isin([0, 1])].copy()
    summary = {
        "fas_threshold": float(cfg.FAS_THRESHOLD),
        "occ_threshold": float(cfg.OCC_THRESHOLD),
        "fas_label_1": "bona_fide",
        "fas_label_0": "attack",
        "test_samples_all": int(len(df)),
        "fas_valid_samples": int(len(fas_df)),
    }

    if not fas_df.empty:
        tn, fp, fn, tp = confusion_matrix(
            fas_df["true_fas"], fas_df["pred_fas"], labels=[0, 1]
        ).ravel()
        # 预测正类表示 bona fide：FP=攻击被接受，FN=真实活体被拒绝。
        apcer = fp / (tn + fp) if (tn + fp) else np.nan
        bpcer = fn / (fn + tp) if (fn + tp) else np.nan
        summary.update({
            "tn_attack_rejected": int(tn),
            "fp_attack_accepted": int(fp),
            "fn_bona_fide_rejected": int(fn),
            "tp_bona_fide_accepted": int(tp),
            "accuracy_percent": float((fas_df["true_fas"] == fas_df["pred_fas"]).mean() * 100),
            "precision_percent": float(precision_score(fas_df["true_fas"], fas_df["pred_fas"], zero_division=0) * 100),
            "recall_percent": float(recall_score(fas_df["true_fas"], fas_df["pred_fas"], zero_division=0) * 100),
            "f1_percent": float(f1_score(fas_df["true_fas"], fas_df["pred_fas"], zero_division=0) * 100),
            "APCER_percent": float(apcer * 100),
            "BPCER_percent": float(bpcer * 100),
            "ACER_percent": float((apcer + bpcer) * 50),
            "HTER_percent": float((apcer + bpcer) * 50),
            "AUC_percent": float(roc_auc_score(fas_df["true_fas"], fas_df["fas_score_bona_fide"]) * 100)
                if fas_df["true_fas"].nunique() == 2 else np.nan,
        })

    summary_path = os.path.join(cfg.LOG_DIR, "test_sample_metrics_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[逐样本结果] 已保存: {prediction_path}")
    print(f"[逐样本指标] 已保存: {summary_path}")

def train_and_verify():
    seed_everything(cfg.SEED)

    if not cfg.USE_STAGE_TRAINING:
        raise NotImplementedError("当前版本仅支持 USE_STAGE_TRAINING=True")

    print("\n" + "=" * 80)
    print("Stage 1: 先训练 ID 主任务")
    print("=" * 80)
    stage1_state, stage1_best_epoch, stage1_best_score = train_one_stage("stage1")

    print("\n" + "=" * 80)
    print("Stage 2: 在 Stage 1 基础上训练 ID + FAS")
    print("=" * 80)
    stage2_state, stage2_best_epoch, stage2_best_score = train_one_stage(
        "stage2",
        init_state_dict=stage1_state
    )

    # 最终测试必须恢复 Stage 2 配置；尤其是 ArcFace margin 参数、损失配置等。
    apply_stage_config("stage2")

    _, _, test_loader, train_set, id_to_idx, total_classes = build_dataloaders()
    test_set = test_loader.dataset
    fas_pos_weight_value, _, _ = compute_fas_pos_weight(train_set)

    model = build_model(total_classes)
    model.load_state_dict(stage2_state, strict=True)
    model.eval()

    criterion_id = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
    criterion_bce_logits = nn.BCEWithLogitsLoss()

    test_metrics = run_one_epoch_disentangle(
        model=model,
        dataloader=test_loader,
        optimizer=None,
        criterion_id=criterion_id,
        criterion_bce_logits=criterion_bce_logits,
        device=cfg.DEVICE,
        epoch=stage2_best_epoch,
        phase="test",
        fas_pos_weight_value=fas_pos_weight_value,
        stage_name="stage2"
    )

    print_report(test_metrics, title="测试集最终评估")

    logger = PaperLogger()
    logger.update_test(test_metrics)

    export_final_id_confusion_matrices(model, test_loader, id_to_idx)
    export_test_sample_predictions(model, test_loader, test_set, id_to_idx)

    print(f"\n{'=' * 80}")
    print("分阶段训练总结")
    print("=" * 80)
    print(f"Stage 1 最佳 Epoch       : {stage1_best_epoch}")
    print(f"Stage 1 最佳 Score       : {stage1_best_score:.2f}")
    print(f"Stage 2 最佳 Epoch       : {stage2_best_epoch}")
    print(f"Stage 2 最佳 Score       : {stage2_best_score:.2f}")
    print(f"测试集 ID 准确率         : {test_metrics.get('id_acc', 0.0):.2f}%")
    print(f"测试集 遮挡ID准确率      : {test_metrics.get('occ_id_acc', 0.0):.2f}%")
    print(f"测试集 无遮挡ID准确率    : {test_metrics.get('non_occ_id_acc', 0.0):.2f}%")
    print(f"测试集 ID F1             : {test_metrics.get('id_f1', 0.0):.2f}%")
    print(f"测试集 FAS 准确率        : {test_metrics.get('fas_acc', 0.0):.2f}%")
    print(f"测试集 FAS F1            : {test_metrics.get('fas_f1', 0.0):.2f}%")
    print(f"日志目录                 : {cfg.LOG_DIR}")
    print(f"四宫格曲线图             : {os.path.join(cfg.LOG_DIR, cfg.CURVE_FIG_NAME)}")
    print(f"ID混淆矩阵(全部)         : {os.path.join(cfg.LOG_DIR, 'id_confusion_matrix_all.csv')}")
    print(f"ID混淆矩阵(遮挡)         : {os.path.join(cfg.LOG_DIR, 'id_confusion_matrix_occ_only.csv')}")
    print(f"ID混淆矩阵(无遮挡)       : {os.path.join(cfg.LOG_DIR, 'id_confusion_matrix_non_occ.csv')}")
    print(f"Stage 1 最佳模型         : {cfg.STAGE1_BEST_MODEL_PATH}")
    print(f"Stage 2 最佳模型         : {cfg.STAGE2_BEST_MODEL_PATH}")
    print("=" * 80 + "\n")

    return test_metrics

if __name__ == "__main__":
    try:
        print(f"\n{'=' * 80}")
        print("开始初始化")
        print("=" * 80)
        print(f"RAW_DATA_ROOT      : {cfg.RAW_DATA_ROOT}  (存在: {os.path.exists(cfg.RAW_DATA_ROOT)})")
        print(f"ALIGNED_REAL_ROOT  : {cfg.ALIGNED_REAL_ROOT}  (存在: {os.path.exists(cfg.ALIGNED_REAL_ROOT)})")
        print(f"ALIGNED_MASK_ROOT  : {cfg.ALIGNED_MASK_ROOT}  (存在: {os.path.exists(cfg.ALIGNED_MASK_ROOT)})")
        print(f"训练模式           : {cfg.TRAIN_MODE}")
        print(f"Backbone           : {cfg.BACKBONE_NAME}")
        print(f"分阶段训练         : {cfg.USE_STAGE_TRAINING}")

        if cfg.REQUIRE_ALIGNED_REAL and not os.path.exists(cfg.ALIGNED_REAL_ROOT):
            raise FileNotFoundError(f"real 五点对齐数据集不存在: {cfg.ALIGNED_REAL_ROOT}")

        if cfg.REQUIRE_ALIGNED_MASK and not os.path.exists(cfg.ALIGNED_MASK_ROOT):
            raise FileNotFoundError(f"mask 五点对齐数据集不存在: {cfg.ALIGNED_MASK_ROOT}")

        test_metrics = train_and_verify()

        print("\n" + "=" * 80)
        print("训练成功完成")
        print("=" * 80)
        print(f"测试集 ID 准确率        : {test_metrics.get('id_acc', 0.0):.2f}%")
        print(f"测试集 遮挡ID准确率     : {test_metrics.get('occ_id_acc', 0.0):.2f}%")
        print(f"测试集 无遮挡ID准确率   : {test_metrics.get('non_occ_id_acc', 0.0):.2f}%")
        print(f"测试集 FAS Acc          : {test_metrics.get('fas_acc', 0.0):.2f}%")
        print(f"测试集 FAS F1           : {test_metrics.get('fas_f1', 0.0):.2f}%")
        print(f"曲线图路径              : {os.path.join(cfg.LOG_DIR, cfg.CURVE_FIG_NAME)}")
        print(f"ID混淆矩阵(全部)        : {os.path.join(cfg.LOG_DIR, 'id_confusion_matrix_all.csv')}")
        print(f"ID混淆矩阵(遮挡)        : {os.path.join(cfg.LOG_DIR, 'id_confusion_matrix_occ_only.csv')}")
        print(f"ID混淆矩阵(无遮挡)      : {os.path.join(cfg.LOG_DIR, 'id_confusion_matrix_non_occ.csv')}")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n训练被用户中断")

    except FileNotFoundError as error:
        print("\n文件路径错误:")
        print(error)

    except Exception as error:
        print("\n训练过程中出现错误:")
        print(f"{type(error).__name__}: {error}")
        import traceback
        traceback.print_exc()