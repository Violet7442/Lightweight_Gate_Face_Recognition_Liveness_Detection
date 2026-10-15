# train.py
import os
# 完全关闭torch.compile/dynamo，阻止加载出错的onnx导出模块
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score
)
import numpy as np

from config import cfg
from models import CD_MAGNet
from dataset import (
    AirportFaceDataset,

    get_base_transforms,
    get_test_base_transforms,
    UnconstrainedEnvWrapper
)


# =========================================================
# Focal Loss
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        preds = torch.clamp(preds, 1e-7, 1.0 - 1e-7)
        bce = -(
                targets * torch.log(preds) +
                (1.0 - targets) * torch.log(1.0 - preds)
        )
        pt = torch.where(targets == 1.0, preds, 1.0 - preds)
        alpha_t = torch.where(
            targets == 1.0,
            torch.tensor(self.alpha, device=preds.device),
            torch.tensor(1.0 - self.alpha, device=preds.device)
        )
        focal_loss = alpha_t * (1.0 - pt) ** self.gamma * bce
        return focal_loss.mean()


# =========================================================
# 自动搜索 FAS 最佳阈值
# =========================================================
def find_best_fas_threshold(y_true, y_pred_logits, thresholds=np.arange(0.5, 0.96, 0.01)):
    best_acer = float('inf')
    best_th = 0.5
    best_metrics = {}

    for th in thresholds:
        y_pred = (y_pred_logits > th).astype(int)
        apcer, bpcer, acer = compute_acer(y_true, y_pred)
        if acer < best_acer:
            best_acer = acer
            best_th = th
            best_metrics = {
                'apcer': apcer,
                'bpcer': bpcer,
                'acer': acer,
                'threshold': th,
                'f1': f1_score(y_true, y_pred, zero_division=0) * 100.0,
                'precision': precision_score(y_true, y_pred, zero_division=0) * 100.0,
                'recall': recall_score(y_true, y_pred, zero_division=0) * 100.0
            }

    return best_th, best_metrics


# =========================================================
# 计算 ACER (已修复 SyntaxError)
# =========================================================
def compute_acer(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        apcer = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acer = (apcer + bpcer) / 2.0
        return apcer * 100.0, bpcer * 100.0, acer * 100.0
    except Exception:
        return 0.0, 0.0, 0.0


# =========================================================
# 分阶段训练工具
# =========================================================
def set_all_trainable(model):
    for p in model.parameters():
        p.requires_grad = True


def set_stage1_trainable(model):
    """Stage1: 仅训练 ID/OCC/IDE，冻结 FAS + ADV"""
    for p in model.parameters():
        p.requires_grad = True

    freeze_modules = [
        model.att_fas,
        model.fc_fas,
        model.head_fas,
        model.adv_h_fas,
        model.adv_h_occ
    ]
    for m in freeze_modules:
        for p in m.parameters():
            p.requires_grad = False


def set_stage2_trainable(model):
    """Stage2: 训练 ID/FAS/OCC/IDE，仍冻结 ADV"""
    for p in model.parameters():
        p.requires_grad = True
    for p in model.adv_h_fas.parameters():
        p.requires_grad = False
    for p in model.adv_h_occ.parameters():
        p.requires_grad = False


def get_stage_name(epoch):
    if epoch <= cfg.WARMUP_EPOCHS:
        return "stage1_private_only"
    elif epoch <= cfg.WARMUP_EPOCHS + 20:  # Stage 2 持续 20 轮 (Epoch 11~30)
        return "stage2_joint_finetune"
    else:
        return "stage3_with_rmf"           # Stage 3 持续 30 轮 (Epoch 31~60)

def get_loss_weights(epoch):
    """动态权重：Stage1 重 ID，Stage2 平衡，Stage3 小幅加 RMFRD"""
    if epoch <= cfg.WARMUP_EPOCHS:
        return {
            "id": 10.0, "fas": 0.0, "occ": 2.0, "ide": 1.0, "adv": 0.0
        }
    elif epoch <= cfg.WARMUP_EPOCHS + 20:
        # FAS 权重在 Stage2 的前 5 轮缓慢增加 (Warm-up)
        fas_ramp = min(1.0, (epoch - cfg.WARMUP_EPOCHS) / 5.0)
        return {
            "id": 10.0, "fas": 2.0 * fas_ramp, "occ": 2.0, "ide": 1.0, "adv": 0.0
        }
    else:
        return {
            "id": 8.0, "fas": 2.0, "occ": 2.0, "ide": 1.0, "adv": 0.0
        }
# =========================================================
# Logger
# =========================================================
class PaperLogger:
    def __init__(self):
        self.train_history = []
        self.val_history = []
        self.test_history = []
        self.log_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def update_train(self, epoch, metrics):
        row = copy.deepcopy(metrics)
        row["epoch"] = epoch
        self.train_history.append(row)
        pd.DataFrame(self.train_history).to_csv(
            os.path.join(self.log_dir, "train_log.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    def update_val(self, epoch, metrics):
        row = copy.deepcopy(metrics)
        row["epoch"] = epoch
        self.val_history.append(row)
        pd.DataFrame(self.val_history).to_csv(
            os.path.join(self.log_dir, "val_log.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    def update_test(self, metrics):
        row = copy.deepcopy(metrics)
        self.test_history.append(row)
        pd.DataFrame(self.test_history).to_csv(
            os.path.join(self.log_dir, "final_test_report.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    def draw_curves(self):
        if len(self.train_history) == 0:
            return
        train_df = pd.DataFrame(self.train_history)
        plt.figure(figsize=(12, 7))
        if "total_loss" in train_df.columns:
            plt.plot(train_df["epoch"], train_df["total_loss"], label="Train Total Loss", color="black")
        if "loss_id" in train_df.columns:
            plt.plot(train_df["epoch"], train_df["loss_id"], label="Train ID Loss")
        if "loss_fas" in train_df.columns:
            plt.plot(train_df["epoch"], train_df["loss_fas"], label="Train FAS Loss")
        if "loss_occ" in train_df.columns:
            plt.plot(train_df["epoch"], train_df["loss_occ"], label="Train OCC Loss")
        if "loss_ide" in train_df.columns:
            plt.plot(train_df["epoch"], train_df["loss_ide"], label="Train IDE Loss")
        if len(self.val_history) > 0:
            val_df = pd.DataFrame(self.val_history)
            if "composite_score" in val_df.columns:
                plt.plot(val_df["epoch"], val_df["composite_score"], label="Val Composite Score", linestyle="--",
                         color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training / Validation Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_validation_curves.png"), dpi=300)
        plt.close()


# =========================================================
# Metrics
# =========================================================
def compute_composite_score(metrics):
    id_acc = min(metrics.get('acc_id', 0.0), 100.0) / 100.0
    fas_acer_score = max(0.0, 100.0 - metrics.get('acer_fas', 100.0)) / 100.0
    occ_f1 = min(metrics.get('f1_occ', 0.0), 100.0) / 100.0
    ide_acc = min(metrics.get('acc_ide', 0.0), 100.0) / 100.0
    return round((
                         cfg.SCORE_WEIGHT_ID_ACC * id_acc +
                         cfg.SCORE_WEIGHT_FAS_ACER * fas_acer_score +
                         cfg.SCORE_WEIGHT_OCC_F1 * occ_f1 +
                         cfg.SCORE_WEIGHT_IDE_ACC * ide_acc
                 ) * 100.0, 2)


def get_detailed_metrics(y_true, y_pred, task_name=""):
    if len(y_true) == 0 or len(y_pred) == 0:
        return {f"{task_name}_precision": 0.0, f"{task_name}_recall": 0.0, f"{task_name}_f1": 0.0}
    try:
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        return {
            f"{task_name}_precision": round(precision * 100.0, 2),
            f"{task_name}_recall": round(recall * 100.0, 2),
            f"{task_name}_f1": round(f1 * 100.0, 2)
        }
    except Exception:
        return {f"{task_name}_precision": 0.0, f"{task_name}_recall": 0.0, f"{task_name}_f1": 0.0}


def init_stats():
    return {
        'loss_total': 0.0,
        'loss_id': 0.0,
        'loss_fas': 0.0,
        'loss_occ': 0.0,
        'loss_ide': 0.0,
        'loss_adv': 0.0,
        'correct_id': 0, 'samples_id': 0,
        'correct_ide': 0, 'samples_ide': 0,
        'all_p_fas_logits': [], 'all_t_fas': [],
        'all_p_occ': [], 'all_t_occ': [],
        'all_p_id': [], 'all_t_id': [],
        'all_p_ide': [], 'all_t_ide': [],
        'adv_fas_pred': [], 'adv_fas_true': [],
        'adv_occ_pred': [], 'adv_occ_true': []
    }


def summarize_metrics(stats, num_batches, phase="train", best_fas_th=None):
    num_batches = max(num_batches, 1)

    # FAS：使用 best_th 或默认 0.5
    th = best_fas_th if best_fas_th is not None else 0.5
    pred_fas_binary = (np.array(stats['all_p_fas_logits']) > th).astype(int)
    apcer, bpcer, acer = compute_acer(stats['all_t_fas'], pred_fas_binary)

    id_metrics = get_detailed_metrics(stats['all_t_id'], stats['all_p_id'], "id")
    occ_metrics = get_detailed_metrics(stats['all_t_occ'], stats['all_p_occ'], "occ")
    ide_metrics = get_detailed_metrics(stats['all_t_ide'], stats['all_p_ide'], "ide")

    adv_fas_acc = (np.array(stats['adv_fas_pred']) == np.array(stats['adv_fas_true'])).mean() * 100.0 if stats[
        'adv_fas_true'] else 0.0
    adv_occ_acc = (np.array(stats['adv_occ_pred']) == np.array(stats['adv_occ_true'])).mean() * 100.0 if stats[
        'adv_occ_true'] else 0.0
    ds_score = 0.5 - (abs(adv_fas_acc - 50.0) + abs(adv_occ_acc - 50.0)) / 200.0
    ds_score = max(0.0, min(0.5, ds_score))

    acc_id = stats['correct_id'] / stats['samples_id'] * 100.0 if stats['samples_id'] > 0 else 0.0
    acc_ide = stats['correct_ide'] / stats['samples_ide'] * 100.0 if stats['samples_ide'] > 0 else 0.0
    f1_fas = f1_score(stats['all_t_fas'], pred_fas_binary, zero_division=0) * 100.0 if stats['all_t_fas'] else 0.0
    f1_occ = f1_score(stats['all_t_occ'], stats['all_p_occ'], zero_division=0) * 100.0 if stats['all_t_occ'] else 0.0

    metrics = {
        'phase': phase,
        'loss_id': stats['loss_id'] / num_batches,
        'loss_fas': stats['loss_fas'] / num_batches,
        'loss_occ': stats['loss_occ'] / num_batches,
        'loss_ide': stats['loss_ide'] / num_batches,
        'loss_adv': stats['loss_adv'] / num_batches,
        'total_loss': stats['loss_total'] / num_batches,
        'acc_id': acc_id,
        'id_precision': id_metrics.get('id_precision', 0.0),
        'id_recall': id_metrics.get('id_recall', 0.0),
        'id_f1': id_metrics.get('id_f1', 0.0),
        'samples_id': stats['samples_id'],
        'f1_fas': f1_fas,
        'fas_precision': precision_score(stats['all_t_fas'], pred_fas_binary, zero_division=0) * 100.0 if stats[
            'all_t_fas'] else 0.0,
        'fas_recall': recall_score(stats['all_t_fas'], pred_fas_binary, zero_division=0) * 100.0 if stats[
            'all_t_fas'] else 0.0,
        'apcer_fas': apcer,
        'bpcer_fas': bpcer,
        'acer_fas': acer,
        'samples_fas': len(stats['all_t_fas']),
        'f1_occ': f1_occ,
        'occ_precision': occ_metrics.get('occ_precision', 0.0),
        'occ_recall': occ_metrics.get('occ_recall', 0.0),
        'occ_f1_weighted': occ_metrics.get('occ_f1', 0.0),
        'samples_occ': len(stats['all_t_occ']),
        'acc_ide': acc_ide,
        'ide_precision': ide_metrics.get('ide_precision', 0.0),
        'ide_recall': ide_metrics.get('ide_recall', 0.0),
        'ide_f1': ide_metrics.get('ide_f1', 0.0),
        'samples_ide': stats['samples_ide'],
        'adv_fas_acc': adv_fas_acc,
        'adv_occ_acc': adv_occ_acc,
        'decoupling_score': ds_score,
        'best_fas_threshold': th
    }
    metrics['composite_score'] = compute_composite_score(metrics)
    return metrics


# =========================================================
# One Epoch
# =========================================================
def run_one_epoch(
        model,
        dataloader,
        optimizer,
        criterions,
        device,
        epoch,
        phase="train",
        val_error_samples=None,
        loss_weights=None,
        enable_fas=True,
        enable_adv=False,
        best_fas_th=None
):
    is_train = phase == "train"
    if is_train:
        model.train()
    else:
        model.eval()

    if loss_weights is None:
        loss_weights = {
            "id": cfg.WEIGHT_ID,
            "fas": cfg.WEIGHT_FAS,
            "occ": cfg.WEIGHT_OCC,
            "ide": cfg.WEIGHT_IDE,
            "adv": cfg.WEIGHT_ADV
        }

    stats = init_stats()
    criterion_id, criterion_fas, criterion_occ, criterion_ide = criterions
    criterion_bce_logits = nn.BCEWithLogitsLoss()

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [{phase.upper()}]", ncols=120)

    if phase == "val" and val_error_samples is not None:
        val_error_samples.clear()

    for batch_idx, (images, id_labels, fas_labels, occ_labels, ide_labels) in enumerate(loop):
        images = images.to(device)
        id_labels = id_labels.to(device)
        fas_labels = fas_labels.float().to(device)
        occ_labels = occ_labels.float().to(device)
        ide_labels = ide_labels.to(device)

        with torch.set_grad_enabled(is_train):
            if is_train:
                outputs = model(images, label_id=id_labels)
                pred_id = outputs["out_id"]
                pred_fas = outputs["out_fas"].view(-1)
                pred_occ = outputs["out_occ"].view(-1)
                logits_blur = outputs["logits_blur"].view(-1)
                logits_glare = outputs["logits_glare"].view(-1)
                logits_dark = outputs["logits_dark"].view(-1)
                adv_fas = outputs["adv_fas"].view(-1)
                adv_occ = outputs["adv_occ"].view(-1)
            else:
                pred_id, pred_fas, pred_occ, logits_blur, logits_glare, logits_dark = model(images, label_id=None)
                pred_fas = pred_fas.view(-1)
                pred_occ = pred_occ.view(-1)
                logits_blur, logits_glare, logits_dark = logits_blur.view(-1), logits_glare.view(-1), logits_dark.view(
                    -1)
                adv_fas = adv_occ = None

            # ✅ 核心修复：区分 real/mask/fake，且保证 RMFRD 正确参与 OCC
            valid_id = (id_labels != cfg.IGNORE_INDEX)
            valid_fas = (fas_labels != cfg.IGNORE_INDEX)
            # OCC 参与条件：不能是 fake (即 fas_labels != 0.0) 且有 occ 标签
            valid_occ = (fas_labels != 0.0) & (occ_labels != cfg.IGNORE_INDEX)
            valid_ide = (ide_labels != cfg.IGNORE_INDEX)

            # Stage1 禁用 FAS loss
            if is_train and not enable_fas:
                valid_fas_for_loss = torch.zeros_like(valid_fas, dtype=torch.bool)
            else:
                valid_fas_for_loss = valid_fas

            # ID Loss
            if valid_id.sum() > 0:
                loss_id = criterion_id(pred_id[valid_id], id_labels[valid_id])
            else:
                loss_id = torch.tensor(0.0, device=device)

            # IDE Loss
            if valid_ide.sum() > 0:
                labels_blur = (ide_labels[valid_ide] == 2).float()
                labels_glare = (ide_labels[valid_ide] == 1).float()
                labels_dark = (ide_labels[valid_ide] == 3).float()
                loss_ide = (
                                   criterion_bce_logits(logits_blur[valid_ide], labels_blur) +
                                   criterion_bce_logits(logits_glare[valid_ide], labels_glare) +
                                   criterion_bce_logits(logits_dark[valid_ide], labels_dark)
                           ) / 3.0
            else:
                loss_ide = torch.tensor(0.0, device=device)

            # FAS Loss
            if valid_fas_for_loss.sum() > 0:
                loss_fas = criterion_fas(pred_fas[valid_fas_for_loss], fas_labels[valid_fas_for_loss])
            else:
                loss_fas = torch.tensor(0.0, device=device)

            # FAS logits 收集
            if valid_fas.sum() > 0:
                stats['all_p_fas_logits'].extend(pred_fas[valid_fas].cpu().tolist())
                stats['all_t_fas'].extend(fas_labels[valid_fas].int().cpu().tolist())

            # OCC Loss
            if valid_occ.sum() > 0:
                loss_occ = criterion_occ(pred_occ[valid_occ], occ_labels[valid_occ])
                stats['all_p_occ'].extend((pred_occ[valid_occ] > 0.5).int().cpu().tolist())
                stats['all_t_occ'].extend(occ_labels[valid_occ].int().cpu().tolist())
            else:
                loss_occ = torch.tensor(0.0, device=device)

            # ADV Loss
            loss_adv = torch.tensor(0.0, device=device)
            if is_train and enable_adv:
                if adv_fas is not None and valid_fas_for_loss.sum() > 0:
                    loss_adv_fas = criterion_fas(adv_fas[valid_fas_for_loss], fas_labels[valid_fas_for_loss])
                    stats['adv_fas_pred'].extend((adv_fas[valid_fas_for_loss] > 0.5).int().cpu().tolist())
                    stats['adv_fas_true'].extend(fas_labels[valid_fas_for_loss].int().cpu().tolist())
                else:
                    loss_adv_fas = torch.tensor(0.0, device=device)

                if adv_occ is not None and valid_occ.sum() > 0:
                    loss_adv_occ = criterion_occ(adv_occ[valid_occ], occ_labels[valid_occ])
                    stats['adv_occ_pred'].extend((adv_occ[valid_occ] > 0.5).int().cpu().tolist())
                    stats['adv_occ_true'].extend(occ_labels[valid_occ].int().cpu().tolist())
                else:
                    loss_adv_occ = torch.tensor(0.0, device=device)

                loss_adv = loss_adv_fas + loss_adv_occ

            # ID Metrics
            pred_id_labels = torch.max(pred_id.data, 1)[1]
            if valid_id.sum() > 0:
                stats['all_p_id'].extend(pred_id_labels[valid_id].cpu().tolist())
                stats['all_t_id'].extend(id_labels[valid_id].cpu().tolist())
                stats['correct_id'] += (pred_id_labels[valid_id] == id_labels[valid_id]).sum().item()
                stats['samples_id'] += valid_id.sum().item()

            # 验证集错误样本收集
            if phase == "val" and val_error_samples is not None:
                pred_fas_labels = (pred_fas > (best_fas_th if best_fas_th is not None else 0.5)).float()
                id_err = (pred_id_labels != id_labels) & valid_id
                fas_err = (pred_fas_labels != fas_labels) & valid_fas

                incorrect_mask = id_err | fas_err
                incorrect_indices = torch.where(incorrect_mask)[0]

                if len(incorrect_indices) > 0:
                    if hasattr(dataloader.dataset, 'dataset'):
                        if isinstance(dataloader.dataset.dataset, UnconstrainedEnvWrapper):
                            base_dataset = dataloader.dataset.dataset.dataset
                        else:
                            base_dataset = dataloader.dataset.dataset
                    else:
                        base_dataset = dataloader.dataset

                    start_idx = batch_idx * cfg.BATCH_SIZE

                    for idx_in_batch in incorrect_indices:
                        abs_idx = start_idx + idx_in_batch.item()

                        if abs_idx < len(base_dataset):
                            try:
                                item = base_dataset.data_list[abs_idx] if hasattr(base_dataset, 'data_list') else {}
                                img_path = item.get('path', 'unknown')

                                t_id = id_labels[idx_in_batch].item()
                                p_id = pred_id_labels[idx_in_batch].item()
                                t_fas = fas_labels[idx_in_batch].item()
                                p_fas = pred_fas_labels[idx_in_batch].item()

                                is_id_err = id_err[idx_in_batch].item()
                                is_fas_err = fas_err[idx_in_batch].item()

                                if is_id_err and is_fas_err:
                                    err_type = "ID_and_FAS_Error"
                                elif is_id_err:
                                    err_type = "ID_Error"
                                else:
                                    err_type = "FAS_Error"

                                person_name = 'unknown'
                                if hasattr(base_dataset, 'id_to_idx'):
                                    for name, idx_val in base_dataset.id_to_idx.items():
                                        if idx_val == t_id:
                                            person_name = name
                                            break

                                val_error_samples.append({
                                    'index': abs_idx,
                                    'image_path': img_path,
                                    'error_type': err_type,
                                    'true_id': t_id,
                                    'pred_id': p_id,
                                    'true_fas': t_fas,
                                    'pred_fas': p_fas,
                                    'person_name': person_name
                                })
                            except Exception:
                                pass

            # IDE Metrics
            if valid_ide.sum() > 0:
                stacked_logits = torch.stack([
                    torch.zeros_like(logits_blur[valid_ide]),
                    logits_glare[valid_ide],
                    logits_blur[valid_ide],
                    logits_dark[valid_ide]
                ], dim=1)
                pred_ide_labels = torch.argmax(stacked_logits, dim=1)
                stats['all_p_ide'].extend(pred_ide_labels.cpu().tolist())
                stats['all_t_ide'].extend(ide_labels[valid_ide].cpu().tolist())
                stats['correct_ide'] += (pred_ide_labels == ide_labels[valid_ide]).sum().item()
                stats['samples_ide'] += valid_ide.sum().item()

            # Total Loss
            loss = (
                    loss_id * loss_weights["id"] +
                    loss_fas * loss_weights["fas"] +
                    loss_occ * loss_weights["occ"] +
                    loss_ide * loss_weights["ide"] +
                    loss_adv * loss_weights["adv"]
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if len(trainable_params) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                optimizer.step()

        stats['loss_total'] += loss.item()
        stats['loss_id'] += loss_id.item()
        stats['loss_fas'] += loss_fas.item()
        stats['loss_occ'] += loss_occ.item()
        stats['loss_ide'] += loss_ide.item()
        stats['loss_adv'] += loss_adv.item()

        current_id_acc = stats['correct_id'] / (stats['samples_id'] + 1e-8) * 100.0
        loop.set_postfix(
            ID_Acc=f"{current_id_acc:.2f}%",
            Loss=f"{loss.item():.4f}",
            Stage=get_stage_name(epoch) if is_train else phase
        )

    # 验证阶段：搜索最佳 FAS 阈值
    if phase == "val" and stats['all_t_fas']:
        best_th, best_fas_metrics = find_best_fas_threshold(
            np.array(stats['all_t_fas']),
            np.array(stats['all_p_fas_logits'])
        )
        # 覆盖 metrics 中 FAS 相关字段
        metrics = summarize_metrics(stats, len(dataloader), phase, best_fas_th=best_th)
        metrics.update({
            'best_fas_threshold': best_th,
            'fas_best_apcer': best_fas_metrics['apcer'],
            'fas_best_bpcer': best_fas_metrics['bpcer'],
            'fas_best_acer': best_fas_metrics['acer'],
            'fas_best_f1': best_fas_metrics['f1'],
            'fas_best_precision': best_fas_metrics['precision'],
            'fas_best_recall': best_fas_metrics['recall']
        })
        return metrics
    else:
        return summarize_metrics(stats, len(dataloader), phase, best_fas_th=best_fas_th)


# =========================================================
# Dataloader：三阶段构建
# =========================================================
# =========================================================
# Dataloader：三阶段构建【仅私有数据集，不加载FAS/RMFRD公开集】
# =========================================================
def build_dataloaders():
    print("\n" + "=" * 80)
    print("📦 开始加载数据集（仅私有数据集，不加载FAS/RMFRD公开数据集）")
    print("=" * 80)
    main_train_set = AirportFaceDataset(
        cfg.DATA_ROOT,
        split='train',
        transform=get_base_transforms(),
        id_to_idx_map=None,
        allow_new_ids=True
    )
    # ========== 注释掉公开数据集FAS、RMFRD，暂时不使用 ==========
    # rmfrd_path = r"E:\py\CVproject\CV\dataset\Face-Mask-12K-Images-Dataset\Face Mask Dataset"
    # rmfrd_set = RMFRDDataset(
    #     root_dir=rmfrd_path,
    #     existing_id_map=main_train_set.id_to_idx,
    #     transform=get_base_transforms(),
    #     split='Train'
    # )
    # fas_path = r"E:\py\CVproject\CV\dataset\Real-and-Fake-Face-Detection-Dataset\real_and_fake_face"
    # fas_set = PureFASDataset(
    #     root_dir=fas_path,
    #     existing_id_map=main_train_set.id_to_idx,
    #     transform=get_base_transforms()
    # )

    # Stage1: 仅私有数据
    train_set_stage1 = main_train_set
    # Stage2、Stage3 暂时同样使用私有数据集（原代码结构不动，后续想用公开集再打开注释）
    train_set_stage2 = main_train_set
    train_set_stage3 = main_train_set

    env_train_set_stage1 = UnconstrainedEnvWrapper(train_set_stage1, is_train=True)
    env_train_set_stage2 = UnconstrainedEnvWrapper(train_set_stage2, is_train=True)
    env_train_set_stage3 = UnconstrainedEnvWrapper(train_set_stage3, is_train=True)

    val_set = AirportFaceDataset(
        cfg.DATA_ROOT,
        split='val',
        transform=get_test_base_transforms(),
        id_to_idx_map=main_train_set.id_to_idx,
        allow_new_ids=False
    )
    env_val_set = UnconstrainedEnvWrapper(val_set, is_train=False)

    test_set = AirportFaceDataset(
        cfg.DATA_ROOT,
        split='test',
        transform=get_test_base_transforms(),
        id_to_idx_map=main_train_set.id_to_idx,
        allow_new_ids=False
    )
    env_test_set = UnconstrainedEnvWrapper(test_set, is_train=False)

    train_loader_stage1 = DataLoader(env_train_set_stage1, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                     num_workers=cfg.NUM_WORKERS,
                                     pin_memory=True if cfg.DEVICE.type == "cuda" else False)
    train_loader_stage2 = DataLoader(env_train_set_stage2, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                     num_workers=cfg.NUM_WORKERS,
                                     pin_memory=True if cfg.DEVICE.type == "cuda" else False)
    train_loader_stage3 = DataLoader(env_train_set_stage3, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                     num_workers=cfg.NUM_WORKERS,
                                     pin_memory=True if cfg.DEVICE.type == "cuda" else False)
    val_loader = DataLoader(env_val_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.NUM_WORKERS, pin_memory=True if cfg.DEVICE.type == "cuda" else False)
    test_loader = DataLoader(env_test_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True if cfg.DEVICE.type == "cuda" else False)
    print("\n" + "-" * 80)
    print(f"Stage1 (私有数据): {len(env_train_set_stage1)}")
    print(f"Stage2 (私有数据): {len(env_train_set_stage2)}")
    print(f"Stage3 (私有数据): {len(env_train_set_stage3)}")
    print(f"验证集: {len(env_val_set)} | 测试集: {len(env_test_set)} | 身份数: {len(main_train_set.id_to_idx)}")
    print("-" * 80 + "\n")
    return train_loader_stage1, train_loader_stage2, train_loader_stage3, val_loader, test_loader, main_train_set.id_to_idx



# =========================================================
# Print
# =========================================================
def print_detailed_evaluation(metrics, epoch=None, title="Evaluation Report"):
    if epoch is not None:
        head = f"📊 Epoch {epoch} {title}"
    else:
        head = f"📊 {title}"

    print(f"\n{'=' * 80}")
    print(head)
    print(f"{'=' * 80}\n")

    print(f"[👤 身份识别 ID Recognition]")
    print(f"   • Accuracy : {metrics['acc_id']:.2f}%")
    print(f"   • Precision: {metrics['id_precision']:.2f}%")
    print(f"   • Recall   : {metrics['id_recall']:.2f}%")
    print(f"   • F1 Score : {metrics['id_f1']:.2f}%")
    print(f"   • Samples  : {metrics['samples_id']}")

    print(f"\n[🛡️ 活体防伪 FAS Anti-Spoofing]")
    print(f"   • Best Threshold : {metrics.get('best_fas_threshold', 0.5):.3f}")
    print(f"   • F1 Score       : {metrics['f1_fas']:.2f}%")
    print(f"   • Precision      : {metrics['fas_precision']:.2f}%")
    print(f"   • Recall         : {metrics['fas_recall']:.2f}%")
    print(f"   • APCER          : {metrics['apcer_fas']:.2f}%")
    print(f"   • BPCER          : {metrics['bpcer_fas']:.2f}%")
    print(f"   • ACER           : {metrics['acer_fas']:.2f}%")
    print(
        f"   • Best ACER      : {metrics.get('fas_best_acer', 0.0):.2f}% @ th={metrics.get('best_fas_threshold', 0.5):.3f}")
    print(f"   • Samples        : {metrics['samples_fas']}")

    print(f"\n[😷 遮挡检测 OCC Detection]")
    print(f"   • F1 Score : {metrics['f1_occ']:.2f}%")
    print(f"   • Precision: {metrics['occ_precision']:.2f}%")
    print(f"   • Recall   : {metrics['occ_recall']:.2f}%")
    print(f"   • Samples  : {metrics['samples_occ']}")

    print(f"\n[☁️ 环境感知 IDE Robustness]")
    print(f"   • Accuracy : {metrics['acc_ide']:.2f}%")
    print(f"   • Precision: {metrics['ide_precision']:.2f}%")
    print(f"   • Recall   : {metrics['ide_recall']:.2f}%")
    print(f"   • F1 Score : {metrics['ide_f1']:.2f}%")
    print(f"   • Samples  : {metrics['samples_ide']}")

    print(f"\n[✨ 综合评分 Composite Score]")
    print(f"   • Score: {metrics['composite_score']:.2f}/100.00")

    print(f"\n{'=' * 80}\n")


# =========================================================
# Train
# =========================================================
def train_and_verify():
    logger = PaperLogger()
    train_loader_stage1, train_loader_stage2, train_loader_stage3, val_loader, test_loader, id_map = build_dataloaders()

    total_classes = len(id_map)
    if total_classes <= 0:
        raise RuntimeError("身份类别数为 0，请检查 train 目录。")

    model = CD_MAGNet(
        num_classes=total_classes,
        embedding_dim=cfg.EMBEDDING_DIM,
        adv_alpha=0.0  # 默认禁用 ADV
    ).to(cfg.DEVICE)

    # 使用分层学习率：backbone 更小 lr
    backbone_params = [p for name, p in model.named_parameters() if 'stem' in name or 'layer' in name]
    head_params = [p for name, p in model.named_parameters() if 'stem' not in name and 'layer' not in name]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg.LR * 0.1},
        {'params': head_params, 'lr': cfg.LR}
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    criterion_fas = FocalLoss(alpha=0.35, gamma=2.0)
    criterion_occ = FocalLoss(alpha=0.75, gamma=2.0)
    criterions = (
        nn.CrossEntropyLoss(),
        criterion_fas,
        criterion_occ,
        nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)
    )

    best_val_score = -1.0
    best_epoch = -1
    val_error_samples = []

    print("\n" + "=" * 80)
    print("🚀 开始训练：CD_MAGNet v2.1 —— 三阶段增强训练 (60 Epochs)")
    print("=" * 80)
    print(f"Stage1 (Epoch 1~{cfg.WARMUP_EPOCHS}): 仅私有数据 → ID + OCC + IDE，禁用 FAS/ADV")
    print(
        f"Stage2 (Epoch {cfg.WARMUP_EPOCHS + 1}~{cfg.WARMUP_EPOCHS + 20}): 私有 + FAS → ID + FAS + OCC + IDE，仍禁用 ADV")
    print(f"Stage3 (Epoch {cfg.WARMUP_EPOCHS + 21}~{cfg.EPOCHS}): 加入少量 RMFRD → 全任务微调")
    print("=" * 80 + "\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        stage_name = get_stage_name(epoch)
        loss_weights = get_loss_weights(epoch)

        print(f"\n{'=' * 80}")
        print(f"🚀 Epoch {epoch}/{cfg.EPOCHS} | Current LR: {current_lr:.6e}")
        print(f"📌 Current Stage: {stage_name}")
        print(
            f"⚖️ Loss Weights | ID={loss_weights['id']:.2f}, FAS={loss_weights['fas']:.2f}, OCC={loss_weights['occ']:.2f}, IDE={loss_weights['ide']:.2f}, ADV={loss_weights['adv']:.2f}")
        print(f"{'=' * 80}\n")

        # 选择 dataloader
        if epoch <= cfg.WARMUP_EPOCHS:
            train_loader = train_loader_stage1
            set_stage1_trainable(model)
            enable_fas = False
            enable_adv = False
        elif epoch <= cfg.WARMUP_EPOCHS + 20:
            train_loader = train_loader_stage2
            set_stage2_trainable(model)
            enable_fas = True
            enable_adv = False
        else:
            train_loader = train_loader_stage3
            set_all_trainable(model)
            enable_fas = True
            enable_adv = False

        train_metrics = run_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterions=criterions,
            device=cfg.DEVICE,
            epoch=epoch,
            phase="train",
            loss_weights=loss_weights,
            enable_fas=enable_fas,
            enable_adv=enable_adv
        )
        print_detailed_evaluation(train_metrics, epoch, "训练集评估")
        logger.update_train(epoch, train_metrics)

        # 验证（自动搜索 FAS 阈值）
        set_all_trainable(model)
        val_metrics = run_one_epoch(
            model=model,
            dataloader=val_loader,
            optimizer=None,
            criterions=criterions,
            device=cfg.DEVICE,
            epoch=epoch,
            phase="val",
            val_error_samples=val_error_samples,
            loss_weights={
                "id": 10.0, "fas": 2.0, "occ": 2.0, "ide": 1.0, "adv": 0.0
            },
            enable_fas=True,
            enable_adv=False
        )
        print_detailed_evaluation(val_metrics, epoch, "验证集评估")
        logger.update_val(epoch, val_metrics)
        logger.draw_curves()

        scheduler.step()

        if len(val_error_samples) > 0:
            error_df = pd.DataFrame(val_error_samples)
            error_file_path = os.path.join("./logs", f"val_error_samples_epoch_{epoch}.csv")
            error_df.to_csv(error_file_path, index=False, encoding="utf-8-sig")
            print(f"   [!] 保存 {len(val_error_samples)} 个错误样本到: {error_file_path}")

        current_val_score = val_metrics['composite_score']
        if current_val_score > best_val_score:
            best_val_score = current_val_score
            best_epoch = epoch
            torch.save({
                'state_dict': model.state_dict(),
                'id_map': id_map,
                'best_fas_threshold': val_metrics.get('best_fas_threshold', 0.5),
                'val_composite_score': best_val_score,
                'epoch': best_epoch,
                'config': {
                    'input_size': cfg.INPUT_SIZE,
                    'embedding_dim': cfg.EMBEDDING_DIM,
                    'num_classes': total_classes
                }
            }, cfg.BEST_MODEL_PATH)
            print(
                f"   [+] 🎯 保存最优模型至 {cfg.BEST_MODEL_PATH} | Epoch: {best_epoch} | Score: {best_val_score:.2f}"
            )
        else:
            print(f"   [-] 当前 Score {current_val_score:.2f} ≤ 最佳 {best_val_score:.2f}")

        print()

    # 测试
    print("\n" + "=" * 80)
    print("🏁 加载最优模型，在测试集上评估")
    print("=" * 80 + "\n")
    checkpoint = torch.load(cfg.BEST_MODEL_PATH, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    set_all_trainable(model)

    test_metrics = run_one_epoch(
        model=model,
        dataloader=test_loader,
        optimizer=None,
        criterions=criterions,
        device=cfg.DEVICE,
        epoch=best_epoch,
        phase="test",
        loss_weights={"id": 10.0, "fas": 2.0, "occ": 2.0, "ide": 1.0, "adv": 0.0},
        enable_fas=True,
        enable_adv=False,
        best_fas_th=checkpoint.get('best_fas_threshold', 0.5)
    )
    print_detailed_evaluation(
        test_metrics,
        title=f"最终测试报告 | Best Val Epoch: {best_epoch} | Score: {best_val_score:.2f}"
    )
    logger.update_test(test_metrics)

    print(f"✅ 测试结果已保存至 ./logs/final_test_report.csv")
    print(f"✅ 最优模型: {cfg.BEST_MODEL_PATH}")


if __name__ == '__main__':
    train_and_verify()