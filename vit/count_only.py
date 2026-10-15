# count_only.py  只统计模型结构参数量，不读取任何pth权重
# -*- coding: utf-8 -*-
import torch
from models import CD_MAGNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}\n")

    # 真实训练类别：86
    model = CD_MAGNet(
        num_classes=86,
        embedding_dim=512,
        deg_embedding_dim=256,
        adv_alpha=0.1,
        arcface_s=32.0,
        arcface_m=0.35,
        backbone_name="dinov2_vits14",
        backbone_dim=384,
        backbone_input_size=224,
        use_dinov2_pretrained=True,
        use_fas_local_branch=True,
        fas_local_dim=256,
        use_fas_sobel=True,
        fas_sobel_alpha=0.35,
        fas_adv_alpha=0.05
    ).to(device)

    # --------------------------
    # 模拟 stage1 状态：冻结非id分支
    # --------------------------
    model.freeze_non_id_branches()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_s1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_s1 = total_params - trainable_params_s1

    print("==================== Stage1 状态( freeze_non_id_branches() ) ====================")
    print(f"🔢 全部参数量(total): {total_params:,}")
    print(f"🔢 可训练参数量(trainable): {trainable_params_s1:,}")
    print(f"📊 冻结参数(frozen): {frozen_s1:,}")

    # --------------------------
    # 模拟 stage2 状态：全部头放开可训练（backbone依旧冻结）
    # --------------------------
    for p in model.parameters():
        p.requires_grad = False
    for m in model.get_all_head_modules(include_dpm=True):
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True

    trainable_params_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_s2 = total_params - trainable_params_s2

    print("\n==================== Stage2 状态(全部任务头可训练) ====================")
    print(f"🔢 全部参数量(total): {total_params:,}")
    print(f"🔢 可训练参数量(trainable): {trainable_params_s2:,}")
    print(f"📊 冻结参数(frozen): {frozen_s2:,}")

    bb = model.backbone
    bb_total = sum(p.numel() for p in bb.parameters())
    bb_trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    print(f"\n💡 Backbone(DINOv2‑vits14) 总参数: {bb_total:,}")
    print(f"💡 Backbone(DINOv2‑vits14) 可训练参数: {bb_trainable:,}")

    print("\n----------子模块参数量明细----------")
    for name, submodule in model.named_children():
        cnt = sum(p.numel() for p in submodule.parameters())
        print(f"{name:<20} {cnt:>12,}")
