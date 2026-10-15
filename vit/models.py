# -*- coding: utf-8 -*-
# models.py

import math
import os
import sys

# 把dinov2仓库外层目录加入搜索路径
DINOV2_ROOT = "/root/autodl-tmp/CV/dinov2"
if DINOV2_ROOT not in sys.path:
    sys.path.insert(0, DINOV2_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input):
        if self.training:
            if input.dim() == 2:
                values_per_channel = input.size(0)
            elif input.dim() == 3:
                values_per_channel = input.size(0) * input.size(2)
            else:
                values_per_channel = input.numel() // input.size(1)

            if values_per_channel <= 1:
                return F.batch_norm(
                    input,
                    self.running_mean,
                    self.running_var,
                    self.weight,
                    self.bias,
                    training=False,
                    momentum=0.0,
                    eps=self.eps
                )

        return super().forward(input)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=32.0, m=0.35):
        super().__init__()

        self.s = s
        self.m = m
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        if labels is None:
            return cosine * self.s

        labels = labels.clone().long()
        invalid = (labels < 0) | (labels >= self.num_classes)

        if invalid.any():
            labels[invalid] = 0

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        if invalid.any():
            one_hot[invalid] = 0.0

        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.s

class DPM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.local_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        self.global_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

        self.head_blur = self._make_head(in_channels)
        self.head_glare = self._make_head(in_channels)
        self.head_dark = self._make_head(in_channels)

    @staticmethod
    def _make_head(in_channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x, compute_degradation=True):
        m_deg = self.local_net(x)
        g = self.global_net(x).unsqueeze(-1).unsqueeze(-1)
        x_robust = x + x * m_deg * g

        if not compute_degradation:
            return x_robust, (None, None, None)

        return x_robust, (
            self.head_blur(x_robust),
            self.head_glare(x_robust),
            self.head_dark(x_robust)
        )

class FASLocalTextureBranch(nn.Module):
    def __init__(self, in_channels, out_dim=256, use_sobel=True, sobel_alpha=0.35):
        super().__init__()

        self.use_sobel = use_sobel
        self.sobel_alpha = sobel_alpha

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
        )

        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1, groups=in_channels // 2, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 2, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def apply_sobel(self, x):
        _, c, _, _ = x.shape
        sobel_x = self.sobel_x.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=c)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def forward(self, x):
        if self.use_sobel:
            x = x + self.sobel_alpha * self.apply_sobel(x)

        return self.pool(self.local_conv(x)).flatten(1)

class CD_MAGNet(nn.Module):
    def __init__(
            self,
            num_classes,
            embedding_dim=512,
            deg_embedding_dim=256,
            adv_alpha=0.1,
            arcface_s=32.0,
            arcface_m=0.35,
            backbone_name="dinov2_vits14",
            backbone_dim=384,
            backbone_input_size=224,
            use_dinov2_pretrained=True,
            use_dpm=True,               # 新增
            use_fas_local_branch=True,
            fas_local_dim=256,
            use_fas_sobel=True,
            fas_sobel_alpha=0.35,
            fas_adv_alpha=0.05
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.deg_embedding_dim = deg_embedding_dim
        self.backbone_name = backbone_name
        self.backbone_dim = backbone_dim
        self.backbone_input_size = backbone_input_size
        self.use_fas_local_branch = use_fas_local_branch

        # self.backbone = self._load_dinov2(backbone_name, use_dinov2_pretrained)

        # self.dpm = DPM(backbone_dim)
        self.backbone = self._load_dinov2(backbone_name, use_dinov2_pretrained)

        self.use_dpm = use_dpm
        self.dpm = DPM(backbone_dim) if use_dpm else None

        self.id_fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(backbone_dim, embedding_dim),
            SafeBatchNorm1d(embedding_dim)
        )
        self.arcface = ArcFaceHead(embedding_dim, num_classes, s=arcface_s, m=arcface_m)

        self.deg_fc = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(backbone_dim, deg_embedding_dim),
            SafeBatchNorm1d(deg_embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.id_to_deg_dim = nn.Linear(embedding_dim, deg_embedding_dim)
        self.deg_emb_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(deg_embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
        self.adv_deg_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

        self.fas_global_proj = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(backbone_dim, 256),
            SafeBatchNorm1d(256),
            nn.GELU()
        )

        if use_fas_local_branch:
            self.fas_local_branch = FASLocalTextureBranch(
                in_channels=backbone_dim,
                out_dim=fas_local_dim,
                use_sobel=use_fas_sobel,
                sobel_alpha=fas_sobel_alpha
            )
            fas_in_dim = 256 + fas_local_dim
        else:
            self.fas_local_branch = None
            fas_in_dim = 256

        self.fas_head = nn.Sequential(
            nn.Dropout(0.45),
            nn.Linear(fas_in_dim, 256),
            SafeBatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            SafeBatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1)
        )

        self.occ_head = nn.Sequential(
            nn.Dropout(0.45),
            nn.Linear(backbone_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(256, 1)
        )

        self.grl = GRL(alpha=adv_alpha)
        self.grl_fas = GRL(alpha=fas_adv_alpha)

        self.adv_fas_head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(embedding_dim, 256),
            SafeBatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

        self.adv_occ_head = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def _load_dinov2(self, backbone_name, use_pretrained):
        print("\n" + "=" * 80)
        print("[Backbone] 离线加载 DINOv2-vits14（无 register）")
        print("=" * 80)
    
        # 你下载的 dinov2 完整源码目录（含 hubconf.py）
        dinov2_repo_root = os.path.join(os.path.dirname(cfg.PROJECT_ROOT), "dinov2")
        if dinov2_repo_root not in sys.path:
            sys.path.append(dinov2_repo_root)
    
        import hubconf
    
        builders = {
            "dinov2_vits14": hubconf.dinov2_vits14,
        }
        if backbone_name not in builders:
            raise ValueError(f"不支持的 backbone 名称: {backbone_name}")
    
        backbone = builders[backbone_name]()
    
        if use_pretrained:
            weight_path = os.path.join(cfg.PROJECT_ROOT, "dinov2_vits14_pretrain.pth")
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"本地权重文件缺失: {weight_path}")
    
            state = torch.load(weight_path, map_location=cfg.DEVICE, weights_only=True)
            backbone.load_state_dict(state)
            print(f"离线加载本地权重成功: {weight_path}")
    
        return backbone





        



    def freeze_non_id_branches(self):
        for module in [
            self.dpm,
            self.fas_global_proj,
            self.fas_head,
            self.fas_local_branch,
            self.occ_head,
            self.adv_fas_head,
            self.adv_occ_head,
            self.deg_fc,
            self.deg_emb_head,
            self.adv_deg_head,
        ]:
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = False

    def train_last_blocks_only(self, num_blocks=4):
        for p in self.parameters():
            p.requires_grad = False

        if hasattr(self.backbone, "blocks") and num_blocks > 0:
            total_blocks = len(self.backbone.blocks)
            start_idx = max(0, total_blocks - num_blocks)
            for i in range(start_idx, total_blocks):
                for p in self.backbone.blocks[i].parameters():
                    p.requires_grad = True
            print(f"解冻 backbone.blocks[{start_idx}:{total_blocks}]")

        for module in self.get_all_head_modules(include_dpm=True):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

    def get_all_head_modules(self, include_dpm=True):
        modules = [
            self.id_fc,
            self.arcface,
            self.deg_fc,
            self.id_to_deg_dim,
            self.deg_emb_head,
            self.adv_deg_head,
            self.fas_global_proj,
            self.fas_head,
            self.occ_head,
            self.adv_fas_head,
            self.adv_occ_head,
        ]

        if include_dpm:
            modules.append(self.dpm)

        if self.fas_local_branch is not None:
            modules.append(self.fas_local_branch)

        return modules

    def _dinov2_forward_features(self, x):
        if x.shape[-2:] != (self.backbone_input_size, self.backbone_input_size):
            x = F.interpolate(
                x,
                size=(self.backbone_input_size, self.backbone_input_size),
                mode="bilinear",
                align_corners=False
            )

        features = self.backbone.forward_features(x)

        if isinstance(features, dict):
            if "x_norm_clstoken" in features:
                cls_feat = features["x_norm_clstoken"]
            elif "x_prenorm" in features:
                cls_feat = features["x_prenorm"][:, 0]
            else:
                raise KeyError(f"DINOv2 未找到 cls token，可用 keys: {features.keys()}")

            if "x_norm_patchtokens" in features:
                patch_tokens = features["x_norm_patchtokens"]
            elif "x_prenorm" in features:
                patch_tokens = features["x_prenorm"][:, 1:]
            else:
                raise KeyError(f"DINOv2 未找到 patch tokens，可用 keys: {features.keys()}")
        else:
            if features.dim() == 3:
                cls_feat = features[:, 0]
                patch_tokens = features[:, 1:]
            elif features.dim() == 2:
                cls_feat = features
                patch_tokens = None
            else:
                raise RuntimeError(f"无法解析 DINOv2 输出维度: {features.shape}")

        if patch_tokens is None:
            raise RuntimeError("DINOv2 未返回 patch tokens，无法构造 FAS 局部分支。")

        b, n, c = patch_tokens.shape
        h = int(math.sqrt(n))

        if h * h != n:
            raise RuntimeError(f"patch token 数量无法还原为二维特征图: n={n}")

        spatial_2d = patch_tokens.transpose(1, 2).contiguous().view(b, c, h, h)
        return cls_feat, spatial_2d

    def extract_feat(self, x, id_only=False):
        cls_feat, spatial_2d = self._dinov2_forward_features(x)
    
        if id_only:
            return cls_feat, None, None, None, None
    
        # ========= 替换原来第461行 =========
        if self.use_dpm and self.dpm is not None:
            spatial_robust, dpm_logits = self.dpm(spatial_2d, compute_degradation=True)
        else:
            spatial_robust = spatial_2d
            dpm_logits = (None, None, None)
        # =================================
    
        fas_local_feat = None
    
        if self.use_fas_local_branch and self.fas_local_branch is not None:
            fas_local_feat = self.fas_local_branch(spatial_robust)
    
        robust_global = F.adaptive_avg_pool2d(spatial_robust, 1).flatten(1)
        feat = cls_feat + robust_global
    
        return feat, fas_local_feat, *dpm_logits


    def forward(self, x, label_id=None, id_only=False):
        feat, fas_local_feat, logits_blur, logits_glare, logits_dark = self.extract_feat(
            x,
            id_only=id_only
        )

        embedding = self.id_fc(feat)
        out_id = self.arcface(embedding, label_id)

        if id_only:
            return {
                "out_id": out_id,
                "embedding": embedding,
                "feat": feat
            }

        fas_global_feat = self.fas_global_proj(feat)

        if self.use_fas_local_branch and fas_local_feat is not None:
            fas_input = torch.cat([fas_global_feat, fas_local_feat], dim=1)
        else:
            fas_input = fas_global_feat

        out_fas = self.fas_head(fas_input)
        out_occ = self.occ_head(feat)

        deg_embedding = self.deg_fc(feat)
        id_for_orth = self.id_to_deg_dim(embedding)
        deg_emb_logits = self.deg_emb_head(deg_embedding)

        if self.training:
            adv_occ = self.adv_occ_head(self.grl(feat))
            adv_deg_logits = self.adv_deg_head(self.grl(embedding))
            adv_fas = self.adv_fas_head(self.grl_fas(embedding))
        else:
            adv_occ = self.adv_occ_head(feat)
            adv_deg_logits = self.adv_deg_head(embedding)
            adv_fas = self.adv_fas_head(embedding)

        return {
            "out_id": out_id,
            "out_fas": out_fas,
            "out_occ": out_occ,
            "logits_blur": logits_blur,
            "logits_glare": logits_glare,
            "logits_dark": logits_dark,
            "adv_fas": adv_fas,
            "adv_occ": adv_occ,
            "feat": feat,
            "embedding": embedding,
            "deg_embedding": deg_embedding,
            "id_for_orth": id_for_orth,
            "deg_emb_logits": deg_emb_logits,
            "adv_deg_logits": adv_deg_logits
        }