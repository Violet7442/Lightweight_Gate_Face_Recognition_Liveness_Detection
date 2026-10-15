import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from config import cfg

# ===================== 手写ResNet50，不依赖torchvision =====================
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width))
        return nn.Sequential(*layers)

def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        # 剔除fc层权重
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        # strict=False 允许缺少key
        model.load_state_dict(state_dict, strict=False)
    return model

# ===================== 下面是你原来CD_MAGNet代码，完全不变 =====================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class TaskAwareAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_ca = x * self.ca(x)
        max_p, _ = torch.max(x_ca, dim=1, keepdim=True)
        avg_p = torch.mean(x_ca, dim=1, keepdim=True)
        x_sa = x_ca * self.sa(torch.cat([max_p, avg_p], dim=1))
        return x + x_sa

class ArcFaceLinear(nn.Module):
    def __init__(self, in_features, out_features, s=32.0, m=0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        self.th, self.mm = math.cos(math.pi - m), math.sin(math.pi - m) * m
    def forward(self, input, label=None):
        cos = F.linear(F.normalize(input), F.normalize(self.weight))
        cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
        if label is None:
            return cos * self.s
        sin = torch.sqrt(1.0 - torch.pow(cos, 2))
        phi = torch.where(cos > self.th, cos * math.cos(self.m) - sin * math.sin(self.m), cos - self.mm)
        one_hot = torch.zeros(cos.size(), device=input.device)
        valid_mask = (label != cfg.IGNORE_INDEX)
        valid_labels = label[valid_mask].long()
        if valid_labels.size(0) > 0:
            batch_indices = valid_mask.nonzero(as_tuple=True)[0]
            one_hot[batch_indices, valid_labels] = 1
        return ((one_hot * phi) + ((1.0 - one_hot) * cos)) * self.s

class DegradationPerceptionModule(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.local_net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.global_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        self.head_blur = nn.Linear(256, 1)
        self.head_glare = nn.Linear(256, 1)
        self.head_dark = nn.Linear(256, 1)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels + 256, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        M_deg = self.local_net(x)
        M_clean = 1.0 - M_deg
        feat_local = x * M_clean
        v_deg = self.global_net(x)
        logits_blur = self.head_blur(v_deg)
        logits_glare = self.head_glare(v_deg)
        logits_dark = self.head_dark(v_deg)
        v_deg_expand = v_deg.view(B, 256, 1, 1).expand(-1, -1, H, W)
        fused = torch.cat([feat_local, v_deg_expand], dim=1)
        x_injected = x + self.adapter(fused)
        return x_injected, (logits_blur, logits_glare, logits_dark)

class CD_MAGNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, adv_alpha=0.1):
        super().__init__()
        # 这里调用手写resnet50，pretrained=True加载imagenet预训练权重
        base = resnet50(pretrained=True)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.dpm = DegradationPerceptionModule(in_channels=512)
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.att_id = TaskAwareAttention(2048)
        self.att_fas = TaskAwareAttention(2048)
        self.att_occ = TaskAwareAttention(2048)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout_id = nn.Dropout(p=0.3)
        self.dropout_fas = nn.Dropout(p=0.3)
        self.dropout_occ = nn.Dropout(p=0.5)
        self.fc_id = nn.Linear(2048, embedding_dim)
        self.fc_fas = nn.Linear(2048, embedding_dim)
        self.fc_occ = nn.Linear(2048, embedding_dim)
        self.head_id = ArcFaceLinear(embedding_dim, num_classes, s=32.0, m=0.35)
        self.head_fas = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.head_occ = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.grl = GRL(alpha=adv_alpha)
        self.adv_h_fas = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv_h_occ = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, label_id=None):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_robust, (logits_blur, logits_glare, logits_dark) = self.dpm(x)
        x = self.layer3(x_robust)
        s = self.layer4(x)
        flat_id = self.dropout_id(self.flatten(self.gap(self.att_id(s))))
        flat_fas = self.dropout_fas(self.flatten(self.gap(self.att_fas(s))))
        flat_occ = self.dropout_occ(self.flatten(self.gap(self.att_occ(s))))
        v_id = F.normalize(self.fc_id(flat_id), dim=1)
        v_fas = self.fc_fas(flat_fas)
        v_occ = self.fc_occ(flat_occ)
        out_id = self.head_id(v_id, label=label_id)
        out_fas = torch.sigmoid(self.head_fas(v_fas))
        out_occ = torch.sigmoid(self.head_occ(v_occ))
        if self.training and label_id is not None:
            id_rev = self.grl(v_id)
            return {
                "out_id": out_id,
                "out_fas": out_fas,
                "out_occ": out_occ,
                "logits_blur": logits_blur,
                "logits_glare": logits_glare,
                "logits_dark": logits_dark,
                "adv_fas": torch.sigmoid(self.adv_h_fas(id_rev)),
                "adv_occ": torch.sigmoid(self.adv_h_occ(id_rev))
            }
        return out_id, out_fas, out_occ, logits_blur, logits_glare, logits_dark
