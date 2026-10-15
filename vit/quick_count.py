# quick_count.py 不初始化任何模型，直接读取权重文件统计参数量
import torch

# DINOv2‑vits14主干网络固定参数量（官方公开）
BACKBONE_PARAMS = 21623808


def analyze_ckpt(ckpt_path):
    print(f"\n===== 读取 {ckpt_path} =====")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    backbone_sum = 0
    head_sum = 0
    for key, tensor in sd.items():
        num = tensor.numel()
        if key.startswith("backbone."):
            backbone_sum += num
        else:
            head_sum += num

    total = backbone_sum + head_sum
    print(f"Backbone参数量: {backbone_sum:,}")
    print(f"任务头(heads)参数量: {head_sum:,}")
    print(f"模型总参数量 Total: {total:,}")
    return backbone_sum, head_sum, total


if __name__ == "__main__":
    # 修改成你真实的两个权重文件名
    file1 = "./exp_a_stage1_id_best.pth"
    file2 = "./exp_a_stage2_id_occ_fas_best.pth"

    analyze_ckpt(file1)
    analyze_ckpt(file2)

    print("\n---说明---")
    print("1. backbone主干全程冻结训练，不计入可训练参数；")
    print("2. Stage1可训练参数：仅ID分支部分头参数；")
    print("3. Stage2可训练参数：ID + OCC + Fake全部任务头参数。")
