# test.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# [!] 增加了完整的指标库导入
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import os


from config import cfg
from dataset import AirportFaceDataset, RMFRDDataset, get_test_base_transforms, UnconstrainedEnvWrapper
from models import CD_MAGNet

import numpy
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])


def denormalize_for_vision(tensor):
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1).permute(1, 2, 0).numpy()


def generate_paper_report(t_id, p_id, t_fas, p_fas, t_occ, p_occ, t_ide, p_ide, v_id_mask):
    print("\n" + "=" * 70)
    print(" 📑 机场安检级学术评估报告 (Airport-Grade Paper Evaluation Report)")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. 核心任务性能 (ID)
    # ---------------------------------------------------------
    print("\n[🎯 任务 1: 身份识别 (ID Recognition) - 核心认证通道]")
    t_id_valid = np.array(t_id)[v_id_mask]
    p_id_valid = np.array(p_id)[v_id_mask]

    if len(t_id_valid) > 0:
        overall_id_acc = accuracy_score(t_id_valid, p_id_valid)
        print(f" -> 整体准确率 (Overall Accuracy): {overall_id_acc * 100:.2f}%")
        print(
            f" -> 精确率 (Precision)       : {precision_score(t_id_valid, p_id_valid, average='weighted', zero_division=0) * 100:.2f}%")
        print(
            f" -> 召回率 (Recall)          : {recall_score(t_id_valid, p_id_valid, average='weighted', zero_division=0) * 100:.2f}%")
        print(
            f" -> F1 分数 (F1 Score)       : {f1_score(t_id_valid, p_id_valid, average='weighted', zero_division=0) * 100:.2f}%")
    else:
        print(" -> 本次测试未提供有效 ID 标签样本。")

    # ---------------------------------------------------------
    # 2. FAS 防伪 (安全第一道防线)
    # ---------------------------------------------------------
    if len(t_fas) > 0:
        print("\n[🛡️ 任务 2: 活体防伪 (FAS Anti-Spoofing) - 安全生死线]")
        print("--- 详细防伪分类报告 ---")
        print(classification_report(t_fas, p_fas, target_names=['Fake(0)', 'Real(1)'], digits=4, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(t_fas, p_fas, labels=[0, 1]).ravel()
        apcer = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
        bpcer = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
        acer = (apcer + bpcer) / 2

        print("--- 安防核心验证指标 ---")
        print(f" -> APCER (攻击被误作真人的概率): {apcer:.2f}%  <- 安全漏洞率")
        print(f" -> BPCER (真人被误作攻击的概率): {bpcer:.2f}%  <- 旅客误拒率")
        print(f" -> ACER  (平均错误率, 越低越好): {acer:.2f}%")
        print(f" -> 整体准确率 (Accuracy)       : {accuracy_score(t_fas, p_fas) * 100:.2f}%")
        print(f" -> 活体精确率 (Precision)      : {precision_score(t_fas, p_fas, zero_division=0) * 100:.2f}%")
        print(f" -> 活体召回率 (Recall)         : {recall_score(t_fas, p_fas, zero_division=0) * 100:.2f}%")
        print(f" -> 活体 F1 分数 (F1 Score)     : {f1_score(t_fas, p_fas, zero_division=0) * 100:.2f}%")

        # 绘制防伪混淆矩阵
        plt.figure(figsize=(6, 5))
        sns.heatmap(np.array([[tn, fp], [fn, tp]]), annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Fake', 'Pred Real'], yticklabels=['True Fake', 'True Real'])
        plt.title('FAS Confusion Matrix (Airport Security)')
        plt.savefig(os.path.join(cfg.VIS_DIR, "fas_confusion_matrix.png"))
        plt.close()

    # ---------------------------------------------------------
    # 3. OCC 遮挡
    # ---------------------------------------------------------
    if len(t_occ) > 0:
        print("\n[😷 任务 3: 异物遮挡 (Mask/OCC Detection) - 防御辅助通道]")
        print("--- 详细遮挡分类报告 ---")
        print(classification_report(t_occ, p_occ, target_names=['No Mask(0)', 'Masked(1)'], digits=4, zero_division=0))
        print("--- 遮挡核心验证指标 ---")
        print(f" -> 整体准确率 (Accuracy)       : {accuracy_score(t_occ, p_occ) * 100:.2f}%")
        print(f" -> 遮挡精确率 (Precision)      : {precision_score(t_occ, p_occ, zero_division=0) * 100:.2f}%")
        print(f" -> 遮挡召回率 (Recall)         : {recall_score(t_occ, p_occ, zero_division=0) * 100:.2f}%")
        print(f" -> 遮挡 F1 分数 (F1 Score)     : {f1_score(t_occ, p_occ, zero_division=0) * 100:.2f}%")

    # ---------------------------------------------------------
    # 4. IDE 环境
    # ---------------------------------------------------------
    print("\n[☁️ 任务 4: 环境感知分析 (EAFM Effectiveness Proof) - 拓扑自适应]")
    print("--- 环境感知分类报告 ---")
    print(classification_report(t_ide, p_ide, target_names=["Normal", "Glare", "Blur", "Dark"], digits=4,
                                zero_division=0))
    print(f" -> 整体环境感知准确率 (Accuracy): {accuracy_score(t_ide, p_ide) * 100:.2f}%")

    ENV_MAP = {0: "Normal(正常)", 1: "Light-Glared(强光)", 2: "Blurred(模糊)", 3: "Darkness(低照度)"}
    ide_accs = []
    env_names = []

    print("\n--- 极限环境 ID 解码鲁棒性 (Robustness) ---")
    for ide_class in range(4):
        # 筛选出当前环境标签下，并且 ID 标签有效（!=-1）的样本进行条件测试
        condition = (np.array(t_ide) == ide_class) & v_id_mask
        if np.sum(condition) > 0:
            env_acc = accuracy_score(np.array(t_id)[condition], np.array(p_id)[condition]) * 100
            print(f" -> 在 {ENV_MAP[ide_class]} 环境下，身份解析准确率: {env_acc:.2f}%")
            ide_accs.append(env_acc)
            env_names.append(ENV_MAP[ide_class].split('(')[0])

    # 绘制抗干扰柱状图 (证明模型无惧光照和模糊)
    if len(ide_accs) > 0:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(env_names, ide_accs, color=['#4CAF50', '#FF9800', '#2196F3', '#9C27B0'])
        plt.ylim(0, 105)
        plt.ylabel('ID Recognition Accuracy (%)')
        plt.title('Robustness across Unconstrained Environments (EAFM)')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom',
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.VIS_DIR, "robustness_bar_chart.png"), dpi=200)
        plt.close()

    print(f"\n✅ 细粒度图表已自动生成，保存在: {cfg.VIS_DIR}/ 文件夹下！")
    print("=" * 70 + "\n")


def final_test():
    if not os.path.exists(cfg.BEST_MODEL_PATH):
        raise FileNotFoundError(f"找不到权重文件 {cfg.BEST_MODEL_PATH}，请先运行 train.py 进行训练。")

    checkpoint = torch.load(
        cfg.BEST_MODEL_PATH,
        map_location=cfg.DEVICE,
        weights_only=False  # 核心：显式关闭新版的纯权重检查
    )
    idx_to_id = {v: k for k, v in checkpoint['id_map'].items()}

    # 挂载数据集
    test_main = AirportFaceDataset(cfg.DATA_ROOT, split='test', transform=get_test_base_transforms(),
                                   id_to_idx_map=checkpoint['id_map'])
    rmfrd_path = r"E:\py\CVproject\CV\dataset\Face-Mask-12K-Images-Dataset\Face Mask Dataset"
    test_mask = RMFRDDataset(root_dir=rmfrd_path, existing_id_map=checkpoint['id_map'],
                             transform=get_test_base_transforms(), split='Test')

    # 在测试集中开启环境污染 is_train=True
    env_test_set = UnconstrainedEnvWrapper(ConcatDataset([test_main, test_mask]), is_train=True)
    test_loader = DataLoader(env_test_set, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # 实例化模型
    model = CD_MAGNet(num_classes=len(idx_to_id), embedding_dim=cfg.EMBEDDING_DIM).to(cfg.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 结果收集桶
    all_t_id, all_p_id = [], []
    all_t_fas, all_p_fas = [], []
    all_t_occ, all_p_occ = [], []
    all_t_ide, all_p_ide = [], []

    has_plotted = False

    with torch.no_grad():
        for imgs, lbl_id, lbl_fas, lbl_occ, lbl_ide in tqdm(test_loader, desc="[执行全链路深度考核]"):
            imgs = imgs.to(cfg.DEVICE)
            p_id, p_fas, p_occ, p_ide = model(imgs)

            pr_id = torch.max(p_id.data, 1)[1].cpu().numpy()
            pr_fas = (p_fas.view(-1) > 0.5).int().cpu().numpy()
            pr_occ = (p_occ.view(-1) > 0.5).int().cpu().numpy()
            pr_ide = torch.max(p_ide.data, 1)[1].cpu().numpy()

            all_t_id.extend(lbl_id.numpy())
            all_p_id.extend(pr_id)

            all_t_ide.extend(lbl_ide.numpy())
            all_p_ide.extend(pr_ide)

            v_fas_mask = lbl_fas.numpy() != float(cfg.IGNORE_INDEX)
            all_t_fas.extend(lbl_fas.numpy()[v_fas_mask].astype(int))
            all_p_fas.extend(pr_fas[v_fas_mask])

            v_occ_mask = lbl_occ.numpy() != float(cfg.IGNORE_INDEX)
            all_t_occ.extend(lbl_occ.numpy()[v_occ_mask].astype(int))
            all_p_occ.extend(pr_occ[v_occ_mask])

            # 将表现最好/最差的一批预测结果作图保存
            if not has_plotted:
                plt.figure(figsize=(16, int(np.ceil(imgs.size(0) / 4)) * 4))
                ENV_MAP = {0: "Clear", 1: "Glare", 2: "Blur", 3: "Dark"}
                for i in range(min(12, imgs.size(0))):
                    plt.subplot(3, 4, i + 1)
                    plt.imshow(denormalize_for_vision(imgs[i].cpu()))
                    plt.axis('off')
                    gt_id_name = idx_to_id.get(lbl_id[i].item(), 'Unk') if lbl_id[i].item() != -1 else "N/A"
                    pr_id_name = idx_to_id.get(pr_id[i], 'Unk')
                    color = 'green' if (pr_id_name == gt_id_name or gt_id_name == 'N/A') else 'red'
                    plt.title(f"ID: {pr_id_name} (T:{gt_id_name})\nEnv: {ENV_MAP[pr_ide[i]]}", fontsize=10, loc='left',
                              color=color)
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.VIS_DIR, "test_predictions_visual.png"))
                plt.close()
                has_plotted = True

    # 有效 ID 掩码
    v_id_mask = np.array(all_t_id) != cfg.IGNORE_INDEX

    # 打印超级学术审查报告
    generate_paper_report(all_t_id, all_p_id, all_t_fas, all_p_fas, all_t_occ, all_p_occ, all_t_ide, all_p_ide,
                          v_id_mask)


if __name__ == '__main__':
    final_test()