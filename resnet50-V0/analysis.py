# analysis.py
import os
import re
import glob
from collections import Counter

import pandas as pd

# =========================================================
# 配置区
# =========================================================
LOG_DIR = "./logs"
ERROR_FILE_PATTERN = "val_error_samples_epoch_*.csv"
OUTPUT_DIR = "./logs/simple_error_analysis"

PERSON_MIN_ERROR_EPOCH_RATIO = 0.5
IMAGE_MIN_ERROR_EPOCH_RATIO = 0.5

TOP_K_PERSONS = 50
TOP_K_IMAGES = 100


# =========================================================
# 工具函数
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv_safely(path):
    encodings = ["utf-8-sig", "utf-8", "gbk", "ansi"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"CSV 读取失败: {path}, 错误: {last_error}")


def extract_epoch(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r"epoch_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def normalize_path(path):
    if pd.isna(path):
        return ""
    return str(path).replace("\\", "/")


def parse_from_path(image_path):
    norm_path = normalize_path(image_path)
    parts = norm_path.split("/")

    person = "unknown"
    status = "unknown"
    filename = os.path.basename(norm_path)

    if "val" in parts:
        val_idx = parts.index("val")
        if len(parts) > val_idx + 1:
            person = parts[val_idx + 1]
        if len(parts) > val_idx + 2:
            status = parts[val_idx + 2]
    else:
        if len(parts) >= 3:
            person = parts[-3]
        if len(parts) >= 2:
            status = parts[-2]

    stem = os.path.splitext(filename)[0]
    lower_stem = stem.lower()

    env = "unknown"
    if lower_stem.endswith("_origin"):
        env = "origin"
    elif lower_stem.endswith("_blur"):
        env = "blur"
    elif lower_stem.endswith("_dark"):
        env = "dark"
    elif lower_stem.endswith("_glare"):
        env = "glare"

    return person, status, filename, env, norm_path


def counter_to_text(counter, top_k=10):
    if not counter:
        return ""
    items = counter.most_common(top_k)
    return "; ".join([f"{k}:{v}" for k, v in items])


# =========================================================
# 读取全部错误日志
# =========================================================
def load_all_errors():
    pattern = os.path.join(LOG_DIR, ERROR_FILE_PATTERN)
    files = glob.glob(pattern)
    files = sorted(files, key=lambda x: extract_epoch(x) if extract_epoch(x) is not None else 999999)

    if len(files) == 0:
        raise FileNotFoundError(f"没有找到错误日志文件。\n请检查路径是否正确：{pattern}")

    all_dfs = []
    valid_epochs = []

    print("=" * 100)
    print("开始读取验证集错误 CSV")
    print("=" * 100)

    for file_path in files:
        epoch = extract_epoch(file_path)
        if epoch is None:
            continue

        df = read_csv_safely(file_path)

        # 兼容旧版 CSV（如果没有这些列，就填默认值）
        if "error_type" not in df.columns: df["error_type"] = "ID_Error"
        if "true_fas" not in df.columns: df["true_fas"] = -1
        if "pred_fas" not in df.columns: df["pred_fas"] = -1

        df["epoch"] = epoch
        parsed = df["image_path"].apply(parse_from_path)
        df["person"] = parsed.apply(lambda x: x[0])
        df["status"] = parsed.apply(lambda x: x[1])
        df["filename"] = parsed.apply(lambda x: x[2])
        df["env"] = parsed.apply(lambda x: x[3])
        df["norm_image_path"] = parsed.apply(lambda x: x[4])

        all_dfs.append(df)
        valid_epochs.append(epoch)
        print(f"读取文件: {os.path.basename(file_path)} | epoch={epoch} | 错误数量={len(df)}")

    all_errors = pd.concat(all_dfs, ignore_index=True)
    valid_epochs = sorted(list(set(valid_epochs)))
    return all_errors, valid_epochs


# =========================================================
# 统计经常出错的人 (针对 ID 识别)
# =========================================================
def analyze_id_error_persons(id_errors, epochs):
    total_epoch_count = len(epochs)
    rows = []
    group_cols = ["true_id", "person"]

    for (true_id, person), g in id_errors.groupby(group_cols):
        error_epochs = sorted(g["epoch"].unique().tolist())
        error_epoch_count = len(error_epochs)
        error_epoch_ratio = error_epoch_count / total_epoch_count if total_epoch_count > 0 else 0

        total_error_count = len(g)
        unique_error_images = g["norm_image_path"].nunique()
        avg_error_per_epoch = total_error_count / total_epoch_count if total_epoch_count > 0 else 0

        status_counter = Counter(g["status"].tolist())
        env_counter = Counter(g["env"].tolist())
        pred_counter = Counter(g["pred_id"].astype(str).tolist())

        rows.append({
            "真实ID": true_id,
            "人员名称": person,
            "出现错误的轮数": error_epoch_count,
            "总轮数": total_epoch_count,
            "出错轮数百分比": round(error_epoch_ratio * 100, 2),
            "总错误次数": total_error_count,
            "涉及错误图片数量": unique_error_images,
            "平均每轮错误次数": round(avg_error_per_epoch, 2),
            "是否经常出错": error_epoch_ratio >= PERSON_MIN_ERROR_EPOCH_RATIO,
            "错误出现在哪些轮": ",".join(map(str, error_epochs)),
            "real_fake_mask错误分布": counter_to_text(status_counter),
            "环境错误分布": counter_to_text(env_counter),
            "最常被错认成的pred_id": counter_to_text(pred_counter, top_k=10),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values(
            by=["是否经常出错", "出现错误的轮数", "总错误次数"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
    return result


# =========================================================
# 统计防伪经常出错的人 (针对 FAS 防伪)
# =========================================================
def analyze_fas_error_persons(fas_errors, epochs):
    total_epoch_count = len(epochs)
    rows = []

    # 按照 人员 和 status 分组
    for (person, status), g in fas_errors.groupby(["person", "status"]):
        error_epochs = sorted(g["epoch"].unique().tolist())
        error_epoch_count = len(error_epochs)
        total_error_count = len(g)
        unique_error_images = g["norm_image_path"].nunique()

        if status == "fake":
            err_desc = "把假脸当成真脸 (APCER 漏网之鱼)"
        else:
            err_desc = "把真脸当成假脸 (BPCER 误杀好人)"

        env_counter = Counter(g["env"].tolist())

        rows.append({
            "人员名称": person,
            "图片类型(status)": status,
            "防伪错误类型": err_desc,
            "出现错误的轮数": error_epoch_count,
            "总轮数": total_epoch_count,
            "出错轮数百分比": round(error_epoch_count / total_epoch_count * 100, 2),
            "总错误次数": total_error_count,
            "涉及错误图片数量": unique_error_images,
            "环境错误分布": counter_to_text(env_counter),
            "错误出现在哪些轮": ",".join(map(str, error_epochs)),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values(by=["出现错误的轮数", "总错误次数"], ascending=[False, False]).reset_index(
            drop=True)
    return result


# =========================================================
# 统计经常出错的图片
# =========================================================
def analyze_error_images(df_errors, epochs, is_fas=False):
    total_epoch_count = len(epochs)
    rows = []

    for image_path, g in df_errors.groupby("norm_image_path"):
        first = g.iloc[0]
        error_epochs = sorted(g["epoch"].unique().tolist())
        error_epoch_count = len(error_epochs)
        error_epoch_ratio = error_epoch_count / total_epoch_count if total_epoch_count > 0 else 0

        row = {
            "图片路径": first["image_path"],
            "人员名称": first["person"],
            "图片类型": first["status"],
            "环境类型": first["env"],
            "文件名": first["filename"],
            "出错轮数": error_epoch_count,
            "总轮数": total_epoch_count,
            "出错轮数百分比": round(error_epoch_ratio * 100, 2),
            "是否经常出错": error_epoch_ratio >= IMAGE_MIN_ERROR_EPOCH_RATIO,
            "错误出现在哪些轮": ",".join(map(str, error_epochs)),
        }

        if is_fas:
            row["防伪错误类型"] = "把假当真(APCER)" if first["status"] == "fake" else "把真当假(BPCER)"
        else:
            pred_counter = Counter(g["pred_id"].astype(str).tolist())
            row["真实ID"] = first["true_id"]
            row["最常被错认成的pred_id"] = counter_to_text(pred_counter, top_k=10)

        rows.append(row)

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result = result.sort_values(
            by=["是否经常出错", "出错轮数", "人员名称"],
            ascending=[False, False, True]
        ).reset_index(drop=True)
    return result


# =========================================================
# 保存结果
# =========================================================
def save_outputs(id_persons, id_images, fas_persons, fas_images, all_errors, epochs):
    ensure_dir(OUTPUT_DIR)

    id_persons.to_csv(os.path.join(OUTPUT_DIR, "经常出错的人(ID识别).csv"), index=False, encoding="utf-8-sig")
    id_images.to_csv(os.path.join(OUTPUT_DIR, "经常出错的图片(ID识别).csv"), index=False, encoding="utf-8-sig")
    fas_persons.to_csv(os.path.join(OUTPUT_DIR, "防伪经常出错的人(FAS).csv"), index=False, encoding="utf-8-sig")
    fas_images.to_csv(os.path.join(OUTPUT_DIR, "防伪经常出错的图片(FAS).csv"), index=False, encoding="utf-8-sig")

    summary_txt = os.path.join(OUTPUT_DIR, "分析结果摘要.txt")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("验证集错误分析结果摘要 (包含 ID识别 和 活体防伪)\n")
        f.write("=" * 100 + "\n\n")

        f.write("【一、活体防伪 (FAS) 错误分析】\n")
        f.write("-" * 100 + "\n")
        if len(fas_persons) == 0:
            f.write("太棒了！没有发现任何防伪错误。\n\n")
        else:
            show_cols = ["人员名称", "图片类型(status)", "防伪错误类型", "出错轮数百分比", "总错误次数",
                         "涉及错误图片数量"]
            f.write(fas_persons[show_cols].head(TOP_K_PERSONS).to_string(index=False))
            f.write("\n\n")

        f.write("【二、身份识别 (ID) 错误分析】\n")
        f.write("-" * 100 + "\n")
        if len(id_persons) == 0:
            f.write("没有发现 ID 识别错误。\n\n")
        else:
            show_cols = ["真实ID", "人员名称", "出错轮数百分比", "总错误次数", "涉及错误图片数量",
                         "最常被错认成的pred_id"]
            f.write(id_persons[show_cols].head(TOP_K_PERSONS).to_string(index=False))
            f.write("\n\n")

        f.write("=" * 100 + "\n")
        f.write("文件说明\n")
        f.write("=" * 100 + "\n")
        f.write("1. 防伪经常出错的人(FAS).csv：专门分析 Fake 图片漏网之鱼，以及 Real 图片被误杀的情况。\n")
        f.write("2. 经常出错的人(ID识别).csv：专门分析真人身份被认错的情况。\n")

    print("\n" + "=" * 100)
    print("分析完成，结果已保存至:", OUTPUT_DIR)
    print("建议优先查看: 分析结果摘要.txt")
    print("=" * 100 + "\n")


# =========================================================
# 主函数
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)
    all_errors, epochs = load_all_errors()

    if len(all_errors) == 0:
        print("没有读取到任何错误记录，无法分析。")
        return

    # 拆分错误类型
    id_errors = all_errors[all_errors["error_type"].isin(["ID_Error", "ID_and_FAS_Error"])]
    fas_errors = all_errors[all_errors["error_type"].isin(["FAS_Error", "ID_and_FAS_Error"])]

    print(f"发现 ID 识别错误记录: {len(id_errors)} 条")
    print(f"发现 活体防伪错误记录: {len(fas_errors)} 条")

    print("正在分析 ID 识别错误...")
    id_person_result = analyze_id_error_persons(id_errors, epochs)
    id_image_result = analyze_error_images(id_errors, epochs, is_fas=False)

    print("正在分析 活体防伪(FAS) 错误...")
    fas_person_result = analyze_fas_error_persons(fas_errors, epochs)
    fas_image_result = analyze_error_images(fas_errors, epochs, is_fas=True)

    save_outputs(id_person_result, id_image_result, fas_person_result, fas_image_result, all_errors, epochs)


if __name__ == "__main__":
    main()