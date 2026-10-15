# -*- coding: utf-8 -*-
# run_ablation.py —— 逐个跑消融变体，每个变体独立进程 + 独立日志目录
import os
import sys
import subprocess
import pandas as pd

# 每个变体：要覆盖的环境变量（ABL_ 前缀 + ABLATION_TAG）
ABLATIONS = {
    "full":    {},
    "wo_dpm":  {"USE_DPM": "0"},
    "wo_local":{"USE_FAS_LOCAL_BRANCH": "0"},
    "wo_orth": {"STAGE1_LAMBDA_ORTHO": "0", "STAGE2_LAMBDA_ORTHO": "0"},
}

def main():
    rows = []
    for name, overrides in ABLATIONS.items():
        env = os.environ.copy()
        env["ABLATION_TAG"] = name
        for k, v in overrides.items():
            env[f"ABL_{k}"] = v

        print(f"\n{'='*70}\n[运行] 消融变体: {name}\n{'='*70}")
        ret = subprocess.run([sys.executable, "train.py"], env=env)
        if ret.returncode != 0:
            print(f"[失败] {name}，跳过汇总")
            continue

        report = os.path.join(f"./logs_ablation_{name}", "final_test_report.csv")
        if not os.path.exists(report):
            print(f"[警告] 未生成报告: {report}")
            continue

        df = pd.read_csv(report)
        row = {"variant": name}
        for col in ["id_acc", "occ_id_acc", "non_occ_id_acc", "fas_acc", "fas_f1", "occ_det_acc"]:
            row[col] = df[col].iloc[0] if col in df.columns else None
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv("ablation_summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== 消融结果汇总 =====\n")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()