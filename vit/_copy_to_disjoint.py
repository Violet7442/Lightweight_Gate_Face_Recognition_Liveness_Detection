# -*- coding: utf-8 -*-
"""按 source-disjoint 重新划分清单，把文件复制到统一的新目录。

输入 : dataset_resplit_disjoint_manifest.csv（由 _resplit_source_disjoint.py 生成）
输出 : {DS_DISJOINT}/{new_split}/{person_name}/{category}/{filename}

real/mask/fake 统一落在同一目录结构下，且已排除标签冲突样本、消除跨 split 泄漏。
dataset.py 后续把 RAW_DATA_ROOT / ALIGNED_REAL_ROOT 都指向 data_disjoint 即可
（同一 root 下 real/mask/fake 并列）。
"""
import csv
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST = Path(os.environ.get("DS_OUT", r"/root/autodl-tmp/CV/dataset_resplit_disjoint_manifest.csv"))
DISJOINT = Path(os.environ.get("DS_DISJOINT", r"/root/autodl-tmp/CV/dataset/data_disjoint"))


def main():
    rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8-sig")))
    kept = [r for r in rows if r["excluded"] != "1"]
    skipped = [r for r in rows if r["excluded"] == "1"]

    # 清空旧目标目录，保证幂等
    if DISJOINT.exists():
        shutil.rmtree(DISJOINT)

    copied = 0
    missing_src = 0
    per_split = Counter()
    per_cat = Counter()
    for r in kept:
        src = Path(r["source_path"])
        if not src.exists():
            missing_src += 1
            print(f"[缺失源文件] {src}")
            continue
        dst_dir = DISJOINT / r["new_split"] / r["person_name"] / r["category"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied += 1
        per_split[r["new_split"]] += 1
        per_cat[(r["new_split"], r["category"])] += 1

    print(f"MANIFEST_ROWS={len(rows)} KEPT={len(kept)} EXCLUDED={len(skipped)}")
    print(f"COPIED={copied} MISSING_SRC={missing_src}")
    print(f"SPLITS={dict(per_split)}")
    for s in ("train", "val", "test"):
        print(f"  {s}: real={per_cat[(s,'real')]} mask={per_cat[(s,'mask')]} fake={per_cat[(s,'fake')]}")
    print(f"OUTPUT_ROOT={DISJOINT}")


if __name__ == "__main__":
    main()