# -*- coding: utf-8 -*-
"""Source-disjoint 重新划分脚本。

目标：
1. 同一 original_group_id（同一原图的 origin/blur/dark/glare 变体家族）必须落到同一 split；
2. 同一 content_sha256（文件字节完全相同）必须落到同一 split；
3. 同一哈希被标成多个不同 category（real/mask/fake）的标签冲突样本 —— 排除；
4. 按身份分层、贪心匹配目标比例地重划 train/val/test；
5. 自检：重划后跨 split 的 group / hash 泄漏必须为 0。

本脚本只生成新清单 + 自检报告，不移动任何物理文件。
"""
import csv
import hashlib
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

RAW_ROOT = Path(os.environ.get("DS_RAW", r"E:\py\CVproject\CV\dataset\data"))
ALIGNED_ROOT = Path(os.environ.get("DS_ALIGNED", r"E:\py\CVproject\CV\dataset\data_mtcnn_aligned"))
OUTPUT = Path(os.environ.get("DS_OUT", r"E:\work\dataset_resplit_disjoint_manifest.csv"))
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
VARIANT_RE = re.compile(r"_(origin|blur|dark|glare)$", re.IGNORECASE)
MIN_IMAGES_PER_ID = 5
SEED = 42
TARGET = {"train": 0.77, "val": 0.115, "test": 0.115}


def images(directory):
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.name.lower(),
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 1) 身份树（与 dataset.py 的 build_global_id_to_idx 一致）
train_root = RAW_ROOT / "train"
counts = {}
for person_dir in sorted((p for p in train_root.iterdir() if p.is_dir()), key=lambda p: p.name):
    counts[person_dir.name] = len(images(person_dir / "real")) + len(images(person_dir / "mask"))
valid_people = sorted(name for name, c in counts.items() if c >= MIN_IMAGES_PER_ID)
identity_map = {name: f"ID{i:03d}" for i, name in enumerate(valid_people, 1)}

# 2) 扫描实际加载样本（与 loader 口径一致）
loaded = []  # (path, split, person, category, source)
for split in SPLITS:
    raw_split = RAW_ROOT / split
    for raw_person in sorted((p for p in raw_split.iterdir() if p.is_dir()), key=lambda p: p.name):
        person = raw_person.name
        if person not in identity_map:
            continue
        for category in ("real", "mask"):
            aligned_dir = ALIGNED_ROOT / split / person / category
            selected = aligned_dir if aligned_dir.is_dir() else raw_person / category
            source = f"private_{category}_aligned" if aligned_dir.is_dir() else f"private_{category}_raw_fallback"
            for path in images(selected):
                loaded.append((path, split, person, category, source))
        raw_fake = {p.stem: p for p in images(raw_person / "fake")}
        alg_fake = {p.stem: p for p in images(ALIGNED_ROOT / split / person / "fake")}
        for stem in sorted(raw_fake):
            path = alg_fake[stem] if stem in alg_fake else raw_fake[stem]
            src_label = "private_fake_aligned" if stem in alg_fake else "private_fake_raw"
            loaded.append((path, split, person, "fake", src_label))

# 3) 计算逐样本属性
rows = []
for path, split, person, category, source in loaded:
    stem = path.stem
    m = VARIANT_RE.search(stem)
    variant = m.group(1).lower() if m else "unspecified"
    base_name = VARIANT_RE.sub("", stem)
    logical_key = f"{person}|{category}|{base_name}".lower()
    group_id = "GRP_" + sha256_text(logical_key)[:16]
    content_hash = sha256_file(path)
    rows.append({
        "sample_id": "SMP_" + sha256_text(f"{split}|{person}|{category}|{path.name}|{source}")[:16],
        "identity_id": identity_map[person],
        "person_name": person,
        "category": category,
        "id_label_available": int(category != "fake"),
        "fas_label": 0 if category == "fake" else 1,
        "occ_label": 1 if category == "mask" else (0 if category == "real" else -1),
        "original_split": split,
        "source_id": source,
        "original_group_id": group_id,
        "variant": variant,
        "file_extension": path.suffix.lower(),
        "content_sha256": content_hash,
        "source_path": str(path),
    })

print(f"IDENTITIES={len(identity_map)} TOTAL={len(rows)}")
print("ORIG_SPLITS=" + repr(dict(Counter(r['original_split'] for r in rows))))

# 4) Union-Find：合并同一 group_id 与同一 content_hash
parent = {}

def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

# 用样本下标做节点
for i, r in enumerate(rows):
    union(("g", r["original_group_id"]), ("g", r["original_group_id"]))
for i in range(len(rows)):
    find(("g", rows[i]["original_group_id"]))

group_first = {}
hash_first = {}
for i, r in enumerate(rows):
    gk = ("g", r["original_group_id"])
    hk = ("h", r["content_sha256"])
    if gk in group_first:
        union(("n", group_first[gk]), ("n", i))
    else:
        group_first[gk] = i
    if hk in hash_first:
        union(("n", hash_first[hk]), ("n", i))
    else:
        hash_first[hk] = i

# 5) 标签冲突检测：同一 content_hash 出现多个 category
hash_cats = defaultdict(set)
for r in rows:
    hash_cats[r["content_sha256"]].add(r["category"])
conflict_hashes = {h for h, cats in hash_cats.items() if len(cats) > 1}
conflict_rows = [i for i, r in enumerate(rows) if r["content_sha256"] in conflict_hashes]
print(f"LABEL_CONFLICT_HASHES={len(conflict_hashes)} ROWS={len(conflict_rows)}")

for i in range(len(rows)):
    r = rows[i]
    r["excluded"] = int(r["content_sha256"] in conflict_hashes)
    r["exclude_reason"] = "hash_category_conflict" if r["content_sha256"] in conflict_hashes else ""

kept_idx = [i for i, r in enumerate(rows) if not r["excluded"]]

# 6) 每个 cluster（union-find 根）汇总
cluster_rows = defaultdict(list)
for i in kept_idx:
    cluster_rows[find(("n", i))].append(i)

clusters = []
for cid, idxs in cluster_rows.items():
    idents = Counter(rows[i]["identity_id"] for i in idxs)
    cats = Counter(rows[i]["category"] for i in idxs)
    clusters.append({
        "idxs": idxs,
        "n": len(idxs),
        "main_ident": idents.most_common(1)[0][0],
        "main_cat": cats.most_common(1)[0][0],
    })

# 7) 按身份分层贪心重划
random.seed(SEED)
by_ident = defaultdict(list)
for c in clusters:
    by_ident[c["main_ident"]].append(c)

assign_ident = {}  # ident -> {'train':n,'val':n,'test':n}
new_split = {}
for ident in sorted(by_ident):
    clist = by_ident[ident]
    clist.sort(key=lambda c: -c["n"])  # 大 cluster 优先
    counts = {"train": 0, "val": 0, "test": 0}
    total = sum(c["n"] for c in clist)
    # 保底：最大 cluster 放 train
    if clist:
        big = clist[0]
        for j in big["idxs"]:
            new_split[j] = "train"
        counts["train"] += big["n"]
        clist = clist[1:]
    ident_seed = int(sha256_text(ident)[:8], 16)
    rng = random.Random(SEED ^ ident_seed)
    rng.shuffle(clist)
    for c in clist:
        # 贪心：选当前相对目标比例“缺口最大”的 split（最需要补充）
        best, best_deficit = None, None
        for s in ("train", "val", "test"):
            deficit = TARGET[s] - counts[s] / (total + 1e-9)
            if best_deficit is None or deficit > best_deficit:
                best, best_deficit = s, deficit
        for j in c["idxs"]:
            new_split[j] = best
        counts[best] += c["n"]

for i in kept_idx:
    rows[i]["new_split"] = new_split[i]
for i in conflict_rows:
    rows[i]["new_split"] = "excluded"

# 8) 自检：重划后跨 split 泄漏
group_splits = defaultdict(set)
hash_splits = defaultdict(set)
for i in kept_idx:
    r = rows[i]
    s = r["new_split"]
    group_splits[r["original_group_id"]].add(s)
    hash_splits[r["content_sha256"]].add(s)
cross_group = {k: v for k, v in group_splits.items() if len(v) > 1}
cross_hash = {k: v for k, v in hash_splits.items() if len(v) > 1}
print(f"POST-RESPLIT CROSS_SPLIT_GROUPS={len(cross_group)} CROSS_SPLIT_HASHES={len(cross_hash)}")

new_dist = Counter(r["new_split"] for r in rows)
print("NEW_SPLITS=" + repr(dict(new_dist)))

# 9) 输出
fieldnames = ["sample_id", "identity_id", "person_name", "category", "id_label_available", "fas_label",
              "occ_label", "original_split", "new_split", "excluded", "exclude_reason",
              "source_id", "original_group_id", "variant", "file_extension",
              "content_sha256", "source_path"]
rows.sort(key=lambda r: (r["new_split"], r["identity_id"], r["category"],
                         r["original_group_id"], r["variant"], r["sample_id"]))
with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"OUTPUT={OUTPUT}")
# 身份覆盖检查
kept = [r for r in rows if r["excluded"] == 0]
train_idents = {r["identity_id"] for r in kept if r["new_split"] == "train"}
print(f"TRAIN_IDENTITIES={len(train_idents)}/86")
print(f"TRAIN_MIN_PER_ID=" +
      repr(dict(Counter(r['identity_id'] for r in kept if r['new_split']=='train').most_common(1))))
# 每身份 train 样本数最小值
per = Counter(r['identity_id'] for r in kept if r['new_split']=='train')
print("TRAIN_PER_IDENT_MIN=", min(per.values()), "MAX=", max(per.values()))