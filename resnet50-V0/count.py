import os

# 你的两个验证集路径
val_main = r"E:\py\CVproject\CV\dataset\data\verify"
val_mask = r"E:\py\CVproject\CV\dataset\Face-Mask-12K-Images-Dataset\Face Mask Dataset\Validation"

total_val = 0

# 统计主验证集
for person_name in os.listdir(val_main):
    person_dir = os.path.join(val_main, person_name)
    if not os.path.isdir(person_dir):  # 修正：把 os.isdir 改为 os.path.isdir
        continue
    for status in ['real', 'fake', 'mask']:
        s_dir = os.path.join(person_dir, status)
        if not os.path.exists(s_dir):
            continue
        for f in os.listdir(s_dir):
            if f.lower().endswith(('png','jpg','jpeg')):
                total_val +=1

# 统计口罩验证集
for cat in ['WithMask','WithoutMask']:
    cat_dir = os.path.join(val_mask, cat)
    if not os.path.exists(cat_dir):
        continue
    for f in os.listdir(cat_dir):
        if f.lower().endswith(('png','jpg','jpeg')):
            total_val +=1

print(f"✅ 验证集总数：{total_val}")