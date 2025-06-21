import os
import shutil
import random

#####Kelly
# 设置随机种子，保证划分可复现
random.seed(42)

# 原始图像数据集路径（包含各类子目录）
dataset_dir = r'D:/practicalTrainingCode/Images'

# 训练集和验证集输出路径
train_dir = r'D:/practicalTrainingCode/dataset/image2/train'
val_dir = r'D:/practicalTrainingCode/dataset/image2/val'

# 训练集占比
train_ratio = 0.7

# 创建输出目录（若不存在则创建）
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 遍历每个类别的子文件夹
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)

    # 跳过不是目录的项，例如 .DS_Store 等文件
    if not os.path.isdir(class_path) or class_name in ["train", "val"]:
        continue

    # 读取该类别下所有图片
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = [os.path.join(class_path, img) for img in images]

    # 打乱顺序并划分训练集和验证集
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # 拷贝训练图片
    for img in train_imgs:
        class_train_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)
        shutil.copy(img, class_train_dir)

    # 拷贝验证图片
    for img in val_imgs:
        class_val_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_val_dir, exist_ok=True)
        shutil.copy(img, class_val_dir)

##### Kelly
print("✅ 数据划分完成，训练集和验证集已生成！")
