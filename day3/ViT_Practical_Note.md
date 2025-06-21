
# 💡 Vision Transformer 实践笔记（专属路径版）

---

## 📁 项目目录结构一览

```text
D:/practicalTrainingCode/
├── TrainingImageDataset/
│   ├── deal_with_datasets.py      # 数据集划分
│   ├── prepare.py                 # 索引生成
│   ├── model.py                   # 模型定义
│   ├── train.py                   # 模型训练
│   ├── dataset.py                 # Dataset 类
│   ├── activationFunction.py      # 激活函数拓展
│   └── chihua.py                  # 拓展功能
├── Images/                        # 原始图像
├── dataset/image2/train/         # 训练集
├── dataset/image2/val/           # 验证集
└── logs_train/                    # TensorBoard 日志
```

---

## 🧩 一、数据预处理阶段

### 📌 原图路径设置
```python
dataset_dir = r'D:/practicalTrainingCode/Images'
```

### 📌 输出路径
```python
train_dir = r'D:/practicalTrainingCode/dataset/image2/train'
val_dir   = r'D:/practicalTrainingCode/dataset/image2/val'
```

### ✨ 步骤一：划分数据集
```bash
python deal_with_datasets.py
```

⏱ 输出结构：
```
image2/train/class_x/*.jpg
image2/val/class_x/*.jpg
```

### ✨ 步骤二：生成索引文件
```bash
python prepare.py
```

输出内容示例（每行对应一个样本）：
```
class1/img001.jpg 0
class2/img045.jpg 1
```

---

## 🧠 二、ViT 模型结构详解（model.py）

### 📸 模型输入输出流程图

![ViT结构图](https://raw.githubusercontent.com/google-research/vision_transformer/main/vit_architecture.png)

---

### 🔹 Patch Embedding 模块

将输入图像 $X \in \mathbb{R}^{B \times 3 \times H \times W}$ 切分成若干 patch：
$$
N = \frac{H \times W}{P^2}, \quad x_i \in \mathbb{R}^{3 \cdot P^2}
$$

再进行线性映射：
$$
z_i = x_i W + b \in \mathbb{R}^D
$$

Py代码片段：
```python
Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch, p2=patch)
```

---

### 🔹 多头注意力机制（Multi-head Self-Attention）

注意力权重计算公式如下：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

每个头计算后拼接：
$$
\text{MultiHead}(X) = \text{Concat}(head_1, \dots, head_h)W^O
$$

---

### 🔹 Transformer 编码器结构

每层由如下模块组成：

```text
LN → MSA → Add → LN → MLP → Add
```

- LayerNorm
- 多头注意力
- 前馈神经网络（MLP）
- 残差连接

---

## 🏋️ 三、模型训练流程（train.py）

### ✅ 启动训练
```bash
python train.py
```

### ⏱ 日志输出
```
[Epoch 1] Loss: 2075.35, Accuracy: 28.53%
Validation Accuracy: 35.67%
```

📁 TensorBoard 日志会存入：
```
D:/practicalTrainingCode/logs_train/
```

---

## 🔬 四、ViT 时间序列扩展版（1D）

时间序列输入形状：`[B, C, L]`  
Patch 生成方式：
```python
Rearrange('b c (n p) -> b n (p c)', p=patch_size)
```

📌 分类 token 使用 `repeat()` 方法拼接，Transformer 编码结构不变。

---

## 📚 我的感悟

> 这次项目让我真正理解了 Transformer 并非只用于文本任务。ViT 在图像与时间序列任务中展现了优秀的通用性。通过完整实现划分、索引、模型构建与训练流程，我对 PyTorch 框架的模块化设计也有了更深的理解。

---
