
# 📌 ViT (Vision Transformer) 结构与实现要点

## 📂 一、针对 2D 图像的 ViT 模型

### 🗂️ 1. Patch 分块与编码

- **目的**：把整张图片拆成相同大小的小块（patch），再展平并映射到固定维度空间，便于后续送入 Transformer 处理。
- **输入格式**：形如 `[Batch, Channels, Height, Width]`，例如 `[B, 3, 256, 256]`。
- **实现思路**：
  - 先利用 `Rearrange`（来自 `einops` 库）把高宽切割成小块并拉平成向量。
  - 然后用线性层把每个 patch 投影到统一的特征维度 `dim`。

  ```python
  Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
  ```

### 🧩 2. 核心 Attention 层

- **功能**：执行多头自注意力，用于捕捉 patch 间的关系。
- **特点**：
  - Query、Key、Value 一次性通过一个线性映射生成。
  - 继续使用 `einops` 简化张量维度转换。

### 🧱 3. Transformer 编码器

- **结构**：
  - 多层编码块，每层包含：
    - 层归一化（LayerNorm）
    - 多头注意力
    - 前向传播（FeedForward）
  - 块与块之间使用残差连接（Residual），归一化通常放在前（Post-LN）。

### 🎯 4. ViT 总体流程

- 输入图片 -> Patch 嵌入 -> 加上可训练的分类 token -> 叠加位置编码 -> 经过 Transformer 编码 -> 最后用分类头生成类别预测。

- 典型使用：

  ```python
  model = ViT(image_size=256, patch_size=16, num_classes=100, ...)
  img = torch.randn(1, 3, 256, 256)
  logits = model(img)  # 输出 shape: [1, 100]
  ```

---

## 📂 二、针对 1D 时序信号的 ViT 模型

### 🎢 1. 输入设计

- **形状**：`[Batch, Channels, Length]`，例如 `[4, 3, 256]`。
- **分块**：把一维数据按长度切成等长小段（patch），每个 patch 包含若干时间点的数据。

### 🗂️ 2. Patch 嵌入（1D版）

- 使用 `Rearrange` 将一维序列按 patch 大小切块并拉平：

  ```python
  Rearrange('b c (n p) -> b n (p c)', p=patch_size)
  ```

- 每个 patch 再通过线性层投影到统一维度。

### 🔑 3. 分类 Token 与拼接

- 1D 模型同样需要一个分类 token，形状是 `[dim]`。
- 利用 `repeat` 扩充为 batch 大小，并借助 `einops` 的 `pack` 和 `unpack` 方便与 patch 序列拼接和拆分。

### 🧠 4. Transformer 主体

- 编码层结构与 2D 版本完全一致，重复堆叠多个块。

### 📈 5. 最终输出

- 最终输出为每条序列的分类结果，形状 `[Batch, num_classes]`。

---

## 🔗 参考仓库

- 代码示例来源：[lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)

---

## ✅ 总结

本笔记覆盖了 ViT 在图像和一维时序场景下的主要实现思路，包括 Patch 的切分方式、Token 的拼接方法以及编码结构设计，结合 `einops` 实现高效且清晰的张量操作。适用于从零开始理解 ViT 的结构与 PyTorch 实现。
