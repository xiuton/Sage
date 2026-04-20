# Sage 多模态功能指南

本文档详细介绍 Sage 项目的多模态功能，包括架构设计、使用方法和配置选项。

## 目录

1. [架构概述](#架构概述)
2. [视觉编码器](#视觉编码器)
3. [多模态融合](#多模态融合)
4. [图像预处理](#图像预处理)
5. [训练和推理](#训练和推理)
6. [配置选项](#配置选项)
7. [使用示例](#使用示例)
8. [完整使用指南](#完整使用指南)

---

## 完整使用指南

**📖 如需详细的使用方法、代码示例和配置说明，请查看：**

**[MULTIMODAL_USAGE.md](./MULTIMODAL_USAGE.md)** - 完整的多模态使用指南，包含：
- ✅ 快速开始教程
- ✅ 所有功能模块的详细说明
- ✅ 完整的代码示例
- ✅ 高级配置和最佳实践
- ✅ 常见问题解答

---

## 架构概述

Sage 的多模态功能通过以下模块实现：

```
图像输入 → 图像预处理 → 视觉编码器 → 多模态融合 → 文本模型
                                                       ↑
文本输入 ──────────────────────────────────────────────┘
```

**核心组件：**
- **ImagePreprocessor**：图像预处理流水线
- **VisionEncoder**：视觉特征编码器（支持 ResNet 和 Vision Transformer）
- **CrossAttention**：跨模态注意力机制
- **MultimodalFusion**：多模态特征融合层
- **MultimodalModule**：完整的多模态集成模块

---

## 视觉编码器

Sage 提供两种视觉编码器选项：

### ResNet 编码器（推荐用于初步使用）

**特点：**
- 基于残差网络的 CNN 架构
- 包含 4 个残差块
- 支持 224x224 图像输入
- 计算效率高，适合快速迭代

**配置参数：**
```rust
VisionEncoderConfig {
    in_channels: 3,           // 输入通道数（RGB图像）
    hidden_channels: 64,      // 隐藏层通道数
    out_dim: 512,             // 输出特征维度
    encoder_type: "resnet".into(),  // 编码器类型
    num_layers: 4,            // 层数
    patch_size: 16,           // Patch大小（ResNet不使用，保留用于兼容性）
    image_size: 224,          // 图像尺寸
}
```

### Vision Transformer (ViT) 编码器

**特点：**
- 基于 Transformer 的自注意力架构
- 将图像分割为 patches 处理
- 适合捕捉长距离视觉依赖
- 配置灵活，可扩展性强

**配置参数：**
```rust
VisionEncoderConfig {
    in_channels: 3,
    hidden_channels: 64,
    out_dim: 512,
    encoder_type: "vit".into(),  // 使用 ViT 编码器
    num_layers: 4,                // Transformer 层数
    patch_size: 16,               // Patch 大小（16x16）
    image_size: 224,              // 图像尺寸
}
```

**选择建议：**
- **ResNet**：快速原型、计算资源有限、小数据集
- **ViT**：大数据集、高质量视觉理解、灵活配置

---

## 多模态融合

Sage 提供多种融合策略：

### 1. 门控融合（Gated Fusion）

**原理：**
```
融合特征 = gate * 文本特征 + (1-gate) * 视觉特征
```

**特点：**
- 自适应权重分配
- 简单高效
- 适合大多数场景

### 2. 拼接融合（Concatenation）

**原理：**
```
融合特征 = [文本特征; 视觉特征]
```

**特点：**
- 保留完整信息
- 适合需要丰富特征的任务

### 3. 加法融合（Addition）

**原理：**
```
融合特征 = 文本特征 + 视觉特征
```

**特点：**
- 参数少
- 计算快

### 4. 跨模态注意力融合（Cross Attention）

**原理：**
使用跨模态注意力机制让文本特征关注视觉特征

**特点：**
- 最灵活
- 可学习的视觉注意力
- 适合复杂多模态任务

---

## 图像预处理

### 预处理流程

```
原始图像 → 归一化 → 标准化 → 预处理后图像
```

### ImagePreprocessingConfig

```rust
ImagePreprocessingConfig {
    target_size: 224,              // 目标图像尺寸
    normalize: true,               // 是否标准化
    mean: [0.485, 0.456, 0.406],  // ImageNet 均值
    std: [0.229, 0.224, 0.225],   // ImageNet 标准差
    random_crop: false,            // 随机裁剪（训练增强）
    random_flip: false,            // 随机翻转（训练增强）
    center_crop: true,             // 中心裁剪
}
```

### 预处理细节

**归一化：**
```
归一化图像 = 原始图像 / 255.0
```
将像素值从 [0, 255] 转换为 [0, 1]

**标准化：**
```
标准化图像 = (归一化图像 - mean) / std
```
使用 ImageNet 统计量进行标准化

---

## 训练和推理

### 训练

**启用多模态训练：**
```bash
cargo run --release --bin train -- `
    --multimodal `
    --sft-jsonl data/multimodal_data.jsonl `
    --output-dir ./models/mm_model `
    --vision-out-dim 512 `
    --fusion-strategy cross_attention
```

**数据格式：**
```json
{
    "prompt": "描述这张图片",
    "response": "这是一张...的图片",
    "image_path": "data/images/sample.jpg"
}
```

或使用 messages 格式：
```json
{
    "messages": [
        {
            "role": "user",
            "content": "描述这张图片",
            "image_path": "data/images/sample.jpg"
        },
        {
            "role": "assistant",
            "content": "这是一张...的图片"
        }
    ]
}
```

### 推理

**启用多模态推理：**
```bash
cargo run --bin infer -- `
    --model-dir ./models/mm_model `
    --multimodal `
    --image-path data/images/test.jpg `
    --prompt "描述这张图片"
```

**完整参数示例：**
```bash
cargo run --bin infer -- `
    --model-dir ./models/mm_model `
    --use-best `
    --multimodal `
    --image-path data/images/test.jpg `
    --prompt "描述这张图片" `
    --num-tokens 100 `
    --temperature 0.7 `
    --backend gpu
```

---

## 配置选项

### MultimodalConfig 配置

完整的多模态配置示例：

```rust
MultimodalConfig {
    // 视觉编码器配置
    vision_encoder: VisionEncoderConfig {
        in_channels: 3,
        hidden_channels: 64,
        out_dim: 512,
        encoder_type: "resnet".into(),
        num_layers: 4,
        patch_size: 16,
        image_size: 224,
    },
    // 融合配置
    fusion: MultimodalFusionConfig {
        text_dim: 512,
        vision_dim: 512,
        output_dim: 512,
        strategy: "cross_attention".into(),
    },
    // 预处理配置
    preprocessing: ImagePreprocessingConfig {
        target_size: 224,
        normalize: true,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
        random_crop: false,
        random_flip: false,
        center_crop: true,
    },
    // 是否启用多模态
    enable_multimodal: true,
}
```

### 配置文件

在 `config.json` 中配置：

```json
{
    "multimodal": {
        "enable_multimodal": true,
        "vision_encoder": {
            "encoder_type": "vit",
            "out_dim": 512
        },
        "fusion": {
            "strategy": "cross_attention"
        }
    }
}
```

---

## 使用示例

### 示例 1: 基础 ResNet 多模态

```bash
# 训练
cargo run --release --bin train -- `
    --multimodal `
    --sft-jsonl data/multimodal_simple.jsonl `
    --output-dir ./models/resnet_mm `
    --vision-out-dim 512 `
    --fusion-strategy gated

# 推理
cargo run --bin infer -- `
    --model-dir ./models/resnet_mm `
    --multimodal `
    --image-path data/images/sample.jpg `
    --prompt "这是什么？"
```

### 示例 2: Vision Transformer 高级配置

```bash
# 训练
cargo run --release --bin train -- `
    --multimodal `
    --sft-jsonl data/multimodal_advanced.jsonl `
    --output-dir ./models/vit_mm `
    --vision-out-dim 512 `
    --fusion-strategy cross_attention

# 推理
cargo run --bin infer -- `
    --model-dir ./models/vit_mm `
    --use-best `
    --multimodal `
    --image-path data/images/complex.jpg `
    --prompt "详细描述这张图片" `
    --num-tokens 200 `
    --temperature 0.8
```

### 示例 3: 集成测试

```bash
# 运行多模态集成测试
cargo test test_multimodal_resnet_integration -- --nocapture
cargo test test_multimodal_vit_integration -- --nocapture
cargo test test_vision_encoders -- --nocapture
```

---

## 高级功能

### 跨模态注意力机制详解

CrossAttention 模块结构：

```
文本特征 → Query 投影 → 多头注意力 → 输出投影 → 融合特征
                                                        ↑
视觉特征 → Key 投影 ─────────────────────────────────┘
视觉特征 → Value 投影 ───────────────────────────────┘
```

**配置选项：**
```rust
CrossAttentionConfig {
    text_dim: 512,      // 文本特征维度
    vision_dim: 512,    // 视觉特征维度
    num_heads: 8,       // 注意力头数
    dropout: 0.1,       // Dropout 率
}
```

### 多模态融合策略对比

| 策略 | 参数复杂度 | 灵活性 | 推荐场景 |
|------|-----------|--------|---------|
| add | 低 | 低 | 简单任务 |
| concatenate | 低 | 中 | 一般任务 |
| gated | 中 | 中 | 大多数场景 |
| cross_attention | 高 | 高 | 复杂任务 |

---

## 常见问题

### Q: 如何选择视觉编码器？

**A:** 
- **ResNet**：快速原型、小数据集、GPU 显存有限
- **ViT**：大数据集、高质量需求、灵活配置

### Q: 图像预处理中使用的归一化参数为什么是这些值？

**A:** 使用 ImageNet 数据集的统计量（mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]），这样可以利用在 ImageNet 上预训练的知识。

### Q: 多模态训练需要多少数据？

**A:**
- 最小：~100 条图文对（仅用于验证流程）
- 推荐：~1000 条以上图文对
- 高质量：~10000 条以上多样化图文对

### Q: 如何评估多模态模型？

**A:**
- 定性评估：人工观察生成的描述是否准确
- 定量评估：可以添加专门的多模态评估指标（如 CIDEr、BLEU 等）

---

## 下一步扩展

当前多模态功能已具备完整的基础架构，以下功能已实现：

### ✅ 已实现

1. **增强视觉编码器**
   - ✅ 更深的 ResNet 变体（ResNet18/34/50/101/152）
   - ✅ 完整的 ViT 架构（TransformerEncoderBlock、MultiHeadAttention、MLPBlock）
   - ✅ 预训练权重加载（WeightLoader、PretrainedWeightConfig）

2. **数据增强**
   - ✅ 随机裁剪（Random Crop）
   - ✅ 随机翻转（Horizontal Flip）
   - ✅ 颜色抖动（Color Jitter）
   - ✅ 中心裁剪（Center Crop）

3. **图像生成**
   - ✅ VAE 图像自编码器（VAEEncoder、VAEDecoder）
   - ✅ Diffusion 扩散模型（UNet、TimeEmbedding、DiffusionModel）
   - ✅ TextToImagePipeline（文生图流水线）

4. **评测指标**
   - ✅ MultimodalEvaluator（多模态评估器）
   - ✅ MultimodalMetrics（评估指标结构）
   - ✅ MetricsLogger（指标日志记录）

### 🚧 待实现

- 预训练权重加载（需要权重文件格式支持）
- FID、IS 等图像生成质量评估指标
- 更高级的数据增强（如 CutMix、MixUp）

---

## 更新日志

### v1.2 (2026-04-19)
- ✅ 实现 VAE 图像自编码器
- ✅ 实现 Diffusion 扩散模型（文生图核心）
- ✅ 实现完整的 Vision Transformer 架构
- ✅ 实现更深的 ResNet 变体（ResNet18/34/50/101/152）
- ✅ 实现数据增强功能（随机裁剪、翻转、颜色抖动）
- ✅ 实现预训练权重加载功能
- ✅ 实现多模态评估指标

### v1.1 (2026-04-19)
- ✅ 实现 ResNet 视觉编码器
- ✅ 实现 Vision Transformer (ViT) 编码器选项
- ✅ 实现跨模态注意力机制
- ✅ 实现多种融合策略（gated, concatenate, add, cross_attention）
- ✅ 实现图像预处理流水线
- ✅ 添加完整的多模态集成测试
- ✅ 更新多模态训练和推理文档

---

**作者：** Sage 团队  
**最后更新：** 2026-04-19  
**版本：** v1.2
