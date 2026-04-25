# Sage 多模态功能 - 快速开始

本指南帮助您快速上手 Sage 的多模态功能。

## 📋 目录

1. [快速安装](#快速安装)
2. [10 分钟上手](#10-分钟上手)
3. [核心功能一览](#核心功能一览)
4. [示例代码](#示例代码)
5. [文档索引](#文档索引)

---

## 🚀 快速安装

### 前置要求

- Rust 1.75 或更高版本
- Cargo 包管理器

### 安装步骤

```bash
# 克隆项目
cd Sage

# 构建项目
cargo build --release

# 运行多模态快速开始示例
cargo run --example multimodal_quickstart
```

---

## ⚡ 10 分钟上手

### 步骤 1：基础多模态初始化

```rust
use sage::core::{VisionEncoder, VisionEncoderConfig, MultimodalFusion, MultimodalConfig};

let device = Default::default();

// 初始化视觉编码器
let vision_config = VisionEncoderConfig::default();
let vision_encoder = VisionEncoder::new(&vision_config, &device);

// 初始化多模态融合
let multimodal_config = MultimodalConfig {
    vision_dim: 512,
    text_dim: 512,
    fusion: "gated".to_string(),
};
let multimodal_fusion = MultimodalFusion::new(&multimodal_config, &device);
```

### 步骤 2：使用数据增强

```rust
use sage::core::DataAugmentation;

let data_aug = DataAugmentation {
    random_crop: true,
    random_flip: true,
    color_jitter: true,
    random_rotation: false,
    device: device.clone(),
};

let augmented_image = data_aug.apply(raw_image);
```

### 步骤 3：图像生成（文生图）

```rust
use sage::core::{TextToImagePipeline, DiffusionConfig};

let config = DiffusionConfig::default();
let text_to_image = TextToImagePipeline::new(&config, &device);

// 生成图像
let generated = text_to_image.generate(text_embedding, 50);
```

---

## 🎯 核心功能一览

### 1. 视觉编码器

| 功能 | 描述 | 状态 |
|------|------|------|
| ResNet18/34 | 轻量级 CNN | ✅ |
| ResNet50/101/152 | 深层 CNN | ✅ |
| Vision Transformer | 完整 ViT 架构 | ✅ |
| 预训练权重加载 | 迁移学习支持 | ✅ |

### 2. 多模态融合

| 功能 | 描述 | 状态 |
|------|------|------|
| Gated Fusion | 门控融合 | ✅ |
| Concatenation | 拼接融合 | ✅ |
| Cross Attention | 跨模态注意力 | ✅ |

### 3. 数据增强

| 功能 | 描述 | 状态 |
|------|------|------|
| 随机裁剪 | Random Crop | ✅ |
| 随机翻转 | Horizontal Flip | ✅ |
| 颜色抖动 | Color Jitter | ✅ |
| 中心裁剪 | Center Crop | ✅ |

### 4. 图像生成

| 功能 | 描述 | 状态 |
|------|------|------|
| VAE 自编码器 | 编码/解码 | ✅ |
| Diffusion 模型 | 去噪扩散 | ✅ |
| Text-to-Image | 文生图流水线 | ✅ |

### 5. 评估指标

| 功能 | 描述 | 状态 |
|------|------|------|
| BLEU 分数 | 图像描述评估 | ✅ |
| 视觉-语言对齐 | 特征相似度 | ✅ |
| PSNR | 生成质量 | ✅ |
| FID | Fréchet 距离 | ✅ |

---

## 💻 示例代码

### 运行示例

```bash
# 快速开始示例
cargo run --example multimodal_quickstart
```

### 更多示例

查看 `examples/` 目录获取更多完整代码示例。

---

## 📚 文档索引

| 文档 | 描述 |
|------|------|
| **[MULTIMODAL_USAGE.md](./MULTIMODAL_USAGE.md)** | 完整详细的使用指南 |
| **[MULTIMODAL_GUIDE.md](./MULTIMODAL_GUIDE.md)** | 架构设计和功能概览 |
| **[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)** | 训练指南 |
| **[COMMANDS.md](./COMMANDS.md)** | 命令参考 |

---

## 🚀 API 服务器使用（进阶）

API 服务器在启动时会加载 LLM 模型，同时提供 LLM 对话和多模态图像生成服务。

### 快速启动

```bash
cargo run --release --features="api" --bin api_server -- `
    --model-dir ./models/sage_model_formal `
    --backend gpu `
    --port 8000
```

### API 调用示例

```bash
# 加载 Diffusion 模型
curl -X POST http://localhost:8000/api/v1/diffusion/load `
  -H "Content-Type: application/json" `
  -d '{"model_path": "./models/text_to_image_full", "config_path": "./configs/config_vae_diffusion.json"}'

# 生成图像
curl -X POST http://localhost:8000/api/v1/images/generate `
  -H "Content-Type: application/json" `
  -d '{"prompt": "一只可爱的小猫", "steps": 100}'
```

详细文档请查看 **[API_GUIDE.md](./API_GUIDE.md)**。

---

## 🎉 下一步

1. 阅读 **[完整使用指南](./MULTIMODAL_USAGE.md)**
2. 尝试运行示例代码
3. 根据您的需求配置模型
4. 开始训练您的多模态模型！

---

**有问题？** 查看 **[常见问题解答](./MULTIMODAL_USAGE.md#常见问题)** 或提交 Issue。

**最后更新：** 2026-04-25
