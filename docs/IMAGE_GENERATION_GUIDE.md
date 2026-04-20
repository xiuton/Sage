# 图像生成指南 (IMAGE_GENERATION_GUIDE)

## 目录

- [概述](#概述)
- [核心架构](#核心架构)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [技术实现细节](#技术实现细节)
- [架构设计](#架构设计)
- [使用示例](#使用示例)
- [生成结果](#生成结果)
- [注意事项](#注意事项)

---

## 概述

Sage 项目实现了完整的 **VAE (Variational Autoencoder)** 和 **Diffusion Model (扩散模型)** 用于图像生成，包括 **Text-to-Image（文生图）** 功能。项目使用 Rust 语言和 Burn 深度学习框架实现，具备以下特性：

- ✅ **完整的 VAE 架构**：编码器、解码器、重参数化技巧
- ✅ **UNet 扩散模型**：用于噪声预测的神经网络架构
- ✅ **时间嵌入**：Sinusoidal 位置编码用于时间步表示
- ✅ **多种生成模式**：支持 VAE 直接生成、Diffusion 采样和文生图
- ✅ **文本条件生成**：通过提示词引导图像生成
- ✅ **GPU 加速支持**：使用 Wgpu 后端进行硬件加速
- ✅ **命令行工具**：便捷的图像生成接口

---

## 核心架构

### 1. VAE (变分自编码器)

VAE 是一种生成模型，它学习数据的潜在表示。包含以下组件：

#### 编码器 (VAEEncoder)
```rust
pub struct VAEEncoder<B: Backend> {
    conv1, bn1,  // 第1个卷积块
    conv2, bn2,  // 第2个卷积块
    conv3, bn3,  // 第3个卷积块
    conv4, bn4,  // 第4个卷积块
    fc_mu,       // 均值层
    fc_log_var,  // 对数方差层
}
```

**工作流程**：
1. 输入图像通过 4 个卷积块进行特征提取
2. 每个卷积块包含：Conv2d → BatchNorm → Gelu 激活
3. 特征图被展平并通过两个全连接层生成均值 (μ) 和对数方差 (log σ²)

#### 解码器 (VAEDecoder)
```rust
pub struct VAEDecoder<B: Backend> {
    fc1,  // 第一层全连接
    fc2,  // 第二层全连接
}
```

**工作流程**：
1. 潜在向量 z 通过 fc1 变换到隐藏维度
2. Gelu 激活函数
3. fc2 变换到图像空间 (3 × 64 × 64)
4. Reshape 为 4D 张量输出

#### 重参数化技巧
```rust
pub fn reparameterize(&self, mu: Tensor<B, 2>, log_var: Tensor<B, 2>) -> Tensor<B, 2> {
    let std = ((log_var / 2.0).exp() + 1e-8);
    let z = mu + std * epsilon;
    z
}
```

### 2. Diffusion Model (扩散模型)

扩散模型通过逐步去噪来生成图像，包含两个过程：

#### 前向过程 (Forward Process)
逐步向图像添加高斯噪声，最终变成纯噪声。

```rust
// 在 t 时刻的带噪图像
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - β_t) * x_{t-1}, β_t * I)
```

#### 反向过程 (Reverse Process)
学习从噪声中恢复图像。

```rust
// 预测噪声并去噪
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

#### UNet 架构
UNet 用于预测噪声，由以下部分组成：

```rust
pub struct UNet<B: Backend> {
    time_embedding: TimeEmbedding<B>,  // 时间嵌入
    down_blocks: Vec<UNetBlock<B>>,     // 下采样块
    mid_conv1, mid_bn1,                  // 中间层
    mid_conv2, mid_bn2,                  // 中间层
    up_blocks: Vec<UNetBlock<B>>,        // 上采样块
    final_conv: Conv2d<B>,               // 最终卷积
}
```

**UNetBlock 结构**：
```rust
pub struct UNetBlock<B: Backend> {
    conv1, bn1,   // 第一个卷积
    conv2, bn2,   // 第二个卷积
    time_mlp,     // 时间步MLP
    dropout,      // Dropout
}
```

#### 时间嵌入 (TimeEmbedding)
```rust
pub struct TimeEmbedding<B: Backend> {
    linear1: Linear<B>,  // 维度扩展
    linear2: Linear<B>,   // 维度收缩
}

pub fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
    let x = self.linear1.forward(t);   // dim -> dim * 4
    let x = Gelu::new().forward(x);
    self.linear2.forward(x)            // dim * 4 -> dim
}
```

---

## 快速开始

### 环境要求

- Rust 2024 edition
- Burn 0.19 深度学习框架
- NdArray 后端（CPU 训练/推理）

### 基础命令

#### 1. VAE 快速生成（推荐用于测试）

```bash
cargo run --bin image_gen -- --generate-only --image-size 64 --latent-dim 128
```

**参数说明**：
- `--generate-only`：使用 VAE 直接生成，跳过 Diffusion 采样
- `--image-size 64`：生成 64×64 像素图像
- `--latent-dim 128`：潜在空间维度为 128

#### 2. Diffusion 完整生成

```bash
cargo run --bin image_gen -- --image-size 64 --latent-dim 128 --steps 20
```

**参数说明**：
- `--steps 20`：采样步数，越多越精细但越慢
- 其他参数同上

---

## 命令行参数

### 完整参数列表

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--generate-only` | 仅使用 VAE 生成 | false | `--generate-only` |
| `--image-size` | 输出图像尺寸 | 64 | `--image-size 64` |
| `--latent-dim` | 潜在空间维度 | 128 | `--latent-dim 128` |
| `--steps` | Diffusion 采样步数 | 50 | `--steps 20` |
| `--output` | 输出文件路径 | 自动生成 | `--output my_image.png` |
| `--prompt` | 文生图文本提示词 | "a beautiful landscape" | `--prompt "a cat"` |
| `--seed` | 随机种子（可选） | None | `--seed 42` |

### 使用示例

#### 示例 1：生成测试图像

```bash
cargo run --bin image_gen -- --generate-only --image-size 64 --latent-dim 128
```

**输出**：
```
========================================
  Sage 图像生成工具 v1.0
========================================

📦 初始化模型配置...
✨ 生成模式: VAE 图像生成
📐 图像尺寸: 64x64
🔢 采样步数: 50

✅ 图像生成完成！
💾 保存到: ./assets/image_xxx.png
```

#### 示例 2：完整 Diffusion 生成

```bash
cargo run --bin image_gen -- --image-size 64 --latent-dim 128 --steps 10
```

**输出**：
```
========================================
  Sage 图像生成工具 v1.0
========================================

📦 初始化模型配置...
✨ 扩散模型生成模式
📐 图像尺寸: 64x64
🔢 采样步数: 10

✅ 图像生成完成！
💾 保存到: generated_image.png
```

#### 示例 3：自定义输出路径

```bash
cargo run --bin image_gen -- --generate-only --output ./my_images/test.png --image-size 64
```

#### 示例 4：文生图生成（Text-to-Image）

```bash
cargo run --bin image_gen -- --backend gpu --prompt "a beautiful landscape" --image-size 64 --steps 20
```

**输出**：
```
========================================
  Sage 图像生成工具 v1.0
========================================

📦 初始化模型配置...
🚀 使用 GPU 后端进行图像生成...
✨ 生成模式: 文生图 (Text-to-Image)
📝 提示词: a beautiful landscape
📐 图像尺寸: 64x64
🔢 采样步数: 20

✅ 图像生成完成！
💾 保存到: ./assets/image_xxx.png
```

#### 示例 5：GPU 加速文生图

```bash
cargo run --bin image_gen -- --backend gpu --prompt "a red rose in the rain" --image-size 64 --steps 50 --output ./my_rose.png
```

---

## 技术实现细节

### Burn 0.19 API 使用

项目使用 Burn 0.19 深度学习框架，关键 API：

#### 卷积配置
```rust
Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
    .with_padding(burn::nn::PaddingConfig2d::Same)
    .with_stride([stride, stride])
    .init(device)
```

#### 批归一化配置
```rust
BatchNormConfig::new(num_features).init(device)
```

#### 线性层配置
```rust
LinearConfig::new(in_features, out_features).init(device)
```

#### Dropout 配置
```rust
DropoutConfig::new(p).init()  // p 为丢弃概率
```

### 张量操作

#### 创建全零张量
```rust
Tensor::zeros([batch, channels, height, width], device)
```

#### 创建全填充张量
```rust
Tensor::full([batch, channels], fill_value, device)
```

#### 张量切片
```rust
tensor.slice([start_idx..end_idx, ...])
```

#### 张量重塑
```rust
tensor.reshape([new_batch, new_channels, new_height, new_width])
```

---

## 架构设计

### 图像生成流程

#### VAE 直接生成流程

```
潜在向量 z (随机)
    ↓
VAEDecoder.forward(z)
    ↓
图像张量 [1, 3, 64, 64]
    ↓
tensor_to_image_simple()
    ↓
PNG 图像文件
```

#### Diffusion 完整生成流程

```
初始化噪声 x_T [1, 128, 8, 8]
    ↓
for t in (T-1 .. 0):
    ↓
    x_t → UNet → noise_pred
    ↓
    计算去噪图像 x_{t-1}
    ↓
VAEDecoder 解码潜在表示
    ↓
图像张量 [1, 3, 64, 64]
    ↓
tensor_to_image_simple()
    ↓
PNG 图像文件
```

#### 文生图（Text-to-Image）流程

```
文本提示词 "a beautiful landscape"
    ↓
SimpleTokenizer 编码 → token IDs
    ↓
Embedding 层 → 文本特征向量
    ↓
均值池化 → 条件向量 [1, 128]
    ↓
条件向量注入到去噪过程
    ↓
UNet 噪声预测时加入条件信息
    ↓
生成与文本匹配的图像
```

### 文本编码器 (SimpleTokenizer)

为了支持文生图功能，我们实现了一个简单的分词器：

```rust
pub struct SimpleTokenizer {
    vocab: Vec<String>,              // 词汇表
    char_to_id: HashMap<char, usize>,  // 字符到ID映射
}

impl SimpleTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        // 创建基础词汇表：a-z, A-Z, 0-9 及其他符号
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        // 将文本转换为token IDs
    }
}
```

**工作流程**：
1. 提示词通过 `SimpleTokenizer::encode()` 转换为 token IDs
2. Token IDs 通过 Embedding 层获得文本特征
3. 文本特征经过均值池化得到条件向量
4. 条件向量在 UNet 去噪时被添加到噪声预测中
5. 这样不同提示词会引导生成不同的图像特征

### 维度变换

| 阶段 | 张量形状 | 说明 |
|------|----------|------|
| 初始噪声 | [1, 128, 8, 8] | latent_dim × latent_h × latent_h |
| 时间张量 | [1, 128] | 潜在维度 |
| 编码器输出 | [1, 128] | 潜在向量 |
| 解码器输出 | [1, 3, 64, 64] | 图像 |

### 模型配置

```rust
pub struct DiffusionConfig {
    pub image_size: 64,        // 图像尺寸
    pub in_channels: 3,        // 输入通道 (RGB)
    pub hidden_channels: 128,  // 隐藏层通道数
    pub num_timesteps: 1000,   // 总时间步数
    pub latent_dim: 128,       // 潜在空间维度
    pub beta_start: 0.0001,     // 噪声调度起始值
    pub beta_end: 0.02,         // 噪声调度结束值
}
```

### 损失函数与优化

#### VAE 损失

VAE 的损失函数由两部分组成：

1. **重构损失**：重建图像与原图像的均方误差
```rust
let recon_loss = (recon - x).powf(2.0).mean();
```

2. **KL 散度**：潜在空间的正则化
```rust
let kl_loss = -0.5 * (1 + log_var - mu.powf(2.0) - log_var.exp()).mean();
```

3. **总损失**
```rust
let total_loss = recon_loss + beta * kl_loss;
```
其中 beta 是 KL 散度的权重系数。

#### Diffusion 损失

扩散模型的训练目标是预测添加的噪声：

```rust
let noise_pred = unet.forward(x_t, t_emb);
let loss = (noise_pred - noise).powf(2.0).mean();
```

---

## 使用示例

### 示例 1：生成随机 VAE 图像

```bash
cargo run --bin image_gen -- --generate-only --image-size 64 --latent-dim 128
```

**输出图像**：`./assets/image_xxx.png`

### 示例 2：多步 Diffusion 生成

```bash
cargo run --bin image_gen -- --image-size 64 --latent-dim 128 --steps 20
```

**说明**：步数越多，生成质量越高，但速度越慢

### 示例 3：快速测试模式

```bash
cargo run --bin image_gen -- --generate-only --image-size 32 --latent-dim 64 --steps 5
```

**用途**：快速验证功能，生成小尺寸图像

### 示例 4：高质量生成

```bash
cargo run --bin image_gen -- --image-size 64 --latent-dim 256 --steps 100
```

**说明**：
- 更大的潜在维度 (256) 提供更丰富的表示
- 更多采样步数 (100) 提供更精细的去噪

---

## 生成结果

### 输出位置

| 生成模式 | 输出路径 |
|----------|----------|
| VAE 生成模式 (`--generate-only`) | `./assets/image_xxx.png` |
| Diffusion 生成模式 | `./generated_image.png` |

### 图像格式

- **格式**：PNG
- **颜色空间**：RGB
- **像素值**：0-255 (8位)

### 输出文件名

#### VAE 模式
文件名格式：`image_xxx.png`
- 包含 8 字符哈希值确保唯一性
- 示例：`image_050b88e85af48aa0.png`

#### Diffusion 模式
文件名格式：`generated_image.png`
- 固定文件名，方便覆盖测试

---

## 注意事项

### 1. 性能考虑

- **CPU 后端**：默认使用 NdArray 后端，CPU 运行
- **批处理**：当前仅支持 batch_size=1
- **内存占用**：Diffusion 完整生成比 VAE 直接生成需要更多内存

### 2. 生成质量

- **VAE 直接生成**：快速但质量有限，适合测试
- **Diffusion 生成**：质量更高，但需要更多步数
- **模型未训练**：当前模型参数是随机初始化的，生成的是随机噪声图像

### 3. 参数调优建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--image-size` | 64 | 64×64 是性能和质量的平衡点 |
| `--latent-dim` | 128 | 128 是隐藏通道数的 1 倍 |
| `--steps` | 10-50 | 10 步快速测试，50 步高质量 |

### 4. 已知限制

- ⚠️ 模型参数是随机初始化的，需要训练才能生成有意义图像
- ❌ 图像到图像（img2img）功能尚未实现
- ❌ ControlNet 功能尚未实现

### 5. 未来计划

- [ ] 添加预训练的 VAE/Diffusion 模型权重加载
- [ ] 实现更高级的文本到图像生成
- [ ] 添加图像到图像的转换功能
- [ ] 支持 ControlNet 条件控制
- [ ] 添加更高分辨率图像生成

---

## 模型训练指南

要让文生图模型能够根据提示词生成具体内容，需要进行专门的训练。以下是完整的训练流程：

### 1. 准备训练数据

**数据格式**：JSONL 格式，每行一个文本-图像对

```json
{"prompt": "a cat sitting on a chair", "image_path": "data/cat1.jpg"}
{"prompt": "a dog playing in the park", "image_path": "data/dog1.jpg"}
{"prompt": "a beautiful sunset over the ocean", "image_path": "data/sunset1.jpg"}
```

**数据要求**：
- 每个样本包含 `prompt`（文本描述）和 `image_path`（图像路径）
- 图像分辨率建议 64x64 或 128x128
- 建议至少 10,000 对数据以获得良好效果
- 数据多样性：包含多种场景、物体、风格

### 2. 配置文件

创建训练配置文件，建议存放在 `configs/` 目录：

```bash
# 创建配置目录
mkdir -p configs

# 创建配置文件 configs/vae_diffusion.json
```

**配置文件内容**：

```json
{
  "image_size": 64,        // 生成图像的尺寸（宽×高）
  "latent_dim": 128,       // 潜在空间维度
  "hidden_channels": 128,   // 隐藏层通道数
  "num_timesteps": 1000,    // 扩散模型的总时间步数
  "beta_start": 0.0001,     // 噪声调度的起始值
  "beta_end": 0.02,         // 噪声调度的结束值
  "batch_size": 16,         // 训练批次大小
  "learning_rate": 0.0001,  // 学习率
  "num_epochs": 100,        // 训练轮数
  "text_embedding_dim": 128, // 文本嵌入维度
  "vocab_size": 50000       // 词汇表大小
}
```

**参数说明**：

| 参数 | 说明 | 推荐值 | 调整建议 |
|------|------|--------|----------|
| `image_size` | 生成图像的尺寸（宽×高） | 64 | 64-128，尺寸越大需要更多显存 |
| `latent_dim` | 潜在空间维度，控制生成图像的细节 | 128 | 64-256，维度越大细节越丰富 |
| `hidden_channels` | UNet 隐藏层通道数 | 128 | 64-256，通道数越多能力越强 |
| `num_timesteps` | 扩散模型的总时间步数 | 1000 | 500-2000，步数越多生成质量越高 |
| `beta_start` | 噪声调度的起始值 | 0.0001 | 固定值，一般不需要调整 |
| `beta_end` | 噪声调度的结束值 | 0.02 | 固定值，一般不需要调整 |
| `batch_size` | 训练批次大小 | 16 | 根据 GPU 显存调整，8-32 |
| `learning_rate` | 学习率 | 0.0001 | 1e-5 到 1e-3，建议使用余弦退火 |
| `num_epochs` | 训练轮数 | 100 | 50-200，根据数据量和模型大小调整 |
| `text_embedding_dim` | 文本嵌入维度 | 128 | 与 latent_dim 保持一致 |
| `vocab_size` | 词汇表大小 | 50000 | 10000-100000，根据文本复杂度调整 |

### 3. 启动训练

```bash
# 文生图模型训练
cargo run --release --bin train -- \
    --text-to-image \
    --image-text-data data/text_image_pairs.jsonl \
    --config-path configs/vae_diffusion.json \
    --output-dir ./models/text_to_image \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --num-epochs 100 \
    --backend gpu
```

### 4. 训练过程

训练会包含以下阶段：

1. **VAE 预训练**：训练编码器和解码器
   - 学习图像的潜在表示
   - 重建损失：MSE + KL 散度

2. **扩散模型训练**：训练 UNet 噪声预测网络
   - 前向加噪过程
   - 反向去噪学习

3. **文本条件训练**：将文本特征与图像生成关联
   - 文本编码学习
   - 条件扩散模型训练

### 5. 使用训练好的模型

训练完成后，使用训练好的模型进行生成：

```bash
# 使用训练好的模型生成图像
cargo run --bin image_gen -- \
    --model-path ./models/text_to_image \
    --prompt "a cat wearing sunglasses" \
    --steps 50 \
    --output ./generated_cat.png

# GPU 加速生成
cargo run --bin image_gen -- \
    --backend gpu \
    --model-path ./models/text_to_image \
    --prompt "a beautiful landscape with mountains" \
    --steps 100 \
    --output ./generated_landscape.png
```

### 6. 训练技巧

**数据准备**：
- **数据清洗**：确保文本描述与图像内容匹配
- **数据增强**：对图像进行随机裁剪、翻转、亮度调整
- **文本处理**：标准化文本格式，去除噪声

**训练配置**：
- **批次大小**：根据 GPU 显存调整，建议 16-32
- **学习率**：使用余弦退火调度，初始 1e-4
- **正则化**：添加 Dropout (0.1-0.3) 防止过拟合
- **梯度裁剪**：设置梯度范数阈值 1.0

**加速训练**：
- **混合精度**：启用 FP16 训练
- **分布式训练**：多 GPU 并行训练
- **梯度累积**：当显存不足时使用

### 7. 评估生成质量

训练过程中，定期生成样本评估质量：

```bash
# 评估生成质量
cargo run --bin image_gen -- \
    --model-path ./models/text_to_image/checkpoint_50 \
    --prompt "a red rose in a vase" \
    --output ./eval/rose_50.png

cargo run --bin image_gen -- \
    --model-path ./models/text_to_image/checkpoint_100 \
    --prompt "a red rose in a vase" \
    --output ./eval/rose_100.png
```

**评估指标**：
- **视觉质量**：图像清晰度、色彩自然度
- **文本相关性**：生成内容与提示词的匹配程度
- **多样性**：不同提示词生成不同风格的图像
- **稳定性**：相同提示词多次生成的一致性

**详细文档**：完整的训练指南请参考 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。

---

## 相关文档

- [COMMANDS.md](COMMANDS.md) - 完整命令行参数手册
- [MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md) - 多模态功能指南
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 训练指南
- [PROJECT_STATUS.md](PROJECT_STATUS.md) - 项目开发状态

---

## 技术栈

- **语言**：Rust 2024 edition
- **深度学习框架**：Burn 0.19
- **张量后端**：NdArray (CPU)
- **图像处理**：image crate
- **命令行解析**：clap

---

## 许可与贡献

本项目是 Sage 大模型项目的一部分，采用相同的开源许可证。

如有问题或建议，请提交 Issue 或 Pull Request。
