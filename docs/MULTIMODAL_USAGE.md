# Sage 多模态功能完整使用指南

本文档提供 Sage 多模态功能的完整详细使用说明，包括所有已实现功能的配置、代码示例和最佳实践。

## 目录

1. [快速开始](#快速开始)
2. [核心功能模块](#核心功能模块)
   - [视觉编码器](#视觉编码器)
   - [数据增强](#数据增强)
   - [预训练权重加载](#预训练权重加载)
   - [图像生成](#图像生成)
   - [多模态评估](#多模态评估)
3. [完整代码示例](#完整代码示例)
4. [高级配置](#高级配置)
5. [常见问题](#常见问题)

---

## 快速开始

### 1. 环境准备

确保已安装 Rust 1.75+ 和必要的依赖：

```bash
cd Sage
cargo build --release
```

### 2. 快速测试多模态功能

**运行多模态集成测试：**

```bash
# 运行 ResNet 多模态集成测试
cargo test test_multimodal_resnet_integration -- --nocapture

# 运行 ViT 多模态集成测试  
cargo test test_multimodal_vit_integration -- --nocapture

# 运行视觉编码器测试
cargo test test_vision_encoders -- --nocapture
```

### 3. 完整的多模态训练和推理流程

#### Step 1: 多模态模型训练

**参数详解：**

- `--bin train`：指定运行训练二进制文件
- `--multimodal`：启用多模态训练模式，支持图像和文本的联合训练，用于训练能够理解和生成多模态内容的模型
- `--sft-jsonl data/mm_test.jsonl`：指定SFT（监督微调）训练数据文件路径，JSONL格式每行包含图像路径和文本对
- `--output-dir models/mm_model`：指定模型输出目录，训练完成的模型权重和配置将保存在此目录
- `--vision-out-dim 512`：指定视觉编码器的输出特征维度，维度越高视觉特征表达能力越强
- `--fusion-strategy cross_attention`：指定多模态融合策略为跨模态注意力机制，让文本和视觉特征通过注意力交互融合
- `--batch-size 2`：训练批次大小，每批次处理2个样本，批次越小越节省显存但训练速度较慢
- `--learning-rate 0.0001`：学习率设置为0.0001，标准深度学习训练学习率，控制权重更新幅度
- `--num-epochs 1`：训练1轮，仅用于快速验证训练流程是否正常工作
- `--backend cpu`：使用CPU进行训练，适合调试和没有GPU硬件的环境

**完整命令：**

```bash
cargo run --bin train -- `
    --multimodal `
    --sft-jsonl data/mm_test.jsonl `
    --output-dir models/mm_model `
    --vision-out-dim 512 `
    --fusion-strategy cross_attention `
    --batch-size 2 `
    --learning-rate 0.0001 `
    --num-epochs 1 `
    --backend cpu
```

#### Step 2: 多模态模型推理

**参数详解：**

- `--bin infer`：指定运行推理二进制文件
- `--model-dir models/mm_model`：指定训练好的模型目录路径，包含模型权重和配置文件
- `--multimodal`：启用多模态推理模式，处理图像和文本的联合输入
- `--image-path data/text_to_images/cat.png`：指定输入图像的路径，支持PNG、JPG等常见图像格式
- `--prompt "描述这张图片"`：指定输入的文本提示词，用于指导模型理解或描述图像内容

**完整命令：**

```bash
cargo run --bin infer -- `
    --model-dir models/mm_model `
    --multimodal `
    --image-path data/text_to_images/cat.png `
    --prompt "描述这张图片"
```

#### Step 3: 文生图模型训练

**参数详解：**

- `--bin train`：指定运行训练二进制文件
- `--text-to-image`：启用文生图训练模式，用于训练文本到图像的扩散模型
- `--image-text-data data/text_to_image_pairs.jsonl`：指定图文训练数据文件路径，格式为JSONL，每行包含图像路径和文本描述
- `--config-path configs/config_vae_diffusion.json`：指定模型配置文件路径，包含VAE和Diffusion模型的超参数配置
- `--output-dir models/text_to_image`：指定模型输出目录，训练完成后会生成config.json和diffusion_model.mpk文件
- `--batch-size 2`：训练批次大小，表示每次迭代使用的样本数量，批次越小越节省显存但训练速度较慢
- `--learning-rate 0.001`：学习率设置为0.001，较大的学习率，适合快速收敛测试
- `--num-epochs 1`：训练1轮，仅用于快速验证训练流程是否正常
- `--backend cpu`：使用CPU进行训练，适合调试和没有GPU硬件的环境

**完整命令：**

```bash
cargo run --bin train -- `
    --text-to-image `
    --image-text-data data/text_to_image_pairs.jsonl `
    --config-path configs/config_vae_diffusion.json `
    --output-dir models/text_to_image `
    --batch-size 2 `
    --learning-rate 0.001 `
    --num-epochs 1 `
    --backend cpu
```

#### Step 4: 文生图模型推理

**参数详解：**

- `--bin image_gen`：指定运行图像生成二进制文件
- `--backend cpu`：使用CPU进行图像生成，适合在没有GPU硬件的环境下使用
- `--model-path models/text_to_image`：指定训练好的模型目录路径，目录中应包含config.json和diffusion_model.mpk文件
- `--prompt "一只可爱的小猫"`：文本提示词，描述想要生成的图像内容，支持中文和英文
- `--steps 50`：Diffusion采样步数，步数越多生成质量越高但速度越慢，50步是质量和速度的平衡点

**完整命令：**

```bash
cargo run --bin image_gen -- `
    --backend cpu `
    --model-path models/text_to_image `
    --prompt "一只可爱的小猫" `
    --steps 50
```

#### Step 5: 使用完整训练模型生成高质量图像

**参数详解：**

- `--bin image_gen`：指定运行图像生成二进制文件
- `--backend gpu`：使用GPU进行图像生成，大幅提升生成速度和质量
- `--model-path models/text_to_image_full`：指定完整训练的模型目录路径，包含充分训练的模型权重
- `--prompt "一只可爱的小猫，毛茸茸的，蓝色眼睛，在草地上玩耍"`：详细的文本提示词，引导模型生成更具体的图像
- `--steps 100`：使用100步采样，获得更高质量的生成结果
- `--output ./cat_generated.png`：指定输出图像路径

**完整命令：**

```bash
cargo run --bin image_gen -- `
    --backend gpu `
    --model-path models/text_to_image_full `
    --prompt "一只可爱的小猫，毛茸茸的，蓝色眼睛，在草地上玩耍" `
    --steps 100 `
    --output ./cat_generated.png
```

**解决马赛克图问题的完整流程：**

1. **数据准备**：确保 `data/text_to_image_pairs.jsonl` 中有足够的高质量图文对（建议1000+）
2. **完整训练**：使用 GPU 进行 100 轮训练，学习率 0.0001
3. **模型加载**：使用 `models/text_to_image_full` 目录中的完整训练模型
4. **高质量生成**：使用 100 步采样，详细的提示词，GPU 加速

**注意事项：**
- 快速测试训练（1轮）只能验证流程，生成的图像会是马赛克图
- 完整训练（100轮）才能生成有意义的高质量图像
- 训练时间取决于硬件，GPU 通常需要 6-24 小时完成 100 轮训练

### 3. 通过 API 服务器使用多模态功能

API 服务器在启动时会加载 LLM 模型，同时提供 LLM 对话和多模态图像生成服务。

#### 3.1 启动 API 服务器

```bash
cargo run --release --features="api" --bin api_server -- `
    --model-dir ./models/lm_100m `
    --backend gpu `
    --port 8000
```

**参数详解：**

- `--model-dir`：模型目录路径，需要包含 LLM 模型文件（`tokenizer.json` 和 `model.mpk`）
- `--backend gpu`：使用 GPU 后端进行推理，显著提升性能
- `--port 8000`：服务器监听端口

**启动日志示例：**
```
[INFO] api_server: 正在启动API服务器...
[INFO] api_server: 模型目录: ./models/lm_100m
[INFO] api_server: 端口: 8000
[INFO] api_server: 找到tokenizer，加载中...
[INFO] api_server: Tokenizer加载成功
[INFO] api_server: 使用GPU后端进行推理...
[INFO] api_server: 初始化GPU懒加载模型...
[INFO] api_server: API服务器启动在 http://0.0.0.0:8000
```

#### 3.2 加载 Diffusion 模型

在使用图像生成功能前，需要先加载 Diffusion 模型：

```bash
curl -X POST http://localhost:8000/api/v1/diffusion/load `
  -H "Content-Type: application/json" `
  -d '{
    "model_path": "./models/text_to_image_full",
    "config_path": "./configs/config_vae_diffusion.json"
  }'
```

**参数详解：**

- `model_path`：扩散模型的文件路径，API 服务器会在此路径下查找 `diffusion_model.mpk`
- `config_path`：模型配置文件路径，API 服务器从此文件读取模型超参数

#### 3.3 生成图像

加载模型后，可以通过 API 生成图像：

```bash
curl -X POST http://localhost:8000/api/v1/images/generate `
  -H "Content-Type: application/json" `
  -d '{
    "prompt": "一只可爱的小猫，毛茸茸的，蓝色眼睛",
    "steps": 100,
    "image_size": 64
  }'
```

**参数详解：**

- `prompt`：文本提示词，描述想要生成的图像内容
- `steps`：Diffusion 采样步数，步数越多生成质量越高（建议 50-100）
- `image_size`：输出图像尺寸，默认 64

**响应示例：**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "image_path": "./assets/550e8400-e29b-41d4-a716-446655440000.png",
  "message": "Image generated successfully"
}
```

#### 3.4 完整 API 使用流程

```bash
# 1. 启动 API 服务器（需要 LLM 模型文件）
cargo run --release --features="api" --bin api_server -- `
    --model-dir ./models/lm_100m `
    --backend gpu `
    --port 8000

# 2. 在另一个终端中，加载 Diffusion 模型
curl -X POST http://localhost:8000/api/v1/diffusion/load `
  -H "Content-Type: application/json" `
  -d '{
    "model_path": "./models/text_to_image_full",
    "config_path": "./configs/config_vae_diffusion.json"
  }'

# 3. 生成图像
curl -X POST http://localhost:8000/api/v1/images/generate `
  -H "Content-Type: application/json" `
  -d '{
    "prompt": "一只可爱的小猫",
    "steps": 100
  }'
```

**前置条件**：确保 LLM 模型目录（`./models/lm_100m`）包含：
- `tokenizer.json` - 分词器文件
- `model.mpk` - LLM 模型权重文件

确保 Diffusion 模型目录（`./models/text_to_image_full`）包含：
- `config.json` - 模型配置文件
- `diffusion_model.mpk` - 训练好的模型权重文件

### 2. 最小化多模态使用示例

```rust
use burn::prelude::*;
use sage::core::{
    VisionEncoder, VisionEncoderConfig, MultimodalConfig, MultimodalFusion,
};

type Backend = burn::tensor::backend::NdArray;

fn main() {
    let device = burn::tensor::backend::NdArrayDevice::Cpu;
    
    // 1. 创建视觉编码器配置
    let vision_config = VisionEncoderConfig {
        encoder_type: "resnet".to_string(),
        out_dim: 512,
        image_size: 224,
        ..Default::default()
    };
    
    // 2. 初始化视觉编码器
    let vision_encoder = VisionEncoder::new(&vision_config, &device);
    
    // 3. 创建多模态配置
    let multimodal_config = MultimodalConfig {
        vision_dim: 512,
        text_dim: 512,
        fusion: "gated".to_string(),
    };
    
    // 4. 初始化多模态融合层
    let multimodal_fusion = MultimodalFusion::new(&multimodal_config, &device);
    
    println!("✅ 多模态模块初始化成功！");
}
```

---

## 核心功能模块

### 视觉编码器

Sage 支持两种主要的视觉编码器架构，每种都有多个变体。

#### 1. ResNet 视觉编码器

**支持的变体：**
- ResNet18（默认，轻量级）
- ResNet34
- ResNet50
- ResNet101
- ResNet152

**使用示例：**

```rust
use sage::core::{VisionEncoder, VisionEncoderConfig, ResNetVariant};

// ResNet18 配置（快速原型）
let resnet18_config = VisionEncoderConfig {
    encoder_type: "resnet".to_string(),
    num_layers: 4,
    resnet_variant: Some(ResNetVariant::ResNet18),
    ..Default::default()
};

// ResNet50 配置（高质量）
let resnet50_config = VisionEncoderConfig {
    encoder_type: "resnet".to_string(),
    num_layers: 6,
    resnet_variant: Some(ResNetVariant::ResNet50),
    out_dim: 1024,
    ..Default::default()
};
```

**ResNet 变体对比：**

| 变体 | 参数数量 | 适用场景 | 推荐任务 |
|------|---------|---------|---------|
| ResNet18 | 11M | 快速原型、小数据集 | 分类、快速演示 |
| ResNet34 | 21M | 平衡性能与速度 | 通用视觉任务 |
| ResNet50 | 25M | 高精度任务 | 视觉问答、图像描述 |
| ResNet101 | 44M | 复杂视觉理解 | 多模态推理 |
| ResNet152 | 60M | 最高精度 | 研究、复杂场景 |

#### 2. Vision Transformer (ViT) 视觉编码器

**完整 ViT 架构使用：**

```rust
use sage::core::{
    VisionTransformer, VisionEncoderConfig, TransformerEncoderBlock,
    MultiHeadAttention, MLPBlock,
};

// 创建完整的 ViT 配置
let vit_config = VisionEncoderConfig {
    encoder_type: "vit".to_string(),
    in_channels: 3,
    hidden_channels: 768,
    out_dim: 768,
    num_layers: 12,              // 12 个 Transformer 层
    patch_size: 16,               // Patch 大小 16x16
    image_size: 224,              // 图像尺寸 224x224
};

// 初始化 ViT
let vit = VisionTransformer::new(&vit_config, &device);

// 完整的 Transformer 编码器块使用示例
let transformer_block = TransformerEncoderBlock::new(
    768,    // embed_dim
    12,     // num_heads
    3072,   // mlp_dim
    0.1,    // dropout
    &device,
);
```

**ViT 配置调优建议：**

```rust
// 小模型（快速）
let small_vit = VisionEncoderConfig {
    encoder_type: "vit".to_string(),
    hidden_channels: 256,
    num_layers: 4,
    patch_size: 32,
    out_dim: 256,
    ..Default::default()
};

// 基础模型（推荐）
let base_vit = VisionEncoderConfig {
    encoder_type: "vit".to_string(),
    hidden_channels: 768,
    num_layers: 12,
    patch_size: 16,
    out_dim: 768,
    ..Default::default()
};

// 大模型（高精度）
let large_vit = VisionEncoderConfig {
    encoder_type: "vit".to_string(),
    hidden_channels: 1024,
    num_layers: 24,
    patch_size: 14,
    out_dim: 1024,
    ..Default::default()
};
```

---

### 数据增强

Sage 提供完整的数据增强功能，用于训练时提升模型泛化能力。

#### 数据增强配置

```rust
use sage::core::DataAugmentation;

// 创建数据增强器
let data_aug = DataAugmentation::new(device.clone());

// 配置增强选项
let data_aug = DataAugmentation {
    random_crop: true,           // 随机裁剪
    random_flip: true,           // 随机水平翻转
    random_rotation: false,      // 随机旋转（待实现）
    color_jitter: true,          // 颜色抖动
    device,
};
```

#### 数据增强使用示例

```rust
use burn::tensor::Tensor;

// 假设我们有一个图像 batch
let batch_size = 8;
let image_tensor: Tensor<Backend, 4> = Tensor::ones(
    [batch_size, 3, 224, 224], &device
);

// 1. 随机水平翻转
let flipped = data_aug.random_horizontal_flip(image_tensor.clone());

// 2. 颜色抖动（亮度、对比度）
let jittered = data_aug.color_jitter(image_tensor.clone(), 0.2, 0.2);

// 3. 中心裁剪
let cropped = data_aug.center_crop(image_tensor.clone(), 200, 200);

// 4. 随机裁剪
let random_cropped = data_aug.random_crop(image_tensor, 200, 200);

// 完整的数据增强流水线
let augmented = data_aug.apply(image_tensor);
```

#### 训练时数据增强

```rust
use sage::core::{ImagePreprocessor, DataAugmentation};

// 完整的预处理 + 增强流水线
struct TrainingDataPipeline<B: Backend> {
    preprocessor: ImagePreprocessor<B>,
    augmentation: DataAugmentation<B>,
    device: B::Device,
}

impl<B: Backend> TrainingDataPipeline<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            preprocessor: ImagePreprocessor::new(224, device.clone()),
            augmentation: DataAugmentation {
                random_crop: true,
                random_flip: true,
                random_rotation: false,
                color_jitter: true,
                device: device.clone(),
            },
            device: device.clone(),
        }
    }
    
    pub fn process_image(&self, image: Tensor<B, 3>) -> Tensor<B, 4> {
        // 1. 基础预处理
        let preprocessed = self.preprocessor.preprocess_single(image);
        
        // 2. 应用数据增强（仅训练时）
        self.augmentation.apply(preprocessed)
    }
}
```

---

### 预训练权重加载

Sage 支持从文件加载预训练权重，方便迁移学习。

#### 基本配置

```rust
use sage::core::{WeightLoader, PretrainedWeightConfig};

// 创建权重加载器配置
let weight_config = PretrainedWeightConfig {
    weight_path: "./models/vision_encoder_weights.pt".to_string(),
    strict_loading: false,           // 宽松加载，允许缺失键
    ignore_missing_keys: vec![],     // 可忽略的键列表
};

// 初始化权重加载器
let weight_loader = WeightLoader::new(
    device.clone(),
    weight_config.strict_loading,
    weight_config.ignore_missing_keys,
);
```

#### 加载权重到模型

```rust
// 假设我们有一个已初始化的视觉编码器
let mut vision_encoder = VisionEncoder::new(&vision_config, &device);

// 加载预训练权重
match weight_loader.load_pretrained_weights(&mut vision_encoder, &weight_config) {
    Ok(_) => println!("✅ 预训练权重加载成功！"),
    Err(e) => eprintln!("❌ 权重加载失败: {}", e),
}

// 或者使用更简单的方法
let weight_loader = WeightLoader::simple(&device);
weight_loader.load_from_file(&mut vision_encoder, "./weights/resnet50.pt")?;
```

#### 权重管理工具

```rust
use sage::core::WeightLoader;

// 权重过滤示例
let state_dict = load_weights_from_file()?;
let filtered = weight_loader.filter_state_dict_for_model(
    &state_dict,
    &vision_encoder,
);

// 检查键匹配
let (matched, missing, extra) = weight_loader.check_weight_compatibility(
    &state_dict,
    &vision_encoder,
);

println!("匹配键: {}", matched);
println!("缺失键: {}", missing);
println!("额外键: {}", extra);
```

---

### 图像生成

Sage 提供完整的图像生成功能，包括 VAE 自编码器和 Diffusion 扩散模型。

#### 1. VAE 图像自编码器

**VAE 配置：**

```rust
use sage::core::{VAE, VAEConfig};

// 创建 VAE 配置
let vae_config = VAEConfig {
    in_channels: 3,
    latent_dim: 128,
    hidden_channels: 64,
    image_size: 64,
};

// 初始化 VAE
let vae = VAE::new(&vae_config, &device);
```

**VAE 使用示例：**

```rust
// 编码图像到潜在空间
let image: Tensor<Backend, 4> = Tensor::ones([1, 3, 64, 64], &device);
let (mu, logvar) = vae.encode(image.clone());

// 重参数化采样
let z = vae.reparameterize(mu.clone(), logvar.clone());

// 从潜在空间重建图像
let reconstructed = vae.decode(z.clone());

// 完整的前向传播
let (recon, mu, logvar) = vae.forward(image);
```

**训练 VAE：**

```rust
use burn::tensor::backend::Autodiff;

fn train_vae<B: Backend<Autodiff>>(
    vae: &VAE<B::AutodiffBackend>,
    images: Tensor<B::AutodiffBackend, 4>,
) -> Tensor<B::AutodiffBackend, 0> {
    // 前向传播
    let (recon, mu, logvar) = vae.forward(images);
    
    // 重建损失
    let recon_loss = (recon.clone() - images.clone()).powf(2.0).mean();
    
    // KL 散度
    let kl = -0.5 * (1.0 + logvar - mu.powf(2.0) - logvar.exp()).sum_dim(1).mean();
    
    // 总损失
    recon_loss + kl * 0.1
}
```

#### 2. Diffusion 扩散模型

**Diffusion 配置：**

```rust
use sage::core::{DiffusionModel, DiffusionConfig};

// 创建扩散模型配置
let diffusion_config = DiffusionConfig {
    image_size: 64,
    in_channels: 3,
    hidden_channels: 128,
    num_timesteps: 1000,
    latent_dim: 128,
    beta_start: 0.0001,
    beta_end: 0.02,
};

// 初始化扩散模型（包含 VAE）
let diffusion = DiffusionModel::new(&diffusion_config, &device);
```

**扩散模型生成图像：**

```rust
// 从噪声生成图像
let batch_size = 4;
let num_inference_steps = 50;

let generated_images = diffusion.generate(
    Tensor::randn([batch_size, diffusion_config.latent_dim], &device),
    num_inference_steps,
);

// 使用不同的采样步数（步数越多质量越高）
let high_quality = diffusion.generate(latent, 100);  // 高质量
let fast_generation = diffusion.generate(latent, 20);  // 快速生成
```

**扩散过程可视化：**

```rust
// 获取完整的扩散过程
let initial_noise = Tensor::randn([1, 3, 64, 64], &device);

let mut current = initial_noise;
let mut intermediate_images = Vec::new();

for t in (0..num_inference_steps).rev() {
    current = diffusion.sample_step(current, t as i64);
    intermediate_images.push(current.clone());
}
```

#### 3. Text-to-Image (文生图) 流水线

**初始化文生图流水线：**

```rust
use sage::core::{TextToImagePipeline, DiffusionModel, DiffusionConfig, VAE};

// 创建完整的文生图流水线
let text_to_image = TextToImagePipeline {
    diffusion: DiffusionModel::new(&diffusion_config, &device),
    vae: VAE::new(&vae_config, &device),
    config: diffusion_config.clone(),
};
```

**文生图使用示例：**

```rust
// 假设有文本编码器输出的文本嵌入
let text_embedding: Tensor<Backend, 2> = Tensor::ones([1, 512], &device);

// 生成图像
let num_steps = 50;
let generated_image = text_to_image.generate(text_embedding, num_steps);

println!("✅ 文生图完成！图像形状: {:?}", generated_image.dims());
```

**高级文生图配置：**

```rust
// 多步采样，质量更高
let high_quality_image = text_to_image.generate(text_embedding, 100);

// 批量生成
let batch_text_embeddings = Tensor::ones([4, 512], &device);
let batch_images = text_to_image.generate(batch_text_embeddings, 50);
```

---

### 多模态评估

Sage 提供完整的多模态评估功能，用于衡量模型性能。

#### 评估器初始化

```rust
use sage::core::{MultimodalEvaluator, MultimodalMetrics, MetricsLogger};

// 创建评估器
let evaluator = MultimodalEvaluator::new(device.clone());

// 创建指标日志记录器
let mut metrics_logger = MetricsLogger::new();
```

#### 1. 图像描述评估

```rust
// BLEU 分数评估
let generated_caption = "a cat sitting on a couch";
let reference_caption = "a cat sitting on a sofa";

let bleu_score = evaluator.evaluate_image_captioning(
    generated_caption,
    reference_caption,
);

println!("图像描述 BLEU 分数: {:.4}", bleu_score);
```

#### 2. 视觉-语言对齐评估

```rust
// 计算视觉特征和文本特征的对齐度
let vision_features: Tensor<Backend, 2> = Tensor::ones([1, 512], &device);
let text_features: Tensor<Backend, 2> = Tensor::ones([1, 512], &device);

let alignment_score = evaluator.evaluate_vision_language_alignment(
    vision_features,
    text_features,
);

println!("视觉-语言对齐分数: {:.4}", alignment_score);
```

#### 3. 图像生成质量评估

```rust
// 计算生成图像和目标图像的 PSNR
let generated_image: Tensor<Backend, 4> = Tensor::ones([1, 3, 64, 64], &device);
let target_image: Tensor<Backend, 4> = Tensor::ones([1, 3, 64, 64], &device);

let generation_quality = evaluator.evaluate_generation_quality(
    generated_image,
    target_image,
);

println!("生成质量（PSNR）: {:.2} dB", generation_quality);
```

#### 4. 图像相似度计算

```rust
// 计算两张图像的余弦相似度
let similarity = evaluator.compute_image_similarity(
    generated_image,
    target_image,
);

println!("图像相似度: {:.4}", similarity);
```

#### 5. 完整的评估流程

```rust
fn run_complete_evaluation<B: Backend>(
    evaluator: &MultimodalEvaluator<B>,
    generated_captions: &[String],
    reference_captions: &[String],
    generated_images: Tensor<B, 4>,
    real_images: Tensor<B, 4>,
) -> MultimodalMetrics {
    let mut avg_caption_score = 0.0;
    for (gen, ref_c) in generated_captions.iter().zip(reference_captions) {
        avg_caption_score += evaluator.evaluate_image_captioning(gen, ref_c);
    }
    avg_caption_score /= generated_captions.len() as f64;
    
    let alignment_score = 0.75;  // 示例值
    let generation_quality = evaluator.evaluate_generation_quality(
        generated_images.clone(),
        real_images.clone(),
    );
    
    MultimodalMetrics {
        image_captioning_score: avg_caption_score,
        vision_language_alignment: alignment_score,
        generation_quality,
    }
}

// 使用示例
let metrics = run_complete_evaluation(
    &evaluator,
    &generated_captions,
    &reference_captions,
    generated_images,
    real_images,
);

// 记录和打印指标
metrics_logger.log(metrics.clone());
println!("\n{}", metrics_logger.summary());
```

---

## 完整代码示例

### 示例 1：完整的多模态推理流程

```rust
use burn::prelude::*;
use sage::core::{
    VisionEncoder, VisionEncoderConfig, MultimodalFusion, MultimodalConfig,
    ImagePreprocessor, ResNetVariant,
};

type Backend = burn::tensor::backend::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = burn::tensor::backend::NdArrayDevice::Cpu;
    
    // 1. 配置模型
    let vision_config = VisionEncoderConfig {
        encoder_type: "resnet".to_string(),
        resnet_variant: Some(ResNetVariant::ResNet50),
        out_dim: 1024,
        ..Default::default()
    };
    
    let multimodal_config = MultimodalConfig {
        vision_dim: 1024,
        text_dim: 512,
        fusion: "cross_attention".to_string(),
    };
    
    // 2. 初始化模块
    let vision_encoder = VisionEncoder::new(&vision_config, &device);
    let multimodal_fusion = MultimodalFusion::new(&multimodal_config, &device);
    let preprocessor = ImagePreprocessor::new(224, device.clone());
    
    // 3. 加载和处理图像
    let image = load_image("assets/sample_image.jpg")?;
    let preprocessed_image = preprocessor.preprocess_single(image);
    
    // 4. 编码视觉特征
    let vision_features = vision_encoder.forward(preprocessed_image);
    
    // 5. 假设有文本特征
    let text_features: Tensor<Backend, 2> = Tensor::ones([1, 512], &device);
    
    // 6. 多模态融合
    let fused_features = multimodal_fusion.forward(vision_features, text_features);
    
    println!("✅ 多模态推理完成！融合特征形状: {:?}", fused_features.dims());
    
    Ok(())
}

fn load_image(path: &str) -> Result<Tensor<Backend, 3>, Box<dyn std::error::Error>> {
    // 实际应用中使用图像库加载图像
    Ok(Tensor::ones([3, 224, 224], &burn::tensor::backend::NdArrayDevice::Cpu))
}
```

### 示例 2：训练多模态模型

```rust
use burn::tensor::backend::Autodiff;
use sage::core::{
    VisionEncoder, VisionEncoderConfig, MultimodalFusion, MultimodalConfig,
    DataAugmentation, ImagePreprocessor,
};

type AutodiffBackend = burn::tensor::backend::NdArrayAutodiff;

fn train_multimodal_step(
    vision_encoder: &VisionEncoder<AutodiffBackend>,
    multimodal_fusion: &MultimodalFusion<AutodiffBackend>,
    images: Tensor<AutodiffBackend, 4>,
    text_features: Tensor<AutodiffBackend, 2>,
    targets: Tensor<AutodiffBackend, 2>,
) -> Tensor<AutodiffBackend, 0> {
    // 前向传播
    let vision_features = vision_encoder.forward(images);
    let fused = multimodal_fusion.forward(vision_features, text_features);
    
    // 计算损失（示例）
    let loss = (fused - targets).powf(2.0).mean();
    
    loss
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = burn::tensor::backend::NdArrayDevice::Cpu;
    
    // 初始化
    let vision_config = VisionEncoderConfig::default();
    let multimodal_config = MultimodalConfig {
        vision_dim: 512,
        text_dim: 512,
        fusion: "gated".to_string(),
    };
    
    let mut vision_encoder = VisionEncoder::new(&vision_config, &device);
    let mut multimodal_fusion = MultimodalFusion::new(&multimodal_config, &device);
    let data_aug = DataAugmentation::new(device.clone());
    
    // 训练循环
    let num_epochs = 10;
    for epoch in 0..num_epochs {
        // 获取训练数据（示例）
        let images: Tensor<AutodiffBackend, 4> = Tensor::randn([8, 3, 224, 224], &device);
        let text_features: Tensor<AutodiffBackend, 2> = Tensor::randn([8, 512], &device);
        let targets: Tensor<AutodiffBackend, 2> = Tensor::randn([8, 512], &device);
        
        // 应用数据增强
        let augmented = data_aug.apply(images);
        
        // 训练步骤
        let loss = train_multimodal_step(
            &vision_encoder,
            &multimodal_fusion,
            augmented,
            text_features,
            targets,
        );
        
        // 反向传播（使用 Burn 的优化器）
        // let grads = loss.backward();
        // optimizer.step(&mut vision_encoder, &grads);
        
        println!("Epoch {}, Loss: {:.4}", epoch, loss.val());
    }
    
    Ok(())
}
```

### 示例 3：文生图完整流程

```rust
use sage::core::{
    TextToImagePipeline, DiffusionModel, DiffusionConfig,
    VAE, VAEConfig, MultimodalEvaluator,
};

type Backend = burn::tensor::backend::NdArray;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = burn::tensor::backend::NdArrayDevice::Cpu;
    
    // 1. 配置模型
    let vae_config = VAEConfig {
        in_channels: 3,
        latent_dim: 128,
        hidden_channels: 64,
        image_size: 64,
    };
    
    let diffusion_config = DiffusionConfig {
        image_size: 64,
        in_channels: 3,
        hidden_channels: 128,
        num_timesteps: 1000,
        latent_dim: 128,
        beta_start: 0.0001,
        beta_end: 0.02,
    };
    
    // 2. 初始化文生图流水线
    let text_to_image = TextToImagePipeline {
        diffusion: DiffusionModel::new(&diffusion_config, &device),
        vae: VAE::new(&vae_config, &device),
        config: diffusion_config,
    };
    
    // 3. 准备文本嵌入（实际应用中由文本编码器生成）
    let text_embedding: Tensor<Backend, 2> = Tensor::ones([1, 512], &device);
    
    // 4. 生成图像
    println!("🎨 开始生成图像...");
    let generated_image = text_to_image.generate(text_embedding, 50);
    
    println!("✅ 图像生成完成！形状: {:?}", generated_image.dims());
    
    // 5. 评估生成质量（需要参考图像）
    let evaluator = MultimodalEvaluator::new(device.clone());
    let reference_image: Tensor<Backend, 4> = Tensor::ones([1, 3, 64, 64], &device);
    
    let quality = evaluator.evaluate_generation_quality(generated_image, reference_image);
    println!("生成质量（PSNR）: {:.2} dB", quality);
    
    Ok(())
}
```

---

## 高级配置

### 配置文件示例

创建 `configs/multimodal_config.json`：

```json
{
  "vision_encoder": {
    "encoder_type": "resnet",
    "resnet_variant": "ResNet50",
    "in_channels": 3,
    "hidden_channels": 64,
    "out_dim": 1024,
    "num_layers": 6,
    "patch_size": 16,
    "image_size": 224
  },
  "multimodal": {
    "vision_dim": 1024,
    "text_dim": 512,
    "fusion": "cross_attention",
    "hidden_dim": 768
  },
  "data_augmentation": {
    "random_crop": true,
    "random_flip": true,
    "color_jitter": true,
    "random_rotation": false
  },
  "image_generation": {
    "vae": {
      "latent_dim": 128,
      "hidden_channels": 64
    },
    "diffusion": {
      "num_timesteps": 1000,
      "hidden_channels": 128,
      "beta_start": 0.0001,
      "beta_end": 0.02
    }
  }
}
```

### 加载配置

```rust
use serde::Deserialize;
use std::fs::File;
use std::io::Read;

#[derive(Deserialize)]
struct MultimodalSetup {
    vision_encoder: VisionEncoderConfig,
    multimodal: MultimodalConfig,
    data_augmentation: DataAugmentationConfig,
    image_generation: ImageGenerationConfig,
}

fn load_config(path: &str) -> Result<MultimodalSetup, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let config = serde_json::from_str(&content)?;
    Ok(config)
}

// 使用
let config = load_config("configs/multimodal_config.json")?;
```

---

## 常见问题

### Q: 如何选择 ResNet 还是 ViT？

**A:** 
- **ResNet**：快速原型、小数据集、计算资源有限
- **ViT**：大数据集、高精度任务、长距离视觉依赖

### Q: 文生图需要多少采样步数？

**A:** 
- 快速预览：20-30 步
- 标准质量：50 步
- 高质量：100-200 步

### Q: 如何提升生成质量？

**A:**
1. 增加扩散采样步数
2. 使用更大的模型
3. 添加分类器引导（Classifier Guidance）
4. 训练更多 epochs

### Q: 数据增强应该在什么时候使用？

**A:**
- 训练时：使用全部增强
- 验证/测试时：仅使用基础预处理（不使用随机增强）

### Q: 预训练权重不匹配怎么办？

**A:**
1. 使用 `strict_loading = false`
2. 指定 `ignore_missing_keys` 列表
3. 手动调整权重字典

---

## 更新日志

### v1.2 (2026-04-19)
- ✅ 完整的 VAE 和 Diffusion 图像生成
- ✅ 完整的 Vision Transformer 架构
- ✅ ResNet 多种变体支持
- ✅ 数据增强功能
- ✅ 预训练权重加载
- ✅ 多模态评估指标

---

**作者：** Sage 团队  
**最后更新：** 2026-04-25  
**版本：** v1.3
