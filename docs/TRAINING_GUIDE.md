# Sage 训练指南

## 概述

本文档详细介绍 Sage 模型的各种训练方式、语料获取方法以及训练后的调整优化策略。

## 目录

1. [训练阶段说明（正式 / 非正式）](#训练阶段说明正式--非正式)
2. [训练方式](#训练方式)
3. [语料获取](#语料获取)
4. [实际训练流程](#实际训练流程)
5. [训练参数调整](#训练参数调整)
6. [模型评估与优化](#模型评估与优化)
7. [常见问题排查](#常见问题排查)

## 训练阶段说明（正式 / 非正式）

Sage 里存在两类容易混淆的概念：

- **GPU 显存自动探测**：在默认开启时，会针对多组 batch/序列长度执行 **单次** 前向+反向以估算显存，属于预检步骤，通常**不会**出现 epoch 训练日志。  
- **正式训练**：进入完整训练循环后，按 epoch 使用 DataLoader 迭代数据、记录指标、保存 checkpoint。

`--quick-dev` / `--ultra-quick` 仍是 **正式训练**，只是轮次与数据量压到极小，用于冒烟测试。

**完整时间线、日志如何区分、相关参数**（如 `--no-auto-vram`）请单独阅读：**[TRAINING_PHASES.md](TRAINING_PHASES.md)**。

## 训练方式

### 1. 测试训练（快速验证）

用于快速验证代码和环境是否正常工作：

```bash
# 超快速开发模式（约1秒完成）
cargo run --release --bin train -- --ultra-quick --sft-sample --backend gpu --force-tui --force

# 快速开发模式（约5秒完成）
cargo run --release --bin train -- --quick-dev --sft-sample --backend gpu --force-tui --force
```

**参数说明：**
- `--ultra-quick`：超快速模式，1轮训练，极小批量(2)，极高学习率
- `--quick-dev`：快速开发模式，1轮训练，超小批量(4)，超高学习率
- `--sft-sample`：使用内置示例数据

### 2. 正式训练

使用自定义语料进行完整训练：

```bash
# 使用BPE分词器的正式训练
cargo run --release --bin train -- `
    --sft-jsonl data/your_corpus.jsonl `
    --artifact-dir ./models/your_model `
    --use-bpe `
    --bpe-vocab-size 20000 `
    --num-epochs 50 `
    --batch-size 32 `
    --max-seq-len 256 `
    --lr 5e-5 `
    --backend gpu `
    --model-size 30m `
    --force
```

**关键参数：**
- `--sft-jsonl`：训练语料文件路径
- `--artifact-dir`：模型保存目录
- `--use-bpe`：启用BPE分词器
- `--bpe-vocab-size`：BPE词表大小（建议10000-50000）
- `--model-size`：模型大小（10m/30m/100m/1b/3b/671b）

### 3. 增量训练

在现有模型基础上继续训练：

```bash
# 继续训练
cargo run --release --bin train -- `
    --sft-jsonl data/your_corpus.jsonl `
    --artifact-dir ./models/your_model `
    --continue `
    --num-epochs 100 `
    --backend gpu
```

**参数说明：**
- `--continue`：从现有模型继续训练
- `--resume-epoch <epoch>`：从特定轮次恢复训练

### 4. DPO偏好对齐训练

使用偏好对齐数据进行DPO训练：

```bash
# DPO训练
cargo run --release --bin train -- `
    --dpo `
    --dpo-data data/dpo_data.jsonl `
    --artifact-dir ./models/dpo_model `
    --dpo-beta 0.1 `
    --num-epochs 30 `
    --batch-size 16 `
    --backend gpu `
    --force
```

**参数说明：**
- `--dpo`：启用DPO训练模式
- `--dpo-data`：DPO训练数据文件路径
- `--dpo-beta`：DPO损失的beta参数（默认0.1）
- DPO数据格式：包含`prompt`、`chosen`、`rejected`三个字段

### 5. 分布式训练

当前版本支持基础的分布式训练：

```bash
cargo run --release --bin train -- `
    --distributed `
    --devices cpu,gpu:0 `
    --sft-jsonl data/your_corpus.jsonl `
    --backend gpu
```
- **权重同步**：系统会自动在多个设备间同步权重。
- **数据并行**：每个设备加载独立的数据批次。

### 6. LoRA 轻量化微调

支持仅微调低秩矩阵，大幅节省资源：

```bash
cargo run --release --bin train -- `
    --use-lora `
    --lora-rank 8 `
    --lora-alpha 16 `
    --sft-jsonl data/your_corpus.jsonl `
    --artifact-dir ./models/lora_model `
    --backend gpu
```

### 7. 多模态微调 ✅ **完整功能已实现**

Sage 现在支持完整的多模态训练，包括两种视觉编码器和多种融合策略。

```bash
# 使用 ResNet 编码器 + 门控融合
cargo run --release --bin train -- `
    --multimodal `
    --sft-jsonl data/mm_data.jsonl `
    --output-dir ./models/mm_resnet `
    --vision-out-dim 512 `
    --fusion-strategy gated `
    --backend gpu

# 使用 Vision Transformer 编码器 + 跨模态注意力融合
cargo run --release --bin train -- `
    --multimodal `
    --sft-jsonl data/mm_data.jsonl `
    --output-dir ./models/mm_vit `
    --vision-out-dim 512 `
    --fusion-strategy cross_attention `
    --backend gpu
```

**多模态配置选项：**
- `--multimodal`：启用多模态训练
- `--vision-out-dim`：视觉编码器输出维度（默认 512）
- `--fusion-strategy`：融合策略（gated/concatenate/add/cross_attention）

**视觉编码器选择：**
- **ResNet**（默认）：快速、高效、适合初步使用
- **Vision Transformer**：灵活、高质量、适合大数据集（在配置文件中设置）

**多模态数据格式：**
```json
{
    "prompt": "描述这张图片",
    "response": "这是一张美丽的风景照",
    "image_path": "data/images/landscape.jpg"
}
```

**详细文档：** 完整的多模态使用指南请参考 [MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md)。

### 8. 文生图 (Text-to-Image) 训练 ✅ **新功能**

Sage 支持训练文生图模型，使模型能够根据文本提示词生成相应的图像。

#### 8.1 数据准备

文生图训练需要文本-图像对数据，格式如下：

**JSONL 格式：**
```json
{"prompt": "a cat sitting on a chair", "image_path": "data/cat1.jpg"}
{"prompt": "a dog playing in the park", "image_path": "data/dog1.jpg"}
{"prompt": "a beautiful sunset over the ocean", "image_path": "data/sunset1.jpg"}
```

**数据要求：**
- 每个样本包含 `prompt`（文本描述）和 `image_path`（图像路径）
- 图像分辨率建议 64x64 或 128x128
- 建议至少 10,000 对数据以获得良好效果
- 数据多样性：包含多种场景、物体、风格

**自动生成语料**：

如果已有图片目录，可以使用 `gen_data` 工具自动生成语料文件：

```bash
# 假设图片存放在 data/text_to_images/ 目录
# 执行后会扫描该目录下所有图片，生成 data/text_to_image_pairs.jsonl
cargo run --release --bin gen_data -- --image-dir data/text_to_images
```

```bash
# 指定输出文件路径
cargo run --release --bin gen_data -- --image-dir data/text_to_images --text-to-image-data data/my_training_data.jsonl
```

**参数说明：**
- `--image-dir`：图片所在目录（支持递归扫描子目录）
- `--text-to-image-data`：输出文件路径（默认 `data/text_to_image_pairs.jsonl`）

**生成结果示例**：
```json
{"prompt": "", "image_path": "data/text_to_images/cat1.jpg"}
{"prompt": "", "image_path": "data/text_to_images/dog1.jpg"}
{"prompt": "", "image_path": "data/text_to_images/sunset1.jpg"}
```

> **提示**：生成的语料中 `prompt` 字段为空，需要手动补充文本描述，或使用 AI 模型自动生成描述。

#### 8.2 配置文件

创建文生图训练配置文件，建议存放在 `configs/` 目录：

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

#### 8.3 启动训练

```bash
# 文生图模型训练
cargo run --release --bin train -- `
    --text-to-image `
    --image-text-data data/text_image_pairs.jsonl `
    --config-path configs/vae_diffusion.json `
    --output-dir ./models/text_to_image `
    --batch-size 16 `
    --learning-rate 0.0001 `
    --num-epochs 100 `
    --backend gpu
```

**训练参数说明：**
- `--text-to-image`：启用文生图训练模式
- `--image-text-data`：文本-图像对数据文件路径
- `--config-path`：模型配置文件路径
- `--output-dir`：模型保存目录
- `--batch-size`：批次大小（建议 16-32）
- `--learning-rate`：学习率（建议 1e-4）
- `--num-epochs`：训练轮数（建议 50-200）

#### 8.4 训练过程

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

#### 8.5 训练技巧

**数据准备：**
- **数据清洗**：确保文本描述与图像内容匹配
- **数据增强**：对图像进行随机裁剪、翻转、亮度调整
- **文本处理**：标准化文本格式，去除噪声

**训练配置：**
- **批次大小**：根据 GPU 显存调整，建议 16-32
- **学习率**：使用余弦退火调度，初始 1e-4
- **正则化**：添加 Dropout (0.1-0.3) 防止过拟合
- **梯度裁剪**：设置梯度范数阈值 1.0

**加速训练：**
- **混合精度**：启用 FP16 训练
- **分布式训练**：多 GPU 并行训练
- **梯度累积**：当显存不足时使用

#### 8.6 使用训练好的模型

训练完成后，使用训练好的模型进行文生图生成：

```bash
# 使用训练好的模型生成图像
cargo run --bin image_gen -- `
    --model-path ./models/text_to_image `
    --prompt "a cat wearing sunglasses" `
    --steps 50 `
    --output ./generated_cat.png

# GPU 加速生成
cargo run --bin image_gen -- `
    --backend gpu `
    --model-path ./models/text_to_image `
    --prompt "a beautiful landscape with mountains" `
    --steps 100 `
    --output ./generated_landscape.png
```

#### 8.7 评估生成质量

训练过程中，定期生成样本评估质量：

```bash
# 评估生成质量
cargo run --bin image_gen -- `
    --model-path ./models/text_to_image `
    --prompt "a red rose in a vase" `
    --steps 50 `
    --output ./eval/rose_50.png

cargo run --bin image_gen -- `
    --model-path ./models/text_to_image `
    --prompt "a red rose in a vase" `
    --steps 100 `
    --output ./eval/rose_100.png
```

**评估指标：**
- **视觉质量**：图像清晰度、色彩自然度
- **文本相关性**：生成内容与提示词的匹配程度
- **多样性**：不同提示词生成不同风格的图像
- **稳定性**：相同提示词多次生成的一致性

**详细文档：** 完整的图像生成指南请参考 [IMAGE_GENERATION_GUIDE.md](IMAGE_GENERATION_GUIDE.md)。

## 语料获取

### 1. 使用内置生成工具

```bash
# 生成综合语料（包含普通对话、Web 问答、多模态数据）
cargo run --release --bin gen_data -- --out data/corpus.jsonl --count 1000 --web --multimodal --seed 123
```

**参数说明：**
- `--out`：输出文件路径
- `--count`：生成数据数量
- `--web`：包含网络/Web 风格问答
- `--multimodal`：包含多模态图像路径
- `--seed`：随机种子

### 2. 公开数据集

推荐的中文语料来源：

- **中文维基百科**：https://dumps.wikimedia.org/zhwiki/
- **中文新闻语料**：THUCNews、Sogou新闻等
- **GitHub代码库**：各种开源项目的代码和文档
- **Stack Overflow**：技术问答数据
- **Hugging Face数据集**：https://huggingface.co/datasets

### 3. 网络爬虫采集

可以使用Python爬虫从以下网站获取数据：

- **技术博客**：CSDN、知乎、掘金、SegmentFault
- **文档网站**：官方文档、技术文档站点
- **论坛社区**：技术论坛、问答社区

### 4. 数据格式要求

训练数据需要使用JSONL格式，每行一个JSON对象：

```json
# 格式1：prompt-response格式
{"prompt": "问题内容", "response": "回答内容"}

# 格式2：messages格式（推荐）
{"messages": [
    {"role": "user", "content": "问题内容"},
    {"role": "assistant", "content": "回答内容"}
]}
```

## 实际训练流程

### 1. 数据准备阶段

```bash
# 步骤1：生成或准备训练数据
cargo run --release --bin gen_web_sft -- --out data/train_corpus.jsonl --count 10000 --web --seed 123

# 步骤2：验证数据格式（Windows PowerShell）
Get-Content data/train_corpus.jsonl | Select-Object -First 3
# 或者更简单的方式
Get-Content data/train_corpus.jsonl -Head 3
```

**验证输出示例：**
```json
{"domain":"人工智能","id":0,"messages":[{"content":"关于人工智能，什么是机器学习？","role":"user"},{"content":"机器学习是人工智能的一个分支...","role":"assistant"}]}
{"domain":"深度学习","id":1,"messages":[{"content":"什么是深度学习？","role":"user"},{"content":"深度学习是机器学习的一个分支...","role":"assistant"}]}
{"domain":"神经网络","id":2,"messages":[{"content":"什么是神经网络？","role":"user"},{"content":"神经网络是一种计算模型...","role":"assistant"}]}
```

**验证要点：**
- 每行必须是有效的JSON格式
- 必须包含`messages`字段或`prompt`/`response`字段
- 确保没有语法错误或格式问题

### 2. 模型训练阶段

```bash
# 步骤1：创建BPE分词器并开始训练
cargo run --release --bin train -- `
    --sft-jsonl data/train_corpus.jsonl `
    --artifact-dir ./models/large_model `
    --use-bpe `
    --bpe-vocab-size 30000 `
    --num-epochs 50 `
    --batch-size 32 `
    --max-seq-len 256 `
    --lr 5e-5 `
    --backend gpu `
    --model-size 30m `
    --force `
    --force-tui

# 步骤2：监控训练进度
# 观察TUI界面的损失曲线和学习率变化
```

### 3. 模型评估阶段

```bash
# 使用训练好的模型进行推理测试
cargo run --release --bin infer -- `
    --model-dir ./models/large_model `
    --use-best `
    --prompt "什么是深度学习？" `
    --num-tokens 100 `
    --temperature 0.7
```

## 训练参数调整

### 1. 学习率调整与学习率调度器（推荐）

#### 1.1 学习率调度器（推荐使用）

项目实现了 **Cosine Annealing + Warmup** 学习率调度器，能显著提升训练稳定性和收敛效果。

**推荐设置：**
- **启用调度器**：`--lr-scheduler`
- **学习率最大值（lr-max）**：5e-5 ~ 5e-4（GPU）
  - 小模型（1M/10M）：0.0005 ~ 0.001
  - 中等模型（30M/100M）：0.0003 ~ 0.0005
- **学习率最小值（lr-min）**：1e-5 ~ 5e-6
- **Warmup步数**：总步数的 5%-10%
- **总调度步数**：根据训练轮数和批次计算

**调度阶段：**
1. **Warmup阶段**（前 warmup-steps）：学习率从 0 线性增加到 lr-max
2. **Cosine阶段**（之后）：学习率从 lr-max 余弦衰减到 lr-min

**使用示例：**
```bash
# 基础学习率调度器训练
cargo run --release --bin train -- --sft-jsonl sft_demo_5000.jsonl --artifact-dir ./models/sft_lr_scheduler --lr-scheduler --lr-max 0.0005 --lr-min 0.00001 --warmup-steps 500 --total-steps 10000 --use-bpe --num-epochs 50 --backend gpu

# 学习率调度器 + 大模型 + GPU
cargo run --release --bin train -- --sft-jsonl sft_demo_5000.jsonl --artifact-dir ./models/sft_lr_scheduler_large --lr-scheduler --lr-max 0.0003 --lr-min 0.000005 --warmup-steps 1000 --total-steps 50000 --use-bpe --bpe-vocab-size 10000 --model-size 30m --num-epochs 100 --batch-size 16 --max-seq-len 512 --backend gpu
```

#### 1.2 固定学习率（不推荐）

如果不使用学习率调度器，可以使用固定学习率：

**推荐设置：**
- **小模型（&lt;10M）**：1e-4 ~ 5e-4（GPU）
- **中等模型（10M-100M）**：5e-5 ~ 2e-4

**调整策略：**
- **训练不稳定**：降低学习率（如从1e-4降至5e-5）
- **收敛缓慢**：适当提高学习率
- **损失波动大**：降低学习率并增加批量大小

```bash
# 固定学习率示例
--lr 3e-5
```

### 2. 批量大小调整

**根据GPU内存调整：**
- **8GB GPU**：batch-size = 8 ~ 16
- **16GB GPU**：batch-size = 16 ~ 32
- **24GB+ GPU**：batch-size = 32 ~ 64
- **CPU训练**：batch-size = 4 ~ 16（根据CPU核心数调整）

**批量大小影响：**
- **较大批量**：训练更稳定，梯度估计更准确，但占用更多内存
- **较小批量**：训练更快（每次迭代时间短），但梯度噪声大
- **批量大小选择**：在GPU内存允许的情况下，尽量使用较大的批量

**自动优化：**
- 默认开启 GPU 显存探测（除非 `--no-auto-vram`）：会尝试一组 (batch, seq_len) 的“一步前向+反向”，自动寻找不 OOM 的配置。
- 探测成功后可能会**降低**物理 `--batch-size` 或 `--max-seq-len`，并自动设置 `--gradient-accumulation` 以尽量保持等效 batch。

```bash
# 调整批量大小
--batch-size 16

# GPU 模式下推荐做法：
# - 默认开启显存探测：可能把物理 batch/seq_len 调小，并自动设置梯度累积以保持等效 batch
# - 如需完全手动：加 --no-auto-vram，并自行设置 --gradient-accumulation
--backend gpu --no-auto-vram --batch-size 2 --gradient-accumulation 4  # 等效 batch ≈ 8
```

### 3. 序列长度调整

**根据语料长度调整：**
- **短文本**（问答对）：max-seq-len = 128
- **中等文本**（段落对话）：max-seq-len = 256
- **长文本**（完整对话）：max-seq-len = 512
- **超长文本**：max-seq-len = 1024（需要足够的GPU内存）

**序列长度影响：**
- **较长序列**：可以处理更长的上下文，但占用更多内存
- **较短序列**：训练更快，但可能丢失上下文信息
- **序列长度选择**：根据实际语料的平均长度选择合适的值

**内存消耗：**
- 序列长度每增加一倍，内存消耗大约增加一倍
- 例如：max-seq-len=512 比 max-seq-len=256 占用约两倍内存

```bash
# 调整序列长度
--max-seq-len 256

# 根据语料长度选择
--max-seq-len 128  # 短问答
--max-seq-len 512  # 长对话
```

### 4. 训练轮次调整

**推荐设置：**
- **小数据集**（<1000条）：30 ~ 50轮
- **中等数据集**（1000-10000条）：50 ~ 100轮
- **大数据集**（>10000条）：100 ~ 200轮
- **超大数据集**（>100000条）：200 ~ 500轮

**训练轮次判断：**
- **观察损失曲线**：当训练损失和验证损失趋于稳定时可以停止
- **早期停止**：如果验证损失连续多轮不再下降，可以提前停止
- **过拟合判断**：如果训练损失继续下降但验证损失开始上升，说明过拟合

**增量训练：**
- 可以使用`--continue`参数继续训练
- 每次可以增加50-100轮进行增量训练

```bash
# 调整训练轮次
--num-epochs 100

# 增量训练示例
--continue --num-epochs 200  # 在原有基础上继续训练100轮
```

**训练时间估计：**
- 训练时间 = 每轮时间 × 轮次数
- 可以通过前几轮的时间估算总训练时间
- 例如：每轮5分钟，100轮大约需要8小时

## 模型评估与优化

### 1. 评估指标

#### 1.1 Perplexity（困惑度）

Perplexity 是衡量语言模型质量的重要指标，值越低越好。

**计算公式：**
```
Perplexity = exp(Loss)
```

**解读：**
- **Perplexity &lt; 10**：非常优秀（接近人类水平）
- **Perplexity 10-20**：良好（高质量模型）
- **Perplexity 20-50**：一般（可接受）
- **Perplexity &gt; 50**：较差（需要改进）

**用途：**
- 监控训练过程中模型质量的变化
- 比较不同模型或超参数的性能
- 判断模型是否收敛

**Perplexity 计算位置：**
- 项目在 `src/utils/metrics.rs` 中实现了 Perplexity 计算函数
- 支持从单个损失值或多个损失值计算平均 Perplexity

#### 1.2 BLEU 分数

BLEU（Bilingual Evaluation Understudy）用于评估文本生成质量，比较生成文本与参考文本的相似度。

**范围：** 0.0 ~ 1.0
- **BLEU &gt; 0.5**：非常优秀
- **BLEU 0.3-0.5**：良好
- **BLEU 0.1-0.3**：一般
- **BLEU &lt; 0.1**：较差

**用途：**
- 自动评估文本生成质量
- 比较不同生成策略的效果
- 量化评估模型的文本生成能力

**BLEU 计算位置：**
- 项目在 `src/utils/metrics.rs` 中实现了简化版 BLEU 分数计算
- 支持 n-gram 精度计算和简洁惩罚

#### 1.3 其他评估指标

**损失值（Loss）**：训练和验证损失，反映模型拟合程度
- **Loss下降**：模型正在学习和拟合数据
- **训练Loss远低于验证Loss**：可能存在过拟合
- **Loss不再下降**：模型可能已收敛

**生成质量**：人工评估生成内容的质量、相关性和连贯性
- 这是最可靠的评估方式，但耗时耗力
- 建议定期进行人工评估

**ROUGE分数**：评估摘要质量（可选）
- 用于评估摘要、翻译等任务
- 关注召回率而非精确率

### 2. 评估方法

**自动化评估：**
```bash
# 使用验证集评估
cargo run --release --bin train -- `
    --sft-jsonl data/train_corpus.jsonl `
    --artifact-dir ./models/your_model `
    --continue `
    --num-epochs 1 `
    --backend gpu
```

**人工评估：**
```bash
# 使用训练好的模型进行交互式评估
cargo run --release --bin infer -- `
    --model-dir ./models/your_model `
    --use-best `
    --chat `
    --interactive
```

**批量评估脚本：**
可以编写脚本批量测试模型在测试集上的表现，记录生成结果和评分。

### 3. 优化策略

**常见优化方法：**

#### 1. 数据优化
- **数据增强**：
  - 添加更多样化的训练数据
  - 对现有数据进行改写和扩充
  - 使用数据扩充技术（同义词替换、句式变换等）
  
- **数据清洗**：
  - 去除低质量、重复的训练样本
  - 过滤噪声和无关内容
  - 确保数据质量和一致性

#### 2. 模型调优
- **模型结构优化**：
  - 尝试不同的模型大小（10m vs 30m）
  - 调整层数、注意力头数等超参数
  - 尝试不同的激活函数和归一化方法
  
- **超参数优化**：
  - 使用网格搜索或随机搜索优化超参数
  - 关注学习率、批量大小、dropout率等关键参数

#### 3. 正则化技术
- **防止过拟合**：
  - 使用dropout层（当前已设置为0.1）
  - 添加权重衰减（weight decay）
  - 使用早停策略（early stopping）
  
- **数据增强正则化**：
  - 使用数据扩充减少过拟合
  - 添加噪声和扰动增加模型鲁棒性

#### 4. 学习率优化
- **学习率调度**：
  - 使用学习率预热（warmup）
  - 实现学习率衰减（cosine/linear decay）
  - 自适应学习率调整

#### 5. 训练技巧
- **混合精度训练**：使用FP16加速训练（未来支持）
- **梯度累积**：允许使用更大的等效批量大小
- **梯度裁剪**：防止梯度爆炸

#### 6. 评估和迭代
- **定期评估**：在验证集上评估模型性能
- **模型选择**：保存最佳模型（使用`--use-best`）
- **持续迭代**：基于评估结果调整训练策略

### 4. 过拟合处理

**过拟合迹象：**
- **训练损失持续下降，但验证损失开始上升**
- **生成内容重复或模式化**
- **模型在训练集上表现很好，但在新数据上表现差**
- **困惑度在训练集上很低，但在验证集上很高**

**解决方案：**

#### 1. 数据层面
- **增加数据量**：收集更多训练数据
- **数据增强**：对现有数据进行扩充和变换
- **数据清洗**：去除异常和低质量样本

#### 2. 模型层面
- **降低模型复杂度**：
  - 使用更小的模型（10m vs 30m）
  - 减少层数或隐藏层大小
  - 使用更简单的模型结构

- **增强正则化**：
  - 增加dropout率（当前为0.1，可以尝试0.2-0.5）
  - 添加权重衰减（L2正则化）
  - 使用早停策略（early stopping）

#### 3. 训练策略
- **提前停止**：当验证损失不再下降时停止训练
- **减少训练轮次**：避免训练过度
- **调整学习率**：使用更小的学习率

#### 4. 评估和监控
- **定期评估**：在验证集上评估模型性能
- **监控指标**：同时关注训练和验证损失
- **保存最佳模型**：使用验证集性能选择最佳模型

**实用建议：**
- 先尝试增加数据量和数据增强
- 如果仍然过拟合，再考虑降低模型复杂度
- 使用早停策略自动选择最佳训练轮次

## 常见问题排查

### 1. UTF-8字符边界错误

**错误信息：**
```
byte index XX is not a char boundary; it is inside 'X' (bytes XX..XX)
```

**问题原因：**
- BPE分词器返回的偏移量是字节索引
- 中文字符占用多个字节，切片时可能跨越字符边界

**解决方案：**
- 已在代码中修复，使用安全的字符串匹配方法
- 如果仍然遇到此错误，请更新到最新版本

### 2. GPU内存不足

**错误信息：**
```
Out of memory
```

**问题原因：**
- 批量大小过大
- 序列长度过长
- 模型过大

**解决方案：**
- **减小批量大小**：`--batch-size 8`
- **减小序列长度**：`--max-seq-len 128`
- **使用更小的模型**：`--model-size 10m`（或更大模型：`--model-size 30m/100m/1b/3b/671b`）
- **调整学习率**：可能需要同时调整学习率

### 3. 训练速度慢

**问题表现：**
- 每轮训练时间过长
- GPU利用率低

**优化建议：**
- **使用GPU后端**：`--backend gpu`
- **CPU 后端增加工作线程数**：`--num-workers 16`（GPU 后端会强制使用单线程数据加载，忽略该参数）
- **使用BPE分词器**：`--use-bpe`（通常减少 token 数并改善重复问题；首次构建 BPE tokenizer 可能更慢）
- **调整批处理大小**：找到GPU内存允许的最大批量
- **检查GPU驱动**：确保使用最新的GPU驱动

### 4. TUI不显示

**问题表现：**
- 训练时没有显示进度界面
- 只显示文本输出

**解决方案：**
- **强制启用TUI**：`--force-tui`
- **确保终端支持**：使用支持ANSI颜色的终端
- **检查环境变量**：代码会自动设置必要的环境变量
- **Windows终端**：推荐使用Windows Terminal或PowerShell

### 5. 模型不收敛

**问题表现：**
- 损失值不下降或波动很大
- 生成内容质量差

**解决方案：**
- **调整学习率**：尝试不同的学习率（如5e-5, 1e-4）
- **检查数据质量**：确保训练数据格式正确
- **增加训练轮次**：可能需要更多轮次才能收敛
- **检查模型配置**：确认模型参数设置正确

### 6. JSON解析错误

**错误信息：**
```
Failed to parse line: ...
```

**问题原因：**
- JSON格式错误
- 缺少必要字段
- 编码问题

**解决方案：**
- **验证数据格式**：使用`Get-Content data/file.jsonl -Head 3`检查
- **修复JSON格式**：确保每行都是有效的JSON
- **检查编码**：确保文件使用UTF-8编码

### 7. 分词器构建失败

**错误信息：**
```
Failed to build BPE tokenizer
```

**问题原因：**
- 语料数据太少
- 语料格式不正确

**解决方案：**
- **增加语料量**：至少需要几百条数据
- **检查数据格式**：确保数据包含文本内容
- **使用预训练分词器**：如果无法构建新分词器

### 8. 模型加载失败

**错误信息：**
```
Failed to load model
```

**问题原因：**
- 模型文件损坏
- 模型配置不匹配

**解决方案：**
- **重新训练**：使用`--force`参数重新训练
- **检查模型文件**：确保模型文件完整
- **验证配置**：确认模型配置与代码兼容

## 总结

通过本文档的指导，您将能够：

### 🎯 核心能力
1. **掌握多种训练方式**：从快速验证到正式训练，灵活应对不同场景
2. **获取高质量语料**：使用内置工具或外部数据源准备训练数据
3. **进行有效训练**：合理设置参数，监控训练进度，优化训练效果
4. **解决常见问题**：快速排查和解决训练过程中的各种问题
5. **评估和优化模型**：使用各种指标评估模型性能并进行优化

### 🚀 实践建议
- **循序渐进**：从简单的测试训练开始，逐步过渡到正式训练
- **数据为王**：高质量的训练数据是模型性能的关键
- **参数调优**：耐心调整超参数，找到最佳配置
- **持续监控**：关注训练过程，及时发现和解决问题
- **定期评估**：使用验证集和人工评估确保模型质量

### 💡 已实现功能（v1.1）

#### 训练功能增强
- **多规模模型**：支持1M/10M/30M参数规模
- **多种训练模式**：通用对话、代码生成、数学推理
- **快速开发模式**：超快速验证和快速开发模式
- **流式数据加载**：支持大语料流式训练
- **完整多模态功能**：支持 ResNet 和 Vision Transformer (ViT) 两种视觉编码器，四种融合策略（gated、concatenate、add、cross_attention），完整的端到端训练与推理闭环

#### 分词器优化
- **BPE分词器**：支持字节对编码分词
- **字符级分词**：支持中文字符级分词
- **动态词表**：根据训练数据自动构建词表

#### 训练优化
- **GPU加速**：WGPU后端支持
- **学习率调度**：预热和衰减策略
- **显存探测**：GPU 默认会探测可用的 (batch, seq_len) 并自动设置梯度累积（可用 `--no-auto-vram` 关闭）
- **数据加载线程**：CPU 可多线程；GPU(WGPU) 会强制使用单线程数据加载（num_workers=0）

### 📚 资源推荐
- **Burn框架文档**：https://burn-rs.github.io/
- **Rust深度学习社区**：https://github.com/burn-rs/burn
- **中文NLP资源**：各种开源中文语料库和工具

---

**更新日期：** 2026-04-19  
**版本：** v1.2  
**作者：** Sage团队
