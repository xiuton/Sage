# Sage 命令手册

本文档基于当前仓库代码与 `--help` 输出整理，覆盖所有已支持的命令与参数，并提供推荐用法。

二进制列表：

- `train`：训练
- `infer`：推理/对话
- `api_server`：API 服务器（模型管理、推理服务）
- `gen_data`：综合数据生成工具（SFT/Web/多模态）
- `accuracy_eval`：模型准确率评估工具
- `benchmark`：性能基准测试工具
- `export`：模型导出工具
- `convert`：权重转换工具
- `image_gen`：图像生成工具（VAE/Diffusion/文生图）

相关文档：

- 数据格式：`DATA_FORMAT.md`
- 排错：`TROUBLESHOOTING.md`
- 路线图：`PROJECT_STATUS.md`
- 多模态详细文档：[MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md)
- 多模态使用指南：[MULTIMODAL_USAGE.md](MULTIMODAL_USAGE.md)

---

## 1) train（训练）

运行：

```bash
cargo run --release --bin train -- [OPTIONS]
```

### 1.1 训练模式选择（互斥建议）

训练输入可以来自三类来源（建议只选其一）：

- **纯文本语料 LM 训练（预训练）**
  - `--corpus <path>`：单文件（默认 `corpus_cn.txt`）
  - `--corpus-dir <dir>`：目录（递归读取所有 `.txt`）
- **SFT 训练（指令微调）**
  - `--sft-jsonl <path>`：每行一条 JSON
  - 支持两种 schema：
    - `{"prompt":"...","response":"..."}`
    - `{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`
- **内置样例（用于快速跑通）**
  - `--sft-sample`：prompt/response 样例
  - `--sft-sample-messages`：messages 样例

优先级（当你同时传多个输入参数时）：

1. `--sft-jsonl`
2. `--sft-sample-messages`
3. `--sft-sample`
4. `--corpus-dir`
5. `--corpus`（默认 `corpus_cn.txt`）

### 1.2 常用示例

> 说明：当前 `train` 实际参数以代码为准，核心输出参数是 `--output-dir`，学习率参数是 `--learning-rate`，模型规格通过 `--config-path` 指定；旧文档里的 `--artifact-dir`、`--lr`、`--model-size` 属历史写法。

**A. 目录语料训练（续写 LM - 预训练）**

```bash
cargo run --release --bin train -- --corpus-dir D:\data\cn_texts --output-dir ./models/lm_cn --num-epochs 5 --max-seq-len 64
```

**A4. 分布式训练（权重同步）**

```bash
cargo run --release --bin train -- --distributed --devices gpu:0,gpu:1 --sft-jsonl data.jsonl
```
当前实现了基础的分布式权重平均与同步逻辑，支持多设备协同训练。

**A5. LoRA 轻量化微调**

```bash
cargo run --release --bin train -- --use-lora --lora-rank 8 --lora-alpha 16 --sft-jsonl data.jsonl --output-dir ./models/lora_model
```
LoRA 模式下仅训练低秩矩阵，可大幅降低显存占用。

**A6. DPO偏好对齐训练**

```bash
cargo run --release --bin train -- --dpo --dpo-data dpo_data.jsonl --output-dir ./models/dpo_model --dpo-beta 0.1 --dpo-kl-weight 0.1 --num-epochs 30 --batch-size 16 --backend gpu --force
```

**A2. 目录语料训练（限制读取大小 + 快速验证 - 预训练）**

```bash
cargo run --release --bin train -- --corpus-dir D:\data\cn_texts --output-dir ./models/lm_cn_quick --num-epochs 1 --max-seq-len 64 --max-bytes 10000000 --force --reset-tokenizer
```

**A3. 大规模预训练（使用 GPU 和流式处理）**

```bash
cargo run --release --bin train -- --corpus-dir ./corpus --max-bytes 1000000000 --stream --backend gpu --config-path ./inference/configs/config_1B.json --output-dir ./models/large_pretrained
```

**A4. 基本预训练（使用 CPU）**

```bash
cargo run --release --bin train -- --corpus-dir ./corpus --output-dir ./models/lm_basic --num-epochs 1 --max-seq-len 64 --batch-size 4
```

---

## 2) infer（推理/对话）

运行：

```bash
cargo run --bin infer -- [OPTIONS]
```

### 2.1 基础推理模式

**A. 基础文本生成**

```bash
cargo run --bin infer -- --prompt "你好，请介绍一下自己" --num-tokens 100
```

**B. 使用特定模型**

```bash
cargo run --bin infer -- --model-dir ./models/sage_model_formal --use-best --prompt "写一首关于春天的诗"
```

**C. GPU 加速推理**

```bash
cargo run --bin infer -- --model-dir ./models/sage_model_formal --use-best --prompt "解释量子计算" --backend gpu
```

### 2.2 交互式对话

**A. 交互模式**

```bash
cargo run --bin infer -- --model-dir ./models/sage_model_formal --use-best --interactive
```

**B. 终端模式（推荐）**

```bash
cargo run --bin infer -- --model-dir ./models/sage_model_formal --use-best --terminal
```

**C. Chat 模式**

```bash
cargo run --bin infer -- --model-dir ./models/sage_model_formal --use-best --chat --prompt "你好"
```

### 2.3 采样参数调优

**A. 低温度（确定性输出）**

```bash
cargo run --bin infer -- --prompt "1+1等于几" --temperature 0.1 --top-p 0.9
```

**B. 高温度（创造性输出）**

```bash
cargo run --bin infer -- --prompt "写一个科幻故事开头" --temperature 1.2 --top-p 0.95
```

**C. 避免重复**

```bash
cargo run --bin infer -- --prompt "详细描述..." --repetition-penalty 1.5 --punctuation-penalty 1.8
```

---

## 3) image_gen（图像生成）⭐ 新功能

运行：

```bash
cargo run --bin image_gen -- [OPTIONS]
```

### 3.1 VAE 直接生成模式（快速测试）

使用 VAE 模型直接生成图像，无需 Diffusion 采样过程，适合快速测试：

**A. 基础 VAE 生成**

```bash
cargo run --bin image_gen -- --generate-only --image-size 64 --latent-dim 128
```

生成的图像会自动保存到 `assets/` 目录，文件名使用哈希值确保唯一，例如：
- `image_050b88e85af48aa0.png`

**B. 指定输出路径**

```bash
cargo run --bin image_gen -- --generate-only --output ./my_image.png --image-size 64
```

### 3.2 Diffusion 完整生成模式

使用完整的 Diffusion 模型进行去噪采样，生成质量更高：

**A. 基础 Diffusion 生成**

```bash
cargo run --bin image_gen -- --image-size 64 --latent-dim 128 --steps 10
```

**B. 高质量生成（更多采样步数）**

```bash
cargo run --bin image_gen -- --image-size 64 --latent-dim 128 --steps 50
```

### 3.3 文生图模式（Text-to-Image）⭐

使用文本提示词生成对应图像，支持 GPU 加速：

**A. 基础文生图**

```bash
cargo run --bin image_gen -- --prompt "a beautiful landscape" --image-size 64 --steps 20
```

**B. GPU 加速文生图**

```bash
cargo run --bin image_gen -- --backend gpu --prompt "a sunset over the ocean" --image-size 64 --steps 30
```

**C. 自定义主题生成**

```bash
cargo run --bin image_gen -- --backend gpu --prompt "a red rose in the rain" --image-size 64 --steps 50 --output ./my_rose.png
```

**D. 使用训练好的模型**

```bash
# 使用训练好的模型生成图像
cargo run --bin image_gen -- `
    --backend cpu `
    --model-path models/text_to_image `
    --prompt "一只可爱的小猫" `
    --steps 50
```

**E. GPU 加速模型推理**

```bash
# GPU 加速模型推理
cargo run --bin image_gen -- `
    --backend gpu `
    --model-path models/text_to_image `
    --prompt "a beautiful landscape with mountains" `
    --steps 100 `
    --output ./generated_landscape.png
```

### 3.4 图像生成参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--generate-only` | false | VAE 直接生成模式（跳过 Diffusion） |
| `--output` | ./assets/generated_image.png | 输出图像路径 |
| `--steps` | 50 | Diffusion 采样步数 |
| `--image-size` | 64 | 输出图像尺寸（宽=高） |
| `--latent-dim` | 128 | 潜在空间维度 |
| `--backend` | cpu | 后端类型：`cpu` 或 `gpu` |
| `--prompt` | "a beautiful landscape" | 文生图提示词 |
| `--seed` | None | 随机种子（可选） |

### 3.5 采样步数建议

| 用途 | 推荐步数 | 生成时间 | 质量 |
|------|----------|----------|------|
| 快速预览 | 5-10 | ~1秒 | 中等 |
| 标准质量 | 20-50 | ~3秒 | 良好 |
| 高质量 | 50-100 | ~5秒 | 优秀 |

### 3.6 完整示例

```bash
# 示例 1：VAE 快速生成（CPU）
cargo run --bin image_gen -- --generate-only --image-size 64 --latent-dim 128

# 示例 2：VAE 快速生成（GPU 加速）
cargo run --bin image_gen -- --backend gpu --generate-only --image-size 64 --latent-dim 128

# 示例 3：Diffusion 标准生成（GPU 加速）
cargo run --bin image_gen -- --backend gpu --image-size 64 --latent-dim 128 --steps 20

# 示例 4：高质量 Diffusion 生成（GPU 加速）
cargo run --bin image_gen -- --backend gpu --image-size 64 --latent-dim 128 --steps 50

# 示例 5：文生图 - 风景画
cargo run --bin image_gen -- --backend gpu --prompt "a beautiful landscape with mountains and rivers" --image-size 64 --steps 30

# 示例 6：文生图 - 动物
cargo run --bin image_gen -- --backend gpu --prompt "a white cat sitting on a windowsill" --image-size 64 --steps 50

# 示例 7：文生图 - 自定义输出路径
cargo run --bin image_gen -- --backend gpu --prompt "a starry night sky" --output ./my_night_sky.png --steps 30
```

> **提示**：不同的提示词会产生不同的图像特征。提示词会被编码后作为扩散模型的 Conditioning 信息，引导生成过程。

**详细文档：** 更多图像生成功能说明请参考 [MULTIMODAL_USAGE.md](MULTIMODAL_USAGE.md)

---

## 4) gen_data（数据生成）

运行：

```bash
cargo run --release --bin gen_data -- [OPTIONS]
```

### 4.1 生成 SFT 训练数据

```bash
cargo run --release --bin gen_data -- --out data/sft_demo.jsonl --count 1000
```

### 4.2 生成多模态数据

```bash
cargo run --release --bin gen_data -- --out data/multimodal_data.jsonl --count 500 --multimodal --image-dir ./images
```

### 4.3 生成 Web 问答数据

```bash
cargo run --release --bin gen_data -- --out data/web_qa.jsonl --count 200 --web
```

---

## 5) api_server（API 服务器）

运行：

```bash
cargo run --release --bin api_server -- [OPTIONS]
```

### 5.1 启动 API 服务器

```bash
cargo run --release --bin api_server -- --port 8080 --model-dir ./models/sage_model_formal
```

### 5.2 API 调用示例

```bash
# 文本生成
curl -X POST http://localhost:8080/generate `
  -H "Content-Type: application/json" `
  -d '{"prompt": "你好", "max_tokens": 100}'

# 多模态推理
curl -X POST http://localhost:8080/multimodal `
  -H "Content-Type: application/json" `
  -d '{"prompt": "描述这张图片", "image_path": "./test.jpg"}'
```

---

## 多模态训练与推理

### 多模态训练示例

#### 基础多模态训练（CPU）

**参数详解：**

- `--bin train`：指定运行训练二进制文件
- `--multimodal`：启用多模态训练模式，支持图像和文本的联合训练
- `--sft-jsonl data/mm_test.jsonl`：指定SFT训练数据文件路径，数据格式为JSONL，每行包含图像路径和文本描述
- `--output-dir models/mm_model`：指定模型输出目录，训练完成的模型权重和配置将保存在此目录
- `--vision-out-dim 512`：指定视觉编码器的输出特征维度，维度越高表达能力越强，但计算量越大
- `--fusion-strategy cross_attention`：指定多模态融合策略，cross_attention表示使用跨模态注意力机制进行特征融合
- `--batch-size 2`：指定训练批次大小，表示每次训练使用的样本数量，批次越大训练越快但占用显存越多
- `--learning-rate 0.0001`：指定学习率，控制模型权重更新的幅度，学习率过大会导致训练不稳定，过小会导致收敛速度慢
- `--num-epochs 1`：指定训练轮数，表示整个训练数据集被使用的次数，1轮仅用于快速测试
- `--backend cpu`：指定计算后端为CPU，如果需要使用GPU加速训练可改为`gpu`

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

#### GPU 加速多模态训练

**参数详解：**

- `--bin train`：指定运行训练二进制文件
- `--multimodal`：启用多模态训练模式
- `--sft-jsonl data/mm_test.jsonl`：指定SFT训练数据文件路径
- `--output-dir models/mm_model`：指定模型输出目录
- `--vision-out-dim 512`：视觉编码器输出特征维度为512
- `--fusion-strategy cross_attention`：使用跨模态注意力机制进行特征融合，适合复杂的多模态理解任务
- `--batch-size 4`：训练批次大小为4，相比CPU训练可以设置更大的批次
- `--learning-rate 0.0001`：学习率设置为0.0001，这是深度学习训练中常用的学习率
- `--num-epochs 50`：训练50轮，足够让模型学习到数据中的模式和特征
- `--backend gpu`：指定计算后端为GPU，使用显卡加速训练，大幅提升训练速度

**完整命令：**

```bash
cargo run --bin train -- `
    --multimodal `
    --sft-jsonl data/mm_test.jsonl `
    --output-dir models/mm_model `
    --vision-out-dim 512 `
    --fusion-strategy cross_attention `
    --batch-size 4 `
    --learning-rate 0.0001 `
    --num-epochs 50 `
    --backend gpu
```

#### ResNet 编码器训练

**参数详解：**

- `--bin train`：指定运行训练二进制文件
- `--multimodal`：启用多模态训练模式
- `--sft-jsonl data/mm_test.jsonl`：指定SFT训练数据文件路径
- `--output-dir models/mm_resnet`：指定使用ResNet编码器的模型输出目录
- `--vision-out-dim 512`：视觉编码器输出特征维度为512
- `--fusion-strategy gated`：使用门控融合策略，通过可学习的门控机制自适应控制视觉和文本特征的融合权重，计算效率高
- `--backend cpu`：使用CPU进行训练，适合快速验证和调试

**完整命令：**

```bash
cargo run --bin train -- `
    --multimodal `
    --sft-jsonl data/mm_test.jsonl `
    --output-dir models/mm_resnet `
    --vision-out-dim 512 `
    --fusion-strategy gated `
    --backend cpu
```

#### Vision Transformer 编码器训练

**参数详解：**

- `--bin train`：指定运行训练二进制文件
- `--multimodal`：启用多模态训练模式
- `--sft-jsonl data/mm_test.jsonl`：指定SFT训练数据文件路径
- `--output-dir models/mm_vit`：指定使用Vision Transformer编码器的模型输出目录
- `--vision-out-dim 768`：Vision Transformer输出特征维度为768，比ResNet的512更大，提供更强的特征表达能力
- `--fusion-strategy cross_attention`：使用跨模态注意力机制进行特征融合，充分利用Transformer的自注意力优势
- `--backend gpu`：使用GPU进行训练，ViT模型计算量较大，需要GPU加速

**完整命令：**

```bash
cargo run --bin train -- `
    --multimodal `
    --sft-jsonl data/mm_test.jsonl `
    --output-dir models/mm_vit `
    --vision-out-dim 768 `
    --fusion-strategy cross_attention `
    --backend gpu
```

### 多模态推理示例

#### 基础多模态推理

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

#### GPU 加速多模态推理

**参数详解：**

- `--bin infer`：指定运行推理二进制文件
- `--model-dir models/mm_model`：指定模型目录路径
- `--use-best`：使用模型目录中的最佳检查点（best checkpoint）进行推理，通常是验证集上表现最好的模型
- `--multimodal`：启用多模态推理模式
- `--image-path data/text_to_images/cat.png`：指定输入图像路径
- `--prompt "描述这张图片"`：文本提示词
- `--num-tokens 100`：指定最大生成token数量，控制输出文本的长度，100个token约等于75个中文汉字
- `--temperature 0.7`：指定采样温度参数，控制输出的随机性，值越低输出越确定性，0.7是生成质量和多样性的平衡点
- `--backend gpu`：使用GPU进行推理加速，大幅提升推理速度

**完整命令：**

```bash
cargo run --bin infer -- `
    --model-dir models/mm_model `
    --use-best `
    --multimodal `
    --image-path data/text_to_images/cat.png `
    --prompt "描述这张图片" `
    --num-tokens 100 `
    --temperature 0.7 `
    --backend gpu
```

#### 详细参数多模态推理

**参数详解：**

- `--bin infer`：指定运行推理二进制文件
- `--model-dir models/mm_model`：指定模型目录路径
- `--multimodal`：启用多模态推理模式
- `--image-path data/text_to_images/dog.png`：指定输入图像路径，这里使用狗的图片作为示例
- `--prompt "详细描述这张图片，包括颜色、动作和场景"`：详细的文本提示词，引导模型生成更丰富的描述
- `--num-tokens 200`：最大生成200个token，允许模型输出更长的详细描述
- `--temperature 0.8`：较高的温度参数，增加输出的创造性和多样性，适合需要丰富描述的场景
- `--top-p 0.9`：核采样参数，0.9表示从累积概率达到0.9的token中进行采样，平衡质量和多样性
- `--top-k 50`：限制每次从概率最高的50个token中进行采样，防止低概率token被选中
- `--backend cpu`：使用CPU进行推理，适合在没有GPU的环境下使用

**完整命令：**

```bash
cargo run --bin infer -- `
    --model-dir models/mm_model `
    --multimodal `
    --image-path data/text_to_images/dog.png `
    --prompt "详细描述这张图片，包括颜色、动作和场景" `
    --num-tokens 200 `
    --temperature 0.8 `
    --top-p 0.9 `
    --top-k 50 `
    --backend cpu
```

#### 交互式多模态对话

**参数详解：**

- `--bin infer`：指定运行推理二进制文件
- `--model-dir models/mm_model`：指定模型目录路径
- `--use-best`：使用最佳检查点进行推理
- `--multimodal`：启用多模态推理模式
- `--image-path data/text_to_images/cat.png`：指定输入图像路径
- `--chat`：启用聊天模式，允许进行多轮对话交互
- `--interactive`：启用交互式模式，用户可以在终端中连续输入多个问题或指令

**完整命令：**

```bash
cargo run --bin infer -- `
    --model-dir models/mm_model `
    --use-best `
    --multimodal `
    --image-path data/text_to_images/cat.png `
    --chat `
    --interactive
```

**详细文档：** 更多多模态功能说明请参考 [MULTIMODAL_GUIDE.md](MULTIMODAL_GUIDE.md)

---

## 参数速查表

### train 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--corpus-dir` | 语料目录 | `--corpus-dir ./data` |
| `--sft-jsonl` | SFT数据文件 | `--sft-jsonl train.jsonl` |
| `--output-dir` | 输出目录 | `--output-dir ./output` |
| `--num-epochs` | 训练轮数 | `--num-epochs 5` |
| `--batch-size` | 批次大小 | `--batch-size 8` |
| `--learning-rate` | 学习率 | `--learning-rate 1e-4` |
| `--max-seq-len` | 最大序列长度 | `--max-seq-len 512` |
| `--backend` | 后端 | `--backend gpu` |
| `--use-lora` | 启用LoRA | `--use-lora --lora-rank 8` |
| `--multimodal` | 启用多模态 | `--multimodal` |
| `--fusion-strategy` | 融合策略 | `--fusion-strategy gated` |

### infer 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--prompt` | 输入文本 | `--prompt "你好"` |
| `--model-dir` | 模型目录 | `--model-dir ./model` |
| `--num-tokens` | 生成token数 | `--num-tokens 200` |
| `--temperature` | 温度参数 | `--temperature 0.8` |
| `--top-p` | Top-p采样 | `--top-p 0.9` |
| `--multimodal` | 多模态模式 | `--multimodal` |
| `--image-path` | 图像路径 | `--image-path ./img.jpg` |
| `--chat` | Chat模式 | `--chat` |
| `--terminal` | 终端模式 | `--terminal` |

### image_gen 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--text-to-image` | 文生图模式 | `--text-to-image` |
| `--generate-only` | 纯图像生成 | `--generate-only` |
| `--prompt` | 文本描述 | `--prompt "一只猫"` |
| `--output` | 输出路径 | `--output ./out.png` |
| `--steps` | 采样步数 | `--steps 50` |
| `--image-size` | 图像尺寸 | `--image-size 64` |
| `--latent-dim` | 潜在维度 | `--latent-dim 128` |

---

**最后更新：** 2026-04-21
**版本：** v1.3
