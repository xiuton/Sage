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
cargo run --release --bin train -- --corpus-dir D:\data\cn_texts --output-dir ./tmp/lm_cn --num-epochs 5 --max-seq-len 64
```

**A4. 分布式训练（权重同步）**

```bash
cargo run --release --bin train -- --distributed --devices gpu:0,gpu:1 --sft-jsonl data.jsonl
```
当前实现了基础的分布式权重平均与同步逻辑，支持多设备协同训练。

**A5. LoRA 轻量化微调**

```bash
cargo run --release --bin train -- --use-lora --lora-rank 8 --lora-alpha 16 --sft-jsonl data.jsonl --output-dir ./tmp/lora_model
```
LoRA 模式下仅训练低秩矩阵，可大幅降低显存占用。

**A6. DPO偏好对齐训练**

```bash
cargo run --release --bin train -- --dpo --dpo-data dpo_data.jsonl --output-dir ./tmp/dpo_model --dpo-beta 0.1 --dpo-kl-weight 0.1 --num-epochs 30 --batch-size 16 --backend gpu --force
```

**A2. 目录语料训练（限制读取大小 + 快速验证 - 预训练）**

```bash
cargo run --release --bin train -- --corpus-dir D:\data\cn_texts --output-dir ./tmp/lm_cn_quick --num-epochs 1 --max-seq-len 64 --max-bytes 10000000 --force --reset-tokenizer
```

**A3. 大规模预训练（使用 GPU 和流式处理）**

```bash
cargo run --release --bin train -- --corpus-dir ./corpus --max-bytes 1000000000 --stream --backend gpu --config-path ./inference/configs/config_1B.json --output-dir ./models/large_pretrained
```

**A4. 基本预训练（使用 CPU）**

```bash
cargo run --release --bin train -- --corpus-dir ./corpus --output-dir ./tmp/lm_basic --num-epochs 1 --max-seq-len 64 --batch-size 4
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
cargo run --bin infer -- --model-dir ./tmp/sage_model_formal --use-best --prompt "写一首关于春天的诗"
```

**C. GPU 加速推理**

```bash
cargo run --bin infer -- --model-dir ./tmp/sage_model_formal --use-best --prompt "解释量子计算" --backend gpu
```

### 2.2 交互式对话

**A. 交互模式**

```bash
cargo run --bin infer -- --model-dir ./tmp/sage_model_formal --use-best --interactive
```

**B. 终端模式（推荐）**

```bash
cargo run --bin infer -- --model-dir ./tmp/sage_model_formal --use-best --terminal
```

**C. Chat 模式**

```bash
cargo run --bin infer -- --model-dir ./tmp/sage_model_formal --use-best --chat --prompt "你好"
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
cargo run --release --bin api_server -- --port 8080 --model-dir ./tmp/sage_model_formal
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

```bash
# ResNet 编码器 + 门控融合
cargo run --release --bin train -- `
    --multimodal `
    --sft-jsonl data/multimodal_data.jsonl `
    --output-dir ./tmp/mm_resnet_gated `
    --vision-out-dim 512 `
    --fusion-strategy gated

# Vision Transformer 编码器 + 跨模态注意力融合
cargo run --release --bin train -- `
    --multimodal `
    --sft-jsonl data/multimodal_data.jsonl `
    --output-dir ./tmp/mm_vit_cross `
    --vision-out-dim 512 `
    --fusion-strategy cross_attention
```

### 多模态推理示例

```bash
# 基础多模态推理
cargo run --bin infer -- `
    --model-dir ./tmp/mm_resnet_gated `
    --multimodal `
    --image-path ./data/images/test.jpg `
    --prompt "描述这张图片"

# 使用最佳模型 + GPU + 详细参数
cargo run --bin infer -- `
    --model-dir ./tmp/mm_vit_cross `
    --use-best `
    --multimodal `
    --image-path ./data/images/sample.jpg `
    --prompt "详细描述这张图片，包括场景、物体和颜色" `
    --num-tokens 150 `
    --temperature 0.7 `
    --backend gpu

# 交互式多模态对话
cargo run --bin infer -- `
    --model-dir ./tmp/mm_model `
    --use-best `
    --multimodal `
    --image-path ./data/images/demo.jpg `
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

**最后更新：** 2026-04-19
**版本：** v1.2
