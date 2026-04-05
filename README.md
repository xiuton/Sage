# Sage（Rust 0.001B 小模型 / 训练与推理工程）

## 项目概览

Sage 是一个使用 **Rust + Burn** 实现的小型 Transformer 项目，提供完整的大模型训练与推理闭环。

### 核心特性

- **训练模式**：纯文本自回归训练（LM）、指令/对话 SFT 训练、DPO偏好对齐训练
- **模型规模**：1M / 10M / 30M / 100M / 1B / 3B 参数
- **推理功能**：Chat 模式、流式输出、GPU 加速
- **技术特性**：BPE 分词器、INT4/INT8 量化、可中断训练、快速开发模式
- **分布式训练**：支持多 GPU/多机器数据并行训练
- **多模态能力**：支持图像输入，实现图像编码器和多模态融合层，集成到完整的训练和推理流程

> 目标：提供一个“可跑通、可扩展、可继续工程化”的 Rust 大模型训练最小闭环。

---

## 文档导航

### 核心文档

- **[COMMANDS.md](docs/COMMANDS.md)**：完整命令行参数手册（训练、推理、数据生成）
- **[DATA_FORMAT.md](docs/DATA_FORMAT.md)**：训练数据格式规范（纯文本LM训练、SFT训练）
- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**：详细训练指南
- **[TRAINING_PHASES.md](docs/TRAINING_PHASES.md)**：**显存探测 vs 正式训练**（何时出现 TUI/Learner、阶段说明）
- **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)**：实战部署指南
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**：常见故障排查与解决方案
- **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)**：项目开发状态、已完成功能、未来计划路线图
- **[PROJECT_CHECKLIST.md](docs/PROJECT_CHECKLIST.md)**：功能检查清单与优化计划、模块完整性验证
- **[QUICK_TEST_GUIDE.md](docs/QUICK_TEST_GUIDE.md)**：全流程测试指南（53项测试覆盖所有功能）
- **[ARCHITECTURE_REVIEW.md](docs/ARCHITECTURE_REVIEW.md)**：功能合理性、Rust/目录规范、小模型场景取舍（审阅向）

### 文档职责说明

| 文档 | 主要内容 | 适用读者 |
|------|----------|----------|
| COMMANDS.md | 完整命令行参数参考 | 所有用户 |
| DATA_FORMAT.md | 数据格式规范 | 数据准备人员 |
| TRAINING_GUIDE.md | 训练方法和最佳实践 | 训练工程师 |
| TRAINING_PHASES.md | 非正式预检（显存探测）与正式训练（Learner）分界 | 训练工程师 |
| DEPLOYMENT_GUIDE.md | 实战部署指南 | 部署运维人员 |
| TROUBLESHOOTING.md | 问题排查 | 所有用户 |
| PROJECT_STATUS.md | 项目进展和路线图 | 关注项目发展的用户 |
| PROJECT_CHECKLIST.md | 功能检查清单、优化计划、模块完整性验证 | 开发者 / 维护者 |
| QUICK_TEST_GUIDE.md | 全流程测试指南（53项测试覆盖所有功能） | 测试人员 / 开发者 |
| ARCHITECTURE_REVIEW.md | 架构与规范审阅、功能取舍 | 维护者 / 进阶贡献者 |

## 快速开始

详细的快速开始指南请参考 [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) 和 [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)。

### 环境与工具链

- 本仓库 `Cargo.toml` 为 **`edition = "2024"`**，请使用 **支持该 edition** 的 Rust 工具链（建议通过 [rustup](https://rustup.rs/) 安装的当前 **stable**，并定期 `rustup update`）。
- **GPU 训练**（`--backend gpu`）依赖 **WGPU** 可用的环境与显卡驱动；详见 [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) 与 [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。

### 基础流程

1. **环境准备**：安装 Rust 和必要依赖
2. **数据准备**：准备训练数据或使用内置样例
3. **模型训练**：使用 `train` 命令训练模型
4. **模型推理**：使用 `infer` 命令进行推理

### 示例命令

```bash
# 生成训练数据
cargo run --release --bin gen_sft -- --out data/sft_demo.jsonl --count 5000

# 训练模型
cargo run --release --bin train -- --sft-jsonl data/sft_demo.jsonl --artifact-dir ./tmp/model

# 推理生成
cargo run --bin infer -- --model-dir ./tmp/model --use-best --chat --prompt "你好"

# 多模态训练
cargo run --bin train -- --multimodal --vision-out-dim 512 --fusion-strategy add --ultra-quick --sft-sample

# 多模态推理
cargo run --bin infer -- --multimodal --image-path ./test_image.jpg --prompt "描述这张图片"
```

---

## 项目结构

```
Sage/
  src/
    bin/                    # 可执行文件入口
      train.rs              # 训练入口（LM/SFT/DPO，支持 BPE/字符级分词器）
      infer.rs              # 推理入口（续写/Chat/交互）
      api_server.rs         # API 服务器（模型管理、推理服务）
      accuracy_eval.rs      # 模型准确率评估工具
      benchmark.rs          # 性能基准测试工具
      export.rs             # 模型导出工具
      gen_sft.rs            # 生成 SFT JSONL（测试/压测用）
      gen_web_sft.rs        # 生成网页格式的 SFT JSONL
    core/                   # 核心模型和推理功能
      mod.rs                # 核心模块导出
      model.rs              # Transformer LM（含 TrainStep/ValidStep）
      tokenizer.rs          # 分词器（字符级 tokenizer + BPE，支持 SFT mask 编码）
      generation.rs         # 采样/生成（top-k/top-p/重复惩罚/标点惩罚/context window）
      kv_cache.rs           # KV 缓存实现
      multimodal.rs         # 多模态能力（图像编码器、多模态融合层）
    training/               # 训练相关功能
      mod.rs                # 训练模块导出
      training.rs           # LearnerBuilder 训练封装 + best 模型导出
      streaming.rs          # 流式数据加载（支持大语料训练）
      lora.rs               # LoRA 模块（与主训练路径集成程度见 ARCHITECTURE_REVIEW）
      vram_probe.rs         # GPU 显存预检（单次 step，非 Learner）
      distributed.rs        # 分布式训练支持（数据并行、多GPU训练）
      dpo.rs                # DPO偏好对齐训练框架
    inference/              # 推理相关功能
      mod.rs                # 推理模块导出
      lazy_load.rs          # 懒加载模型功能
    data/                   # 数据处理功能
      mod.rs                # 数据模块导出
      data.rs               # Dataset/Batcher（含 SFT mask → target pad）
    api/                    # API服务器功能
      mod.rs                # API模块导出
    tools/                  # 工具类功能
      mod.rs                # 工具模块导出
      model_download.rs     # 模型下载功能（支持从远程 URL 下载模型）
      export.rs             # 模型导出功能
    utils/                  # 通用工具函数
      mod.rs                # 工具模块导出
      common.rs             # 通用工具函数
      error.rs              # 错误处理定义
      logger.rs             # 日志系统
      performance.rs        # 性能监控工具
    quantization/           # 量化功能
      mod.rs                # 量化模块导出
      quantization.rs       # 模型量化功能（INT4/INT8量化、动态量化）
    lib.rs                  # 库导出
  docs/                     # 文档目录
    COMMANDS.md            # 命令行参数说明
    DATA_FORMAT.md         # 数据格式说明
    PROJECT_STATUS.md      # 项目状态和开发计划
    DEPLOYMENT_GUIDE.md    # 部署指南
    TRAINING_GUIDE.md      # 训练指南
    TRAINING_PHASES.md     # 显存探测 vs 正式训练（Learner）
    ARCHITECTURE_REVIEW.md # 架构审阅与规范/取舍
    TROUBLESHOOTING.md     # 故障排查指南
  tests/                    # 测试目录
    test_api_server.rs     # API服务器测试
    test_kv_cache.rs       # KV缓存测试
    test_model.rs          # 模型测试
    test_performance.rs    # 性能测试
    test_tokenizer.rs      # 分词器测试
  data/                     # 生成的数据文件目录
  .gitignore
  Cargo.toml
  Cargo.lock
  README.md
  Dockerfile
  Dockerfile.gpu
  docker-compose.yml
```

---

## 已实现功能特性（按模块）

#### 模型（Transformer LM）

- Token Embedding + 可学习位置 Embedding
- **TransformerEncoder + 自回归掩码**（实现 Decoder-only 风格的因果注意力）
  - 组件使用 `TransformerEncoder`（Burn 0.20 下的最佳实践）
  - 通过自回归掩码实现因果注意力，确保每个 token 只能关注前面的 token
  - 行为上等价于 Decoder-only 架构
- 语言模型输出头（Linear → vocab logits）
- 参数量统计（估算）
- **多规模模型配置**：
  - `default`：约 1M 参数（默认）
  - `10m`：约 10M 参数
  - `30m`：约 30M 参数
  - `100m`：约 100M 参数
  - `1b`：约 1B 参数
  - `3b`：约 3B 参数
  - `671b`：约 671B 参数（演示用）
- **KV Cache**：用于加速自回归推理的键值缓存
- **RMSNorm 层代码**：现代大模型归一化层（已实现代码，待 API 适配）
- **SwiGLU 激活函数代码**：现代大模型 FFN 激活函数（已实现代码，待 API 适配）

代码入口：[core/model.rs](src/core/model.rs)

### Tokenizer（字符级 + BPE）

- 字符级词表（Unicode `char`，天然支持中文）
- 特殊 token：`pad_id=0`、`unk_id=1`、`bos_id=2`、`eos_id=3`
- 支持保存/加载：`tokenizer.json`
- SFT 专用：`encode_with_assistant_mask` 生成 token 序列 + “只学助手回复”的 mask

代码入口：[core/tokenizer.rs](src/core/tokenizer.rs)

### 数据集与训练数据管线

- `TextDataset`：按 `seq_len` 生成 (input, target)
- SFT mask：对“非助手回复”位置，将 target 置为 `pad_id=0`（并在 loss 中忽略 pad token）

代码入口：[data/data.rs](src/data/data.rs)

### 训练（Burn Learner）

- 可配置训练：epochs / batch_size / lr / max_seq_len
- 自动保存：`config.json` / `tokenizer.json` / `model.mpk`
- checkpoint（按 epoch）
- best 模型：扫描 valid loss 自动导出 `best_model.mpk`
- 继续训练：
  - `--continue` 从 `model.mpk` 加载权重继续训
  - `--resume-epoch N` 从 `checkpoint/model-N.mpk` 加载权重继续训
- **多种训练模式**：
  - `general`：通用对话模式（默认）
  - `code`：代码生成模式（优化代码生成场景）
  - `math`：数学推理模式（优化数学问题解决场景）
- **DPO偏好对齐训练**：支持 beta 参数和 KL 散度正则化
- **多规模模型**：`--model-size default/10m/30m/100m/1b/3b/671b`
- **GPU 加速**：`--backend gpu`（WGPU 后端）
- **分布式训练**：支持多 GPU/多机器数据并行训练
- **多模态能力**：支持图像输入，实现图像编码器和多模态融合层
  - `--multimodal`：启用多模态训练
  - `--vision-out-dim`：视觉编码器输出维度
  - `--fusion-strategy`：融合策略（add/concatenate/attention）
- **真实 Loss 计算**：训练和验证阶段均使用真实损失值
- **梯度累积**：支持梯度累积步数配置
- **学习率调度器配置**：支持 Cosine Annealing + Warmup 学习率调整策略

> 说明：当前"继续训练"是**只恢复模型权重**，不恢复优化器状态（后续计划优化）。

代码入口：[training/training.rs](src/training/training.rs)、[bin/train.rs](src/bin/train.rs)

### 评估指标

- **Perplexity**：语言模型质量评估指标（从损失计算）
- **BLEU**：文本生成质量评估指标（简化版）

代码入口：[utils/metrics.rs](src/utils/metrics.rs)

### 推理生成（Sampling）

- `temperature` 温度
- `top_k` / `top_p`（Nucleus）
- `repetition_penalty`（抑制重复）
- `punctuation_penalty`（抑制连续标点）
- `context_len` 上下文窗口（默认跟随 `model.max_seq_len`，并自动截断避免越界）
- `--chat`：按 `<user>...</user>\n<assistant>...</assistant>` 模板生成并截取回复
- **停止序列**：
  - `--stop-on-user`：遇到 `<user>` 标签时停止（默认启用）
  - `--stop-sequence`：自定义停止序列（可多次使用）
- **多模态推理**：
  - `--multimodal`：启用多模态推理
  - `--image-path`：指定图像文件路径
  - 支持图像描述、图像理解等多模态任务

代码入口：[core/generation.rs](src/core/generation.rs)、[bin/infer.rs](src/bin/infer.rs)

---

## 数据格式

详细的数据格式说明请参考 [DATA_FORMAT.md](docs/DATA_FORMAT.md)。

### 1) 纯文本 LM 训练（续写）

训练目标：预测下一个 token。输入通常是一段长文本（中文/英文都可以）。

你可以使用：

- 单文件：`--corpus corpus_cn.txt`
- 多文件目录：`--corpus-dir D:\data\texts`（递归收集 `.txt`，按路径排序后拼接，并用换行分隔）

### 2) SFT 训练（JSONL）

训练目标：让模型学会按"用户/助手"模板输出回复；并且只对"助手回复段"计算学习信号（mask loss）。

当前支持两种 JSONL schema（每行一个 JSON 对象）：

**A. prompt/response**

```json
{"prompt":"你是谁？","response":"我是一个用 Rust 训练出来的小模型。"}
```

**B. messages（推荐，支持多轮）**

```json
{"messages":[
  {"role":"system","content":"你是一个有帮助的助手。"},
  {"role":"user","content":"你是谁？"},
  {"role":"assistant","content":"我是一个用 Rust 训练出来的小模型。"}
]}
```

说明：

- `role` 支持 `system` / `user` / `assistant`。
- 多轮对话建议以 `user→assistant→user→assistant...` 的顺序组织。
- `system` 角色用于设置系统提示词。

---

## 训练产物与复用

训练输出目录由 `--artifact-dir` 控制，目录结构（示例）：

```
artifact-dir/
  config.json
  tokenizer.json
  model.mpk
  best_model.mpk
  checkpoint/
    model-1.mpk
    model-2.mpk
  train/
  valid/
    epoch-1/Loss.log
```

- `model.mpk`：最后一次训练结束的权重
- `best_model.mpk`：根据 valid loss 自动选择的最优 epoch 权重（推理可用 `--use-best` 优先加载）
- `checkpoint/`：每个 epoch 的权重快照（可用 `--resume-epoch` 从某个 epoch 继续训练）

---

## 大规模训练建议（现阶段工程实践）

详细的大规模训练指南请参考 [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)。

本项目当前仍是“最小闭环”，但已经能支撑更大语料的工程化训练。建议按以下方式逐步放大：

1) **从小规模验证开始**

- 先用 `--sft-max-records 1000` 或 `--max-bytes 10000000` 做快速 smoke test，确认流程与产物无误，再放大规模。

2) **控制内存占用**

- `train --stream`：逐行读取/分块处理，并把 token 写入 `artifact-dir/cache/`，训练时用 memmap 数据集读取，显著降低峰值内存（会落盘 cache）。
- `train --stream --stream-direct`：逐行读取并直接训练，不写入 token cache（不落盘、边读边训；当前仅支持 SFT）。
- 使用 `--max-bytes` 限制读取上限，避免一次性读爆内存。
- 对超大 JSONL，建议先用 `--sft-max-records` 做 smoke test，再放大规模。

3) **避免 tokenizer 词表漂移**

- SFT/LM 训练时，如果换了语料且仍复用旧 `tokenizer.json`，会导致新字符大量映射到 `unk`，效果变差。
- 语料变化较大时建议加 `--reset-tokenizer`。

4) **长上下文**

- 推理 `--context-len` 会被自动截断到 `model.max_seq_len`。
- 如果你确实需要更长上下文：训练时提高 `--max-seq-len` 并重新训练模型。

---

## 硬件加速配置

详细的硬件加速配置请参考 [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)。

### GPU 加速训练

项目支持通过命令行参数选择训练后端：

```bash
# 使用GPU后端（需要支持WGPU的显卡）
cargo run --release --bin train -- --backend gpu --sft-jsonl data.jsonl --artifact-dir ./tmp/gpu_model

# 使用CPU后端（默认）
cargo run --release --bin train -- --backend cpu --sft-jsonl data.jsonl --artifact-dir ./tmp/cpu_model
```

**注意**：GPU后端需要支持WGPU的显卡。

### CPU 多线程优化

```bash
# 自动根据CPU核心数优化工作线程数
cargo run --release --bin train -- --sft-jsonl data.jsonl --num-workers auto

# 手动指定工作线程数
cargo run --release --bin train -- --sft-jsonl data.jsonl --num-workers 16
```

---

## 学习率调度器（推荐使用）

项目实现了 **Cosine Annealing + Warmup** 学习率调度器，能显著提升训练稳定性和收敛效果。

### 快速开始
```bash
# 使用学习率调度器训练（推荐）
cargo run --release --bin train -- --sft-jsonl sft_demo_5000.jsonl --artifact-dir ./tmp/sft_lr_scheduler --lr-scheduler --lr-max 0.0005 --lr-min 0.00001 --warmup-steps 500 --total-steps 10000 --use-bpe --num-epochs 50 --backend gpu
```

### 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr-scheduler` | 禁用 | 启用学习率调度器 |
| `--lr-max` | 0.0001 | 最大学习率（Warmup阶段结束时的值） |
| `--lr-min` | 0.00001 | 最小学习率（Cosine阶段结束时的值） |
| `--warmup-steps` | 1000 | Warmup步数（学习率从0线性增加到lr-max） |
| `--total-steps` | 10000 | 总调度步数 |

### 调度阶段
1. **Warmup阶段**（前 warmup-steps）：学习率从 0 线性增加到 lr-max
2. **Cosine阶段**（之后）：学习率从 lr-max 余弦衰减到 lr-min

### 推荐设置
- Warmup步数：总步数的 5%-10%
- 小模型（1M/10M）：lr-max=0.0005, lr-min=0.00001
- 中等模型（30M/100M）：lr-max=0.0003, lr-min=0.000005

---

## 评估指标

项目支持多种评估指标，用于监控和评估模型性能。

### Perplexity（困惑度）
Perplexity 是衡量语言模型质量的重要指标，值越低越好。
- 计算方式：Perplexity = exp(Loss)
- 理想值：对于高质量语料，通常应低于 10-20

### BLEU 分数
BLEU 用于评估文本生成质量，比较生成文本与参考文本的相似度。
- 范围：0.0 ~ 1.0
- 值越高表示生成质量越好

---

## 编译与构建

本项目使用 **Cargo Features** 进行功能模块化管理，以优化编译内存占用和时间。

### Feature 功能说明

| Feature | 包含内容 | 说明 |
|---------|----------|------|
| `core` (默认) | `train`、`infer`、`gen_sft` | 核心功能，默认编译 |
| `api` | `api_server` | API 服务器 |
| `tools` | `benchmark`、`accuracy_eval`、`export` | 辅助工具 |
| `web` | `gen_web_sft` | Web 数据生成 |
| `full` | 所有功能 | 全部编译 |

### 常用编译命令

```powershell
# 推荐：只编译核心功能（内存占用最小）
cargo build --release

# 编译核心 + API 服务器
cargo build --release --features "api"

# 编译核心 + 辅助工具
cargo build --release --features "tools"

# 编译所有功能
cargo build --release --features "full"
```

### 内存优化建议

如果在 Windows 上遇到编译内存不足（OOM）问题：

1. **使用 `-j 1` 限制并行编译**：
   ```powershell
   cargo build --release -j 1
   ```

2. **只编译需要的二进制**：
   ```powershell
   cargo build --release --bin train --bin infer --bin gen_sft
   ```

3. **使用 Debug 模式**（开发时）：
   ```powershell
   cargo build
   ```

---

## 命令总览

本项目提供 **八个** 可执行目标（`src/bin/*.rs`）：

- `train`：训练
- `infer`：推理
- `api_server`：API 服务器
- `accuracy_eval`：模型准确率评估工具
- `benchmark`：性能基准测试工具
- `export`：模型导出工具
- `gen_sft`：生成可训练的 SFT JSONL 数据
- `gen_web_sft`：生成带网页/网络选项的 SFT JSONL

完整参数说明见：[COMMANDS.md](docs/COMMANDS.md)

**训练阶段说明**（显存探测与 Burn 正式训练何时分界）：见 [TRAINING_PHASES.md](docs/TRAINING_PHASES.md)。

---

## 已知限制（现阶段）

- 已实现 **BPE**；纯字符级在长中文文本上仍可能效率偏低，可按需选用 `--use-bpe`。
- 默认/小档位模型参数量有限，即使 SFT 数据增大，也难以达到生产级助手水平。
- 当前 SFT 的 mask loss 是通过“把非学习位置 target 置为 `pad_id=0` 并在 loss 中忽略 pad token”实现的近似方案；更严格的实现应当使用专门的 ignore_index / loss mask。
- `burn_train` 可能出现 “Failed to install the file logger” 警告（Windows 权限/路径相关），不影响训练主流程。
- Windows 偶尔会遇到 `LNK1104 cannot open file infer.exe`（可执行文件被占用），可用 `cargo clean` 或关闭残留进程后重试。

---

## 未来计划（建议任务清单）

- **更严格的 SFT 损失掩码**：对助手回复以外 token 使用真正的 ignore_index 或 loss mask，而不是 pad 替代。
- **Tokenizer 升级**：~~BPE / SentencePiece（可选 Rust 实现或集成现有 crate）。~~ ✅ **已完成**（BPE 已实现）
- **数据流式加载**：~~对超大 JSONL/多文件语料，支持流式读取而非一次性读入内存。~~ ✅ **已完成**
- **恢复优化器状态**：checkpoint 恢复不仅恢复模型权重，也恢复 optimizer/scheduler。
- **更强的停止策略**：~~支持 stop sequences（例如遇到 `\u{0003}` 或 "用户：" 时停止生成）。~~ ✅ **已完成**
- **学习率调度器**：~~Cosine Annealing + Warmup 学习率调整策略。~~ ✅ **已完成**
- **评测与指标**：Perplexity、BLEU 分数、样例回放等。 ✅ **已完成**（Perplexity 和 BLEU 已实现）
- **GPU 训练与推理**：~~完善 WGPU 后端使用与性能优化（当前默认 NdArray CPU）。~~ ✅ **已完成**（支持 `--backend gpu`）
- **更大模型配置**：~~提供多个预设 config（~1M、~10M、~30M）按硬件选择。~~ ✅ **已完成**（`--model-size` 参数）
- **专项训练模式**：代码生成、数学推理等专项优化。 ✅ **已完成**（`--training-mode` 参数）
- **自回归掩码**：~~实现因果注意力机制，确保每个token只能关注前面的token。~~ ✅ **已完成**（通过自回归掩码实现）
- **KV Cache 优化**：进一步优化 KV Cache 以提升推理性能
- **RoPE 位置编码**：实现旋转位置编码以提升长文本外推能力
- **RMSNorm 与 SwiGLU 启用**：适配 Burn 0.20 API 以启用 RMSNorm 归一化层和 SwiGLU 激活函数
