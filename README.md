# Sage（Rust 0.001B 小模型 / 训练与推理工程）

Sage 是一个使用 **Rust + Burn** 实现的小型 Transformer 项目，支持：

- 纯文本自回归训练（续写 LM）
- 指令/对话 SFT 训练（JSONL：`prompt/response` 或 `messages`）
- 多种模型规模（1M / 10M / 30M 参数）
- 多种训练模式（通用对话 / 代码生成 / 数学推理）
- 推理生成（Top-K / Top-P / 温度 / 重复惩罚 / 标点惩罚）
- Chat 模式（`用户/助手` 模板 + 交互模式 + 停止序列）
- 分词器支持：字符级 + BPE（Byte Pair Encoding）
- 训练产物持久化（配置、Tokenizer、模型权重、checkpoint、best 模型）
- 大语料流式训练（落盘 token cache，或不落盘边读边训）
- **GPU 加速训练和推理**（WGPU 后端支持）
- **快速开发模式**（3轮训练 + 优化参数，适合测试和迭代）
- **可中断训练**（Ctrl+C 优雅关闭并保存检查点）
- **可选进度条**（可禁用 TUI 显示以获得清洁日志）
- **流式输出支持**（SSE Server-Sent Events）
- **INT8 量化优化**（减小模型体积，提升推理速度）

> 目标：提供一个“可跑通、可扩展、可继续工程化”的 Rust 大模型训练最小闭环。

---

## 文档导航

### 核心文档
- [README.md](README.md)：项目总览、快速开始、功能特性、训练配方与常见问题
- [COMMANDS.md](docs/COMMANDS.md)：完整命令行参数手册（训练、推理、数据生成）
- [DATA_FORMAT.md](docs/DATA_FORMAT.md)：训练数据格式规范（纯文本LM训练、SFT训练）
- [TRAINING_DEPLOYMENT.md](docs/TRAINING_DEPLOYMENT.md)：实战训练与部署完整指南，包含环境准备、数据准备、训练配置、模型部署、性能优化等章节的实际操作演示
- [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)：详细训练指南
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)：常见故障排查与解决方案
- [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)：项目开发状态、已完成功能、未来计划路线图

### 文档说明
- **核心文档**：提供项目基础信息和使用指南
- **实战指南**：包含完整的操作演示，适合新手快速上手
- **参考文档**：提供问题解决和项目进展信息

## 快速开始

### 1) 环境

- Windows / PowerShell（本仓库已在 Windows 上验证）
- Rust stable（建议使用最新稳定版）

### 2) 编译检查

```bash
cargo check
```

### 3) 生成一份可训练的 SFT 数据（5000 条）

项目内置了一个数据生成器二进制：[gen_sft.rs](src/bin/gen_sft.rs)

**生成 SFT 训练数据：**

```bash
cargo run --release --bin gen_sft -- --out data/sft_demo_5000.jsonl --count 5000 --seed 42
```

生成文件示例（每行一条 JSON）：

```json
{"id":91,"messages":[{"role":"user","content":"...?"},{"role":"assistant","content":"..."}]}
```

说明：仓库默认忽略 `sft*.jsonl`（避免把大数据文件提交进仓库）。建议用 `gen_sft` 生成到本地路径使用。

### 4) 用 SFT 数据训练一个模型

```bash
cargo run --release --bin train -- --sft-jsonl data/sft_demo_5000.jsonl --artifact-dir ./tmp/sft_demo_model --num-epochs 1 --max-seq-len 64 --force --reset-tokenizer
```

```bash
cargo run --release --bin train -- --sft-jsonl data/sft_demo_5000.jsonl --artifact-dir ./tmp/sft_demo_model --num-epochs 1 --max-seq-len 256 --force --reset-tokenizer
```

**使用 BPE 分词器训练（推荐，减少重复字符问题）：**

```bash
cargo run --release --bin train -- --sft-jsonl data/sft_demo_5000.jsonl --artifact-dir ./tmp/sft_bpe_model --use-bpe --bpe-vocab-size 5000 --num-epochs 50 --batch-size 32 --max-seq-len 256 --force --reset-tokenizer
```

**快速开发模式（用于测试和迭代）：**

```bash
cargo run --release --bin train -- --sft-jsonl data/sft_demo_5000.jsonl --artifact-dir ./tmp/sft_quick --quick-dev --force --reset-tokenizer
```

快速开发模式会自动优化设置以加快训练速度。

训练完成后会在 `./tmp/sft_demo_model` 生成：

- `config.json`：训练/模型配置（包含 `model` 超参）
- `tokenizer.json`：Tokenizer 词表
- `tokenizer.json.meta`：分词器元数据（BPE 时包含类型信息）
- `model.mpk`：最终模型权重
- `checkpoint/`：epoch checkpoint（文件名形如 `model-<epoch>.mpk`）
- `best_model.mpk`：根据 `valid/epoch-*/Loss.log` 选择的最优 epoch 权重（如果 valid 目录存在）

### 5) 推理（chat 模式）

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_demo_model --use-best --chat --prompt "你是谁？" -n 80
```

交互模式（连续对话）：

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_demo_model --use-best --chat --interactive
```

---

## 项目结构

```
Sage/
  src/
    bin/                    # 可执行文件入口
      train.rs              # 训练入口（LM/SFT，支持 BPE/字符级分词器）
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
    training/               # 训练相关功能
      mod.rs                # 训练模块导出
      training.rs           # LearnerBuilder 训练封装 + best 模型导出
      streaming.rs          # 流式数据加载（支持大语料训练）
      lora.rs               # LoRA 训练支持
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
      quantization.rs       # 模型量化功能
    lib.rs                  # 库导出
  docs/                     # 文档目录
    COMMANDS.md            # 命令行参数说明
    DATA_FORMAT.md         # 数据格式说明
    PROJECT_STATUS.md      # 项目状态和开发计划
    TRAINING_DEPLOYMENT.md # 训练部署指南
    TRAINING_GUIDE.md      # 训练指南
    TROUBLESHOOTING.md     # 故障排查指南
  data/                     # 生成的数据文件目录
  .gitignore
  Cargo.toml
  Cargo.lock
  README.md
```

## 已实现功能特性（按模块）

### 模型（Transformer LM）

- Token Embedding + 可学习位置 Embedding
- **Transformer Encoder（Encoder-only Transformer 架构）**（多层、多头注意力、FFN）
- 语言模型输出头（Linear → vocab logits）
- 参数量统计（估算）
- **多规模模型配置**：
  - `default`：约 1M 参数（默认）
  - `10m`：约 10M 参数
  - `30m`：约 30M 参数

代码入口：[model.rs](src/model.rs)

### Tokenizer（字符级）

- 字符级词表（Unicode `char`，天然支持中文）
- 特殊 token：`pad_id=0`、`unk_id=1`、`bos_id=2`、`eos_id=3`
- 支持保存/加载：`tokenizer.json`
- SFT 专用：`encode_with_assistant_mask` 生成 token 序列 + “只学助手回复”的 mask

代码入口：[tokenizer.rs](src/tokenizer.rs)

### 数据集与训练数据管线

- `TextDataset`：按 `seq_len` 生成 (input, target)
- SFT mask：对“非助手回复”位置，将 target 置为 `pad_id=0`（并在 loss 中忽略 pad token）

代码入口：[data.rs](src/data.rs)

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
- **多规模模型**：`--model-size default/10m/30m`
- **GPU 加速**：`--backend gpu`（WGPU 后端）

> 说明：当前"继续训练"是**只恢复模型权重**，不恢复优化器状态（后续计划优化）。

代码入口：[training.rs](src/training.rs)、[train.rs](src/bin/train.rs)

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

代码入口：[generation.rs](src/generation.rs)、[infer.rs](src/bin/infer.rs)

---

## 数据格式

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

**内部对话模板格式**

Sage 会把 SFT 数据转换成内部模板文本进行训练：

```
\u{0002}<s>
<user>用户问题</user>
<assistant>助手回复</assistant>\u{0003}
```

其中：
- `\u{0002}`：BOS（开始标记）
- `\u{0003}`：EOS（结束标记）
- `<s>`：序列开始标签
- `<user>` / `</user>`：用户内容标签
- `<assistant>` / `</assistant>`：助手内容标签
- `<system>` / `</system>`：系统提示标签（可选）

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

### 高速模式（推荐大数据 / 多核CPU）

```bash
cargo run --bin train -- --fast --sft-jsonl data/sft_demo_5000.jsonl --artifact-dir ./tmp/sft_fast --num-workers 12 --batch-size 64 --num-epochs 10 --force --reset-tokenizer
```

- 自动启用大批量、最少进度显示（`--no-progress`）
- `--num-workers` 可选 8~16
- 适合 8+ 核 CPU 和大数据，训练时间显著下降
- 自动根据CPU核心数设置最佳工作线程数

### 1) SFT 快速验证（几十到几百条）

用于确认：数据格式正确、训练能跑通、产物能加载推理。

```bash
cargo run --release --bin gen_sft -- --out data/sft_quick_200.jsonl --count 200 --seed 1
cargo run --release --bin train -- --sft-jsonl data/sft_quick_200.jsonl --sft-max-records 200 --artifact-dir ./tmp/sft_quick --num-epochs 1 --max-seq-len 64 --force --reset-tokenizer
cargo run --bin infer -- --model-dir ./tmp/sft_quick --use-best --chat --prompt "你是谁？" -n 80 -s 1
```

### 1.5) SFT 超快速验证（闪电模式）

用于超快速开发迭代：1轮训练、极小批量、高学习率、自动限制100条数据，适合快速验证代码修改。

```bash
# 使用内置样例数据（3条）
cargo run --bin train -- --ultra-quick --sft-sample --artifact-dir ./tmp/ultra_quick

# 或使用你的JSONL数据（自动限制为100条）
cargo run --bin train -- --ultra-quick --sft-jsonl your_data.jsonl --artifact-dir ./tmp/ultra_quick
```

特点：
- 1个epoch、batch_size=2、lr=1e-2
- 自动限制数据量为100条（如果未指定--sft-max-records）
- 训练时间通常在10-30秒内完成
- 适合BPE调参、代码修改验证

### 2) SFT 标准训练（几千到几万条）

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_main --num-epochs 5 --batch-size 16 --lr 1e-4 --max-seq-len 64 --max-bytes 50000000 --force --reset-tokenizer
```

建议：

- 先用 `--sft-max-records 1000` 做一次 smoke test，再去掉限制跑全量。
- 数据变化较大时加 `--reset-tokenizer`，避免大量字符变成 `unk`。

### 3) SFT 不落盘流式训练（边读边训）

适用：大 JSONL、希望减少磁盘中间产物（不写 token cache）。

```bash
cargo run --release --bin train -- --stream --stream-direct --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_stream_direct --num-epochs 3 --batch-size 16 --lr 1e-4 --max-seq-len 64 --max-bytes 50000000 --force --reset-tokenizer
```

### 4) 纯文本续写（LM）训练（多文件目录）

```bash
cargo run --release --bin train -- --corpus-dir D:\data\lm_texts --artifact-dir ./tmp/lm_cn --num-epochs 5 --max-seq-len 64 --max-bytes 50000000 --force --reset-tokenizer
```

### 5) 断点继续训练

从最终权重继续：

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_main --continue --num-epochs 2
```

从某个 epoch checkpoint 继续：

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_main --resume-epoch 3 --num-epochs 2
```

---

## 数据清洗建议（实用）

你现在的模型规模很小，数据质量对效果影响非常明显。建议至少做以下清洗：

- 去掉乱码与无意义重复（例如连续上百个相同字符/标点）。
- 统一换行与空白（Windows `\r\n` → `\n`）。
- 限制单条样本长度（过长会被 `max_seq_len` 截断，浪费算力）。
- 尽量减少“模板化重复句式”，避免模型学会机械复读。

对于 SFT（JSONL），建议保证：

- `messages` 里 `user/assistant` 成对出现
- `assistant` 回复尽量是完整句子（不是片段）
- 避免把“参考答案编号/题库选项”当成主要内容

---

## 常见问题与排错（FAQ）

### 1) 推理输出全是标点/重复

常见原因：小模型 + 字符级 tokenizer + 数据风格导致“标点”概率被高估。

建议顺序：

1. 降低 `--temperature`（例如 `0.7`）
2. 降低 `--top-p`（例如 `0.85~0.9`）
3. 增大 `--repetition-penalty`（例如 `1.1~1.3`）
4. 增大 `--punctuation-penalty`（例如 `1.6~3.0`）
5. 训练数据换成更口语/更接近对话的 SFT

### 2) `--context-len` 设大了会怎样？

infer 会自动把 `context-len` 截断到训练时的 `model.max_seq_len` 并提示，避免位置编码越界崩溃。

### 3) Windows 报错 `LNK1104 cannot open file infer.exe`

一般是 exe 被占用/未完全退出导致。可以：

- 关闭残留的 `infer.exe` / `train.exe` 进程
- 运行 `cargo clean` 后重试
- 换一个 `--target-dir`（临时解决）

---

## 详细训练指南

### 训练方式

#### 1. 测试训练（快速验证）

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

#### 2. 正式训练

使用自定义语料进行完整训练：

```bash
# 使用BPE分词器的正式训练
cargo run --release --bin train -- \
    --sft-jsonl data/your_corpus.jsonl \
    --artifact-dir ./tmp/your_model \
    --use-bpe \
    --bpe-vocab-size 20000 \
    --num-epochs 50 \
    --batch-size 32 \
    --max-seq-len 256 \
    --lr 5e-5 \
    --num-workers 8 \
    --backend gpu \
    --model-size 30m \
    --force
```

**关键参数：**
- `--sft-jsonl`：训练语料文件路径
- `--artifact-dir`：模型保存目录
- `--use-bpe`：启用BPE分词器
- `--bpe-vocab-size`：BPE词表大小（建议10000-50000）
- `--model-size`：模型大小（10m/30m）

#### 3. 增量训练

在现有模型基础上继续训练：

```bash
# 继续训练
cargo run --release --bin train -- \
    --sft-jsonl data/your_corpus.jsonl \
    --artifact-dir ./tmp/your_model \
    --continue \
    --num-epochs 100 \
    --backend gpu
```

**参数说明：**
- `--continue`：从现有模型继续训练
- `--resume-epoch <epoch>`：从特定轮次恢复训练

### 语料获取

#### 1. 使用内置生成工具

```bash
# 生成1000条多样化对话语料
cargo run --release --bin gen_sft -- --out data/dialogs.jsonl --count 1000 --seed 123

# 生成500条技术问答语料（包含网络数据）
cargo run --release --bin gen_web_sft -- --out data/tech_qa.jsonl --count 500 --web --seed 42
```

#### 2. 公开数据集

推荐的中文语料来源：
- **中文维基百科**：https://dumps.wikimedia.org/zhwiki/
- **中文新闻语料**：THUCNews、Sogou新闻等
- **GitHub代码库**：各种开源项目的代码和文档
- **Stack Overflow**：技术问答数据
- **Hugging Face数据集**：https://huggingface.co/datasets

### 训练参数调整

#### 1. 学习率调整

**推荐设置：**
- **初始学习率**：5e-5 ~ 1e-4（GPU）
- **学习率预热**：前10%轮次线性预热
- **学习率衰减**：使用余弦衰减或线性衰减

**调整策略：**
- **训练不稳定**：降低学习率（如从1e-4降至5e-5）
- **收敛缓慢**：适当提高学习率
- **损失波动大**：降低学习率并增加批量大小

#### 2. 批量大小调整

**根据GPU内存调整：**
- **8GB GPU**：batch-size = 8 ~ 16
- **16GB GPU**：batch-size = 16 ~ 32
- **24GB+ GPU**：batch-size = 32 ~ 64
- **CPU训练**：batch-size = 4 ~ 16（根据CPU核心数调整）

#### 3. 序列长度调整

**根据语料长度调整：**
- **短文本**（问答对）：max-seq-len = 128
- **中等文本**（段落对话）：max-seq-len = 256
- **长文本**（完整对话）：max-seq-len = 512
- **超长文本**：max-seq-len = 1024（需要足够的GPU内存）

#### 4. 训练轮次调整

**推荐设置：**
- **小数据集**（<1000条）：30 ~ 50轮
- **中等数据集**（1000-10000条）：50 ~ 100轮
- **大数据集**（>10000条）：100 ~ 200轮
- **超大数据集**（>100000条）：200 ~ 500轮

### 模型评估与优化

#### 1. 评估指标

**主要评估指标：**
- **损失值（Loss）**：训练和验证损失，反映模型拟合程度
- **困惑度（Perplexity）**：exp(loss)，越低越好，衡量模型预测的不确定性
- **生成质量**：人工评估生成内容的质量、相关性和连贯性

#### 2. 评估方法

```bash
# 使用训练好的模型进行交互式评估
cargo run --release --bin infer -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --chat \
    --interactive
```

#### 3. 优化策略

**数据优化：**
- **数据增强**：添加更多样化的训练数据，对现有数据进行改写和扩充
- **数据清洗**：去除低质量、重复的训练样本，过滤噪声和无关内容

**模型调优：**
- **模型结构优化**：尝试不同的模型大小（10m vs 30m），调整层数、注意力头数等超参数
- **超参数优化**：使用网格搜索或随机搜索优化超参数，关注学习率、批量大小等关键参数

**正则化技术：**
- **防止过拟合**：使用dropout层（当前已设置为0.1），添加权重衰减，使用早停策略
- **数据增强正则化**：使用数据扩充减少过拟合，添加噪声和扰动增加模型鲁棒性

#### 4. 过拟合处理

**过拟合迹象：**
- **训练损失持续下降，但验证损失开始上升**
- **生成内容重复或模式化**
- **模型在训练集上表现很好，但在新数据上表现差**

**解决方案：**
- **数据层面**：增加数据量，数据增强，数据清洗
- **模型层面**：降低模型复杂度，增强正则化
- **训练策略**：提前停止，减少训练轮次，调整学习率

### 常见问题排查

#### 1. UTF-8字符边界错误

**错误信息：**
```
byte index XX is not a char boundary; it is inside 'X' (bytes XX..XX)
```

**解决方案：**
- 已在代码中修复，使用安全的字符串匹配方法
- 如果仍然遇到此错误，请更新到最新版本

#### 2. GPU内存不足

**解决方案：**
- **减小批量大小**：`--batch-size 8`
- **减小序列长度**：`--max-seq-len 128`
- **使用更小的模型**：`--model-size 10m`

#### 3. 训练速度慢

**优化建议：**
- **使用GPU后端**：`--backend gpu`
- **增加工作线程数**：`--num-workers 8`（根据CPU核心数调整）
- **使用BPE分词器**：`--use-bpe`（比字符分词更快）
- **调整批处理大小**：找到GPU内存允许的最大批量

#### 4. TUI不显示

**解决方案：**
- **强制启用TUI**：`--force-tui`
- **确保终端支持**：使用支持ANSI颜色的终端
- **Windows终端**：推荐使用Windows Terminal或PowerShell

#### 5. 模型不收敛

**解决方案：**
- **调整学习率**：尝试不同的学习率（如5e-5, 1e-4）
- **检查数据质量**：确保训练数据格式正确
- **增加训练轮次**：可能需要更多轮次才能收敛
- **检查模型配置**：确认模型参数设置正确

---

## 生产环境部署

### 1. 模型导出

训练完成后，模型会自动保存到指定目录，包含以下文件：

```
./tmp/your_model/
├── model.mpk          # 模型权重文件
├── best_model.mpk     # 最佳模型权重（如果启用）
├── config.json        # 模型配置文件
├── tokenizer.json     # 分词器配置文件
├── tokenizer.vocab    # 词表文件（BPE模式）
└── checkpoint/        # 训练检查点目录
    ├── model-10.mpk   # 第10轮检查点
    ├── model-20.mpk   # 第20轮检查点
    └── ...
```

### 2. API服务部署

#### 启动API服务器：

```bash
# 启动API服务器（默认端口8000，CPU后端）
cargo run --release --bin api_server -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --port 8000

# 使用GPU后端加速推理
cargo run --release --bin api_server -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --backend gpu \
    --port 8000

# 使用INT8量化优化
cargo run --release --bin api_server -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --quantize \
    --port 8000
```

#### API接口说明：

| 端点 | 方法 | 描述 | 认证 |
|------|------|------|------|
| `/v1/chat/completions` | POST | Chat Completion接口（OpenAI标准） | 需要 |
| `/v1/batch-chat/completions` | POST | 批量Chat Completion接口 | 需要 |
| `/v1/async-chat/completions` | POST | 异步Chat Completion接口 | 需要 |
| `/api/task/:task_id` | GET | 查询任务状态 | 需要 |
| `/api/health` | GET | 健康检查接口 | 不需要 |
| `/api/model-info` | GET | 获取模型信息 | 需要 |

### 3. 性能优化

**模型优化：**
- **量化**：支持INT8动态量化和静态量化，减小模型体积并提高推理速度
- **剪枝**：移除不重要的权重（未来支持）
- **蒸馏**：知识蒸馏减小模型大小（未来支持）

**推理优化：**
- **批处理**：实现批处理推理提高吞吐量
- **缓存**：缓存频繁使用的计算结果
- **并行处理**：使用多线程或异步处理

---

## 命令总览

本项目提供七个二进制：

- `train`：训练
- `infer`：推理
- `api_server`：API服务器
- `accuracy_eval`：模型准确率评估工具
- `benchmark`：性能基准测试工具
- `export`：模型导出工具
- `gen_sft`：生成可训练的 SFT JSONL 数据

完整参数说明见：[COMMANDS.md](docs/COMMANDS.md)

---

## 已知限制（现阶段）

- Tokenizer 为字符级，表达能力与效率有限（中文更推荐 BPE/SentencePiece）。
- 模型规模约 0.001B，能力有限：即使 SFT 数据增大，也难以达到成熟助手水平。
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
- **评测与指标**：Perplexity、简单的 QA accuracy、样例回放等。
- **GPU 训练与推理**：~~完善 WGPU 后端使用与性能优化（当前默认 NdArray CPU）。~~ ✅ **已完成**（支持 `--backend gpu`）
- **更大模型配置**：~~提供多个预设 config（~1M、~10M、~30M）按硬件选择。~~ ✅ **已完成**（`--model-size` 参数）
- **专项训练模式**：代码生成、数学推理等专项优化。 ✅ **已完成**（`--training-mode` 参数）
