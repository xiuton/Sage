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
- **GPU 加速训练**（WGPU 后端支持）
- **快速开发模式**（3轮训练 + 优化参数，适合测试和迭代）
- **可中断训练**（Ctrl+C 优雅关闭并保存检查点）
- **可选进度条**（可禁用 TUI 显示以获得清洁日志）

> 目标：提供一个“可跑通、可扩展、可继续工程化”的 Rust 大模型训练最小闭环。

---

## 文档导航

### 核心文档
- [README.md](README.md)：项目总览、快速开始、功能特性、训练配方与常见问题
- [COMMANDS.md](docs/COMMANDS.md)：完整命令行参数手册（训练、推理、数据生成）
- [DATA_FORMAT.md](docs/DATA_FORMAT.md)：训练数据格式规范（纯文本LM训练、SFT训练）
- [TRAINING_DEPLOYMENT.md](docs/TRAINING_DEPLOYMENT.md)：实战训练与部署完整指南，包含环境准备、数据准备、训练配置、模型部署、性能优化等章节的实际操作演示
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
    bin/
      train.rs        # 训练入口（LM/SFT，支持 BPE/字符级分词器）
      infer.rs        # 推理入口（续写/Chat/交互）
      gen_sft.rs      # 生成 SFT JSONL（测试/压测用）
      gen_web_sft.rs  # 生成网页格式的 SFT JSONL
    data.rs           # Dataset/Batcher（含 SFT mask → target pad）
    generation.rs     # 采样/生成（top-k/top-p/重复惩罚/标点惩罚/context window）
    model.rs          # Transformer LM（含 TrainStep/ValidStep）
    tokenizer.rs      # 分词器（字符级 tokenizer + BPE，支持 SFT mask 编码）
    training.rs       # LearnerBuilder 训练封装 + best 模型导出
    streaming.rs      # 流式数据加载（支持大语料训练）
    lib.rs            # 库导出
  docs/
    COMMANDS.md            # 命令行参数说明
    DATA_FORMAT.md         # 数据格式说明
    PROJECT_STATUS.md      # 项目状态和开发计划
    TRAINING_DEPLOYMENT.md # 训练部署指南
    TROUBLESHOOTING.md     # 故障排查指南
  data/                   # 生成的数据文件目录
  .gitignore
  Cargo.toml
  Cargo.lock
  README.md
```

## 已实现功能特性（按模块）

### 模型（Transformer LM）

- Token Embedding + 可学习位置 Embedding
- Transformer Encoder（多层、多头注意力、FFN）
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

## 命令总览

本项目主要提供三个二进制：

- `train`：训练
- `infer`：推理
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
