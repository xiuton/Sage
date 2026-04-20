# Sage 数据格式规范（DATA_FORMAT）

本文档描述 Sage 当前支持的训练数据输入格式，并给出推荐实践、常见陷阱与校验方法。

---

## 0) 总览

Sage 目前支持三类训练目标：

1. **LM（Language Modeling / 续写）**：从纯文本中学习预测下一个 token。
2. **SFT（Supervised Fine-Tuning / 指令微调）**：从对话/问答数据中学习“用户提问 → 助手回答”。
3. **DPO（Direct Preference Optimization / 偏好对齐）**：从偏好数据中学习区分好回复和坏回复。
另外：Sage 已全面支持**多模态训练与推理**。你可以在 SFT 数据中包含图像路径，实现图文理解能力。

---

## 1) 数据生成工具

推荐使用内置的 `gen_data` 工具生成各种格式的示例数据：

```bash
# 生成包含 Web 问答和多模态路径的综合 SFT 数据
cargo run --release --bin gen_data -- --out data/train.jsonl --count 1000 --web --multimodal
```

---

## 1) LM：纯文本语料

### 1.1 输入来源

训练命令（示例）：

```bash
cargo run --release --bin train -- --corpus corpus_cn.txt --artifact-dir ./tmp/lm_cn --num-epochs 3 --max-seq-len 64
```

或目录模式（递归读取所有 `.txt` 文件）：

```bash
cargo run --release --bin train -- --corpus-dir D:\data\lm_texts --artifact-dir ./tmp/lm_cn --num-epochs 3 --max-seq-len 64
```

当语料很大时建议启用 `--stream`：

```bash
cargo run --release --bin train -- --stream --corpus-dir D:\data\lm_texts --artifact-dir ./tmp/lm_cn --num-epochs 3 --max-seq-len 64
```

### 1.2 拼接规则（目录模式）

- 递归遍历目录下所有 `.txt` 文件
- 按路径排序
- 逐个读取并拼接，每个文件后追加一个换行
- 可用 `--max-bytes` 限制读入总字节数（用于防止内存占用过大）

---

## 2) SFT：JSONL

### 2.1 JSONL 基本要求

- **一行一个 JSON 对象**
- 文件编码建议 UTF-8
- 每行尽量不要超长（过长样本会被 `max_seq_len` 截断）

训练命令（示例）：

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_cn --num-epochs 3 --max-seq-len 64 --force --reset-tokenizer
```

大 JSONL 建议启用 `--stream`（逐行读取并写入 token cache）：

```bash
cargo run --release --bin train -- --stream --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_cn --num-epochs 3 --max-seq-len 64 --force --reset-tokenizer
```

如果你希望“不落盘、边读边训”，可以使用 `--stream --stream-direct`（当前仅支持 SFT）：

```bash
cargo run --release --bin train -- --stream --stream-direct --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_cn --num-epochs 3 --max-seq-len 64 --force --reset-tokenizer
```

### 2.2 支持的 schema

#### A) prompt/response（单轮问答）

```json
{"prompt":"你是谁？","response":"我是一个用 Rust 训练出来的小模型。"}
```

语义：

- `prompt`：用户输入/问题
- `response`：助手输出/回答

#### B) messages（推荐，支持多轮）

```json
{"messages":[
  {"role":"system","content":"你是一个有帮助的助手。"},
  {"role":"user","content":"你是谁？"},
  {"role":"assistant","content":"我是一个用 Rust 训练出来的小模型。"}
]}
```

规则：
- `role` 支持：`system` / `user` / `assistant`
- 多轮对话推荐顺序：`system → user → assistant → user → assistant ...`
- 该 schema **必须至少包含一条** `assistant` 消息。

#### C) 多模态消息（含图像路径）

```json
{
  "messages": [
    {"role": "user", "content": "这张图片里有什么？"},
    {"role": "assistant", "content": "这是一张示例图片，包含了一些常见的物体。"}
  ],
  "image_path": "assets/sample_image.jpg"
}
```

规则：
- `image_path`：可选字段，指向本地图像文件路径。
- 如果该行包含 `image_path`，训练时会自动加载该图像并通过视觉编码器处理。

### 2.3 Sage 内部对话模板（训练时）

Sage 会把 SFT 数据转换成内部模板文本进行训练：

**通用对话模式（general）**：

```
\u{0002}<s>
<user>用户问题</user>
<assistant>助手回复</assistant>\u{0003}
```

**代码生成模式（code）**：

```
\u{0002}<s>
<system>你是一个专业的代码助手，擅长编写高质量、可读性强的代码。</system>
<user>用户问题</user>
<assistant>助手回复</assistant>\u{0003}
```

**数学推理模式（math）**：

```
\u{0002}<s>
<system>你是一个数学专家，擅长解决各类数学问题并提供详细的解题步骤。</system>
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

### 2.4 “只学助手回复”（mask loss）

为了让模型更像“助手”，Sage 在 SFT 训练中会只对“助手回复段”计算学习信号：

- 用户段（`<user>...</user>`）对应的 target 会被置为 pad token（id=0）
- 系统提示段（`<system>...</system>`）对应的 target 也会被置为 pad token
- loss 会忽略 pad token

这能显著减少“模型复读用户问题”的倾向。

实现细节（重要）：

- SFT 的 assistant mask 是通过扫描模板文本里的 `<assistant> ... </assistant>` 区间生成的（而不是按 `role` 字段直接生成）。
- 因此**不建议**你在 `prompt/response/content` 里手工插入 `<assistant>` / `</assistant>` / `<user>` 等标签；这些标签会被当作普通文本进入模板，从而影响 mask 与训练目标。
- 推荐做法：数据里只放“纯文本内容”，让 `train` 在内部统一套模板。

---

## 3) 数据质量建议（强烈建议）

### 3.1 SFT 数据（推荐）

- 避免大段无意义标点/重复字符
- 回复尽量完整、语气一致
- 减少模板化重复（例如每条都“好的。\n...”）
- 统一全角/半角标点（可选）
- 尽量让训练集包含你希望模型具备的“说话风格”

### 3.2 LM 文本（推荐）

- 去掉乱码与极长重复行
- 统一换行（`\r\n` → `\n`）
- 尽量不要把多个领域混在同一个超大文件里（可用目录分组）

---

## 4) DPO：偏好对齐数据

### 4.1 DPO 训练命令

```bash
cargo run --release --bin train -- --dpo --dpo-data dpo_data.jsonl --artifact-dir ./tmp/dpo_model --dpo-beta 0.1 --dpo-kl-weight 0.1 --num-epochs 30 --batch-size 16 --backend gpu --force
```

### 4.2 DPO 数据格式

DPO 训练需要包含偏好信息的数据，每行一个 JSON 对象，包含以下字段：

```json
{
  "prompt": "你是谁？",
  "chosen": "我是一个用 Rust 训练出来的小模型。",
  "rejected": "我不知道。"
}
```

字段说明：
- `prompt`：用户输入/问题
- `chosen`：偏好的回复（更好的回答）
- `rejected`：被拒绝的回复（较差的回答）

实现细节（重要）：

- 当前 DPO 的实现是把 token 序列直接拼接：`full_chosen = encode(prompt) + encode(chosen)`，`full_rejected = encode(prompt) + encode(rejected)`。
- DPO 路径**不会**像 SFT 一样自动套 `<user>/<assistant>` 模板，也不会自动插入 BOS/EOS 控制字符。
- 这意味着：如果你希望 DPO 对齐到“对话模型”的输出格式，建议把 `prompt` 写成你希望的前缀（例如以 `<assistant>` 结尾的对话模板前缀），并让 `chosen/rejected` 是对该前缀的续写内容。

### 4.3 数据质量建议

- 确保 `chosen` 和 `rejected` 的差异有意义
- 偏好应该基于相同的 prompt
- 避免极端的偏好差异（例如一个非常好，一个完全无关）
- 偏好应该反映真实的用户偏好

---

## 5) 最小校验（Smoke Test）

当你拿到一份新数据集时，建议先做：

1) 只训练 200~1000 条记录：

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --sft-max-records 500 --artifact-dir ./tmp/sft_smoke --num-epochs 1 --max-seq-len 64 --force --reset-tokenizer
```

2) 用 chat 推理快速看输出是否“像人话”：

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_smoke --use-best --chat --prompt "你是谁？" -n 80 -t 0.7 -p 0.9 -k 20 -r 1.1 --punctuation-penalty 1.8 -s 1
```

---

## 5) 多模态数据

### 5.1 支持的图像格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

### 5.2 自动流水线

在训练与推理中，Sage 提供自动化的图像预处理：
1. **自动加载**：从 `image_path` 指定的路径实时读取。
2. **Resizing**：自动缩放为 224x224 像素。
3. **归一化**：将像素值缩放到 0.0 - 1.0 范围。

### 5.3 数据准备示例

你可以使用 `gen_data --multimodal` 快速生成测试集：

```bash
cargo run --release --bin gen_data -- --multimodal --out data/mm_test.jsonl --count 100
```
生成的数据会自动包含一个模拟的 `assets/sample_image.jpg` 路径。
