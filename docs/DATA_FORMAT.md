# Sage 数据格式规范（DATA_FORMAT）

本文档描述 Sage 当前支持的训练数据输入格式，并给出推荐实践、常见陷阱与校验方法。

---

## 0) 总览

Sage 目前支持两类训练目标：

1. **LM（Language Modeling / 续写）**：从纯文本中学习预测下一个 token。
2. **SFT（Supervised Fine-Tuning / 指令微调）**：从对话/问答数据中学习“用户提问 → 助手回答”。

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
- `system` 角色用于设置系统提示词（可选）
- 多轮对话推荐顺序：`system → user → assistant → user → assistant ...`

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

## 4) 最小校验（Smoke Test）

当你拿到一份新数据集时，建议先做：

1) 只训练 200~1000 条记录：

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --sft-max-records 500 --artifact-dir ./tmp/sft_smoke --num-epochs 1 --max-seq-len 64 --force --reset-tokenizer
```

2) 用 chat 推理快速看输出是否“像人话”：

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_smoke --use-best --chat --prompt "你是谁？" -n 80 -t 0.7 -p 0.9 -k 20 -r 1.1 --punctuation-penalty 1.8 -s 1
```
