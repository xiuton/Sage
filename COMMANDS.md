# Sage 命令手册

本文档基于当前仓库代码与 `--help` 输出整理，覆盖所有已支持的命令与参数，并提供推荐用法。

二进制列表：

- `train`：训练
- `infer`：推理/对话
- `gen_sft`：生成 SFT jsonl 数据（测试/压测用）

相关文档：

- 数据格式：`DATA_FORMAT.md`
- 排错：`TROUBLESHOOTING.md`
- 路线图：`PROJECT_STATUS.md`

---

## 1) train（训练）

运行：

```bash
cargo run --release --bin train -- [OPTIONS]
```

### 1.1 训练模式选择（互斥建议）

训练输入可以来自三类来源（建议只选其一）：

- **纯文本语料 LM 训练**
  - `--corpus <path>`：单文件（默认 `corpus_cn.txt`）
  - `--corpus-dir <dir>`：目录（递归读取所有 `.txt`）
- **SFT 训练（JSONL）**
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

**A. 目录语料训练（续写 LM）**

```bash
cargo run --release --bin train -- --corpus-dir D:\data\cn_texts --artifact-dir ./tmp/lm_cn --num-epochs 5 --max-seq-len 64
```

**A2. 目录语料训练（限制读取大小 + 快速验证）**

```bash
cargo run --release --bin train -- --corpus-dir D:\data\cn_texts --artifact-dir ./tmp/lm_cn_quick --num-epochs 1 --max-seq-len 64 --max-bytes 10000000 --force --reset-tokenizer
```

**B. 使用 JSONL 做 SFT**

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_cn --num-epochs 1 --max-seq-len 64 --force --reset-tokenizer
```

如果你没有现成的数据，可以先用 `gen_sft` 生成一份：

```bash
cargo run --release --bin gen_sft -- --out sft_demo_5000.jsonl --count 5000 --seed 42
```

**B3. SFT 数据 smoke test（先跑 1000 条）**

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --sft-max-records 1000 --artifact-dir ./tmp/sft_smoke --num-epochs 1 --max-seq-len 64 --force --reset-tokenizer
```

**B4. 使用 BPE 分词器训练 SFT（减少重复字符问题）**

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_bpe --use-bpe --bpe-vocab-size 5000 --num-epochs 50 --batch-size 32 --max-seq-len 256 --force --reset-tokenizer
```

**B2. JSONL 的 schema 示例**

prompt/response：

```json
{"prompt":"你是谁？","response":"我是一个用 Rust 训练出来的小模型。"}
```

messages（多轮也可）：

```json
{"messages":[
  {"role":"user","content":"你是谁？"},
  {"role":"assistant","content":"我是一个用 Rust 训练出来的小模型。"}
]}
```

**C. 从已有模型继续训练**

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_cn --continue --num-epochs 1
```

**D. 从 checkpoint 继续训练**

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --artifact-dir ./tmp/sft_cn --resume-epoch 3 --num-epochs 2
```

> 注意：当前继续训练仅恢复“模型权重”，不会恢复优化器状态。
>
> `--force` / `--continue` / `--resume-epoch` 同时出现时：会优先走“继续训练/恢复训练”的逻辑。

### 1.3 参数说明（来自 `train --help`）

`cargo run --bin train -- --help` 输出的参数如下（含补充解释）：

- `--corpus <CORPUS>`：单文件语料路径（不传时默认 `corpus_cn.txt`）。用于语言模型训练，文件内容会被按行分割处理。
- `--corpus-dir <CORPUS_DIR>`：语料目录（递归收集 `.txt`）。会自动遍历目录下的所有 .txt 文件，按文件名排序后拼接成训练数据。
- `--max-bytes <MAX_BYTES>`：最大读取字节数（默认 `50000000`，传 `0` 表示不限制）。
- `--stream`：启用流式读取与磁盘 token 缓存（推荐用于大语料/大 JSONL，避免一次性读入内存）。
- `--stream-direct`：不落盘边读边训（仅 SFT；需同时启用 `--stream`）。
- `--sft-jsonl <SFT_JSONL>`：SFT 数据 jsonl 文件。每行一条 JSON 记录，支持 `{"prompt":"...","response":"..."}` 或 `{"messages":[...], "id": ...}` 格式。
- `--sft-sample`：使用内置 prompt/response 样例数据（约 100 条）。用于快速测试 SFT 训练流程。
- `--sft-sample-messages`：使用内置 messages 样例数据（约 100 条）。用于测试多轮对话格式的 SFT 训练。
- `--sft-max-records <SFT_MAX_RECORDS>`：最多读取多少条 SFT 记录（默认 `0` 不限制）。
- `--artifact-dir <ARTIFACT_DIR>`：输出目录（默认 `./tmp/sage_model_formal`）。存放训练产物，包括模型权重、tokenizer、配置和checkpoint。
- `--num-epochs <NUM_EPOCHS>`：训练 epoch（默认 `50`）。完整数据集被训练的轮数。更多的 epochs 可以提高模型性能，但也增加训练时间。建议监控验证损失来确定合适的 epoch 数。
- `--batch-size <BATCH_SIZE>`：batch size（默认 `32`）。控制每次训练迭代处理的样本数量。较大的 batch-size（32-64）可以提高训练稳定性，减少梯度噪声，但需要更多内存；较小的 batch-size（8-16）可以节省内存，但可能导致训练不稳定。建议根据你的 GPU/CPU 内存情况选择合适的值。
- `--lr <LR>`：学习率（默认 `0.0001`）。控制模型权重更新的步长，过大会导致训练不稳定，过小会使训练过慢收敛。
- `--max-seq-len <MAX_SEQ_LEN>`：序列长度（默认 `256`）。限制输入序列的最大长度，过长会增加内存使用，过短可能丢失上下文信息。
- `--use-bpe`：使用 BPE 分词器替代字符级分词器（推荐用于减少重复字符问题）。
- `--bpe-vocab-size <BPE_VOCAB_SIZE>`：BPE 词汇表大小（默认 `5000`，仅在启用 `--use-bpe` 时有效）。
- `--force`：即使输出目录已有模型也强制重新训练/覆盖。会删除现有的模型文件。
- `--continue`：从 `<artifact-dir>/model.mpk` 加载权重继续训练。用于增量训练或微调。
- `--resume-epoch <RESUME_EPOCH>`：从 `<artifact-dir>/checkpoint/model-<epoch>.mpk` 加载权重继续训练。用于从特定checkpoint恢复训练。
- `--reset-tokenizer`：忽略已有 `tokenizer.json`，从当前语料重新构建词表。当语料发生重大变化时使用。

补充说明：

- `--corpus-dir` 会递归读取所有 `.txt` 并按路径排序拼接；每个文件后追加换行。
- `--max-bytes` 同时作用于 `--corpus` / `--corpus-dir` / `--sft-jsonl`（用于限制读入大小）。
- `--batch-size` 控制每次训练迭代处理的样本数量。较大的 batch-size（32-64）可以提高训练稳定性，减少梯度噪声，但需要更多内存；较小的 batch-size（8-16）可以节省内存，但可能导致训练不稳定。建议根据你的 GPU/CPU 内存情况选择合适的值。
- `--use-bpe` 启用 BPE 分词器，能显著减少高频字符的重复问题（如中文中的"解"字符重复），建议在遇到重复生成问题时使用。
- `--bpe-vocab-size` 控制 BPE 词汇表大小，较大的词汇表能提供更好的 token 多样性，但会增加模型参数量。
- 语料变化较大时建议 `--reset-tokenizer`，否则新字符会大量映射到 `unk`，影响训练质量。
- 训练与推理尽量使用同一个 `artifact-dir` 的 `tokenizer.json`，否则 token id 对不上，输出会严重跑偏。

### 1.4 输出产物说明

训练成功后，`artifact-dir` 内会出现：

- `config.json`：训练配置（包含 `model` 超参）
- `tokenizer.json`：Tokenizer 词表
- `model.mpk`：最终权重
- `checkpoint/`：epoch checkpoint（`model-<epoch>.mpk` 等）
- `best_model.mpk`：根据 valid loss 选出来的最佳权重（如果能找到 valid 指标）

当启用 `--stream` 时，`artifact-dir` 还会包含：

- `cache/tokens.bin`：u32 小端序 token id 序列
- `cache/mask.bin`：u8 mask 序列（SFT：只学习助手回复；LM：全为 1）

当启用 `--stream --stream-direct` 时：

- 不会生成 `cache/` 中间产物（直接流式训练）

---

## 2) infer（推理 / 对话）

运行：

```bash
cargo run --bin infer -- [OPTIONS]
```

### 2.1 常用示例

**A. 单次生成（续写）**

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_cn --prompt "天地玄黄" -n 80
```

**A2. 单次生成（控制可复现）**

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_cn --prompt "天地玄黄" -n 80 -s 42
```

**B. Chat 模式（助手回复）**

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_cn --use-best --chat --prompt "你是谁？" -n 80
```

**B2. Chat 模式（更干净的回复）**

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_cn --use-best --chat --prompt "给我学习 Rust 的建议" \
  -n 120 -t 0.7 -k 20 -p 0.9 -r 1.1 --punctuation-penalty 1.8 -s 42
```

**C. 交互式对话**

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_cn --use-best --chat --interactive
```

**D. 采样参数推荐（减少标点、减少重复）**

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_cn --use-best --chat --prompt "给我学习 Rust 的建议" \
  -n 120 -t 0.7 -k 20 -p 0.9 -r 1.1 --punctuation-penalty 1.6 -s 42
```

**E. context window（避免超长输入/生成导致越界）**

```bash
cargo run --bin infer -- --model-dir ./tmp/sft_cn --prompt "天地玄黄" --context-len 64 -n 80
```

如果 `context-len` 超过训练时 `max_seq_len`，程序会自动截断到 `max_seq_len` 并提示。

### 2.2 参数说明（来自 `infer --help`）

- `--prompt <PROMPT>`：输入提示词（单次模式必需；交互模式下从 stdin 输入）。
- `-n, --num-tokens <NUM_TOKENS>`：生成 token 数（默认 `50`）。
- `-t, --temperature <TEMPERATURE>`：温度（默认 `0.8`，越大越随机）。
- `-k, --top-k <TOP_K>`：Top-K（默认 `10`）。
- `-p, --top-p <TOP_P>`：Top-P（默认 `0.9`）。
- `-r, --repetition-penalty <REPETITION_PENALTY>`：重复惩罚（默认 `1.1`，越大越抑制重复）。
- `--punctuation-penalty <PUNCTUATION_PENALTY>`：标点惩罚（默认 `1.3`，越大越抑制标点/连续标点）。
- `-s, --seed <SEED>`：随机种子（可复现）。
- `--model-dir <MODEL_DIR>`：模型目录（默认 `./tmp/sage_model_formal`）。
- `--use-best`：优先加载 `best_model.mpk`（若不存在则回退到 `model.mpk`）。
- `--context-len <CONTEXT_LEN>`：上下文窗口长度（默认 `0`=自动使用 `model.max_seq_len`；如果传得比 `max_seq_len` 大，会自动截断避免越界）。
- `-i, --interactive`：交互模式。
- `--chat`：chat 模式（使用 `用户/助手` 模板生成，并从输出中提取“助手回复”部分）。

补充说明：
- `-t, --temperature`：控制生成随机性。值越大（>1.0）生成更随机/创造性，值越小（<1.0）生成更保守/确定性。0.7-0.9 适合大多数应用。
- `-k, --top-k`：只从概率最高的 K 个 token 中采样。较小的值（5-10）使输出更保守，较大的值（20-50）增加多样性。
- `-p, --top-p`：核采样，只从累积概率达到 P 的 token 中采样。与 top-k 结合使用效果更好，通常设为 0.8-0.95。
- `-r, --repetition-penalty`：对已生成的 token 施加惩罚，减少重复。1.0 表示无惩罚，>1.0 会抑制重复，1.1-1.3 是常用范围。
- `--punctuation-penalty`：专门对标点符号施加额外惩罚，减少"标点雨"问题。1.0 表示无惩罚，>1.0 会抑制标点，1.3-2.0 是常用范围。
- `-s, --seed`：设置随机种子保证结果可重现。用于调试或生成一致的结果。- `--chat` 会把输入包成 `用户：<prompt>\n助手：`，然后从生成结果里提取最后一次出现的“助手：”之后的内容作为回复。
- 交互模式 `--interactive` 下会维护一个简单的文本 history（多轮），并受到 `context-len` 截断影响（只取最后 N 个 token）。
- 若 `--use-best` 但目录中没有 `best_model.mpk`，会自动回退到 `model.mpk`。

---

## 3) gen_sft（生成 SFT JSONL）

运行：

```bash
cargo run --release --bin gen_sft -- --out <path> --count <n> --seed <seed>
```

这是一个用于快速生成**可训练** SFT 数据的工具，生成格式为 `{"messages":[...], "id": ...}`，与 `train --sft-jsonl` 兼容。

参数：

- `--out <path>`：输出文件（默认 `sft_demo_5000.jsonl`）
- `--count <n>`：生成条数（默认 `5000`）
- `--seed <seed>`：随机种子（默认 `42`）

建议用法：

- 先生成小文件做 smoke test（例如 200 条）
- 确认训练/推理闭环后再生成 5000/20000 条

示例：

```bash
cargo run --release --bin gen_sft -- --out sft_demo_20000.jsonl --count 20000 --seed 123
```

---

## 4) 常见问题

### 4.1 为什么会出现大量标点？

小模型 + 字符级 tokenizer + 语料风格会导致标点容易被高概率采样。可以通过：

- 增大 `--punctuation-penalty`
- 降低 `--temperature` / `--top-p`
- 增大 `--repetition-penalty`
- 使用更多真实 SFT 数据训练

### 4.2 为什么 `--context-len` 不能超过 `max_seq_len`？

因为位置 embedding 只为 `max_seq_len` 个位置训练/初始化，超过会越界。infer 会自动截断，但如果你确实需要更长上下文，请在训练时提高 `--max-seq-len` 并重新训练。

### 4.3 Windows 下遇到 LNK1104 无法写入 infer.exe？

这是 Windows 常见的文件锁问题：之前运行的 exe 进程未完全退出，或被杀软/索引占用。

可尝试：

- 关闭残留的 `infer.exe` / `train.exe` 进程
- 运行 `cargo clean` 后重试
- 使用临时 target 目录：`cargo run --target-dir .\\target_tmp --bin infer -- ...`
