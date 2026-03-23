# Sage 故障排查（TROUBLESHOOTING）

本文档收录在 Sage 项目中最常见、最影响开发体验的问题，并给出可操作的排查步骤与解决方案。

---

## 1) Windows：`LNK1104 cannot open file infer.exe/train.exe`

### 现象

- `cargo run --bin infer ...` 或 `cargo run --bin train ...` 失败
- 链接器报错：`LNK1104 cannot open file '...infer.exe'`

### 原因

Windows 上可执行文件被占用（常见于：上一次运行的进程未退出、被杀软/索引占用、IDE 终端残留）。

### 解决方案（按推荐顺序）

1. 关闭残留的 `infer.exe` / `train.exe` 进程
2. 清理构建缓存

```bash
cargo clean
```

3. 使用临时 target 目录（绕过锁定）

```bash
cargo run --target-dir .\target_tmp --bin infer -- --help
```

---

## 2) 推理崩溃：`context_len` / `max_seq_len` 越界

### 现象

- 运行 `infer` 时 panic
- 类似报错：`Index 32 must be less than axis length 32`

### 原因

位置 embedding 只为训练时的 `max_seq_len` 个位置分配/训练，推理输入序列长度超过它会越界。

### 解决方案

- `infer` 已实现自动保护：如果 `--context-len` 大于 `model.max_seq_len`，会自动截断并提示。
- 若你需要更长上下文：
  1) 训练时提高 `--max-seq-len`
  2) 重新训练得到新的模型权重

---

## 3) 推理输出“标点雨/重复”

### 现象

输出大量 `，。。，。` 或重复短片段，内容不连贯。

### 主要原因

- 模型较小（~0.001B）且 tokenizer 为字符级，容易把高频标点当作“安全预测”。
- 训练数据风格偏书面/标点密集，或 SFT 数据规模太小。

### 快速缓解（推理参数）

推荐按顺序尝试：

1) 降低温度：

```bash
... -t 0.7
```

2) 降低 top-p：

```bash
... -p 0.85
```

3) 增大重复惩罚：

```bash
... -r 1.2
```

4) 增大标点惩罚：

```bash
... --punctuation-penalty 2.0
```

### 根治方向（训练侧）

- 增加高质量 SFT 数据量（至少几千～几万条）
- 减少模板化重复样本
- 未来升级 tokenizer（BPE/SentencePiece）✅ **BPE已完成**
- **使用 BPE 分词器**（推荐）：BPE 能显著减少高频字符重复问题

```bash
cargo run --release --bin train -- --sft-jsonl your_data.jsonl --use-bpe --bpe-vocab-size 5000 --num-epochs 50 --batch-size 32 --max-seq-len 256
```

---

## 4) 训练提示：`experiment logger: Failed to install the file logger.`

### 现象

训练输出中出现上面的提示，但训练仍然继续，最终能保存模型。

### 说明

这是训练日志落盘器在 Windows 下安装失败的告警（通常与权限/路径/文件句柄有关），不影响训练主流程。

### 建议

- 可忽略（当前代码不依赖该日志落盘器）
- 如果你需要训练日志文件，后续可以把日志输出重定向到文件：

```bash
cargo run --release --bin train -- ... > train.log 2>&1
```

---

## 5) 训练崩溃：`Ratio should be between 0 and 1 inclusively.`

### 现象

训练过程中 panic，堆栈指向 `ratatui` 的 gauge/progress 渲染。

### 原因

训练进度条的 `processed/total` 比值超过 1（常见于 “total 估算偏小但训练继续推进” 的流式数据场景）。

### 解决方案

- 已在当前版本修复（流式 DataLoader 的 total 不再小于 processed）。
- 如果你仍遇到该问题：请更新到最新代码并重新编译。

---

## 6) 训练速度很慢 / 卡住

### 现象

- CPU 后端训练慢
- 训练输出进度条更新不明显

### 排查建议

1) 先做小规模 smoke test，确认流程没问题：

```bash
cargo run --release --bin gen_sft -- --out sft_smoke_200.jsonl --count 200 --seed 1
cargo run --release --bin train -- --sft-jsonl sft_smoke_200.jsonl --sft-max-records 200 --artifact-dir ./tmp/smoke --num-epochs 1 --max-seq-len 64 --force --reset-tokenizer
```

2) 降低 `--max-bytes` 或减少 `--num-epochs`
3) 后续考虑切换 GPU 后端（计划任务）

---

## 7) BPE 分词器相关问题

### 7.1 训练时 BPE 编译错误

#### 现象
使用 `--use-bpe` 时出现编译错误，提示 `ModelWrapper`、`TrainerWrapper` 等相关错误。

#### 原因
tokenizers crate API 版本兼容性问题。

#### 解决方案
- 确保使用项目指定的 `tokenizers = "0.19.1"` 版本
- 如果仍有问题，暂时使用字符级分词器：移除 `--use-bpe` 参数

### 7.2 BPE 训练速度慢

#### 现象
BPE 训练比字符级慢很多。

#### 原因
BPE 需要从语料中学习合并规则，计算复杂度较高。

#### 建议
- 对于小数据集（<1M tokens），可以接受
- 对于大语料，考虑预训练 BPE 或使用现有 BPE 模型
- 或者在小数据集上训练 BPE，然后用于大语料

### 7.3 BPE 模型推理不一致

#### 现象
使用 BPE 训练的模型推理结果与字符级差异很大。

#### 原因
BPE 改变了 token 粒度，模型需要重新学习 token 间的关系。

#### 建议
- 这是正常现象，BPE 通常需要更多训练数据和 epochs
- 增加 `--num-epochs` 到 50+，`--batch-size` 到 32+
- 使用更大的 `--bpe-vocab-size`（如 8000-10000）以获得更好平衡

---

## 8) 训练进度条显示混乱/与日志混合

### 现象

- 训练时进度条与日志输出混合，显示混乱
- 终端输出难以阅读

### 原因

Burn 框架的 TUI 进度条在某些终端环境下会与标准输出混合。

### 解决方案

1) 禁用进度条显示：

```bash
cargo run --release --bin train -- --no-progress [其他参数]
```

2) 使用快速开发模式（保留进度条显示）：

```bash
cargo run --release --bin train -- --quick-dev [其他参数]
```

4) **如果问题持续**：进度条实现可能与特定终端不兼容，可以考虑使用 `--no-progress` 选项

---

## 9) 训练无法用 Ctrl+C 中断

### 现象

- 按 Ctrl+C 无法中断训练进程
- 进程继续运行

### 原因

之前的版本没有实现信号处理。

### 解决方案

✅ **已修复**：新版本支持 Ctrl+C 中断，会优雅关闭并保存检查点。

如果遇到中断无效的情况：
1) 等待几秒钟让程序完成清理
2) 如果仍然无效，使用任务管理器强制结束进程

---

## 10) 快速开发模式说明

### 快速开发模式的特点

使用 `--quick-dev` 参数启用快速开发模式，会自动设置以下参数以加快训练速度：

- 训练轮数：3（而非默认50）
- 批次大小：8（而非默认32）
- 学习率：0.001（而非默认0.0001）
- 保留进度条显示（便于观察训练进度）

### 使用场景

- 快速测试训练流程是否正常
- 开发和调试阶段的快速迭代
- 验证数据格式和模型配置
- CI/CD 环境中的自动化测试

### 示例

```bash
cargo run --release --bin train -- --sft-jsonl sft_demo_5000.jsonl --artifact-dir ./tmp/quick_test --quick-dev --force --reset-tokenizer
```

### 注意事项

- 快速开发模式仅用于测试，不要用于生产训练
- 训练结果可能不如完整参数调优的模型
- 如果需要高质量模型，请使用完整训练参数

---

## 11) GPU 训练崩溃：`wgpu error: Validation Error` / `Not enough memory left`

### 现象

- 训练启动后很快 panic
- 典型报错包含：`In Device::create_buffer` 与 `Not enough memory left`

### 原因

- GPU 显存不足（batch 太大、序列太长、模型太大、或者同时跑了其他占显存程序）。
- 旧版本在构建 batch 时会产生大量小的 GPU buffer 分配，容易触发显存分配失败。

### 解决方案

1) 降低显存压力（推荐按顺序尝试）：

- 降低 `--batch-size`（例如 8/16）
- 降低 `--max-seq-len`（例如 64/128）
- 不要开启 `--fast`（会提高 batch/worker/lr）

2) 使用 CPU 后端验证流程：

```bash
cargo run --release --bin train -- --backend cpu [其他参数]
```

3) 更新到最新代码并重新编译：

- 已优化 batch 构建方式，减少 GPU 端临时分配，缓解该类 OOM。

---

## 12) 流式数据加载问题

### 12.1 流式加载速度慢

#### 现象

- 使用 `--stream` 或 `--stream-direct` 时训练速度明显慢于普通模式

#### 原因

- 流式加载需要实时处理数据，IO 操作可能成为瓶颈
- `--stream-direct` 模式下，每次迭代都需要重新读取和处理数据

#### 解决方案

1) 对于大数据集，优先使用 `--stream`（落盘缓存）而非 `--stream-direct`
2) 确保存储设备性能足够（SSD 优于 HDD）
3) 考虑增加 `--batch-size` 以提高数据处理效率

### 12.2 流式加载内存占用仍然很高

#### 现象

- 使用 `--stream` 后内存占用仍然超出预期

#### 原因

- 数据预处理和 tokenization 过程中可能产生临时内存占用
- 批处理过程中需要在内存中保存当前批次的数据

#### 解决方案

1) 降低 `--batch-size` 以减少内存占用
2) 确保 `--max-seq-len` 设置合理，避免过长序列
3) 对于非常大的数据集，考虑分批次处理或使用更小的模型配置

### 12.3 流式加载报错：`Failed to read JSONL`

#### 现象

- 流式加载过程中出现 JSON 解析错误

#### 原因

- JSONL 文件格式不正确或包含无效的 JSON 记录
- 文件编码问题（非 UTF-8）

#### 解决方案

1) 验证 JSONL 文件格式是否正确，确保每行都是有效的 JSON
2) 使用 UTF-8 编码保存 JSONL 文件
3) 对于大文件，可以使用工具（如 `jq`）验证文件格式

```bash
# 使用 jq 验证 JSONL 文件
cat your_data.jsonl | jq -c '. | select(. != null)'
```
