# Sage 故障排查（TROUBLESHOOTING）

本文档收录在 Sage 项目中最常见、最影响开发体验的问题，并给出可操作的排查步骤与解决方案。

---

## 1) Windows：`LNK1104 cannot open file infer.exe/train.exe` / `os error 4551`

### 现象

- `cargo run --bin infer ...` 或 `cargo run --bin train ...` 失败
- 链接器报错：`LNK1104 cannot open file '...infer.exe'`
- 或运行时报错：`could not execute process ... (never executed)`，并提示 `应用程序控制策略已阻止此文件。 (os error 4551)`

### 原因

- `LNK1104`：Windows 上可执行文件被占用（常见于：上一次运行的进程未退出、被杀软/索引占用、IDE 终端残留）。
- `os error 4551`：系统的应用程序控制策略（AppLocker/WDAC/ASR 等）拦截了新生成的 `target\\...\\*.exe` 执行。

### 解决方案（按推荐顺序）

1. 关闭残留的 `infer.exe` / `train.exe` 进程
2. 清理构建缓存

```bash
cargo clean
```

3. 使用自定义 target 目录（同时可绕过“文件锁定”和部分策略拦截点）

```powershell
cargo run --target-dir "$env:LOCALAPPDATA\cargo-target\sage" --bin infer -- --help
```

4. 如果仍是 `os error 4551`

- 尝试把工程移动到 `C:\Users\<你>\...`（部分公司策略会对 D:\ / 共享盘 / 下载目录更严格）。
- 如环境受管控，需要管理员将你的 target 目录或 Rust 构建产物加入白名单。

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
cargo run --release --bin gen_data -- --out sft_smoke_200.jsonl --count 200 --seed 1
cargo run --release --bin train -- --sft-jsonl sft_smoke_200.jsonl --sft-max-records 200 --output-dir ./tmp/smoke --num-epochs 1 --max-seq-len 64 --force --reset-tokenizer
```

2) 降低 `--max-bytes` 或减少 `--num-epochs`
3) 可切换 GPU 后端（如果显卡/驱动支持 WGPU）：

```bash
cargo run --release --bin train -- --backend gpu [其他参数]
```

若 GPU 显存吃紧，优先用 `--no-auto-vram` + `--gradient-accumulation` 手动把物理 batch 降下来（等效 batch 保持不变）。

---

## 7) BPE 分词器相关问题

### 7.1 训练时 BPE 编译错误

#### 现象
使用 `--use-bpe` 时出现编译错误，提示 `ModelWrapper`、`TrainerWrapper` 等相关错误。

#### 原因
tokenizers crate API 版本兼容性问题。

#### 解决方案
- 确保使用项目指定的 `tokenizers = "0.19.*"` 版本
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

- 训练轮数：1（而非默认50）
- 批次大小：CPU 通常为 4；GPU 通常为 8（以实际日志输出为准）
- 学习率：显著提高（CPU 通常为 1e-2；GPU 通常为 2e-2）
- 保留进度条显示（便于观察训练进度）

### 使用场景

- 快速测试训练流程是否正常
- 开发和调试阶段的快速迭代
- 验证数据格式和模型配置
- CI/CD 环境中的自动化测试

### 示例

```bash
cargo run --release --bin train -- --sft-jsonl sft_demo_5000.jsonl --output-dir ./tmp/quick_test --quick-dev --force --reset-tokenizer
```

### 注意事项

- 快速开发模式仅用于测试，不要用于生产训练
- 训练结果可能不如完整参数调优的模型
- 如果需要高质量模型，请使用完整训练参数

---

## 11) GPU 训练崩溃：`wgpu error: Validation Error` / `Out of Memory` / `Not enough memory left`

### 现象

- 训练启动后很快 panic
- 典型报错包含：`In Device::create_buffer` 与 `Not enough memory left`
 - 或出现：`wgpu error: Out of Memory`
 - 或出现：`wgpu error: Validation Error` / `Encoder is invalid`

### 原因

- GPU 显存不足（batch 太大、序列太长、模型太大、或者同时跑了其他占显存程序）。
- 旧版本在构建 batch 时会产生大量小的 GPU buffer 分配，容易触发显存分配失败。
- 如果训练日志中出现 `Loss: NaN`，随后也可能触发 WGPU 的 `Validation Error / Encoder is invalid`（非法数值导致后续 GPU 命令编码失败）。

### 解决方案

1) **优先依赖自动显存探测**（默认开启）：

GPU 训练会自动从小配置开始探测 (batch=1, seq_len=16 → 逐步增大)，找到不 OOM 的最佳配置后自动重启进程重置 WGPU 状态。探测包含 50% 安全系数，为 Adam 优化器留出额外显存。

2) 手动降低显存压力：

- 降低 `--batch-size`（例如 8/16）
- 降低 `--max-seq-len`（例如 64/128）
- 不要开启 `--fast`（会提高 batch/worker/lr）
- 保持等效 batch：例如把 `--batch-size 8` 改为 `--batch-size 2 --gradient-accumulation 4`

3) 用 `--no-auto-vram` 手动控制（跳过探测阶段的自动调整）：

```bash
cargo run --release --bin train -- --backend gpu --no-auto-vram --batch-size 2 --gradient-accumulation 4 [其他参数]
```

4) 使用更小的模型规模：

```bash
cargo run --release --bin train -- --model-size 30m --backend gpu [其他参数]
```

5) 使用 CPU 后端验证流程：

```bash
cargo run --release --bin train -- --backend cpu [其他参数]
```

6) 更新到最新代码并重新编译。

7) 如果出现 `Loss: NaN`

- 优先检查训练数据是否包含模板标签作为普通文本
- 先做 smoke test：`--sft-max-records 200 --num-epochs 1 --max-seq-len 64`

---

## 12) Windows：`STATUS_STACK_OVERFLOW` 推理/训练崩溃

### 现象

```text
thread 'main' has overflowed its stack
error: process didn't exit successfully (exit code: 0xc00000fd, STATUS_STACK_OVERFLOW)
```

### 原因

Windows debug 模式默认栈空间仅 **1MB**，加载大模型（100M+ 参数）时 `spawn_blocking` 等异步调用或深层递归张量操作会耗尽栈空间。

### 解决方案

已通过 `.cargo/config.toml` 全局配置栈空间为 8MB：

```toml
[target.'cfg(windows)']
rustflags = ["-C", "link-args=/STACK:8388608"]
```

重建项目即可生效：

```bash
cargo clean
cargo build --release --bin infer
```

如果仍遇到栈溢出（如加载 1B+ 模型），可手动增加到 16MB：

```bash
$env:RUSTFLAGS="-C link-args=/STACK:16777216"
cargo build --release --bin infer
```

---

## 13) Rust build scripts 构建失败（沙箱环境）

### 现象

```text
called `Result::unwrap()` on an `Err` value: Os { code: 0, kind: Uncategorized, message: "操作成功完成。" }
error: failed to run custom build command for `proc-macro2` / `serde` / `quote`
```

### 原因

Trae 沙箱环境中 `cargo clean` 后重新编译时，build scripts（proc-macro crates）调用 `process::Command` 受沙箱限制，`cargo check` 或 `cargo build` 会报错。

### 解决方案

- 不要手动执行 `cargo clean`，使用 `cargo build --lib --release` 增量编译
- 如已执行 `cargo clean`：重启终端 / IDE 即可恢复

---

## 14) 推理输出全是标点/乱码

### 现象

模型推理输出为大量碎片标点（。，；、），无连贯语义。

### 原因

模型训练不足——epoch 太少或数据量太小，Perplexity 仍然很高（>100）。

### 解决方案

- 增加训练数据量和 epoch
- 降低 temperature（0.5~0.7）使输出更保守
- 使用 `--model-size` 选择更适配硬件的规模

## 15) 分布式训练（`--distributed`）说明

当前版本的分布式训练属于**框架/占位实现**：CLI 参数已提供，但未完成真实的多 GPU 训练与权重/梯度同步。

建议：
- 需要稳定训练时不要使用 `--distributed`，直接用单 GPU（`--backend gpu`）或 CPU
- 如果你在受限环境里需要“多卡训练”，建议先等待该模块补齐或自行实现同步逻辑后再启用

---

## 13) DPO训练问题

### 13.1 DPO数据格式错误

#### 现象
- 使用 `--dpo` 参数时出现数据解析错误
- 报错：`Missing field 'prompt'` 或 `Missing field 'chosen'`

#### 原因
- DPO 数据文件格式不正确
- 缺少必要的 `prompt`、`chosen`、`rejected` 字段

#### 解决方案
1) 验证 DPO 数据格式：
```json
{
  "prompt": "你是谁？",
  "chosen": "我是一个用 Rust 训练出来的小模型。",
  "rejected": "我不知道。"
}
```

2) 先用小文件做 smoke test（例如只保留前 100 行）：
```bash
# PowerShell:
#   Get-Content .\dpo_data.jsonl -TotalCount 100 | Set-Content .\dpo_data_small.jsonl
--dpo-data dpo_data_small.jsonl
```

### 13.2 DPO训练不稳定

#### 现象
- DPO 训练时损失波动很大
- 模型生成质量下降

#### 原因
- beta 参数设置不当
- 偏好数据质量问题

#### 解决方案
1) 调整 beta 参数：
```bash
--dpo-beta 0.05  # 更小的 beta 值
```

2) 确保偏好数据质量良好：
- `chosen` 和 `rejected` 的差异要有意义
- 避免极端的偏好差异

---

## 14) 多模态推理问题

### 14.1 图像加载/预处理错误

#### 现象
- 多模态推理时出现图像加载/预处理错误
- 报错：`Failed to process image`

#### 原因
- 图像格式不支持
- 图像文件损坏或路径不正确

#### 解决方案
1) 确保使用支持的图像格式（JPEG、PNG）
2) 检查传入的图片路径/权限，确保文件可读取
3) 目前训练命令不提供 `--image-size` 参数；如需调整输入分辨率，请在数据预处理阶段完成

### 14.2 多模态融合错误

#### 现象
- 多模态融合层出现维度不匹配错误
- 报错：`Shape mismatch`

#### 原因
- 文本特征和图像特征维度不匹配

#### 解决方案
1) 检查模型配置，确保多模态配置正确
2) 确保图像编码器输出维度与文本特征维度兼容

---

## 15) 量化相关说明

当前版本的量化属于**框架/体积估算**：仓库内存在 `QuantizedModel` 包装器与量化模式枚举，但未提供稳定的“量化推理/量化导出”命令行参数，也未实现权重量化与算子替换带来的真实加速。

因此：
- 文档中若出现 `--quantize-mode`、`--no-quantize-output` 等参数，属于过时内容或其他分支实现，可忽略
- 如需部署侧量化，请先补齐真实量化实现后再对齐文档与 CLI

---

## 16) 流式数据加载问题

### 16.1 流式加载速度慢

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
