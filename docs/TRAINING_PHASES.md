# 训练阶段说明：显存探测 vs 正式训练

本文档说明 Sage 在一条 `train` 命令执行过程中，**哪些步骤属于「非正式训练」**（仅为估显存或诊断），**哪些步骤才是进入完整训练循环的正式训练**，以及日志何时出现。

---

## 1. 术语

| 术语 | 含义 |
|------|------|
| **正式训练** | 当前版本为**自实现训练循环**：按 epoch 遍历 DataLoader、计算 train/valid 损失、写 checkpoint/最佳模型等。代码路径主要在 `src/training/training.rs` 的 `run_training`。 |
| **非正式 / 预检步骤** | **不**进入上述完整训练循环，仅为选择超参或做一次极小计算，**不**产生 epoch 级训练日志，**不**迭代完整数据集。 |

两者都与「模型前向、反向」可能用到同一套算子，但**正式训练**才是你通常所说的「在训练模型」。

---

## 2. 一条命令里的时间线（GPU + 默认开启显存探测）

典型顺序如下（未使用 `--no-auto-vram` 时）：

1. **加载/构建分词器、准备 `TrainingConfig`**  
   - 非训练；仅 I/O 与配置。

2. **（可选）GPU 显存探测 — 非正式训练**  
   - 实现位置：`src/training/vram_probe.rs` 中的 `probe_first_fitting_config`。  
   - 行为：对多组 `(物理 batch_size, max_seq_len)` 依次构造 **单次** `TrainStep::step`（一次前向 + 交叉熵 + 反向），**不**调用优化器 `step`，**不**进入完整训练循环，**不**扫描语料 epoch。  
   - 目的：在显存允许范围内自动选择 batch / seq，并配合梯度累积维持等效 batch。  
   - 控制台可能长时间只有探测相关输出（WGPU 首次编译内核也会发生在此阶段）。  
   - 详见 `train` 在开启探测时打印的「GPU 显存探测（不是正式训练）」横幅说明。

3. **正式训练开始**  
   - 调用 `train` / `train_from_cache` / `train_with_loaders` → `run_training`。  
   - 从这里开始才是 **正式训练**：多 epoch、完整（或流式）数据迭代、验证集评估、checkpoint 等。  
   - `--no-progress` 会减少训练过程中的中间日志输出。
   - 备注：部分启动日志仍会提到 “Burn 训练 TUI”，这是历史遗留提示；当前训练主路径不依赖 Burn Learner/TUI，主要表现为进度条与 epoch/batch 文本日志。

4. **训练结束**  
   - 保存 `model.mpk`、可选 `best_model.mpk` 等。

若使用 `--no-auto-vram`，则 **不存在** 第 2 步，在配置就绪后直接进入第 3 步。

---

## 3. CPU 后端

- 默认**不做**上述 GPU 显存探测（无 WGPU 估显存步骤）。  
- 流程在数据就绪后直接进入「正式训练」（以及你自己选用的 `--quick-dev` 等短跑配置）。

---

## 4. 「快速开发」模式算不算正式训练？

**算正式训练**，只是刻意缩短、用于冒烟：

| 参数 | 作用 |
|------|------|
| `--quick-dev` / `--ultra-quick` | 减少 epoch、批量、数据量等，**仍然会进入完整训练循环**。 |

因此它们 **不是**「显存探测那种非正式步骤」，而是 **极短的正式训练**；适合验代码/环境，不适合当作生产级训练。

---

## 5. 如何对照日志判断当前阶段？

- **看到**「显存探测」「一步前向+反向」「seq_len=… 正在构建模型」等文案 → **探测阶段**，尚无 epoch 训练日志。  
- **看到** `=== Epoch X/Y ===`、或 `train/valid` 下损失日志文件持续更新 → **已进入正式训练**。  
- 若探测阶段感觉「卡住」，常见原因是 WGPU 首次编译；探测实现里带有周期性心跳打印，详见 `vram_probe.rs` 与 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

---

## 6. 相关参数速查

| 参数 | 影响 |
|------|------|
| `--backend gpu` | 启用 WGPU；默认会跑显存探测（除非 `--no-auto-vram`）。 |
| `--no-auto-vram` | 跳过显存探测，直接使用命令行 `batch-size` / `max-seq-len`（可配合 `--gradient-accumulation`）。 |
| `--no-progress` / `--fast` | 减少训练过程中的中间输出（**不**影响探测阶段是否打印说明）。 |
| `--tui` / `--force-tui` | 强制启用训练过程的进度显示（在部分终端可能显示不佳）；本质是让 `no_progress=false`。 |
| `--gradient-accumulation` | 在 CPU 或 `--no-auto-vram` 的 GPU 上由你指定；自动探测成功时由程序写入等效累积步数。 |
| `--distributed` | 分布式训练当前仅有框架/占位实现，尚未实现真实的多 GPU 训练与权重同步；不建议用于实际训练。 |
| `--devices` | 指定设备列表（逗号分隔，示例：`gpu:0,gpu:1`）。仅在 `--distributed` 时读取。 |
| `--dpo` | 启用DPO偏好对齐训练模式。 |
| `--dpo-beta` | DPO损失的beta参数（默认0.1）。 |

更完整的命令说明见 [COMMANDS.md](COMMANDS.md)，训练实操见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。
