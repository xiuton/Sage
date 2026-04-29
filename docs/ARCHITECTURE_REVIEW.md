# Sage 架构审阅：功能合理性、规范与取舍

本文档从维护者与进阶使用者角度，概括 **已实现能力**、**实现上值得注意的点**、**与常见 Rust/深度学习工程的对比**，以及 **对小参数模型场景的冗余与缺口**。适用于 README / PROJECT_STATUS 的补充阅读，非对外「教程」。

---

## 1. 已实现功能概览（按域）

| 域 | 主要内容 | 合理性简评 |
|----|----------|------------|
| **核心模型** | `core/model.rs`：TransformerEncoder + LM head；多档 `ModelConfig`；`core/attention.rs`：Flash Attention、GQA 自定义注意力模块 + SwiGLU MLP；`core/kv_cache.rs`：推理加速 KV 缓存；`core/multimodal.rs`：CNN 视觉编码器与门控融合层 | CNN 编码器比简单投影更具特征提取能力；门控融合支持动态权衡图文权重；Flash/GQA 通过 SageTransformerEncoder 灵活切换注意力类型。 |
| **分词** | `core/tokenizer.rs`：字符级 + BPE | 符合小模型与大一点实验需求；BPE 对中文更实用。 |
| **数据** | `data/data.rs`：LM/SFT 数据集、mask、Batcher、**自动化图像加载流水线** | 自动处理 `image_path` 字段，实现端到端多模态训练。 |
| **训练** | `training/training.rs`：自实现训练循环；`streaming.rs` 大语料；`vram_probe.rs` GPU 预检（从小到大探测 + OOM 自动重启）；`distributed.rs` 分布式权重同步；`dpo.rs` DPO偏好对齐；`precision.rs` 混合精度训练 (FP16/BF16)；`qlora.rs` QLoRA (INT4 + LoRA) | 实现了完整的混合精度基础设施（损失缩放、动态 scale 调整）和 QLoRA 训练显存估算。 |
| **推理** | `inference/generation.rs`、`bin/infer.rs`：Beam Search、Speculative Decoding 推测解码 | 包含高级终端交互模式 (`--terminal`) 与多模态推理；Speculative Decoding 草稿+验证双模型加速。 |
| **量化** | `quantization/`：支持 INT8/INT4 模拟量化推理 | 可在不改变硬件的前提下评估量化对精度和模型体积的影响。 |
| **LoRA** | `training/lora.rs` | **已深度集成到训练路径**；支持仅微调低秩矩阵，大幅节省资源。 |
| **API** | `api/`、`bin/api_server.rs` | 工程化完整闭环；支持 OpenAI 标准格式。 |
| **工具** | `gen_data`、`export`、`benchmark`、`accuracy_eval` | `gen_data` 整合了原有的所有数据生成逻辑。 |

---

## 2. 实现与设计上需注意的点

1. **显存探测 vs 正式训练**  
   探测从最小配置 `(batch=1, seq_len=16)` 开始逐步增大，找到不 OOM 的最佳配置后自动重启进程（通过 `SAGE_VRAM_CONFIG` 环境变量传递配置），确保正式训练在干净的 WGPU 状态下运行。探测包含 50% 安全系数（batch/seq_len 各减半），为 Adam 优化器额外显存留出空间。行为与心理预期需在文档中对齐（见 [TRAINING_PHASES.md](TRAINING_PHASES.md)）。

2. **`--training-mode`（general/code/math）** | **Training_mode**  
   主要影响 **内置样例/模板与部分数据过滤路径**，不是独立的第二套优化器或损失；命名上易让人以为「互斥训练算法」，实为 **场景化默认值/模板**。文档与 README 已逐步澄清；代码注释可继续写清。

3. **继续训练**  
   恢复 **权重** 为主；优化器/调度器状态未完整恢复时，严格复现实验需自行控制学习率日程（见 PROJECT_STATUS）。

4. **WGPU**  
   首次运行编译、线程与队列约束多与 CUDA 栈不同；故障排查集中在 TROUBLESHOOTING 与 train 内注释。

5. **Edition 2024**  
   `Cargo.toml` 使用 `edition = "2024"`，需使用支持该版本的 Rust 工具链；团队协作时应在 README 标明 **最低 Rust 版本**（建议在后续迭代中写入 `rust-toolchain.toml` 或 README「环境」节）。

---

## 3. Rust 与常见开源工程规范

**已对齐或较好的部分**

- 库与二进制分离：`lib.rs` 导出能力，`bin/*.rs` 做 CLI。  
- 按域分目录：`core/`、`training/`、`data/`、`api/`，与许多中型 Rust 应用一致。  
- 集成测试位于 `tests/`。

**目录结构说明**

| 目录 | 作用 | 关键文件 |
|------|------|----------|
| `src/bin/` | 命令行工具入口 | train.rs, infer.rs, api_server.rs, image_gen.rs 等 |
| `src/core/` | 核心模型和功能实现 | model.rs, multimodal.rs, tokenizer.rs, kv_cache.rs |
| `src/training/` | 训练相关功能 | training.rs, lora.rs, dpo.rs, distributed.rs |
| `src/inference/` | 推理相关功能 | generation.rs, kernels.rs, lazy_load.rs |
| `src/data/` | 数据处理 | data.rs |
| `src/api/` | API 服务实现 | mod.rs |
| `src/utils/` | 辅助工具 | logger.rs, metrics.rs, performance.rs |
| `examples/` | 示例代码 | multimodal_quickstart.rs（多模态快速入门示例） |
| `inference/configs/` | 推理配置文件 | config_16B.json, config_1B.json |
| `models/` | 模型配置文件 | config.json, tokenizer.json |
| `tests/` | 集成测试 | test_*.rs 系列测试文件 |

**可继续加强（不强制一次做完）**

- **`rustfmt` / `clippy`**：在 CI 或贡献说明中固定；当前未在本仓库强制。  
- **`examples/`**：常见 Rust 库用于演示 API；本项目以 `bin` 为主，可接受；若库用户增多可考虑 `examples/` 调用 `sage` 库 API。  
- **`benches/`**：性能回归可选。  
- **错误类型**：部分路径仍 `expect`/`panic`；长期可向 `Result` 与用户可读错误枚举收敛。  
- **文档内代码路径**：README 中历史链接曾指向 `src/model.rs`，实际为 `src/core/model.rs`；已在新版 README 中统一修正。

---

## 4. 目录结构 vs「流行」Rust 深度学习项目

当前结构：

```text
src/
  lib.rs
  core/          # 核心层：模型定义、自定义注意力(Flash/GQA)、Tokenizer、KV Cache、多模态实现
  data/          # 数据层：Dataset、Batcher、数据预处理
  inference/     # 推理层：生成策略、采样、懒加载、推理内核
  training/      # 训练层：训练循环、DPO、调度器、流式、显存探测、LoRA
  transformer/   # 底层基础组件
  quantization/  # 量化支持
  configs/       # 配置管理
  api/           # API 服务实现
  tools/         # 开发辅助工具
  utils/         # 辅助工具 (logger, performance, error, etc.)
  bin/           # 命令行入口 (train, infer, api_server, etc.)
tests/           # 集成测试
docs/            # 文档说明
```

与 **Burn / 纯 Rust ML 仓库** 常见模式（`model/`、`data/`、`training/`、单一或少量 bin）**兼容**。  
与 **PyTorch 单仓库 HuggingFace 风格**（`src` 极薄、业务全在 Python）**不同**，这是语言生态差异，不是问题。

---

## 5. 对「参数量较小」场景：哪些偏「多余」、哪些仍值得保留？

**可视为「可选 / 偏大」的能力（依你的目标裁剪）**

- **极大 `model-size` 预设**（如 671B）：除非做配置演示，实际训练不现实。  
- **完整 HTTP API 服务**：若只做研究 + 命令行推理，可不用部署 `api_server`。  
- **LoRA 模块（未接主训练路径）**：维护成本存在；只做 10M 级全量微调时可忽略该文件，或未来接好再启用。  
- **模型远程下载工具**：无中心模型发布时可当占位。

**仍建议保留**

- **流式数据**：小模型也会遇到「语料大、内存紧」。  
- **BPE + SFT mask**：小模型要中文可用性时很关键。  
- **显存探测 + 梯度累积**：在消费级 GPU 上训 10M～30M 仍实用（已增强为从小到大探测 + 自动重启 + 安全系数）。  
- **checkpoint / best_model**：实验管理基础能力。
- **混合精度训练 + QLoRA**：在 8GB 消费级 GPU 上训 100M 模型时显存节省效果显著。
- **SageError 统一错误处理**：结构化错误信息含上下文和修复建议，提升调试效率。

---

## 6. 建议优先补足的能力（与 PROJECT_STATUS 呼应）

- 更严格的 **SFT loss mask**（ignore_index / 显式 mask 张量）。  
- **Checkpoint 恢复优化器与调度器**（实验可复现）。  
- **最低 Rust 版本 / toolchain 文件** 与 CI（fmt + test）。  
- 评测：**困惑度、固定集上简单指标**，便于小模型迭代对比。

---

## 7. 相关文档

- [TRAINING_PHASES.md](TRAINING_PHASES.md)：正式 vs 非正式训练细节  
- [PROJECT_STATUS.md](PROJECT_STATUS.md)：路线图与完成度  
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)：WGPU、显存、DataLoader  
