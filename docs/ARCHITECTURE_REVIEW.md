# Sage 架构审阅：功能合理性、规范与取舍

本文档从维护者与进阶使用者角度，概括 **已实现能力**、**实现上值得注意的点**、**与常见 Rust/深度学习工程的对比**，以及 **对小参数模型场景的冗余与缺口**。适用于 README / PROJECT_STATUS 的补充阅读，非对外「教程」。

---

## 1. 已实现功能概览（按域）

| 域 | 主要内容 | 合理性简评 |
|----|----------|------------|
| **核心模型** | `core/model.rs`：Encoder-style Transformer + LM head；多档 `ModelConfig`（1M～预设大到 671B 量级）；`core/multimodal.rs`：图像编码器与多模态融合层 | 小模型档（default/10m/30m）与 Burn Learner 配套合理；极大档多为「占位/教学」配置，单机 WGPU 通常无法训练。多模态模块提供图像输入支持。 |
| **分词** | `core/tokenizer.rs`：字符级 + BPE | 符合小模型与大一点实验需求；BPE 对中文更实用。 |
| **数据** | `data/data.rs`：LM/SFT 数据集、mask、Batcher | 与 CrossEntropy pad 忽略配套；严谨 loss mask 仍可增强（见 README 已知限制）。 |
| **训练** | `training/training.rs`：Learner、checkpoint、best 模型；`streaming.rs` 大语料；`vram_probe.rs` GPU 预检；`distributed.rs` 分布式训练；`dpo.rs` DPO偏好对齐 | 主路径清晰；WGPU 下 DataLoader `num_workers=0` 属必要取舍。分布式训练和DPO模块完善了训练能力。 |
| **推理** | `core/generation.rs`、`bin/infer.rs` | 与训练闭环匹配。 |
| **量化** | `quantization/`：INT4/INT8量化、动态量化 | 对部署/实验有意义；支持多种量化模式，提升推理效率。 |
| **LoRA** | `training/lora.rs` | 当前多为模块草案，**未与 `train` 主路径深度集成**；对小模型全量微调场景为「备用/超前」。 |
| **API** | `api/`、`bin/api_server.rs` | 工程化完整闭环有用；若个人仅本地 `infer`，可视为可选组件。 |
| **工具** | `gen_sft`、`gen_web_sft`、`export`、`benchmark`、`accuracy_eval` | 按需求选用；benchmark/accuracy_eval 对实验记录有帮助。 |

---

## 2. 实现与设计上需注意的点

1. **显存探测 vs 正式训练**  
   探测只做 **单次** `step`，不经过 `Learner`；行为与心理预期需在文档中对齐（见 [TRAINING_PHASES.md](TRAINING_PHASES.md)）。

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
  core/          # 模型、分词、生成、KV、多模态
  training/      # Learner、流式、LoRA、vram_probe、分布式训练、DPO
  data/
  inference/
  api/
  tools/
  utils/
  quantization/  # INT4/INT8量化、动态量化
  bin/           # 多个入口
tests/
docs/
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
- **显存探测 + 梯度累积**：在消费级 GPU 上训 10M～30M 仍实用。  
- **checkpoint / best_model**：实验管理基础能力。

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
