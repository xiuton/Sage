# 项目状态与路线图（Sage）

本文档用于记录：当前仓库已经完成了哪些工程能力、还缺哪些关键能力、以及推荐的下一步迭代顺序。

> 建议把本文件当作“项目看板/路线图”，每完成一项就更新状态。

---

## 1) 当前定位

Sage 是一个 **0.001B（~1M 参数）** 级别的 Rust 小模型训练工程，用来验证：

- 模型定义 → 数据管线 → 训练 → 保存 → 推理 → chat → SFT 的全流程
- 在 Rust 生态中以 Burn 作为深度学习框架实现可扩展工程结构

它不是“生产级 AI 助手”，但已经具备把它进一步工程化与扩容的骨架。

---

## 2) 已完成任务（Done）

### 2.1 工程结构

- 库化组织：核心逻辑在 `src/`，二进制入口在 `src/bin/`
- 三个可执行程序：
  - `train`：训练入口（LM/SFT）
  - `infer`：推理入口（续写/Chat/交互）
  - `gen_sft`：生成 SFT JSONL（测试/压测）
- 文档体系：
  - [README.md](README.md)
  - [COMMANDS.md](COMMANDS.md)
  - 本文件 [PROJECT_STATUS.md](PROJECT_STATUS.md)

### 2.2 模型（Transformer LM）

- Token Embedding + Position Embedding
- Transformer Encoder + LM Head
- TrainStep / ValidStep 接入 Burn Learner
- 参数规模约 0.001B

### 2.3 Tokenizer（字符级 + BPE + 中文）

- Unicode `char` 字符级词表，天然支持中文
- BPE (Byte Pair Encoding) 分词器，支持自定义词汇表大小
- 特殊 token：`pad/unk/bos/eos`
- 词表可保存/加载（`tokenizer.json` + `tokenizer.json.meta`）
- BPE 能显著减少高频字符重复问题（如中文"解"字符）

### 2.4 训练（LM/SFT）

- LM 训练：单文件 `--corpus`、目录多文件 `--corpus-dir`
- 读取上限：`--max-bytes`
- SFT 训练：`--sft-jsonl`（兼容 `prompt/response` 或 `messages`）
- 内置 SFT 样例：`--sft-sample` / `--sft-sample-messages`
- SFT “只学助手回复”：
  - 通过 mask 把非助手回复位置的 target 置为 pad，并在 loss 中忽略 pad token
- 流式读取与大语料训练：
  - `train --stream`：逐行读取/分块处理，写入 `artifact-dir/cache/`
  - 训练时使用 memmap 数据集读取 token cache，降低峰值内存
  - `train --stream --stream-direct`：不落盘边读边训（当前仅支持 SFT）
- 训练产物：
  - `config.json` / `tokenizer.json` / `model.mpk`
  - `checkpoint/` 保存 epoch 权重
  - 导出 `best_model.mpk`（从 valid loss 自动选择）
- 继续训练（权重级恢复）：
  - `--continue` 从 `model.mpk`
  - `--resume-epoch` 从 `checkpoint/model-<epoch>.mpk`
- **快速开发模式**：`--quick-dev`（3轮 + 优化参数，适合测试迭代）
- **可中断训练**：Ctrl+C 优雅关闭并保存检查点
- **可选进度条**：`--no-progress` 禁用 TUI 显示，支持终端兼容性配置

### 2.5 推理（Sampling + Chat）

- 可控采样：temperature / top-k / top-p
- 重复惩罚：`repetition_penalty`
- 标点惩罚：`punctuation_penalty`（缓解标点雨）
- 上下文窗口：`context_len`（自动截断到 `max_seq_len`，避免越界）
- Chat 模式：`--chat`
  - 输入按 `用户：...\n助手：` 模板组织
  - 输出提取“助手：”之后内容作为回复
- 交互模式：`--interactive`

### 2.6 数据生成工具

- `gen_sft` 支持生成可训练 JSONL：
  - `{"messages":[...], "id": ...}`
  - 用于 smoke test / 压测训练管线

### 2.7 硬件加速支持

- GPU后端：支持 WGPU 训练和推理（`--backend gpu`）
- CPU后端：NdArray（`--backend cpu`，默认）
- 自动工作线程配置：基于 CPU 核心数优化数据加载并行度

---

## 3) 关键待办（Next / TODO）

下面按优先级给出“最影响质量与可用性”的任务序列。

### P0（强烈建议优先做）

1. **更严格的 SFT loss mask**
   - 当前用 pad token 近似 mask，可能引入偏差（特别是 pad=0 会影响分布）
   - 目标：支持 ignore_index 或显式 loss mask（只对 assistant token 计算 loss）
2. **更强的停止策略（Stop sequences）**
   - 目标：推理时遇到 `\u{0003}` 或 “用户：” 立即停止生成
   - 能显著改善 chat 体验与输出长度可控性
3. **数据流式读取**（已完成）
   - 已实现 `train --stream`（落盘 token cache + memmap dataset）
   - 已实现 `train --stream --stream-direct`（不落盘、边读边训；SFT）
   - 后续优化：更高效的 token cache 格式/索引（便于跳读、并行、复用）

### P1（提升工程可用性）

1. **恢复优化器状态**
   - 现有 `--continue/--resume-epoch` 只恢复模型权重
   - 目标：恢复 optimizer/scheduler 状态，真正实现断点续训一致性
2. **更标准的对话模板**
   - 目标：兼容常见 chat template（system/user/assistant），并在训练与推理共用同一模板
3. **更丰富的 metrics 与评测**
   - Perplexity、简单 QA accuracy、固定样例回放
4. **训练日志落盘问题**
   - Windows 下 file logger 安装失败的告警（不影响训练，但影响体验）

### P2（提升模型能力/性能）

1. **Tokenizer 升级（BPE/SentencePiece）** ✅ **已完成**
   - 字符级对中文会导致序列变长、泛化差
   - BPE 分词器已实现，支持自定义词汇表大小
2. **GPU 后端优化**
   - 当前默认 NdArray CPU
   - 目标：WGPU 训练/推理稳定可用，并加速
3. **更大的模型配置**
   - 提供多个预设 config（~1M、~10M、~30M）按硬件选择

---

## 3.1 里程碑（Milestones）

建议把后续工作拆成可验收的里程碑（每个里程碑都应可运行验证）。

### M0：当前版本（已具备）

- 可训练：LM / SFT（JSONL）均能跑通
- 可推理：续写 / chat / 交互
- 可复现：支持 seed、可持久化产物（模型/词表/配置/checkpoint）

### M1：助手可用性增强（P0）

- Stop sequences（遇到 EOS 或 “用户：” 停止）
- 更严格的 loss mask（真正只对 assistant 计算）
- 流式读取大 JSONL（不占满内存）

验收标准：

- `infer --chat` 输出不会无限续写到下一轮“用户：”
- 大 JSONL（>1GB）可以在受控内存下训练（可分块）

### M2：工程体验增强（P1）

- 恢复优化器状态
- 更稳定的日志与实验记录（Windows 可落盘）
- 基础评测脚本（固定样例回放 + 简单指标）

### M3：能力/性能增强（P2）

- Tokenizer 升级（BPE/SentencePiece）✅ **已完成**
- GPU 后端稳定训练/推理（WGPU）
- 多档模型配置（~1M/~10M/~30M）

---

## 3.2 建议拆解（Task Breakdown）

为了方便开发推进，给出 P0 的可拆解子任务建议：

### Stop sequences

- 推理端增加 stop 条件：遇到 `\u{0003}` 立即停止
- 可选：遇到 “用户：” 字符序列也停止
- 增加 CLI 参数：`--stop-on-user`（可选，默认开）

### 更严格的 loss mask

- 在 dataset 里把 “不学习位置” 标记成 `ignore_index`
- 训练端 loss 支持 ignore_index（或显式 loss mask）
- 保持与当前 `pad_tokens` 方案兼容（迁移期）

### 流式读取

- 对 JSONL 逐行读取并逐步 tokenization
- 支持 `--max-records` 与 `--max-bytes` 在流式场景的语义一致
- 可选：把 token 缓存在磁盘（中间产物）以复用

### P1（提升工程可用性）

1. **恢复优化器状态**
   - 现有 `--continue/--resume-epoch` 只恢复模型权重
   - 目标：恢复 optimizer/scheduler 状态，真正实现断点续训一致性
2. **更标准的对话模板**
   - 目标：兼容常见 chat template（system/user/assistant），并在训练与推理共用同一模板
3. **更丰富的 metrics 与评测**
   - Perplexity、简单 QA accuracy、固定样例回放
4. **训练日志落盘问题**
   - Windows 下 file logger 安装失败的告警（不影响训练，但影响体验）

### P2（提升模型能力/性能）

1. **Tokenizer 升级（BPE/SentencePiece）**
   - 字符级对中文会导致序列变长、泛化差
2. **GPU 后端优化**
   - 当前默认 NdArray CPU
   - 目标：WGPU 训练/推理稳定可用，并加速
3. **更大的模型配置**
   - 提供多个预设 config（~1M、~10M、~30M）按硬件选择

---

## 4) “更像 AI 助手”需要什么（现实说明）

要达到“像成熟 AI 助手那样回复”，通常需要：

- 更大的模型（至少千万级参数起步）
- 更高质量、更大规模的 SFT 数据（几万到几十万条高质量问答/对话）
- 更合理的 tokenizer（BPE/SentencePiece）
- 更严格的 loss mask + stop sequences
- 可能还需要 RLHF / DPO 等偏好对齐（此项目可后续探索）

现阶段 Sage 的价值在于：把上述能力拆解为工程任务，逐步迭代实现。

---

## 5) 建议的迭代顺序（建议）

### 当前状态评估
项目已具备完整的训练-推理闭环，包括：
- ✅ 基础Transformer模型架构（约0.001B参数）
- ✅ 字符级和BPE分词器支持
- ✅ LM和SFT训练模式
- ✅ 流式数据加载（支持大规模数据）
- ✅ GPU/CPU后端支持
- ✅ 推理和Chat交互模式

### 优化优先级（按影响程度排序）

#### P0：核心质量提升（立即实施）
1. **Stop sequences** - 解决推理时输出无限续写到下一轮的问题
2. **严格的SFT loss mask** - 真正只对assistant部分计算loss，提升训练效率
3. **优化器状态恢复** - 实现真正的断点续训，支持长时间训练

#### P1：工程体验提升（短期实施）
4. **更标准的对话模板** - 统一训练和推理的chat template
5. **更丰富的metrics** - 增加perplexity等评估指标
6. **训练日志优化** - 解决Windows下file logger问题

#### P2：能力扩展（中期实施）
7. **GPU性能优化** - 优化WGPU训练速度和稳定性
8. **更大模型配置** - 提供~10M、~30M参数的配置选项
9. **更多训练模式** - 支持代码生成、数学推理等特定领域训练

---

## 6) 大模型训练指南

### 训练方式选择

#### 1. 自回归预训练（LM）
适合：通用语言理解、文本生成基础能力  
数据：纯文本语料（如书籍、文章、代码等）  
命令：
```bash
cargo run --release --bin train -- --corpus your_text.txt --artifact-dir ./tmp/lm_model --num-epochs 100 --batch-size 32 --use-bpe --bpe-vocab-size 10000
```

#### 2. 指令微调（SFT）
适合：问答、对话、特定任务  
数据：JSONL格式的指令-回复对  
命令：
```bash
cargo run --release --bin train -- --sft-jsonl sft_data.jsonl --artifact-dir ./tmp/sft_model --num-epochs 50 --batch-size 16 --use-bpe --bpe-vocab-size 10000
```

#### 3. 代码生成训练
适合：代码补全、编程助手  
数据：代码语料或代码问答对  
命令：
```bash
# 代码预训练
cargo run --release --bin train -- --corpus-dir ./code_corpus --artifact-dir ./tmp/code_model --num-epochs 200 --batch-size 32 --use-bpe --bpe-vocab-size 20000

# 代码SFT
cargo run --release --bin train -- --sft-jsonl code_instruction.jsonl --artifact-dir ./tmp/code_sft --num-epochs 100 --batch-size 8 --use-bpe --bpe-vocab-size 20000
```

### 数据准备建议

#### 通用文本数据
- 来源：书籍、新闻、百科、网页等
- 清洗：去除噪声、特殊字符、重复内容
- 大小：建议至少1GB以上文本（约100M tokens）

#### SFT数据
- 格式：`{"prompt": "...", "response": "..."}` 或 `{"messages": [...]}`
- 质量：重点关注回复的准确性、完整性、逻辑性
- 数量：建议至少10,000条高质量样本
- 多样性：覆盖不同领域和任务类型

#### 代码数据
- 来源：GitHub仓库、编程书籍、教程
- 格式：支持纯代码文件或代码问答对
- 预处理：保留缩进和格式，添加语言标记

### 训练参数调优

#### 学习率
- 小模型（<10M）：1e-4 ~ 5e-4
- 中等模型（10M-100M）：5e-5 ~ 2e-4
- 使用学习率衰减（如cosine decay）

#### 批量大小
- 根据GPU/CPU内存调整
- GPU：16-64（视显存而定）
- CPU：8-32（视内存而定）

#### 序列长度
- 通用文本：256-512
- 代码：512-1024
- 长文本：1024-2048（需要更大内存）

### 硬件配置建议

#### CPU训练
- 适合：小型模型（<10M参数）、开发测试
- 核心数：8核以上
- 内存：16GB以上

#### GPU训练（推荐）
- 适合：中等及以上模型
- 显存：至少4GB（推荐8GB以上）
- GPU型号：NVIDIA RTX 3060/3070/4070或更高
- 使用`--backend gpu`启用

### 训练监控与评估

#### 训练中监控
- 观察loss曲线（应平稳下降）
- 检查验证集性能
- 定期保存checkpoint

#### 推理测试
- 使用固定提示测试生成质量
- 检查回复相关性、连贯性
- 评估特殊任务表现（如代码生成的可执行性）

### 模型部署建议

#### 推理优化
- 使用量化（未来支持）
- 启用KV缓存（未来支持）
- 调整生成参数（temperature、top-p等）

#### 服务化
- 封装为API服务
- 添加请求队列和负载均衡
- 支持批量推理

---

## 7) 下一步行动建议

### 立即行动（本周）
1. **实现Stop sequences** - 在`generation.rs`中添加停止条件
2. **优化SFT loss mask** - 在`data.rs`中实现ignore_index支持

### 短期目标（本月）
3. **完善对话模板** - 统一训练和推理的格式
4. **添加更多metrics** - 实现perplexity计算

### 中期目标（季度）
5. **扩展模型规模** - 提供10M、30M参数配置
6. **优化GPU性能** - 提升WGPU训练速度
7. **增加训练模式** - 支持更多特定领域训练

### 长期规划
8. **RLHF/DPO支持** - 实现偏好对齐
9. **多模态能力** - 支持图像输入
10. **分布式训练** - 支持多GPU/多机器训练
