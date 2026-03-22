# Sage 模型训练与部署完全指南

本文档提供了 Sage 模型从环境准备、数据准备、训练配置到模型部署和推理使用的完整实战指南。

## 目录

1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [训练配置与执行](#训练配置与执行)
4. [模型部署](#模型部署)
5. [推理使用](#推理使用)
6. [性能优化](#性能优化)
7. [故障排除](#故障排除)

---

## 1. 环境准备

### 1.1 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **内存**: 最低 8GB，推荐 16GB+
- **GPU**: 可选，支持 WGPU（Windows DirectX/Metal/Vulkan）
- **Rust**: 版本 1.75+（推荐使用 rustup 安装）

### 1.2 安装依赖

```bash
# 安装 Rust（如果尚未安装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 克隆项目
git clone https://github.com/yourusername/sage.git
cd sage

# 安装依赖
cargo build --release
```

### 1.3 验证安装

```bash
# 运行示例训练（超快速模式）
cargo run --release --bin train -- --ultra-quick --sft-sample
```

---

## 2. 数据准备

### 2.1 支持的数据格式

Sage 支持以下数据格式：

#### 2.1.1 SFT 数据格式（推荐）

```json
{"prompt": "你的问题", "response": "回答内容"}
```

或多轮对话格式：

```json
{"messages": [
    {"role": "system", "content": "系统提示词"},
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "助手回答"}
]}
```

#### 2.1.2 纯文本语料

直接提供文本文件或目录，每行一个样本。

### 2.2 数据准备示例

#### 示例 1：创建 SFT 训练数据

```bash
# 创建训练数据文件
echo '{"prompt": "你好，请介绍一下你自己。", "response": "我是一个基于Transformer架构的语言模型..."}' > train_data.jsonl
echo '{"prompt": "什么是机器学习？", "response": "机器学习是人工智能的一个分支..."}' >> train_data.jsonl
```

#### 示例 2：使用预定义样本

```bash
# 使用内置样本进行训练
cargo run --release --bin train -- --sft-sample
```

### 2.3 实际操作演示

#### 演示 1：创建自定义训练数据

```bash
# 创建包含多个对话样本的训练数据文件
echo '{"prompt": "你好，请介绍一下你自己。", "response": "我是一个基于Transformer架构的语言模型，旨在理解和生成自然语言。我可以回答问题、提供信息、进行对话，并帮助完成各种语言相关的任务。"}' > my_training_data.jsonl

# 添加更多训练样本
echo '{"prompt": "什么是机器学习？", "response": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式和规律，而无需明确编程。核心概念包括：监督学习（从标记数据中学习）、无监督学习（从未标记数据中发现模式）、强化学习（通过试错学习最优策略）等。"}' >> my_training_data.jsonl

echo '{"prompt": "如何学习编程？", "response": "学习编程的有效方法包括：1. 选择合适的编程语言；2. 从基础概念开始学习；3. 实践项目；4. 参与开源社区；5. 不断学习和练习。最重要的是保持耐心和持续的实践。"}' >> my_training_data.jsonl
```

#### 演示 2：验证训练数据

```bash
# 查看创建的数据文件内容
Get-Content my_training_data.jsonl
```

**输出示例**：
```
{"prompt": "你好，请介绍一下你自己。", "response": "我是一个基于Transformer架构的语言模型，旨在理解和生成自然语言。我可以回答问题、提供信息、进行对话，并帮助完成各种语言相关的任务。"}
{"prompt": "什么是机器学习？", "response": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式和规律，而无需明确编程。核心概念包括：监督学习（从标记数据中学习）、无监督学习（从未标记数据中发现模式）、强化学习（通过试错学习最优策略）等。"}
{"prompt": "如何学习编程？", "response": "学习编程的有效方法包括：1. 选择合适的编程语言；2. 从基础概念开始学习；3. 实践项目；4. 参与开源社区；5. 不断学习和练习。最重要的是保持耐心和持续的实践。"}
```

#### 演示 3：使用自定义数据训练

```bash
# 使用自定义数据进行训练
cargo run --release --bin train -- --sft-jsonl my_training_data.jsonl --model-size 10m --num-epochs 30 --batch-size 4 --training-mode general --force
```

**注意**：如果自定义数据格式不正确，可能会导致"语料为空"错误。确保JSON格式正确且文件编码为UTF-8。

---

## 3. 训练配置与执行

### 3.1 基本训练命令

```bash
# 基本训练（使用 CPU）
cargo run --release --bin train -- --sft-jsonl train_data.jsonl --num-epochs 50 --batch-size 32

# 使用 GPU 训练
cargo run --release --bin train -- --sft-jsonl train_data.jsonl --backend gpu --num-epochs 50 --batch-size 64
```

### 3.2 模型大小选择

```bash
# 默认模型（约 1M 参数）
cargo run --release --bin train -- --model-size default

# 10M 参数模型
cargo run --release --bin train -- --model-size 10m

# 30M 参数模型
cargo run --release --bin train -- --model-size 30m
```

### 3.3 训练模式选择

```bash
# 通用对话模式
cargo run --release --bin train -- --training-mode general

# 代码生成模式（优化代码生成）
cargo run --release --bin train -- --training-mode code

# 数学推理模式（优化数学问题解决）
cargo run --release --bin train -- --training-mode math
```

### 3.4 开发模式

```bash
# 快速开发模式（1轮训练，小批量）
cargo run --release --bin train -- --quick-dev

# 超快速开发模式（1轮训练，极小批量，仅100条数据）
cargo run --release --bin train -- --ultra-quick

# 高性能模式（更快的数据加载和批处理）
cargo run --release --bin train -- --fast
```

### 3.5 训练配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-epochs` | 训练轮数 | 50 |
| `--batch-size` | 批处理大小 | 32 |
| `--lr` | 学习率 | 1e-4 |
| `--max-seq-len` | 最大序列长度 | 256 |
| `--artifact-dir` | 输出目录 | ./tmp/sage_model_formal |
| `--backend` | 后端（cpu/gpu） | cpu |
| `--model-size` | 模型大小 | default |
| `--training-mode` | 训练模式 | general |

### 3.6 训练监控

训练过程中会显示：
- 训练损失和验证损失
- 学习率变化
- 训练速度（tokens/sec）
- 每轮时间统计
- 最终 perplexity 指标

### 3.7 实际操作演示

#### 演示 1：超快速开发模式

```bash
# 使用内置样本进行超快速训练（1轮，极小批量）
cargo run --release --bin train -- --ultra-quick --sft-sample --training-mode general
```

**输出示例**：
```
使用默认模型配置（约1M参数）
使用CPU后端进行训练...
设备信息: Cpu
词表大小: 56
模型参数总量: 840248 (约 0.001B)
启用超快速开发模式：1轮训练，极小批量(2)，极高学习率，只用100条数据，适合闪电验证
未发现已训练模型，开始正式训练...
使用 8 个工作线程进行数据加载
训练数据: 133 tokens, 验证数据: 15 tokens
批量大小: 2, 序列长度: 256

性能统计:
总训练时间: 81.0234ms
总处理 tokens: 148
处理速度: 1826.63 tokens/sec
每轮平均时间: 81.0234ms

训练流程完成！模型已保存在 './tmp/sage_model_formal'
```

#### 演示 2：标准训练模式（10M参数）

```bash
# 使用10M参数模型进行完整训练
cargo run --release --bin train -- --sft-sample --model-size 10m --num-epochs 20 --batch-size 4 --training-mode general --force
```

**输出示例**：
```
使用约10M参数的模型配置
使用CPU后端进行训练...
设备信息: Cpu
词表大小: 56
模型参数总量: 19102776 (约 0.001B)
未发现已训练模型，开始正式训练...
使用 8 个工作线程进行数据加载
训练数据: 133 tokens, 验证数据: 15 tokens
批量大小: 4, 序列长度: 256

性能统计:
总训练时间: 19.5592463s
总处理 tokens: 148
处理速度: 7.57 tokens/sec
每轮平均时间: 977.962315ms

训练流程完成！模型已保存在 './tmp/sage_model_formal'
```

---

## 4. 模型部署

### 4.1 训练产物

训练完成后，在输出目录会生成以下文件：

```
artifact_dir/
├── model.mpk              # 最终模型权重
├── best_model.mpk         # 最优轮次模型（如果有验证数据）
├── tokenizer.json         # 分词器配置
├── checkpoint/            # 每轮检查点
│   ├── model-1.mpk
│   ├── model-2.mpk
│   └── ...
├── train/                 # 训练日志
│   ├── epoch-1/
│   │   ├── Loss.log
│   │   └── LearningRate.log
│   └── ...
└── valid/                 # 验证日志（如果有）
    ├── epoch-1/
    │   └── Loss.log
    └── ...
```

### 4.2 模型导出

```bash
# 模型已经自动保存为 model.mpk
# 可以复制到其他位置使用
cp ./tmp/sage_model_formal/model.mpk ./deploy/model.mpk
cp ./tmp/sage_model_formal/tokenizer.json ./deploy/tokenizer.json
```

### 4.3 实际部署演示

#### 演示 1：查看训练产物目录

```bash
# 查看训练生成的文件结构
Get-ChildItem ./tmp/sage_model_formal -Recurse
```

**输出示例**：
```
    目录: D:\Code\Rust\Sage\tmp\sage_model_formal

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         2026/3/22      7:31                checkpoint
-a----         2026/3/22      7:30            428 config.json
-a----         2026/3/22      7:14              0 experiment.log
-a----         2026/3/22      7:31       38213063 model.mpk
-a----         2026/3/22      7:30           1161 tokenizer.json

    目录: D:\Code\Rust\Sage\tmp\sage_model_formal\checkpoint

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2026/3/22      7:31       38213063 model-19.mpk
-a----         2026/3/22      7:31       38213063 model-20.mpk
-a----         2026/3/22      7:31            207 optim-19.mpk
-a----         2026/3/22      7:31            207 optim-20.mpk
-a----         2026/3/22      7:31            207 scheduler-19.mpk
-a----         2026/3/22      7:31            207 scheduler-20.mpk
```

#### 演示 2：创建部署目录

```bash
# 创建部署目录
New-Item -ItemType Directory -Path ./deploy -Force
```

**输出示例**：
```
    目录: D:\Code\Rust\Sage

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----         2026/3/22      7:33                deploy
```

#### 演示 3：复制模型文件到部署目录

```bash
# 复制模型权重文件
Copy-Item ./tmp/sage_model_formal/model.mpk ./deploy/model.mpk

# 复制分词器配置文件
Copy-Item ./tmp/sage_model_formal/tokenizer.json ./deploy/tokenizer.json
```

#### 演示 4：验证部署文件

```bash
# 查看部署目录中的文件
Get-ChildItem ./deploy
```

**输出示例**：
```
    目录: D:\Code\Rust\Sage\deploy

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----         2026/3/22      7:31       38213063 model.mpk
-a----         2026/3/22      7:30           1161 tokenizer.json
```

### 4.4 模型加载

使用 Python 或其他语言加载模型（需要相应的推理库）：

```python
# 示例：使用 burn 库加载模型
from burn import Model, ModelConfig

model = ModelConfig.load("model.mpk")
```

---

## 5. 推理使用

### 5.1 命令行推理

```bash
# 使用训练好的模型进行推理
cargo run --release --bin infer -- --model-dir ./tmp/sage_model_formal
```

### 5.2 推理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 模型目录路径 | 必填 |
| `--prompt` | 提示词（单次推理） | 无 |
| `--interactive` | 交互式对话模式 | false |
| `--max-length` | 最大生成长度 | 200 |
| `--temperature` | 采样温度 | 0.7 |
| `--top-k` | Top-K 采样 | 40 |
| `--top-p` | Top-P 采样 | 0.9 |
| `--seed` | 随机种子 | 42 |

### 5.3 推理示例

#### 演示 1：单次推理

```bash
# 使用训练好的模型进行单次推理
cargo run --release --bin infer -- --model-dir ./tmp/sage_model_formal --prompt "你好，请介绍一下你自己"
```

**输出示例**：
```
正在加载模型...
模型加载完成。

--- 模型生成 ---
提示词: "你好，请介绍一下你自己"
生成结果: "你一你个重tas什型s>。成话<个句？"
```

#### 演示 2：交互式对话

```bash
# 启动交互式对话模式
cargo run --release --bin infer -- --model-dir ./tmp/sage_model_formal --interactive
```

**注意**：生成结果质量取决于训练数据量和模型大小。使用内置样本训练的模型生成效果有限，建议使用更多真实数据进行训练。

### 5.4 效果说明与改进建议

#### 当前模型效果说明

从演示结果可以看到，使用内置样本训练的模型生成效果有限：
- **训练数据不足**：内置样本只有几个示例对话
- **词表太小**：只有56个词汇，限制了表达能力
- **训练轮数有限**：20轮对于10M参数模型来说不够充分

#### 改进建议

要获得更好的生成效果，建议：

1. **准备更多训练数据**
   ```bash
   # 创建包含更多对话样本的训练文件
   echo '{"prompt": "问题1", "response": "回答1"}' > my_data.jsonl
   echo '{"prompt": "问题2", "response": "回答2"}' >> my_data.jsonl
   # ... 添加更多样本
   ```

2. **使用 GPU 加速训练**
   ```bash
   cargo run --release --bin train -- --sft-jsonl my_data.jsonl --backend gpu --model-size 10m --num-epochs 50
   ```

3. **增加训练轮数**
   ```bash
   cargo run --release --bin train -- --sft-jsonl my_data.jsonl --model-size 10m --num-epochs 100
   ```

4. **使用更大的模型**
   ```bash
   cargo run --release --bin train -- --sft-jsonl my_data.jsonl --model-size 30m --num-epochs 50
   ```

### 5.5 API 集成

可以将模型集成到 Web API 服务中：

```rust
// 示例：创建简单的 HTTP API
use sage::{model::Model, tokenizer::Tokenizer};

fn main() {
    // 加载模型和分词器
    let model = Model::load("model.mpk");
    let tokenizer = Tokenizer::load("tokenizer.json");
    
    // 创建 API 服务
    // ...
}
```

---

## 6. 性能优化

### 6.1 GPU 优化

```bash
# 使用高性能 GPU 配置
cargo run --release --bin train -- --backend gpu --fast
```

GPU 优化特性：
- 自动选择高性能 GPU
- 更大的批处理大小（最高 256）
- 优化的学习率（提升 1.5 倍）
- 更多的工作线程（最高 12 个）

#### 性能对比演示

**演示 1：CPU 训练性能**

```bash
# 使用 CPU 进行训练
cargo run --release --bin train -- --sft-sample --model-size 10m --num-epochs 20 --batch-size 4 --training-mode general --force
```

**输出示例**：
```
使用约10M参数的模型配置
使用CPU后端进行训练...
设备信息: Cpu
词表大小: 56
模型参数总量: 19102776 (约 0.001B)
未发现已训练模型，开始正式训练...
使用 8 个工作线程进行数据加载
训练数据: 133 tokens, 验证数据: 15 tokens
批量大小: 4, 序列长度: 256

性能统计:
总训练时间: 31.2289031s
总处理 tokens: 148
处理速度: 4.74 tokens/sec
每轮平均时间: 1.561445155s

训练流程完成！模型已保存在 './tmp/sage_model_formal'
```

**演示 2：GPU 训练性能**

```bash
# 使用 GPU 进行训练
cargo run --release --bin train -- --sft-sample --model-size 10m --num-epochs 20 --batch-size 4 --training-mode general --backend gpu --force
```

**输出示例**：
```
使用约10M参数的模型配置
使用优化的GPU后端配置...
设备信息: Gpu(Adrenalin Edition)
词表大小: 56
模型参数总量: 19102776 (约 0.001B)
未发现已训练模型，开始正式训练...
使用 8 个工作线程进行数据加载
训练数据: 133 tokens, 验证数据: 15 tokens
批量大小: 4, 序列长度: 256

性能统计:
总训练时间: 15.4142858s
总处理 tokens: 148
处理速度: 9.60 tokens/sec
每轮平均时间: 770.71429ms

训练流程完成！模型已保存在 './tmp/sage_model_formal'
```

**性能对比**：
- CPU：31.2秒，4.74 tokens/sec
- GPU：15.4秒，9.60 tokens/sec
- **性能提升**：GPU 训练速度约为 CPU 的 2 倍！

### 6.2 内存优化

```bash
# 减少批处理大小以降低内存使用
cargo run --release --bin train -- --batch-size 16 --max-seq-len 128
```

### 6.3 训练速度优化

```bash
# 使用高性能模式
cargo run --release --bin train -- --fast

# 增加工作线程数
cargo run --release --bin train -- --num-workers 8
```

---

## 7. 故障排除

### 7.1 常见错误

#### 错误：`Failed to install the file logger`
**解决方案**：这是 Windows 权限问题，不影响训练，已通过环境变量自动解决。

#### 错误：`Out of memory`
**解决方案**：
- 减少批处理大小：`--batch-size 16`
- 减少序列长度：`--max-seq-len 128`
- 使用更小的模型：`--model-size default`

#### 错误：`CUDA out of memory`
**解决方案**：
- 切换到 CPU：`--backend cpu`
- 减少批处理大小

### 7.2 训练建议

1. **从小模型开始**：先使用 default 模型验证流程
2. **监控损失曲线**：训练损失应持续下降
3. **定期保存检查点**：使用 `--artifact-dir` 指定输出目录
4. **使用验证集**：添加验证数据以防止过拟合

### 7.3 性能调优

- **批量大小**：根据 GPU 内存调整，GPU 模式推荐 64-256
- **学习率**：初始学习率 1e-4，可逐步调整
- **序列长度**：根据数据特点调整，一般 128-512
- **训练轮数**：根据数据量调整，一般 50-200 轮

---

## 8. 高级用法

### 8.1 分布式训练（未来功能）

```bash
# 未来支持多 GPU 训练
cargo run --release --bin train -- --distributed --gpus 2
```

### 8.2 模型量化

```bash
# 未来支持模型量化以减小模型大小
cargo run --release --bin quantize -- --model-path model.mpk --quantization 4bit
```

### 8.3 模型转换

```bash
# 转换为 ONNX 格式（未来功能）
cargo run --release --bin convert -- --format onnx --input model.mpk --output model.onnx
```

---

## 9. 生产部署建议

### 9.1 模型服务

推荐使用以下架构进行生产部署：

1. **FastAPI + Uvicorn**：创建 RESTful API
2. **Redis**：缓存频繁请求的结果
3. **Docker**：容器化部署
4. **Kubernetes**：集群管理（大规模部署）

### 9.2 监控与日志

- 使用 Prometheus 监控推理延迟
- 使用 Grafana 可视化性能指标
- 使用 ELK Stack 收集和分析日志

---

## 10. 资源链接

- [项目 GitHub](https://github.com/yourusername/sage)
- [Rust Burn 框架](https://burn-rs.github.io/)
- [训练示例](examples/)
- [故障排除指南](TROUBLESHOOTING.md)

---

## 11. 版本更新记录

| 版本 | 主要特性 |
|------|----------|
| 0.1.0 | 初始版本，基础训练和推理功能 |
| 0.1.1 | 添加 GPU 优化和更大模型配置 |
| 0.1.2 | 添加特定领域训练模式 |

---

## 联系方式

- 问题反馈：[GitHub Issues](https://github.com/yourusername/sage/issues)
- 技术支持：your@email.com

---

**注意**：本文档会随着项目发展不断更新，请定期查看最新版本。
