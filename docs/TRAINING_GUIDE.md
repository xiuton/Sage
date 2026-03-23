# Sage 训练指南

## 概述

本文档详细介绍 Sage 模型的各种训练方式、语料获取方法以及训练后的调整优化策略。

## 目录

1. [训练方式](#训练方式)
2. [语料获取](#语料获取)
3. [实际训练流程](#实际训练流程)
4. [训练参数调整](#训练参数调整)
5. [模型评估与优化](#模型评估与优化)
6. [常见问题排查](#常见问题排查)

## 训练方式

### 1. 测试训练（快速验证）

用于快速验证代码和环境是否正常工作：

```bash
# 超快速开发模式（约1秒完成）
cargo run --release --bin train -- --ultra-quick --sft-sample --backend gpu --force-tui --force

# 快速开发模式（约5秒完成）
cargo run --release --bin train -- --quick-dev --sft-sample --backend gpu --force-tui --force
```

**参数说明：**
- `--ultra-quick`：超快速模式，1轮训练，极小批量(2)，极高学习率
- `--quick-dev`：快速开发模式，1轮训练，超小批量(4)，超高学习率
- `--sft-sample`：使用内置示例数据

### 2. 正式训练

使用自定义语料进行完整训练：

```bash
# 使用BPE分词器的正式训练
cargo run --release --bin train -- \
    --sft-jsonl data/your_corpus.jsonl \
    --artifact-dir ./tmp/your_model \
    --use-bpe \
    --bpe-vocab-size 20000 \
    --num-epochs 50 \
    --batch-size 32 \
    --max-seq-len 256 \
    --lr 5e-5 \
    --num-workers 8 \
    --backend gpu \
    --model-size 30m \
    --force
```

**关键参数：**
- `--sft-jsonl`：训练语料文件路径
- `--artifact-dir`：模型保存目录
- `--use-bpe`：启用BPE分词器
- `--bpe-vocab-size`：BPE词表大小（建议10000-50000）
- `--model-size`：模型大小（10m/30m）

### 3. 增量训练

在现有模型基础上继续训练：

```bash
# 继续训练
cargo run --release --bin train -- \
    --sft-jsonl data/your_corpus.jsonl \
    --artifact-dir ./tmp/your_model \
    --continue \
    --num-epochs 100 \
    --backend gpu
```

**参数说明：**
- `--continue`：从现有模型继续训练
- `--resume-epoch <epoch>`：从特定轮次恢复训练

## 语料获取

### 1. 使用内置生成工具

#### 生成对话语料
```bash
# 生成1000条多样化对话语料
cargo run --release --bin gen_sft -- --out data/dialogs.jsonl --count 1000 --seed 123
```

#### 生成技术问答语料
```bash
# 生成500条技术问答语料（包含网络数据）
cargo run --release --bin gen_web_sft -- --out data/tech_qa.jsonl --count 500 --web --seed 42
```

**参数说明：**
- `--out`：输出文件路径
- `--count`：生成数据数量
- `--web`：启用网络数据获取
- `--seed`：随机种子

### 2. 公开数据集

推荐的中文语料来源：

- **中文维基百科**：https://dumps.wikimedia.org/zhwiki/
- **中文新闻语料**：THUCNews、Sogou新闻等
- **GitHub代码库**：各种开源项目的代码和文档
- **Stack Overflow**：技术问答数据
- **Hugging Face数据集**：https://huggingface.co/datasets

### 3. 网络爬虫采集

可以使用Python爬虫从以下网站获取数据：

- **技术博客**：CSDN、知乎、掘金、SegmentFault
- **文档网站**：官方文档、技术文档站点
- **论坛社区**：技术论坛、问答社区

### 4. 数据格式要求

训练数据需要使用JSONL格式，每行一个JSON对象：

```json
# 格式1：prompt-response格式
{"prompt": "问题内容", "response": "回答内容"}

# 格式2：messages格式（推荐）
{"messages": [
    {"role": "user", "content": "问题内容"},
    {"role": "assistant", "content": "回答内容"}
]}
```

## 实际训练流程

### 1. 数据准备阶段

```bash
# 步骤1：生成或准备训练数据
cargo run --release --bin gen_web_sft -- --out data/train_corpus.jsonl --count 10000 --web --seed 123

# 步骤2：验证数据格式（Linux/macOS）
head -n 3 data/train_corpus.jsonl

# 步骤2：验证数据格式（Windows PowerShell）
Get-Content data/train_corpus.jsonl | Select-Object -First 3
# 或者更简单的方式
Get-Content data/train_corpus.jsonl -Head 3
```

**验证输出示例：**
```json
{"domain":"人工智能","id":0,"messages":[{"content":"关于人工智能，什么是机器学习？","role":"user"},{"content":"机器学习是人工智能的一个分支...","role":"assistant"}]}
{"domain":"深度学习","id":1,"messages":[{"content":"什么是深度学习？","role":"user"},{"content":"深度学习是机器学习的一个分支...","role":"assistant"}]}
{"domain":"神经网络","id":2,"messages":[{"content":"什么是神经网络？","role":"user"},{"content":"神经网络是一种计算模型...","role":"assistant"}]}
```

**验证要点：**
- 每行必须是有效的JSON格式
- 必须包含`messages`字段或`prompt`/`response`字段
- 确保没有语法错误或格式问题

### 2. 模型训练阶段

```bash
# 步骤1：创建BPE分词器并开始训练
cargo run --release --bin train -- \
    --sft-jsonl data/train_corpus.jsonl \
    --artifact-dir ./tmp/large_model \
    --use-bpe \
    --bpe-vocab-size 30000 \
    --num-epochs 50 \
    --batch-size 32 \
    --max-seq-len 256 \
    --lr 5e-5 \
    --num-workers 8 \
    --backend gpu \
    --model-size 30m \
    --force \
    --force-tui

# 步骤2：监控训练进度
# 观察TUI界面的损失曲线和学习率变化
```

### 3. 模型评估阶段

```bash
# 使用训练好的模型进行推理测试
cargo run --release --bin infer -- \
    --model-dir ./tmp/large_model \
    --use-best \
    --prompt "什么是深度学习？" \
    --num-tokens 100 \
    --temperature 0.7
```

## 训练参数调整

### 1. 学习率调整

**推荐设置：**
- **初始学习率**：5e-5 ~ 1e-4（GPU）
- **学习率预热**：前10%轮次线性预热
- **学习率衰减**：使用余弦衰减或线性衰减

**调整策略：**
- **训练不稳定**：降低学习率（如从1e-4降至5e-5）
- **收敛缓慢**：适当提高学习率
- **损失波动大**：降低学习率并增加批量大小

```bash
# 调整学习率示例
--lr 3e-5
```

**学习率监控：**
- 通过TUI界面观察学习率变化
- 训练开始时学习率会逐渐上升（预热阶段）
- 后期学习率会逐渐下降（衰减阶段）

### 2. 批量大小调整

**根据GPU内存调整：**
- **8GB GPU**：batch-size = 8 ~ 16
- **16GB GPU**：batch-size = 16 ~ 32
- **24GB+ GPU**：batch-size = 32 ~ 64
- **CPU训练**：batch-size = 4 ~ 16（根据CPU核心数调整）

**批量大小影响：**
- **较大批量**：训练更稳定，梯度估计更准确，但占用更多内存
- **较小批量**：训练更快（每次迭代时间短），但梯度噪声大
- **批量大小选择**：在GPU内存允许的情况下，尽量使用较大的批量

**自动优化：**
- 使用GPU后端时，系统会自动优化批量大小
- 例如：`--batch-size 8` 在GPU模式下可能自动调整为32

```bash
# 调整批量大小
--batch-size 16

# GPU模式下的批量优化示例
--batch-size 8 --backend gpu  # 可能自动调整为32
```

### 3. 序列长度调整

**根据语料长度调整：**
- **短文本**（问答对）：max-seq-len = 128
- **中等文本**（段落对话）：max-seq-len = 256
- **长文本**（完整对话）：max-seq-len = 512
- **超长文本**：max-seq-len = 1024（需要足够的GPU内存）

**序列长度影响：**
- **较长序列**：可以处理更长的上下文，但占用更多内存
- **较短序列**：训练更快，但可能丢失上下文信息
- **序列长度选择**：根据实际语料的平均长度选择合适的值

**内存消耗：**
- 序列长度每增加一倍，内存消耗大约增加一倍
- 例如：max-seq-len=512 比 max-seq-len=256 占用约两倍内存

```bash
# 调整序列长度
--max-seq-len 256

# 根据语料长度选择
--max-seq-len 128  # 短问答
--max-seq-len 512  # 长对话
```

### 4. 训练轮次调整

**推荐设置：**
- **小数据集**（<1000条）：30 ~ 50轮
- **中等数据集**（1000-10000条）：50 ~ 100轮
- **大数据集**（>10000条）：100 ~ 200轮
- **超大数据集**（>100000条）：200 ~ 500轮

**训练轮次判断：**
- **观察损失曲线**：当训练损失和验证损失趋于稳定时可以停止
- **早期停止**：如果验证损失连续多轮不再下降，可以提前停止
- **过拟合判断**：如果训练损失继续下降但验证损失开始上升，说明过拟合

**增量训练：**
- 可以使用`--continue`参数继续训练
- 每次可以增加50-100轮进行增量训练

```bash
# 调整训练轮次
--num-epochs 100

# 增量训练示例
--continue --num-epochs 200  # 在原有基础上继续训练100轮
```

**训练时间估计：**
- 训练时间 = 每轮时间 × 轮次数
- 可以通过前几轮的时间估算总训练时间
- 例如：每轮5分钟，100轮大约需要8小时

## 模型评估与优化

### 1. 评估指标

**主要评估指标：**
- **损失值（Loss）**：训练和验证损失，反映模型拟合程度
- **困惑度（Perplexity）**：exp(loss)，越低越好，衡量模型预测的不确定性
- **生成质量**：人工评估生成内容的质量、相关性和连贯性
- **BLEU分数**：衡量生成文本与参考文本的相似度（可选）
- **ROUGE分数**：评估摘要质量（可选）

**指标解读：**
- **Loss下降**：模型正在学习和拟合数据
- **Perplexity降低**：模型预测能力提高
- **训练Loss远低于验证Loss**：可能存在过拟合
- **Loss不再下降**：模型可能已收敛

### 2. 评估方法

**自动化评估：**
```bash
# 使用验证集评估
cargo run --release --bin train -- \
    --sft-jsonl data/train_corpus.jsonl \
    --artifact-dir ./tmp/your_model \
    --continue \
    --num-epochs 1 \
    --backend gpu
```

**人工评估：**
```bash
# 使用训练好的模型进行交互式评估
cargo run --release --bin infer -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --chat \
    --interactive
```

**批量评估脚本：**
可以编写脚本批量测试模型在测试集上的表现，记录生成结果和评分。

### 3. 优化策略

**常见优化方法：**

#### 1. 数据优化
- **数据增强**：
  - 添加更多样化的训练数据
  - 对现有数据进行改写和扩充
  - 使用数据扩充技术（同义词替换、句式变换等）
  
- **数据清洗**：
  - 去除低质量、重复的训练样本
  - 过滤噪声和无关内容
  - 确保数据质量和一致性

#### 2. 模型调优
- **模型结构优化**：
  - 尝试不同的模型大小（10m vs 30m）
  - 调整层数、注意力头数等超参数
  - 尝试不同的激活函数和归一化方法
  
- **超参数优化**：
  - 使用网格搜索或随机搜索优化超参数
  - 关注学习率、批量大小、dropout率等关键参数

#### 3. 正则化技术
- **防止过拟合**：
  - 使用dropout层（当前已设置为0.1）
  - 添加权重衰减（weight decay）
  - 使用早停策略（early stopping）
  
- **数据增强正则化**：
  - 使用数据扩充减少过拟合
  - 添加噪声和扰动增加模型鲁棒性

#### 4. 学习率优化
- **学习率调度**：
  - 使用学习率预热（warmup）
  - 实现学习率衰减（cosine/linear decay）
  - 自适应学习率调整

#### 5. 训练技巧
- **混合精度训练**：使用FP16加速训练（未来支持）
- **梯度累积**：允许使用更大的等效批量大小
- **梯度裁剪**：防止梯度爆炸

#### 6. 评估和迭代
- **定期评估**：在验证集上评估模型性能
- **模型选择**：保存最佳模型（使用`--use-best`）
- **持续迭代**：基于评估结果调整训练策略

### 4. 过拟合处理

**过拟合迹象：**
- **训练损失持续下降，但验证损失开始上升**
- **生成内容重复或模式化**
- **模型在训练集上表现很好，但在新数据上表现差**
- **困惑度在训练集上很低，但在验证集上很高**

**解决方案：**

#### 1. 数据层面
- **增加数据量**：收集更多训练数据
- **数据增强**：对现有数据进行扩充和变换
- **数据清洗**：去除异常和低质量样本

#### 2. 模型层面
- **降低模型复杂度**：
  - 使用更小的模型（10m vs 30m）
  - 减少层数或隐藏层大小
  - 使用更简单的模型结构

- **增强正则化**：
  - 增加dropout率（当前为0.1，可以尝试0.2-0.5）
  - 添加权重衰减（L2正则化）
  - 使用早停策略（early stopping）

#### 3. 训练策略
- **提前停止**：当验证损失不再下降时停止训练
- **减少训练轮次**：避免训练过度
- **调整学习率**：使用更小的学习率

#### 4. 评估和监控
- **定期评估**：在验证集上评估模型性能
- **监控指标**：同时关注训练和验证损失
- **保存最佳模型**：使用验证集性能选择最佳模型

**实用建议：**
- 先尝试增加数据量和数据增强
- 如果仍然过拟合，再考虑降低模型复杂度
- 使用早停策略自动选择最佳训练轮次

## 常见问题排查

### 1. UTF-8字符边界错误

**错误信息：**
```
byte index XX is not a char boundary; it is inside 'X' (bytes XX..XX)
```

**问题原因：**
- BPE分词器返回的偏移量是字节索引
- 中文字符占用多个字节，切片时可能跨越字符边界

**解决方案：**
- 已在代码中修复，使用安全的字符串匹配方法
- 如果仍然遇到此错误，请更新到最新版本

### 2. GPU内存不足

**错误信息：**
```
Out of memory
```

**问题原因：**
- 批量大小过大
- 序列长度过长
- 模型过大

**解决方案：**
- **减小批量大小**：`--batch-size 8`
- **减小序列长度**：`--max-seq-len 128`
- **使用更小的模型**：`--model-size 10m`
- **调整学习率**：可能需要同时调整学习率

### 3. 训练速度慢

**问题表现：**
- 每轮训练时间过长
- GPU利用率低

**优化建议：**
- **使用GPU后端**：`--backend gpu`
- **增加工作线程数**：`--num-workers 8`（根据CPU核心数调整）
- **使用BPE分词器**：`--use-bpe`（比字符分词更快）
- **调整批处理大小**：找到GPU内存允许的最大批量
- **检查GPU驱动**：确保使用最新的GPU驱动

### 4. TUI不显示

**问题表现：**
- 训练时没有显示进度界面
- 只显示文本输出

**解决方案：**
- **强制启用TUI**：`--force-tui`
- **确保终端支持**：使用支持ANSI颜色的终端
- **检查环境变量**：代码会自动设置必要的环境变量
- **Windows终端**：推荐使用Windows Terminal或PowerShell

### 5. 模型不收敛

**问题表现：**
- 损失值不下降或波动很大
- 生成内容质量差

**解决方案：**
- **调整学习率**：尝试不同的学习率（如5e-5, 1e-4）
- **检查数据质量**：确保训练数据格式正确
- **增加训练轮次**：可能需要更多轮次才能收敛
- **检查模型配置**：确认模型参数设置正确

### 6. JSON解析错误

**错误信息：**
```
Failed to parse line: ...
```

**问题原因：**
- JSON格式错误
- 缺少必要字段
- 编码问题

**解决方案：**
- **验证数据格式**：使用`Get-Content data/file.jsonl -Head 3`检查
- **修复JSON格式**：确保每行都是有效的JSON
- **检查编码**：确保文件使用UTF-8编码

### 7. 分词器构建失败

**错误信息：**
```
Failed to build BPE tokenizer
```

**问题原因：**
- 语料数据太少
- 语料格式不正确

**解决方案：**
- **增加语料量**：至少需要几百条数据
- **检查数据格式**：确保数据包含文本内容
- **使用预训练分词器**：如果无法构建新分词器

### 8. 模型加载失败

**错误信息：**
```
Failed to load model
```

**问题原因：**
- 模型文件损坏
- 模型配置不匹配

**解决方案：**
- **重新训练**：使用`--force`参数重新训练
- **检查模型文件**：确保模型文件完整
- **验证配置**：确认模型配置与代码兼容

## 生产环境部署

### 1. 模型导出

**模型文件结构：**
训练完成后，模型会自动保存到指定目录，包含以下文件：

```
./tmp/your_model/
├── model.mpk          # 模型权重文件
├── best_model.mpk     # 最佳模型权重（如果启用）
├── config.json        # 模型配置文件
├── tokenizer.json     # 分词器配置文件
├── tokenizer.vocab    # 词表文件（BPE模式）
└── checkpoint/        # 训练检查点目录
    ├── model-10.mpk   # 第10轮检查点
    ├── model-20.mpk   # 第20轮检查点
    └── ...
```

**模型验证：**
```bash
# 检查模型文件是否完整
ls -la ./tmp/your_model/

# 验证模型配置
cat ./tmp/your_model/config.json
```

### 2. 模型服务部署

#### 交互式部署
```bash
# 使用infer工具提供交互式推理服务
cargo run --release --bin infer -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --chat \
    --interactive
```

#### API服务部署

**启动API服务器：**
```bash
# 启动API服务器（默认端口8000，CPU后端）
cargo run --release --bin api_server -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --port 8000

# 使用GPU后端加速推理
cargo run --release --bin api_server -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --backend gpu \
    --port 8000

# 使用INT8量化优化
cargo run --release --bin api_server -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --quantize \
    --port 8000

# 同时使用GPU和量化
cargo run --release --bin api_server -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --backend gpu \
    --quantize \
    --port 8000
```

**API服务器参数说明：**
- `--model-dir`：模型目录路径
- `--use-best`：使用最佳模型权重
- `--port`：服务器端口（默认8000）
- `--backend`：推理后端（`cpu` 或 `gpu`，默认 `cpu`）
- `--quantize`：启用INT8量化优化
- `--context-len`：上下文长度（默认跟随模型配置）

#### Docker容器部署

**构建Docker镜像：**
```bash
# 构建Docker镜像
docker build -t sage-api .

# 运行Docker容器
docker run -d \
  -p 8000:8000 \
  -v ./tmp/your_model:/app/models \
  --name sage-api \
  sage-api
```

**使用docker-compose部署：**
```bash
# 启动服务
docker-compose up -d sage-api

# 查看日志
docker-compose logs sage-api

# 停止服务
docker-compose down
```

**GPU支持（可选）：**
```bash
# 需要NVIDIA Docker支持
docker-compose up -d sage-api-gpu
```

**Docker部署说明：**
- 模型文件需要挂载到 `/app/models` 目录
- 默认端口为8000
- 支持环境变量 `RUST_LOG` 控制日志级别
- 提供自动重启机制

**API接口说明：**

| 端点 | 方法 | 描述 | 认证 |
|------|------|------|------|
| `/v1/chat/completions` | POST | Chat Completion接口（OpenAI标准） | 需要 |
| `/v1/batch-chat/completions` | POST | 批量Chat Completion接口 | 需要 |
| `/v1/async-chat/completions` | POST | 异步Chat Completion接口 | 需要 |
| `/api/task/:task_id` | GET | 查询任务状态 | 需要 |
| `/api/health` | GET | 健康检查接口 | 不需要 |
| `/api/model-info` | GET | 获取模型信息 | 需要 |

### 认证机制

API使用Bearer Token认证，通过环境变量 `SAGE_API_KEY` 配置。如果未配置API密钥，则所有接口都不需要认证。

**使用示例：**
```bash
# 设置API密钥环境变量
export SAGE_API_KEY="your-secret-key"

# 使用API密钥调用接口
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "sage-model",
    "messages": [
      {"role": "user", "content": "什么是深度学习？"}
    ]
  }'
```

### Chat Completion接口（OpenAI标准）

**请求示例（普通模式）：**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "sage-model",
    "messages": [
      {"role": "system", "content": "你是一个助手"},
      {"role": "user", "content": "什么是深度学习？"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "top_p": 0.9,
    "top_k": 10
  }'
```

**请求示例（流式输出）：**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "sage-model",
    "messages": [
      {"role": "system", "content": "你是一个助手"},
      {"role": "user", "content": "什么是深度学习？"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "top_p": 0.9,
    "top_k": 10,
    "stream": true
  }'
```

**流式输出响应说明：**
流式输出使用Server-Sent Events (SSE)格式，每个token生成后立即返回，格式如下：
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"sage-model","choices":[{"index":0,"message":{"role":"assistant","content":"深"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"sage-model","choices":[{"index":0,"message":{"role":"assistant","content":"深度学习"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"sage-model","choices":[{"index":0,"message":{"role":"assistant","content":"深度学习是"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"sage-model","choices":[{"index":0,"message":{"role":"assistant","content":"深度学习是机器学习的一个分支"},"finish_reason":"stop"}]}
```

**响应示例：**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "sage-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 80,
    "total_tokens": 100
  }
}
```

### 批量Chat Completion接口

**请求示例：**
```bash
curl -X POST http://localhost:8000/v1/batch-chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "requests": [
      {
        "model": "sage-model",
        "messages": [{"role": "user", "content": "什么是人工智能？"}],
        "max_tokens": 50
      },
      {
        "model": "sage-model",
        "messages": [{"role": "user", "content": "什么是机器学习？"}],
        "max_tokens": 50
      }
    ]
  }'
```

**响应示例：**
```json
{
  "responses": [
    {
      "id": "chatcmpl-124",
      "object": "chat.completion",
      "created": 1677858243,
      "model": "sage-model",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "人工智能是计算机科学的一个分支..."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 35,
        "total_tokens": 50
      }
    },
    {
      "id": "chatcmpl-125",
      "object": "chat.completion",
      "created": 1677858244,
      "model": "sage-model",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "机器学习是人工智能的一个分支..."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 16,
        "completion_tokens": 34,
        "total_tokens": 50
      }
    }
  ],
  "total_duration_ms": 920,
  "request_count": 2
}
```

### 异步Chat Completion接口

**请求示例：**
```bash
curl -X POST http://localhost:8000/v1/async-chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "sage-model",
    "messages": [{"role": "user", "content": "什么是深度学习？"}],
    "max_tokens": 200
  }'
```

**响应示例：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "Pending",
  "result": null,
  "error": null
}
```

### 任务状态查询接口

**请求示例：**
```bash
curl -X GET http://localhost:8000/api/task/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer your-secret-key"
```

**响应示例：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "Completed",
  "result": {
    "id": "chatcmpl-126",
    "object": "chat.completion",
    "created": 1677858245,
    "model": "sage-model",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "深度学习是机器学习的一个分支..."
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 20,
      "completion_tokens": 180,
      "total_tokens": 200
    }
  },
  "error": null,
  "created_at": 1000,
  "started_at": 1500,
  "completed_at": 2700
}
```

### 健康检查

```bash
curl http://localhost:8000/api/health
```

### 获取模型信息

```bash
curl -X GET http://localhost:8000/api/model-info \
  -H "Authorization: Bearer your-secret-key"
```

**API参数说明（OpenAI标准）：**
- `model`：模型名称（可选，默认"sage-model"）
- `messages`：消息数组（必填），包含role和content字段
  - `role`：消息角色（system、user、assistant）
  - `content`：消息内容
- `max_tokens`：生成的最大token数（可选，默认50）
- `temperature`：采样温度（可选，默认0.8）
- `top_p`：top-p采样参数（可选，默认0.9）
- `top_k`：top-k采样参数（可选，默认10）
- `n`：生成的回复数量（可选，默认1）
- `stop`：停止序列（可选）
- `presence_penalty`：存在惩罚（可选）
- `frequency_penalty`：频率惩罚（可选）
- `seed`：随机种子（可选）

**批量推理限制：**
- 最大批量大小：100个请求
- 批量请求不能为空

**异步任务状态：**
- `Pending`：任务等待处理
- `Running`：任务正在处理中
- `Completed`：任务处理完成
- `Failed`：任务处理失败

#### 批量推理
```bash
# 批量处理文本文件
cargo run --release --bin infer -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --prompt-file input.txt \
    --output-file output.txt
```

### 3. 性能优化

**模型优化：**
- **量化**：使用INT8/INT4量化减小模型体积（未来支持）
- **剪枝**：移除不重要的权重（未来支持）
- **蒸馏**：知识蒸馏减小模型大小（未来支持）

**推理优化：**
- **批处理**：实现批处理推理提高吞吐量
- **缓存**：缓存频繁使用的计算结果
- **并行处理**：使用多线程或异步处理

**部署优化：**
- **使用ONNX格式**：导出为ONNX格式进行部署（未来支持）
- **GPU加速**：在生产环境使用GPU加速推理
- **内存管理**：优化内存使用减少资源消耗

### 4. 监控和维护

**性能监控：**
- 监控推理延迟和吞吐量
- 跟踪资源使用情况（CPU/GPU/内存）
- 设置告警机制

**模型更新：**
- 定期重新训练模型
- 使用增量学习更新模型
- A/B测试新模型

**故障恢复：**
- 定期备份模型文件
- 实现模型回滚机制
- 监控系统健康状态

## 总结

通过本文档的指导，您将能够：

### 🎯 核心能力
1. **掌握多种训练方式**：从快速验证到正式训练，灵活应对不同场景
2. **获取高质量语料**：使用内置工具或外部数据源准备训练数据
3. **进行有效训练**：合理设置参数，监控训练进度，优化训练效果
4. **解决常见问题**：快速排查和解决训练过程中的各种问题
5. **部署和维护模型**：将训练好的模型部署到生产环境

### 🚀 实践建议
- **循序渐进**：从简单的测试训练开始，逐步过渡到正式训练
- **数据为王**：高质量的训练数据是模型性能的关键
- **参数调优**：耐心调整超参数，找到最佳配置
- **持续监控**：关注训练过程，及时发现和解决问题
- **定期评估**：使用验证集和人工评估确保模型质量

### 💡 未来展望
- **更大模型**：支持更大规模的模型训练
- **多模态支持**：扩展到图像、音频等多模态数据
- **分布式训练**：支持多GPU并行训练
- **量化和优化**：提供模型压缩和推理加速
- **更多训练模式**：支持指令微调、强化学习等高级训练方法

### 📚 资源推荐
- **Burn框架文档**：https://burn-rs.github.io/
- **Rust深度学习社区**：https://github.com/burn-rs/burn
- **中文NLP资源**：各种开源中文语料库和工具

---

**更新日期：** 2026-03-24  
**版本：** v1.1  
**作者：** Sage团队