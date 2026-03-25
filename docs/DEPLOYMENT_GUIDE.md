# Sage 部署指南

本文档提供 Sage 模型从训练完成到生产环境部署的完整指南。

## 目录

1. [模型部署概述](#模型部署概述)
2. [模型导出与验证](#模型导出与验证)
3. [交互式部署](#交互式部署)
4. [API服务部署](#api服务部署)
5. [Docker容器部署](#docker容器部署)
6. [性能优化](#性能优化)
7. [监控和维护](#监控和维护)
8. [认证机制](#认证机制)
9. [API接口说明](#api接口说明)

---

## 1. 模型部署概述

### 1.1 部署前准备

在开始部署之前，请确保：
- 训练已完成并生成了模型文件
- 模型文件完整且可用
- 部署环境满足系统要求

### 1.2 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **内存**: 最低 8GB，推荐 16GB+
- **GPU**: 可选，支持 WGPU（Windows DirectX/Metal/Vulkan）
- **Rust**: 版本 1.75+（推荐使用 rustup 安装）

### 1.3 环境准备

```bash
# 安装 Rust（如果尚未安装）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 克隆项目
git clone https://github.com/yourusername/sage.git
cd sage

# 安装依赖
cargo build --release
```

---

## 2. 模型导出与验证

### 2.1 模型文件结构

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

### 2.2 模型验证

```bash
# 检查模型文件是否完整
ls -la ./tmp/your_model/

# 验证模型配置
cat ./tmp/your_model/config.json
```

### 2.3 创建部署目录

```bash
# 创建部署目录
mkdir -p ./deploy

# 复制模型文件
cp ./tmp/your_model/model.mpk ./deploy/model.mpk
cp ./tmp/your_model/best_model.mpk ./deploy/best_model.mpk
cp ./tmp/your_model/config.json ./deploy/config.json
cp ./tmp/your_model/tokenizer.json ./deploy/tokenizer.json
```

### 2.4 模型验证测试

```bash
# 使用训练好的模型进行推理测试
cargo run --release --bin infer -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --prompt "什么是深度学习？" \
    --num-tokens 100 \
    --temperature 0.7
```

---

## 3. 交互式部署

### 3.1 命令行交互式部署

```bash
# 使用infer工具提供交互式推理服务
cargo run --release --bin infer -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --chat \
    --interactive
```

### 3.2 流式输出部署

```bash
# 启用流式输出的交互式部署
cargo run --release --bin infer -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --chat \
    --interactive \
    --stream \
    --stream-speed 10
```

### 3.3 GPU加速交互式部署

```bash
# 使用GPU加速的交互式部署
cargo run --release --bin infer -- \
    --model-dir ./tmp/your_model \
    --use-best \
    --chat \
    --interactive \
    --backend gpu
```

---

## 4. API服务部署

### 4.1 启动API服务器

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

### 4.2 API服务器参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 模型目录路径 | 必填 |
| `--use-best` | 使用最佳模型权重 | false |
| `--port` | 服务器端口 | 8000 |
| `--backend` | 推理后端（`cpu` 或 `gpu`） | cpu |
| `--quantize` | 启用INT8量化优化 | false |
| `--context-len` | 上下文长度 | 默认跟随模型配置 |
| `--host` | 服务器地址 | 0.0.0.0 |
| `--log-level` | 日志级别 | info |

### 4.3 健康检查

```bash
# 检查API服务器是否正常运行
curl http://localhost:8000/api/health
```

### 4.4 获取模型信息

```bash
curl -X GET http://localhost:8000/api/model-info \
  -H "Authorization: Bearer your-secret-key"
```

---

## 5. Docker容器部署

### 5.1 构建Docker镜像

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

### 5.2 使用docker-compose部署

```bash
# 启动服务
docker-compose up -d sage-api

# 查看日志
docker-compose logs sage-api

# 停止服务
docker-compose down
```

### 5.3 GPU支持（可选）

```bash
# 需要NVIDIA Docker支持
docker-compose up -d sage-api-gpu
```

### 5.4 Docker部署说明

- 模型文件需要挂载到 `/app/models` 目录
- 默认端口为8000
- 支持环境变量 `RUST_LOG` 控制日志级别
- 提供自动重启机制

---

## 6. 性能优化

### 6.1 模型优化

- **量化**：支持INT8动态量化和静态量化，减小模型体积并提高推理速度
- **剪枝**：移除不重要的权重（未来支持）
- **蒸馏**：知识蒸馏减小模型大小（未来支持）

### 6.2 推理优化

- **批处理**：实现批处理推理提高吞吐量
- **缓存**：缓存频繁使用的计算结果
- **并行处理**：使用多线程或异步处理

### 6.3 部署优化

- **使用ONNX格式**：导出为ONNX格式进行部署（未来支持）
- **GPU加速**：在生产环境使用GPU加速推理
- **内存管理**：优化内存使用减少资源消耗

### 6.4 量化评估

**性能测试：**
```bash
# 运行量化性能基准测试
cargo run --release --bin benchmark
```

**精度评估：**
```bash
# 运行量化精度评估
cargo run --release --bin accuracy_eval
```

**评估指标：**
- **性能提升**：量化模型相对于原始模型的加速比
- **精度损失**：量化模型与原始模型的输出相似度
- **内存节省**：量化模型的内存占用减少比例

**测试结果示例：**
```
=== 量化性能测试 ===
配置文件: config.toml
模型文件: sage_model.burn

=== CPU性能测试 ===
测试提示: 今天天气很好，我们去
------------------------
原始模型 (10次推理): 10.5s
动态量化模型 (10次推理): 6.2s
静态量化模型 (10次推理): 5.8s
动态量化加速比: 1.7x
静态量化加速比: 1.8x

=== 量化精度评估 ===
完全匹配率: 95.00% (19/20)
字符匹配率: 98.75%
```

---

## 7. 监控和维护

### 7.1 性能监控

- 监控推理延迟和吞吐量
- 跟踪资源使用情况（CPU/GPU/内存）
- 设置告警机制

### 7.2 模型更新

- 定期重新训练模型
- 使用增量学习更新模型
- A/B测试新模型

### 7.3 故障恢复

- 定期备份模型文件
- 实现模型回滚机制
- 监控系统健康状态

### 7.4 Systemd服务管理

#### 安装服务
```bash
# 复制服务文件到systemd目录
sudo cp deploy/sage-api.service /etc/systemd/system/

# 修改服务文件中的用户和路径
sudo nano /etc/systemd/system/sage-api.service

# 重新加载systemd配置
sudo systemctl daemon-reload
```

#### 管理服务
```bash
# 启动服务
sudo systemctl start sage-api

# 设置开机自启
sudo systemctl enable sage-api

# 查看服务状态
sudo systemctl status sage-api

# 停止服务
sudo systemctl stop sage-api

# 重启服务
sudo systemctl restart sage-api
```

#### 查看日志
```bash
# 查看实时日志
sudo journalctl -u sage-api -f

# 查看最近日志
sudo journalctl -u sage-api --since "1 hour ago"
```

---

## 8. 认证机制

API使用Bearer Token认证，通过环境变量 `SAGE_API_KEY` 配置。如果未配置API密钥，则所有接口都不需要认证。

### 8.1 设置API密钥

```bash
# Windows PowerShell
$env:SAGE_API_KEY="your-secret-key"

# Linux/macOS
export SAGE_API_KEY="your-secret-key"
```

### 8.2 使用API密钥

```bash
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

---

## 9. API接口说明

### 9.1 接口列表

| 端点 | 方法 | 描述 | 认证 |
|------|------|------|------|
| `/v1/chat/completions` | POST | Chat Completion接口（OpenAI标准） | 需要 |
| `/v1/batch-chat/completions` | POST | 批量Chat Completion接口 | 需要 |
| `/v1/async-chat/completions` | POST | 异步Chat Completion接口 | 需要 |
| `/api/task/:task_id` | GET | 查询任务状态 | 需要 |
| `/api/health` | GET | 健康检查接口 | 不需要 |
| `/api/model-info` | GET | 获取模型信息 | 需要 |

### 9.2 Chat Completion接口（OpenAI标准）

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

### 9.3 批量Chat Completion接口

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

### 9.4 异步Chat Completion接口

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

### 9.5 任务状态查询接口

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

### 9.6 API参数说明（OpenAI标准）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model` | 模型名称 | sage-model |
| `messages` | 消息数组 | 必填 |
| `max_tokens` | 生成的最大token数 | 50 |
| `temperature` | 采样温度 | 0.8 |
| `top_p` | top-p采样参数 | 0.9 |
| `top_k` | top-k采样参数 | 10 |
| `n` | 生成的回复数量 | 1 |
| `stop` | 停止序列 | 无 |
| `presence_penalty` | 存在惩罚 | 无 |
| `frequency_penalty` | 频率惩罚 | 无 |
| `seed` | 随机种子 | 无 |

### 9.7 批量推理限制

- 最大批量大小：100个请求
- 批量请求不能为空

### 9.8 异步任务状态

- `Pending`：任务等待处理
- `Running`：任务正在处理中
- `Completed`：任务处理完成
- `Failed`：任务处理失败

---

## 10. 增强功能使用

### 10.1 模型管理接口使用

#### 1. 列出所有模型
```bash
curl -X GET http://localhost:8000/api/models \
  -H "Authorization: Bearer your-api-key"
```

#### 2. 加载新模型
```bash
curl -X POST http://localhost:8000/api/models \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my-custom-model",
    "model_dir": "./models/custom-model"
  }'
```

#### 3. 切换活动模型
```bash
curl -X POST http://localhost:8000/api/models/my-custom-model/activate \
  -H "Authorization: Bearer your-api-key"
```

#### 4. 热更新模型
```bash
curl -X POST http://localhost:8000/api/models/my-custom-model/reload \
  -H "Authorization: Bearer your-api-key"
```

#### 5. 卸载模型
```bash
curl -X DELETE http://localhost:8000/api/models/my-custom-model \
  -H "Authorization: Bearer your-api-key"
```

#### 6. 下载模型
```bash
curl -X POST http://localhost:8000/api/models/download \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "downloaded-model",
    "url": "https://example.com/models/model.mpk"
  }'
```

#### 7. 更新模型
```bash
curl -X POST http://localhost:8000/api/models/my-model/update \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/models/model_v2.mpk"
  }'
```

### 10.2 模型导出使用

#### 1. 导出为ONNX格式
```bash
cargo run --bin export -- \
  --model-dir ./models/my-model \
  --output ./exports/model.onnx \
  --format onnx
```

#### 2. 导出为GGUF格式
```bash
cargo run --bin export -- \
  --model-dir ./models/my-model \
  --output ./exports/model.gguf \
  --format gguf
```

---

## 11. 生产部署最佳实践

### 11.1 架构建议

推荐使用以下架构进行生产部署：

1. **FastAPI + Uvicorn**：创建 RESTful API
2. **Redis**：缓存频繁请求的结果
3. **Docker**：容器化部署
4. **Kubernetes**：集群管理（大规模部署）

### 11.2 监控与日志

- 使用 Prometheus 监控推理延迟
- 使用 Grafana 可视化性能指标
- 使用 ELK Stack 收集和分析日志

### 11.3 安全建议

- 使用 HTTPS 加密传输
- 配置适当的认证机制
- 限制 API 访问频率
- 定期更新模型和依赖

---

## 12. 故障排除

### 12.1 常见问题

#### 问题：API服务器无法启动
**解决方案：**
- 检查端口是否被占用
- 验证模型文件是否完整
- 查看日志文件获取详细错误信息

#### 问题：推理速度慢
**解决方案：**
- 使用 GPU 加速：`--backend gpu`
- 启用量化：`--quantize`
- 调整批量大小和序列长度

#### 问题：内存不足
**解决方案：**
- 减少批量大小
- 使用更小的模型
- 启用流式处理

### 12.2 性能调优建议

- **批量大小**：根据 GPU 内存调整，GPU 模式推荐 64-256
- **学习率**：初始学习率 1e-4，可逐步调整
- **序列长度**：根据数据特点调整，一般 128-512
- **训练轮数**：根据数据量调整，一般 50-200 轮

---

## 13. 版本更新记录

| 版本 | 主要特性 |
|------|----------|
| 0.1.0 | 初始版本，基础部署功能 |
| 0.1.1 | 添加 GPU 优化和量化支持 |
| 0.1.2 | 添加完整 API 接口支持 |

---

**更新日期：** 2026-03-25  
**版本：** v1.1  
**作者：** Sage团队