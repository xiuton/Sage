
# Sage API 使用指南

本文档提供 Sage API 服务器的完整使用说明，包括启动方法、API 端点、参数说明和使用示例。

## 目录

1. [快速开始](#快速开始)
2. [启动参数](#启动参数)
3. [认证机制](#认证机制)
4. [API 端点列表](#api-端点列表)
5. [聊天完成](#聊天完成)
6. [文本补全](#文本补全)
7. [图像生成](#图像生成)
8. [Diffusion 模型管理](#diffusion-模型管理)
9. [训练任务管理](#训练任务管理)
10. [性能监控](#性能监控)
11. [限流信息](#限流信息)
12. [实时通信](#实时通信)

---

## 快速开始

### 1. 启动 API 服务器

#### 完整模式（LLM + 多模态）
需要同时提供 tokenizer.json 和 model.mpk 文件：

```bash
# 基本启动（CPU 后端）
cargo run --release --features="api" --bin api_server -- `
    --model-dir ./models/sage_model_formal `
    --port 8000

# 使用 GPU 后端
cargo run --release --features="api" --bin api_server -- `
    --model-dir ./models/sage_model_formal `
    --backend gpu `
    --port 8000

# 启用 API Key 认证
$env:SAGE_API_KEY="your-secret-key"
cargo run --release --features="api" --bin api_server -- `
    --model-dir ./models/sage_model_formal `
    --port 8000 `
    --max-concurrent 4
```

#### 多模态专用模式
仅提供多模态图像生成功能，不需要 LLM 模型文件：

```bash
# 启动多模态专用服务器（使用训练后的多模态模型）
cargo run --release --features="api" --bin api_server -- `
    --model-dir ./models/text_to_image_full `
    --backend gpu `
    --port 8000

# 启动 CPU 版本
cargo run --release --features="api" --bin api_server -- `
    --model-dir ./models/text_to_image_full `
    --backend cpu `
    --port 8000
```

**注意**：多模态专用模式下，聊天完成和文本补全功能不可用，但可以加载和使用 Diffusion 模型进行图像生成。

### 2. 健康检查

```bash
curl http://localhost:8000/health
```

响应示例：
```json
{
  "status": "ok",
  "timestamp": 1713945600,
  "version": "0.1.0"
}
```

---

## 启动参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 模型目录路径 | `./models/sage_model_formal` |
| `--port` | 服务器端口 | 8000 |
| `--backend` | 推理后端（`cpu` 或 `gpu`） | cpu |
| `--log-level` | 日志级别（`error`, `warn`, `info`, `debug`, `trace`） | info |
| `--max-concurrent` | 最大并发请求数 | 4 |
| `--api-key` | API密钥（也可通过环境变量 `SAGE_API_KEY` 设置） | 无 |

---

## 服务模式

API 服务器支持两种运行模式：

### 完整模式
- 需要 `model-dir` 目录下存在 `tokenizer.json` 和 `model.mpk`
- 提供完整的 LLM 对话功能
- 同时支持多模态图像生成（通过 `/api/v1/diffusion/*` 端点）

### 多模态专用模式
- 仅需要 `model-dir` 目录下存在扩散模型文件（`config.json`、`diffusion_model.mpk`）
- LLM 对话功能不可用（`/api/v1/chat/completions` 和 `/api/v1/completions` 返回错误）
- 专用提供多模态图像生成功能

启动时会自动检测模型文件并选择合适的模式：
- 如果同时存在 LLM 和多模态模型，提供完整服务
- 如果只有多模态模型，自动切换到多模态专用模式

---

## 认证机制

- 若设置 `--api-key` 或环境变量 `SAGE_API_KEY`，则除 `/health` 和 `/api/v1/models` 外的接口需要 `Authorization: Bearer &lt;API_KEY&gt;` 头。
- 若未设置，则不做认证。

---

## API 端点列表

| 方法 | 端点 | 描述 | 认证 |
|------|------|------|------|
| GET | `/health` | 健康检查 | 不需要 |
| GET | `/api/v1/models` | 列出可用模型 | 不需要 |
| GET | `/api/v1/model/info` | 获取模型详细信息 | 需要 |
| POST | `/api/v1/chat/completions` | 聊天完成（支持流式） | 需要 |
| POST | `/api/v1/completions` | 文本补全 | 需要 |
| POST | `/api/v1/images/generate` | 图像生成 | 需要 |
| POST | `/api/v1/images/generations` | 批量图像生成 | 需要 |
| POST | `/api/v1/diffusion/load` | 加载 Diffusion 模型 | 需要 |
| POST | `/api/v1/diffusion/unload` | 卸载 Diffusion 模型 | 需要 |
| POST | `/api/v1/training/start` | 启动训练任务 | 需要 |
| GET | `/api/v1/training/status/:id` | 查询训练状态 | 需要 |
| POST | `/api/v1/training/cancel/:id` | 取消训练任务 | 需要 |
| GET | `/api/v1/training/list` | 列出所有训练任务 | 需要 |
| GET | `/api/v1/performance` | 获取性能统计 | 需要 |
| GET | `/api/v1/rate-limit` | 获取限流信息 | 需要 |
| WS | `/ws` | WebSocket 实时通信 | 需要 |
| GET | `/events` | SSE 事件流 | 需要 |

---

## 聊天完成

> **注意**：此功能需要在完整模式下启动API服务器（需要 LLM 模型文件）。

### 普通模式

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "model": "sage-llm",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手"},
      {"role": "user", "content": "什么是深度学习？"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,
    "top_p": 0.9,
    "top_k": 10
  }'
```

响应示例：
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1713945600,
  "model": "sage-llm",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "深度学习是机器学习的一个分支，使用多层神经网络..."
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

### 流式输出

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "model": "sage-llm",
    "messages": [{"role": "user", "content": "什么是深度学习？"}],
    "temperature": 0.7,
    "max_tokens": 100,
    "stream": true
  }'
```

流式响应使用 SSE 格式：
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1713945600,"model":"sage-llm","choices":[{"index":0,"message":{"role":"assistant","content":"深"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1713945600,"model":"sage-llm","choices":[{"index":0,"message":{"role":"assistant","content":"深度学习"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1713945600,"model":"sage-llm","choices":[{"index":0,"message":{"role":"assistant","content":"深度学习是"},"finish_reason":null}]}
```

### 请求参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `model` | string | 模型名称 | `sage-llm` |
| `messages` | array | 消息数组（必填） | - |
| `temperature` | float | 采样温度（0-2） | 0.8 |
| `max_tokens` | integer | 最大生成 token 数 | 50 |
| `top_p` | float | Top-p 采样（0-1） | 0.9 |
| `top_k` | integer | Top-k 采样 | 10 |
| `n` | integer | 生成的回复数量 | 1 |
| `stop` | array | 停止序列 | [] |
| `presence_penalty` | float | 存在惩罚 | 0.0 |
| `frequency_penalty` | float | 频率惩罚 | 0.0 |
| `seed` | integer | 随机种子 | null |
| `stream` | boolean | 是否启用流式输出 | false |

---

## 文本补全

```bash
curl -X POST http://localhost:8000/api/v1/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "prompt": "什么是深度学习？",
    "max_length": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 10
  }'
```

响应示例：
```json
{
  "prompt": "什么是深度学习？",
  "text": "深度学习是机器学习的一个分支，使用多层神经网络..."
}
```

---

## 图像生成

> **注意**：在多模态专用模式下，图像生成功能可用，但需要先调用 `/api/v1/diffusion/load` 加载模型。

### 加载 Diffusion 模型

在生成图像前，需要先加载 Diffusion 模型：

```bash
# 完整路径方式
curl -X POST http://localhost:8000/api/v1/diffusion/load `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "model_path": "./models/text_to_image_full",
    "config_path": "./configs/config_vae_diffusion.json"
  }'

# 简化方式（自动在 models 目录中查找）
curl -X POST http://localhost:8000/api/v1/diffusion/load `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "model_path": "text_to_image_full",
    "config_path": ""
  }'
```

响应示例：
```json
{
  "status": "loaded",
  "model_path": "./models/text_to_image_full",
  "config": {
    "image_size": 64,
    "hidden_channels": 128,
    "latent_dim": 128,
    "num_timesteps": 1000
  }
}
```

### 生成图像

```bash
curl -X POST http://localhost:8000/api/v1/images/generate `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "prompt": "一只可爱的小猫",
    "model_path": "./models/text_to_image_full",
    "steps": 100,
    "latent_dim": 128,
    "image_size": 64
  }'
```

响应示例：
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "image_path": "./assets/550e8400-e29b-41d4-a716-446655440000.png",
  "message": "Image generated successfully"
}
```

### 批量图像生成

```bash
curl -X POST http://localhost:8000/api/v1/images/generations `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "prompt": "一只可爱的小猫",
    "steps": 100
  }'
```

---

## Diffusion 模型管理

### 加载模型

参见 [图像生成](#图像生成) 部分。

### 卸载模型

```bash
curl -X POST http://localhost:8000/api/v1/diffusion/unload `
  -H "Authorization: Bearer your-secret-key"
```

响应示例：
```json
{
  "status": "unloaded"
}
```

---

## 训练任务管理

### 启动训练

```bash
curl -X POST http://localhost:8000/api/v1/training/start `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "mode": "text_to_image",
    "data_path": "./data/text_to_image_pairs.jsonl",
    "config_path": "./configs/config_vae_diffusion.json",
    "output_dir": "./models/training_output",
    "batch_size": 4,
    "learning_rate": 0.0001,
    "num_epochs": 10,
    "backend": "cpu"
  }'
```

响应示例：
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "message": "Training started with 10 epochs. Check status at /api/v1/training/status/550e8400-e29b-41d4-a716-446655440000",
  "output_dir": "./models/training_output"
}
```

### 查询训练状态

```bash
curl -X GET http://localhost:8000/api/v1/training/status/550e8400-e29b-41d4-a716-446655440000 `
  -H "Authorization: Bearer your-secret-key"
```

响应示例：
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "mode": "text_to_image",
  "start_time": 1713945600,
  "current_epoch": 3,
  "total_epochs": 10,
  "progress_percent": 30.0,
  "loss": 0.1234,
  "output_dir": "./models/training_output",
  "message": "Epoch 3/10, loss: 0.1234",
  "error": null
}
```

状态值：
- `running`: 训练进行中
- `completed`: 训练完成
- `cancelled`: 训练已取消

### 取消训练

```bash
curl -X POST http://localhost:8000/api/v1/training/cancel/550e8400-e29b-41d4-a716-446655440000 `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "force": false
  }'
```

### 列出所有训练任务

```bash
curl -X GET http://localhost:8000/api/v1/training/list `
  -H "Authorization: Bearer your-secret-key"
```

响应示例：
```json
{
  "object": "list",
  "data": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "running",
      "mode": "text_to_image",
      "current_epoch": 3,
      "total_epochs": 10,
      "progress_percent": 30.0,
      "loss": 0.1234,
      "output_dir": "./models/training_output",
      "start_time": 1713945600,
      "message": "Epoch 3/10, loss: 0.1234"
    }
  ],
  "total": 1
}
```

---

## 性能监控

```bash
curl -X GET http://localhost:8000/api/v1/performance `
  -H "Authorization: Bearer your-secret-key"
```

响应示例：
```json
{
  "total_requests": 100,
  "total_tokens": 5000,
  "total_errors": 2,
  "avg_response_time_ms": 123.45,
  "requests_by_endpoint": {
    "/api/v1/chat/completions": 80,
    "/api/v1/images/generate": 20
  },
  "inference_stats": {
    "total_inference_requests": 100,
    "total_inference_time_ms": 12345,
    "avg_inference_time_ms": 123.45
  },
  "model_loading": {
    "total_loads": 2,
    "loaded_models": [
      {
        "name": "diffusion",
        "type": "diffusion",
        "loaded_at": 1713945600,
        "reference_count": 1
      }
    ]
  }
}
```

---

## 限流信息

```bash
curl -X GET http://localhost:8000/api/v1/rate-limit `
  -H "Authorization: Bearer your-secret-key"
```

响应示例：
```json
{
  "requests_remaining": 58,
  "tokens_remaining": 49800,
  "limit": {
    "requests_per_minute": 60,
    "tokens_per_minute": 60000
  }
}
```

**限流规则：**
- 每分钟最多 60 个请求
- 每分钟最多 60000 个 tokens
- 超过限制会返回 `429 Too Many Requests`

---

## 实时通信

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('WebSocket connected');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch (data.type) {
    case 'chat':
      console.log('Chat message:', data.content);
      break;
    case 'image':
      console.log('Image generated:', data.path);
      break;
    case 'training':
      console.log('Training update:', data.id, 'Progress:', data.progress);
      break;
    case 'error':
      console.error('Error:', data.message);
      break;
  }
};

ws.onclose = () => {
  console.log('WebSocket disconnected');
};
```

### SSE 事件流

```javascript
const eventSource = new EventSource('http://localhost:8000/events');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  switch (data.type) {
    case 'chat':
      console.log('Chat message:', data.content);
      break;
    case 'image':
      console.log('Image generated:', data.path);
      break;
    case 'training':
      console.log('Training update:', data.id, 'Progress:', data.progress);
      break;
    case 'error':
      console.error('Error:', data.message);
      break;
  }
};

eventSource.onerror = (error) => {
  console.error('EventSource error:', error);
  eventSource.close();
};
```

---

## 错误处理

API 使用统一的错误响应格式：

```json
{
  "error": "error_code",
  "message": "详细错误信息"
}
```

常见错误码：
- `bad_request`: 请求参数错误（400）
- `unauthorized`: 认证失败（401）
- `rate_limit_exceeded`: 超过限流（429）
- `model_not_loaded`: Diffusion 模型未加载（400）
- `task_not_found`: 训练任务不存在（404）
- `invalid_state`: 任务状态无效（400）

---

## 最佳实践

1. **使用流式输出**：对于长文本生成，启用 `stream: true` 获得更好的用户体验
2. **合理设置参数**：根据需求调整 `max_tokens` 和 `temperature`
3. **监控限流**：定期检查 `/api/v1/rate-limit` 避免被限制
4. **错误重试**：对于临时错误（如 `429`），实现指数退避重试
5. **使用 GPU**：生产环境建议使用 `--backend gpu` 提升推理速度

---

**更新日期：** 2026-04-25  
**版本：** v1.3  
**作者：** Sage团队
