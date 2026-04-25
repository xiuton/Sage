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
9. [API使用指南](#api使用指南)

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
- **Rust**: 建议 1.85+（本项目使用 `edition = "2024"`；推荐使用 rustup 安装）

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
./models/your_model/
├── model.mpk          # 模型权重文件
├── best_model.mpk     # 最佳模型权重（如果启用）
├── config.json        # 模型配置文件
├── tokenizer.json     # 分词器配置文件
└── checkpoint/        # 训练检查点目录
    ├── model-10.mpk   # 第10轮检查点
    ├── model-20.mpk   # 第20轮检查点
    └── ...
```

### 2.2 模型验证

```bash
# 检查模型文件是否完整
ls -la ./models/your_model/

# 验证模型配置
cat ./models/your_model/config.json
```

### 2.3 创建部署目录

```bash
# 创建部署目录
mkdir -p ./deploy

# 复制模型文件
cp ./models/your_model/model.mpk ./deploy/model.mpk
cp ./models/your_model/best_model.mpk ./deploy/best_model.mpk
cp ./models/your_model/config.json ./deploy/config.json
cp ./models/your_model/tokenizer.json ./deploy/tokenizer.json
```

### 2.4 模型验证测试

```bash
# 使用训练好的模型进行推理测试
cargo run --release --bin infer -- `
    --model-dir ./models/your_model `
    --use-best `
    --prompt "什么是深度学习？" `
    --num-tokens 100 `
    --temperature 0.7
```

---

## 3. 交互式部署

### 3.1 命令行交互式部署

```bash
# 使用infer工具提供交互式推理服务
cargo run --release --bin infer -- `
    --model-dir ./models/your_model `
    --use-best `
    --chat `
    --interactive
```

### 3.2 流式输出部署

```bash
# 启用流式输出的交互式部署
cargo run --release --bin infer -- `
    --model-dir ./models/your_model `
    --use-best `
    --chat `
    --interactive `
    --stream `
    --stream-speed 10
```

### 3.3 GPU加速交互式部署

```bash
# 使用GPU加速的交互式部署
cargo run --release --bin infer -- `
    --model-dir ./models/your_model `
    --use-best `
    --chat `
    --interactive `
    --backend gpu
```

### 3.4 多模态推理部署

Sage 支持完整的多模态能力，包括图像生成和多模态理解。

#### 支持的功能
- **推理侧**：使用 `infer --multimodal --image-path <图像路径>` 进行图像输入推理
- **图像生成**：使用 `image_gen` 工具进行文本到图像生成
- **核心组件**：已实现 `VisionEncoder`、`MultimodalFusion`、`VAE` 和 `DiffusionModel` 模块
- **模型架构**：支持文本和图像特征融合，以及文本到图像生成

#### 使用示例

##### 1. 多模态推理
```bash
# 基本多模态推理
cargo run --bin infer -- `
    --multimodal `
    --image-path ./test_image.jpg `
    --prompt "描述这张图片"

# 多模态推理（使用GPU加速）
cargo run --bin infer -- `
    --multimodal `
    --image-path ./test_image.jpg `
    --prompt "描述这张图片" `
    --backend gpu
```

##### 2. 图像生成（文生图）
```bash
# 使用CPU生成图像
cargo run --bin image_gen -- `
    --backend cpu `
    --model-path models/text_to_image `
    --prompt "一只可爱的小猫" `
    --steps 50

# 使用GPU生成高质量图像
cargo run --bin image_gen -- `
    --backend gpu `
    --model-path models/text_to_image_full `
    --prompt "一只可爱的小猫，毛茸茸的，蓝色眼睛，在草地上玩耍" `
    --steps 100 `
    --output ./cat_generated.png
```

---

## 4. API服务部署

### 4.1 启动API服务器

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

### 4.2 API服务器参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-dir` | 模型目录路径 | `./models/sage_model_formal` |
| `--port` | 服务器端口 | 8000 |
| `--backend` | 推理后端（`cpu` 或 `gpu`） | cpu |
| `--log-level` | 日志级别（`error`, `warn`, `info`, `debug`, `trace`） | info |
| `--max-concurrent` | 最大并发请求数 | 4 |
| `--api-key` | API密钥（也可通过环境变量 `SAGE_API_KEY` 设置） | 无 |

认证机制：
- 若设置 `--api-key` 或环境变量 `SAGE_API_KEY`，则除 `/health` 和 `/api/v1/models` 外的接口需要 `Authorization: Bearer <API_KEY>`。
- 若未设置，则不做认证。

### 4.3 健康检查

```bash
# 检查API服务器是否正常运行
curl http://localhost:8000/health
```

### 4.4 获取模型信息

```bash
curl -X GET http://localhost:8000/api/v1/model/info `
  -H "Authorization: Bearer your-secret-key"
```

---

## 5. Docker容器部署

### 5.1 构建Docker镜像

```bash
# 构建Docker镜像
docker build -t sage-api .

# 运行Docker容器（完整模式）
docker run -d `
  -p 8000:8000 `
  -v ./models/sage_model_formal:/app/models `
  --name sage-api `
  sage-api

# 运行Docker容器（多模态专用模式）
docker run -d `
  -p 8000:8000 `
  -v ./models/text_to_image_full:/app/models `
  --name sage-api-multimodal `
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

- **量化**：当前量化为框架/体积估算，未提供真实的权重量化推理加速；如需部署侧加速，需要补齐算子替换与端到端验证
- **剪枝**：移除不重要的权重（未来支持）
- **蒸馏**：知识蒸馏减小模型大小（未来支持）
- **多模态支持**：支持图像输入的模型部署

### 6.2 推理优化

- **批处理**：实现批处理推理提高吞吐量
- **缓存**：缓存频繁使用的计算结果
- **并行处理**：使用多线程或异步处理

### 6.3 GPU 利用率优化

Sage 使用 CubeCL 作为 GPU 计算后端。可以通过以下方式优化 GPU 利用率：

**CubeCL Autotune 级别：**

| 级别 | 描述 | 首次启动速度 | GPU 利用率 |
|------|------|-------------|-----------|
| `minimal` | 最快但不充分 | 最快 | 低 |
| `balanced` | 良好平衡（默认） |较快 | 较高 |
| `extensive` | 更彻底 | 较慢 | 高 |
| `full` | 最彻底 | 最慢 | 最高 |

**设置方式（环境变量）：**
```bash
# Linux/macOS
export CUBECL_AUTOTUNE_LEVEL=balanced

# Windows PowerShell
$env:CUBECL_AUTOTUNE_LEVEL="balanced"
```

**注意：** 更高的 autotune 级别会在首次运行时进行更充分的 kernel 优化测试，选择最优的 GPU kernel 配置。优化结果会被缓存，后续启动会直接使用。

### 6.5 部署优化

- **使用ONNX格式**：导出为ONNX格式进行部署（未来支持）
- **GPU加速**：在生产环境使用GPU加速推理
- **内存管理**：优化内存使用减少资源消耗

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
curl -X POST http://localhost:8000/api/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
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
| `/health` | GET | 健康检查 | 不需要 |
| `/api/v1/models` | GET | 列出可用模型 | 不需要 |
| `/api/v1/model/info` | GET | 获取模型详细信息 | 需要 |
| `/api/v1/chat/completions` | POST | 聊天完成（支持流式） | 需要 |
| `/api/v1/completions` | POST | 文本补全 | 需要 |
| `/api/v1/images/generate` | POST | 图像生成 | 需要 |
| `/api/v1/images/generations` | POST | 批量图像生成 | 需要 |
| `/api/v1/diffusion/load` | POST | 加载 Diffusion 模型 | 需要 |
| `/api/v1/diffusion/unload` | POST | 卸载 Diffusion 模型 | 需要 |
| `/api/v1/training/start` | POST | 启动训练任务 | 需要 |
| `/api/v1/training/status/:id` | GET | 查询训练状态 | 需要 |
| `/api/v1/training/cancel/:id` | POST | 取消训练任务 | 需要 |
| `/api/v1/training/list` | GET | 列出所有训练任务 | 需要 |
| `/api/v1/performance` | GET | 获取性能统计 | 需要 |
| `/api/v1/rate-limit` | GET | 获取限流信息 | 需要 |
| `/ws` | WS | WebSocket 实时通信 | 需要 |
| `/events` | GET | SSE 事件流 | 需要 |

### 9.2 /api/v1/completions 文本补全接口

**请求示例：**
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

**响应示例：**
```json
{
  "prompt": "什么是深度学习？",
  "text": "深度学习是机器学习的一个分支，使用多层神经网络..."
}
```

### 9.3 /api/v1/chat/completions 聊天完成接口（OpenAI标准）

**请求示例（普通模式）：**
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

**请求示例（流式输出）：**
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

**流式输出响应说明：**
流式输出使用Server-Sent Events (SSE)格式，每个token生成后立即返回，格式如下：
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"sage-llm","choices":[{"index":0,"message":{"role":"assistant","content":"深"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677858242,"model":"sage-llm","choices":[{"index":0,"message":{"role":"assistant","content":"深度学习"},"finish_reason":null}]}
```

**响应示例：**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "sage-llm",
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

### 9.4 图像生成接口

**请求示例：**
```bash
curl -X POST http://localhost:8000/api/v1/images/generate `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "prompt": "一只可爱的小猫",
    "model_path": "text_to_image_full",
    "steps": 100,
    "latent_dim": 128,
    "image_size": 64
  }'
```

**响应示例：**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "image_path": "./assets/550e8400-e29b-41d4-a716-446655440000.png",
  "message": "Image generated successfully"
}
```

### 9.5 Diffusion 模型管理接口

**加载模型：**
```bash
curl -X POST http://localhost:8000/api/v1/diffusion/load `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your-secret-key" `
  -d '{
    "model_path": "text_to_image_full",
    "config_path": ""
  }'
```

**卸载模型：**
```bash
curl -X POST http://localhost:8000/api/v1/diffusion/unload `
  -H "Authorization: Bearer your-secret-key"
```

### 9.6 训练任务管理接口

**启动训练：**
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

**查询训练状态：**
```bash
curl -X GET http://localhost:8000/api/v1/training/status/550e8400-e29b-41d4-a716-446655440000 `
  -H "Authorization: Bearer your-secret-key"
```

**列出所有训练任务：**
```bash
curl -X GET http://localhost:8000/api/v1/training/list `
  -H "Authorization: Bearer your-secret-key"
```

---



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
- 调小上下文长度：`--context-len`（过大时每步计算更慢）
- 使用更小的模型配置（训练侧 `--model-size`），或降低生成长度（`infer -n`）

#### 问题：内存不足
**解决方案：**
- 减少批量大小
- 使用更小的模型
- 启用流式处理

### 12.2 性能调优建议

- **推理上下文长度**：控制 `--context-len`，不超过训练时的 `max_seq_len`
- **生成长度**：通过 `infer -n` 限制输出 token 数
- **生成策略**：调低 `-t/-p/-k` 可提升稳定性并减少无效生成

---

## 10. 版本更新记录

| 版本 | 主要特性 |
|------|----------|
| 0.1.0 | 初始版本，基础部署功能 |
| 0.1.1 | 添加 GPU 推理支持与部署指南完善 |
| 0.1.2 | 添加完整 API 接口支持 |
| 1.0.0 | 添加 DPO 偏好对齐与多模态推理支持 |
| 1.1.0 | 完善 API 服务模式：流式输出、WebSocket、训练任务管理、性能监控 |
| 1.2.0 | 扩展 API 服务器支持多模态，添加专门的多模态路由 |
| 1.3.0 | 优化 API 接口，支持通过模型名称自动查找模型（无需完整路径） |

---

**更新日期：** 2026-04-25  
**版本：** v1.3  
**作者：** Sage团队
