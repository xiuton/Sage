use burn::backend::{ndarray::{NdArray, NdArrayDevice}, wgpu::Wgpu};
use burn::prelude::Backend;

use clap::Parser;
use sage::{
    generation::GenerateOptions,
    lazy_load::LazyModel,
    logger::init_logger,
    performance::PerformanceMonitor,
    tokenizer::Tokenizer,
    TrainingConfig,
};
use sage::{log_info, log_error, model_download::ModelDownloader};
use axum::{
    extract::{Json, Path, Request},
    http::{header, StatusCode},
    middleware::{self, Next},
    routing::{get, post, delete},
    Router,
};

use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    env,
    fs,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::{
    net::TcpListener,
    sync::mpsc,
    task,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "./tmp/sage_model_formal")]
    model_dir: String,

    #[arg(long, default_value_t = false)]
    use_best: bool,

    #[arg(long, default_value_t = 8000)]
    port: u16,

    #[arg(long, default_value_t = 0)]
    context_len: usize,

    #[arg(long, default_value = "cpu", value_name = "cpu|gpu")]
    backend: String,
    
    #[arg(long)]
    quantize: bool,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    n: Option<usize>,
    stop: Option<Vec<String>>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    stream: Option<bool>,
}

#[derive(Serialize, Clone, Debug)]
struct ChatCompletionChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize, Clone, Debug)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

#[derive(Deserialize)]
struct BatchChatCompletionRequest {
    requests: Vec<ChatCompletionRequest>,
}

#[derive(Serialize)]
struct BatchChatCompletionResponse {
    responses: Vec<ChatCompletionResponse>,
    total_duration_ms: u64,
    request_count: usize,
}

#[derive(Serialize, Debug, Clone)]
enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug)]
struct AsyncTask {
    task_id: String,
    status: TaskStatus,
    request: ChatCompletionRequest,
    result: Option<ChatCompletionResponse>,
    error: Option<String>,
    created_at: Instant,
    started_at: Option<Instant>,
    completed_at: Option<Instant>,
}

#[derive(Deserialize)]
struct AsyncChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    n: Option<usize>,
    stop: Option<Vec<String>>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    stream: Option<bool>,
}

#[derive(Serialize)]
struct AsyncTaskResponse {
    task_id: String,
    status: TaskStatus,
    result: Option<ChatCompletionResponse>,
    error: Option<String>,
}

#[derive(Serialize)]
struct TaskStatusResponse {
    task_id: String,
    status: TaskStatus,
    result: Option<ChatCompletionResponse>,
    error: Option<String>,
    created_at: u64,
    started_at: Option<u64>,
    completed_at: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ModelInfo {
    model_id: String,
    model_dir: String,
    status: String,
    size: String,
    backend: String,
    loaded_at: u64,
}

#[derive(Serialize, Deserialize, Debug)]
struct ModelLoadRequest {
    model_id: String,
    model_dir: String,
    use_best: bool,
    quantize: bool,
}

#[derive(Serialize)]
struct ModelLoadResponse {
    model_id: String,
    status: String,
    message: String,
}

#[derive(Serialize)]
struct ModelListResponse {
    models: Vec<ModelInfo>,
    active_model: Option<String>,
}

#[derive(Serialize)]
struct ModelSwitchResponse {
    model_id: String,
    status: String,
    message: String,
}

struct AppState {
    tokenizer: Arc<Mutex<Tokenizer>>,
    lazy_model: Arc<Mutex<LazyModel<NdArray>>>,
    lazy_model_gpu: Arc<Mutex<Option<LazyModel<Wgpu>>>>,
    config: Arc<Mutex<TrainingConfig>>,
    context_len: usize,
    api_key: Option<String>,
    tasks: Arc<Mutex<HashMap<String, AsyncTask>>>,
    task_queue: Arc<Mutex<VecDeque<String>>>,
    task_sender: mpsc::Sender<String>,
    _backend: String,
    performance_monitor: PerformanceMonitor,
    
    // 模型管理
    models: Arc<Mutex<HashMap<String, ModelInfo>>>,
    active_model: Arc<Mutex<String>>,
    
    // 模型下载器
    model_downloader: Arc<ModelDownloader>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    
    // 初始化日志系统
    init_logger();
    
    log_info!("正在启动API服务器...");
    log_info!("模型目录: {}", args.model_dir);
    log_info!("端口: {}", args.port);

    // 加载模型配置
    let config_path = format!("{}/config.json", args.model_dir);
    let config_str = fs::read_to_string(&config_path).expect("Failed to read config");
    let config: TrainingConfig = serde_json::from_str(&config_str).expect("Failed to parse config");

    // 加载分词器
    let tokenizer_path = format!("{}/tokenizer.json", args.model_dir);
    let tokenizer = Tokenizer::load(&tokenizer_path).expect("Failed to load tokenizer");

    // 根据后端类型加载模型
    let model_path = if args.use_best {
        let best_path = format!("{}/best_model.mpk", args.model_dir);
        if fs::metadata(&best_path).is_ok() {
            best_path
        } else {
            format!("{}/model.mpk", args.model_dir)
        }
    } else {
        format!("{}/model.mpk", args.model_dir)
    };

    let (lazy_model, lazy_model_gpu, backend) = if args.backend == "gpu" {
        log_info!("使用GPU后端进行推理...");
        unsafe {
            std::env::set_var("WGPU_POWER_PREFERENCE", "HighPerformance");
        }
        log_info!("初始化GPU懒加载模型...");
        let lazy_model_gpu = Some(LazyModel::new(config.model.clone(), model_path.clone()));
        let lazy_model = LazyModel::new(config.model.clone(), model_path.clone());
        (lazy_model, lazy_model_gpu, "gpu".to_string())
    } else {
        log_info!("使用CPU后端进行推理...");
        log_info!("初始化CPU懒加载模型...");
        let lazy_model = LazyModel::new(config.model.clone(), model_path.clone());
        (lazy_model, None, "cpu".to_string())
    };

    // 从环境变量加载API密钥
    let api_key = env::var("SAGE_API_KEY").ok();

    // 创建任务队列和通信通道
    let (task_sender, task_receiver) = mpsc::channel(100);
    let tasks = Arc::new(Mutex::new(HashMap::new()));
    let task_queue = Arc::new(Mutex::new(VecDeque::new()));

    // 初始化模型管理
    let mut models = HashMap::new();
    let model_info = ModelInfo {
        model_id: "default".to_string(),
        model_dir: args.model_dir.clone(),
        status: "loaded".to_string(),
        size: format!("{}m", config.model.d_model / 1024 / 1024),
        backend: backend.clone(),
        loaded_at: Instant::now().elapsed().as_millis() as u64,
    };
    models.insert("default".to_string(), model_info);
    
    // 创建模型下载器
    let model_downloader = Arc::new(ModelDownloader::new(&args.model_dir));
    
    let state = Arc::new(AppState {
        tokenizer: Arc::new(Mutex::new(tokenizer)),
        lazy_model: Arc::new(Mutex::new(lazy_model)),
        lazy_model_gpu: Arc::new(Mutex::new(lazy_model_gpu)),
        config: Arc::new(Mutex::new(config)),
        context_len: args.context_len,
        api_key,
        tasks: tasks.clone(),
        task_queue: task_queue.clone(),
        task_sender,
        _backend: backend,
        performance_monitor: PerformanceMonitor::new(),
        models: Arc::new(Mutex::new(models)),
        active_model: Arc::new(Mutex::new("default".to_string())),
        model_downloader,
    });

    // 启动后台任务处理器
    let state_clone = state.clone();
    task::spawn(async move {
        task_processor(state_clone, task_receiver).await;
    });

    let app = Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/model-info", get(model_info_handler))
        .route("/api/performance", get(performance_handler))
        .route("/v1/chat/completions", post(infer_handler))
        .route("/v1/batch-chat/completions", post(batch_infer_handler))
        .route("/v1/async-chat/completions", post(async_infer_handler))
        .route("/api/task/:task_id", get(task_status_handler))
        
        // 模型管理接口
        .route("/api/models", get(list_models_handler))
        .route("/api/models", post(load_model_handler))
        .route("/api/models/:model_id", delete(unload_model_handler))
        .route("/api/models/:model_id/activate", post(switch_model_handler))
        .route("/api/models/:model_id/reload", post(reload_model_handler))
        
        // 模型下载和更新接口
        .route("/api/models/download", post(download_model_handler))
        
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .with_state(state.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    log_info!("API服务器启动在 http://{}", addr);
    
    let listener = TcpListener::bind(addr).await.unwrap();
    
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

async fn health_handler() -> StatusCode {
    StatusCode::OK
}

async fn model_info_handler(state: axum::extract::State<Arc<AppState>>) -> Json<serde_json::Value> {
    let config = state.config.lock().unwrap();
    let tokenizer = state.tokenizer.lock().unwrap();
    
    let info = serde_json::json!({
        "model_config": {
            "vocab_size": config.model.vocab_size,
            "max_seq_len": config.model.max_seq_len,
            "d_model": config.model.d_model,
            "d_ff": config.model.d_ff,
            "n_layers": config.model.n_layers,
            "n_heads": config.model.n_heads,
        },
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
        },
        "training_config": {
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "lr": config.lr,
        },
    });
    
    Json(info)
}

async fn performance_handler(state: axum::extract::State<Arc<AppState>>) -> Json<serde_json::Value> {
    let all_metrics = state.performance_monitor.get_all_metrics();
    
    let mut performance_data = serde_json::json!({});
    
    for (endpoint, metrics_list) in all_metrics {
        let avg_metrics = state.performance_monitor.get_average_metrics(&endpoint);
        
        let endpoint_data = serde_json::json!({
            "request_count": metrics_list.len(),
            "average": avg_metrics.map(|m| serde_json::json!({
                "inference_time_ms": m.inference_time_ms,
                "tokens_per_second": m.tokens_per_second,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "total_tokens": m.total_tokens,
            })).unwrap_or(serde_json::json!(null)),
            "samples": metrics_list.into_iter().map(|m| serde_json::json!({
                "inference_time_ms": m.inference_time_ms,
                "tokens_per_second": m.tokens_per_second,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "total_tokens": m.total_tokens,
            })).collect::<Vec<_>>(),
        });
        
        performance_data[endpoint] = endpoint_data;
    }
    
    Json(performance_data)
}

async fn infer_handler(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    log_info!("收到ChatCompletion请求: messages数量={}, stream={:?}", req.messages.len(), req.stream);
    
    let start_time = std::time::Instant::now();

    if req.messages.is_empty() {
        log_error!("错误: messages数组为空");
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "bad_request".to_string(),
                message: "Messages array cannot be empty".to_string(),
            }),
        ));
    }

    // 格式化messages为聊天格式
    let formatted_prompt = format_messages_to_prompt(&req.messages);
    log_info!("格式化后的提示: {}", formatted_prompt);

    let options = GenerateOptions {
        max_new_tokens: req.max_tokens.unwrap_or(50),
        temperature: req.temperature.unwrap_or(0.8),
        top_k: req.top_k.unwrap_or(10),
        top_p: req.top_p.unwrap_or(0.9),
        repetition_penalty: 1.1,
        punctuation_penalty: 1.3,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        seed: req.seed,
        context_len: state.context_len,
        stop_on_user: true,
        stop_sequences: req.stop.unwrap_or(Vec::new()),
    };

    // 根据后端类型选择设备
    let (reply, _duration_ms) = if state._backend == "gpu" && state.lazy_model_gpu.lock().unwrap().is_some() {
        let device = <Wgpu as Backend>::Device::default();
        let lazy_model_gpu = state.lazy_model_gpu.lock().unwrap();
        let model = lazy_model_gpu.as_ref().unwrap().get_model(&device);
        let model = model.lock().unwrap();
        let tokenizer = state.tokenizer.lock().unwrap();
        
        if req.stream.unwrap_or(false) {
            // 流式输出处理
            let created_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            let request_id = uuid::Uuid::new_v4().to_string();
            let model_name = req.model.unwrap_or_else(|| "sage-model".to_string());
            
            let mut generation_state = sage::generation::GenerationState::new(sage::generation::ModelType::Normal(&*model), &tokenizer, &formatted_prompt, &options, &device);
                
                let mut full_content = String::new();
                let mut chunks = Vec::new();
                
                while !generation_state.is_stopped() {
                    if let Some(token) = generation_state.next_token() {
                        full_content.push_str(&token);
                        
                        let choice = ChatCompletionChoice {
                            index: 0,
                            message: ChatMessage {
                                role: "assistant".to_string(),
                                content: full_content.clone(),
                            },
                            finish_reason: None,
                        };
                        
                        let chunk = ChatCompletionResponse {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: created_time,
                            model: model_name.clone(),
                            choices: vec![choice],
                            usage: Usage {
                                prompt_tokens: formatted_prompt.len() / 4,
                                completion_tokens: full_content.len() / 4,
                                total_tokens: (formatted_prompt.len() + full_content.len()) / 4,
                            },
                        };
                        
                        chunks.push(format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()));
                    }
                }
                
                // 添加最后一个chunk，包含finish_reason
                let choice = ChatCompletionChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: full_content.clone(),
                    },
                    finish_reason: Some("stop".to_string()),
                };
                
                let final_chunk = ChatCompletionResponse {
                    id: request_id,
                    object: "chat.completion.chunk".to_string(),
                    created: created_time,
                    model: model_name,
                    choices: vec![choice],
                    usage: Usage {
                        prompt_tokens: formatted_prompt.len() / 4,
                        completion_tokens: full_content.len() / 4,
                        total_tokens: (formatted_prompt.len() + full_content.len()) / 4,
                    },
                };
                
                chunks.push(serde_json::to_string(&final_chunk).unwrap() + "\n");
                
                // 将所有chunk合并为一个响应
                let response_body = chunks.join("");
                
                let response = axum::response::Response::builder()
                    .header("Content-Type", "text/event-stream")
                    .header("Cache-Control", "no-cache")
                    .header("Connection", "keep-alive")
                    .body(axum::body::Body::from(response_body))
                    .unwrap();
                    
                return Ok(response);
            } else {
                // 非流式输出处理
                log_info!("开始执行GPU推理...");
                
                let response = sage::generation::generate(&*model, &tokenizer, &formatted_prompt, &options, &device);
                log_info!("推理完成，原始响应: {}", response);
                
                // 提取助手回复
                let reply = extract_assistant_reply(&response);
                log_info!("提取的助手回复: {}", reply);
                
                let duration_ms = start_time.elapsed().as_millis();
                log_info!("推理耗时: {}ms", duration_ms);
                (reply, duration_ms)
            }
        } else {
            let device = NdArrayDevice::Cpu;
            let lazy_model = state.lazy_model.lock().unwrap();
            let model = lazy_model.get_model(&device);
            let model = model.lock().unwrap();
            let tokenizer = state.tokenizer.lock().unwrap();
            
            if req.stream.unwrap_or(false) {
                let created_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let request_id = uuid::Uuid::new_v4().to_string();
                let model_name = req.model.unwrap_or_else(|| "sage-model".to_string());
                
                let mut generation_state = sage::generation::GenerationState::new(sage::generation::ModelType::Normal(&*model), &tokenizer, &formatted_prompt, &options, &device);
                
                let mut full_content = String::new();
                let mut chunks = Vec::new();
                
                while !generation_state.is_stopped() {
                    if let Some(token) = generation_state.next_token() {
                        full_content.push_str(&token);
                        
                        let choice = ChatCompletionChoice {
                            index: 0,
                            message: ChatMessage {
                                role: "assistant".to_string(),
                                content: full_content.clone(),
                            },
                            finish_reason: None,
                        };
                        
                        let chunk = ChatCompletionResponse {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: created_time,
                            model: model_name.clone(),
                            choices: vec![choice],
                            usage: Usage {
                                prompt_tokens: formatted_prompt.len() / 4,
                                completion_tokens: full_content.len() / 4,
                                total_tokens: (formatted_prompt.len() + full_content.len()) / 4,
                            },
                        };
                        
                        chunks.push(format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()));
                    }
                }
                
                let response_body = axum::body::Body::from(chunks.join(""));
                
                let response = axum::response::Response::builder()
                    .header("Content-Type", "text/event-stream")
                    .header("Cache-Control", "no-cache")
                    .header("Connection", "keep-alive")
                    .body(response_body)
                    .unwrap();
                    
                return Ok(response);
            } else {
                log_info!("开始执行CPU推理...");
                
                let response = sage::generation::generate(&*model, &tokenizer, &formatted_prompt, &options, &device);
                log_info!("推理完成，原始响应: {}", response);
                
                // 提取助手回复
                let reply = extract_assistant_reply(&response);
                log_info!("提取的助手回复: {}", reply);
                
                let duration_ms = start_time.elapsed().as_millis();
                log_info!("推理耗时: {}ms", duration_ms);
                (reply, duration_ms)
            }
        };

    // 构建OpenAI格式的响应
    let choice = ChatCompletionChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: reply.clone(),
        },
        finish_reason: Some("stop".to_string()),
    };

    let prompt_tokens = formatted_prompt.len() / 4; // 估算
    let completion_tokens = reply.len() / 4; // 估算
    
    // 记录性能指标
    let metrics = state.performance_monitor.record_inference(
        "/v1/chat/completions",
        start_time,
        prompt_tokens,
        completion_tokens,
    );
    
    log_info!("性能指标: 推理时间={:.2}ms, 速度={:.2} tokens/s", 
             metrics.inference_time_ms, metrics.tokens_per_second);

    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };

    let response = ChatCompletionResponse {
        id: uuid::Uuid::new_v4().to_string(),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model.unwrap_or_else(|| "sage-model".to_string()),
        choices: vec![choice],
        usage,
    };

    let response = axum::response::Response::builder()
        .header("Content-Type", "application/json")
        .body(axum::body::Body::from(serde_json::to_string(&response).unwrap()))
        .unwrap();
        
    Ok(response)
}

async fn batch_infer_handler(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<BatchChatCompletionRequest>,
) -> Result<Json<BatchChatCompletionResponse>, (StatusCode, Json<ErrorResponse>)> {
    log_info!("收到批量ChatCompletion请求，共{}个请求", req.requests.len());
    
    let start_time = std::time::Instant::now();
    
    // 验证请求数量
    if req.requests.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "bad_request".to_string(),
                message: "Batch requests cannot be empty".to_string(),
            }),
        ));
    }
    
    // 限制批量大小
    if req.requests.len() > 100 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "bad_request".to_string(),
                message: "Batch size exceeds maximum limit of 100".to_string(),
            }),
        ));
    }
    
    let mut responses = Vec::with_capacity(req.requests.len());
    
    if state._backend == "gpu" && state.lazy_model_gpu.lock().unwrap().is_some() {
        let device = <Wgpu as Backend>::Device::default();
        let lazy_model_gpu = state.lazy_model_gpu.lock().unwrap();
        let model = lazy_model_gpu.as_ref().unwrap().get_model(&device);
        let model = model.lock().unwrap();
        let tokenizer = state.tokenizer.lock().unwrap();
        
        for (i, request) in req.requests.iter().enumerate() {
            log_info!("处理批量请求 #{}/{} (GPU)", i + 1, req.requests.len());
            
            if request.messages.is_empty() {
                // 创建空响应
                let choice = ChatCompletionChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: "".to_string(),
                    },
                    finish_reason: Some("stop".to_string()),
                };
                
                let response = ChatCompletionResponse {
                    id: uuid::Uuid::new_v4().to_string(),
                    object: "chat.completion".to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: request.model.clone().unwrap_or_else(|| "sage-model".to_string()),
                    choices: vec![choice],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                };
                
                responses.push(response);
                continue;
            }
            
            let formatted_prompt = format_messages_to_prompt(&request.messages);
            
            let options = GenerateOptions {
                max_new_tokens: request.max_tokens.unwrap_or(50),
                temperature: request.temperature.unwrap_or(0.8),
                top_k: request.top_k.unwrap_or(10),
                top_p: request.top_p.unwrap_or(0.9),
                repetition_penalty: 1.1,
                punctuation_penalty: 1.3,
                presence_penalty: request.presence_penalty.unwrap_or(0.0),
                frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
                seed: request.seed,
                context_len: state.context_len,
                stop_on_user: true,
                stop_sequences: request.stop.clone().unwrap_or(Vec::new()),
            };
            
            let response_text = sage::generation::generate(&*model, &tokenizer, &formatted_prompt, &options, &device);
            let reply = extract_assistant_reply(&response_text);
            
            let choice = ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: reply.clone(),
                },
                finish_reason: Some("stop".to_string()),
            };
            
            let usage = Usage {
                prompt_tokens: formatted_prompt.len() / 4,
                completion_tokens: reply.len() / 4,
                total_tokens: (formatted_prompt.len() + reply.len()) / 4,
            };
            
            let response = ChatCompletionResponse {
                id: uuid::Uuid::new_v4().to_string(),
                object: "chat.completion".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                model: request.model.clone().unwrap_or_else(|| "sage-model".to_string()),
                choices: vec![choice],
                usage,
            };
            
            responses.push(response);
        }
    } else {
        let device = NdArrayDevice::Cpu;
        let lazy_model = state.lazy_model.lock().unwrap();
        let model = lazy_model.get_model(&device);
        let model = model.lock().unwrap();
        let tokenizer = state.tokenizer.lock().unwrap();
        
        for (i, request) in req.requests.iter().enumerate() {
            log_info!("处理批量请求 #{}/{} (CPU)", i + 1, req.requests.len());
            
            if request.messages.is_empty() {
                // 创建空响应
                let choice = ChatCompletionChoice {
                    index: 0,
                    message: ChatMessage {
                        role: "assistant".to_string(),
                        content: "".to_string(),
                    },
                    finish_reason: Some("stop".to_string()),
                };
                
                let response = ChatCompletionResponse {
                    id: uuid::Uuid::new_v4().to_string(),
                    object: "chat.completion".to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: request.model.clone().unwrap_or_else(|| "sage-model".to_string()),
                    choices: vec![choice],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                };
                
                responses.push(response);
                continue;
            }
            
            let formatted_prompt = format_messages_to_prompt(&request.messages);
            
            let options = GenerateOptions {
                max_new_tokens: request.max_tokens.unwrap_or(50),
                temperature: request.temperature.unwrap_or(0.8),
                top_k: request.top_k.unwrap_or(10),
                top_p: request.top_p.unwrap_or(0.9),
                repetition_penalty: 1.1,
                punctuation_penalty: 1.3,
                presence_penalty: request.presence_penalty.unwrap_or(0.0),
                frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
                seed: request.seed,
                context_len: state.context_len,
                stop_on_user: true,
                stop_sequences: request.stop.clone().unwrap_or(Vec::new()),
            };
            
            let response_text = sage::generation::generate(&*model, &tokenizer, &formatted_prompt, &options, &device);
            let reply = extract_assistant_reply(&response_text);
            
            let choice = ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: reply.clone(),
                },
                finish_reason: Some("stop".to_string()),
            };
            
            let usage = Usage {
                prompt_tokens: formatted_prompt.len() / 4,
                completion_tokens: reply.len() / 4,
                total_tokens: (formatted_prompt.len() + reply.len()) / 4,
            };
            
            let response = ChatCompletionResponse {
                id: uuid::Uuid::new_v4().to_string(),
                object: "chat.completion".to_string(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                model: request.model.clone().unwrap_or_else(|| "sage-model".to_string()),
                choices: vec![choice],
                usage,
            };
            
            responses.push(response);
        }
    }

    
    let total_duration_ms = start_time.elapsed().as_millis();
    
    // 记录批量推理性能指标
    let total_prompt_tokens: usize = responses.iter().map(|r| r.usage.prompt_tokens).sum();
    let total_completion_tokens: usize = responses.iter().map(|r| r.usage.completion_tokens).sum();
    
    let metrics = state.performance_monitor.record_inference(
        "/v1/batch-chat/completions",
        start_time,
        total_prompt_tokens,
        total_completion_tokens,
    );
    
    log_info!("批量推理完成，总耗时: {}ms, 平均速度={:.2} tokens/s", 
             total_duration_ms, metrics.tokens_per_second);
    
    Ok(Json(BatchChatCompletionResponse {
        responses,
        total_duration_ms: total_duration_ms as u64,
        request_count: req.requests.len(),
    }))
}

async fn async_infer_handler(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<AsyncChatCompletionRequest>,
) -> Result<Json<AsyncTaskResponse>, (StatusCode, Json<ErrorResponse>)> {
    log_info!("收到异步ChatCompletion请求: messages数量={}", req.messages.len());
    
    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "bad_request".to_string(),
                message: "Messages array cannot be empty".to_string(),
            }),
        ));
    }
    
    // 生成任务ID
    let task_id = uuid::Uuid::new_v4().to_string();
    
    // 创建任务
    let task = AsyncTask {
        task_id: task_id.clone(),
        status: TaskStatus::Pending,
        request: ChatCompletionRequest {
            model: req.model.clone(),
            messages: req.messages.clone(),
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            top_p: req.top_p,
            top_k: req.top_k,
            n: req.n,
            stop: req.stop.clone(),
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
            seed: req.seed,
            stream: req.stream,
        },
        result: None,
        error: None,
        created_at: Instant::now(),
        started_at: None,
        completed_at: None,
    };
    
    // 保存任务
    {
        let mut tasks = state.tasks.lock().unwrap();
        tasks.insert(task_id.clone(), task);
    }
    
    // 将任务加入队列
    {
        let mut queue = state.task_queue.lock().unwrap();
        queue.push_back(task_id.clone());
    }
    
    // 通知任务处理器
    state.task_sender.send(task_id.clone()).await.unwrap();
    
    log_info!("异步任务已创建，任务ID: {}", task_id);
    
    Ok(Json(AsyncTaskResponse {
        task_id,
        status: TaskStatus::Pending,
        result: None,
        error: None,
    }))
}

async fn task_status_handler(
    state: axum::extract::State<Arc<AppState>>,
    Path(task_id): Path<String>,
) -> Result<Json<TaskStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let tasks = state.tasks.lock().unwrap();
    
    match tasks.get(&task_id) {
        Some(task) => {
            Ok(Json(TaskStatusResponse {
                task_id: task.task_id.clone(),
                status: task.status.clone(),
                result: task.result.clone(),
                error: task.error.clone(),
                created_at: task.created_at.elapsed().as_millis() as u64,
                started_at: task.started_at.map(|t| t.elapsed().as_millis() as u64),
                completed_at: task.completed_at.map(|t| t.elapsed().as_millis() as u64),
            }))
        },
        None => {
            Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "not_found".to_string(),
                    message: "Task not found".to_string(),
                }),
            ))
        }
    }
}

async fn task_processor(state: Arc<AppState>, mut receiver: mpsc::Receiver<String>) {
    log_info!("后台任务处理器已启动");
    
    while let Some(task_id) = receiver.recv().await {
        log_info!("开始处理任务: {}", task_id);
        
        // 获取任务
        let task = {
            let tasks = state.tasks.lock().unwrap();
            tasks.get(&task_id).map(|t| AsyncTask {
                task_id: t.task_id.clone(),
                status: t.status.clone(),
                request: t.request.clone(),
                result: t.result.clone(),
                error: t.error.clone(),
                created_at: t.created_at,
                started_at: t.started_at,
                completed_at: t.completed_at,
            })
        };
        
        if let Some(mut task) = task {
            // 更新任务状态为运行中
            {
                let mut tasks = state.tasks.lock().unwrap();
                task.status = TaskStatus::Running;
                task.started_at = Some(Instant::now());
                tasks.insert(task_id.clone(), AsyncTask {
                    task_id: task.task_id.clone(),
                    status: task.status.clone(),
                    request: task.request.clone(),
                    result: task.result.clone(),
                    error: task.error.clone(),
                    created_at: task.created_at,
                    started_at: task.started_at,
                    completed_at: task.completed_at,
                });
            }
            
            let formatted_prompt = format_messages_to_prompt(&task.request.messages);
            
            let options = GenerateOptions {
                max_new_tokens: task.request.max_tokens.unwrap_or(50),
                temperature: task.request.temperature.unwrap_or(0.8),
                top_k: task.request.top_k.unwrap_or(10),
                top_p: task.request.top_p.unwrap_or(0.9),
                repetition_penalty: 1.1,
                punctuation_penalty: 1.3,
                presence_penalty: task.request.presence_penalty.unwrap_or(0.0),
                frequency_penalty: task.request.frequency_penalty.unwrap_or(0.0),
                seed: task.request.seed,
                context_len: state.context_len,
                stop_on_user: true,
                stop_sequences: task.request.stop.clone().unwrap_or(Vec::new()),
            };
            
            // 执行推理
            let reply = if state._backend == "gpu" && state.lazy_model_gpu.lock().unwrap().is_some() {
                let device = <Wgpu as Backend>::Device::default();
                let lazy_model_gpu = state.lazy_model_gpu.lock().unwrap();
                let model = lazy_model_gpu.as_ref().unwrap().get_model(&device);
                let model = model.lock().unwrap();
                let tokenizer = state.tokenizer.lock().unwrap();
                let response_text = sage::generation::generate(&*model, &tokenizer, &formatted_prompt, &options, &device);
                extract_assistant_reply(&response_text)
            } else {
                let device = NdArrayDevice::Cpu;
                let lazy_model = state.lazy_model.lock().unwrap();
                let model = lazy_model.get_model(&device);
                let model = model.lock().unwrap();
                let tokenizer = state.tokenizer.lock().unwrap();
                let response_text = sage::generation::generate(&*model, &tokenizer, &formatted_prompt, &options, &device);
                extract_assistant_reply(&response_text)
            };
            
            // 更新任务状态
            {
                let mut tasks = state.tasks.lock().unwrap();
                let completed_at = Instant::now();
                
                if reply.is_empty() {
                    // 推理失败
                    task.status = TaskStatus::Failed;
                    task.completed_at = Some(completed_at);
                    task.error = Some("Inference failed: empty response".to_string());
                } else {
                    // 推理成功
                    task.status = TaskStatus::Completed;
                    task.completed_at = Some(completed_at);
                    
                    let choice = ChatCompletionChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".to_string(),
                            content: reply.clone(),
                        },
                        finish_reason: Some("stop".to_string()),
                    };
                    
                    let usage = Usage {
                        prompt_tokens: formatted_prompt.len() / 4,
                        completion_tokens: reply.len() / 4,
                        total_tokens: (formatted_prompt.len() + reply.len()) / 4,
                    };
                    
                    let result = ChatCompletionResponse {
                        id: uuid::Uuid::new_v4().to_string(),
                        object: "chat.completion".to_string(),
                        created: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        model: task.request.model.clone().unwrap_or_else(|| "sage-model".to_string()),
                        choices: vec![choice],
                        usage,
                    };
                    
                    task.result = Some(result);
                }
                
                tasks.insert(task_id.clone(), AsyncTask {
                    task_id: task.task_id.clone(),
                    status: task.status.clone(),
                    request: task.request.clone(),
                    result: task.result.clone(),
                    error: task.error.clone(),
                    created_at: task.created_at,
                    started_at: task.started_at,
                    completed_at: task.completed_at,
                });
            }
            
            log_info!("任务处理完成: {}", task_id);
        }
    }
}

fn format_messages_to_prompt(messages: &[ChatMessage]) -> String {
    let mut estimated_len = 10;
    for msg in messages {
        estimated_len += msg.role.len() + msg.content.len() + 20;
    }
    
    let mut out = String::with_capacity(estimated_len);
    out.push('\u{0002}');
    out.push_str("<s>\n");
    
    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                out.push_str("<system>");
                out.push_str(&msg.content);
                out.push_str("</system>\n");
            },
            "user" => {
                out.push_str("<user>");
                out.push_str(&msg.content);
                out.push_str("</user>\n");
            },
            "assistant" => {
                out.push_str("<assistant>");
                out.push_str(&msg.content);
                out.push_str("</assistant>\n");
            },
            _ => {
                out.push_str("<user>");
                out.push_str(&msg.content);
                out.push_str("</user>\n");
            },
        }
    }
    
    out.push_str("<assistant>");
    out
}

async fn auth_middleware(
    state: axum::extract::State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    // 健康检查接口不需要认证
    if request.uri().path() == "/api/health" {
        let response = next.run(request).await;
        return Ok(response);
    }

    // 如果没有配置API密钥，则跳过认证
    if state.api_key.is_none() {
        let response = next.run(request).await;
        return Ok(response);
    }

    // 从请求头获取API密钥
    let auth_header = request.headers().get(header::AUTHORIZATION);
    
    let api_key = match auth_header {
        Some(header) => header.to_str().ok(),
        None => None,
    };

    // 验证API密钥
    if let Some(api_key) = api_key
        && let Some(key) = api_key.strip_prefix("Bearer ")
        && key == state.api_key.as_ref().unwrap()
    {
        let response = next.run(request).await;
        return Ok(response);
    }

    // 认证失败
    Err((
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "unauthorized".to_string(),
            message: "Invalid or missing API key".to_string(),
        }),
    ))
}

fn extract_assistant_reply(full: &str) -> String {
    let Some(idx) = full.rfind("<assistant>") else {
        return full.trim().to_string();
    };
    let start = idx + "<assistant>".len();
    let Some(end) = full[start..].find("</assistant>") else {
        return full[start..].trim().to_string();
    };
    full[start..start + end].trim().to_string()
}

async fn list_models_handler(
    state: axum::extract::State<Arc<AppState>>,
) -> axum::Json<ModelListResponse> {
    let models = state.models.lock().unwrap();
    let active_model = state.active_model.lock().unwrap();
    
    let models_list: Vec<ModelInfo> = models.values().cloned().collect();
    
    axum::Json(ModelListResponse {
        models: models_list,
        active_model: Some(active_model.clone()),
    })
}

async fn load_model_handler(
    state: axum::extract::State<Arc<AppState>>,
    axum::Json(req): axum::Json<ModelLoadRequest>,
) -> Result<axum::Json<ModelLoadResponse>, (StatusCode, axum::Json<ErrorResponse>)> {
    let model_id = req.model_id;
    let model_dir = req.model_dir;
    
    // 检查模型目录是否存在
    if !std::path::Path::new(&model_dir).exists() {
        return Err((
            StatusCode::NOT_FOUND,
            axum::Json(ErrorResponse {
                error: "not_found".to_string(),
                message: format!("Model directory not found: {}", model_dir),
            }),
        ));
    }
    
    // 检查配置文件是否存在
    let config_path = format!("{}/config.json", model_dir);
    if !std::path::Path::new(&config_path).exists() {
        return Err((
            StatusCode::BAD_REQUEST,
            axum::Json(ErrorResponse {
                error: "invalid_model".to_string(),
                message: format!("Config file not found: {}", config_path),
            }),
        ));
    }
    
    // 读取模型配置
    let config_str = fs::read_to_string(&config_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(ErrorResponse {
                error: "read_error".to_string(),
                message: format!("Failed to read config: {}", e),
            }),
        )
    })?;
    
    let config: TrainingConfig = serde_json::from_str(&config_str).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            axum::Json(ErrorResponse {
                error: "parse_error".to_string(),
                message: format!("Failed to parse config: {}", e),
            }),
        )
    })?;
    
    // 创建模型信息
    let model_info = ModelInfo {
        model_id: model_id.clone(),
        model_dir: model_dir.clone(),
        status: "loading".to_string(),
        size: format!("{}m", config.model.d_model / 1024 / 1024),
        backend: "cpu".to_string(),
        loaded_at: Instant::now().elapsed().as_millis() as u64,
    };
    
    // 添加到模型列表
    let mut models = state.models.lock().unwrap();
    models.insert(model_id.clone(), model_info);
    
    log_info!("Model {} loaded from {}", model_id, model_dir);
    
    Ok(axum::Json(ModelLoadResponse {
        model_id,
        status: "loaded".to_string(),
        message: "Model loaded successfully".to_string(),
    }))
}

async fn unload_model_handler(
    state: axum::extract::State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<axum::Json<ModelLoadResponse>, (StatusCode, axum::Json<ErrorResponse>)> {
    // 不允许卸载默认模型
    if model_id == "default" {
        return Err((
            StatusCode::BAD_REQUEST,
            axum::Json(ErrorResponse {
                error: "forbidden".to_string(),
                message: "Cannot unload default model".to_string(),
            }),
        ));
    }
    
    let mut models = state.models.lock().unwrap();
    
    if !models.contains_key(&model_id) {
        return Err((
            StatusCode::NOT_FOUND,
            axum::Json(ErrorResponse {
                error: "not_found".to_string(),
                message: format!("Model not found: {}", model_id),
            }),
        ));
    }
    
    // 检查是否是当前活动模型
    let active_model = state.active_model.lock().unwrap();
    if *active_model == model_id {
        return Err((
            StatusCode::BAD_REQUEST,
            axum::Json(ErrorResponse {
                error: "active_model".to_string(),
                message: "Cannot unload active model".to_string(),
            }),
        ));
    }
    
    models.remove(&model_id);
    
    log_info!("Model {} unloaded", model_id);
    
    Ok(axum::Json(ModelLoadResponse {
        model_id,
        status: "unloaded".to_string(),
        message: "Model unloaded successfully".to_string(),
    }))
}

async fn switch_model_handler(
    state: axum::extract::State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<axum::Json<ModelSwitchResponse>, (StatusCode, axum::Json<ErrorResponse>)> {
    let models = state.models.lock().unwrap();
    
    if !models.contains_key(&model_id) {
        return Err((
            StatusCode::NOT_FOUND,
            axum::Json(ErrorResponse {
                error: "not_found".to_string(),
                message: format!("Model not found: {}", model_id),
            }),
        ));
    }
    
    let mut active_model = state.active_model.lock().unwrap();
    *active_model = model_id.clone();
    
    log_info!("Switched active model to {}", model_id);
    
    Ok(axum::Json(ModelSwitchResponse {
        model_id,
        status: "switched".to_string(),
        message: "Model switched successfully".to_string(),
    }))
}

async fn reload_model_handler(
    state: axum::extract::State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<axum::Json<ModelLoadResponse>, (StatusCode, axum::Json<ErrorResponse>)> {
    let models = state.models.lock().unwrap();
    
    let model_info = models.get(&model_id).ok_or((
        StatusCode::NOT_FOUND,
        axum::Json(ErrorResponse {
            error: "not_found".to_string(),
            message: format!("Model not found: {}", model_id),
        }),
    ))?;
    
    let model_dir = model_info.model_dir.clone();
    
    // 检查配置文件是否存在
    let config_path = format!("{}/config.json", model_dir);
    if !std::path::Path::new(&config_path).exists() {
        return Err((
            StatusCode::BAD_REQUEST,
            axum::Json(ErrorResponse {
                error: "invalid_model".to_string(),
                message: format!("Config file not found: {}", config_path),
            }),
        ));
    }
    
    // 读取模型配置
    let config_str = fs::read_to_string(&config_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(ErrorResponse {
                error: "read_error".to_string(),
                message: format!("Failed to read config: {}", e),
            }),
        )
    })?;
    
    let config: TrainingConfig = serde_json::from_str(&config_str).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            axum::Json(ErrorResponse {
                error: "parse_error".to_string(),
                message: format!("Failed to parse config: {}", e),
            }),
        )
    })?;
    
    // 加载分词器
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);
    let tokenizer = Tokenizer::load(&tokenizer_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(ErrorResponse {
                error: "tokenizer_error".to_string(),
                message: format!("Failed to load tokenizer: {}", e),
            }),
        )
    })?;
    
    // 加载模型权重
    let model_path = format!("{}/model.mpk", model_dir);
    let best_model_path = format!("{}/best_model.mpk", model_dir);
    let final_model_path = if std::path::Path::new(&best_model_path).exists() {
        best_model_path
    } else {
        model_path
    };
    
    // 更新模型
    let mut state_tokenizer = state.tokenizer.lock().unwrap();
    *state_tokenizer = tokenizer;
    
    let mut state_config = state.config.lock().unwrap();
    *state_config = config.clone();
    
    let mut state_lazy_model = state.lazy_model.lock().unwrap();
    *state_lazy_model = LazyModel::new(config.model.clone(), final_model_path.clone());
    
    // 更新GPU模型（如果启用）
    if let Some(_) = state.lazy_model_gpu.lock().unwrap().as_ref() {
        let mut state_lazy_model_gpu = state.lazy_model_gpu.lock().unwrap();
        *state_lazy_model_gpu = Some(LazyModel::new(config.model.clone(), final_model_path));
    }
    
    log_info!("Model {} reloaded from {}", model_id, model_dir);
    
    Ok(axum::Json(ModelLoadResponse {
        model_id,
        status: "reloaded".to_string(),
        message: "Model reloaded successfully".to_string(),
    }))
}

#[derive(Debug, Deserialize)]
struct DownloadModelRequest {
    model_id: String,
    url: String,
}

async fn download_model_handler(
    state: axum::extract::State<Arc<AppState>>,
    axum::extract::Json(req): axum::extract::Json<DownloadModelRequest>,
) -> Result<axum::Json<ModelLoadResponse>, (StatusCode, axum::Json<ErrorResponse>)> {
    log_info!("开始下载模型: {}", req.model_id);
    
    match state.model_downloader.download_model(&req.model_id, &req.url).await {
        Ok(model_dir) => {
            log_info!("模型下载完成: {}", req.model_id);
            
            // 添加到模型列表
            let mut models = state.models.lock().unwrap();
            let model_info = ModelInfo {
                model_id: req.model_id.clone(),
                model_dir: model_dir.to_str().unwrap().to_string(),
                status: "loaded".to_string(),
                size: "unknown".to_string(),
                backend: "cpu".to_string(),
                loaded_at: std::time::Instant::now().elapsed().as_millis() as u64,
            };
            models.insert(req.model_id.clone(), model_info);
            
            Ok(axum::Json(ModelLoadResponse {
                model_id: req.model_id,
                status: "downloaded".to_string(),
                message: "Model downloaded successfully".to_string(),
            }))
        }
        Err(e) => {
            log_error!("模型下载失败: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(ErrorResponse {
                    error: "download_error".to_string(),
                    message: format!("Failed to download model: {}", e),
                }),
            ))
        }
    }
}


