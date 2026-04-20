#![recursion_limit = "1024"]

use burn::backend::{ndarray::{NdArray, NdArrayDevice}, wgpu::Wgpu};
use burn::prelude::Backend;

use clap::Parser;
use sage::{
    inference::{GenerateOptions, LazyModel},
    logger::init_logger_with_level,
    models::ModelConfig,
    performance::PerformanceMonitor,
    models::Tokenizer,
    TrainingConfig,
};
use burn::optim::AdamConfig;
use sage::{log_info, log_error};
#[cfg(feature = "web")]
use sage::model_download::ModelDownloader;
use axum::{
    extract::{Json, Request},
    http::{header, StatusCode},
    middleware::{self, Next},
    routing::{get, post},
    Router,
};

use serde::{Deserialize, Serialize};
use std::{
    fs,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::{  
    net::TcpListener,
    sync::Semaphore,
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

    #[arg(long, default_value = "info")]
    log_level: String,

    #[arg(long, default_value_t = 2)]
    max_concurrent: usize,
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
struct GenerateRequest {
    prompt: String,
    max_length: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
}

#[derive(Serialize)]
struct GenerateResponse {
    prompt: String,
    text: String,
}

struct AppState {
    tokenizer: Arc<Tokenizer>,
    lazy_model: Arc<LazyModel<NdArray>>,
    lazy_model_gpu: Arc<Option<LazyModel<Wgpu>>>,
    config: Arc<Mutex<TrainingConfig>>,
    context_len: usize,
    api_key: Option<String>,
    _backend: String,
    performance_monitor: PerformanceMonitor,
    
    // 并发控制
    concurrency_semaphore: Arc<Semaphore>,
    max_concurrent: usize,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    
    // 初始化日志系统
    init_logger_with_level(Some(&args.log_level));
    
    log_info!("正在启动API服务器...");
    log_info!("模型目录: {}", args.model_dir);
    log_info!("端口: {}", args.port);

    // 设置 cubecl autotune 级别为 balanced，提高 GPU 利用率
    // minimal: 最快启动但 GPU 利用率低
    // balanced: 较好的启动速度和 GPU 利用率平衡
    // extensive/full: 最高 GPU 利用率但首次启动慢
    unsafe {
        std::env::set_var("CUBECL_AUTOTUNE_LEVEL", "balanced");
    }

    // 加载模型配置
    // 为了简化启动，使用默认的模型配置
    let model_config = ModelConfig {
        d_model: 512,
        n_heads: 4,
        n_layers: 2,
        vocab_size: 30522,
        d_ff: 2048,
        dropout: 0.1,
        max_seq_len: 512,
        quantized: false,
        multimodal: None,
        use_moe: false,
        num_experts: 0,
        top_k_experts: 0,
    };
    
    let optimizer_config = AdamConfig::new();
    let config = TrainingConfig::create(model_config.clone(), optimizer_config);

    // 计算正确的 context_len
    let requested_context_len = if args.context_len == 0 {
        config.model.max_seq_len
    } else {
        args.context_len
    };
    let context_len = requested_context_len.min(config.model.max_seq_len);
    if requested_context_len > config.model.max_seq_len {
        log_info!("context_len {} 超过模型 max_seq_len {}，已自动截断。", requested_context_len, config.model.max_seq_len);
    }

    // 加载分词器
    let tokenizer_path = format!("{}/tokenizer.json", args.model_dir);
    let tokenizer = Tokenizer::load(&tokenizer_path).expect("Failed to load tokenizer");

    // 根据后端类型加载模型
    let primary_model_path = format!("{}/model.mpk", args.model_dir);
    let best_model_path = format!("{}/best_model.mpk", args.model_dir);
    
    let model_path = if args.use_best {
        if fs::metadata(&best_model_path).is_ok() {
            best_model_path
        } else {
            primary_model_path
        }
    } else {
        if fs::metadata(&primary_model_path).is_ok() {
            primary_model_path
        } else if fs::metadata(&best_model_path).is_ok() {
            log_info!("model.mpk 不存在，自动使用 best_model.mpk");
            best_model_path
        } else {
            primary_model_path
        }
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
    let api_key = std::env::var("SAGE_API_KEY").ok();

    // 初始化并发控制信号量
    let max_concurrent = args.max_concurrent.max(1);
    let concurrency_semaphore = Arc::new(Semaphore::new(max_concurrent));
    log_info!("初始化并发控制：最大并发数 = {}", max_concurrent);

    let state = Arc::new(AppState {
        tokenizer: Arc::new(tokenizer),
        lazy_model: Arc::new(lazy_model),
        lazy_model_gpu: Arc::new(lazy_model_gpu),
        config: Arc::new(Mutex::new(config)),
        context_len,
        api_key,
        _backend: backend,
        performance_monitor: PerformanceMonitor::new(),
        concurrency_semaphore: concurrency_semaphore.clone(),
        max_concurrent,
    });

    let app = Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/model-info", get(model_info_handler))
        .route("/api/performance", get(performance_handler))
        .route("/api/generate", post(generate_handler))
        .route("/v1/chat/completions", post(infer_handler));
    
    let final_app = app
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .with_state(state.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    log_info!("API服务器启动在 http://{}", addr);
    
    let listener = TcpListener::bind(addr).await.unwrap();
    
    axum::serve(listener, final_app)
        .await
        .unwrap();
}

async fn health_handler() -> StatusCode {
    StatusCode::OK
}

async fn model_info_handler(state: axum::extract::State<Arc<AppState>>) -> Json<serde_json::Value> {
    let config = state.config.lock().unwrap();
    let tokenizer = &*state.tokenizer;
    
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

    // 尝试获取并发控制信号量
    let semaphore = state.concurrency_semaphore.clone();
    let permit = semaphore.try_acquire()
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "service_unavailable".to_string(),
                    message: format!("Server is at capacity (max concurrent: {})", state.max_concurrent),
                }),
            )
        })?;

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
    let (reply, _duration_ms) = if state._backend == "gpu" && state.lazy_model_gpu.is_some() {
        let device = <Wgpu as Backend>::Device::default();
        let lazy_model_gpu = &*state.lazy_model_gpu;
        
        if lazy_model_gpu.is_none() {
            drop(permit);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "internal_server_error".to_string(),
                    message: "GPU model not loaded".to_string(),
                }),
            ));
        }
        
        let model = lazy_model_gpu.as_ref().unwrap().get_model(&device);
        let model = model.lock().unwrap();
        let tokenizer = &*state.tokenizer;
        
        if req.stream.unwrap_or(false) {
            // 流式输出特殊处理
            drop(permit);
            return Ok(perform_gpu_streaming_inference(&*model, &tokenizer, &formatted_prompt, &options, &device, req.model.clone()));
        } else {
            log_info!("开始执行GPU推理...");
            let (reply, duration_ms) = perform_gpu_non_streaming_inference(&*model, &tokenizer, &formatted_prompt, &options, &device, start_time);
            log_info!("推理完成，提取的助手回复: {}", reply);
            log_info!("推理耗时: {}ms", duration_ms);
            (reply, duration_ms)
        }
    } else {
        let device = NdArrayDevice::Cpu;
        let lazy_model = &*state.lazy_model;
        let model = lazy_model.get_model(&device);
        let model = model.lock().unwrap();
        let tokenizer = &*state.tokenizer;
        
        if req.stream.unwrap_or(false) {
            // 流式输出特殊处理
            drop(permit);
            return Ok(perform_cpu_streaming_inference(&*model, &tokenizer, &formatted_prompt, &options, &device, req.model.clone()));
        } else {
            log_info!("开始执行CPU推理...");
            let (reply, duration_ms) = perform_cpu_non_streaming_inference(&*model, &tokenizer, &formatted_prompt, &options, &device, start_time);
            log_info!("推理完成，提取的助手回复: {}", reply);
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
        1, // batch_size
        formatted_prompt.len(), // sequence_length
        0, // model_parameters (简化实现)
    );
    
    log_info!("性能指标: 推理时间={:.2}ms, 速度={:.2} tokens/s", 
             metrics.inference_time_ms, metrics.tokens_per_second);

    // 释放信号量
    drop(permit);

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

async fn perform_inference_inner(
    state: Arc<AppState>,
    formatted_prompt: String,
    options: GenerateOptions,
    stream: bool,
    _model: Option<String>,
) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    let start_time = std::time::Instant::now();
    
    // 尝试获取并发控制信号量
    let semaphore = state.concurrency_semaphore.clone();
    let permit = semaphore.try_acquire()
        .map_err(|_| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: "service_unavailable".to_string(),
                    message: format!("Server is at capacity (max concurrent: {})", state.max_concurrent),
                }),
            )
        })?;
    
    if stream {
        drop(permit);
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "bad_request".to_string(),
                message: "Stream is not supported for /api/generate".to_string(),
            }),
        ));
    }
    
    let result = if state._backend == "gpu" && state.lazy_model_gpu.is_some() {
        let device = <Wgpu as Backend>::Device::default();
        let lazy_model_gpu = &*state.lazy_model_gpu;
        
        if lazy_model_gpu.is_none() {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "internal_server_error".to_string(),
                    message: "GPU model not loaded".to_string(),
                }),
            ))
        } else {
            let model_ref = lazy_model_gpu.as_ref().unwrap().get_model(&device);
            match model_ref.lock() {
                Ok(model_guard) => {
                    let tokenizer = &*state.tokenizer;
                    log_info!("开始执行GPU推理...");
                    let (reply, duration_ms) = perform_gpu_non_streaming_inference(
                        &*model_guard, 
                        &tokenizer, 
                        &formatted_prompt, 
                        &options, 
                        &device, 
                        start_time
                    );
                    log_info!("推理完成，提取的助手回复: {}", reply);
                    log_info!("推理耗时: {}ms", duration_ms);
                    Ok(reply)
                },
                Err(_e) => {
                    Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: "internal_server_error".to_string(),
                            message: "Model lock poisoned".to_string(),
                        }),
                    ))
                }
            }
        }
    } else {
        let device = NdArrayDevice::Cpu;
        let lazy_model = &*state.lazy_model;
        let model_ref = lazy_model.get_model(&device);
        match model_ref.lock() {
            Ok(model_guard) => {
                let tokenizer = &*state.tokenizer;
                log_info!("开始执行CPU推理...");
                let (reply, duration_ms) = perform_cpu_non_streaming_inference(
                    &*model_guard, 
                    &tokenizer, 
                    &formatted_prompt, 
                    &options, 
                    &device, 
                    start_time
                );
                log_info!("推理完成，提取的助手回复: {}", reply);
                log_info!("推理耗时: {}ms", duration_ms);
                Ok(reply)
            },
            Err(_e) => {
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "internal_server_error".to_string(),
                        message: "Model lock poisoned".to_string(),
                    }),
                ))
            }
        }
    };
    
    // 释放信号量
    drop(permit);
    
    result
}

async fn generate_handler(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    log_info!("收到Generate请求: prompt长度={}", req.prompt.len());

    let formatted_prompt = format!("<user>\n{}\n</user>\n<assistant>", req.prompt);
    log_info!("格式化后的提示: {}", formatted_prompt);

    let options = GenerateOptions {
        max_new_tokens: req.max_length.unwrap_or(50),
        temperature: req.temperature.unwrap_or(0.8),
        top_k: req.top_k.unwrap_or(10),
        top_p: req.top_p.unwrap_or(0.9),
        repetition_penalty: 1.1,
        punctuation_penalty: 1.3,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: None,
        context_len: state.context_len,
        stop_on_user: true,
        stop_sequences: Vec::new(),
    };

    let reply = perform_inference_inner(state.0.clone(), formatted_prompt, options, false, None).await?;

    Ok(Json(GenerateResponse { prompt: req.prompt, text: reply }))
}

/// 为 WGPU 后端执行流式推理
fn perform_gpu_streaming_inference(
    model: &sage::core::Model<Wgpu>,
    tokenizer: &sage::core::Tokenizer,
    formatted_prompt: &str,
    options: &sage::inference::GenerateOptions,
    device: &<Wgpu as Backend>::Device,
    req_model: Option<String>,
) -> axum::response::Response {
    let created_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let request_id = uuid::Uuid::new_v4().to_string();
    let model_name = req_model.unwrap_or_else(|| "sage-model".to_string());
    
    let mut generation_state = sage::inference::GenerationState::new(
        sage::inference::ModelType::Normal(model),
        tokenizer,
        formatted_prompt,
        options,
        device,
    );
    
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
    
    let final_choice = ChatCompletionChoice {
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
        choices: vec![final_choice],
        usage: Usage {
            prompt_tokens: formatted_prompt.len() / 4,
            completion_tokens: full_content.len() / 4,
            total_tokens: (formatted_prompt.len() + full_content.len()) / 4,
        },
    };
    
    chunks.push(serde_json::to_string(&final_chunk).unwrap() + "\n");
    
    let response_body = chunks.join("");
    
    axum::response::Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(axum::body::Body::from(response_body))
        .unwrap()
}

/// 为 NdArray 后端执行流式推理
fn perform_cpu_streaming_inference(
    model: &sage::core::Model<NdArray>,
    tokenizer: &sage::core::Tokenizer,
    formatted_prompt: &str,
    options: &sage::inference::GenerateOptions,
    device: &NdArrayDevice,
    req_model: Option<String>,
) -> axum::response::Response {
    let created_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let request_id = uuid::Uuid::new_v4().to_string();
    let model_name = req_model.unwrap_or_else(|| "sage-model".to_string());
    
    let mut generation_state = sage::inference::GenerationState::new(
        sage::inference::ModelType::Normal(model),
        tokenizer,
        formatted_prompt,
        options,
        device,
    );
    
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
    
    let final_choice = ChatCompletionChoice {
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
        choices: vec![final_choice],
        usage: Usage {
            prompt_tokens: formatted_prompt.len() / 4,
            completion_tokens: full_content.len() / 4,
            total_tokens: (formatted_prompt.len() + full_content.len()) / 4,
        },
    };
    
    chunks.push(serde_json::to_string(&final_chunk).unwrap() + "\n");
    
    let response_body = chunks.join("");
    
    axum::response::Response::builder()
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(axum::body::Body::from(response_body))
        .unwrap()
}

/// 为 WGPU 后端执行非流式推理
fn perform_gpu_non_streaming_inference(
    model: &sage::core::Model<Wgpu>,
    tokenizer: &sage::core::Tokenizer,
    formatted_prompt: &str,
    options: &sage::inference::GenerateOptions,
    device: &<Wgpu as Backend>::Device,
    start_time: std::time::Instant,
) -> (String, u128) {
    let response = sage::inference::generate(model, tokenizer, formatted_prompt, options, device);
    let reply = extract_assistant_reply(&response);
    let duration_ms = start_time.elapsed().as_millis();
    (reply, duration_ms)
}

/// 为 NdArray 后端执行非流式推理
fn perform_cpu_non_streaming_inference(
    model: &sage::core::Model<NdArray>,
    tokenizer: &sage::core::Tokenizer,
    formatted_prompt: &str,
    options: &sage::inference::GenerateOptions,
    device: &NdArrayDevice,
    start_time: std::time::Instant,
) -> (String, u128) {
    let response = sage::inference::generate(model, tokenizer, formatted_prompt, options, device);
    let reply = extract_assistant_reply(&response);
    let duration_ms = start_time.elapsed().as_millis();
    (reply, duration_ms)
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

