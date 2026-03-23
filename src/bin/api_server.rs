use burn::backend::{ndarray::{NdArray, NdArrayDevice}, wgpu::Wgpu};
use burn::prelude::Backend;
use burn::module::Module;
use clap::Parser;
use sage::{
    generation::{GenerateOptions, generate},
    tokenizer::Tokenizer,
    training::TrainingConfig,
};
use axum::{
    extract::{Json, Path, Request},
    http::{header, StatusCode},
    middleware::{self, Next},
    routing::{get, post},
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
use uuid;

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

enum ModelBackend {
    Cpu(Arc<Mutex<sage::model::Model<NdArray>>>),
    Gpu(Arc<Mutex<sage::model::Model<Wgpu>>>),
}

struct AppState {
    tokenizer: Tokenizer,
    model: ModelBackend,
    config: TrainingConfig,
    context_len: usize,
    api_key: Option<String>,
    tasks: Arc<Mutex<HashMap<String, AsyncTask>>>,
    task_queue: Arc<Mutex<VecDeque<String>>>,
    task_sender: mpsc::Sender<String>,
    backend: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    
    println!("正在启动API服务器...");
    println!("模型目录: {}", args.model_dir);
    println!("端口: {}", args.port);

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

    let (model, backend) = if args.backend == "gpu" {
        println!("使用GPU后端进行推理...");
        // GPU后端优化配置
        unsafe {
            std::env::set_var("WGPU_POWER_PREFERENCE", "HighPerformance");
        }
        let device = <Wgpu as Backend>::Device::default();
        let mut model = config.model
            .init::<Wgpu>(&device)
            .load_file(&model_path, &burn::record::CompactRecorder::new(), &device)
            .expect("Failed to load model");
        
        if args.quantize {
            println!("正在进行模型量化...");
            model = model.quantize();
            println!("量化完成");
        }
        
        (ModelBackend::Gpu(Arc::new(Mutex::new(model))), "gpu".to_string())
    } else {
        println!("使用CPU后端进行推理...");
        let device = NdArrayDevice::Cpu;
        let mut model = config.model
            .init::<NdArray>(&device)
            .load_file(&model_path, &burn::record::CompactRecorder::new(), &device)
            .expect("Failed to load model");
        
        if args.quantize {
            println!("正在进行模型量化...");
            model = model.quantize();
            println!("量化完成");
        }
        
        (ModelBackend::Cpu(Arc::new(Mutex::new(model))), "cpu".to_string())
    };

    // 从环境变量加载API密钥
    let api_key = env::var("SAGE_API_KEY").ok();

    // 创建任务队列和通信通道
    let (task_sender, task_receiver) = mpsc::channel(100);
    let tasks = Arc::new(Mutex::new(HashMap::new()));
    let task_queue = Arc::new(Mutex::new(VecDeque::new()));

    let state = Arc::new(AppState {
        tokenizer,
        model,
        config,
        context_len: args.context_len,
        api_key,
        tasks: tasks.clone(),
        task_queue: task_queue.clone(),
        task_sender,
        backend,
    });

    // 启动后台任务处理器
    let state_clone = state.clone();
    task::spawn(async move {
        task_processor(state_clone, task_receiver).await;
    });

    let app = Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/model-info", get(model_info_handler))
        .route("/v1/chat/completions", post(infer_handler))
        .route("/v1/batch-chat/completions", post(batch_infer_handler))
        .route("/v1/async-chat/completions", post(async_infer_handler))
        .route("/api/task/:task_id", get(task_status_handler))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .with_state(state.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    println!("API服务器启动在 http://{}", addr);
    
    let listener = TcpListener::bind(addr).await.unwrap();
    
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}

async fn health_handler() -> StatusCode {
    StatusCode::OK
}

async fn model_info_handler(state: axum::extract::State<Arc<AppState>>) -> Json<serde_json::Value> {
    let config = &state.config;
    let tokenizer = &state.tokenizer;
    
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

async fn infer_handler(
    state: axum::extract::State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    println!("收到ChatCompletion请求: messages数量={}, stream={:?}", req.messages.len(), req.stream);
    
    let start_time = std::time::Instant::now();

    if req.messages.is_empty() {
        println!("错误: messages数组为空");
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
    println!("格式化后的提示: {}", formatted_prompt);

    let options = GenerateOptions {
        max_new_tokens: req.max_tokens.unwrap_or(50),
        temperature: req.temperature.unwrap_or(0.8),
        top_k: req.top_k.unwrap_or(10),
        top_p: req.top_p.unwrap_or(0.9),
        repetition_penalty: 1.1,
        punctuation_penalty: 1.3,
        seed: req.seed,
        context_len: state.context_len,
        stop_on_user: true,
        stop_sequences: req.stop.unwrap_or(Vec::new()),
    };

    // 根据后端类型选择设备
    let (reply, duration_ms) = match &state.model {
        ModelBackend::Cpu(model) => {
            let device = NdArrayDevice::Cpu;
            let model = model.lock().unwrap();
            
            if req.stream.unwrap_or(false) {
                // 流式输出处理
                let created_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let request_id = uuid::Uuid::new_v4().to_string();
                let model_name = req.model.unwrap_or_else(|| "sage-model".to_string());
                
                let mut generation_state = sage::generation::GenerationState::new(&*model, &state.tokenizer, &formatted_prompt, &options, &device);
                
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
                        
                        chunks.push(serde_json::to_string(&chunk).unwrap() + "\n");
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
                println!("开始执行CPU推理...");
                let response = sage::generation::generate(&*model, &state.tokenizer, &formatted_prompt, &options, &device);
                println!("推理完成，原始响应: {}", response);
                
                // 提取助手回复
                let reply = extract_assistant_reply(&response);
                println!("提取的助手回复: {}", reply);
                
                let duration_ms = start_time.elapsed().as_millis();
                println!("推理耗时: {}ms", duration_ms);
                (reply, duration_ms)
            }
        }
        ModelBackend::Gpu(model) => {
            let device = <Wgpu as Backend>::Device::default();
            let model = model.lock().unwrap();
            
            if req.stream.unwrap_or(false) {
                // 流式输出处理
                let created_time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let request_id = uuid::Uuid::new_v4().to_string();
                let model_name = req.model.unwrap_or_else(|| "sage-model".to_string());
                
                let mut generation_state = sage::generation::GenerationState::new(&*model, &state.tokenizer, &formatted_prompt, &options, &device);
                
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
                        
                        chunks.push(serde_json::to_string(&chunk).unwrap() + "\n");
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
                println!("开始执行GPU推理...");
                let response = sage::generation::generate(&*model, &state.tokenizer, &formatted_prompt, &options, &device);
                println!("推理完成，原始响应: {}", response);
                
                // 提取助手回复
                let reply = extract_assistant_reply(&response);
                println!("提取的助手回复: {}", reply);
                
                let duration_ms = start_time.elapsed().as_millis();
                println!("推理耗时: {}ms", duration_ms);
                (reply, duration_ms)
            }
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

    let usage = Usage {
        prompt_tokens: formatted_prompt.len() / 4, // 估算
        completion_tokens: reply.len() / 4, // 估算
        total_tokens: (formatted_prompt.len() + reply.len()) / 4, // 估算
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
    println!("收到批量ChatCompletion请求，共{}个请求", req.requests.len());
    
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
    
    match &state.model {
        ModelBackend::Cpu(model) => {
            let device = NdArrayDevice::Cpu;
            let model = model.lock().unwrap();
            
            for (i, request) in req.requests.iter().enumerate() {
                println!("处理批量请求 #{}/{} (CPU)", i + 1, req.requests.len());
                
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
                    seed: request.seed,
                    context_len: state.context_len,
                    stop_on_user: true,
                    stop_sequences: request.stop.clone().unwrap_or(Vec::new()),
                };
                
                let response_text = generate(&*model, &state.tokenizer, &formatted_prompt, &options, &device);
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
        ModelBackend::Gpu(model) => {
            let device = <Wgpu as Backend>::Device::default();
            let model = model.lock().unwrap();
            
            for (i, request) in req.requests.iter().enumerate() {
                println!("处理批量请求 #{}/{} (GPU)", i + 1, req.requests.len());
                
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
                    seed: request.seed,
                    context_len: state.context_len,
                    stop_on_user: true,
                    stop_sequences: request.stop.clone().unwrap_or(Vec::new()),
                };
                
                let response_text = generate(&*model, &state.tokenizer, &formatted_prompt, &options, &device);
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
    }
    
    let total_duration_ms = start_time.elapsed().as_millis();
    println!("批量推理完成，总耗时: {}ms", total_duration_ms);
    
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
    println!("收到异步ChatCompletion请求: messages数量={}", req.messages.len());
    
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
    
    println!("异步任务已创建，任务ID: {}", task_id);
    
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
    println!("后台任务处理器已启动");
    
    while let Some(task_id) = receiver.recv().await {
        println!("开始处理任务: {}", task_id);
        
        // 获取任务
        let task = {
            let tasks = state.tasks.lock().unwrap();
            if let Some(t) = tasks.get(&task_id) {
                Some(AsyncTask {
                    task_id: t.task_id.clone(),
                    status: t.status.clone(),
                    request: t.request.clone(),
                    result: t.result.clone(),
                    error: t.error.clone(),
                    created_at: t.created_at,
                    started_at: t.started_at,
                    completed_at: t.completed_at,
                })
            } else {
                None
            }
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
                seed: task.request.seed,
                context_len: state.context_len,
                stop_on_user: true,
                stop_sequences: task.request.stop.clone().unwrap_or(Vec::new()),
            };
            
            // 执行推理
            let reply = match &state.model {
                ModelBackend::Cpu(model) => {
                    let device = NdArrayDevice::Cpu;
                    let model = model.lock().unwrap();
                    let response_text = generate(&*model, &state.tokenizer, &formatted_prompt, &options, &device);
                    extract_assistant_reply(&response_text)
                }
                ModelBackend::Gpu(model) => {
                    let device = <Wgpu as Backend>::Device::default();
                    let model = model.lock().unwrap();
                    let response_text = generate(&*model, &state.tokenizer, &formatted_prompt, &options, &device);
                    extract_assistant_reply(&response_text)
                }
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
            
            println!("任务处理完成: {}", task_id);
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
    if let Some(api_key) = api_key {
        if api_key.starts_with("Bearer ") {
            let key = &api_key[7..];
            if key == state.api_key.as_ref().unwrap() {
                let response = next.run(request).await;
                return Ok(response);
            }
        }
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
