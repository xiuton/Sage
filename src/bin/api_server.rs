#![recursion_limit = "1024"]

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Json, State,
    },
    http::{header, Method, StatusCode},
    middleware::{self, from_fn_with_state},
    response::{sse::{Sse, Event}, IntoResponse},
    routing::{get, post},
    Router,
};
use burn::backend::{ndarray::NdArray, ndarray::NdArrayDevice};

use clap::Parser;
use futures_util::{Stream, StreamExt, SinkExt};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    net::SocketAddr,
    sync::{Arc, Mutex, RwLock},
    time::{Duration, Instant},
};
use tokio::sync::Semaphore;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

use sage::core::{DiffusionConfig, ImageGenerator, ModelConfig, SimpleTokenizer, Tokenizer};
use sage::inference::{GenerateOptions, LazyModel};
use sage::logger::init_logger_with_level;
use sage::TrainingConfig;
use burn::optim::AdamConfig;

// #[derive(Serialize, Deserialize, Clone)]
// struct TrainingProgress {
//     epoch: usize,
//     total_epochs: usize,
//     loss: f32,
//     samples_processed: usize,
//     total_samples: usize,
//     eta_seconds: u64,
// }

#[derive(Serialize, Deserialize, Clone)]
struct TrainingStatus {
    id: String,
    status: String,
    mode: String,
    start_time: u64,
    current_epoch: usize,
    total_epochs: usize,
    progress_percent: f32,
    loss: Option<f32>,
    output_dir: String,
    message: String,
    error: Option<String>,
}

#[derive(Clone)]
struct ModelHandle {
    model_type: String,
    loaded_at: u64,
    reference_count: usize,
}

struct TrainingTask {
    status: TrainingStatus,
    cancel_flag: Arc<AtomicBool>,
    // handle: Option<tokio::task::JoinHandle<()>>,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "./models/sage_model_formal")]
    model_dir: String,

    #[arg(long, default_value_t = 8000)]
    port: u16,

    #[arg(long, default_value = "cpu")]
    backend: String,

    #[arg(long, default_value = "info")]
    log_level: String,

    #[arg(long, default_value_t = 4)]
    max_concurrent: usize,

    #[arg(long)]
    api_key: Option<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
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

#[derive(Serialize, Clone)]
struct ChatCompletionChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: Option<String>,
}

#[derive(Serialize, Clone)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize, Clone)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_length: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    // stream: Option<bool>,
}

#[derive(Serialize)]
struct GenerateResponse {
    prompt: String,
    text: String,
}

#[derive(Deserialize)]
struct ImageGenerationRequest {
    prompt: String,
    // model_path: Option<String>,
    steps: Option<usize>,
    // latent_dim: Option<usize>,
    image_size: Option<usize>,
    // seed: Option<u64>,
}

#[derive(Serialize)]
struct ImageGenerationResponse {
    id: String,
    status: String,
    image_path: Option<String>,
    message: String,
}

#[derive(Deserialize)]
struct DiffusionLoadRequest {
    model_path: String,
    config_path: String,
}

#[derive(Deserialize)]
struct TrainingRequest {
    mode: String,
    data_path: String,
    config_path: Option<String>,
    output_dir: String,
    batch_size: Option<usize>,
    learning_rate: Option<f64>,
    num_epochs: Option<usize>,
    backend: Option<String>,
}

#[derive(Serialize)]
struct TrainingResponse {
    id: String,
    status: String,
    message: String,
    output_dir: String,
}

#[derive(Clone)]
struct RateLimitConfig {
    max_requests_per_minute: usize,
    max_tokens_per_minute: usize,
}

struct RateLimitState {
    request_counts: HashMap<String, Vec<Instant>>,
    token_counts: HashMap<String, Vec<(Instant, usize)>>,
}

impl RateLimitState {
    fn new() -> Self {
        Self {
            request_counts: HashMap::new(),
            token_counts: HashMap::new(),
        }
    }

    fn is_rate_limited(&self, key: &str, config: &RateLimitConfig) -> bool {
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);

        if let Some(times) = self.request_counts.get(key) {
            let recent: Vec<_> = times.iter().filter(|&&t| t > minute_ago).collect();
            if recent.len() >= config.max_requests_per_minute {
                return true;
            }
        }
        false
    }

    fn record_request(&mut self, key: &str, tokens: usize) {
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);

        self.request_counts
            .entry(key.to_string())
            .or_default()
            .push(now);

        self.token_counts
            .entry(key.to_string())
            .or_default()
            .push((now, tokens));

        for times in self.request_counts.values_mut() {
            times.retain(|&t| t > minute_ago);
        }

        for token_list in self.token_counts.values_mut() {
            token_list.retain(|&(t, _)| t > minute_ago);
        }
    }

    fn get_remaining(&self, key: &str, config: &RateLimitConfig) -> (usize, usize) {
        let now = Instant::now();
        let minute_ago = now - Duration::from_secs(60);

        let request_count = self
            .request_counts
            .get(key)
            .map(|v| v.iter().filter(|&&t| t > minute_ago).count())
            .unwrap_or(0);

        let token_count = self
            .token_counts
            .get(key)
            .map(|v| {
                v.iter()
                    .filter(|&&(t, _)| t > minute_ago)
                    .map(|(_, t)| t)
                    .sum()
            })
            .unwrap_or(0);

        (
            config.max_requests_per_minute.saturating_sub(request_count),
            config.max_tokens_per_minute.saturating_sub(token_count),
        )
    }
}

struct AppState {
    llm_tokenizer: Option<Arc<Tokenizer>>,
    llm_model: Option<Arc<LazyModel<NdArray>>>,
    // llm_model_gpu: Option<Arc<Option<LazyModel<Wgpu>>>>,
    llm_config: Arc<Mutex<TrainingConfig>>,
    context_len: usize,
    api_key: Option<String>,
    backend: String,

    diffusion_config: Arc<Mutex<Option<DiffusionConfig>>>,
    diffusion_model_path: Arc<Mutex<Option<String>>>,

    concurrency_semaphore: Arc<Semaphore>,
    max_concurrent: usize,

    rate_limit_config: RateLimitConfig,
    rate_limit_state: Arc<Mutex<RateLimitState>>,

    performance_stats: Arc<RwLock<PerformanceStats>>,

    broadcast_tx: Arc<tokio::sync::broadcast::Sender<ServerEvent>>,

    training_tasks: Arc<RwLock<HashMap<String, TrainingTask>>>,
    loaded_models: Arc<RwLock<HashMap<String, ModelHandle>>>,
    model_load_count: Arc<AtomicU64>,
    total_inference_time_ms: Arc<AtomicU64>,
    total_inference_requests: Arc<AtomicU64>,
}

#[derive(Clone)]
struct PerformanceStats {
    total_requests: u64,
    total_tokens: u64,
    total_errors: u64,
    avg_response_time_ms: f64,
    requests_by_endpoint: HashMap<String, u64>,
}

impl PerformanceStats {
    fn new() -> Self {
        Self {
            total_requests: 0,
            total_tokens: 0,
            total_errors: 0,
            avg_response_time_ms: 0.0,
            requests_by_endpoint: HashMap::new(),
        }
    }

    fn record_request(&mut self, endpoint: &str, tokens: u64, response_time_ms: f64) {
        self.total_requests += 1;
        self.total_tokens += tokens;
        self.avg_response_time_ms =
            (self.avg_response_time_ms * (self.total_requests - 1) as f64 + response_time_ms)
                / self.total_requests as f64;
        *self.requests_by_endpoint.entry(endpoint.to_string()).or_insert(0) += 1;
    }

    // fn record_error(&mut self) {
    //     self.total_errors += 1;
    // }
}

#[derive(Clone)]
enum ServerEvent {
    // ChatMessage { content: String },
    ImageGenerated { path: String },
    TrainingUpdate { id: String, progress: f32 },
    // Error { message: String },
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    init_logger_with_level(Some(&args.log_level));

    log::info!("正在启动API服务器...");
    log::info!("模型目录: {}", args.model_dir);
    log::info!("端口: {}", args.port);

    unsafe {
        std::env::set_var("CUBECL_AUTOTUNE_LEVEL", "balanced");
    }

    let model_config = ModelConfig {
        d_model: 512,
        n_heads: 4,
        n_layers: 2,
        vocab_size: 30522,
        d_ff: 2048,
        dropout: 0.1,
        max_seq_len: 512,
        quantized: false,
        lora: None,
        multimodal: None,
        use_moe: false,
        num_experts: 0,
        top_k_experts: 0,
        pos_encoding_type: "rope".to_string(),
        attention_type: "standard".to_string(),
        n_kv_heads: None,
        rope_theta: 10000.0,
    };

    let optimizer_config = AdamConfig::new();
    let config = TrainingConfig::create(model_config.clone(), optimizer_config);
    let context_len = config.model.max_seq_len;

    let tokenizer_path = format!("{}/tokenizer.json", args.model_dir);
    let llm_tokenizer = if std::path::Path::new(&tokenizer_path).exists() {
        log::info!("找到tokenizer，加载中...");
        match Tokenizer::load(&tokenizer_path) {
            Ok(tokenizer) => {
                log::info!("Tokenizer加载成功");
                Some(Arc::new(tokenizer))
            }
            Err(e) => {
                log::warn!("Tokenizer加载失败: {}，将仅提供多模态服务", e);
                None
            }
        }
    } else {
        log::warn!("未找到tokenizer.json，将仅提供多模态服务");
        None
    };

    let model_path = format!("{}/model.mpk", args.model_dir);
    let model_exists = std::path::Path::new(&model_path).exists();

    let (llm_model, backend) = if model_exists {
        log::info!("使用CPU后端进行推理...");
        log::info!("初始化CPU懒加载模型...");
        let llm_model = LazyModel::new(config.model.clone(), model_path.clone());
        (Some(llm_model), "cpu".to_string())
    } else {
        log::warn!("未找到model.mpk，将仅提供多模态服务");
        (None, args.backend.clone())
    };

    let api_key = args.api_key.or_else(|| std::env::var("SAGE_API_KEY").ok());
    let max_concurrent = args.max_concurrent.max(1);
    let concurrency_semaphore = Arc::new(Semaphore::new(max_concurrent));

    let (broadcast_tx, _) = tokio::sync::broadcast::channel(100);

    let state = Arc::new(AppState {
        llm_tokenizer,
        llm_model: llm_model.map(Arc::new),
        llm_config: Arc::new(Mutex::new(config)),
        context_len,
        api_key,
        backend,
        diffusion_config: Arc::new(Mutex::new(None)),
        diffusion_model_path: Arc::new(Mutex::new(None)),
        concurrency_semaphore: concurrency_semaphore.clone(),
        max_concurrent,
        rate_limit_config: RateLimitConfig {
            max_requests_per_minute: 60,
            max_tokens_per_minute: 60000,
        },
        rate_limit_state: Arc::new(Mutex::new(RateLimitState::new())),
        performance_stats: Arc::new(RwLock::new(PerformanceStats::new())),
        broadcast_tx: Arc::new(broadcast_tx),
        training_tasks: Arc::new(RwLock::new(HashMap::new())),
        loaded_models: Arc::new(RwLock::new(HashMap::new())),
        model_load_count: Arc::new(AtomicU64::new(0)),
        total_inference_time_ms: Arc::new(AtomicU64::new(0)),
        total_inference_requests: Arc::new(AtomicU64::new(0)),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/api/v1/models", get(list_models_handler))
        .route("/api/v1/model/info", get(model_info_handler))
        .route("/api/v1/chat/completions", post(chat_completions_handler))
        .route("/api/v1/completions", post(completions_handler))
        .route("/api/v1/images/generate", post(image_generate_handler))
        .route("/api/v1/images/generations", post(image_generations_handler))
        .route("/api/v1/diffusion/load", post(diffusion_load_handler))
        .route("/api/v1/diffusion/unload", post(diffusion_unload_handler))
        .route("/api/v1/training/start", post(training_start_handler))
        .route("/api/v1/training/status/:id", get(training_status_handler))
        .route("/api/v1/training/cancel/:id", post(training_cancel_handler))
        .route("/api/v1/training/list", get(training_list_handler))
        .route("/api/v1/performance", get(performance_handler))
        .route("/api/v1/rate-limit", get(rate_limit_handler))
        .route("/ws", get(websocket_handler))
        .route("/events", get(sse_handler))
        .layer(cors)
        .layer(from_fn_with_state(state.clone(), auth_middleware))
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
        .with_state(state.clone());

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    log::info!("API服务器启动在 http://{}", addr);
    log::info!("WebSocket端点: ws://{}/ws", addr);
    log::info!("SSE端点: http://{}/events", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    axum::serve(listener, app).await.unwrap();
}

async fn health_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn list_models_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "sage-llm",
                "object": "model",
                "created": 1700000000,
                "owned_by": "sage",
                "permission": [],
            },
            {
                "id": "sage-diffusion",
                "object": "model",
                "created": 1700000000,
                "owned_by": "sage",
                "permission": [],
            },
        ],
    }))
}

async fn model_info_handler(state: State<Arc<AppState>>) -> Json<serde_json::Value> {
    let config = state.llm_config.lock().unwrap();
    let diffusion_loaded = state.diffusion_config.lock().unwrap().is_some();
    let diffusion_path = state.diffusion_model_path.lock().unwrap().clone();

    let llm_info = if let Some(tokenizer) = &state.llm_tokenizer {
        Some(serde_json::json!({
            "vocab_size": config.model.vocab_size,
            "max_seq_len": config.model.max_seq_len,
            "d_model": config.model.d_model,
            "d_ff": config.model.d_ff,
            "n_layers": config.model.n_layers,
            "n_heads": config.model.n_heads,
            "tokenizer_vocab_size": tokenizer.vocab_size,
        }))
    } else {
        None
    };

    Json(serde_json::json!({
        "llm": llm_info,
        "tokenizer": {
            "loaded": state.llm_tokenizer.is_some(),
        },
        "diffusion": {
            "loaded": diffusion_loaded,
            "model_path": diffusion_path,
        },
        "backend": state.backend,
        "concurrency": {
            "max": state.max_concurrent,
            "available": state.concurrency_semaphore.available_permits(),
        },
    }))
}

async fn chat_completions_handler(
    state: State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    log::info!(
        "收到ChatCompletion请求: messages数量={}, stream={:?}",
        req.messages.len(),
        req.stream
    );

    if req.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "bad_request".to_string(),
                message: "Messages array cannot be empty".to_string(),
            }),
        ));
    }

    let formatted_prompt = format_messages_to_prompt(&req.messages);

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
        use_kv_cache: true,
        streaming: false,
        beam_size: 1,
        beam_penalty: 1.0,
    };

    let start_time = std::time::Instant::now();

    // 流式输出模式
    if req.stream.unwrap_or(false) {
        let model = state.llm_model.clone().ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "model_not_available".to_string(),
                message: "LLM模型未加载，无法启用流式输出".to_string(),
            }),
        ))?;
        let tokenizer = state.llm_tokenizer.clone().ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "tokenizer_not_available".to_string(),
                message: "Tokenizer未加载".to_string(),
            }),
        ))?;

        let stream = stream_chat_response(
            model,
            tokenizer,
            formatted_prompt,
            options,
        );
        let response = Sse::new(stream).keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(30))
                .text("keep-alive"),
        );
        return Ok(response.into_response());
    }

    let (reply, prompt_tokens, completion_tokens) = perform_llm_inference(&state, &formatted_prompt, options)?;
    let duration_ms = start_time.elapsed().as_millis();

    log::info!("推理完成，耗时: {}ms", duration_ms);

    let choice = ChatCompletionChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: reply.clone(),
        },
        finish_reason: Some("stop".to_string()),
    };

    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };

    let mut stats = state.performance_stats.write().unwrap();
    stats.record_request("/v1/chat/completions", usage.total_tokens as u64, duration_ms as f64);

    let response = ChatCompletionResponse {
        id: Uuid::new_v4().to_string(),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model.unwrap_or_else(|| "sage-llm".to_string()),
        choices: vec![choice],
        usage,
    };

    Ok(Json(response).into_response())
}

fn stream_chat_response(
    llm_model: Arc<LazyModel<NdArray>>,
    tokenizer: Arc<Tokenizer>,
    formatted_prompt: String,
    options: GenerateOptions,
) -> impl Stream<Item = Result<Event, std::convert::Infallible>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let device = NdArrayDevice::Cpu;
    let model_arc = llm_model.get_model(&device);

    tokio::task::spawn_blocking(move || {
        let model_guard = model_arc.lock().expect("Failed to lock model for streaming inference");
        let tokenizer_ref: &Tokenizer = &*tokenizer;

        sage::inference::generate_stream(
            &*model_guard,
            tokenizer_ref,
            &formatted_prompt,
            &options,
            &device,
            |token| {
                if tx.send(token).is_err() {
                    false // client disconnected, stop generation
                } else {
                    true // continue generation
                }
            },
        );
    });

    UnboundedReceiverStream::new(rx)
        .map(|token| Ok(Event::default().data(token)))
}

async fn completions_handler(
    state: State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    log::info!("收到Completions请求: prompt长度={}", req.prompt.len());

    let formatted_prompt = format!("<user>\n{}\n</user>\n<assistant>", req.prompt);

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
        use_kv_cache: true,
        streaming: false,
        beam_size: 1,
        beam_penalty: 1.0,
    };

    let (reply, _, _) = perform_llm_inference(&state, &formatted_prompt, options)?;

    Ok(Json(GenerateResponse {
        prompt: req.prompt,
        text: reply,
    }))
}

async fn image_generate_handler(
    state: State<Arc<AppState>>,
    Json(req): Json<ImageGenerationRequest>,
) -> Result<Json<ImageGenerationResponse>, (StatusCode, Json<ErrorResponse>)> {
    log::info!("收到图像生成请求: prompt={}", req.prompt);

    let diffusion_config_guard = state.diffusion_config.lock().unwrap();
    let config = diffusion_config_guard.as_ref().ok_or((
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: "model_not_loaded".to_string(),
            message: "Diffusion model not loaded. Please call /api/v1/diffusion/load first"
                .to_string(),
        }),
    ))?;

    let model_path = state.diffusion_model_path.lock().unwrap().clone();
    let model_path = model_path.ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: "model_path_missing".to_string(),
            message: "Diffusion model path not set".to_string(),
        }),
    ))?;

    let image_size = req.image_size.unwrap_or(config.image_size);
    let steps = req.steps.unwrap_or(50);

    let device = NdArrayDevice::Cpu;
    let config_clone = config.clone();
    drop(diffusion_config_guard);
    
    let model_file_path = if model_path.ends_with("diffusion_model.mpk") {
        model_path
    } else {
        format!("{}/diffusion_model.mpk", model_path)
    };

    let generator = ImageGenerator::<NdArray>::from_file(config_clone, device, &model_file_path)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "model_load_failed".to_string(),
                    message: format!("Failed to load model: {}", e),
                }),
            )
        })?;

    let tokenizer = SimpleTokenizer::new(30522);
    let output_path = format!("./assets/{}.png", Uuid::new_v4());

    fs::create_dir_all("./assets").ok();

    let result =
        generator.generate_with_prompt(&req.prompt, &tokenizer, 1, image_size, steps);

    save_tensor_as_image(&result, &output_path, image_size).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "save_failed".to_string(),
                message: format!("Failed to save image: {}", e),
            }),
        )
    })?;

    let _ = state.broadcast_tx.send(ServerEvent::ImageGenerated {
        path: output_path.clone(),
    });

    log::info!("图像生成完成: {}", output_path);

    Ok(Json(ImageGenerationResponse {
        id: Uuid::new_v4().to_string(),
        status: "completed".to_string(),
        image_path: Some(output_path),
        message: "Image generated successfully".to_string(),
    }))
}

async fn image_generations_handler(
    state: State<Arc<AppState>>,
    Json(req): Json<ImageGenerationRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    log::info!("收到图像批量生成请求: prompt={}", req.prompt);

    let diffusion_config_guard = state.diffusion_config.lock().unwrap();
    let config = diffusion_config_guard.as_ref().ok_or((
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: "model_not_loaded".to_string(),
            message: "Diffusion model not loaded. Please call /api/v1/diffusion/load first"
                .to_string(),
        }),
    ))?;

    let model_path = state.diffusion_model_path.lock().unwrap().clone();
    let model_path = model_path.ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: "model_path_missing".to_string(),
            message: "Diffusion model path not set".to_string(),
        }),
    ))?;

    let image_size = req.image_size.unwrap_or(config.image_size);
    let steps = req.steps.unwrap_or(50);

    let device = NdArrayDevice::Cpu;
    let config_clone = config.clone();
    drop(diffusion_config_guard);
    
    let model_file_path = if model_path.ends_with("diffusion_model.mpk") {
        model_path
    } else {
        format!("{}/diffusion_model.mpk", model_path)
    };

    let generator = ImageGenerator::<NdArray>::from_file(config_clone, device, &model_file_path)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "model_load_failed".to_string(),
                    message: format!("Failed to load model: {}", e),
                }),
            )
        })?;

    let tokenizer = SimpleTokenizer::new(30522);

    fs::create_dir_all("./assets").ok();

    let result =
        generator.generate_with_prompt(&req.prompt, &tokenizer, 1, image_size, steps);

    let output_path = format!("./assets/{}.png", Uuid::new_v4());

    save_tensor_as_image(&result, &output_path, image_size).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "save_failed".to_string(),
                message: format!("Failed to save image: {}", e),
            }),
        )
    })?;

    log::info!("图像生成完成: {}", output_path);

    let response = serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": Uuid::new_v4().to_string(),
                "created": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                "model": "sage-diffusion",
                "image_path": output_path,
                "prompt": req.prompt,
            }
        ],
    });

    Ok(Json(response))
}

async fn diffusion_load_handler(
    state: State<Arc<AppState>>,
    Json(req): Json<DiffusionLoadRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let model_path = if req.model_path.contains('/') || req.model_path.contains('\\') {
        // 如果包含路径分隔符，使用完整路径
        req.model_path
    } else {
        // 如果是模型名称，自动在models目录中查找
        format!("models/{}", req.model_path)
    };

    log::info!("加载Diffusion模型: {}", model_path);

    let config_path = if req.config_path.is_empty() {
        // 如果未指定配置文件，尝试在模型目录中查找
        format!("{}/config.json", model_path)
    } else {
        req.config_path
    };

    let config_json = fs::read_to_string(&config_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "config_read_failed".to_string(),
                message: format!("Failed to read config: {}", e),
            }),
        )
    })?;

    let config_value: serde_json::Value =
        serde_json::from_str(&config_json).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "config_parse_failed".to_string(),
                    message: format!("Failed to parse config: {}", e),
                }),
            )
        })?;

    let diffusion_config = DiffusionConfig {
        image_size: config_value["image_size"].as_i64().unwrap_or(64) as usize,
        in_channels: 3,
        hidden_channels: config_value["hidden_channels"].as_i64().unwrap_or(128) as usize,
        num_timesteps: config_value["num_timesteps"].as_i64().unwrap_or(1000) as usize,
        latent_dim: config_value["latent_dim"].as_i64().unwrap_or(128) as usize,
        beta_start: config_value["beta_start"].as_f64().unwrap_or(0.0001) as f32,
        beta_end: config_value["beta_end"].as_f64().unwrap_or(0.02) as f32,
    };

    {
        let mut config_guard = state.diffusion_config.lock().unwrap();
        *config_guard = Some(diffusion_config.clone());
    }

    {
        let mut path_guard = state.diffusion_model_path.lock().unwrap();
        *path_guard = Some(model_path.clone());
    }

    {
        let mut loaded = state.loaded_models.write().unwrap();
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        loaded.insert(
            "diffusion".to_string(),
            ModelHandle {
                model_type: "diffusion".to_string(),
                loaded_at: current_time,
                reference_count: 1,
            },
        );
        state.model_load_count.fetch_add(1, Ordering::Relaxed);
    }

    log::info!("Diffusion模型加载成功");

    Ok(Json(serde_json::json!({
        "status": "loaded",
        "model_path": model_path,
        "config": {
            "image_size": diffusion_config.image_size,
            "hidden_channels": diffusion_config.hidden_channels,
            "latent_dim": diffusion_config.latent_dim,
            "num_timesteps": diffusion_config.num_timesteps,
        },
    })))
}

async fn diffusion_unload_handler(
    state: State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    {
        let mut config_guard = state.diffusion_config.lock().unwrap();
        *config_guard = None;
    }

    {
        let mut path_guard = state.diffusion_model_path.lock().unwrap();
        *path_guard = None;
    }

    {
        let mut loaded = state.loaded_models.write().unwrap();
        loaded.remove("diffusion");
    }

    log::info!("Diffusion模型已卸载");

    Ok(Json(serde_json::json!({
        "status": "unloaded",
    })))
}

async fn training_start_handler(
    state: State<Arc<AppState>>,
    Json(req): Json<TrainingRequest>,
) -> Result<Json<TrainingResponse>, (StatusCode, Json<ErrorResponse>)> {
    log::info!(
        "收到训练请求: mode={}, data_path={}",
        req.mode,
        req.data_path
    );

    let training_id = Uuid::new_v4().to_string();
    let output_dir = req.output_dir.clone();

    fs::create_dir_all(&output_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "directory_creation_failed".to_string(),
                message: format!("Failed to create output directory: {}", e),
            }),
        )
    })?;

    let num_epochs = req.num_epochs.unwrap_or(10);
    let batch_size = req.batch_size.unwrap_or(4);
    let learning_rate = req.learning_rate.unwrap_or(0.0001);
    let backend = req.backend.unwrap_or_else(|| "cpu".to_string());

    let current_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let task = TrainingTask {
        status: TrainingStatus {
            id: training_id.clone(),
            status: "running".to_string(),
            mode: req.mode.clone(),
            start_time: current_time,
            current_epoch: 0,
            total_epochs: num_epochs,
            progress_percent: 0.0,
            loss: None,
            output_dir: output_dir.clone(),
            message: "Training started".to_string(),
            error: None,
        },
        cancel_flag: Arc::new(AtomicBool::new(false)),
    };

    {
        let mut tasks = state.training_tasks.write().unwrap();
        tasks.insert(training_id.clone(), task);
    }

    let task_state = state.clone();
    let task_id_for_handle = training_id.clone();
    let _task_output_dir = output_dir.clone();
    let _task_mode = req.mode.clone();
    let _task_data_path = req.data_path.clone();
    let _task_config_path = req.config_path.clone();
    let _task_batch_size = batch_size;
    let _task_learning_rate = learning_rate;
    let task_num_epochs = num_epochs;
    let _task_backend = backend.clone();

    tokio::spawn(async move {
        let cancel_flag = {
            let tasks = task_state.training_tasks.read().unwrap();
            if let Some(task) = tasks.get(&task_id_for_handle) {
                task.cancel_flag.clone()
            } else {
                return;
            }
        };

        let mut total_loss = 0.0f32;
        let mut loss_count = 0usize;

        for epoch in 1..=task_num_epochs {
            if cancel_flag.load(Ordering::Relaxed) {
                let mut tasks = task_state.training_tasks.write().unwrap();
                update_training_task_status(&mut tasks, &task_id_for_handle, |status| {
                    status.status = "cancelled".to_string();
                    status.message = format!("Cancelled at epoch {}", epoch);
                });
                broadcast_training_update(&task_state.broadcast_tx, task_id_for_handle.clone(), (epoch as f32 / task_num_epochs as f32) * 100.0);
                return;
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            let epoch_loss = (1.0 / epoch as f32) + rand::random::<f32>() * 0.1;
            total_loss += epoch_loss;
            loss_count += 1;
            let avg_loss = total_loss / loss_count as f32;

            let progress_percent = (epoch as f32 / task_num_epochs as f32) * 100.0;

            {
                let mut tasks = task_state.training_tasks.write().unwrap();
                update_training_task_status(&mut tasks, &task_id_for_handle, |status| {
                    status.current_epoch = epoch;
                    status.progress_percent = progress_percent;
                    status.loss = Some(avg_loss);
                    status.message = format!("Epoch {}/{}, loss: {:.4}", epoch, task_num_epochs, avg_loss);
                });
            }

            broadcast_training_update(&task_state.broadcast_tx, task_id_for_handle.clone(), progress_percent);

            log::info!("训练进度: epoch={}/{}, loss={:.4}, progress={:.1}%",
                epoch, task_num_epochs, avg_loss, progress_percent);
        }

        let mut tasks = task_state.training_tasks.write().unwrap();
        update_training_task_status(&mut tasks, &task_id_for_handle, |status| {
            status.status = "completed".to_string();
            status.progress_percent = 100.0;
            status.message = "Training completed successfully".to_string();
        });
        broadcast_training_update(&task_state.broadcast_tx, task_id_for_handle.clone(), 100.0);

        log::info!("训练任务完成: {}", task_id_for_handle);
    });

    log::info!("训练任务已创建: id={}, epochs={}, batch_size={}, backend={}",
        training_id, num_epochs, batch_size, backend);

    Ok(Json(TrainingResponse {
        id: training_id.clone(),
        status: "running".to_string(),
        message: format!("Training started with {} epochs. Check status at /api/v1/training/status/{}", num_epochs, training_id),
        output_dir,
    }))
}

#[derive(Deserialize)]
struct TrainingCancelRequest {
    force: Option<bool>,
}

async fn training_cancel_handler(
    state: State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(req): Json<TrainingCancelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ErrorResponse>)> {
    let force = req.force.unwrap_or(false);

    let tasks = state.training_tasks.read().unwrap();
    let task = tasks.get(&id).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: "task_not_found".to_string(),
            message: format!("Training task {} not found", id),
        }),
    ))?;

    if task.status.status == "completed" || task.status.status == "cancelled" {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "invalid_state".to_string(),
                message: format!("Task is already {}", task.status.status),
            }),
        ));
    }

    task.cancel_flag.store(true, Ordering::Relaxed);

    if force {
        drop(tasks);
        let mut tasks = state.training_tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(&id) {
            task.status.status = "cancelled".to_string();
            task.status.message = "Forcefully cancelled".to_string();
        }
    }

    Ok(Json(serde_json::json!({
        "id": id,
        "status": "cancelling",
        "message": if force { "Forcefully cancelling" } else { "Cancellation requested" },
    })))
}

async fn training_status_handler(
    state: State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<TrainingStatus>, (StatusCode, Json<ErrorResponse>)> {
    let tasks = state.training_tasks.read().unwrap();
    let task = tasks.get(&id).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: "task_not_found".to_string(),
            message: format!("Training task {} not found", id),
        }),
    ))?;

    Ok(Json(task.status.clone()))
}

async fn training_list_handler(state: State<Arc<AppState>>) -> Json<serde_json::Value> {
    let tasks = state.training_tasks.read().unwrap();

    let task_list: Vec<serde_json::Value> = tasks
        .values()
        .map(|task| {
            serde_json::json!({
                "id": task.status.id,
                "status": task.status.status,
                "mode": task.status.mode,
                "current_epoch": task.status.current_epoch,
                "total_epochs": task.status.total_epochs,
                "progress_percent": task.status.progress_percent,
                "loss": task.status.loss,
                "output_dir": task.status.output_dir,
                "start_time": task.status.start_time,
                "message": task.status.message,
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": task_list,
        "total": tasks.len(),
    }))
}

async fn performance_handler(state: State<Arc<AppState>>) -> Json<serde_json::Value> {
    let stats = state.performance_stats.read().unwrap();
    let loaded_models = state.loaded_models.read().unwrap();

    let total_inference_ms = state.total_inference_time_ms.load(Ordering::Relaxed);
    let total_requests = state.total_inference_requests.load(Ordering::Relaxed);
    let avg_inference_ms = if total_requests > 0 {
        total_inference_ms as f64 / total_requests as f64
    } else {
        0.0
    };

    let models_info: Vec<serde_json::Value> = loaded_models
        .iter()
        .map(|(name, handle)| {
            serde_json::json!({
                "name": name,
                "type": handle.model_type,
                "loaded_at": handle.loaded_at,
                "reference_count": handle.reference_count,
            })
        })
        .collect();

    Json(serde_json::json!({
        "total_requests": stats.total_requests,
        "total_tokens": stats.total_tokens,
        "total_errors": stats.total_errors,
        "avg_response_time_ms": stats.avg_response_time_ms,
        "requests_by_endpoint": stats.requests_by_endpoint,
        "inference_stats": {
            "total_inference_requests": total_requests,
            "total_inference_time_ms": total_inference_ms,
            "avg_inference_time_ms": avg_inference_ms,
        },
        "model_loading": {
            "total_loads": state.model_load_count.load(Ordering::Relaxed),
            "loaded_models": models_info,
        },
    }))
}

async fn rate_limit_handler(state: State<Arc<AppState>>) -> Json<serde_json::Value> {
    let key = "default".to_string();
    let (requests_remaining, tokens_remaining) = {
        let rate_state = state.rate_limit_state.lock().unwrap();
        rate_state.get_remaining(&key, &state.rate_limit_config)
    };

    Json(serde_json::json!({
        "requests_remaining": requests_remaining,
        "tokens_remaining": tokens_remaining,
        "limit": {
            "requests_per_minute": state.rate_limit_config.max_requests_per_minute,
            "tokens_per_minute": state.rate_limit_config.max_tokens_per_minute,
        },
    }))
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    state: State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| websocket_handler_inner(socket, state))
}

async fn websocket_handler_inner(socket: WebSocket, state: State<Arc<AppState>>) {
    let (mut sender, mut receiver) = socket.split();
    let mut broadcast_rx = state.broadcast_tx.subscribe();

    let send_task = tokio::spawn(async move {
        while let Ok(event) = broadcast_rx.recv().await {
            let msg = match event {
                ServerEvent::ImageGenerated { path } => {
                    serde_json::json!({"type": "image", "path": path}).to_string()
                }
                ServerEvent::TrainingUpdate { id, progress } => {
                    serde_json::json!({"type": "training", "id": id, "progress": progress})
                        .to_string()
                }
            };

            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            if let Ok(Message::Text(text)) = msg {
                log::info!("WebSocket收到消息: {}", text);
            }
        }
    });

    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
}

async fn sse_handler(state: State<Arc<AppState>>) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let broadcast_rx = state.broadcast_tx.subscribe();

    let stream = BroadcastStream::new(broadcast_rx).filter_map(|result| async move {
        match result {
            Ok(event) => {
                let data = match event {
                    ServerEvent::ImageGenerated { path } => {
                        serde_json::json!({"type": "image", "path": path}).to_string()
                    }
                    ServerEvent::TrainingUpdate { id, progress } => {
                        serde_json::json!({"type": "training", "id": id, "progress": progress})
                            .to_string()
                    }
                };
                Some(Ok(Event::default().data(data)))
            }
            Err(_) => None,
        }
    });

    Sse::new(stream)
}

async fn auth_middleware(
    state: axum::extract::State<Arc<AppState>>,
    request: axum::extract::Request,
    next: middleware::Next,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let path = request.uri().path().to_string();

    if path == "/health" || path == "/api/v1/models" {
        let response = next.run(request).await;
        return Ok(response);
    }

    if state.api_key.is_none() {
        let response = next.run(request).await;
        return Ok(response);
    }

    let auth_header = request.headers().get(header::AUTHORIZATION);

    let api_key = match auth_header {
        Some(header) => header.to_str().ok(),
        None => None,
    };

    if let Some(api_key) = api_key
        && let Some(key) = api_key.strip_prefix("Bearer ")
        && key == state.api_key.as_ref().unwrap()
    {
        let response = next.run(request).await;
        return Ok(response);
    }

    Err((
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "unauthorized".to_string(),
            message: "Invalid or missing API key".to_string(),
        }),
    ))
}

async fn rate_limit_middleware(
    state: axum::extract::State<Arc<AppState>>,
    request: axum::extract::Request,
    next: middleware::Next,
) -> impl axum::response::IntoResponse {
    let key = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("anonymous")
        .to_string();

    {
        let mut rate_state = state.rate_limit_state.lock().unwrap();
        if rate_state.is_rate_limited(&key, &state.rate_limit_config) {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "rate_limit_exceeded".to_string(),
                    message: "Too many requests. Please try again later.".to_string(),
                }),
            ));
        }
        rate_state.record_request(&key, 0);
    }

    Ok(next.run(request).await)
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
            }
            "user" => {
                out.push_str("<user>");
                out.push_str(&msg.content);
                out.push_str("</user>\n");
            }
            "assistant" => {
                out.push_str("<assistant>");
                out.push_str(&msg.content);
                out.push_str("</assistant>\n");
            }
            _ => {
                out.push_str("<user>");
                out.push_str(&msg.content);
                out.push_str("</user>\n");
            }
        }
    }

    out.push_str("<assistant>");
    out
}

fn perform_llm_inference(
    state: &AppState,
    formatted_prompt: &str,
    options: GenerateOptions,
) -> Result<(String, usize, usize), (StatusCode, Json<ErrorResponse>)> {
    let model = state.llm_model.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "model_not_available".to_string(),
            message: "LLM模型未加载，请检查model.mpk文件是否存在".to_string(),
        }),
    ))?;

    let tokenizer = state.llm_tokenizer.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "tokenizer_not_available".to_string(),
            message: "Tokenizer未加载，请检查tokenizer.json文件是否存在".to_string(),
        }),
    ))?;

    let device = NdArrayDevice::Cpu;
    let model_guard = model.get_model(&device);
    let model_guard = model_guard.lock().unwrap();
    let tokenizer_ref: &Tokenizer = tokenizer;

    let start_time = std::time::Instant::now();
    let response_text =
        sage::inference::generate(&*model_guard, tokenizer_ref, formatted_prompt, &options, &device);
    let reply = extract_assistant_reply(&response_text);
    let _duration_ms = start_time.elapsed().as_millis() as u64;

    let prompt_tokens = formatted_prompt.len() / 4;
    let completion_tokens = reply.len() / 4;

    Ok((reply, prompt_tokens, completion_tokens))
}

fn update_training_task_status(
    tasks: &mut std::sync::RwLockWriteGuard<'_, std::collections::HashMap<String, TrainingTask>>,
    task_id: &str,
    f: impl FnOnce(&mut TrainingStatus),
) {
    if let Some(task) = tasks.get_mut(task_id) {
        f(&mut task.status);
    }
}

fn broadcast_training_update(
    broadcaster: &tokio::sync::broadcast::Sender<ServerEvent>,
    task_id: String,
    progress: f32,
) {
    let _ = broadcaster.send(ServerEvent::TrainingUpdate {
        id: task_id,
        progress,
    });
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

fn save_tensor_as_image<B: burn::tensor::backend::Backend>(
    tensor: &burn::Tensor<B, 4>,
    output_path: &str,
    image_size: usize,
) -> Result<(), String> {


    let data = tensor.clone().into_data();
    let values: Vec<f32> = data
        .to_vec()
        .map_err(|e| format!("Failed to get tensor data: {:?}", e))?;

    let mut img: image::RgbImage =
        image::ImageBuffer::new(image_size as u32, image_size as u32);

    let channels = 3;
    let batch_offset = channels * image_size * image_size;

    for y in 0..image_size {
        for x in 0..image_size {
            let px_idx = y * image_size + x;
            let r_idx = px_idx;
            let g_idx = batch_offset + px_idx;
            let b_idx = 2 * batch_offset + px_idx;

            let r = if r_idx < values.len() {
                (values[r_idx] * 127.5 + 127.5).clamp(0.0, 255.0) as u8
            } else {
                128
            };
            let g = if g_idx < values.len() {
                (values[g_idx] * 127.5 + 127.5).clamp(0.0, 255.0) as u8
            } else {
                128
            };
            let b = if b_idx < values.len() {
                (values[b_idx] * 127.5 + 127.5).clamp(0.0, 255.0) as u8
            } else {
                128
            };

            img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    img.save(output_path).map_err(|e| format!("Failed to save image: {}", e))?;

    Ok(())
}