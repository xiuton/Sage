use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Module;
use clap::Parser;
use sage::{
    generation::{GenerateOptions, generate},
    tokenizer::Tokenizer,
    training::TrainingConfig,
};
use axum::{
    extract::Json,
    http::StatusCode,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    net::SocketAddr,
    sync::{Arc, Mutex},
};
use tokio::net::TcpListener;

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
}

#[derive(Deserialize)]
struct InferenceRequest {
    prompt: String,
    num_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    punctuation_penalty: Option<f32>,
    seed: Option<u64>,
}

#[derive(Serialize)]
struct InferenceResponse {
    response: String,
    prompt: String,
    num_tokens: usize,
    duration_ms: u64,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    message: String,
}

struct AppState {
    tokenizer: Tokenizer,
    model: Arc<Mutex<sage::model::Model<NdArray>>>,
    config: TrainingConfig,
    context_len: usize,
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

    // 加载模型
    let device = NdArrayDevice::Cpu;
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

    let model = config.model
        .init::<NdArray>(&device)
        .load_file(&model_path, &burn::record::CompactRecorder::new(), &device)
        .expect("Failed to load model");

    let state = Arc::new(AppState {
        tokenizer,
        model: Arc::new(Mutex::new(model)),
        config,
        context_len: args.context_len,
    });

    let app = Router::new()
        .route("/api/infer", post(infer_handler))
        .route("/api/health", get(health_handler))
        .route("/api/model-info", get(model_info_handler))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    println!("API服务器启动在 http://{}", addr);
    
    let listener = TcpListener::bind(addr).await.unwrap();
    
    axum::serve(listener, app)
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
    Json(req): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, (StatusCode, Json<ErrorResponse>)> {
    println!("收到推理请求: prompt='{}', num_tokens={}", req.prompt, req.num_tokens.unwrap_or(50));
    
    let start_time = std::time::Instant::now();

    if req.prompt.trim().is_empty() {
        println!("错误: prompt为空");
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "bad_request".to_string(),
                message: "Prompt cannot be empty".to_string(),
            }),
        ));
    }

    let options = GenerateOptions {
        max_new_tokens: req.num_tokens.unwrap_or(50),
        temperature: req.temperature.unwrap_or(0.8),
        top_k: req.top_k.unwrap_or(10),
        top_p: req.top_p.unwrap_or(0.9),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.1),
        punctuation_penalty: req.punctuation_penalty.unwrap_or(1.3),
        seed: req.seed,
        context_len: state.context_len,
        stop_on_user: true,
        stop_sequences: Vec::new(),
    };

    let device = NdArrayDevice::Cpu;
    let prompt = req.prompt.clone();
    
    // 格式化提示
    let formatted_prompt = format_chat_prefix(&prompt);
    println!("格式化后的提示: {}", formatted_prompt);
    
    // 执行推理
    println!("开始执行推理...");
    let model = state.model.lock().unwrap();
    let response = generate(&*model, &state.tokenizer, &formatted_prompt, &options, &device);
    println!("推理完成，原始响应: {}", response);
    
    // 提取助手回复
    let reply = extract_assistant_reply(&response);
    println!("提取的助手回复: {}", reply);
    
    let duration_ms = start_time.elapsed().as_millis();
    println!("推理耗时: {}ms", duration_ms);

    Ok(Json(InferenceResponse {
        response: reply,
        prompt,
        num_tokens: options.max_new_tokens,
        duration_ms: duration_ms as u64,
    }))
}

fn format_chat_prefix(user_text: &str) -> String {
    let estimated_len = 10 + user_text.len();
    let mut out = String::with_capacity(estimated_len);
    out.push('\u{0002}');
    out.push_str("<s>\n<user>");
    out.push_str(user_text);
    out.push_str("</user>\n<assistant>");
    out
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
