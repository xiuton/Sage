use log::{debug, info, warn};
use std::io::Write;
use std::sync::Once;
use std::env;

static INIT_LOGGER: Once = Once::new();

pub fn init_logger() {
    init_logger_with_level(None);
}

pub fn init_logger_with_level(log_level: Option<&str>) {
    INIT_LOGGER.call_once(|| {
        let log_level = log_level
            .map(|s| s.to_string())
            .or_else(|| env::var("RUST_LOG").ok())
            .unwrap_or_else(|| "info".to_string());
        let env_filter = env_logger::Env::default().default_filter_or(&log_level);
        
        let mut binding = env_logger::Builder::from_env(env_filter);
        let builder = binding
            .format(|buf, record| {
                writeln!(
                    buf,
                    "[{}] {}: {}",
                    record.level(),
                    record.target(),
                    record.args()
                )
            });
        
        // 在Windows上使用更健壮的日志配置，避免文件锁定问题
        match builder.try_init() {
            Ok(()) => {}
            Err(e) => {
                warn!("日志初始化失败: {}", e);
                warn!("继续运行，但日志可能不完整");
            }
        }
    });
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {
        log::debug!($($arg)*);
    };
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        log::info!($($arg)*);
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        log::warn!($($arg)*);
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        log::error!($($arg)*);
    };
}

pub fn log_performance(endpoint: &str, time_ms: f64, tokens_per_second: f64) {
    info!(
        "[PERFORMANCE] {} - 耗时: {:.2}ms, 速度: {:.2} tokens/s",
        endpoint, time_ms, tokens_per_second
    );
}

pub fn log_model_loading(model_name: &str, time_ms: u64) {
    info!("[MODEL] 加载模型 {} 完成，耗时: {}ms", model_name, time_ms);
}

pub fn log_inference_start(prompt_length: usize, max_tokens: usize) {
    debug!(
        "[INFERENCE] 开始推理 - Prompt长度: {}, 最大生成长度: {}",
        prompt_length, max_tokens
    );
}

pub fn log_inference_end(completion_length: usize, time_ms: f64) {
    debug!(
        "[INFERENCE] 推理完成 - 生成长度: {}, 耗时: {:.2}ms",
        completion_length, time_ms
    );
}
