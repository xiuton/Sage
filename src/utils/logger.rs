use log::{debug, info};
use std::io::Write;
use std::sync::Once;

static INIT_LOGGER: Once = Once::new();

pub fn init_logger() {
    INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .format(|buf, record| {
                writeln!(
                    buf,
                    "[{}] {}: {}",
                    record.level(),
                    record.target(),
                    record.args()
                )
            })
            .init();
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
