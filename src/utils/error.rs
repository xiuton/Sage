use std::fmt;

#[derive(Debug)]
pub enum SageError {
    ModelLoadingError(String),
    TokenizerError(String),
    InferenceError(String),
    ConfigurationError(String),
    IOError(std::io::Error),
    SerdeError(serde_json::Error),
    BackendError(String),
    QuantizationError(String),
    ApiError(String),
}

impl fmt::Display for SageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SageError::ModelLoadingError(msg) => write!(f, "模型加载错误: {}", msg),
            SageError::TokenizerError(msg) => write!(f, "分词器错误: {}", msg),
            SageError::InferenceError(msg) => write!(f, "推理错误: {}", msg),
            SageError::ConfigurationError(msg) => write!(f, "配置错误: {}", msg),
            SageError::IOError(e) => write!(f, "IO错误: {}", e),
            SageError::SerdeError(e) => write!(f, "序列化错误: {}", e),
            SageError::BackendError(msg) => write!(f, "后端错误: {}", msg),
            SageError::QuantizationError(msg) => write!(f, "量化错误: {}", msg),
            SageError::ApiError(msg) => write!(f, "API错误: {}", msg),
        }
    }
}

impl std::error::Error for SageError {}

impl From<std::io::Error> for SageError {
    fn from(e: std::io::Error) -> Self {
        SageError::IOError(e)
    }
}

impl From<serde_json::Error> for SageError {
    fn from(e: serde_json::Error) -> Self {
        SageError::SerdeError(e)
    }
}

pub type Result<T> = std::result::Result<T, SageError>;
