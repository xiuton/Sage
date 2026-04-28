use std::fmt;

#[derive(Debug)]
pub enum SageError {
    ModelLoadingError {
        message: String,
        file_path: Option<String>,
        suggestion: String,
    },
    TokenizerError {
        message: String,
        text: Option<String>,
        suggestion: String,
    },
    InferenceError {
        message: String,
        input_shape: Option<String>,
        model_info: Option<String>,
        suggestion: String,
    },
    ConfigurationError {
        message: String,
        config_file: Option<String>,
        suggestion: String,
    },
    IOError(std::io::Error),
    SerdeError(serde_json::Error),
    BackendError {
        message: String,
        backend_type: String,
        suggestion: String,
    },
    QuantizationError {
        message: String,
        quantization_type: String,
        suggestion: String,
    },
    ApiError {
        message: String,
        endpoint: Option<String>,
        suggestion: String,
    },
    TrainingError {
        message: String,
        epoch: Option<usize>,
        batch: Option<usize>,
        suggestion: String,
    },
    ValidationError {
        message: String,
        field_name: Option<String>,
        expected_value: Option<String>,
        suggestion: String,
    },
}

impl fmt::Display for SageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SageError::ModelLoadingError { message, file_path, suggestion } => {
                if let Some(path) = file_path {
                    write!(f, "🚨 模型加载错误\n\n错误信息: {}\n文件路径: {}\n\n💡 建议解决方案:\n{}", message, path, suggestion)
                } else {
                    write!(f, "🚨 模型加载错误\n\n错误信息: {}\n\n💡 建议解决方案:\n{}", message, suggestion)
                }
            }
            SageError::TokenizerError { message, text, suggestion } => {
                if let Some(txt) = text {
                    write!(f, "🚨 分词器错误\n\n错误信息: {}\n输入文本: {}\n\n💡 建议解决方案:\n{}", message, txt, suggestion)
                } else {
                    write!(f, "🚨 分词器错误\n\n错误信息: {}\n\n💡 建议解决方案:\n{}", message, suggestion)
                }
            }
            SageError::InferenceError { message, input_shape, model_info, suggestion } => {
                let mut details = String::new();
                if let Some(shape) = input_shape {
                    details.push_str(&format!("输入形状: {}\n", shape));
                }
                if let Some(info) = model_info {
                    details.push_str(&format!("模型信息: {}\n", info));
                }
                
                write!(f, "🚨 推理错误\n\n错误信息: {}\n{}\n💡 建议解决方案:\n{}", message, details, suggestion)
            }
            SageError::ConfigurationError { message, config_file, suggestion } => {
                if let Some(file) = config_file {
                    write!(f, "🚨 配置错误\n\n错误信息: {}\n配置文件: {}\n\n💡 建议解决方案:\n{}", message, file, suggestion)
                } else {
                    write!(f, "🚨 配置错误\n\n错误信息: {}\n\n💡 建议解决方案:\n{}", message, suggestion)
                }
            }
            SageError::IOError(e) => write!(f, "🚨 IO错误\n\n错误信息: {}\n\n💡 建议检查文件权限和路径是否正确", e),
            SageError::SerdeError(e) => write!(f, "🚨 序列化错误\n\n错误信息: {}\n\n💡 建议检查JSON格式是否正确", e),
            SageError::BackendError { message, backend_type, suggestion } => {
                write!(f, "🚨 后端错误\n\n错误信息: {}\n后端类型: {}\n\n💡 建议解决方案:\n{}", message, backend_type, suggestion)
            }
            SageError::QuantizationError { message, quantization_type, suggestion } => {
                write!(f, "🚨 量化错误\n\n错误信息: {}\n量化类型: {}\n\n💡 建议解决方案:\n{}", message, quantization_type, suggestion)
            }
            SageError::ApiError { message, endpoint, suggestion } => {
                if let Some(ep) = endpoint {
                    write!(f, "🚨 API错误\n\n错误信息: {}\n端点: {}\n\n💡 建议解决方案:\n{}", message, ep, suggestion)
                } else {
                    write!(f, "🚨 API错误\n\n错误信息: {}\n\n💡 建议解决方案:\n{}", message, suggestion)
                }
            }
            SageError::TrainingError { message, epoch, batch, suggestion } => {
                let mut details = String::new();
                if let Some(ep) = epoch {
                    details.push_str(&format!("当前轮次: {}\n", ep));
                }
                if let Some(bt) = batch {
                    details.push_str(&format!("当前批次: {}\n", bt));
                }
                
                write!(f, "🚨 训练错误\n\n错误信息: {}\n{}\n💡 建议解决方案:\n{}", message, details, suggestion)
            }
            SageError::ValidationError { message, field_name, expected_value, suggestion } => {
                let mut details = String::new();
                if let Some(field) = field_name {
                    details.push_str(&format!("字段名: {}\n", field));
                }
                if let Some(expected) = expected_value {
                    details.push_str(&format!("期望值: {}\n", expected));
                }
                
                write!(f, "🚨 验证错误\n\n错误信息: {}\n{}\n💡 建议解决方案:\n{}", message, details, suggestion)
            }
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

impl From<String> for SageError {
    fn from(s: String) -> Self {
        SageError::configuration(s, None::<String>)
    }
}

pub type Result<T> = std::result::Result<T, SageError>;

// 便捷的错误创建函数
impl SageError {
    /// 创建模型加载错误
    pub fn model_loading(message: impl Into<String>, file_path: Option<String>) -> Self {
        SageError::ModelLoadingError {
            message: message.into(),
            file_path,
            suggestion: "请检查模型文件是否存在、格式是否正确，或尝试重新下载模型".to_string(),
        }
    }

    /// 创建分词器错误
    pub fn tokenizer(message: impl Into<String>, text: Option<String>) -> Self {
        SageError::TokenizerError {
            message: message.into(),
            text,
            suggestion: "请检查输入文本的编码格式，或尝试使用不同的分词策略".to_string(),
        }
    }

    /// 创建推理错误
    pub fn inference(message: impl Into<String>, input_shape: Option<String>, model_info: Option<String>) -> Self {
        SageError::InferenceError {
            message: message.into(),
            input_shape,
            model_info,
            suggestion: "请检查输入数据的形状和类型，确保与模型期望的输入匹配".to_string(),
        }
    }

    /// 创建配置错误
    pub fn configuration(message: impl Into<String>, config_file: Option<String>) -> Self {
        SageError::ConfigurationError {
            message: message.into(),
            config_file,
            suggestion: "请检查配置文件格式是否正确，或使用默认配置重新生成".to_string(),
        }
    }

    /// 创建后端错误
    pub fn backend(message: impl Into<String>, backend_type: impl Into<String>) -> Self {
        SageError::BackendError {
            message: message.into(),
            backend_type: backend_type.into(),
            suggestion: "请检查后端依赖是否正确安装，或尝试使用不同的后端".to_string(),
        }
    }

    /// 创建量化错误
    pub fn quantization(message: impl Into<String>, quantization_type: impl Into<String>) -> Self {
        SageError::QuantizationError {
            message: message.into(),
            quantization_type: quantization_type.into(),
            suggestion: "请检查量化参数设置是否正确，或尝试使用不同的量化策略".to_string(),
        }
    }

    /// 创建API错误
    pub fn api(message: impl Into<String>, endpoint: Option<String>) -> Self {
        SageError::ApiError {
            message: message.into(),
            endpoint,
            suggestion: "请检查API端点是否正确，或查看服务器日志获取更多信息".to_string(),
        }
    }

    /// 创建训练错误
    pub fn training(message: impl Into<String>, epoch: Option<usize>, batch: Option<usize>) -> Self {
        SageError::TrainingError {
            message: message.into(),
            epoch,
            batch,
            suggestion: "请检查训练数据格式、学习率设置，或尝试减小批量大小".to_string(),
        }
    }

    /// 创建验证错误
    pub fn validation(message: impl Into<String>, field_name: Option<String>, expected_value: Option<String>) -> Self {
        SageError::ValidationError {
            message: message.into(),
            field_name,
            expected_value,
            suggestion: "请检查输入数据的有效性，确保符合预期的格式和范围".to_string(),
        }
    }
}

// 错误上下文扩展
pub trait ErrorContext<T> {
    fn context(self, context: &str) -> Result<T>;
}

impl<T, E: Into<SageError>> ErrorContext<T> for std::result::Result<T, E> {
    fn context(self, context: &str) -> Result<T> {
        self.map_err(|e| {
            let base_error: SageError = e.into();
            match base_error {
                SageError::ModelLoadingError { message, file_path, suggestion } => 
                    SageError::ModelLoadingError { 
                        message: format!("{} (上下文: {})", message, context), 
                        file_path, 
                        suggestion 
                    },
                SageError::TokenizerError { message, text, suggestion } => 
                    SageError::TokenizerError { 
                        message: format!("{} (上下文: {})", message, context), 
                        text, 
                        suggestion 
                    },
                SageError::InferenceError { message, input_shape, model_info, suggestion } => 
                    SageError::InferenceError { 
                        message: format!("{} (上下文: {})", message, context), 
                        input_shape, 
                        model_info, 
                        suggestion 
                    },
                SageError::ConfigurationError { message, config_file, suggestion } => 
                    SageError::ConfigurationError { 
                        message: format!("{} (上下文: {})", message, context), 
                        config_file, 
                        suggestion 
                    },
                SageError::BackendError { message, backend_type, suggestion } => 
                    SageError::BackendError { 
                        message: format!("{} (上下文: {})", message, context), 
                        backend_type, 
                        suggestion 
                    },
                SageError::QuantizationError { message, quantization_type, suggestion } => 
                    SageError::QuantizationError { 
                        message: format!("{} (上下文: {})", message, context), 
                        quantization_type, 
                        suggestion 
                    },
                SageError::ApiError { message, endpoint, suggestion } => 
                    SageError::ApiError { 
                        message: format!("{} (上下文: {})", message, context), 
                        endpoint, 
                        suggestion 
                    },
                SageError::TrainingError { message, epoch, batch, suggestion } => 
                    SageError::TrainingError { 
                        message: format!("{} (上下文: {})", message, context), 
                        epoch, 
                        batch, 
                        suggestion 
                    },
                SageError::ValidationError { message, field_name, expected_value, suggestion } => 
                    SageError::ValidationError { 
                        message: format!("{} (上下文: {})", message, context), 
                        field_name, 
                        expected_value, 
                        suggestion 
                    },
                _ => base_error,
            }
        })
    }
}
