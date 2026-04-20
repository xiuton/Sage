use serde::{Deserialize, Serialize};
use crate::core::ModelConfig;
use crate::training::DPOConfig;
use std::env;
use std::collections::HashMap;

/// 学习率调度器配置
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LRSchedulerConfig {
    pub lr_max: f64,
    pub lr_min: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

/// 训练配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: burn::optim::AdamConfig,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub lr: f64,
    pub max_seq_len: usize,
    pub save_dir: String,
    pub save_interval: usize,
    pub log_interval: usize,
    pub eval_interval: usize,
    pub eval_steps: usize,
    pub use_amp: bool,
    pub use_lora: bool,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub lora_dropout: f32,
    pub distributed: bool,
    pub num_devices: usize,
    pub devices: Vec<String>,
    pub use_wandb: bool,
    pub wandb_project: String,
    pub wandb_run_name: String,
    pub num_workers: usize,
    pub seed: u64,
    pub no_progress: bool,
    pub gradient_accumulation_steps: usize,
    pub dpo_config: Option<DPOConfig>,
    pub lr_scheduler: Option<LRSchedulerConfig>,
}

impl TrainingConfig {
    pub fn create(model: ModelConfig, optimizer: burn::optim::AdamConfig) -> Self {
        Self {
            model,
            optimizer,
            batch_size: 32,
            num_epochs: 50,
            lr: 5.0e-4,
            max_seq_len: 512,
            save_dir: "./tmp".to_string(),
            save_interval: 1000,
            log_interval: 100,
            eval_interval: 1000,
            eval_steps: 100,
            use_amp: false,
            use_lora: false,
            lora_rank: 8,
            lora_alpha: 16.0,
            lora_dropout: 0.1,
            distributed: false,
            num_devices: 1,
            devices: Vec::new(),
            use_wandb: false,
            wandb_project: "sage".to_string(),
            wandb_run_name: format!("run-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
            num_workers: 4,
            seed: 42,
            no_progress: false,
            gradient_accumulation_steps: 1,
            dpo_config: None,
            lr_scheduler: None,
        }
    }
    
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        config.load_from_env();
        Ok(config)
    }
    
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// 验证配置的有效性
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.batch_size == 0 {
            return Err("batch_size must be greater than 0".into());
        }
        if self.num_epochs == 0 {
            return Err("num_epochs must be greater than 0".into());
        }
        if self.lr <= 0.0 {
            return Err("lr must be greater than 0".into());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be greater than 0".into());
        }
        if self.gradient_accumulation_steps == 0 {
            return Err("gradient_accumulation_steps must be greater than 0".into());
        }
        if self.use_lora && self.lora_rank == 0 {
            return Err("lora_rank must be greater than 0 when use_lora is true".into());
        }
        Ok(())
    }
    
    /// 从环境变量加载配置
    pub fn load_from_env(&mut self) {
        if let Ok(value) = env::var("SAGE_BATCH_SIZE") {
            if let Ok(value) = value.parse() {
                self.batch_size = value;
            }
        }
        if let Ok(value) = env::var("SAGE_NUM_EPOCHS") {
            if let Ok(value) = value.parse() {
                self.num_epochs = value;
            }
        }
        if let Ok(value) = env::var("SAGE_LR") {
            if let Ok(value) = value.parse() {
                self.lr = value;
            }
        }
        if let Ok(value) = env::var("SAGE_MAX_SEQ_LEN") {
            if let Ok(value) = value.parse() {
                self.max_seq_len = value;
            }
        }
        if let Ok(value) = env::var("SAGE_SAVE_DIR") {
            self.save_dir = value;
        }
        if let Ok(value) = env::var("SAGE_USE_AMP") {
            if let Ok(value) = value.parse() {
                self.use_amp = value;
            }
        }
        if let Ok(value) = env::var("SAGE_USE_LORA") {
            if let Ok(value) = value.parse() {
                self.use_lora = value;
            }
        }
        if let Ok(value) = env::var("SAGE_DISTRIBUTED") {
            if let Ok(value) = value.parse() {
                self.distributed = value;
            }
        }
    }
    
    /// 合并两个配置，当前配置优先
    pub fn merge(&mut self, other: &Self) {
        if self.batch_size == 0 {
            self.batch_size = other.batch_size;
        }
        if self.num_epochs == 0 {
            self.num_epochs = other.num_epochs;
        }
        if self.lr == 0.0 {
            self.lr = other.lr;
        }
        if self.max_seq_len == 0 {
            self.max_seq_len = other.max_seq_len;
        }
        if self.save_dir.is_empty() {
            self.save_dir = other.save_dir.clone();
        }
        if self.save_interval == 0 {
            self.save_interval = other.save_interval;
        }
        if self.log_interval == 0 {
            self.log_interval = other.log_interval;
        }
        if self.eval_interval == 0 {
            self.eval_interval = other.eval_interval;
        }
        if self.eval_steps == 0 {
            self.eval_steps = other.eval_steps;
        }
        if self.num_workers == 0 {
            self.num_workers = other.num_workers;
        }
        if self.gradient_accumulation_steps == 0 {
            self.gradient_accumulation_steps = other.gradient_accumulation_steps;
        }
        if self.dpo_config.is_none() {
            self.dpo_config = other.dpo_config.clone();
        }
        if self.lr_scheduler.is_none() {
            self.lr_scheduler = other.lr_scheduler.clone();
        }
    }
    
    /// 从HashMap创建配置
    pub fn from_hashmap(map: &HashMap<String, String>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = Self::create(ModelConfig::default(), burn::optim::AdamConfig::new());
        
        if let Some(value) = map.get("batch_size") {
            if let Ok(value) = value.parse() {
                config.batch_size = value;
            }
        }
        if let Some(value) = map.get("num_epochs") {
            if let Ok(value) = value.parse() {
                config.num_epochs = value;
            }
        }
        if let Some(value) = map.get("lr") {
            if let Ok(value) = value.parse() {
                config.lr = value;
            }
        }
        if let Some(value) = map.get("max_seq_len") {
            if let Ok(value) = value.parse() {
                config.max_seq_len = value;
            }
        }
        if let Some(value) = map.get("save_dir") {
            config.save_dir = value.clone();
        }
        
        config.validate()?;
        Ok(config)
    }
}

/// 推理配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub max_seq_len: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub punctuation_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub use_gpu: bool,
    pub use_quantization: bool,
    pub quantization_mode: String,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_path: "./tmp/model.mpk".to_string(),
            tokenizer_path: "./tmp/tokenizer.json".to_string(),
            max_seq_len: 512,
            temperature: 0.8,
            top_k: 10,
            top_p: 0.9,
            repetition_penalty: 1.1,
            punctuation_penalty: 1.3,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            use_gpu: false,
            use_quantization: false,
            quantization_mode: "dynamic".to_string(),
        }
    }
}

impl InferenceConfig {
    /// 验证配置的有效性
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.model_path.is_empty() {
            return Err("model_path must be set".into());
        }
        if self.tokenizer_path.is_empty() {
            return Err("tokenizer_path must be set".into());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be greater than 0".into());
        }
        if self.temperature < 0.0 {
            return Err("temperature must be non-negative".into());
        }
        if self.top_k == 0 {
            return Err("top_k must be greater than 0".into());
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err("top_p must be between 0.0 and 1.0".into());
        }
        if self.repetition_penalty < 1.0 {
            return Err("repetition_penalty must be at least 1.0".into());
        }
        Ok(())
    }
    
    /// 从环境变量加载配置
    pub fn load_from_env(&mut self) {
        if let Ok(value) = env::var("SAGE_MODEL_PATH") {
            self.model_path = value;
        }
        if let Ok(value) = env::var("SAGE_TOKENIZER_PATH") {
            self.tokenizer_path = value;
        }
        if let Ok(value) = env::var("SAGE_MAX_SEQ_LEN") {
            if let Ok(value) = value.parse() {
                self.max_seq_len = value;
            }
        }
        if let Ok(value) = env::var("SAGE_TEMPERATURE") {
            if let Ok(value) = value.parse() {
                self.temperature = value;
            }
        }
        if let Ok(value) = env::var("SAGE_TOP_K") {
            if let Ok(value) = value.parse() {
                self.top_k = value;
            }
        }
        if let Ok(value) = env::var("SAGE_TOP_P") {
            if let Ok(value) = value.parse() {
                self.top_p = value;
            }
        }
        if let Ok(value) = env::var("SAGE_REPETITION_PENALTY") {
            if let Ok(value) = value.parse() {
                self.repetition_penalty = value;
            }
        }
        if let Ok(value) = env::var("SAGE_USE_GPU") {
            if let Ok(value) = value.parse() {
                self.use_gpu = value;
            }
        }
        if let Ok(value) = env::var("SAGE_USE_QUANTIZATION") {
            if let Ok(value) = value.parse() {
                self.use_quantization = value;
            }
        }
    }
    
    /// 合并两个配置，当前配置优先
    pub fn merge(&mut self, other: &Self) {
        if self.model_path.is_empty() {
            self.model_path = other.model_path.clone();
        }
        if self.tokenizer_path.is_empty() {
            self.tokenizer_path = other.tokenizer_path.clone();
        }
        if self.max_seq_len == 0 {
            self.max_seq_len = other.max_seq_len;
        }
        if self.temperature == 0.0 {
            self.temperature = other.temperature;
        }
        if self.top_k == 0 {
            self.top_k = other.top_k;
        }
        if self.top_p == 0.0 {
            self.top_p = other.top_p;
        }
        if self.repetition_penalty == 0.0 {
            self.repetition_penalty = other.repetition_penalty;
        }
        if self.punctuation_penalty == 0.0 {
            self.punctuation_penalty = other.punctuation_penalty;
        }
        if self.quantization_mode.is_empty() {
            self.quantization_mode = other.quantization_mode.clone();
        }
    }
    
    /// 从HashMap创建配置
    pub fn from_hashmap(map: &HashMap<String, String>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = Self::default();
        
        if let Some(value) = map.get("model_path") {
            config.model_path = value.clone();
        }
        if let Some(value) = map.get("tokenizer_path") {
            config.tokenizer_path = value.clone();
        }
        if let Some(value) = map.get("max_seq_len") {
            if let Ok(value) = value.parse() {
                config.max_seq_len = value;
            }
        }
        if let Some(value) = map.get("temperature") {
            if let Ok(value) = value.parse() {
                config.temperature = value;
            }
        }
        if let Some(value) = map.get("top_k") {
            if let Ok(value) = value.parse() {
                config.top_k = value;
            }
        }
        if let Some(value) = map.get("top_p") {
            if let Ok(value) = value.parse() {
                config.top_p = value;
            }
        }
        
        config.validate()?;
        Ok(config)
    }
    
    /// 加载配置文件
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        config.load_from_env();
        Ok(config)
    }
    
    /// 保存配置文件
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// API配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiConfig {
    pub port: u16,
    pub host: String,
    pub max_concurrent: usize,
    pub api_key: Option<String>,
    pub log_level: String,
    pub use_gpu: bool,
    pub model_dir: String,
    pub use_best: bool,
    pub context_len: usize,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            port: 8000,
            host: "0.0.0.0".to_string(),
            max_concurrent: 2,
            api_key: None,
            log_level: "info".to_string(),
            use_gpu: false,
            model_dir: "./tmp/sage_model_formal".to_string(),
            use_best: false,
            context_len: 0,
        }
    }
}

impl ApiConfig {
    /// 验证配置的有效性
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.port == 0 {
            return Err("port must be greater than 0".into());
        }
        if self.host.is_empty() {
            return Err("host must be set".into());
        }
        if self.max_concurrent == 0 {
            return Err("max_concurrent must be greater than 0".into());
        }
        if self.model_dir.is_empty() {
            return Err("model_dir must be set".into());
        }
        Ok(())
    }
    
    /// 从环境变量加载配置
    pub fn load_from_env(&mut self) {
        if let Ok(value) = env::var("SAGE_API_PORT") {
            if let Ok(value) = value.parse() {
                self.port = value;
            }
        }
        if let Ok(value) = env::var("SAGE_API_HOST") {
            self.host = value;
        }
        if let Ok(value) = env::var("SAGE_API_MAX_CONCURRENT") {
            if let Ok(value) = value.parse() {
                self.max_concurrent = value;
            }
        }
        if let Ok(value) = env::var("SAGE_API_KEY") {
            self.api_key = Some(value);
        }
        if let Ok(value) = env::var("SAGE_API_LOG_LEVEL") {
            self.log_level = value;
        }
        if let Ok(value) = env::var("SAGE_API_USE_GPU") {
            if let Ok(value) = value.parse() {
                self.use_gpu = value;
            }
        }
        if let Ok(value) = env::var("SAGE_API_MODEL_DIR") {
            self.model_dir = value;
        }
        if let Ok(value) = env::var("SAGE_API_USE_BEST") {
            if let Ok(value) = value.parse() {
                self.use_best = value;
            }
        }
        if let Ok(value) = env::var("SAGE_API_CONTEXT_LEN") {
            if let Ok(value) = value.parse() {
                self.context_len = value;
            }
        }
    }
    
    /// 合并两个配置，当前配置优先
    pub fn merge(&mut self, other: &Self) {
        if self.port == 0 {
            self.port = other.port;
        }
        if self.host.is_empty() {
            self.host = other.host.clone();
        }
        if self.max_concurrent == 0 {
            self.max_concurrent = other.max_concurrent;
        }
        if self.api_key.is_none() {
            self.api_key = other.api_key.clone();
        }
        if self.log_level.is_empty() {
            self.log_level = other.log_level.clone();
        }
        if self.model_dir.is_empty() {
            self.model_dir = other.model_dir.clone();
        }
    }
    
    /// 从HashMap创建配置
    pub fn from_hashmap(map: &HashMap<String, String>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = Self::default();
        
        if let Some(value) = map.get("port") {
            if let Ok(value) = value.parse() {
                config.port = value;
            }
        }
        if let Some(value) = map.get("host") {
            config.host = value.clone();
        }
        if let Some(value) = map.get("max_concurrent") {
            if let Ok(value) = value.parse() {
                config.max_concurrent = value;
            }
        }
        if let Some(value) = map.get("api_key") {
            config.api_key = Some(value.clone());
        }
        if let Some(value) = map.get("model_dir") {
            config.model_dir = value.clone();
        }
        
        config.validate()?;
        Ok(config)
    }
    
    /// 加载配置文件
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        config.load_from_env();
        Ok(config)
    }
    
    /// 保存配置文件
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
