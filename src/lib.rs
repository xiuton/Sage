#![recursion_limit = "1024"]

// =============================================================================
// Sage - Rust 深度学习框架项目
// 框架: Burn (https://burn.dev/) - Rust 原生深度学习框架
// =============================================================================

// -----------------------------------------------------------------------------
// 框架模块 (基于 Burn 深度学习框架构建)
// -----------------------------------------------------------------------------
pub mod core;      // 核心模型模块 (Transformer, RMSNorm, SwiGLU) - 基于 Burn layers
pub mod data;      // 数据处理模块 - 基于 Burn Dataset trait
pub mod inference;  // 推理模块 - 基于 Burn 自动微分和推理 API
pub mod training;  // 训练模块 - 基于 Burn Optimizer 和 TrainingRunner
pub mod transformer; // Transformer 组件 - 基于 Burn TransformerEncoder

// -----------------------------------------------------------------------------
// 自定义模块 (Sage 项目特定实现)
// -----------------------------------------------------------------------------
pub mod configs;   // 配置管理
pub mod api;       // API 服务器
pub mod tools;     // 工具函数 (模型导出, 下载等)
pub mod utils;     // 通用工具 (错误处理, 日志, 性能监控)
pub mod quantization; // 量化推理支持

// Canonical top-level exports
pub use inference::generation as core_generation;
pub use core::kv_cache;
pub use core::model;
pub use core::multimodal;
pub use core::tokenizer;
pub use data::{MmapTextDataset, TextBatch, TextBatcher, TextDataset};
pub use inference::generation;
pub use inference::kernels;
pub use inference::lazy_load;
pub use inference::model as inference_model;
pub use training::dpo;
pub use training::lora;
pub use training::streaming;
pub use training::{probe_first_fitting_config, probe_training_step_fits};
pub use training::{train, train_dpo, train_from_cache, train_with_loaders};
pub use training::{MixedPrecisionTrainer, PrecisionConfig, PrecisionKind, QloraConfig, QloraModel};

// Legacy compatibility exports
pub use configs::config::{TrainingConfig, LRSchedulerConfig, InferenceConfig, ApiConfig};
pub use tools::export;
#[cfg(feature = "web")]
pub use tools::model_download;
pub use utils::error::*;
pub use utils::logger;
pub use utils::performance;
pub use utils::metrics;
