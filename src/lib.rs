#![recursion_limit = "1024"]

pub mod core;
pub mod data;
pub mod inference;
pub mod training;
pub mod transformer;
pub mod configs;
pub mod api;
pub mod tools;
pub mod utils;
pub mod quantization;

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

// Legacy compatibility exports
pub use configs::config::{TrainingConfig, LRSchedulerConfig, InferenceConfig, ApiConfig};
pub use tools::export;
#[cfg(feature = "web")]
pub use tools::model_download;
pub use utils::error::*;
pub use utils::logger;
pub use utils::performance;
pub use utils::metrics;
