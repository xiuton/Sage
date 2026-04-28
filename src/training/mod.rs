//! Canonical training layer for optimization, schedulers, and runtime helpers.

pub mod distributed;
pub mod dpo;
pub mod lora;
pub mod lr_scheduler;
pub mod precision;
pub mod qlora;
pub mod streaming;
pub mod training;
pub mod vram_probe;

pub use dpo::{load_dpo_jsonl, DPOBatcher, DPOConfig, DPOItem, DPOTrainer};
pub use lr_scheduler::LRScheduler;
pub use precision::{MixedPrecisionTrainer, PrecisionConfig, PrecisionKind};
pub use qlora::{QloraConfig, QloraModel};
pub use streaming::{SftInput, StreamingSftDataLoader};
pub use training::{train, train_dpo, train_from_cache, train_with_loaders};
pub use vram_probe::{probe_first_fitting_config, probe_training_step_fits};
