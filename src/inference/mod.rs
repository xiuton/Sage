//! Canonical inference layer for generation and model loading helpers.

pub mod generation;
pub mod kernels;
pub mod lazy_load;
pub mod model;

pub use generation::{
    generate,
    generate_multimodal,
    generate_quantized,
    GenerateOptions,
    GenerationState,
    ModelType,
};
pub use lazy_load::LazyModel;
