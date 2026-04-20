//! Canonical core layer for model, tokenizer, generation, KV cache, and multimodal.

pub mod kv_cache;
pub mod model;
pub mod multimodal;
pub mod tokenizer;
pub mod image_generation;
pub mod multimodal_metrics;

pub use crate::inference::generation::{GenerateOptions, GenerationState, ModelType, generate, generate_multimodal, generate_quantized};
pub use kv_cache::KVCache;
pub use model::{Model, ModelConfig};
pub use multimodal::{
    MultimodalConfig,
    MultimodalFusion,
    MultimodalFusionConfig,
    VisionEncoder,
    VisionEncoderConfig,
    ResNetVariant,
    BottleneckBlock,
    ResidualBlock,
    TransformerEncoderBlock,
    MultiHeadAttention,
    MLPBlock,
    VisionTransformer,
    DataAugmentation,
    WeightLoader,
    PretrainedWeightConfig,
};
pub use tokenizer::Tokenizer;
pub use image_generation::{
    VAEConfig,
    DiffusionConfig,
    VAE,
    VAEEncoder,
    VAEDecoder,
    UNet,
    UNetBlock,
    TimeEmbedding,
    DiffusionModel,
    ImageGenerator,
    SimpleTokenizer,
};
pub use multimodal_metrics::{
    MultimodalEvaluator,
    MultimodalMetrics,
    MetricsLogger,
};
