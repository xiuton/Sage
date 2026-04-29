use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::prelude::*;
use sage::model::ModelConfig;

#[test]
fn test_model_config() {
    let config = ModelConfig {
        vocab_size: 1000,
        max_seq_len: 512,
        d_model: 512,
        d_ff: 2048,
        n_layers: 6,
        n_heads: 8,
        dropout: 0.1,
        quantized: false,
        multimodal: None,
        ..Default::default()
    };
    
    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.max_seq_len, 512);
    assert_eq!(config.d_model, 512);
    assert_eq!(config.n_layers, 6);
    assert_eq!(config.n_heads, 8);
    assert_eq!(config.dropout, 0.1);
    assert!(!config.quantized);
}

#[test]
fn test_model_init() {
    let device = NdArrayDevice::Cpu;
    let config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 128,
        d_model: 64,
        d_ff: 256,
        n_layers: 2,
        n_heads: 4,
        dropout: 0.1,
        quantized: false,
        multimodal: None,
        ..Default::default()
    };
    
    let model = config.init::<NdArray>(&device);
    
    // 验证模型可以进行前向传播
    let input = Tensor::<NdArray, 2, Int>::zeros([1, 2], &device);
    let output = model.forward(input);
    
    assert_eq!(output.dims(), [1, 2, 100]);
}

#[test]
fn test_model_config_with_quantized() {
    let config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 128,
        d_model: 64,
        d_ff: 256,
        n_layers: 2,
        n_heads: 4,
        dropout: 0.1,
        quantized: true,
        multimodal: None,
        ..Default::default()
    };
    
    assert!(config.quantized);
}

#[test]
fn test_forward_with_cache() {
    let device = NdArrayDevice::Cpu;
    let config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 128,
        d_model: 64,
        d_ff: 256,
        n_layers: 2,
        n_heads: 4,
        dropout: 0.1,
        quantized: false,
        multimodal: None,
        ..Default::default()
    };
    
    let model = config.init::<NdArray>(&device);
    
    // 测试不带缓存的前向传播
    let input = Tensor::<NdArray, 2, Int>::zeros([1, 2], &device);
    let output1 = model.forward_with_cache(input, None);
    
    assert_eq!(output1.dims(), [1, 2, 100]);
}

#[test]
fn test_model_config_small_10m() {
    let config = ModelConfig::small_10m();
    
    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.max_seq_len, 256);
    assert_eq!(config.d_model, 512);
    assert_eq!(config.n_layers, 6);
    assert_eq!(config.n_heads, 8);
}

#[test]
fn test_model_config_medium_30m() {
    let config = ModelConfig::medium_30m();
    
    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.max_seq_len, 512);
    assert_eq!(config.d_model, 768);
    assert_eq!(config.n_layers, 12);
    assert_eq!(config.n_heads, 12);
}

#[test]
fn test_model_config_100m() {
    let config = ModelConfig::small_100m();
    assert_eq!(config.d_model, 1024);
    assert_eq!(config.n_layers, 16);
    assert_eq!(config.n_heads, 16);
    assert_eq!(config.max_seq_len, 1024);
    assert!(config.num_params() > 90_000_000);
}

#[test]
fn test_model_config_1b() {
    let config = ModelConfig::medium_1b();
    assert_eq!(config.d_model, 1536);
    assert_eq!(config.n_layers, 24);
    assert_eq!(config.n_heads, 24);
    assert!(config.num_params() > 600_000_000);
}

#[test]
fn test_model_config_ro_fields() {
    let config = ModelConfig::small_10m();
    assert_eq!(config.pos_encoding_type, "learned");
    assert_eq!(config.rope_theta, 10000.0);
    assert!(!config.use_moe);
    assert_eq!(config.num_experts, 8);
    assert_eq!(config.top_k_experts, 2);
}
