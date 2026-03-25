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
