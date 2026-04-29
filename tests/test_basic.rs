use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::prelude::*;
use sage::{model::ModelConfig, tokenizer::Tokenizer};

#[test]
fn test_model_initialization() {
    let device = NdArrayDevice::Cpu;
    
    let config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 16,
        d_model: 32,
        d_ff: 128,
        n_layers: 1,
        n_heads: 2,
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
fn test_tokenizer_creation() {
    let tokenizer = Tokenizer::new("");
    
    // 测试编码和解码
    let text = "测试文本";
    let ids = tokenizer.encode(text);
    let decoded = tokenizer.decode(&ids);
    
    assert!(!ids.is_empty());
    assert!(!decoded.is_empty());
}

#[test]
fn test_model_config_presets() {
    let small_config = ModelConfig::small_10m();
    assert_eq!(small_config.vocab_size, 1000);
    assert_eq!(small_config.max_seq_len, 256);
    
    let medium_config = ModelConfig::medium_30m();
    assert_eq!(medium_config.vocab_size, 1000);
    assert_eq!(medium_config.max_seq_len, 512);
}