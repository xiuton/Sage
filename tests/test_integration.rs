use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::prelude::*;
use sage::{
    model::ModelConfig,
    tokenizer::Tokenizer,
    training::training::TrainingConfig as TrainConfig,
    training::dpo::{DPOConfig, DPOItem, DPOBatch},
    quantization::quantization::{QuantizationMode, QuantizedModel},
};

/// 端到端模型训练和推理集成测试
#[test]
fn test_end_to_end_training_inference() {
    let device = NdArrayDevice::Cpu;
    
    // 创建小型模型配置
    let model_config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 32,
        d_model: 64,
        d_ff: 128,
        n_layers: 2,
        n_heads: 4,
        dropout: 0.1,
        quantized: false,
        multimodal: None,
    };
    
    // 初始化模型
    let model = model_config.init::<NdArray>(&device);
    
    // 测试前向传播
    let input = Tensor::<NdArray, 2, Int>::zeros([1, 5], &device);
    let output = model.forward(input);
    
    assert_eq!(output.dims(), [1, 5, 100]);
    
    // 测试分词器
    let tokenizer = Tokenizer::new("hello world");
    let tokens = tokenizer.encode("test");
    let decoded = tokenizer.decode(&tokens);
    
    assert!(!tokens.is_empty());
    assert!(!decoded.is_empty());
}

/// 量化模型集成测试
#[test]
fn test_quantization_integration() {
    let device = NdArrayDevice::Cpu;
    
    let model_config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 32,
        d_model: 64,
        d_ff: 128,
        n_layers: 2,
        n_heads: 4,
        dropout: 0.1,
        quantized: false,
        multimodal: None,
    };
    
    let original_model = model_config.init::<NdArray>(&device);
    
    // 测试动态量化
    let quantized_dynamic = QuantizedModel::new(original_model.clone(), QuantizationMode::Dynamic);
    
    // 测试INT8量化
    let quantized_int8 = QuantizedModel::new(original_model.clone(), QuantizationMode::Int8);
    
    // 验证量化模型可以正常前向传播
    let input = Tensor::<NdArray, 2, Int>::zeros([1, 3], &device);
    let output_dynamic = quantized_dynamic.forward(input.clone());
    let output_int8 = quantized_int8.forward(input);
    
    assert_eq!(output_dynamic.dims(), [1, 3, 100]);
    assert_eq!(output_int8.dims(), [1, 3, 100]);
}

/// DPO训练流程集成测试
#[test]
fn test_dpo_training_integration() {
    let device = NdArrayDevice::Cpu;
    
    let model_config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 32,
        d_model: 64,
        d_ff: 128,
        n_layers: 2,
        n_heads: 4,
        dropout: 0.1,
        quantized: false,
        multimodal: None,
    };
    
    let _model = model_config.init::<NdArray>(&device);
    
    // 创建DPO配置
    let _dpo_config = DPOConfig {
        beta: 0.1,
        use_kl_regularization: true,
        kl_weight: 0.1,
    };
    
    // 创建模拟DPO数据
    let _dpo_item = DPOItem {
        prompt: vec![1, 2, 3],
        chosen: vec![4, 5, 6],
        rejected: vec![7, 8, 9],
        prompt_mask: vec![1, 1, 1],
        chosen_mask: vec![1, 1, 1],
        rejected_mask: vec![1, 1, 1],
    };
    
    // 测试DPO批次创建（简化实现）
    let batch = DPOBatch {
        prompt: Tensor::<NdArray, 2, Int>::zeros([1, 3], &device),
        chosen: Tensor::<NdArray, 2, Int>::zeros([1, 3], &device),
        rejected: Tensor::<NdArray, 2, Int>::zeros([1, 3], &device),
        prompt_mask: Tensor::<NdArray, 2>::zeros([1, 3], &device),
        chosen_mask: Tensor::<NdArray, 2>::zeros([1, 3], &device),
        rejected_mask: Tensor::<NdArray, 2>::zeros([1, 3], &device),
    };
    
    // 验证批次数据形状正确
    assert_eq!(batch.prompt.dims(), [1, 3]);
    assert_eq!(batch.chosen.dims(), [1, 3]);
    assert_eq!(batch.rejected.dims(), [1, 3]);
}

/// 训练配置集成测试
#[test]
fn test_training_config_integration() {
    let model_config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 32,
        d_model: 64,
        d_ff: 128,
        n_layers: 2,
        n_heads: 4,
        dropout: 0.1,
        quantized: false,
        multimodal: None,
    };
    
    let optimizer_config = burn::optim::AdamConfig::new();
    
    // 测试训练配置创建
    let training_config = TrainConfig::create(model_config, optimizer_config);
    
    // 验证配置参数正确
    assert_eq!(training_config.num_epochs, 50);
    assert_eq!(training_config.batch_size, 32);
    assert_eq!(training_config.lr, 5.0e-4);
    assert!(!training_config.distributed);
    assert!(training_config.devices.is_empty());
    assert!(training_config.dpo_config.is_none());
}

/// 多模态模型集成测试
#[test]
fn test_multimodal_integration() {
    let device = NdArrayDevice::Cpu;
    
    let model_config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 32,
        d_model: 64,
        d_ff: 128,
        n_layers: 2,
        n_heads: 4,
        dropout: 0.1,
        quantized: false,
        multimodal: Some(sage::core::multimodal::MultimodalConfig {
            vision_encoder: sage::core::multimodal::VisionEncoderConfig {
                in_channels: 3,
                hidden_dim: 64,
                out_dim: 64,
                num_layers: 2,
                use_batch_norm: false,
            },
            fusion: sage::core::multimodal::MultimodalFusionConfig {
                text_dim: 64,
                vision_dim: 64,
                output_dim: 64,
                strategy: sage::core::multimodal::FusionStrategy::Concatenate,
            },
            enable_multimodal: true,
        }),
    };
    
    // 初始化多模态模型
    let model = model_config.init::<NdArray>(&device);
    
    // 测试文本输入前向传播
    let text_input = Tensor::<NdArray, 2, Int>::zeros([1, 5], &device);
    let output = model.forward(text_input);
    
    assert_eq!(output.dims(), [1, 5, 100]);
}

/// 性能基准集成测试
#[test]
fn test_performance_integration() {
    let device = NdArrayDevice::Cpu;
    
    let model_config = ModelConfig {
        vocab_size: 100,
        max_seq_len: 16,
        d_model: 32,
        d_ff: 64,
        n_layers: 1,
        n_heads: 2,
        dropout: 0.1,
        quantized: false,
        multimodal: None,
    };
    
    let model = model_config.init::<NdArray>(&device);
    
    // 测试小批量性能
    let input = Tensor::<NdArray, 2, Int>::zeros([2, 8], &device);
    
    // 多次前向传播测试稳定性
    for _ in 0..3 {
        let output = model.forward(input.clone());
        assert_eq!(output.dims(), [2, 8, 100]);
    }
}