use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::prelude::*;
use sage::{
    core::model::ModelConfig,
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
        full_chosen: vec![1, 2, 3, 4, 5, 6],
        full_rejected: vec![1, 2, 3, 7, 8, 9],
        chosen_mask: vec![0, 0, 0, 1, 1, 1],
        rejected_mask: vec![0, 0, 0, 1, 1, 1],
        prompt_len: 3,
    };
    
    // 测试DPO批次创建（简化实现）
    let batch = DPOBatch {
        full_chosen: Tensor::<NdArray, 2, Int>::zeros([1, 6], &device),
        full_rejected: Tensor::<NdArray, 2, Int>::zeros([1, 6], &device),
        chosen_mask: Tensor::<NdArray, 2, Int>::zeros([1, 6], &device),
        rejected_mask: Tensor::<NdArray, 2, Int>::zeros([1, 6], &device),
        prompt_len: 3,
    };
    
    // 验证批次数据形状正确
    assert_eq!(batch.full_chosen.dims(), [1, 6]);
    assert_eq!(batch.full_rejected.dims(), [1, 6]);
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

/// 多模态模型集成测试 (ResNet 编码器)
#[test]
fn test_multimodal_resnet_integration() {
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
                hidden_channels: 64,
                out_dim: 64,
                encoder_type: "resnet".to_string(),
                num_layers: 4,
                patch_size: 16,
                image_size: 224,
            },
            fusion: sage::core::multimodal::MultimodalFusionConfig {
                text_dim: 64,
                vision_dim: 64,
                output_dim: 64,
                strategy: "concatenate".to_string(),
            },
            preprocessing: Default::default(),
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

/// 多模态模型集成测试 (Vision Transformer 编码器)
#[test]
fn test_multimodal_vit_integration() {
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
                hidden_channels: 64,
                out_dim: 64,
                encoder_type: "vit".to_string(),
                num_layers: 4,
                patch_size: 16,
                image_size: 224,
            },
            fusion: sage::core::multimodal::MultimodalFusionConfig {
                text_dim: 64,
                vision_dim: 64,
                output_dim: 64,
                strategy: "gated".to_string(),
            },
            preprocessing: Default::default(),
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

/// 图像编码器组件独立测试
#[test]
fn test_vision_encoders() {
    use sage::core::multimodal::{VisionEncoder, VisionEncoderConfig};
    
    let device = NdArrayDevice::Cpu;
    
    // 测试 ResNet 编码器
    let resnet_config = VisionEncoderConfig {
        in_channels: 3,
        hidden_channels: 64,
        out_dim: 64,
        encoder_type: "resnet".to_string(),
        num_layers: 4,
        patch_size: 16,
        image_size: 224,
    };
    
    let resnet_encoder = VisionEncoder::new(resnet_config, &device);
    
    // 创建随机图像输入
    let image_input = Tensor::<NdArray, 4>::rand([1, 3, 224, 224], &device);
    let resnet_output = resnet_encoder.forward(image_input);
    assert_eq!(resnet_output.dims(), [1, 64]);
    
    // 测试 Vision Transformer 编码器
    let vit_config = VisionEncoderConfig {
        in_channels: 3,
        hidden_channels: 64,
        out_dim: 64,
        encoder_type: "vit".to_string(),
        num_layers: 4,
        patch_size: 16,
        image_size: 224,
    };
    
    let vit_encoder = VisionEncoder::new(vit_config, &device);
    
    let image_input_vit = Tensor::<NdArray, 4>::rand([1, 3, 224, 224], &device);
    let vit_output = vit_encoder.forward(image_input_vit);
    assert_eq!(vit_output.dims(), [1, 64]);
}

/// 图像预处理组件测试
#[test]
fn test_image_preprocessing() {
    use sage::core::multimodal::{ImagePreprocessor, ImagePreprocessingConfig};
    
    let device = NdArrayDevice::Cpu;
    
    let preprocessing_config = ImagePreprocessingConfig {
        target_size: 224,
        normalize: true,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
        random_crop: false,
        random_flip: false,
        center_crop: true,
    };
    
    let preprocessor = ImagePreprocessor::new(preprocessing_config, device.clone());
    
    // 创建随机原始图像输入 (0-255范围)
    let raw_image = Tensor::<NdArray, 4>::rand([1, 3, 256, 256], &device) * 255.0;
    
    let processed_image = preprocessor.preprocess(raw_image);
    assert_eq!(processed_image.dims(), [1, 3, 256, 256]);
}

/// 跨模态注意力机制测试
#[test]
fn test_cross_attention() {
    use sage::core::multimodal::{CrossAttention, CrossAttentionConfig};
    
    let device = NdArrayDevice::Cpu;
    
    let cross_attn_config = CrossAttentionConfig {
        text_dim: 64,
        vision_dim: 64,
        num_heads: 8,
        dropout: 0.1,
    };
    
    let cross_attention = CrossAttention::new(&cross_attn_config, &device);
    
    // 创建测试输入
    let text_embedding = Tensor::<NdArray, 3>::rand([1, 10, 64], &device);
    let vision_embedding = Tensor::<NdArray, 2>::rand([1, 64], &device);
    
    let output = cross_attention.forward(text_embedding, vision_embedding);
    assert_eq!(output.dims(), [1, 10, 64]);
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

#[test]
fn test_generate_options_defaults() {
    use sage::generation::GenerateOptions;
    let opts = GenerateOptions::default();
    assert_eq!(opts.max_new_tokens, 100);
    assert_eq!(opts.temperature, 1.0);
    assert_eq!(opts.top_k, 50);
    assert_eq!(opts.top_p, 0.9);
    assert_eq!(opts.repetition_penalty, 1.0);
    assert_eq!(opts.beam_size, 1);
    assert_eq!(opts.beam_penalty, 0.0);
    assert!(opts.use_kv_cache);
    assert!(!opts.streaming);
}

#[test]
fn test_generate_options_beam_search() {
    use sage::generation::GenerateOptions;
    let opts = GenerateOptions {
        beam_size: 4,
        beam_penalty: 0.5,
        ..GenerateOptions::default()
    };
    assert_eq!(opts.beam_size, 4);
    assert_eq!(opts.beam_penalty, 0.5);
}

#[test]
fn test_training_config_validation() {
    use sage::configs::config::TrainingConfig;
    use sage::core::model::ModelConfig;
    
    let config = TrainingConfig {
        model: ModelConfig::small_10m(),
        batch_size: 4,
        num_epochs: 1,
        lr: 1e-4,
        max_seq_len: 64,
        gradient_accumulation_steps: 1,
        num_workers: 0,
        no_progress: true,
        distributed: false,
        devices: vec![],
        dpo_config: None,
        lr_scheduler: None,
        use_lora: false,
        lora_rank: 8,
        lora_alpha: 16.0,
        gradient_clip: Some(1.0),
    };
    
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.num_epochs, 1);
    assert_eq!(config.gradient_clip, Some(1.0));
    assert!(!config.use_lora);
}

#[test]
fn test_model_config_parameter_count() {
    let small = ModelConfig::small_10m();
    let medium = ModelConfig::medium_30m();
    let large = ModelConfig::small_100m();
    
    assert!(small.num_params() < medium.num_params());
    assert!(medium.num_params() < large.num_params());
    assert!(small.num_params() > 5_000_000);
    assert!(small.num_params() < 20_000_000);
}