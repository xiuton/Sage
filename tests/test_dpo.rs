use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::optim::AdamConfig;
use burn_autodiff::Autodiff;
use sage::core::model::ModelConfig;
use sage::training::dpo::{DPOConfig, DPOItem, DPOBatcher, DPOLossCalculator, DPOTrainer};
use burn::data::dataloader::batcher::Batcher;

#[test]
fn test_dpo_loss_calculator() {
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
    };
    
    let model = config.init::<Autodiff<NdArray>>(&device);
    let dpo_config = DPOConfig::default();
    let loss_calculator = DPOLossCalculator::<Autodiff<NdArray>>::new(dpo_config, device.clone());
    
    // 创建简单的测试数据
    let prompt_len = 4;
    let prompt = vec![1, 2, 3, 4];
    let chosen = vec![5, 6, 7];
    let rejected = vec![8, 9, 10];
    
    let mut full_chosen = prompt.clone();
    full_chosen.extend(chosen.clone());
    
    let mut full_rejected = prompt.clone();
    full_rejected.extend(rejected.clone());
    
    let mut chosen_mask = vec![0; prompt_len];
    chosen_mask.extend(vec![1; chosen.len()]);
    
    let mut rejected_mask = vec![0; prompt_len];
    rejected_mask.extend(vec![1; rejected.len()]);
    
    let item = DPOItem {
        full_chosen,
        full_rejected,
        chosen_mask,
        rejected_mask,
        prompt_len,
    };
    
    let batcher = DPOBatcher::<Autodiff<NdArray>>::new(device.clone(), 128, 64);
    let batch = batcher.batch(vec![item], &device);
    
    // 计算损失
    let loss = loss_calculator.calculate_loss(&model, &batch);
    
    // 验证损失是有限值
    let loss_value: f32 = loss.to_data().to_vec().unwrap()[0];
    assert!(!loss_value.is_nan());
    assert!(!loss_value.is_infinite());
    println!("DPO 损失测试通过！损失值: {}", loss_value);
}

#[test]
fn test_dpo_trainer() {
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
    };
    
    let model = config.init::<Autodiff<NdArray>>(&device);
    let optimizer_config = AdamConfig::new();
    let optimizer = optimizer_config.init();
    let dpo_config = DPOConfig::default();
    
    let mut trainer = DPOTrainer::new(
        model,
        optimizer,
        dpo_config,
        device.clone(),
    );
    
    // 创建测试数据
    let prompt_len = 4;
    let prompt = vec![1, 2, 3, 4];
    let chosen = vec![5, 6, 7];
    let rejected = vec![8, 9, 10];
    
    let mut full_chosen = prompt.clone();
    full_chosen.extend(chosen.clone());
    
    let mut full_rejected = prompt.clone();
    full_rejected.extend(rejected.clone());
    
    let mut chosen_mask = vec![0; prompt_len];
    chosen_mask.extend(vec![1; chosen.len()]);
    
    let mut rejected_mask = vec![0; prompt_len];
    rejected_mask.extend(vec![1; rejected.len()]);
    
    let item = DPOItem {
        full_chosen,
        full_rejected,
        chosen_mask,
        rejected_mask,
        prompt_len,
    };
    
    let batcher = DPOBatcher::<Autodiff<NdArray>>::new(device.clone(), 128, 64);
    let batch = batcher.batch(vec![item], &device);
    
    // 训练一步
    let loss_value = trainer.train_batch(batch, 0.001);
    
    // 验证损失是有限值
    assert!(!loss_value.is_nan());
    assert!(!loss_value.is_infinite());
    println!("DPO 训练器测试通过！训练损失: {}", loss_value);
}
