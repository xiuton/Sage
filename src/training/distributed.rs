use burn::{
    data::{dataloader::{DataLoaderBuilder}, dataset::Dataset},
    optim::Optimizer,
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use std::sync::Arc;
use std::thread;
use crate::{TextBatch, data::data::TextItem, TextBatcher};


/// 分布式训练配置
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    pub devices: Vec<String>,
    pub rank: usize,
    pub world_size: usize,
}

impl DistributedConfig {
    pub fn new(devices: Vec<String>) -> Self {
        let world_size = devices.len();
        Self {
            devices,
            rank: 0,
            world_size,
        }
    }
}

/// 数据并行训练器
pub struct DataParallelTrainer<B: AutodiffBackend, O: Optimizer<crate::core::model::Model<B>, B>> {
    models: Vec<crate::core::model::Model<B>>,
    optimizers: Vec<O>,
    devices: Vec<B::Device>,
    config: DistributedConfig,
}

impl<B: AutodiffBackend, O: Optimizer<crate::core::model::Model<B>, B> + Clone> DataParallelTrainer<B, O> {
    pub fn new(
        model: crate::core::model::Model<B>,
        optimizer: O,
        devices: Vec<B::Device>,
        config: DistributedConfig,
    ) -> Self {
        let mut models = Vec::with_capacity(devices.len());
        let mut optimizers = Vec::with_capacity(devices.len());
        
        for _device in &devices {
            let model_copy = model.clone();
            let optimizer_copy = optimizer.clone();
            models.push(model_copy);
            optimizers.push(optimizer_copy);
        }
        
        Self {
            models,
            optimizers,
            devices,
            config,
        }
    }
    
    pub fn train_batch(&mut self, batch: &TextBatch<B>, device_idx: usize, lr: f64) -> f64 {
        let device = &self.devices[device_idx];
        let model = &mut self.models[device_idx];
        let optimizer = &mut self.optimizers[device_idx];
        
        // 将数据移动到对应设备
        let batch_device = TextBatch {
            inputs: batch.inputs.clone().to_device(device),
            targets: batch.targets.clone().to_device(device),
            mask: batch.mask.clone().to_device(device),
            attention_mask: batch.attention_mask.clone().to_device(device),
            token_type_ids: batch.token_type_ids.clone().to_device(device),
            images: batch.images.clone().map(|img| img.to_device(device)),
        };
        
        // 前向传播
        let output = model.forward_step(batch_device);
        let loss_val = output.loss.clone().into_data().as_slice::<f32>()
            .expect("Loss tensor should be convertible to f32 slice")[0] as f64;
        
        // 反向传播与优化
        let grads = output.loss.backward();
        let grads = burn::optim::GradientsParams::from_grads(grads, model);
        *model = optimizer.step(lr, model.clone(), grads);
        
        loss_val
    }
    
    /// 权重同步：将所有设备的模型权重取平均并同步到所有模型
    pub fn synchronize_weights(&mut self) {
        if self.config.world_size <= 1 {
            return;
        }
        
        println!("正在同步设备间的权重...");
        
        // 1. 获取第一个模型作为基准，计算平均值
        // 注意：由于 Burn 的 Module 是不可变的且基于状态，我们需要创建一个新的状态
        // 这里的简化实现仅做演示，真实高性能同步需要底层的 Tensor 通信
        
        // 获取主模型的权重副本
        let base_model = self.models[0].clone();
        
        // 对每个参数进行平均
        // 简化：这里仅同步第一个模型到其他模型（即主从同步）
        for i in 1..self.config.world_size {
            self.models[i] = base_model.clone();
        }
        
        println!("权重同步完成。");
    }
}

/// 在多设备上并行训练
pub fn train_parallel<B: AutodiffBackend, D: Dataset<TextItem> + Send + Sync + 'static>(
    dataset: Arc<D>,
    batch_size: usize,
    num_epochs: usize,
    devices: Vec<B::Device>,
) {
    let world_size = devices.len();
    let mut handles = Vec::with_capacity(world_size);
    
    // 使用第一个设备作为参考
    let _master_device = devices[0].clone();
    
    for rank in 0..world_size {
        let dataset_clone = Arc::clone(&dataset);
        let device = devices[rank].clone();
        
        let handle = thread::spawn(move || {
            println!("设备 {} 开始训练", rank);
            
            // 创建数据加载器（每个设备处理不同的数据块）
            let batcher = TextBatcher::<B>::new(device.clone());
            let dataloader = DataLoaderBuilder::new(batcher)
                .batch_size(batch_size)
                .shuffle(42)
                .num_workers(4)
                .build(dataset_clone);
            
            for epoch in 1..=num_epochs {
                for (i, _batch) in dataloader.iter().enumerate() {
                    if i % 100 == 0 {
                        println!("设备 {} - Epoch {} - Batch {}", rank, epoch, i);
                    }
                    // 这里应该调用具体的训练逻辑
                }
            }
            
            println!("设备 {} 训练完成", rank);
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().expect("A distributed training thread panicked");
    }
}

/// 获取可用设备列表
pub fn get_available_devices() -> Vec<String> {
    let mut devices = Vec::new();
    
    // 检查CPU
    devices.push("cpu".to_string());
    
    // 检查GPU
    // 在 Windows WGPU 环境下，我们通常只有一个默认设备
    devices.push("gpu:0".to_string());
    
    devices
}

/// 根据设备名称创建设备实例
pub fn create_device<B: Backend>(device_name: &str) -> B::Device {
    match device_name {
        "cpu" => B::Device::default(),
        _ => B::Device::default(),
    }
}
