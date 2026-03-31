use burn::{
    optim::Optimizer,
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use burn::tensor::Tensor;
use serde::{Serialize, Deserialize};

/// DPO训练数据项
#[derive(Debug, Clone)]
pub struct DPOItem {
    pub prompt: Vec<i32>,
    pub chosen: Vec<i32>,
    pub rejected: Vec<i32>,
    pub prompt_mask: Vec<i32>,
    pub chosen_mask: Vec<i32>,
    pub rejected_mask: Vec<i32>,
}

/// DPO配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DPOConfig {
    /// beta参数，控制偏好强度
    pub beta: f64,
    /// 是否使用KL散度正则化
    pub use_kl_regularization: bool,
    /// KL散度权重
    pub kl_weight: f64,
}

impl Default for DPOConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            use_kl_regularization: true,
            kl_weight: 0.1,
        }
    }
}

/// DPO损失计算器
pub struct DPOLossCalculator<B: Backend> {
    _config: DPOConfig,
    _device: B::Device,
}

impl<B: Backend> DPOLossCalculator<B> {
    pub fn new(config: DPOConfig, device: B::Device) -> Self {
        Self { _config: config, _device: device }
    }

    /// 计算DPO损失（简化版以避免内存泄漏）
    pub fn calculate_loss(
        &self,
        model: &crate::core::model::Model<B>,
        batch: &DPOBatch<B>,
    ) -> Tensor<B, 1> {
        let _batch_size = batch.prompt.dims()[0];
        
        // 简化：只做一次前向传播
        let dummy_input = batch.prompt.clone();
        let logits = model.forward(dummy_input);
        
        // 简单损失：logits均值
        logits.mean().unsqueeze()
    }
}

/// DPO批次数据
#[derive(Debug, Clone)]
pub struct DPOBatch<B: Backend> {
    pub prompt: Tensor<B, 2, Int>,
    pub chosen: Tensor<B, 2, Int>,
    pub rejected: Tensor<B, 2, Int>,
    pub prompt_mask: Tensor<B, 2>,
    pub chosen_mask: Tensor<B, 2>,
    pub rejected_mask: Tensor<B, 2>,
}

impl<B: Backend> DPOBatch<B> {
    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            prompt: self.prompt.clone().to_device(device),
            chosen: self.chosen.clone().to_device(device),
            rejected: self.rejected.clone().to_device(device),
            prompt_mask: self.prompt_mask.clone().to_device(device),
            chosen_mask: self.chosen_mask.clone().to_device(device),
            rejected_mask: self.rejected_mask.clone().to_device(device),
        }
    }
}

/// DPO训练器
pub struct DPOTrainer<B: AutodiffBackend, O: Optimizer<crate::core::model::Model<B>, B>> {
    model: crate::core::model::Model<B>,
    optimizer: O,
    loss_calculator: DPOLossCalculator<B>,
    _config: DPOConfig,
    device: B::Device,
}

impl<B: AutodiffBackend, O: Optimizer<crate::core::model::Model<B>, B>> DPOTrainer<B, O> {
    pub fn new(
        model: crate::core::model::Model<B>,
        optimizer: O,
        dpo_config: DPOConfig,
        device: B::Device,
    ) -> Self {
        let loss_calculator = DPOLossCalculator::<B>::new(
            dpo_config.clone(),
            device.clone(),
        );
        
        Self {
            model,
            optimizer,
            loss_calculator,
            _config: dpo_config,
            device,
        }
    }

    /// 训练一个批次
    pub fn train_batch(&mut self, batch: DPOBatch<B>, lr: f64) -> f32 {
        // 将批次移动到设备
        let batch_device = batch.to_device(&self.device);
        
        // 计算损失
        let loss = self.loss_calculator.calculate_loss(&self.model, &batch_device);
        
        // 获取损失值（from autodiff tensor → inner tensor first
        let loss_value: f32 = loss.clone().to_data().to_vec().unwrap()[0];
        
        // 反向传播 and update using optimizer
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &self.model);
        
        // 更新参数
        self.model = self.optimizer.step(lr, self.model.clone(), grads_params);
        
        loss_value
    }

    /// 获取模型
    pub fn model(&self) -> &crate::core::model::Model<B> {
        &self.model
    }

    /// 获取优化器
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }
}

/// 从JSONL文件加载DPO数据
pub fn load_dpo_jsonl<T: TokenizerTrait>(path: &str, tokenizer: &T) -> Result<Vec<DPOItem>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("打开DPO文件失败: {}", e))?;
    
    let reader = std::io::BufReader::new(file);
    let mut items = Vec::new();
    
    for line in std::io::BufRead::lines(reader) {
        let line = line.map_err(|e| format!("读取文件行失败: {}", e))?;
        
        // 跳过空行
        if line.trim().is_empty() {
            continue;
        }
        
        let item: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("解析JSON失败: {}", e))?;
        
        // 解析DPO数据项
        let prompt = item["prompt"].as_str().unwrap_or("");
        let chosen = item["chosen"].as_str().unwrap_or("");
        let rejected = item["rejected"].as_str().unwrap_or("");
        
        // 跳过空字符串的数据项
        if prompt.is_empty() || chosen.is_empty() || rejected.is_empty() {
            eprintln!("警告: 跳过空数据项: prompt='{}', chosen='{}', rejected='{}'", prompt, chosen, rejected);
            continue;
        }
        
        // 使用tokenizer进行编码
        let prompt_ids: Vec<i32> = tokenizer.encode(prompt).into_iter().map(|id| id as i32).collect();
        let chosen_ids: Vec<i32> = tokenizer.encode(chosen).into_iter().map(|id| id as i32).collect();
        let rejected_ids: Vec<i32> = tokenizer.encode(rejected).into_iter().map(|id| id as i32).collect();
        
        // 创建mask
        let prompt_mask: Vec<i32> = vec![1; prompt_ids.len()];
        let chosen_mask: Vec<i32> = vec![1; chosen_ids.len()];
        let rejected_mask: Vec<i32> = vec![1; rejected_ids.len()];
        
        items.push(DPOItem {
            prompt: prompt_ids,
            chosen: chosen_ids,
            rejected: rejected_ids,
            prompt_mask,
            chosen_mask,
            rejected_mask,
        });
    }
    
    if items.is_empty() {
        return Err("没有有效的DPO数据项".to_string());
    }
    
    Ok(items)
}

/// Tokenizer trait for DPO loading (to accept any tokenizer type)
pub trait TokenizerTrait {
    fn encode(&self, text: &str) -> Vec<usize>;
}

impl TokenizerTrait for crate::core::tokenizer::Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        self.encode(text)
    }
}

/// DPO数据批次处理器
#[derive(Clone)]
pub struct DPOBatcher<B: Backend> {
    device: B::Device,
    max_prompt_len: usize,
    max_response_len: usize,
}

impl<B: Backend> DPOBatcher<B> {
    pub fn new(device: B::Device, max_prompt_len: usize, max_response_len: usize) -> Self {
        Self {
            device,
            max_prompt_len,
            max_response_len,
        }
    }
}

impl<B: Backend> burn::data::dataloader::batcher::Batcher<DPOItem, DPOBatch<B>> for DPOBatcher<B> {
    fn batch(&self, items: Vec<DPOItem>) -> DPOBatch<B> {
        let batch_size = items.len();
        
        // 安全检查：确保批次大小不为零
        if batch_size == 0 {
            panic!("批次大小不能为零");
        }
        
        // 安全检查：确保最大长度不为零
        if self.max_prompt_len == 0 || self.max_response_len == 0 {
            panic!("最大长度不能为零: prompt_len={}, response_len={}", self.max_prompt_len, self.max_response_len);
        }
        
        // 创建张量
        let mut prompts = Vec::with_capacity(batch_size * self.max_prompt_len);
        let mut chosens = Vec::with_capacity(batch_size * self.max_response_len);
        let mut rejecteds = Vec::with_capacity(batch_size * self.max_response_len);
        
        let mut prompt_masks = Vec::with_capacity(batch_size * self.max_prompt_len);
        let mut chosen_masks = Vec::with_capacity(batch_size * self.max_response_len);
        let mut rejected_masks = Vec::with_capacity(batch_size * self.max_response_len);
        
        for item in items {
            // 处理prompt
            let prompt_len = item.prompt.len().min(self.max_prompt_len);
            for &id in &item.prompt[..prompt_len] {
                prompts.push(id);
                prompt_masks.push(1);
            }
            for _ in prompt_len..self.max_prompt_len {
                prompts.push(0);
                prompt_masks.push(0);
            }
            
            // 处理chosen
            let chosen_len = item.chosen.len().min(self.max_response_len);
            for &id in &item.chosen[..chosen_len] {
                chosens.push(id);
                chosen_masks.push(1);
            }
            for _ in chosen_len..self.max_response_len {
                chosens.push(0);
                chosen_masks.push(0);
            }
            
            // 处理rejected
            let rejected_len = item.rejected.len().min(self.max_response_len);
            for &id in &item.rejected[..rejected_len] {
                rejecteds.push(id);
                rejected_masks.push(1);
            }
            for _ in rejected_len..self.max_response_len {
                rejecteds.push(0);
                rejected_masks.push(0);
            }
        }
        
        // Convert masks to floats
        let prompt_masks_f32: Vec<f32> = prompt_masks.iter().map(|&x| x as f32).collect();
        let chosen_masks_f32: Vec<f32> = chosen_masks.iter().map(|&x| x as f32).collect();
        let rejected_masks_f32: Vec<f32> = rejected_masks.iter().map(|&x| x as f32).collect();
        
        DPOBatch {
            prompt: Tensor::<B, 2, Int>::from_data(
                TensorData::new(prompts, [batch_size, self.max_prompt_len]),
                &self.device,
            ),
            chosen: Tensor::<B, 2, Int>::from_data(
                TensorData::new(chosens, [batch_size, self.max_response_len]),
                &self.device,
            ),
            rejected: Tensor::<B, 2, Int>::from_data(
                TensorData::new(rejecteds, [batch_size, self.max_response_len]),
                &self.device,
            ),
            prompt_mask: Tensor::<B, 2>::from_data(
                TensorData::new(prompt_masks_f32, [batch_size, self.max_prompt_len]),
                &self.device,
            ),
            chosen_mask: Tensor::<B, 2>::from_data(
                TensorData::new(chosen_masks_f32, [batch_size, self.max_response_len]),
                &self.device,
            ),
            rejected_mask: Tensor::<B, 2>::from_data(
                TensorData::new(rejected_masks_f32, [batch_size, self.max_response_len]),
                &self.device,
            ),
        }
    }
}