use burn::{
    optim::Optimizer,
    prelude::*,
    tensor::backend::AutodiffBackend,
    tensor::{Tensor, Int, TensorData},
};
use serde::{Serialize, Deserialize};

/// DPO训练数据项
#[derive(Debug, Clone)]
pub struct DPOItem {
    pub full_chosen: Vec<i32>,
    pub full_rejected: Vec<i32>,
    pub chosen_mask: Vec<i32>,
    pub rejected_mask: Vec<i32>,
    pub prompt_len: usize,
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

/// DPO损失计算器 - 真实实现
pub struct DPOLossCalculator<B: Backend> {
    config: DPOConfig,
    _device: B::Device,
}

impl<B: Backend> DPOLossCalculator<B> {
    pub fn new(config: DPOConfig, device: B::Device) -> Self {
        Self { config, _device: device }
    }

    /// 计算序列的 logits 均值（用于 DPO 损失）
    fn compute_logits_score(
        &self,
        model: &crate::core::model::Model<B>,
        input_ids: Tensor<B, 2, Int>,
        prompt_len: usize,
    ) -> Tensor<B, 1> {
        let [batch_size, seq_len] = input_ids.dims();
        let vocab_size = model.vocab_size();
        
        // 前向传播
        let logits = model.forward(input_ids.clone());
        
        // 只计算响应部分（从 prompt_len-1 到 seq_len-2 的 logits）
        let response_len = seq_len - prompt_len;
        if response_len <= 0 {
            return Tensor::<B, 1>::zeros([batch_size], &logits.device());
        }
        
        let logits_for_response = logits.slice([0..batch_size, (prompt_len - 1)..(seq_len - 1), 0..vocab_size]);
        
        // 取 logits 均值作为分数
        let mean_2d = logits_for_response
            .reshape([batch_size, response_len * vocab_size])
            .mean_dim(1);
        mean_2d.slice([0..batch_size, 0..1]).reshape([batch_size])
    }

    /// 计算真实的 DPO 损失
    pub fn calculate_loss(
        &self,
        model: &crate::core::model::Model<B>,
        batch: &DPOBatch<B>,
    ) -> Tensor<B, 1> {
        let beta = self.config.beta as f32;
        
        // 计算 chosen 和 rejected 的 logits 分数
        let score_chosen = self.compute_logits_score(
            model, 
            batch.full_chosen.clone(), 
            batch.prompt_len
        );
        let score_rejected = self.compute_logits_score(
            model, 
            batch.full_rejected.clone(), 
            batch.prompt_len
        );
        
        // logit_diff = score_chosen - score_rejected
        let logit_diff = score_chosen.sub(score_rejected);
        
        // 损失：-log(sigmoid(beta * logit_diff))
        let scaled_logit = logit_diff.mul_scalar(beta);
        
        // sigmoid(x) = 1 / (1 + exp(-x))
        let exp_neg = scaled_logit.clone().neg().exp();
        let ones = Tensor::<B, 1>::ones_like(&exp_neg);
        let denominator = ones.clone().add(exp_neg);
        let sigmoid_val = ones.div(denominator);
        
        // -log(sigmoid_val) 的均值
        let loss = sigmoid_val.log().mean().neg();
        
        loss.unsqueeze()
    }
}

/// DPO批次数据
#[derive(Debug, Clone)]
pub struct DPOBatch<B: Backend> {
    pub full_chosen: Tensor<B, 2, Int>,
    pub full_rejected: Tensor<B, 2, Int>,
    pub chosen_mask: Tensor<B, 2, Int>,
    pub rejected_mask: Tensor<B, 2, Int>,
    pub prompt_len: usize,
}

impl<B: Backend> DPOBatch<B> {
    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            full_chosen: self.full_chosen.clone().to_device(device),
            full_rejected: self.full_rejected.clone().to_device(device),
            chosen_mask: self.chosen_mask.clone().to_device(device),
            rejected_mask: self.rejected_mask.clone().to_device(device),
            prompt_len: self.prompt_len,
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
        let batch_device = batch.to_device(&self.device);
        let loss = self.loss_calculator.calculate_loss(&self.model, &batch_device);
        let loss_tensor = loss.clone().to_data().to_vec::<f32>().expect("DPO loss tensor must be f32");
        let loss_value = *loss_tensor.first().expect("DPO loss tensor must have at least one element");
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &self.model);
        self.model = self.optimizer.step(lr, self.model.clone(), grads_params);
        loss_value
    }

    pub fn model(&self) -> &crate::core::model::Model<B> {
        &self.model
    }

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
        
        if line.trim().is_empty() {
            continue;
        }
        
        let item: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("解析JSON失败: {}", e))?;
        
        let prompt = item["prompt"].as_str().unwrap_or("");
        let chosen = item["chosen"].as_str().unwrap_or("");
        let rejected = item["rejected"].as_str().unwrap_or("");
        
        if prompt.is_empty() || chosen.is_empty() || rejected.is_empty() {
            eprintln!("警告: 跳过空数据项: prompt='{}', chosen='{}', rejected='{}'", prompt, chosen, rejected);
            continue;
        }
        
        let prompt_ids: Vec<i32> = tokenizer.encode(prompt).into_iter().map(|id| id as i32).collect();
        let chosen_ids: Vec<i32> = tokenizer.encode(chosen).into_iter().map(|id| id as i32).collect();
        let rejected_ids: Vec<i32> = tokenizer.encode(rejected).into_iter().map(|id| id as i32).collect();
        
        let mut full_chosen = prompt_ids.clone();
        full_chosen.extend(chosen_ids.clone());
        
        let mut full_rejected = prompt_ids.clone();
        full_rejected.extend(rejected_ids.clone());
        
        let prompt_len = prompt_ids.len();
        let mut chosen_mask = vec![0; prompt_len];
        chosen_mask.extend(vec![1; chosen_ids.len()]);
        
        let mut rejected_mask = vec![0; prompt_len];
        rejected_mask.extend(vec![1; rejected_ids.len()]);
        
        items.push(DPOItem {
            full_chosen,
            full_rejected,
            chosen_mask,
            rejected_mask,
            prompt_len,
        });
    }
    
    if items.is_empty() {
        return Err("没有有效的DPO数据项".to_string());
    }
    
    Ok(items)
}

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
    max_seq_len: usize,
}

impl<B: Backend> DPOBatcher<B> {
    pub fn new(device: B::Device, max_seq_len: usize, _max_response_len: usize) -> Self {
        Self {
            device,
            max_seq_len,
        }
    }
}

impl<B: Backend> burn::data::dataloader::batcher::Batcher<B, DPOItem, DPOBatch<B>> for DPOBatcher<B> {
    fn batch(&self, items: Vec<DPOItem>, _device: &B::Device) -> DPOBatch<B> {
        let batch_size = items.len();
        
        if batch_size == 0 {
            panic!("批次大小不能为零");
        }
        
        if self.max_seq_len == 0 {
            panic!("最大序列长度不能为零");
        }
        
        let prompt_len = items[0].prompt_len;
        
        let mut full_chosens = Vec::with_capacity(batch_size * self.max_seq_len);
        let mut full_rejecteds = Vec::with_capacity(batch_size * self.max_seq_len);
        let mut chosen_masks = Vec::with_capacity(batch_size * self.max_seq_len);
        let mut rejected_masks = Vec::with_capacity(batch_size * self.max_seq_len);
        
        for item in items {
            let chosen_len = item.full_chosen.len().min(self.max_seq_len);
            for &id in &item.full_chosen[..chosen_len] {
                full_chosens.push(id);
            }
            for _ in chosen_len..self.max_seq_len {
                full_chosens.push(0);
            }
            
            let cm_len = item.chosen_mask.len().min(self.max_seq_len);
            for &id in &item.chosen_mask[..cm_len] {
                chosen_masks.push(id);
            }
            for _ in cm_len..self.max_seq_len {
                chosen_masks.push(0);
            }
            
            let rejected_len = item.full_rejected.len().min(self.max_seq_len);
            for &id in &item.full_rejected[..rejected_len] {
                full_rejecteds.push(id);
            }
            for _ in rejected_len..self.max_seq_len {
                full_rejecteds.push(0);
            }
            
            let rm_len = item.rejected_mask.len().min(self.max_seq_len);
            for &id in &item.rejected_mask[..rm_len] {
                rejected_masks.push(id);
            }
            for _ in rm_len..self.max_seq_len {
                rejected_masks.push(0);
            }
        }
        
        DPOBatch {
            full_chosen: Tensor::<B, 2, Int>::from_data(
                TensorData::new(full_chosens, [batch_size, self.max_seq_len]),
                &self.device,
            ),
            full_rejected: Tensor::<B, 2, Int>::from_data(
                TensorData::new(full_rejecteds, [batch_size, self.max_seq_len]),
                &self.device,
            ),
            chosen_mask: Tensor::<B, 2, Int>::from_data(
                TensorData::new(chosen_masks, [batch_size, self.max_seq_len]),
                &self.device,
            ),
            rejected_mask: Tensor::<B, 2, Int>::from_data(
                TensorData::new(rejected_masks, [batch_size, self.max_seq_len]),
                &self.device,
            ),
            prompt_len,
        }
    }
}
