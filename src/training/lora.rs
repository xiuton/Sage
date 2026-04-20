use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
};
use serde::{Deserialize, Serialize};

/// LoRA 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f64,
    pub target_modules: Vec<String>, // 例如 ["q_proj", "v_proj", "output_head"]
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: vec!["output_head".to_string()],
        }
    }
}

/// LoRA 层：包装一个原始 Linear 层并添加低秩矩阵 A 和 B
#[derive(Module, Debug)]
pub struct LoRALinear<B: Backend> {
    pub linear: Linear<B>,
    pub lora_a: Linear<B>,
    pub lora_b: Linear<B>,
    pub alpha: f64,
    pub rank: usize,
    pub enabled: bool,
}

impl<B: Backend> LoRALinear<B> {
    pub fn new(
        linear: Linear<B>,
        rank: usize,
        alpha: f64,
        device: &B::Device,
    ) -> Self {
        let in_features = linear.weight.dims()[1];
        let out_features = linear.weight.dims()[0];
        
        // LoRA A 使用正态分布初始化 (通常是 Kaiming uniform/normal)
        let lora_a = LinearConfig::new(in_features, rank)
            .with_bias(false)
            .init(device);
        
        // LoRA B 初始化为全 0，确保训练开始时 LoRA 部分输出为 0，不影响原始权重
        let lora_b_config = LinearConfig::new(rank, out_features).with_bias(false);
        let lora_b = lora_b_config.init(device);
        
        Self {
            linear,
            lora_a,
            lora_b,
            alpha,
            rank,
            enabled: true,
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let original_output = self.linear.forward(input.clone());
        
        if !self.enabled {
            return original_output;
        }

        // LoRA path: (input * A) * B * (alpha / rank)
        let lora_output = self.lora_b.forward(self.lora_a.forward(input));
        let scaled_lora_output = lora_output * (self.alpha / self.rank as f64);
        
        original_output + scaled_lora_output
    }
    
    /// 合并 LoRA 权重到原始 Linear 层并返回新的 Linear 层
    pub fn merge_weights(&self) -> Linear<B> {
        let device = self.linear.weight.device();
        let wa = self.lora_a.weight.val(); // [rank, in_features]
        let wb = self.lora_b.weight.val(); // [out_features, rank]
        
        // W_lora = (B * A) * (alpha / r)
        let w_lora = wb.matmul(wa) * (self.alpha / self.rank as f64);
        
        // 创建一个新的 Linear 层，权重为原始权重加上 LoRA 权重
        let merged_weight = self.linear.weight.val() + w_lora;
        let merged_linear = LinearConfig::new(merged_weight.dims()[1], merged_weight.dims()[0])
            .with_bias(self.linear.bias.is_some())
            .init(&device);
        
        merged_linear
    }
}
