use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
};

/// LoRA 层
#[derive(Module, Debug)]
pub struct LoRALayer<B: Backend> {
    linear: Linear<B>,
    lora_a: Linear<B>,
    lora_b: Linear<B>,
    alpha: f64,
    rank: usize,
}

impl<B: Backend> LoRALayer<B> {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f64,
        device: &B::Device,
    ) -> Self {
        let linear = LinearConfig::new(in_features, out_features).init(device);
        
        let lora_a = LinearConfig::new(in_features, rank)
            .with_bias(false)
            .init(device);
        
        let lora_b = LinearConfig::new(rank, out_features)
            .with_bias(false)
            .init(device);
        
        Self {
            linear,
            lora_a,
            lora_b,
            alpha,
            rank,
        }
    }
    
    /// 从已有的 Linear 层创建 LoRA 层
    pub fn from_linear(linear: Linear<B>, rank: usize, alpha: f64) -> Self {
        let device = linear.weight.device();
        let in_features = linear.weight.dims()[1];
        let out_features = linear.weight.dims()[0];
        
        let lora_a = LinearConfig::new(in_features, rank)
            .with_bias(false)
            .init(&device);
        
        let lora_b = LinearConfig::new(rank, out_features)
            .with_bias(false)
            .init(&device);
        
        Self {
            linear,
            lora_a,
            lora_b,
            alpha,
            rank,
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let original_output = self.linear.forward(input.clone());
        
        let lora_output = self.lora_b.forward(self.lora_a.forward(input));
        let scaled_lora_output = lora_output * self.alpha / self.rank as f64;
        
        original_output + scaled_lora_output
    }
    
    /// 冻结原始 Linear 层（只训练 LoRA 部分）
    pub fn freeze_original(&self) {
        // 在实际 Burn 使用中，通常是通过不计算这些参数的梯度来实现
        // 这里作为示例框架
    }
    
    /// 合并 LoRA 权重到原始 Linear 层
    pub fn merge_weights(&self) -> Linear<B> {
        // 简单实现：返回原始 Linear 层
        // 真实合并需要计算 W = W0 + (alpha/r) * B * A
        self.linear.clone()
    }
    
    /// 获取可训练参数（仅 LoRA 部分）
    pub fn trainable_params(&self) -> Vec<String> {
        vec!["lora_a".to_string(), "lora_b".to_string()]
    }
}

#[derive(Debug, Clone)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f64,
    pub target_modules: Vec<String>,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

#[derive(Debug, Clone)]
pub struct QLoRAConfig {
    pub lora_config: LoRAConfig,
    pub quant_bit: usize,
    pub quant_type: QuantType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantType {
    Int8,
    NF4,
    FP4,
}

impl Default for QLoRAConfig {
    fn default() -> Self {
        Self {
            lora_config: LoRAConfig::default(),
            quant_bit: 4,
            quant_type: QuantType::NF4,
        }
    }
}

/// LoRA 训练辅助工具
pub mod utils {
    /// 计算 LoRA 可训练参数数量
    pub fn calculate_lora_params(
        in_features: usize,
        out_features: usize,
        rank: usize,
        num_layers: usize,
    ) -> usize {
        let lora_params_per_layer = (in_features * rank) + (rank * out_features);
        lora_params_per_layer * num_layers
    }
    
    /// 计算完整模型参数数量（用于对比）
    pub fn calculate_full_params(
        in_features: usize,
        out_features: usize,
        num_layers: usize,
    ) -> usize {
        let full_params_per_layer = in_features * out_features;
        full_params_per_layer * num_layers
    }
}
