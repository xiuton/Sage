use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
};

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
    
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let original_output = self.linear.forward(input.clone());
        
        let lora_output = self.lora_b.forward(self.lora_a.forward(input));
        let scaled_lora_output = lora_output * self.alpha / self.rank as f64;
        
        original_output + scaled_lora_output
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

#[derive(Debug, Clone)]
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
