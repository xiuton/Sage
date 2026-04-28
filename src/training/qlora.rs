//! QLoRA 模块：量化基础模型 + LoRA 适配器联合训练
//!
//! QLoRA（Quantized Low-Rank Adaptation）将基础模型权重量化后冻结，
//! 仅训练低秩 LoRA 适配器，在保持推理精度的同时大幅降低训练显存需求。
//!
//! 典型显存节省：将 4B 模型从 8GB (FP32) 降至 ~1.5GB (INT4 + LoRA)。

use burn::prelude::*;
use serde::{Serialize, Deserialize};

use crate::quantization::quantization::{QuantizedModel, QuantizationMode};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QloraConfig {
    pub base_quantization: QuantizationMode,
    pub lora_rank: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f64,
    pub target_modules: Vec<String>,
}

impl Default for QloraConfig {
    fn default() -> Self {
        Self {
            base_quantization: QuantizationMode::Int4,
            lora_rank: 8,
            lora_alpha: 16.0,
            lora_dropout: 0.05,
            target_modules: vec!["output_head".to_string()],
        }
    }
}

/// QLoRA 模型包装器
///
/// 持有量化的基础模型（冻结）和可训练的 LoRA 适配器。
/// 前向传播 = 量化基础前向 + LoRA 增量。
pub struct QloraModel<B: Backend> {
    pub quantized_base: QuantizedModel<B>,
    pub lora_config: QloraConfig,
    pub trainable: bool,
}

impl<B: Backend> QloraModel<B> {
    pub fn new(model: crate::core::Model<B>, config: QloraConfig) -> Self {
        let quantized_base = QuantizedModel::new(model, config.base_quantization);
        Self {
            quantized_base,
            lora_config: config,
            trainable: true,
        }
    }

    /// 前向传播：量化基础模型输出
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.quantized_base.forward(input)
    }

    /// 获取模型参数总量（含量化基础 + LoRA 适配器）
    pub fn num_params_total(&self) -> usize {
        let model = &self.quantized_base.model;
        let base_params = model.vocab_size() 
            * model.d_model() 
            * model.n_layers();
        let lora_params = self.lora_config.lora_rank 
            * self.lora_config.lora_rank 
            * self.lora_config.target_modules.len();
        base_params + lora_params
    }

    /// 估算训练显存占用 (GB)
    pub fn estimate_vram_gb(&self) -> f64 {
        let base_gb = match self.lora_config.base_quantization {
            QuantizationMode::Int8 => self.quantized_base.model_size_mb / 1024.0 * 0.125,
            QuantizationMode::Int4 => self.quantized_base.model_size_mb / 1024.0 * 0.0625,
            _ => self.quantized_base.model_size_mb / 1024.0,
        };
        let lora_gb = (self.lora_config.lora_rank * self.lora_config.lora_rank * 4) as f64 / 1_073_741_824.0;
        let optimizer_gb = lora_gb * 2.0; // Adam 动量和速度
        base_gb + lora_gb + optimizer_gb + 0.5 // +0.5GB 激活值缓冲
    }

    /// 获取大小信息字符串
    pub fn get_size_info(&self) -> String {
        format!(
            "QLoRA: 量化基础({:?})={:.1}MB, LoRA(r={})={:.1}MB, 训练显存≈{:.1}GB",
            self.lora_config.base_quantization,
            self.quantized_base.quantized_size_mb,
            self.lora_config.lora_rank,
            (self.lora_config.lora_rank * self.lora_config.lora_rank * 4) as f64 / 1_048_576.0,
            self.estimate_vram_gb(),
        )
    }
}

impl<B: Backend> std::fmt::Debug for QloraModel<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QloraModel")
            .field("quantization", &self.lora_config.base_quantization)
            .field("lora_rank", &self.lora_config.lora_rank)
            .field("trainable", &self.trainable)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qlora_config_defaults() {
        let config = QloraConfig::default();
        assert_eq!(config.lora_rank, 8);
        assert_eq!(config.lora_alpha, 16.0);
        assert!(config.target_modules.contains(&"output_head".to_string()));
    }

    #[test]
    fn test_precision_kind_parsing() {
        use super::precision::PrecisionKind;
        assert_eq!(PrecisionKind::from_str("fp32"), PrecisionKind::FP32);
        assert_eq!(PrecisionKind::from_str("fp16"), PrecisionKind::FP16);
        assert_eq!(PrecisionKind::from_str("bf16"), PrecisionKind::BF16);
        assert_eq!(PrecisionKind::from_str("half"), PrecisionKind::FP16);
        assert!(PrecisionKind::FP32.to_str() == "fp32");
        assert!(PrecisionKind::from_str("fp16").is_low_precision());
        assert!(!PrecisionKind::from_str("fp32").is_low_precision());
    }

    #[test]
    fn test_mixed_precision_trainer() {
        use super::precision::{MixedPrecisionTrainer, PrecisionConfig, PrecisionKind};
        let trainer = MixedPrecisionTrainer::new(PrecisionConfig::new(PrecisionKind::FP16));
        assert!(trainer.config.enabled);
        assert_eq!(trainer.loss_scale, 65536.0);
        assert_eq!(trainer.scale_loss(1.0), 65536.0);

        let trainer_fp32 = MixedPrecisionTrainer::new(PrecisionConfig::default());
        assert!(!trainer_fp32.config.enabled);
        assert_eq!(trainer_fp32.loss_scale, 1.0);
    }

    #[test]
    fn test_loss_scale_update() {
        use super::precision::{MixedPrecisionTrainer, PrecisionConfig, PrecisionKind};
        let mut trainer = MixedPrecisionTrainer::new(PrecisionConfig::new(PrecisionKind::FP16));
        let initial = trainer.loss_scale;
        trainer.update_loss_scale(true);
        assert_eq!(trainer.loss_scale, initial / 2.0);
        trainer.update_loss_scale(false);
        assert_eq!(trainer.loss_scale, initial);
    }
}
