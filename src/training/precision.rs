//! 混合精度训练模块
//!
//! 支持 FP32（全精度）、FP16（半精度）、BF16（Brain Float）三种精度模式。
//! 在 Burn 0.19 中，Autocast 支持受后端限制；本模块提供精度配置和转换基础设施，
//! 为未来 Burn 版本的原生 autocast 集成预留接口。

use burn::prelude::*;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionKind {
    FP32,
    FP16,
    BF16,
}

impl PrecisionKind {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "fp16" | "half" => PrecisionKind::FP16,
            "bf16" | "bfloat16" => PrecisionKind::BF16,
            _ => PrecisionKind::FP32,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            PrecisionKind::FP32 => "fp32",
            PrecisionKind::FP16 => "fp16",
            PrecisionKind::BF16 => "bf16",
        }
    }

    /// 是否启用低精度训练
    pub fn is_low_precision(&self) -> bool {
        matches!(self, PrecisionKind::FP16 | PrecisionKind::BF16)
    }

    /// 估算相对于 FP32 的显存节省比例
    pub fn memory_savings_ratio(&self) -> f64 {
        match self {
            PrecisionKind::FP32 => 1.0,
            PrecisionKind::FP16 => 0.5,
            PrecisionKind::BF16 => 0.5,
        }
    }
}

/// 精度配置
#[derive(Debug, Clone)]
pub struct PrecisionConfig {
    pub kind: PrecisionKind,
    pub enabled: bool,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            kind: PrecisionKind::FP32,
            enabled: false,
        }
    }
}

impl PrecisionConfig {
    pub fn new(kind: PrecisionKind) -> Self {
        Self {
            kind,
            enabled: kind != PrecisionKind::FP32,
        }
    }

    pub fn from_config(use_amp: bool, precision_str: &str) -> Self {
        let kind = PrecisionKind::from_str(precision_str);
        Self {
            kind,
            enabled: use_amp || kind.is_low_precision(),
        }
    }
}

/// 混合精度训练包装器
///
/// 管理模型参数的精度转换和梯度缩放。
/// 在 Burn 0.19 中，实际低精度计算取决于后端实现；
/// 本包装器提供配置管理和未来 autocast 集成点。
pub struct MixedPrecisionTrainer {
    pub config: PrecisionConfig,
    pub loss_scale: f32,
}

impl MixedPrecisionTrainer {
    pub fn new(config: PrecisionConfig) -> Self {
        let loss_scale = if config.enabled { 65536.0 } else { 1.0 };
        Self { config, loss_scale }
    }

    pub fn from_training_config(use_amp: bool, precision: &str) -> Self {
        let config = PrecisionConfig::from_config(use_amp, precision);
        Self::new(config)
    }

    /// 将 FP32 损失放大以保持小梯度精度（用于 FP16 训练）
    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.loss_scale
    }

    /// 将梯度缩小回原始比例
    pub fn unscale_gradients<B: Backend>(&self, grads: Tensor<B, 1>) -> Tensor<B, 1> {
        grads / self.loss_scale
    }

    /// 检查是否需要更新 loss scale
    pub fn update_loss_scale(&mut self, grad_overflow: bool) {
        if !self.config.enabled {
            return;
        }
        if grad_overflow {
            self.loss_scale = (self.loss_scale / 2.0).max(1.0);
        } else if self.loss_scale < 1_048_576.0 {
            self.loss_scale = (self.loss_scale * 2.0).min(1_048_576.0);
        }
    }

    /// 获取当前精度模式的描述信息
    pub fn info(&self) -> String {
        if self.config.enabled {
            format!(
                "混合精度: {} (loss_scale={}, 显存节省≈{:.0}%)",
                self.config.kind.to_str(),
                self.loss_scale as i64,
                (1.0 - self.config.kind.memory_savings_ratio()) * 100.0
            )
        } else {
            "全精度: fp32".to_string()
        }
    }
}
