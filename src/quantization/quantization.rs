use burn::{
    prelude::*,
    tensor::backend::Backend,
};
use serde::{Deserialize, Serialize};

/// 量化模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// 动态量化（仅权重量化）
    Dynamic,
    /// INT8量化
    Int8,
    /// INT4量化
    Int4,
}

/// 量化参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    pub mode: QuantizationMode,
    pub group_size: usize,
    pub zero_point: bool,
}

impl Default for QuantizationParams {
    fn default() -> Self {
        Self {
            mode: QuantizationMode::Dynamic,
            group_size: 128,
            zero_point: true,
        }
    }
}

/// 量化后的 Linear 层（模拟量化）
#[derive(Debug)]
pub struct QuantizedLinear<B: Backend> {
    pub weight: Tensor<B, 2>,
    pub bias: Option<Tensor<B, 1>>,
    pub scale: Tensor<B, 1>,
    pub zero_point: Option<Tensor<B, 1>>,
    pub mode: QuantizationMode,
}

impl<B: Backend> QuantizedLinear<B> {
    pub fn from_linear(linear: burn::nn::Linear<B>, mode: QuantizationMode) -> Self {
        let weight = linear.weight.val();
        let bias = linear.bias.map(|b| b.val());
        let device = weight.device();

        // 简单的对称 Min-Max 量化 (INT8)
        let (q_weight, scale) = match mode {
            QuantizationMode::Int8 | QuantizationMode::Dynamic => {
                let max_val = weight.clone().abs().max().into_data().as_slice::<f32>().unwrap()[0];
                let scale = max_val / 127.0;
                let q_weight = (weight / scale).round().clamp(-128.0, 127.0) * scale;
                (q_weight, Tensor::<B, 1>::from_data([scale], &device))
            }
            QuantizationMode::Int4 => {
                let max_val = weight.clone().abs().max().into_data().as_slice::<f32>().unwrap()[0];
                let scale = max_val / 7.0;
                let q_weight = (weight / scale).round().clamp(-8.0, 7.0) * scale;
                (q_weight, Tensor::<B, 1>::from_data([scale], &device))
            }
        };

        Self {
            weight: q_weight,
            bias,
            scale,
            zero_point: None,
            mode,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = input.dims();
        let [out_features, _in_features] = self.weight.dims();
        
        // 模拟量化推理：使用已量化（并恢复）的权重
        let input_2d = input.reshape([batch_size * seq_len, d_model]);
        let mut output = input_2d.matmul(self.weight.clone().transpose());
        
        if let Some(bias) = &self.bias {
            output = output + bias.clone().unsqueeze();
        }
        
        output.reshape([batch_size, seq_len, out_features])
    }
}

/// 量化模型（包装原始模型，提供大小计算功能）
pub struct QuantizedModel<B: Backend> {
    pub model: crate::core::Model<B>,
    pub mode: QuantizationMode,
    /// 模型大小（MB）
    pub model_size_mb: f64,
    /// 量化后大小（MB）
    pub quantized_size_mb: f64,
    /// 量化后的输出层
    pub quantized_output_head: QuantizedLinear<B>,
}

impl<B: Backend> QuantizedModel<B> {
    pub fn new(model: crate::core::Model<B>, mode: QuantizationMode) -> Self {
        let model_size_mb = Self::calculate_model_size(&model);
        let quantized_size_mb = Self::calculate_quantized_size(&model, &mode);
        
        // 对输出层进行量化
        let output_head_linear = model.output_head().linear.clone();
        let quantized_output_head = QuantizedLinear::from_linear(output_head_linear, mode);
        
        Self {
            model,
            mode,
            model_size_mb,
            quantized_size_mb,
            quantized_output_head,
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // 1. 正常的嵌入和 Transformer 层（目前尚未完全量化所有层，仅模拟）
        let [batch_size, seq_len] = input.dims();
        let device = input.device();
        let x = self.model.embedding().forward(input.clone());
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
        let positions = pos_ids.reshape([1, seq_len]).repeat(&[batch_size, 1]);
        let pos_embeddings = self.model.pos_embedding().forward(positions);
        
        let mut x = x + pos_embeddings;
        x = self.model.transformer_encoder().forward(burn::nn::transformer::TransformerEncoderInput::new(x));
        
        // 2. 使用量化后的输出层
        self.quantized_output_head.forward(x)
    }
    
    /// 获取模型大小信息
    pub fn get_size_info(&self) -> String {
        format!(
            "原始大小: {:.2} MB, 量化后大小: {:.2} MB, 压缩比: {:.1}x",
            self.model_size_mb,
            self.quantized_size_mb,
            self.model_size_mb / self.quantized_size_mb
        )
    }
    
    /// 计算模型大小（MB）
    fn calculate_model_size(model: &crate::core::Model<B>) -> f64 {
        let mut total_bytes = 0.0;
        
        // Token Embedding
        total_bytes += model.vocab_size() as f64 * model.d_model() as f64 * 4.0;
        
        // Positional Embedding
        total_bytes += model.max_seq_len() as f64 * model.d_model() as f64 * 4.0;
        
        // Transformer Encoder (简化计算)
        let attention_params = 4 * model.d_model() * model.d_model();
        let mlp_params = 2 * model.d_model() * model.d_ff();
        let layer_params = attention_params + mlp_params;
        total_bytes += layer_params as f64 * model.n_layers() as f64 * 4.0;
        
        // Output Head
        total_bytes += model.d_model() as f64 * model.vocab_size() as f64 * 4.0;
        
        total_bytes / (1024.0 * 1024.0)
    }
    
    /// 计算量化后模型大小（MB）
    fn calculate_quantized_size(model: &crate::core::Model<B>, mode: &QuantizationMode) -> f64 {
        let mut total_bytes = 0.0;
        
        // 嵌入层保持 float32
        total_bytes += model.vocab_size() as f64 * model.d_model() as f64 * 4.0;
        total_bytes += model.max_seq_len() as f64 * model.d_model() as f64 * 4.0;
        
        // 计算量化权重的字节数
        let weight_bytes = match mode {
            QuantizationMode::Dynamic => 4.0,  // float32
            QuantizationMode::Int8 => 1.0,     // int8
            QuantizationMode::Int4 => 0.5,     // int4 (位打包)
        };
        
        // Transformer Encoder
        let attention_params = 4 * model.d_model() * model.d_model();
        let mlp_params = 2 * model.d_model() * model.d_ff();
        let layer_params = attention_params + mlp_params;
        total_bytes += layer_params as f64 * model.n_layers() as f64 * weight_bytes;
        
        // Output Head
        total_bytes += model.d_model() as f64 * model.vocab_size() as f64 * weight_bytes;
        
        total_bytes / (1024.0 * 1024.0)
    }
}

/// 量化工具函数
pub mod utils {
    use super::*;
    
    /// 计算模型大小（MB）
    pub fn calculate_model_size<B: Backend>(model: &crate::core::Model<B>) -> f64 {
        QuantizedModel::<B>::calculate_model_size(model)
    }
    
    /// 计算量化后模型大小（MB）
    pub fn calculate_quantized_size<B: Backend>(model: &crate::core::Model<B>, mode: QuantizationMode) -> f64 {
        QuantizedModel::<B>::calculate_quantized_size(model, &mode)
    }
}
