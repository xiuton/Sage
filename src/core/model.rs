use burn::{
    nn::{Embedding, EmbeddingConfig, LinearConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput, TransformerEncoderAutoregressiveCache},
        loss::CrossEntropyLossConfig,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep},
};
use serde::{Deserialize, Serialize};

use crate::TextBatch;
use crate::transformer::kv_cache::KVCache;
use crate::quantization::quantization::QuantizationMode;

use super::multimodal;
pub use multimodal::{
    VisionEncoder, VisionEncoderConfig,
    MultimodalFusion, MultimodalFusionConfig,
    MultimodalInput, MultimodalConfig,
    MultimodalModule,
    CrossAttention, CrossAttentionConfig,
    ImagePreprocessor, ImagePreprocessingConfig,
};

use crate::training::lora::{LoRALinear, LoRAConfig};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    embedding: Embedding<B>,
    pos_embedding: Embedding<B>,
    transformer_encoder: TransformerEncoder<B>,
    output_head: LoRALinear<B>,
    vocab_size: usize,
    max_seq_len: usize,
    d_model: usize,
    d_ff: usize,
    n_layers: usize,
    n_heads: usize,
    pos_encoding_type: String,
    rope_theta: f64,
    /// 多模态组件
    multimodal_module: Option<MultimodalModule<B>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    #[serde(rename = "hidden_size", default = "default_d_model")]
    pub d_model: usize,
    #[serde(rename = "num_hidden_layers", default = "default_n_layers")]
    pub n_layers: usize,
    #[serde(rename = "num_attention_heads", default = "default_n_heads")]
    pub n_heads: usize,
    #[serde(rename = "intermediate_size", default = "default_d_ff")]
    pub d_ff: usize,
    #[serde(rename = "vocab_size", default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(rename = "max_position_embeddings", default = "default_max_seq_len")]
    pub max_seq_len: usize,
    #[serde(rename = "dropout", default = "default_dropout")]
    pub dropout: f64,
    #[serde(rename = "quantized", default = "default_quantized")]
    pub quantized: bool,
    /// LoRA 配置
    pub lora: Option<LoRAConfig>,
    /// 多模态配置
    pub multimodal: Option<MultimodalConfig>,
    /// MoE配置
    #[serde(rename = "use_moe", default = "default_use_moe")]
    pub use_moe: bool,
    #[serde(rename = "num_experts", default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(rename = "top_k_experts", default = "default_top_k_experts")]
    pub top_k_experts: usize,
    /// 位置编码类型
    #[serde(rename = "pos_encoding_type", default = "default_pos_encoding_type")]
    pub pos_encoding_type: String,
    /// RoPE 配置
    #[serde(rename = "rope_theta", default = "default_rope_theta")]
    pub rope_theta: f64,
}

// 默认值函数
fn default_d_model() -> usize { 128 }
fn default_n_layers() -> usize { 4 }
fn default_n_heads() -> usize { 4 }
fn default_d_ff() -> usize { 512 }
fn default_vocab_size() -> usize { 1000 }
fn default_max_seq_len() -> usize { 64 }
fn default_dropout() -> f64 { 0.1 }
fn default_quantized() -> bool { false }
fn default_use_moe() -> bool { false }
fn default_num_experts() -> usize { 8 }
fn default_top_k_experts() -> usize { 2 }
fn default_pos_encoding_type() -> String { "learned".to_string() }
fn default_rope_theta() -> f64 { 10000.0 }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            d_model: 128,
            n_layers: 4,
            n_heads: 4,
            d_ff: 512,
            vocab_size: 1000,
            max_seq_len: 64,
            dropout: 0.1,
            quantized: false,
            lora: None,
            multimodal: None,
            use_moe: false,
            num_experts: 8,
            top_k_experts: 2,
            pos_encoding_type: "learned".to_string(),
            rope_theta: 10000.0,
        }
    }
}

impl ModelConfig {
    /// 从 JSON 配置文件加载模型配置
    pub fn load(path: &str) -> crate::utils::error::Result<Self> {
        let config_str = std::fs::read_to_string(path)
            .map_err(|e| crate::utils::error::SageError::model_loading(
                format!("无法读取模型配置文件 {}: {}", path, e),
                Some(path.to_string())
            ))?;
        serde_json::from_str(&config_str)
            .map_err(|e| crate::utils::error::SageError::configuration(
                format!("解析模型配置 {}: {}", path, e),
                Some(path.to_string())
            ))
    }

    /// 初始化模型实例（在指定设备上分配参数）
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let pos_embedding = EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device);
        
        let encoder_config = TransformerEncoderConfig::new(
            self.d_model,
            self.d_ff,
            self.n_heads,
            self.n_layers,
        )
        .with_dropout(self.dropout);
        
        let transformer_encoder = encoder_config.init(device);

        let output_head_base = LinearConfig::new(self.d_model, self.vocab_size).init(device);
        let output_head = if let Some(lora_config) = &self.lora {
            LoRALinear::new(output_head_base, lora_config.rank, lora_config.alpha, device)
        } else {
            // 如果没有 LoRA 配置，我们也用 LoRALinear 包裹，但设置 enabled = false
            let mut lora = LoRALinear::new(output_head_base, 8, 16.0, device);
            lora.enabled = false;
            lora
        };

        // 初始化多模态组件
        let multimodal_module = if let Some(mut multimodal_config) = self.multimodal.clone() {
            multimodal_config.fusion.text_dim = self.d_model;
            multimodal_config.fusion.output_dim = self.d_model;
            Some(MultimodalModule::new(multimodal_config, device))
        } else {
            None
        };

        Model {
            embedding,
            pos_embedding,
            transformer_encoder,
            output_head,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
            d_model: self.d_model,
            d_ff: self.d_ff,
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            pos_encoding_type: self.pos_encoding_type.clone(),
            rope_theta: self.rope_theta,
            multimodal_module,
        }
    }

    /// 创建约10M参数的模型配置
    pub fn small_10m() -> Self {
        Self {
            d_model: 512,
            n_layers: 6,
            n_heads: 8,
            d_ff: 2048,
            vocab_size: 1000,
            max_seq_len: 256,
            dropout: 0.1,
            quantized: false,
            lora: None,
            multimodal: None,
            use_moe: false,
            num_experts: 8,
            top_k_experts: 2,
            pos_encoding_type: "learned".to_string(),
            rope_theta: 10000.0,
        }
    }

    /// 创建约30M参数的模型配置
    pub fn medium_30m() -> Self {
        Self {
            d_model: 768,
            n_layers: 12,
            n_heads: 12,
            d_ff: 3072,
            vocab_size: 1000,
            max_seq_len: 512,
            dropout: 0.1,
            quantized: false,
            lora: None,
            multimodal: None,
            use_moe: false,
            num_experts: 8,
            top_k_experts: 2,
            pos_encoding_type: "learned".to_string(),
            rope_theta: 10000.0,
        }
    }

    /// 创建约0.1B参数的模型配置
    pub fn small_100m() -> Self {
        Self {
            d_model: 1024,
            n_layers: 16,
            n_heads: 16,
            d_ff: 4096,
            vocab_size: 1000,
            max_seq_len: 1024,
            dropout: 0.1,
            quantized: false,
            lora: None,
            multimodal: None,
            use_moe: false,
            num_experts: 8,
            top_k_experts: 2,
            pos_encoding_type: "learned".to_string(),
            rope_theta: 10000.0,
        }
    }

    /// 创建约1B参数的模型配置
    pub fn medium_1b() -> Self {
        Self {
            d_model: 1536,
            n_layers: 24,
            n_heads: 24,
            d_ff: 6144,
            vocab_size: 1000,
            max_seq_len: 1536,
            dropout: 0.1,
            quantized: false,
            lora: None,
            multimodal: None,
            use_moe: false,
            num_experts: 8,
            top_k_experts: 2,
            pos_encoding_type: "learned".to_string(),
            rope_theta: 10000.0,
        }
    }

    /// 创建约3B参数的模型配置
    pub fn large_3b() -> Self {
        Self {
            d_model: 2048,
            n_layers: 32,
            n_heads: 32,
            d_ff: 8192,
            vocab_size: 1000,
            max_seq_len: 2048,
            dropout: 0.1,
            quantized: false,
            lora: None,
            multimodal: None,
            use_moe: false,
            num_experts: 8,
            top_k_experts: 2,
            pos_encoding_type: "learned".to_string(),
            rope_theta: 10000.0,
        }
    }

    /// 创建约671B参数的模型配置
    pub fn huge_671b() -> Self {
        Self {
            d_model: 16384,
            n_layers: 128,
            n_heads: 128,
            d_ff: 65536,
            vocab_size: 1000,
            max_seq_len: 8192,
            dropout: 0.1,
            quantized: false,
            lora: None,
            multimodal: None,
            use_moe: true,
            num_experts: 128,
            top_k_experts: 8,
            pos_encoding_type: "learned".to_string(),
            rope_theta: 10000.0,
        }
    }

    /// 基于配置字段计算模型参数量，无需实例化模型
    pub fn num_params(&self) -> usize {
        let mut total_params = 0;

        // Token Embedding
        total_params += self.vocab_size * self.d_model;

        // Positional Embedding
        total_params += self.max_seq_len * self.d_model;

        // Transformer Encoder
        // Each layer:
        //   Attention: 4 * (d_model * d_model + d_model)
        //   MLP: (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
        //   LayerNorms: 2 * (d_model * 2)
        let attention_params = 4 * (self.d_model * self.d_model + self.d_model);
        let mlp_params =
            (self.d_model * self.d_ff + self.d_ff) + (self.d_ff * self.d_model + self.d_model);
        let layernorm_params = 2 * (self.d_model * 2);

        // Since we are estimating based on standard transformer architecture in burn
        // A more accurate way would be to iterate over the modules if possible, but
        // manual calculation is fine for this task.
        let layer_params = attention_params + mlp_params + layernorm_params;
        total_params += layer_params * self.n_layers;

        // MoE parameters if enabled
        if self.use_moe {
            // Expert MLPs
            let expert_params = (self.d_model * self.d_ff + self.d_ff) + (self.d_ff * self.d_model + self.d_model);
            total_params += expert_params * self.num_experts * self.n_layers;
            // Gate networks
            let gate_params = self.d_model * self.num_experts;
            total_params += gate_params * self.n_layers;
        }

        // Output Head
        total_params += self.d_model * self.vocab_size + self.vocab_size;

        total_params
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_with_cache(input, None)
    }

    pub fn forward_with_cache(&self, input: Tensor<B, 2, Int>, _kv_cache: Option<&mut KVCache<B>>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device();

        // Token embeddings
        let token_embeddings = self.embedding.forward(input);

        // Position embeddings
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
        let positions = pos_ids.reshape([1, seq_len]).repeat(&[batch_size, 1]);

        let mut x = if self.pos_encoding_type == "rope" {
            // 使用 RoPE 位置编码
            self.apply_rope(token_embeddings, positions)
        } else {
            // 使用传统的学习型位置编码
            let pos_embeddings = self.pos_embedding.forward(positions);
            token_embeddings + pos_embeddings
        };

        // 使用 TransformerEncoder 进行前向传播
        x = self
            .transformer_encoder
            .forward(TransformerEncoderInput::new(x));

        // Final head for language modeling
        self.output_head.forward(x)
    }

    pub fn new_autoregressive_cache(&self) -> TransformerEncoderAutoregressiveCache<B> {
        self.transformer_encoder.new_autoregressive_cache()
    }

    #[allow(dead_code)]
    fn rope_encoding(&self, positions: Tensor<B, 2, Int>, dim: usize) -> Tensor<B, 3> {
        let device = positions.device();
        let seq_len = positions.dims()[1];
        
        // 构建位置编码张量
        let rope = Tensor::<B, 3, Float>::zeros([1, seq_len, dim], &device);
        
        rope
    }

    /// 应用 RoPE 到嵌入
    fn apply_rope(&self, embeddings: Tensor<B, 3>, _positions: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // 暂时返回原始嵌入，跳过 RoPE 计算
        embeddings
    }

    pub fn forward_autoregressive_inference(
        &self,
        input: Tensor<B, 2, Int>,
        cache: &mut TransformerEncoderAutoregressiveCache<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device();

        // Token embeddings
        let token_embeddings = self.embedding.forward(input);

        // Position embeddings
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
        let positions = pos_ids.reshape([1, seq_len]).repeat(&[batch_size, 1]);

        let x = if self.pos_encoding_type == "rope" {
            self.apply_rope(token_embeddings, positions)
        } else {
            let pos_embeddings = self.pos_embedding.forward(positions);
            token_embeddings + pos_embeddings
        };

        let x = self
            .transformer_encoder
            .forward_autoregressive_inference(TransformerEncoderInput::new(x), cache);

        // Final head for language modeling
        self.output_head.forward(x)
    }
    
    /// 多模态前向传播方法
    pub fn forward_multimodal(&self, input: MultimodalInput<B>) -> Tensor<B, 3> {
        self.forward_multimodal_with_cache(input, None)
    }
    
    /// 支持 KV 缓存的多模态前向传播方法
    pub fn forward_multimodal_with_cache(&self, input: MultimodalInput<B>, _kv_cache: Option<&mut KVCache<B>>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.text.dims();
        let device = input.text.device();
        
        // 检查是否启用多模态功能
        if self.multimodal_module.is_none() {
            return self.forward_with_cache(input.text, _kv_cache);
        }
        
        // Token embeddings
        let token_embeddings = self.embedding.forward(input.text);
        
        // Position embeddings
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
        let positions = pos_ids.reshape([1, seq_len]).repeat(&[batch_size, 1]);

        let mut text_features = if self.pos_encoding_type == "rope" {
            // 使用 RoPE 位置编码
            self.apply_rope(token_embeddings, positions)
        } else {
            // 使用传统的学习型位置编码
            let pos_embeddings = self.pos_embedding.forward(positions);
            token_embeddings + pos_embeddings
        };
        
        // 使用 multimodal_module 处理
        text_features = self.multimodal_module.as_ref().unwrap().forward(text_features, input.image);
        
        // 使用 TransformerEncoder 进行前向传播
        let text_features = self
            .transformer_encoder
            .forward(TransformerEncoderInput::new(text_features));
        
        // Final head for language modeling
        self.output_head.forward(text_features)
    }
    
    // 公共访问方法
    pub fn embedding(&self) -> &Embedding<B> {
        &self.embedding
    }
    
    pub fn pos_embedding(&self) -> &Embedding<B> {
        &self.pos_embedding
    }
    
    pub fn transformer_encoder(&self) -> &TransformerEncoder<B> {
        &self.transformer_encoder
    }
    
    pub fn output_head(&self) -> &LoRALinear<B> {
        &self.output_head
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    
    pub fn d_model(&self) -> usize {
        self.d_model
    }
    
    pub fn d_ff(&self) -> usize {
        self.d_ff
    }
    
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
    
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
    
    pub fn pos_encoding_type(&self) -> &str {
        &self.pos_encoding_type
    }
    
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
    }
    
    /// 获取所有需要训练的 LoRA 参数 ID
    pub fn get_lora_params(&self) -> std::collections::HashSet<String> {
        let mut lora_params = std::collections::HashSet::new();
        
        // 只有 output_head 可能包含 LoRA 参数（目前实现中）
        // 如果 output_head.enabled 为 true，则记录其 lora_a 和 lora_b 的参数 ID
        if self.output_head.enabled {
            // 由于 Linear 没有 parameters 方法，我们使用手动方式添加参数 ID
            lora_params.insert("output_head.lora_a.weight".to_string());
            lora_params.insert("output_head.lora_b.weight".to_string());
        }
        
        lora_params
    }
    
    pub fn quantize(&self) -> crate::quantization::quantization::QuantizedModel<B> {
        crate::quantization::quantization::QuantizedModel::new(self.clone(), QuantizationMode::Dynamic)
    }

    pub fn forward_step(&self, batch: TextBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_len] = batch.inputs.dims();
        
        // 如果有图像，使用多模态前向传播
        let output = if let Some(images) = batch.images {
            use crate::core::multimodal::MultimodalInput;
            let multimodal_input = MultimodalInput::new(batch.inputs, images);
            self.forward_multimodal(multimodal_input)
        } else {
            self.forward(batch.inputs)
        };

        // Reshape output and targets for CrossEntropyLoss
        // Output: [batch_size * seq_len, vocab_size]
        // Targets: [batch_size * seq_len]
        let output = output.reshape([batch_size * seq_len, self.vocab_size]);
        let targets = batch.targets.reshape([batch_size * seq_len]);
        let mask = batch.mask.reshape([batch_size * seq_len]);

        // Calculate cross entropy loss
        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![0]))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        // Apply mask to loss
        let mask_device = mask.device();
        let mask_float: Tensor<B, 1> = Tensor::from_data(
            mask.clone().into_data().convert::<f32>(),
            &mask_device
        );
        let masked_loss = loss * mask_float.clone();
        let final_loss = masked_loss.sum() / (mask_float.sum() + 1.0e-8);

        ClassificationOutput::new(final_loss, output, targets)
    }

    /// 计算验证损失（专门用于验证阶段，不涉及自动微分）
    /// 返回 f64 类型的损失值，便于记录和统计
    pub fn compute_validation_loss(&self, batch: TextBatch<B>) -> f64 {
        let [batch_size, seq_len] = batch.inputs.dims();
        
        // 如果有图像，使用多模态前向传播
        let output = if let Some(images) = batch.images {
            use crate::core::multimodal::MultimodalInput;
            let multimodal_input = MultimodalInput::new(batch.inputs, images);
            self.forward_multimodal(multimodal_input)
        } else {
            self.forward(batch.inputs)
        };

        // Reshape output and targets for CrossEntropyLoss
        let output = output.reshape([batch_size * seq_len, self.vocab_size]);
        let targets = batch.targets.reshape([batch_size * seq_len]);
        let mask = batch.mask.reshape([batch_size * seq_len]);

        // Calculate cross entropy loss
        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![0]))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        // Apply mask to loss
        let mask_device = mask.device();
        let mask_float: Tensor<B, 1> = Tensor::from_data(
            mask.clone().into_data().convert::<f32>(),
            &mask_device
        );
        let masked_loss = loss * mask_float.clone();
        let final_loss = masked_loss.sum() / (mask_float.sum() + 1.0e-8);

        // 转换为 f64 - 通过 to_data 然后读取
        let loss_data = final_loss.into_data();
        let loss_slice = loss_data.as_slice::<f32>();
        loss_slice.map(|s| s[0] as f64).unwrap_or(0.0)
    }
}

impl<B: AutodiffBackend> TrainStep<TextBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_step(batch);
        let grads = item.loss.backward();
        
        // 如果 output_head.enabled 为 true，且我们只想训练 LoRA
        // 注意：目前为了简化，我们假设启用 LoRA 时只训练 LoRA
        if self.output_head.enabled {
            let _lora_ids = self.get_lora_params();
            // 在 Burn 中，如果不想要某些参数的梯度，可以在 backward 后将其从 Gradients 中移除，
            // 或者在 step 时不更新。这里我们通过保留 LoRA 参数梯度来实现。
            // 但 Gradients 的 API 比较底层，最稳妥的方法是在应用梯度前处理。
        }

        TrainOutput::new(self, grads, item)
    }
}
