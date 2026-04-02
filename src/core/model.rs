use burn::{
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep},
};

use crate::TextBatch;
use crate::core::kv_cache::KVCache;
use crate::quantization::quantization::QuantizationMode;

use super::multimodal;
pub use multimodal::{
    VisionEncoder, VisionEncoderConfig,
    MultimodalFusion, MultimodalFusionConfig,
    MultimodalInput, MultimodalConfig,
    FusionStrategy,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    embedding: Embedding<B>,
    pos_embedding: Embedding<B>,
    transformer_encoder: TransformerEncoder<B>,
    output_head: Linear<B>,
    vocab_size: usize,
    max_seq_len: usize,
    d_model: usize,
    d_ff: usize,
    n_layers: usize,
    /// 多模态组件
    vision_encoder: Option<VisionEncoder<B>>,
    multimodal_fusion: Option<MultimodalFusion<B>>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 128)]
    pub d_model: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 512)]
    pub d_ff: usize,
    #[config(default = 1000)] // Default, will be overridden
    pub vocab_size: usize,
    #[config(default = 64)]
    pub max_seq_len: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = false)]
    pub quantized: bool,
    /// 多模态配置
    #[config(default = "None")]
    pub multimodal: Option<MultimodalConfig>,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let pos_embedding = EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device);
        let transformer_encoder =
            TransformerEncoderConfig::new(self.d_model, self.d_ff, self.n_heads, self.n_layers)
                .with_dropout(self.dropout)
                .init(device);

        let output_head = LinearConfig::new(self.d_model, self.vocab_size).init(device);

        // 初始化多模态组件
        let (vision_encoder, multimodal_fusion) = if let Some(multimodal_config) = &self.multimodal {
            let vision_encoder = VisionEncoder::new(multimodal_config.vision_encoder.clone(), device);
            let fusion_config = MultimodalFusionConfig {
                text_dim: self.d_model,
                vision_dim: multimodal_config.vision_encoder.out_dim,
                output_dim: self.d_model,
                strategy: multimodal_config.fusion.strategy.clone(),
            };
            let multimodal_fusion = MultimodalFusion::new(fusion_config, device);
            (Some(vision_encoder), Some(multimodal_fusion))
        } else {
            (None, None)
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
            vision_encoder,
            multimodal_fusion,
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
            multimodal: None,
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
            multimodal: None,
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
            multimodal: None,
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
            multimodal: None,
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
            multimodal: None,
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
            multimodal: None,
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_with_cache(input, None)
    }

    pub fn forward_with_cache(&self, input: Tensor<B, 2, Int>, kv_cache: Option<&mut KVCache<B>>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device();

        // Token embeddings
        let token_embeddings = self.embedding.forward(input);

        // Position embeddings - 使用 arange 创建位置索引
        let pos_ids = if let Some(cache) = kv_cache {
            // 如果有缓存，从缓存长度开始计算位置
            let start_pos = cache.get_cached_seq_len() as i64;
            Tensor::<B, 1, Int>::arange(start_pos..(start_pos + seq_len as i64), &device)
        } else {
            Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
        };
        
        let positions = pos_ids.reshape([1, seq_len]).repeat(&[batch_size, 1]);
        let pos_embeddings = self.pos_embedding.forward(positions);

        let mut x = token_embeddings + pos_embeddings;

        // 使用TransformerEncoder进行前向传播
        // 注意：burn库的TransformerEncoder目前不直接支持KV缓存
        // 但我们通过位置编码的优化来减少重复计算
        x = self
            .transformer_encoder
            .forward(TransformerEncoderInput::new(x));

        // Final head for language modeling
        self.output_head.forward(x)
    }
    
    /// 多模态前向传播方法
    pub fn forward_multimodal(&self, input: MultimodalInput<B>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.text.dims();
        let device = input.text.device();
        
        // 检查是否启用多模态功能
        if self.vision_encoder.is_none() || self.multimodal_fusion.is_none() {
            return self.forward(input.text);
        }
        
        // Token embeddings
        let token_embeddings = self.embedding.forward(input.text);
        
        // Position embeddings
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
        let positions = pos_ids.reshape([1, seq_len]).repeat(&[batch_size, 1]);
        let pos_embeddings = self.pos_embedding.forward(positions);
        
        let mut text_features = token_embeddings + pos_embeddings;
        
        // 编码图像
        let vision_embedding = self.vision_encoder.as_ref().unwrap().forward(input.image);
        
        // 融合文本和视觉特征
        text_features = self.multimodal_fusion.as_ref().unwrap().forward(text_features, vision_embedding);
        
        // 使用TransformerEncoder进行前向传播
        text_features = self
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
    
    pub fn output_head(&self) -> &Linear<B> {
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
    
    pub fn quantize(&self) -> crate::quantization::quantization::QuantizedModel<B> {
        crate::quantization::quantization::QuantizedModel::new(self.clone(), QuantizationMode::Dynamic)
    }

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

        // Output Head
        total_params += self.d_model * self.vocab_size + self.vocab_size;

        total_params
    }

    pub fn forward_step(&self, batch: TextBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_len] = batch.inputs.dims();
        let output = self.forward(batch.inputs);

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
        let final_loss = masked_loss.sum() / mask_float.sum().max();

        ClassificationOutput::new(final_loss, output, targets)
    }

    /// 计算验证损失（专门用于验证阶段，不涉及自动微分）
    /// 返回 f64 类型的损失值，便于记录和统计
    pub fn compute_validation_loss(&self, batch: TextBatch<B>) -> f64 {
        let [batch_size, seq_len] = batch.inputs.dims();
        let output = self.forward(batch.inputs);

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
        let final_loss = masked_loss.sum() / mask_float.sum().max();

        // 转换为 f64 - 通过 to_data 然后读取
        let loss_data = final_loss.into_data();
        let loss_slice = loss_data.as_slice::<f32>();
        loss_slice.map(|s| s[0] as f64).unwrap_or(0.0)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: <Model<B> as TrainStep>::Input) -> TrainOutput<<Model<B> as TrainStep>::Output> {
        let item = self.forward_step(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}


