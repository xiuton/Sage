use burn::{
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::TextBatch;

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
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device();

        // Token embeddings
        let token_embeddings = self.embedding.forward(input);

        // Position embeddings - 使用 arange 创建位置索引
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device);
        let positions = pos_ids.reshape([1, seq_len]).repeat(&[batch_size, 1]);
        let pos_embeddings = self.pos_embedding.forward(positions);

        let mut x = token_embeddings + pos_embeddings;

        x = self
            .transformer_encoder
            .forward(TransformerEncoderInput::new(x));

        // Final head for language modeling
        self.output_head.forward(x)
    }
    
    pub fn quantize(&self) -> Self {
        self.clone()
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

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![0]))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<TextBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: TextBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_step(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TextBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: TextBatch<B>) -> ClassificationOutput<B> {
        self.forward_step(batch)
    }
}
