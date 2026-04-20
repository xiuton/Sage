use burn::{
    nn::{
        Linear, LinearConfig, conv::{Conv2d, Conv2dConfig},
        pool::{AvgPool2d, AvgPool2dConfig},
        BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig, Embedding, EmbeddingConfig,
        Dropout, DropoutConfig,
        activation,
    },
    prelude::*,
    tensor::backend::Backend,
};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionEncoderConfig {
    pub in_channels: usize,
    pub hidden_channels: usize,
    pub out_dim: usize,
    pub encoder_type: String,
    pub num_layers: usize,
    pub patch_size: usize,
    pub image_size: usize,
}

impl Default for VisionEncoderConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            hidden_channels: 64,
            out_dim: 512,
            encoder_type: "resnet".to_string(),
            num_layers: 4,
            patch_size: 16,
            image_size: 224,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResNetVariant {
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
}

impl Default for ResNetVariant {
    fn default() -> Self {
        ResNetVariant::ResNet18
    }
}

#[derive(Module, Debug)]
pub struct BottleneckBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B>,
    downsample: Option<(Conv2d<B>, BatchNorm<B>)>,
}

impl<B: Backend> BottleneckBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
        let expansion = 4;
        
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);
        
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .with_stride([stride, stride])
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);
        
        let conv3 = Conv2dConfig::new([out_channels, out_channels * expansion], [1, 1])
            .init(device);
        let bn3 = BatchNormConfig::new(out_channels * expansion).init(device);
        
        let downsample = if in_channels != out_channels * expansion || stride != 1 {
            let ds_conv = Conv2dConfig::new([in_channels, out_channels * expansion], [1, 1])
                .with_stride([stride, stride])
                .init(device);
            let ds_bn = BatchNormConfig::new(out_channels * expansion).init(device);
            Some((ds_conv, ds_bn))
        } else {
            None
        };
        
        Self {
            conv1, bn1, conv2, bn2, conv3, bn3, downsample,
        }
    }
    
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = if let Some((conv, _bn)) = &self.downsample {
            conv.forward(x.clone())
        } else {
            x.clone()
        };
        
        let mut x = self.conv1.forward(x);
        x = self.bn1.forward(x);
        x = activation::Relu::new().forward(x);
        
        x = self.conv2.forward(x);
        x = self.bn2.forward(x);
        x = activation::Relu::new().forward(x);
        
        x = self.conv3.forward(x);
        x = self.bn3.forward(x);
        
        if let Some((_, bn)) = &self.downsample {
            x = bn.forward(x);
        }
        
        activation::Relu::new().forward(x + residual)
    }
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    downsample: Option<(Conv2d<B>, BatchNorm<B>)>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .with_stride([stride, stride])
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);
        
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);
        
        let downsample = if in_channels != out_channels || stride != 1 {
            let ds_conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
                .with_stride([stride, stride])
                .init(device);
            let ds_bn = BatchNormConfig::new(out_channels).init(device);
            Some((ds_conv, ds_bn))
        } else {
            None
        };
        
        Self { conv1, bn1, conv2, bn2, downsample }
    }
    
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = if let Some((conv, _bn)) = &self.downsample {
            conv.forward(x.clone())
        } else {
            x.clone()
        };
        
        let mut x = self.conv1.forward(x);
        x = self.bn1.forward(x);
        x = activation::Relu::new().forward(x);
        
        x = self.conv2.forward(x);
        x = self.bn2.forward(x);
        
        if let Some((_, bn)) = &self.downsample {
            x = bn.forward(x);
        }
        
        activation::Relu::new().forward(x + residual)
    }
}

#[derive(Module, Debug)]
pub struct ResNetEncoder<B: Backend> {
    initial_conv: Conv2d<B>,
    initial_bn: BatchNorm<B>,
    layer1: ResidualBlock<B>,
    layer2: ResidualBlock<B>,
    layer3: ResidualBlock<B>,
    layer4: ResidualBlock<B>,
    pool: AvgPool2d,
    projection: Linear<B>,
}

impl<B: Backend> ResNetEncoder<B> {
    pub fn new(config: &VisionEncoderConfig, device: &B::Device) -> Self {
        let channels = config.hidden_channels;

        let initial_conv = Conv2dConfig::new([config.in_channels, channels], [7, 7])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .with_stride([2, 2])
            .init(device);
        let initial_bn = BatchNormConfig::new(channels).init(device);

        let layer1 = ResidualBlock::new(channels, channels, 1, device);
        let layer2 = ResidualBlock::new(channels, channels * 2, 2, device);
        let layer3 = ResidualBlock::new(channels * 2, channels * 4, 2, device);
        let layer4 = ResidualBlock::new(channels * 4, channels * 8, 2, device);

        let pool = AvgPool2dConfig::new([7, 7]).with_strides([7, 7]).init();
        let projection = LinearConfig::new(channels * 8, config.out_dim).init(device);

        Self {
            initial_conv, initial_bn, layer1, layer2, layer3, layer4, pool, projection,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.initial_conv.forward(x);
        let x = self.initial_bn.forward(x);
        let x = activation::Relu::new().forward(x);

        let x = self.layer1.forward(x);
        let x = self.layer2.forward(x);
        let x = self.layer3.forward(x);
        let x = self.layer4.forward(x);

        let x = self.pool.forward(x);
        let [batch, channels, _, _] = x.dims();
        let x = x.reshape([batch, channels]);

        self.projection.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbedding<B: Backend> {
    proj: Conv2d<B>,
}

impl<B: Backend> PatchEmbedding<B> {
    pub fn new(patch_size: usize, in_channels: usize, hidden_channels: usize, device: &B::Device) -> Self {
        let proj = Conv2dConfig::new([in_channels, hidden_channels], [patch_size, patch_size])
            .with_stride([patch_size, patch_size])
            .init(device);
        Self { proj }
    }
    
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.proj.forward(x);
        let [batch, channels, h, w] = x.dims();
        x.reshape([batch, channels, h * w]).permute([0, 2, 1])
    }
}

#[derive(Module, Debug)]
pub struct TransformerEncoderBlock<B: Backend> {
    pub attention: MultiHeadAttention<B>,
    pub mlp: MLPBlock<B>,
    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub dropout: Dropout,
}

impl<B: Backend> TransformerEncoderBlock<B> {
    pub fn new(num_heads: usize, embed_dim: usize, mlp_ratio: f64, dropout: f64, device: &B::Device) -> Self {
        let attention = MultiHeadAttention::new(num_heads, embed_dim, device);
        let mlp = MLPBlock::new(embed_dim, (embed_dim as f64 * mlp_ratio) as usize, dropout, device);
        let norm1 = LayerNormConfig::new(embed_dim).init(device);
        let norm2 = LayerNormConfig::new(embed_dim).init(device);
        let dropout = DropoutConfig::new(dropout).init();
        
        Self { attention, mlp, norm1, norm2, dropout }
    }
    
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let attn_out = self.attention.forward(x.clone(), x.clone(), x.clone());
        let x = x + self.dropout.forward(attn_out);
        let x = self.norm1.forward(x);
        
        let mlp_out = self.mlp.forward(x.clone());
        x + self.dropout.forward(mlp_out)
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    num_heads: usize,
    embed_dim: usize,
    head_dim: usize,
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    scale: f64,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(num_heads: usize, embed_dim: usize, device: &B::Device) -> Self {
        let head_dim = embed_dim / num_heads;
        let q_proj = LinearConfig::new(embed_dim, embed_dim).init(device);
        let k_proj = LinearConfig::new(embed_dim, embed_dim).init(device);
        let v_proj = LinearConfig::new(embed_dim, embed_dim).init(device);
        let out_proj = LinearConfig::new(embed_dim, embed_dim).init(device);
        
        let scale = (head_dim as f64).sqrt().recip();
        
        Self {
            num_heads,
            embed_dim,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            scale,
        }
    }
    
    pub fn forward(&self, query: Tensor<B, 3>, key: Tensor<B, 3>, value: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, _] = query.dims();

        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);

        let q = self.reshape_to_heads(q);
        let k = self.reshape_to_heads(k);
        let v = self.reshape_to_heads(v);

        let mut attn_weights = q.matmul(k.permute([0, 1, 3, 2])) * self.scale;
        let dims = attn_weights.dims();
        let batch_sz = dims[0];
        let heads = dims[1];
        let seq1 = dims[2];
        let seq2 = dims[3];

        for i in 0..batch_sz {
            for j in 0..heads {
                for k in 0..seq1 {
                    let slice = attn_weights.clone().slice([i..i+1, j..j+1, k..k+1, 0..seq2]);
                    let max_val = slice.clone().reshape([1, 1, seq2]).max_dim(2);
                    let shifted = slice.clone() - max_val.clone().unsqueeze_dim(3);
                    let exp_val = shifted.exp();
                    let sum_exp = exp_val.clone().reshape([1, 1, seq2]).sum_dim(2);
                    let softmax = exp_val / sum_exp.unsqueeze_dim(3);
                    attn_weights = attn_weights.slice_assign([i..i+1, j..j+1, k..k+1, 0..seq2], softmax);
                }
            }
        }

        let attn_output = attn_weights.matmul(v);
        let attn_output = self.reshape_from_heads(attn_output);

        self.out_proj.forward(attn_output)
    }
    
    fn reshape_to_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, seq_len, _] = x.dims();
        x.reshape([batch, seq_len, self.num_heads, self.head_dim])
         .permute([0, 2, 1, 3])
    }
    
    fn reshape_from_heads(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, _, seq_len, head_dim] = x.dims();
        x.permute([0, 2, 1, 3])
          .reshape([batch, seq_len, self.num_heads * head_dim])
    }
}

#[derive(Module, Debug)]
pub struct MLPBlock<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub activation: activation::Gelu,
    pub dropout: Dropout,
}

impl<B: Backend> MLPBlock<B> {
    pub fn new(embed_dim: usize, hidden_dim: usize, dropout: f64, device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(embed_dim, hidden_dim).init(device);
        let linear2 = LinearConfig::new(hidden_dim, embed_dim).init(device);
        let activation = activation::Gelu::new();
        let dropout = DropoutConfig::new(dropout).init();
        
        Self { linear1, linear2, activation, dropout }
    }
    
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct VisionTransformer<B: Backend> {
    patch_embedding: PatchEmbedding<B>,
    class_token: Embedding<B>,
    position_embeddings: Embedding<B>,
    transformer_blocks: Vec<TransformerEncoderBlock<B>>,
    norm: LayerNorm<B>,
    projection: Linear<B>,
    num_patches: usize,
    num_tokens: usize,
}

impl<B: Backend> VisionTransformer<B> {
    pub fn new(config: &VisionEncoderConfig, device: &B::Device) -> Self {
        let patch_size = config.patch_size;
        let image_size = config.image_size;
        let num_patches_per_side = image_size / patch_size;
        let num_patches = num_patches_per_side * num_patches_per_side;
        let num_tokens = num_patches + 1;
        let hidden_dim = config.out_dim;
        let num_heads = 8.min(hidden_dim / 64);
        let num_layers = config.num_layers;
        
        let patch_embedding = PatchEmbedding::new(patch_size, config.in_channels, hidden_dim, device);
        let class_token = EmbeddingConfig::new(num_tokens, hidden_dim).init(device);
        let position_embeddings = EmbeddingConfig::new(num_tokens, hidden_dim).init(device);
        
        let mut transformer_blocks = Vec::new();
        for _ in 0..num_layers {
            transformer_blocks.push(TransformerEncoderBlock::new(num_heads, hidden_dim, 4.0, 0.1, device));
        }
        
        let norm = LayerNormConfig::new(hidden_dim).init(device);
        let projection = LinearConfig::new(hidden_dim, config.out_dim).init(device);
        
        Self {
            patch_embedding,
            class_token,
            position_embeddings,
            transformer_blocks,
            norm,
            projection,
            num_patches,
            num_tokens,
        }
    }
    
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, _, _, _] = x.dims();

        let patches = self.patch_embedding.forward(x);

        let class_tokens = self.class_token.forward(
            Tensor::arange(0..self.num_tokens as i64, &patches.device())
                .unsqueeze_dim(0)
                .repeat(&[batch_size, 1])
        );

        let mut x = Tensor::cat(vec![class_tokens, patches], 1);

        let positions = Tensor::arange(0..self.num_tokens as i64, &x.device())
            .unsqueeze_dim(0)
            .repeat(&[batch_size, 1]);
        let pos_embeddings = self.position_embeddings.forward(positions);

        x = x + pos_embeddings;

        for block in &self.transformer_blocks {
            x = block.forward(x);
        }

        let x = self.norm.forward(x);

        let class_output = x.slice([0..batch_size, 0..1]);
        let hidden_dim = class_output.dims()[2];
        let class_output = class_output.reshape([batch_size, hidden_dim]);

        self.projection.forward(class_output)
    }
}

#[derive(Module, Debug)]
pub struct VisionEncoder<B: Backend> {
    resnet: Option<ResNetEncoder<B>>,
    vit: Option<VisionTransformer<B>>,
    encoder_type: String,
}

impl<B: Backend> VisionEncoder<B> {
    pub fn new(config: VisionEncoderConfig, device: &B::Device) -> Self {
        let encoder_type = config.encoder_type.clone();
        
        let (resnet, vit) = match config.encoder_type.as_str() {
            "resnet" => {
                (Some(ResNetEncoder::new(&config, device)), None)
            },
            "vit" => {
                (None, Some(VisionTransformer::new(&config, device)))
            },
            _ => {
                (Some(ResNetEncoder::new(&config, device)), None)
            }
        };
        
        Self { resnet, vit, encoder_type }
    }
    
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        match self.encoder_type.as_str() {
            "resnet" => {
                self.resnet.as_ref().unwrap().forward(x)
            },
            "vit" => {
                self.vit.as_ref().unwrap().forward(x)
            },
            _ => {
                self.resnet.as_ref().unwrap().forward(x)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAttentionConfig {
    pub text_dim: usize,
    pub vision_dim: usize,
    pub num_heads: usize,
    pub dropout: f64,
}

impl Default for CrossAttentionConfig {
    fn default() -> Self {
        Self {
            text_dim: 512,
            vision_dim: 512,
            num_heads: 8,
            dropout: 0.1,
        }
    }
}

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    query_proj: Linear<B>,
    key_proj: Linear<B>,
    value_proj: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    scale: f64,
}

impl<B: Backend> CrossAttention<B> {
    pub fn new(config: &CrossAttentionConfig, device: &B::Device) -> Self {
        let text_dim = config.text_dim;
        let vision_dim = config.vision_dim;
        let num_heads = config.num_heads;
        
        let query_proj = LinearConfig::new(text_dim, text_dim).init(device);
        let key_proj = LinearConfig::new(vision_dim, text_dim).init(device);
        let value_proj = LinearConfig::new(vision_dim, text_dim).init(device);
        let out_proj = LinearConfig::new(text_dim, text_dim).init(device);
        
        let head_dim = text_dim / num_heads;
        let scale = (head_dim as f64).sqrt().recip();
        
        Self {
            query_proj,
            key_proj,
            value_proj,
            out_proj,
            num_heads,
            scale,
        }
    }
    
    pub fn forward(&self, text_embedding: Tensor<B, 3>, vision_embedding: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = text_embedding.dims();
        
        // 1. 线性投影
        let q = self.query_proj.forward(text_embedding);
        let k = self.key_proj.forward(vision_embedding.clone());
        let v = self.value_proj.forward(vision_embedding);
        
        // 2. Reshape 到多头格式: [batch, heads, seq_len, dim]
        let q = self.split_heads(q, batch_size, seq_len);
        let k = self.split_heads(k.unsqueeze_dim(1).repeat(&[1, seq_len, 1]), batch_size, seq_len);
        let v = self.split_heads(v.unsqueeze_dim(1).repeat(&[1, seq_len, 1]), batch_size, seq_len);
        
        // 3. 注意力计算
        let attention = q.matmul(k.permute([0, 1, 3, 2])) * self.scale;
        
        // 4. Softmax
        let attention = self.softmax_last_dim(attention);
        
        // 5. 应用注意力到 value
        let out = attention.matmul(v);
        
        // 6. 合并多头
        let out = self.combine_heads(out, batch_size, seq_len);
        
        // 7. 最后的投影
        self.out_proj.forward(out)
    }
    
    fn split_heads(&self, x: Tensor<B, 3>, batch_size: usize, seq_len: usize) -> Tensor<B, 4> {
        let last_dim = x.dims()[2] / self.num_heads;
        x.reshape([batch_size, seq_len, self.num_heads, last_dim])
            .permute([0, 2, 1, 3])
    }
    
    fn combine_heads(&self, x: Tensor<B, 4>, batch_size: usize, seq_len: usize) -> Tensor<B, 3> {
        let last_dim = x.dims()[2] * x.dims()[3];
        x.permute([0, 2, 1, 3])
            .reshape([batch_size, seq_len, last_dim])
    }
    
    fn softmax_last_dim(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut result = x.clone();
        let dims = x.dims();
        let batch = dims[0];
        let heads = dims[1];
        let seq1 = dims[2];
        let seq2 = dims[3];
        
        for i in 0..batch {
            for j in 0..heads {
                for k in 0..seq1 {
                    let slice = x.clone().slice([i..i+1, j..j+1, k..k+1, 0..seq2]);
                    let max_val = slice.clone().reshape([1, 1, seq2]).max_dim(2);
                    let shifted = slice.clone() - max_val.clone().unsqueeze_dim(3);
                    let exp_val = shifted.exp();
                    let sum_exp = exp_val.clone().reshape([1, 1, seq2]).sum_dim(2);
                    let softmax = exp_val / sum_exp.unsqueeze_dim(3);
                    result = result.slice_assign([i..i+1, j..j+1, k..k+1, 0..seq2], softmax);
                }
            }
        }
        result
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalFusionConfig {
    pub text_dim: usize,
    pub vision_dim: usize,
    pub output_dim: usize,
    pub strategy: String,
}

impl Default for MultimodalFusionConfig {
    fn default() -> Self {
        Self {
            text_dim: 512,
            vision_dim: 512,
            output_dim: 512,
            strategy: "gated".to_string(),
        }
    }
}

#[derive(Module, Debug)]
pub struct MultimodalFusion<B: Backend> {
    projection_text: Linear<B>,
    projection_vision: Linear<B>,
    gate_layer: Linear<B>,
    output_layer: Linear<B>,
    cross_attention: CrossAttention<B>,
}

impl<B: Backend> MultimodalFusion<B> {
    pub fn new(config: MultimodalFusionConfig, device: &B::Device) -> Self {
        let projection_text = LinearConfig::new(config.text_dim, config.output_dim).init(device);
        let projection_vision = LinearConfig::new(config.vision_dim, config.output_dim).init(device);
        let gate_layer = LinearConfig::new(config.output_dim * 2, config.output_dim).init(device);
        let output_layer = LinearConfig::new(config.output_dim, config.output_dim).init(device);
        
        let cross_attn_config = CrossAttentionConfig {
            text_dim: config.output_dim,
            vision_dim: config.output_dim,
            num_heads: 8,
            dropout: 0.1,
        };
        let cross_attention = CrossAttention::new(&cross_attn_config, device);
        
        Self {
            projection_text,
            projection_vision,
            gate_layer,
            output_layer,
            cross_attention,
        }
    }
    
    pub fn forward(&self, text_embedding: Tensor<B, 3>, vision_embedding: Tensor<B, 2>, strategy: &str) -> Tensor<B, 3> {
        let text_proj = self.projection_text.forward(text_embedding);
        let [_batch_size, seq_len, _] = text_proj.dims();
        
        let vision_proj_2d = self.projection_vision.forward(vision_embedding);
        let vision_proj_3d = vision_proj_2d.clone().unsqueeze_dim(1).repeat(&[1, seq_len, 1]);
        
        let fused = match strategy {
            "cross_attention" => {
                self.cross_attention.forward(text_proj, vision_proj_2d)
            },
            "gated" => {
                let concat = Tensor::cat(vec![text_proj.clone(), vision_proj_3d.clone()], 2);
                let gate_values = self.gate_layer.forward(concat);
                let gate_values = activation::Sigmoid::new().forward(gate_values);
                text_proj * gate_values.clone() + vision_proj_3d * (Tensor::ones_like(&gate_values) - gate_values)
            },
            "concatenate" => {
                Tensor::cat(vec![text_proj, vision_proj_3d], 2)
            },
            "add" => {
                text_proj + vision_proj_3d
            },
            _ => {
                let concat = Tensor::cat(vec![text_proj.clone(), vision_proj_3d.clone()], 2);
                let gate_values = self.gate_layer.forward(concat);
                let gate_values = activation::Sigmoid::new().forward(gate_values);
                text_proj * gate_values.clone() + vision_proj_3d * (Tensor::ones_like(&gate_values) - gate_values)
            }
        };
        
        self.output_layer.forward(fused)
    }
}

#[derive(Debug, Clone)]
pub struct MultimodalInput<B: Backend> {
    pub text: Tensor<B, 2, Int>,
    pub image: Tensor<B, 4>,
}

impl<B: Backend> MultimodalInput<B> {
    pub fn new(text: Tensor<B, 2, Int>, image: Tensor<B, 4>) -> Self {
        Self { text, image }
    }
    
    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            text: self.text.clone().to_device(device),
            image: self.image.clone().to_device(device),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePreprocessingConfig {
    pub target_size: usize,
    pub normalize: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub random_crop: bool,
    pub random_flip: bool,
    pub center_crop: bool,
}

impl Default for ImagePreprocessingConfig {
    fn default() -> Self {
        Self {
            target_size: 224,
            normalize: true,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            random_crop: false,
            random_flip: false,
            center_crop: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalConfig {
    pub vision_encoder: VisionEncoderConfig,
    pub fusion: MultimodalFusionConfig,
    pub preprocessing: ImagePreprocessingConfig,
    pub enable_multimodal: bool,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            vision_encoder: VisionEncoderConfig::default(),
            fusion: MultimodalFusionConfig::default(),
            preprocessing: ImagePreprocessingConfig::default(),
            enable_multimodal: false,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultimodalModule<B: Backend> {
    vision_encoder: VisionEncoder<B>,
    fusion: MultimodalFusion<B>,
    fusion_strategy: String,
}

impl<B: Backend> MultimodalModule<B> {
    pub fn new(config: MultimodalConfig, device: &B::Device) -> Self {
        let vision_encoder = VisionEncoder::new(config.vision_encoder, device);
        let fusion_strategy = config.fusion.strategy.clone();
        let fusion = MultimodalFusion::new(config.fusion, device);
        
        Self {
            vision_encoder,
            fusion,
            fusion_strategy,
        }
    }
    
    pub fn forward(&self, text_embedding: Tensor<B, 3>, image: Tensor<B, 4>) -> Tensor<B, 3> {
        let vision_embedding = self.vision_encoder.forward(image);
        self.fusion.forward(text_embedding, vision_embedding, &self.fusion_strategy)
    }
}

pub struct ImagePreprocessor<B: Backend> {
    config: ImagePreprocessingConfig,
    device: B::Device,
}

impl<B: Backend> ImagePreprocessor<B> {
    pub fn new(config: ImagePreprocessingConfig, device: B::Device) -> Self {
        Self { config, device }
    }
    
    pub fn preprocess(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut result = image.clone();
        
        // 1. 归一化 (将 [0, 255] 转换为 [0, 1])
        result = result / 255.0;
        
        // 2. 如果需要，应用标准化
        if self.config.normalize {
            result = self.normalize(result);
        }
        
        result
    }
    
    fn normalize(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = image.dims();
        
        // 使用单通道方式处理，避免复杂的 tensor 操作
        let mut result = image.clone();
        for c in 0..channels {
            let slice = image.clone().slice([0..batch, c..c+1, 0..height, 0..width]);
            let mean_val = self.config.mean[c];
            let std_val = self.config.std[c];
            let normalized = (slice - mean_val) / std_val;
            result = result.slice_assign([0..batch, c..c+1, 0..height, 0..width], normalized);
        }
        
        result
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainedWeightConfig {
    pub weight_path: String,
    pub strict_loading: bool,
    pub ignore_missing_keys: Vec<String>,
}

impl Default for PretrainedWeightConfig {
    fn default() -> Self {
        Self {
            weight_path: String::new(),
            strict_loading: true,
            ignore_missing_keys: Vec::new(),
        }
    }
}

pub struct WeightLoader<B: Backend> {
    device: B::Device,
    strict_loading: bool,
    ignore_missing_keys: Vec<String>,
}

impl<B: Backend> WeightLoader<B> {
    pub fn new(device: B::Device, config: PretrainedWeightConfig) -> Self {
        Self {
            device,
            strict_loading: config.strict_loading,
            ignore_missing_keys: config.ignore_missing_keys,
        }
    }
    
    pub fn load_weights(&self, weight_path: &str) -> Result<HashMap<String, Tensor<B, 2>>, String> {
        let path = Path::new(weight_path);
        
        if !path.exists() {
            return Err(format!("权重文件不存在: {}", weight_path));
        }
        
        let contents = std::fs::read(weight_path)
            .map_err(|e| format!("读取权重文件失败: {}", e))?;
        
        let weights: HashMap<String, Vec<f32>> = serde_json::from_slice(&contents)
            .map_err(|e| format!("解析权重文件失败: {}", e))?;

        let mut tensor_weights: HashMap<String, Tensor<B, 2>> = HashMap::new();

        for (key, value) in weights {
            let shape = value.len();
            let data: Vec<f32> = value;
            let tensor_data = TensorData::new(data, [1, shape]);
            let tensor: Tensor<B, 2> = Tensor::from_data(tensor_data, &self.device);
            tensor_weights.insert(key, tensor);
        }

        Ok(tensor_weights)
    }

    pub fn load_state_dict(&self, weight_path: &str) -> Result<HashMap<String, Tensor<B, 2>>, String> {
        self.load_weights(weight_path)
    }
    
    pub fn filter_weights(
        &self,
        weights: HashMap<String, Tensor<B, 2>>,
        expected_keys: &[&str],
    ) -> HashMap<String, Tensor<B, 2>> {
        let mut filtered = HashMap::new();
        
        for key in expected_keys {
            if weights.contains_key(*key) {
                filtered.insert(key.to_string(), weights.get(*key).unwrap().clone());
            } else if !self.ignore_missing_keys.contains(&key.to_string()) {
                if self.strict_loading {
                    panic!("缺少必需权重: {}", key);
                } else {
                    eprintln!("警告: 缺少权重 {}, 将使用随机初始化", key);
                }
            }
        }
        
        filtered
    }
}

pub struct DataAugmentation<B: Backend> {
    pub random_crop: bool,
    pub random_flip: bool,
    pub random_rotation: bool,
    pub color_jitter: bool,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> DataAugmentation<B> {
    pub fn new() -> Self {
        Self {
            random_crop: false,
            random_flip: false,
            random_rotation: false,
            color_jitter: false,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_random_crop(mut self, enabled: bool) -> Self {
        self.random_crop = enabled;
        self
    }

    pub fn with_random_flip(mut self, enabled: bool) -> Self {
        self.random_flip = enabled;
        self
    }

    pub fn with_random_rotation(mut self, enabled: bool) -> Self {
        self.random_rotation = enabled;
        self
    }

    pub fn with_color_jitter(mut self, enabled: bool) -> Self {
        self.color_jitter = enabled;
        self
    }

    pub fn augment(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image.clone()
    }

    pub fn center_crop(&self, image: Tensor<B, 4>, target_size: usize) -> Tensor<B, 4> {
        let [batch, channels, height, width] = image.dims();

        let start_h = (height - target_size) / 2;
        let start_w = (width - target_size) / 2;

        image.slice([0..batch, 0..channels, start_h..start_h+target_size, start_w..start_w+target_size])
    }

    pub fn random_crop(&self, image: Tensor<B, 4>, target_size: usize) -> Tensor<B, 4> {
        let [batch, channels, height, width] = image.dims();

        let start_h = (height - target_size) / 2;
        let start_w = (width - target_size) / 2;

        image.slice([0..batch, 0..channels, start_h..start_h+target_size, start_w..start_w+target_size])
    }
}
