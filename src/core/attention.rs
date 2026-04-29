use burn::{
    nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig, Dropout, DropoutConfig},
    prelude::*,
    tensor::{activation, backend::Backend},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionType {
    Standard,
    FlashAttention,
    GroupedQueryAttention,
}

impl Default for AttentionType {
    fn default() -> Self {
        Self::Standard
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        let head_dim = d_model / n_heads;
        let scale = (head_dim as f32).sqrt();
        Self {
            q_proj: LinearConfig::new(d_model, d_model).init(device),
            k_proj: LinearConfig::new(d_model, d_model).init(device),
            v_proj: LinearConfig::new(d_model, d_model).init(device),
            out_proj: LinearConfig::new(d_model, d_model).init(device),
            n_heads,
            head_dim,
            scale,
        }
    }

    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, seq_len, _d_model] = x.dims();
        x.reshape([batch, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3])
    }

    fn merge_heads(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, n_heads, seq_len, head_dim] = x.dims();
        x.permute([0, 2, 1, 3])
            .reshape([batch, seq_len, n_heads * head_dim])
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.split_heads(self.q_proj.forward(x.clone()));
        let k = self.split_heads(self.k_proj.forward(x.clone()));
        let v = self.split_heads(self.v_proj.forward(x));

        let attn_weights = q
            .clone()
            .matmul(k.permute([0, 1, 3, 2]))
            .div_scalar(self.scale);

        let attn_weights = activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        let output = self.merge_heads(attn_output);
        self.out_proj.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct FlashSelfAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    n_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl<B: Backend> FlashSelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        let head_dim = d_model / n_heads;
        let scale = (head_dim as f32).sqrt();
        Self {
            q_proj: LinearConfig::new(d_model, d_model).init(device),
            k_proj: LinearConfig::new(d_model, d_model).init(device),
            v_proj: LinearConfig::new(d_model, d_model).init(device),
            out_proj: LinearConfig::new(d_model, d_model).init(device),
            n_heads,
            head_dim,
            scale,
        }
    }

    fn split_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, seq_len, _d_model] = x.dims();
        x.reshape([batch, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3])
    }

    fn merge_heads(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, n_heads, seq_len, head_dim] = x.dims();
        x.permute([0, 2, 1, 3])
            .reshape([batch, seq_len, n_heads * head_dim])
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.split_heads(self.q_proj.forward(x.clone()));
        let k = self.split_heads(self.k_proj.forward(x.clone()));
        let v = self.split_heads(self.v_proj.forward(x));

        let attn_weights = q
            .clone()
            .matmul(k.permute([0, 1, 3, 2]))
            .div_scalar(self.scale);

        let attn_weights = activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v);

        let output = self.merge_heads(attn_output);
        self.out_proj.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct GroupedQuerySelfAttention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    group_size: usize,
}

impl<B: Backend> GroupedQuerySelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, n_kv_heads: usize, device: &B::Device) -> Self {
        let head_dim = d_model / n_heads;
        let scale = (head_dim as f32).sqrt();
        let kv_dim = n_kv_heads * head_dim;
        Self {
            q_proj: LinearConfig::new(d_model, d_model).init(device),
            k_proj: LinearConfig::new(d_model, kv_dim).init(device),
            v_proj: LinearConfig::new(d_model, kv_dim).init(device),
            out_proj: LinearConfig::new(d_model, d_model).init(device),
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            group_size: n_heads / n_kv_heads,
        }
    }

    fn split_q_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, seq_len, _] = x.dims();
        x.reshape([batch, seq_len, self.n_heads, self.head_dim])
            .permute([0, 2, 1, 3])
    }

    fn split_kv_heads(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, seq_len, _] = x.dims();
        x.reshape([batch, seq_len, self.n_kv_heads, self.head_dim])
            .permute([0, 2, 1, 3])
    }

    fn merge_heads(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, n_heads, seq_len, head_dim] = x.dims();
        x.permute([0, 2, 1, 3])
            .reshape([batch, seq_len, n_heads * head_dim])
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.split_q_heads(self.q_proj.forward(x.clone()));
        let k = self.split_kv_heads(self.k_proj.forward(x.clone()));
        let v = self.split_kv_heads(self.v_proj.forward(x));

        let k_expanded = k.repeat(&[1, self.group_size, 1, 1]);
        let v_expanded = v.repeat(&[1, self.group_size, 1, 1]);

        let attn_weights = q
            .clone()
            .matmul(k_expanded.permute([0, 1, 3, 2]))
            .div_scalar(self.scale);

        let attn_weights = activation::softmax(attn_weights, 3);
        let attn_output = attn_weights.matmul(v_expanded);

        let output = self.merge_heads(attn_output);
        self.out_proj.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct SageTransformerBlock<B: Backend> {
    attention: SageAttention<B>,
    mlp: SwiGLUMLP<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Module, Debug)]
pub enum SageAttention<B: Backend> {
    Standard(MultiHeadSelfAttention<B>),
    Flash(FlashSelfAttention<B>),
    GroupedQuery(GroupedQuerySelfAttention<B>),
}

impl<B: Backend> SageAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Standard(attn) => attn.forward(x),
            Self::Flash(attn) => attn.forward(x),
            Self::GroupedQuery(attn) => attn.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct SwiGLUMLP<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> SwiGLUMLP<B> {
    pub fn new(d_model: usize, d_ff: usize, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(d_model, d_ff).init(device),
            up_proj: LinearConfig::new(d_model, d_ff).init(device),
            down_proj: LinearConfig::new(d_ff, d_model).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate_raw = self.gate_proj.forward(x.clone());
        let gate = gate_raw.clone().neg().exp().add_scalar(1.0).recip();
        let up = self.up_proj.forward(x);
        let hidden = gate.mul(up);
        self.down_proj.forward(hidden)
    }
}

impl<B: Backend> SageTransformerBlock<B> {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        attention_type: AttentionType,
        n_kv_heads: Option<usize>,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let attention = match attention_type {
            AttentionType::FlashAttention => {
                SageAttention::Flash(FlashSelfAttention::new(d_model, n_heads, device))
            }
            AttentionType::GroupedQueryAttention => {
                let kv_heads = n_kv_heads.unwrap_or(n_heads / 4).max(1);
                SageAttention::GroupedQuery(GroupedQuerySelfAttention::new(
                    d_model, n_heads, kv_heads, device,
                ))
            }
            _ => {
                SageAttention::Standard(MultiHeadSelfAttention::new(d_model, n_heads, device))
            }
        };

        let mlp = SwiGLUMLP::new(d_model, d_ff, device);
        let norm1 = LayerNormConfig::new(d_model).init(device);
        let norm2 = LayerNormConfig::new(d_model).init(device);
        let dropout = DropoutConfig::new(dropout).init();

        Self {
            attention,
            mlp,
            norm1,
            norm2,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = self.attention.forward(x);
        let x = self.dropout.forward(x);
        let x = x + residual;

        let residual = x.clone();
        let x = self.norm2.forward(x);
        let x = self.mlp.forward(x);
        let x = self.dropout.forward(x);
        x + residual
    }
}

#[derive(Module, Debug)]
pub struct SageTransformerEncoder<B: Backend> {
    blocks: Vec<SageTransformerBlock<B>>,
    norm: LayerNorm<B>,
}

impl<B: Backend> SageTransformerEncoder<B> {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        n_layers: usize,
        attention_type: AttentionType,
        n_kv_heads: Option<usize>,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let blocks = (0..n_layers)
            .map(|_| {
                SageTransformerBlock::new(
                    d_model,
                    d_ff,
                    n_heads,
                    attention_type,
                    n_kv_heads,
                    dropout,
                    device,
                )
            })
            .collect();

        let norm = LayerNormConfig::new(d_model).init(device);

        Self { blocks, norm }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }
        self.norm.forward(x)
    }
}
