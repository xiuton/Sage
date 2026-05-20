use burn::{
    nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig, Dropout, DropoutConfig},
    prelude::*,
    tensor::{activation, backend::Backend, TensorData},
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

fn make_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2, Float> {
    let range = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device);
    let rows = range.clone().reshape([seq_len, 1]).repeat(&[1, seq_len]);
    let cols = range.reshape([1, seq_len]).repeat(&[seq_len, 1]);
    cols.lower_equal(rows).float()
}

fn apply_rotary_emb<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    seq_len: usize,
    head_dim: usize,
    rope_theta: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = head_dim / 2;
    if half == 0 {
        return (q, k);
    }

    let mut cos_vals = Vec::with_capacity(seq_len * half);
    let mut sin_vals = Vec::with_capacity(seq_len * half);
    for pos in 0..seq_len {
        for i in 0..half {
            let theta = rope_theta.powf(-2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * theta;
            cos_vals.push(angle.cos() as f32);
            sin_vals.push(angle.sin() as f32);
        }
    }

    let cos = Tensor::<B, 3>::from_data(TensorData::new(cos_vals, [1, seq_len, half]), device);
    let sin = Tensor::<B, 3>::from_data(TensorData::new(sin_vals, [1, seq_len, half]), device);

    let [batch, n_heads, seq, hd] = q.dims();
    let cos_q = cos.clone()
        .reshape([1, 1, seq_len, half])
        .repeat(&[batch, n_heads, 1, 1]);
    let sin_q = sin.clone()
        .reshape([1, 1, seq_len, half])
        .repeat(&[batch, n_heads, 1, 1]);

    let q1 = q.clone().slice([0..batch, 0..n_heads, 0..seq, 0..half]);
    let q2 = q.slice([0..batch, 0..n_heads, 0..seq, half..hd]);
    let q_rotated = Tensor::cat(
        vec![
            q1.clone() * cos_q.clone() - q2.clone() * sin_q.clone(),
            q1 * sin_q + q2 * cos_q,
        ],
        3,
    );

    let [batch_k, n_kv, seq_k, hd_k] = k.dims();
    let cos_k = cos.reshape([1, 1, seq_len, half]).repeat(&[batch_k, n_kv, 1, 1]);
    let sin_k = sin.reshape([1, 1, seq_len, half]).repeat(&[batch_k, n_kv, 1, 1]);

    let k1 = k.clone().slice([0..batch_k, 0..n_kv, 0..seq_k, 0..half]);
    let k2 = k.slice([0..batch_k, 0..n_kv, 0..seq_k, half..hd_k]);
    let k_rotated = Tensor::cat(
        vec![
            k1.clone() * cos_k.clone() - k2.clone() * sin_k.clone(),
            k1 * sin_k + k2 * cos_k,
        ],
        3,
    );

    (q_rotated, k_rotated)
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

    pub fn forward(&self, x: Tensor<B, 3>, rope_theta: Option<f64>) -> Tensor<B, 3> {
        let device = x.device();
        let [_batch, seq_len, _d_model] = x.dims();
        let q = self.split_heads(self.q_proj.forward(x.clone()));
        let k = self.split_heads(self.k_proj.forward(x.clone()));
        let v = self.split_heads(self.v_proj.forward(x));

        let (q, k) = if let Some(theta) = rope_theta {
            apply_rotary_emb(q, k, seq_len, self.head_dim, theta, &device)
        } else {
            (q, k)
        };

        let mut attn_weights = q
            .clone()
            .matmul(k.permute([0, 1, 3, 2]))
            .div_scalar(self.scale);

        // 因果掩码：禁止关注未来 token
        let causal_mask = make_causal_mask(seq_len, &device)
            .reshape([1, 1, seq_len, seq_len]);
        attn_weights = attn_weights * causal_mask.clone() + (1.0 - causal_mask) * (-1e9);

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

    pub fn forward(&self, x: Tensor<B, 3>, rope_theta: Option<f64>) -> Tensor<B, 3> {
        let device = x.device();
        let [_batch, seq_len, _d_model] = x.dims();
        let q = self.split_q_heads(self.q_proj.forward(x.clone()));
        let k = self.split_kv_heads(self.k_proj.forward(x.clone()));
        let v = self.split_kv_heads(self.v_proj.forward(x));

        let (q, k) = if let Some(theta) = rope_theta {
            apply_rotary_emb(q, k, seq_len, self.head_dim, theta, &device)
        } else {
            (q, k)
        };

        let k_expanded = k.repeat(&[1, self.group_size, 1, 1]);
        let v_expanded = v.repeat(&[1, self.group_size, 1, 1]);

        let mut attn_weights = q
            .clone()
            .matmul(k_expanded.permute([0, 1, 3, 2]))
            .div_scalar(self.scale);

        let causal_mask = make_causal_mask(seq_len, &device)
            .reshape([1, 1, seq_len, seq_len]);
        attn_weights = attn_weights * causal_mask.clone() + (1.0 - causal_mask) * (-1e9);

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
    GroupedQuery(GroupedQuerySelfAttention<B>),
}

impl<B: Backend> SageAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, rope_theta: Option<f64>) -> Tensor<B, 3> {
        match self {
            Self::Standard(attn) => attn.forward(x, rope_theta),
            Self::GroupedQuery(attn) => attn.forward(x, rope_theta),
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
        let sigmoid = gate_raw.clone().neg().exp().add_scalar(1.0).recip();
        let gate = gate_raw * sigmoid;
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
            AttentionType::FlashAttention | AttentionType::Standard => {
                SageAttention::Standard(MultiHeadSelfAttention::new(d_model, n_heads, device))
            }
            AttentionType::GroupedQueryAttention => {
                let kv_heads = n_kv_heads.unwrap_or(n_heads / 4).max(1);
                SageAttention::GroupedQuery(GroupedQuerySelfAttention::new(
                    d_model, n_heads, kv_heads, device,
                ))
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

    pub fn forward(&self, x: Tensor<B, 3>, rope_theta: Option<f64>) -> Tensor<B, 3> {
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = self.attention.forward(x, rope_theta);
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
    pos_encoding_type: String,
    rope_theta: f64,
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
        pos_encoding_type: &str,
        rope_theta: f64,
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

        Self { blocks, norm, pos_encoding_type: pos_encoding_type.to_string(), rope_theta }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let rope_theta = if self.pos_encoding_type == "rope" {
            Some(self.rope_theta)
        } else {
            None
        };
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, rope_theta);
        }
        self.norm.forward(x)
    }
}
