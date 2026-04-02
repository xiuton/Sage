use burn::{
    nn::{conv::{Conv2d, Conv2dConfig}, BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu},
    prelude::*,
    tensor::backend::Backend,
};
use serde::{Serialize, Deserialize};

/// 图像编码器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionEncoderConfig {
    /// 输入通道数（RGB图像为3）
    pub in_channels: usize,
    /// 隐藏维度
    pub hidden_dim: usize,
    /// 输出维度（与文本嵌入维度匹配）
    pub out_dim: usize,
    /// 卷积层数
    pub num_layers: usize,
    /// 是否使用批归一化
    pub use_batch_norm: bool,
}

impl Default for VisionEncoderConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            hidden_dim: 64,
            out_dim: 512,
            num_layers: 4,
            use_batch_norm: true,
        }
    }
}

/// 图像编码器
#[derive(Module, Debug)]
pub struct VisionEncoder<B: Backend> {
    conv_layers: Vec<Conv2d<B>>,
    batch_norms: Option<Vec<BatchNorm<B>>>,
    relu: Relu,
    output_projection: Linear<B>,
}

impl<B: Backend> VisionEncoder<B> {
    pub fn new(config: VisionEncoderConfig, device: &B::Device) -> Self {
        let mut conv_layers = Vec::with_capacity(config.num_layers);
        let mut batch_norms = if config.use_batch_norm {
            Some(Vec::with_capacity(config.num_layers))
        } else {
            None
        };
        
        let mut in_channels = config.in_channels;
        
        for i in 0..config.num_layers {
            let out_channels = config.hidden_dim * (1 << i);
            let conv_config = Conv2dConfig::new([3, 3], [in_channels, out_channels])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1));
            conv_layers.push(conv_config.init(device));
            
            if let Some(batch_norms) = batch_norms.as_mut() {
                let bn_config = BatchNormConfig::new(out_channels);
                batch_norms.push(bn_config.init(device));
            }
            
            in_channels = out_channels;
        }
        
        let linear_config = LinearConfig::new(in_channels * 8 * 8, config.out_dim);
        let output_projection = linear_config.init(device);
        
        Self {
            conv_layers,
            batch_norms,
            relu: Relu::new(),
            output_projection,
        }
    }
    
    pub fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut x = image;
        
        for (i, conv) in self.conv_layers.iter().enumerate() {
            x = conv.forward(x);
            
            if let Some(batch_norms) = &self.batch_norms {
                x = batch_norms[i].forward(x);
            }
            
            x = self.relu.forward(x);
        }
        
        // 展平特征图
        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);
        
        // 投影到输出维度
        self.output_projection.forward(x)
    }
}

/// 多模态融合策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// 拼接融合
    Concatenate,
    /// 加法融合
    Add,
    /// 注意力融合
    Attention,
}

/// 多模态融合层配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalFusionConfig {
    pub text_dim: usize,
    pub vision_dim: usize,
    pub output_dim: usize,
    pub strategy: FusionStrategy,
}

/// 多模态融合层
#[derive(Module, Debug)]
pub struct MultimodalFusion<B: Backend> {
    projection_text: Linear<B>,
    projection_vision: Linear<B>,
    output_layer: Linear<B>,
}

impl<B: Backend> MultimodalFusion<B> {
    pub fn new(config: MultimodalFusionConfig, device: &B::Device) -> Self {
        let projection_text = LinearConfig::new(config.text_dim, config.output_dim).init(device);
        let projection_vision = LinearConfig::new(config.vision_dim, config.output_dim).init(device);
        let output_layer = LinearConfig::new(config.output_dim, config.output_dim).init(device);
        
        Self {
            projection_text,
            projection_vision,
            output_layer,
        }
    }
    
    pub fn forward(&self, text_embedding: Tensor<B, 3>, _vision_embedding: Tensor<B, 2>) -> Tensor<B, 3> {
        // 简化实现：只使用文本嵌入
        self.projection_text.forward(text_embedding)
    }
}

/// 多模态输入
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

/// 多模态模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalConfig {
    pub vision_encoder: VisionEncoderConfig,
    pub fusion: MultimodalFusionConfig,
    pub enable_multimodal: bool,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            vision_encoder: VisionEncoderConfig::default(),
            fusion: MultimodalFusionConfig {
                text_dim: 512,
                vision_dim: 512,
                output_dim: 512,
                strategy: FusionStrategy::Add,
            },
            enable_multimodal: false,
        }
    }
}