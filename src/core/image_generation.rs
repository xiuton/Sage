use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        Dropout, DropoutConfig,
        Linear, LinearConfig,
        BatchNorm, BatchNormConfig,
        activation::Gelu,
    },
    prelude::*,
    tensor::backend::Backend,
};
use serde::{Serialize, Deserialize};
use image::GenericImageView;

pub fn load_image_as_tensor(path: &str, target_size: usize) -> Result<Vec<f32>, String> {
    let img = image::open(path).map_err(|e| format!("Failed to open image {}: {}", path, e))?;
    
    let (width, height) = img.dimensions();
    let resized = if width != target_size as u32 || height != target_size as u32 {
        img.resize_exact(target_size as u32, target_size as u32, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    
    let rgb = resized.to_rgb8();
    let pixels = rgb.as_raw();
    
    let mut data = Vec::with_capacity(3 * target_size * target_size);
    for pixel in pixels.chunks(3) {
        let r = pixel[0] as f32 / 127.5 - 1.0;
        let g = pixel[1] as f32 / 127.5 - 1.0;
        let b = pixel[2] as f32 / 127.5 - 1.0;
        data.push(r);
        data.push(g);
        data.push(b);
    }
    
    Ok(data)
}

#[derive(Debug, Clone)]
pub struct SimpleTokenizer {
    vocab: Vec<String>,
    char_to_id: std::collections::HashMap<char, usize>,
}

impl SimpleTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        let vocab: Vec<String> = (0..vocab_size)
            .map(|i| if i < 26 {
                format!("{}", ('a' as u8 + i as u8) as char)
            } else if i < 52 {
                format!("{}", ('A' as u8 + (i - 26) as u8) as char)
            } else if i < 62 {
                format!("{}", ('0' as u8 + (i - 52) as u8) as char)
            } else {
                format!("tok{}", i)
            })
            .collect();

        let mut char_to_id = std::collections::HashMap::new();
        for (i, s) in vocab.iter().take(62).enumerate() {
            if let Some(c) = s.chars().next() {
                char_to_id.insert(c, i);
            }
        }

        Self { vocab, char_to_id }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        for c in text.chars() {
            if let Some(&id) = self.char_to_id.get(&c) {
                tokens.push(id);
            } else {
                tokens.push(self.vocab.len() - 1);
            }
        }
        if tokens.is_empty() {
            tokens.push(0);
        }
        tokens
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VAEConfig {
    pub image_channels: usize,
    pub latent_dim: usize,
    pub hidden_channels: usize,
    pub image_size: usize,
}

impl Default for VAEConfig {
    fn default() -> Self {
        Self {
            image_channels: 3,
            latent_dim: 128,
            hidden_channels: 64,
            image_size: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionConfig {
    pub image_size: usize,
    pub in_channels: usize,
    pub hidden_channels: usize,
    pub num_timesteps: usize,
    pub latent_dim: usize,
    pub beta_start: f32,
    pub beta_end: f32,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            image_size: 64,
            in_channels: 3,
            hidden_channels: 128,
            num_timesteps: 1000,
            latent_dim: 128,
            beta_start: 0.0001,
            beta_end: 0.02,
        }
    }
}

#[derive(Module, Debug)]
pub struct VAEEncoder<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B>,
    conv4: Conv2d<B>,
    bn4: BatchNorm<B>,
    fc_mu: Conv2d<B>,
    fc_log_var: Conv2d<B>,
}

impl<B: Backend> VAEEncoder<B> {
    pub fn new(config: &VAEConfig, device: &B::Device) -> Self {
        let hidden = config.hidden_channels;
        let latent = config.latent_dim;
        let channels = config.image_channels;

        let conv1 = Conv2dConfig::new([channels, hidden], [4, 4])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(hidden).init(device);

        let conv2 = Conv2dConfig::new([hidden, hidden * 2], [4, 4])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn2 = BatchNormConfig::new(hidden * 2).init(device);

        let conv3 = Conv2dConfig::new([hidden * 2, hidden * 4], [4, 4])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn3 = BatchNormConfig::new(hidden * 4).init(device);

        let conv4 = Conv2dConfig::new([hidden * 4, hidden * 8], [4, 4])
            .with_stride([2, 2])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn4 = BatchNormConfig::new(hidden * 8).init(device);

        // 输出 2 * latent 个通道的 4D 特征图，这样我们可以直接输出 mu 和 log_var 作为 4D 张量
        let conv_mu = Conv2dConfig::new([hidden * 8, latent], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let conv_log_var = Conv2dConfig::new([hidden * 8, latent], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        Self {
            conv1, bn1, conv2, bn2, conv3, bn3, conv4, bn4,
            fc_mu: conv_mu, fc_log_var: conv_log_var,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = burn::nn::activation::Gelu::new().forward(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = burn::nn::activation::Gelu::new().forward(x);

        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = burn::nn::activation::Gelu::new().forward(x);

        let x = self.conv4.forward(x);
        let x = self.bn4.forward(x);
        let x = burn::nn::activation::Gelu::new().forward(x);

        let mu = self.fc_mu.forward(x.clone());
        let log_var = self.fc_log_var.forward(x);

        (mu, log_var)
    }
}

#[derive(Module, Debug)]
pub struct VAEDecoder<B: Backend> {
    conv_in: Conv2d<B>,
    conv1: ConvTranspose2d<B>,
    bn1: BatchNorm<B>,
    conv2: ConvTranspose2d<B>,
    bn2: BatchNorm<B>,
    conv3: ConvTranspose2d<B>,
    bn3: BatchNorm<B>,
    conv4: ConvTranspose2d<B>,
    bn4: BatchNorm<B>,
}

impl<B: Backend> VAEDecoder<B> {
    pub fn new(config: &VAEConfig, device: &B::Device) -> Self {
        let hidden = config.hidden_channels;
        let latent = config.latent_dim;

        let conv_in = Conv2dConfig::new([latent, hidden * 8], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let conv1 = ConvTranspose2dConfig::new([hidden * 8, hidden * 4], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);
        let bn1 = BatchNormConfig::new(hidden * 4).init(device);

        let conv2 = ConvTranspose2dConfig::new([hidden * 4, hidden * 2], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);
        let bn2 = BatchNormConfig::new(hidden * 2).init(device);

        let conv3 = ConvTranspose2dConfig::new([hidden * 2, hidden], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);
        let bn3 = BatchNormConfig::new(hidden).init(device);

        let conv4 = ConvTranspose2dConfig::new([hidden, 3], [4, 4])
            .with_stride([2, 2])
            .with_padding([1, 1])
            .init(device);
        let bn4 = BatchNormConfig::new(3).init(device);

        Self {
            conv_in, conv1, bn1, conv2, bn2, conv3, bn3, conv4, bn4,
        }
    }

    pub fn forward(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.conv_in.forward(z);
        x = self.conv1.forward(x);
        x = self.bn1.forward(x);
        x = Gelu::new().forward(x);

        x = self.conv2.forward(x);
        x = self.bn2.forward(x);
        x = Gelu::new().forward(x);

        x = self.conv3.forward(x);
        x = self.bn3.forward(x);
        x = Gelu::new().forward(x);

        x = self.conv4.forward(x);
        x = self.bn4.forward(x);

        x
    }
}

#[derive(Module, Debug)]
pub struct VAE<B: Backend> {
    pub encoder: VAEEncoder<B>,
    pub decoder: VAEDecoder<B>,
    latent_dim: usize,
    image_size: usize,
}

impl<B: Backend> VAE<B> {
    pub fn new(config: &VAEConfig, device: &B::Device) -> Self {
        let encoder = VAEEncoder::new(config, device);
        let decoder = VAEDecoder::new(config, device);
        Self {
            encoder,
            decoder,
            latent_dim: config.latent_dim,
            image_size: config.image_size,
        }
    }

    pub fn reparameterize(&self, mu: Tensor<B, 4>, log_var: Tensor<B, 4>) -> Tensor<B, 4> {
        let std = log_var.clone().mul_scalar(0.5).exp();
        let epsilon: Tensor<B, 4> = Tensor::zeros(mu.dims(), &mu.device());
        mu + std * epsilon
    }

    pub fn encode(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        self.encoder.forward(x)
    }

    pub fn decode(&self, z: Tensor<B, 4>) -> Tensor<B, 4> {
        self.decoder.forward(z)
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let (mu, log_var) = self.encode(x.clone());
        let z = self.reparameterize(mu.clone(), log_var.clone());
        let recon = self.decode(z);
        (recon, mu, log_var)
    }

    pub fn generate_random(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 4> {
        let z: Tensor<B, 4> = Tensor::zeros([batch_size, self.latent_dim, 4, 4], device);
        self.decode(z)
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn image_size(&self) -> usize {
        self.image_size
    }
}

#[derive(Module, Debug)]
pub struct TimeEmbedding<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> TimeEmbedding<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(dim, dim * 4).init(device);
        let linear2 = LinearConfig::new(dim * 4, dim).init(device);
        Self { linear1, linear2 }
    }

    pub fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(t);
        let x = Gelu::new().forward(x);
        self.linear2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct UNetBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    time_mlp: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> UNetBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, time_dim: usize, device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let time_mlp = LinearConfig::new(time_dim, out_channels).init(device);
        let dropout = DropoutConfig::new(0.1).init();

        Self { conv1, conv2, time_mlp, dropout }
    }

    pub fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        let h = self.conv1.forward(x);

        let t_out = self.time_mlp.forward(t_emb);
        let c = t_out.dims()[1];
        let batch_size = t_out.dims()[0];
        let t_out = t_out.reshape([batch_size, c, 1, 1]);
        let h = h + t_out;

        let h = Gelu::new().forward(h);
        let h = self.dropout.forward(h);

        let h = self.conv2.forward(h);
        let h = Gelu::new().forward(h);

        h
    }
}

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    time_embedding: TimeEmbedding<B>,
    down_blocks: Vec<UNetBlock<B>>,
    mid_conv1: Conv2d<B>,
    mid_conv2: Conv2d<B>,
    up_blocks: Vec<UNetBlock<B>>,
    final_conv: Conv2d<B>,
}

impl<B: Backend> UNet<B> {
    pub fn new(config: &DiffusionConfig, device: &B::Device) -> Self {
        let hidden = config.hidden_channels;
        let latent = config.latent_dim;
        let time_dim = latent;

        let time_embedding = TimeEmbedding::new(time_dim, device);

        let mut down_blocks = Vec::new();
        let channels = [latent, hidden, hidden * 2, hidden * 4]; // 减小通道数
        for i in 0..channels.len() - 1 {
            down_blocks.push(UNetBlock::new(channels[i], channels[i + 1], time_dim, device));
        }

        let mid_conv1 = Conv2dConfig::new([hidden * 4, hidden * 4], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let mid_conv2 = Conv2dConfig::new([hidden * 4, hidden * 4], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        let mut up_blocks = Vec::new();
        let up_channels = [hidden * 4, hidden * 2, hidden, hidden / 2]; // 减小通道数
        for i in 0..up_channels.len() - 1 {
            up_blocks.push(UNetBlock::new(up_channels[i], up_channels[i + 1], time_dim, device));
        }

        let final_conv = Conv2dConfig::new([hidden / 2, latent], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);

        Self {
            time_embedding,
            down_blocks,
            mid_conv1, mid_conv2,
            up_blocks,
            final_conv,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, t: Tensor<B, 2>) -> Tensor<B, 4> {
        let t_emb = self.time_embedding.forward(t);

        let mut h = x;

        for block in &self.down_blocks {
            h = block.forward(h, t_emb.clone());
        }

        h = self.mid_conv1.forward(h);
        h = Gelu::new().forward(h);
        h = self.mid_conv2.forward(h);
        h = Gelu::new().forward(h);

        for block in &self.up_blocks {
            h = block.forward(h, t_emb.clone());
        }

        h = self.final_conv.forward(h);
        h
    }
}

#[derive(Module, Debug)]
pub struct DiffusionModel<B: Backend> {
    pub vae: VAE<B>,
    pub unet: UNet<B>,
    latent_dim: usize,
    image_size: usize,
    num_timesteps: usize,
    betas: Vec<f32>,
    alphas: Vec<f32>,
    alpha_bar: Vec<f32>,
}

impl<B: Backend> DiffusionModel<B> {
    pub fn from_file<P: AsRef<std::path::Path>>(config: &DiffusionConfig, device: &B::Device, path: P) -> Result<Self, String> {
        use burn::record::{CompactRecorder, Recorder};
        
        let mut model = Self::new(config, device);
        
        let path_buf = path.as_ref().to_path_buf();
        let record = CompactRecorder::new()
            .load(path_buf, device)
            .map_err(|e| format!("Failed to load model: {}", e))?;
        
        model = model.load_record(record);
        
        Ok(model)
    }

    pub fn new(config: &DiffusionConfig, device: &B::Device) -> Self {
        let vae_config = VAEConfig {
            image_channels: config.in_channels,
            latent_dim: config.latent_dim,
            hidden_channels: config.hidden_channels / 2,
            image_size: config.image_size,
        };
        let vae = VAE::new(&vae_config, device);
        let unet = UNet::new(config, device);

        let timesteps = config.num_timesteps;
        let beta_start = config.beta_start;
        let beta_end = config.beta_end;

        let betas: Vec<f32> = (0..timesteps)
            .map(|t| beta_start + (beta_end - beta_start) * (t as f32 / timesteps as f32))
            .collect();

        let alphas: Vec<f32> = betas.iter().map(|b| 1.0 - b).collect();

        let mut alpha_bar_data = vec![1.0f32];
        for i in 1..timesteps {
            alpha_bar_data.push(alpha_bar_data[i - 1] * alphas[i]);
        }

        Self {
            vae,
            unet,
            latent_dim: config.latent_dim,
            image_size: config.image_size,
            num_timesteps: config.num_timesteps,
            betas,
            alphas,
            alpha_bar: alpha_bar_data,
        }
    }

    pub fn get_alpha_bar(&self, t: usize) -> f32 {
        if t < self.alpha_bar.len() {
            self.alpha_bar[t]
        } else {
            0.0
        }
    }

    pub fn vae(&self) -> &VAE<B> {
        &self.vae
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn image_size(&self) -> usize {
        self.image_size
    }

    pub fn num_timesteps(&self) -> usize {
        self.num_timesteps
    }

    pub fn denoise_step(&self, x: Tensor<B, 4>, t: usize, device: &B::Device) -> Tensor<B, 4> {
        self.denoise_step_with_condition(x, t, None, device)
    }

    pub fn denoise_step_with_condition(&self, x: Tensor<B, 4>, t: usize, condition: Option<Tensor<B, 2>>, device: &B::Device) -> Tensor<B, 4> {
        let batch_size = x.dims()[0];
        let time_dim = self.latent_dim;
        let t_tensor: Tensor<B, 2> = Tensor::full([batch_size, time_dim], t as i64, device);

        let mut noise_pred = self.unet.forward(x.clone(), t_tensor);

        if let Some(ref cond) = condition {
            let [b, c, h, w] = noise_pred.dims();
            let cond_4d = cond.clone().reshape([b, 1, 1, c]).repeat(&[1, c, h, w]);
            let cond_scaled = cond_4d.slice([0..b, 0..c, 0..h, 0..w]);
            noise_pred = noise_pred + cond_scaled;
        }

        let alpha_bar_t = self.get_alpha_bar(t);
        let alpha_t = if t > 0 { 1.0 - self.get_alpha_bar(t - 1) } else { 1.0 };
        let beta_t = 1.0 - alpha_t;

        let sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).sqrt();

        let coef1 = beta_t / sqrt_one_minus_alpha_bar;
        let coef2 = alpha_t.sqrt();

        if t == 0 {
            (x - coef1 * noise_pred) / coef2
        } else {
            let [b, c, h, w] = x.dims();
            let noise: Tensor<B, 4> = Tensor::zeros([b, c, h, w], device);
            (x - coef1 * noise_pred) / coef2 + beta_t.sqrt() * noise
        }
    }
}

pub struct ImageGenerator<B: Backend> {
    diffusion: DiffusionModel<B>,
    device: B::Device,
}

impl<B: Backend> ImageGenerator<B> {
    pub fn new(config: DiffusionConfig, device: B::Device) -> Self {
        let diffusion = DiffusionModel::new(&config, &device);
        Self { diffusion, device }
    }

    pub fn from_file<P: AsRef<std::path::Path>>(config: DiffusionConfig, device: B::Device, model_path: P) -> Result<Self, String> {
        let diffusion = DiffusionModel::from_file(&config, &device, model_path)?;
        Ok(Self { diffusion, device })
    }

    pub fn generate_simple(&self, batch_size: usize, _image_size: usize) -> Tensor<B, 4> {
        let vae = self.diffusion.vae();
        vae.generate_random(batch_size, &self.device)
    }

    pub fn generate_with_prompt(&self, prompt: &str, _tokenizer: &SimpleTokenizer, batch_size: usize, _image_size: usize, steps: usize) -> Tensor<B, 4> {
        let latent_dim = self.diffusion.latent_dim();

        let hash: u32 = prompt.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
        let hash_f: f32 = (hash % 1000) as f32 / 1000.0;
        let cond_tensor: Tensor<B, 2> = Tensor::full([batch_size, latent_dim], hash_f, &self.device);

        self.generate_with_condition(batch_size, 8, steps, Some(cond_tensor))
    }

    pub fn generate(&self, batch_size: usize, _image_size: usize, steps: usize) -> Tensor<B, 4> {
        self.generate_with_condition(batch_size, 8, steps, None)
    }

    fn generate_with_condition(&self, batch_size: usize, latent_h: usize, steps: usize, condition: Option<Tensor<B, 2>>) -> Tensor<B, 4> {
        let latent_dim = self.diffusion.latent_dim();
        let mut x: Tensor<B, 4> = Tensor::zeros([batch_size, latent_dim, latent_h, latent_h], &self.device);

        let cond_clone = condition.clone();
        if let Some(ref cond) = condition {
            let cond_expanded = cond.clone().reshape([batch_size, 1, 1, latent_dim]);
            let cond_expanded = cond_expanded.repeat(&[1, latent_dim, latent_h, latent_h]);
            x = x + cond_expanded.slice([0..batch_size, 0..latent_dim, 0..latent_h, 0..latent_h]);
        }

        let num_timesteps = self.diffusion.num_timesteps();
        let start_step = if steps > num_timesteps { 0 } else { num_timesteps - steps };

        for t in (start_step..num_timesteps).rev() {
            x = self.diffusion.denoise_step_with_condition(x, t, cond_clone.clone(), &self.device);
        }

        // 对整个批次生成图像
        self.diffusion.vae().decode(x)
    }
}