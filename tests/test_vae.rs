#[cfg(test)]
mod tests {
    use burn::nn::{Conv2d, Conv2dConfig, BatchNorm, BatchNormConfig};
    use burn::prelude::*;
    use burn::nn::activation::Gelu;

    #[derive(Config, Debug)]
    pub struct VAEConfig {
        pub image_channels: usize,
        pub hidden_channels: usize,
        pub latent_dim: usize,
        pub image_size: usize,
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

            let fc_mu = Conv2dConfig::new([hidden * 8, latent], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device);
            let fc_log_var = Conv2dConfig::new([hidden * 8, latent], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device);

            Self {
                conv1, bn1, conv2, bn2, conv3, bn3, conv4, bn4,
                fc_mu, fc_log_var,
            }
        }

        pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
            let x = self.conv1.forward(x);
            let x = self.bn1.forward(x);
            let x = Gelu::new().forward(x);

            let x = self.conv2.forward(x);
            let x = self.bn2.forward(x);
            let x = Gelu::new().forward(x);

            let x = self.conv3.forward(x);
            let x = self.bn3.forward(x);
            let x = Gelu::new().forward(x);

            let x = self.conv4.forward(x);
            let x = self.bn4.forward(x);
            let x = Gelu::new().forward(x);

            let mu = self.fc_mu.forward(x.clone());
            let log_var = self.fc_log_var.forward(x);

            (mu, log_var)
        }
    }

    #[test]
    fn test_vae_encoder_output_shape() {
        let device = Default::default();
        let config = VAEConfig {
            image_channels: 3,
            hidden_channels: 128,
            latent_dim: 128,
            image_size: 64,
        };

        let encoder = VAEEncoder::new(&config, &device);
        let input = Tensor::ones([1, 3, 64, 64], &device);

        let (mu, log_var) = encoder.forward(input);

        assert_eq!(mu.dims(), &[1, 128, 4, 4]);
        assert_eq!(log_var.dims(), &[1, 128, 4, 4]);
    }

    #[test]
    fn test_vae_encoder_latent_dim_64() {
        let device = Default::default();
        let config = VAEConfig {
            image_channels: 3,
            hidden_channels: 64,
            latent_dim: 64,
            image_size: 64,
        };

        let encoder = VAEEncoder::new(&config, &device);
        let input = Tensor::ones([2, 3, 64, 64], &device);

        let (mu, log_var) = encoder.forward(input);

        assert_eq!(mu.dims(), &[2, 64, 4, 4]);
        assert_eq!(log_var.dims(), &[2, 64, 4, 4]);
    }
}