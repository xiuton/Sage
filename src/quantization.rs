use burn::tensor::backend::Backend;

pub trait QuantizableModule<B: Backend> {
    fn quantize(&self) -> Self;
}

pub struct QuantizationConfig {
    pub bits: usize,
    pub enable: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            enable: false,
        }
    }
}
