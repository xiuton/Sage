use burn::prelude::*;

pub struct KVCache<B: Backend> {
    pub key_cache: Vec<Tensor<B, 4>>,
    pub value_cache: Vec<Tensor<B, 4>>,
}

impl<B: Backend> Default for KVCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> KVCache<B> {
    pub fn new() -> Self {
        Self {
            key_cache: Vec::new(),
            value_cache: Vec::new(),
        }
    }

    pub fn update(&mut self, key: Tensor<B, 4>, value: Tensor<B, 4>) {
        self.key_cache.push(key);
        self.value_cache.push(value);
    }

    pub fn clear(&mut self) {
        self.key_cache.clear();
        self.value_cache.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.key_cache.is_empty()
    }

    pub fn get_combined_keys(&self) -> Option<Tensor<B, 4>> {
        if self.key_cache.is_empty() {
            return None;
        }
        
        Some(Tensor::cat(self.key_cache.clone(), 2))
    }

    pub fn get_combined_values(&self) -> Option<Tensor<B, 4>> {
        if self.value_cache.is_empty() {
            return None;
        }
        
        Some(Tensor::cat(self.value_cache.clone(), 2))
    }
}
