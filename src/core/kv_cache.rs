use burn::prelude::*;

pub struct KVCache<B: Backend> {
    pub key_cache: Vec<Tensor<B, 4>>,
    pub value_cache: Vec<Tensor<B, 4>>,
    pub cached_seq_len: usize,
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
            cached_seq_len: 0,
        }
    }

    pub fn update(&mut self, key: Tensor<B, 4>, value: Tensor<B, 4>) {
        let [_, _, seq_len, _] = key.dims();
        self.key_cache.push(key);
        self.value_cache.push(value);
        self.cached_seq_len += seq_len;
    }

    pub fn clear(&mut self) {
        self.key_cache.clear();
        self.value_cache.clear();
        self.cached_seq_len = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.key_cache.is_empty()
    }

    pub fn get_cached_seq_len(&self) -> usize {
        self.cached_seq_len
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
