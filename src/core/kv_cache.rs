use burn::prelude::*;

/// 单个 Transformer 层的 KV Cache
pub struct LayerKVCache<B: Backend> {
    key_cache: Option<Tensor<B, 4>>,
    value_cache: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for LayerKVCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> LayerKVCache<B> {
    pub fn new() -> Self {
        Self {
            key_cache: None,
            value_cache: None,
        }
    }

    pub fn update(&mut self, new_key: Tensor<B, 4>, new_value: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (key, value) = match (&self.key_cache, &self.value_cache) {
            (Some(key), Some(value)) => {
                let key = Tensor::cat(vec![key.clone(), new_key], 2);
                let value = Tensor::cat(vec![value.clone(), new_value], 2);
                (key, value)
            }
            _ => (new_key, new_value),
        };
        
        self.key_cache = Some(key.clone());
        self.value_cache = Some(value.clone());
        
        (key, value)
    }

    pub fn get(&self) -> Option<(Tensor<B, 4>, Tensor<B, 4>)> {
        match (&self.key_cache, &self.value_cache) {
            (Some(key), Some(value)) => Some((key.clone(), value.clone())),
            _ => None,
        }
    }

    pub fn clear(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
    }

    pub fn is_empty(&self) -> bool {
        self.key_cache.is_none()
    }
}

/// 完整的 KV Cache，用于存储所有层的 key 和 value
pub struct KVCache<B: Backend> {
    layer_caches: Vec<LayerKVCache<B>>,
    cached_seq_len: usize,
}

impl<B: Backend> Default for KVCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> KVCache<B> {
    pub fn new() -> Self {
        Self {
            layer_caches: Vec::new(),
            cached_seq_len: 0,
        }
    }

    pub fn with_capacity(n_layers: usize) -> Self {
        Self {
            layer_caches: (0..n_layers).map(|_| LayerKVCache::new()).collect(),
            cached_seq_len: 0,
        }
    }

    pub fn get_cached_seq_len(&self) -> usize {
        self.cached_seq_len
    }

    pub fn set_cached_seq_len(&mut self, len: usize) {
        self.cached_seq_len = len;
    }

    pub fn get_layer_cache(&mut self, layer_idx: usize) -> &mut LayerKVCache<B> {
        while self.layer_caches.len() <= layer_idx {
            self.layer_caches.push(LayerKVCache::new());
        }
        &mut self.layer_caches[layer_idx]
    }

    pub fn clear(&mut self) {
        for cache in &mut self.layer_caches {
            cache.clear();
        }
        self.cached_seq_len = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.layer_caches.is_empty() || self.layer_caches.iter().all(|c| c.is_empty())
    }
}
