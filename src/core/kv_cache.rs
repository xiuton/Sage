use burn::prelude::*;

/// KV Cache 用于加速自回归生成
/// 
/// 注意：当前的 TransformerEncoder 架构不直接支持完整的 KV Cache
/// 但该结构体已为未来的 Decoder-only 架构做好准备
/// 
/// 当前使用的优化：
/// - 位置编码优化：根据缓存长度计算新 token 的位置
pub struct KVCache<B: Backend> {
    /// 按层存储的 key 缓存
    pub key_cache: Vec<Tensor<B, 4>>,
    /// 按层存储的 value 缓存
    pub value_cache: Vec<Tensor<B, 4>>,
    /// 已缓存的序列长度
    pub cached_seq_len: usize,
}

impl<B: Backend> Default for KVCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> KVCache<B> {
    /// 创建新的 KV Cache
    pub fn new() -> Self {
        Self {
            key_cache: Vec::new(),
            value_cache: Vec::new(),
            cached_seq_len: 0,
        }
    }

    /// 更新 KV Cache（未来用于 Decoder 层）
    /// 
    /// # Arguments
    /// * `key` - 形状为 [batch_size, num_heads, seq_len, head_dim]
    /// * `value` - 形状为 [batch_size, num_heads, seq_len, head_dim]
    pub fn update(&mut self, key: Tensor<B, 4>, value: Tensor<B, 4>) {
        let [_, _, seq_len, _] = key.dims();
        self.key_cache.push(key);
        self.value_cache.push(value);
        self.cached_seq_len += seq_len;
    }

    /// 清空 KV Cache
    pub fn clear(&mut self) {
        self.key_cache.clear();
        self.value_cache.clear();
        self.cached_seq_len = 0;
    }

    /// 检查 KV Cache 是否为空
    pub fn is_empty(&self) -> bool {
        self.key_cache.is_empty() && self.cached_seq_len == 0
    }

    /// 获取已缓存的序列长度
    pub fn get_cached_seq_len(&self) -> usize {
        self.cached_seq_len
    }

    /// 拼接所有层的 key 缓存（未来用于完整的 KV Cache）
    pub fn get_combined_keys(&self) -> Option<Tensor<B, 4>> {
        if self.key_cache.is_empty() {
            return None;
        }
        
        Some(Tensor::cat(self.key_cache.clone(), 2))
    }

    /// 拼接所有层的 value 缓存（未来用于完整的 KV Cache）
    pub fn get_combined_values(&self) -> Option<Tensor<B, 4>> {
        if self.value_cache.is_empty() {
            return None;
        }
        
        Some(Tensor::cat(self.value_cache.clone(), 2))
    }

    /// 设置缓存的序列长度（用于位置编码优化）
    pub fn set_cached_seq_len(&mut self, len: usize) {
        self.cached_seq_len = len;
    }
}
