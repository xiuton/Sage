use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::prelude::*;
use sage::kv_cache::KVCache;

#[test]
fn test_kv_cache_new() {
    let cache = KVCache::<NdArray>::new();
    
    assert!(cache.key_cache.is_empty());
    assert!(cache.value_cache.is_empty());
}

#[test]
fn test_kv_cache_update() {
    let device = NdArrayDevice::Cpu;
    let mut cache = KVCache::<NdArray>::new();
    
    // 创建测试张量 [batch_size, num_heads, seq_len, head_dim]
    let key = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);
    let value = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);
    
    cache.update(key, value);
    
    assert_eq!(cache.key_cache.len(), 1);
    assert_eq!(cache.value_cache.len(), 1);
    assert_eq!(cache.key_cache[0].dims(), [1, 2, 1, 32]);
}

#[test]
fn test_kv_cache_get_combined_keys() {
    let device = NdArrayDevice::Cpu;
    let mut cache = KVCache::<NdArray>::new();
    
    // 添加第一个键值对
    let key1 = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);
    let value1 = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);
    cache.update(key1, value1);
    
    // 添加第二个键值对
    let key2 = Tensor::<NdArray, 4>::ones([1, 2, 1, 32], &device);
    let value2 = Tensor::<NdArray, 4>::ones([1, 2, 1, 32], &device);
    cache.update(key2, value2);
    
    // 获取合并后的键
    let combined_keys = cache.get_combined_keys().unwrap();
    assert_eq!(combined_keys.dims(), [1, 2, 2, 32]);
}

#[test]
fn test_kv_cache_clear() {
    let device = NdArrayDevice::Cpu;
    let mut cache = KVCache::<NdArray>::new();
    
    let key = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);
    let value = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);
    cache.update(key, value);
    
    assert!(!cache.key_cache.is_empty());
    
    cache.clear();
    
    assert!(cache.key_cache.is_empty());
    assert!(cache.value_cache.is_empty());
}
