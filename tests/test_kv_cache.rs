use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::prelude::*;
use sage::kv_cache::KVCache;

#[test]
fn test_kv_cache_new() {
    let cache = KVCache::<NdArray>::new();
    assert!(cache.is_empty());
    assert_eq!(cache.get_cached_seq_len(), 0);
}

#[test]
fn test_kv_cache_with_capacity() {
    let cache = KVCache::<NdArray>::with_capacity(4);
    assert!(cache.is_empty());
    assert_eq!(cache.get_cached_seq_len(), 0);
}

#[test]
fn test_layer_cache_update() {
    let device = NdArrayDevice::Cpu;
    let mut cache = KVCache::<NdArray>::new();

    let key = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);
    let value = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);

    let layer_cache = cache.get_layer_cache(0);
    let (out_key, out_value) = layer_cache.update(key.clone(), value.clone());

    assert_eq!(out_key.dims(), [1, 2, 1, 32]);
    assert_eq!(out_value.dims(), [1, 2, 1, 32]);
    assert!(!layer_cache.is_empty());
}

#[test]
fn test_kv_cache_clear() {
    let device = NdArrayDevice::Cpu;
    let mut cache = KVCache::<NdArray>::new();

    let key = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);
    let value = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);

    let layer_cache = cache.get_layer_cache(0);
    layer_cache.update(key, value);
    assert!(!layer_cache.is_empty());

    cache.clear();
    assert!(cache.is_empty());
}

#[test]
fn test_kv_cache_multi_layer() {
    let device = NdArrayDevice::Cpu;
    let mut cache = KVCache::<NdArray>::new();

    let key = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);
    let value = Tensor::<NdArray, 4>::zeros([1, 2, 1, 32], &device);

    cache.get_layer_cache(0).update(key.clone(), value.clone());
    cache.get_layer_cache(1).update(key.clone(), value.clone());

    assert!(!cache.is_empty());

    let layer0 = cache.get_layer_cache(0);
    assert!(!layer0.is_empty());

    let layer1 = cache.get_layer_cache(1);
    assert!(!layer1.is_empty());
}

#[test]
fn test_kv_cache_seq_len() {
    let mut cache = KVCache::<NdArray>::new();
    assert_eq!(cache.get_cached_seq_len(), 0);

    cache.set_cached_seq_len(42);
    assert_eq!(cache.get_cached_seq_len(), 42);
}
