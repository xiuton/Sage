use burn::prelude::*;
use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use std::cell::UnsafeCell;

pub struct LazyModel<B: Backend> {
    inner: UnsafeCell<InnerLazyModel<B>>,
}

enum InnerLazyModel<B: Backend> {
    Unloaded {
        config: crate::core::model::ModelConfig,
        model_path: String,
    },
    Loaded {
        models: Vec<Arc<Mutex<crate::core::model::Model<B>>> >,
        next_index: AtomicUsize,
    },
}

unsafe impl<B: Backend> Sync for LazyModel<B> {}
unsafe impl<B: Backend> Send for LazyModel<B> {}

impl<B: Backend> LazyModel<B> {
    pub fn new(config: crate::core::model::ModelConfig, model_path: String) -> Self {
        Self {
            inner: UnsafeCell::new(InnerLazyModel::Unloaded { config, model_path }),
        }
    }

    pub fn get_model(&self, device: &B::Device) -> Arc<Mutex<crate::core::model::Model<B>>> {
        let inner = unsafe { &mut *self.inner.get() };
        match inner {
            InnerLazyModel::Loaded { models, next_index } => {
                let index = next_index.fetch_add(1, Ordering::SeqCst) % models.len();
                models[index].clone()
            }
            InnerLazyModel::Unloaded { config, model_path } => {
                log::info!("懒加载模型权重...");
                let start_time = std::time::Instant::now();
                
                // 尝试加载模型文件，如果不存在则创建一个默认模型
                let model = match config
                    .init::<B>(device)
                    .load_file(&model_path, &burn::record::CompactRecorder::new(), device)
                {
                    Ok(model) => {
                        log::info!("模型文件加载成功");
                        model
                    }
                    Err(e) => {
                        log::warn!("模型文件不存在，创建默认模型: {}", e);
                        config.init::<B>(device)
                    }
                };
                
                let load_duration = start_time.elapsed();
                log::info!("模型懒加载完成！耗时: {:?}", load_duration);
                
                // 创建多个模型实例以支持并发
                let model_count = 4; // 默认为4个实例
                let mut models = Vec::with_capacity(model_count);
                for _ in 0..model_count {
                    let model_clone = model.clone();
                    models.push(Arc::new(Mutex::new(model_clone)));
                }
                
                let model = models[0].clone();
                *inner = InnerLazyModel::Loaded { models, next_index: AtomicUsize::new(1) };
                model
            }
        }
    }
}
