use burn::prelude::*;
use std::sync::{Arc, Mutex};
use std::cell::UnsafeCell;
use crate::SageError;

pub struct LazyModel<B: Backend> {
    inner: UnsafeCell<InnerLazyModel<B>>,
}

enum InnerLazyModel<B: Backend> {
    Unloaded {
        config: crate::model::ModelConfig,
        model_path: String,
    },
    Loaded {
        model: Arc<Mutex<crate::model::Model<B>>>,
    },
}

unsafe impl<B: Backend> Sync for LazyModel<B> {}
unsafe impl<B: Backend> Send for LazyModel<B> {}

impl<B: Backend> LazyModel<B> {
    pub fn new(config: crate::model::ModelConfig, model_path: String) -> Self {
        Self {
            inner: UnsafeCell::new(InnerLazyModel::Unloaded { config, model_path }),
        }
    }

    pub fn get_model(&self, device: &B::Device) -> Arc<Mutex<crate::model::Model<B>>> {
        let inner = unsafe { &mut *self.inner.get() };
        match inner {
            InnerLazyModel::Loaded { model } => model.clone(),
            InnerLazyModel::Unloaded { config, model_path } => {
                log::info!("懒加载模型权重...");
                let start_time = std::time::Instant::now();
                let model = config
                    .init::<B>(device)
                    .load_file(&model_path, &burn::record::CompactRecorder::new(), device)
                    .map_err(|e| {
                        let error_msg = format!("模型加载失败: {}", e);
                        log::error!("{}", error_msg);
                        SageError::ModelLoadingError(error_msg)
                    })
                    .unwrap();
                let load_duration = start_time.elapsed();
                log::info!("模型懒加载完成！耗时: {:?}", load_duration);
                
                let model = Arc::new(Mutex::new(model));
                *inner = InnerLazyModel::Loaded { model: model.clone() };
                model
            }
        }
    }
}
