use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder,
        metric::{LearningRateMetric, LossMetric},
    },
};

use crate::data::{MmapTextDataset, TextBatcher, TextDataset};
use crate::model::{Model, ModelConfig};
use crate::tokenizer::Tokenizer;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 5.0e-4)]
    pub lr: f64,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: &Tokenizer,
    tokens: Vec<usize>,
    mask: Vec<u8>,
    init_model: Option<Model<B>>,
) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{}/config.json", artifact_dir))
        .expect("Config should be saved");

    // 保存分词器
    tokenizer
        .save(&format!("{}/tokenizer.json", artifact_dir))
        .expect("Tokenizer should be saved");

    B::seed(config.seed);

    let n_tokens = tokens.len();
    let train_split = (n_tokens as f32 * 0.9) as usize; // 90% 训练

    let tokens_train = tokens[..train_split].to_vec();
    let tokens_test = tokens[train_split..].to_vec();
    let mask_train = mask[..train_split].to_vec();
    let mask_test = mask[train_split..].to_vec();

    let batcher_train = TextBatcher::<B>::new(device.clone());
    let batcher_valid = TextBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TextDataset::new(
            tokens_train,
            mask_train,
            config.model.max_seq_len,
        ));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TextDataset::new(
            tokens_test,
            mask_test,
            config.model.max_seq_len,
        ));

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            init_model.unwrap_or_else(|| config.model.init::<B>(&device)),
            config.optimizer.init(),
            config.lr,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{}/model", artifact_dir), &CompactRecorder::new())
        .expect("Model should be saved");

    if let Some(best_epoch) = find_best_epoch(Path::new(artifact_dir)) {
        let from = Path::new(artifact_dir)
            .join("checkpoint")
            .join(format!("model-{}.mpk", best_epoch));
        let to = Path::new(artifact_dir).join("best_model.mpk");
        let _ = fs::copy(from, to);
    }
}

pub fn train_from_cache<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: &Tokenizer,
    tokens_path: &str,
    mask_path: &str,
    init_model: Option<Model<B>>,
) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{}/config.json", artifact_dir))
        .expect("Config should be saved");

    tokenizer
        .save(&format!("{}/tokenizer.json", artifact_dir))
        .expect("Tokenizer should be saved");

    B::seed(config.seed);

    let dataset_full = MmapTextDataset::open(tokens_path, mask_path, config.model.max_seq_len);
    let n_tokens = dataset_full.total_tokens();
    let train_split = (n_tokens as f32 * 0.9) as usize;

    let dataset_train = dataset_full.with_range(0, train_split);
    let dataset_test = dataset_full.with_range(train_split, n_tokens);

    let batcher_train = TextBatcher::<B>::new(device.clone());
    let batcher_valid = TextBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            init_model.unwrap_or_else(|| config.model.init::<B>(&device)),
            config.optimizer.init(),
            config.lr,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{}/model", artifact_dir), &CompactRecorder::new())
        .expect("Model should be saved");

    if let Some(best_epoch) = find_best_epoch(Path::new(artifact_dir)) {
        let from = Path::new(artifact_dir)
            .join("checkpoint")
            .join(format!("model-{}.mpk", best_epoch));
        let to = Path::new(artifact_dir).join("best_model.mpk");
        let _ = fs::copy(from, to);
    }
}

pub fn train_with_loaders<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: &Tokenizer,
    dataloader_train: Arc<dyn burn::data::dataloader::DataLoader<crate::data::TextBatch<B>>>,
    dataloader_valid: Arc<
        dyn burn::data::dataloader::DataLoader<crate::data::TextBatch<B::InnerBackend>>,
    >,
    init_model: Option<Model<B>>,
) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{}/config.json", artifact_dir))
        .expect("Config should be saved");

    tokenizer
        .save(&format!("{}/tokenizer.json", artifact_dir))
        .expect("Tokenizer should be saved");

    B::seed(config.seed);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            init_model.unwrap_or_else(|| config.model.init::<B>(&device)),
            config.optimizer.init(),
            config.lr,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(format!("{}/model", artifact_dir), &CompactRecorder::new())
        .expect("Model should be saved");

    if let Some(best_epoch) = find_best_epoch(Path::new(artifact_dir)) {
        let from = Path::new(artifact_dir)
            .join("checkpoint")
            .join(format!("model-{}.mpk", best_epoch));
        let to = Path::new(artifact_dir).join("best_model.mpk");
        let _ = fs::copy(from, to);
    }
}

fn find_best_epoch(artifact_dir: &Path) -> Option<usize> {
    let valid_dir = artifact_dir.join("valid");
    let mut best: Option<(usize, f64)> = None;

    let entries = fs::read_dir(&valid_dir).ok()?;
    for entry in entries {
        let entry = entry.ok()?;
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy();
        let epoch = name.strip_prefix("epoch-")?.parse::<usize>().ok()?;

        let loss_path = path.join("Loss.log");
        let loss = read_last_loss(&loss_path)?;

        match best {
            None => best = Some((epoch, loss)),
            Some((_, best_loss)) if loss < best_loss => best = Some((epoch, loss)),
            _ => {}
        }
    }

    best.map(|(epoch, _)| epoch)
}

fn read_last_loss(path: &PathBuf) -> Option<f64> {
    let text = fs::read_to_string(path).ok()?;
    let last = text.lines().rev().find(|l| !l.trim().is_empty())?;
    let value = last.split(',').next()?.trim().parse::<f64>().ok()?;
    Some(value)
}
