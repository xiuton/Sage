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

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use crate::data::{MmapTextDataset, TextBatcher, TextDataset};
use crate::model::{Model, ModelConfig};
use crate::tokenizer::Tokenizer;

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
    #[config(default = false)]
    pub no_progress: bool,
}

fn run_training<B: AutodiffBackend>(
    artifact_dir: &str,
    config: &TrainingConfig,
    device: &B::Device,
    tokenizer: &Tokenizer,
    init_model: Option<Model<B>>,
    dataloader_train: Arc<dyn burn::data::dataloader::DataLoader<crate::data::TextBatch<B>>>,
    dataloader_valid: Arc<dyn burn::data::dataloader::DataLoader<crate::data::TextBatch<B::InnerBackend>>>,
    total_items: usize,
) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{}/config.json", artifact_dir))
        .expect("Config should be saved");

    tokenizer
        .save(&format!("{}/tokenizer.json", artifact_dir))
        .expect("Tokenizer should be saved");

    B::seed(config.seed);

    println!(
        "批量大小: {}, 序列长度: {}",
        config.batch_size, config.model.max_seq_len
    );
    println!("工作线程数: {}", config.num_workers);
    println!("总训练样本数: {}", total_items);

    let mut learner_builder = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs);

    if !config.no_progress {
        learner_builder = learner_builder.summary();
    }

    let learner = learner_builder.build(
        init_model.unwrap_or_else(|| config.model.init::<B>(device)),
        config.optimizer.init(),
        config.lr,
    );

    let start_time = Instant::now();
    let model_trained = learner.fit(dataloader_train, dataloader_valid);
    let elapsed = start_time.elapsed();

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

    print_training_stats(elapsed, total_items, config.num_epochs, artifact_dir);
}

fn print_training_stats(elapsed: std::time::Duration, total_items: usize, num_epochs: usize, artifact_dir: &str) {
    let items_per_second = total_items as f64 / elapsed.as_secs_f64();
    println!("\n性能统计:");
    println!("总训练时间: {:?}", elapsed);
    println!("总处理样本数: {}", total_items);
    println!("处理速度: {:.2} samples/sec", items_per_second);
    println!("每轮平均时间: {:?}", elapsed / num_epochs as u32);

    if let Some(last_epoch_loss) = find_last_epoch_loss(Path::new(artifact_dir)) {
        let perplexity = last_epoch_loss.exp();
        println!("最后一轮训练损失: {:.4}", last_epoch_loss);
        println!("Perplexity: {:.4}", perplexity);
    }
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
    let n_tokens = tokens.len();
    let train_split = (n_tokens as f32 * 0.9) as usize;

    let tokens_train = tokens[..train_split].to_vec();
    let tokens_test = tokens[train_split..].to_vec();
    let mask_train = mask[..train_split].to_vec();
    let mask_test = mask[train_split..].to_vec();

    println!(
        "训练数据: {} tokens, 验证数据: {} tokens",
        tokens_train.len(),
        tokens_test.len()
    );

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

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TextDataset::new(
            tokens_test,
            mask_test,
            config.model.max_seq_len,
        ));

    run_training(
        artifact_dir,
        &config,
        &device,
        tokenizer,
        init_model,
        dataloader_train,
        dataloader_valid,
        n_tokens,
    );
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
    let dataset_full = MmapTextDataset::open(tokens_path, mask_path, config.model.max_seq_len);
    let n_tokens = dataset_full.total_tokens();
    let train_split = (n_tokens as f32 * 0.9) as usize;

    let dataset_train = dataset_full.with_range(0, train_split);
    let dataset_test = dataset_full.with_range(train_split, n_tokens);

    println!(
        "训练数据: {} tokens, 验证数据: {} tokens",
        train_split,
        n_tokens - train_split
    );

    let batcher_train = TextBatcher::<B>::new(device.clone());
    let batcher_valid = TextBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);

    run_training(
        artifact_dir,
        &config,
        &device,
        tokenizer,
        init_model,
        dataloader_train,
        dataloader_valid,
        n_tokens,
    );
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
    let total_items = dataloader_train.num_items();
    run_training(
        artifact_dir,
        &config,
        &device,
        tokenizer,
        init_model,
        dataloader_train,
        dataloader_valid,
        total_items,
    );
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

fn find_last_epoch_loss(artifact_dir: &Path) -> Option<f64> {
    let train_dir = artifact_dir.join("train");
    let entries = fs::read_dir(&train_dir).ok()?;

    let mut max_epoch = None;
    let mut max_epoch_path = None;

    for entry in entries {
        let entry = entry.ok()?;
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy();
        if let Some(epoch_str) = name.strip_prefix("epoch-")
            && let Ok(epoch) = epoch_str.parse::<usize>()
            && max_epoch.map(|e| epoch > e).unwrap_or(true)
        {
            max_epoch = Some(epoch);
            max_epoch_path = Some(path);
        }
    }

    max_epoch_path.and_then(|path| {
        let loss_path = path.join("Loss.log");
        read_last_loss(&loss_path)
    })
}
