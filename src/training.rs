use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{LossMetric, LearningRateMetric},
        LearnerBuilder,
    },
};

use crate::data::{TextBatcher, TextDataset};
use crate::model::ModelConfig;
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
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: &Tokenizer,
    text: &str,
) {
    std::fs::create_dir_all(artifact_dir).ok();
    config.save(format!("{}/config.json", artifact_dir)).expect("Config should be saved");
    
    // 保存分词器
    tokenizer.save(&format!("{}/tokenizer.json", artifact_dir)).expect("Tokenizer should be saved");

    B::seed(config.seed);

    let tokens = tokenizer.encode(text);
    let n_tokens = tokens.len();
    let train_split = (n_tokens as f32 * 0.9) as usize; // 90% 训练

    let tokens_train = tokens[..train_split].to_vec();
    let tokens_test = tokens[train_split..].to_vec();

    let batcher_train = TextBatcher::<B>::new(device.clone());
    let batcher_valid = TextBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TextDataset::new(tokens_train, config.model.max_seq_len));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TextDataset::new(tokens_test, config.model.max_seq_len));

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.lr,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{}/model", artifact_dir), &CompactRecorder::new())
        .expect("Model should be saved");
}
