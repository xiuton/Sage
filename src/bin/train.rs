use burn::backend::{
    ndarray::{NdArray, NdArrayDevice},
    Autodiff,
};
use burn::optim::AdamConfig;
use sage::{
    model::ModelConfig,
    tokenizer::Tokenizer,
    training::{self, TrainingConfig},
};
use std::path::Path;

fn main() {
    // 初始化日志
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }
    env_logger::init();

    let artifact_dir = "./tmp/sage_model_formal";
    let corpus_path = "corpus_cn.txt";
    let device = NdArrayDevice::Cpu;

    // 1. 加载或创建语料
    if !Path::new(corpus_path).exists() {
        panic!("语料文件 {} 不存在，请确保文件已创建。", corpus_path);
    }
    let text = std::fs::read_to_string(corpus_path).expect("Should read corpus file");

    // 2. 初始化/加载分词器
    let tokenizer_path = format!("{}/tokenizer.json", artifact_dir);
    let tokenizer = if Path::new(&tokenizer_path).exists() {
        println!("正在加载现有分词器...");
        Tokenizer::load(&tokenizer_path).expect("Should load tokenizer")
    } else {
        println!("正在从语料创建新分词器...");
        Tokenizer::new(&text)
    };
    println!("词表大小: {}", tokenizer.vocab_size);

    // 3. 配置模型
    let model_config = ModelConfig {
        d_model: 128,
        n_layers: 4,
        n_heads: 4,
        d_ff: 512,
        vocab_size: tokenizer.vocab_size,
        max_seq_len: 64,
        dropout: 0.1,
    };

    let model_init = model_config.init::<NdArray>(&device);
    println!("模型参数总量: {} (约 0.001B)", model_init.num_params());

    // 4. 训练流程
    if !Path::new(&format!("{}/model.mpk", artifact_dir)).exists() {
        println!("未发现已训练模型，开始正式训练...");
        let mut training_config = TrainingConfig::new(model_config.clone(), AdamConfig::new());
        training_config.num_epochs = 20;
        training_config.batch_size = 16;
        training_config.lr = 1e-4;

        training::train::<Autodiff<NdArray>>(
            artifact_dir,
            training_config,
            device.clone(),
            &tokenizer,
            &text,
        );
    } else {
        println!("发现已存在模型，跳过训练。");
    }

    println!("\n训练流程完成！模型已保存在 '{}'", artifact_dir);
}
