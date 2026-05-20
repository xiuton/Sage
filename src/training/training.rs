
//! 训练循环引擎
//!
//! 实现完整的 LLM 训练流程：
//! - `train()` - 标准训练循环（LM 预训练 / SFT 微调）
//! - `train_dpo()` - DPO 偏好对齐训练
//! - `train_from_cache()` - 从预构建 token 缓存训练
//! - `train_with_loaders()` - 自定义 DataLoader 训练
//!
//! 训练参数由 [TrainingConfig] 控制，详见 configs/config.rs。

use burn::{
    data::{dataloader::{batcher::Batcher, DataLoaderBuilder}},
    module::AutodiffModule,
    optim::{adaptor::OptimizerAdaptor, Optimizer, GradientsParams, GradientsAccumulator},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};

use crate::training::distributed::{create_device, get_available_devices, train_parallel, DistributedConfig};
use crate::training::dpo::{DPOBatcher, DPOConfig, DPOItem, DPOTrainer};
use crate::configs::config::TrainingConfig;

use std::{
    fs,
    path::Path,
    sync::Arc,
};

use crate::{MmapTextDataset, TextBatch, TextBatcher, TextDataset};
use crate::core::{Model, Tokenizer};
use crate::training::lr_scheduler::LRScheduler;



struct TrainingContext<B: AutodiffBackend> {
    output_dir: String,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: Arc<Tokenizer>,
    init_model: Option<Model<B>>,
    init_optimizer: Option<OptimizerAdaptor<burn::optim::Adam, Model<B>, B>>,
    dataloader_train: Arc<dyn burn::data::dataloader::DataLoader<B, TextBatch<B>>>,
    dataloader_valid: Arc<dyn burn::data::dataloader::DataLoader<B::InnerBackend, TextBatch<B::InnerBackend>>>,
    total_items: usize,
}

fn create_artifact_dirs(artifact_dir: &str) {
    if let Err(e) = std::fs::create_dir_all(artifact_dir) {
        log::warn!("创建工作目录失败 {}: {}", artifact_dir, e);
    }
    if let Err(e) = std::fs::create_dir_all(format!("{}/train", artifact_dir)) {
        log::warn!("创建训练日志目录失败: {}", e);
    }
    if let Err(e) = std::fs::create_dir_all(format!("{}/valid", artifact_dir)) {
        log::warn!("创建验证日志目录失败: {}", e);
    }
}

fn save_config_and_tokenizer(artifact_dir: &str, config: &TrainingConfig, tokenizer: &Tokenizer) -> Result<(), String> {
    let config_path = format!("{}/config.json", artifact_dir);
    let config_json = serde_json::to_string_pretty(config).map_err(|e| format!("序列化配置文件失败: {}", e))?;
    std::fs::write(&config_path, config_json).map_err(|e| format!("保存配置文件失败: {} ({})", e, config_path))?;
    
    let tokenizer_path = format!("{}/tokenizer.json", artifact_dir);
    tokenizer.save(&tokenizer_path).map_err(|e| format!("保存分词器文件失败: {} ({})", e, tokenizer_path))?;
    
    Ok(())
}

fn print_training_config(batch_size: usize, seq_len: usize, accum: usize, workers: usize, total_items: usize) {
    let effective_bs = batch_size.saturating_mul(accum);
    println!(
        "最终使用配置: 物理 batch = {}, 序列长度 = {}, 梯度累积 = {} (等效 batch ≈ {})",
        batch_size, seq_len, accum, effective_bs
    );
    println!("工作线程数: {}", workers);
    println!("总训练样本数: {}", total_items);
}

fn run_training<B: AutodiffBackend>(context: TrainingContext<B>) {
    create_artifact_dirs(&context.output_dir);
    
    if let Err(e) = save_config_and_tokenizer(&context.output_dir, &context.config, &context.tokenizer) {
        eprintln!("警告: {}", e);
        eprintln!("继续运行，但配置和分词器可能未保存");
    }
    
    B::seed(&context.device, context.config.seed);

    let batch_size = context.config.batch_size;
    let seq_len = context.config.model.max_seq_len;
    
    // 更新配置为调整后的值
    let mut adjusted_context = context;
    adjusted_context.config.batch_size = batch_size;
    adjusted_context.config.model.max_seq_len = seq_len;
    
    let accum = adjusted_context.config.gradient_accumulation_steps.max(1);
    print_training_config(batch_size, seq_len, accum, adjusted_context.config.num_workers, adjusted_context.total_items);

    // 直接使用调整后的配置进行训练
    // 在Windows上避免使用可能导致文件锁定问题的应用日志记录器
    let _application_logger: Option<()> = if cfg!(windows) {
        None
    } else {
        None
    };

    // 创建模型和优化器
    let mut model = adjusted_context.init_model.unwrap_or_else(|| {
        adjusted_context.config.model.init::<B>(&adjusted_context.device)
    });
    let mut optim_config = adjusted_context.config.optimizer.clone();
    if let Some(clip_value) = adjusted_context.config.gradient_clip {
        use burn::optim::grad_clipping::GradientClippingConfig;
        optim_config = optim_config.with_grad_clipping(Some(GradientClippingConfig::Norm(clip_value as f32)));
        println!("梯度裁剪已启用 (Norm), clip_value={}", clip_value);
    }
    let mut optim: OptimizerAdaptor<burn::optim::Adam, Model<B>, B> = adjusted_context.init_optimizer.unwrap_or_else(|| {
        optim_config.init()
    });

    let lr = adjusted_context.config.lr;
    let num_epochs = adjusted_context.config.num_epochs;
    let no_progress = adjusted_context.config.no_progress;
    let output_dir = adjusted_context.output_dir.clone();
    let dataloader_train = adjusted_context.dataloader_train.clone();
    let dataloader_valid = adjusted_context.dataloader_valid.clone();

    // 初始化学习率调度器
    let mut lr_scheduler = if let Some(config) = &adjusted_context.config.lr_scheduler {
        println!("学习率调度器已启用:");
        println!("  - lr_max: {:.6}", config.lr_max);
        println!("  - lr_min: {:.6}", config.lr_min);
        println!("  - warmup_steps: {}", config.warmup_steps);
        println!("  - total_steps: {}", config.total_steps);
        Some(LRScheduler::new(config.lr_max, config.lr_min, config.warmup_steps, config.total_steps))
    } else {
        None
    };

    // 创建训练日志目录
    let checkpoint_dir = Path::new(&output_dir).join("checkpoint");
    if let Err(e) = fs::create_dir_all(&checkpoint_dir) {
        log::warn!("创建检查点目录失败: {}", e);
    }

    let start_time = std::time::Instant::now();

    // 创建验证损失日志目录
    let valid_dir = Path::new(&output_dir).join("valid");
    let train_dir = Path::new(&output_dir).join("train");
    if let Err(e) = fs::create_dir_all(&valid_dir) {
        log::warn!("创建验证日志目录失败: {}", e);
    }
    if let Err(e) = fs::create_dir_all(&train_dir) {
        log::warn!("创建训练日志目录失败: {}", e);
    }

    // 训练循环
    for epoch in 1..=num_epochs {
        println!("\n=== Epoch {}/{} ===", epoch, num_epochs);
        
        let mut accumulator = GradientsAccumulator::new();
        let epoch_valid_dir = valid_dir.join(format!("epoch-{}", epoch));
        let epoch_train_dir = train_dir.join(format!("epoch-{}", epoch));
        if let Err(e) = fs::create_dir_all(&epoch_valid_dir) {
            log::warn!("创建轮次验证日志目录失败: {}", e);
        }
        if let Err(e) = fs::create_dir_all(&epoch_train_dir) {
            log::warn!("创建轮次训练日志目录失败: {}", e);
        }

        // 训练阶段
        let mut train_loss_sum = 0.0f64;
        let mut train_batches = 0usize;
        let mut train_perplexity_sum = 0.0f64;
        
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward_step(batch);
            
            let batch_loss = {
                let loss_data = output.loss.clone().into_data();
                let loss_slice = loss_data.as_slice::<f32>();
                loss_slice.map(|s| s[0] as f64).unwrap_or(0.0)
            };
            train_loss_sum += batch_loss;
            train_perplexity_sum += batch_loss.exp();
            train_batches += 1;
            
            // 反向传播
            let grads = output.loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            accumulator.accumulate(&model, grads);

                // 梯度累积
            if (iteration + 1) % accum == 0 {
                let grads = accumulator.grads();

                // 如果启用了 LoRA，打印提示（LoRA 参数冻结在模型层处理）
                if adjusted_context.config.use_lora {
                    println!("LoRA 模式：仅训练 LoRA 适配器参数（非 LoRA 权重已在模型层冻结）");
                }

                let current_lr = lr_scheduler.as_mut().map(|s| s.get_lr()).unwrap_or(lr);
                model = optim.step(current_lr, model, grads);
                accumulator = GradientsAccumulator::new();
                if let Some(scheduler) = &mut lr_scheduler {
                    scheduler.step();
                }
            }

            if !no_progress && (iteration + 1) % 10 == 0 {
                let avg_loss = train_loss_sum / train_batches as f64;
                let avg_perplexity = train_perplexity_sum / train_batches as f64;
                println!("[Train] Epoch {} - Batch {} - Loss: {:.6} - Perplexity: {:.4}", epoch, iteration + 1, avg_loss, avg_perplexity);
            }
        }

        // 计算并记录训练损失
        let avg_train_loss = if train_batches > 0 {
            train_loss_sum / train_batches as f64
        } else {
            0.0
        };
        let avg_train_perplexity = if train_batches > 0 {
            train_perplexity_sum / train_batches as f64
        } else {
            0.0
        };
        let _ = fs::write(epoch_train_dir.join("Loss.log"), format!("{}\n", avg_train_loss));
        let _ = fs::write(epoch_train_dir.join("Perplexity.log"), format!("{}\n", avg_train_perplexity));

        println!("[Train] Epoch {} complete - Average Loss: {:.6} - Perplexity: {:.4}", epoch, avg_train_loss, avg_train_perplexity);

        // 验证阶段
        let model_valid = model.valid();
        let mut valid_loss_sum = 0.0f64;
        let mut valid_batches = 0usize;
        let mut valid_perplexity_sum = 0.0f64;

        for (iteration, batch) in dataloader_valid.iter().enumerate() {
            let batch_loss = model_valid.compute_validation_loss(batch);
            valid_loss_sum += batch_loss;
            valid_perplexity_sum += batch_loss.exp();
            valid_batches += 1;

            if !no_progress && (iteration + 1) % 10 == 0 {
                let avg_loss = valid_loss_sum / valid_batches as f64;
                let avg_perplexity = valid_perplexity_sum / valid_batches as f64;
                println!("[Valid] Epoch {} - Batch {} - Loss: {:.6} - Perplexity: {:.4}", epoch, iteration + 1, avg_loss, avg_perplexity);
            }
        }

        // 计算并记录验证损失
        let avg_valid_loss = if valid_batches > 0 {
            valid_loss_sum / valid_batches as f64
        } else {
            0.0
        };
        let avg_valid_perplexity = if valid_batches > 0 {
            valid_perplexity_sum / valid_batches as f64
        } else {
            0.0
        };
        let _ = fs::write(epoch_valid_dir.join("Loss.log"), format!("{}\n", avg_valid_loss));
        let _ = fs::write(epoch_valid_dir.join("Perplexity.log"), format!("{}\n", avg_valid_perplexity));

        println!("[Valid] Epoch {} complete - Average Loss: {:.6} - Perplexity: {:.4}", epoch, avg_valid_loss, avg_valid_perplexity);

        // 保存检查点
        let checkpoint_path = checkpoint_dir.join(format!("model-{}.mpk", epoch));
        if let Err(e) = model.clone().save_file(&checkpoint_path, &CompactRecorder::new()) {
            eprintln!("警告: 保存检查点失败: {}", e);
        } else {
            println!("检查点已保存: {}", checkpoint_path.display());
        }
    }

    let elapsed = start_time.elapsed();

    // 使用 find_best_epoch 函数找到最佳 epoch
    if let Some(best_epoch) = find_best_epoch(Path::new(&output_dir)) {
        let from = Path::new(&output_dir)
            .join("checkpoint")
            .join(format!("model-{}.mpk", best_epoch));
        let to = Path::new(&output_dir).join("best_model.mpk");
        
        if let Err(e) = fs::copy(&from, &to) {
            eprintln!("警告: 复制最佳模型失败: {} ({} -> {})", e, from.display(), to.display());
        } else {
            println!("最佳模型已复制到: {}", to.display());
        }
    }

    // 保存最终模型
    let final_model_path = Path::new(&output_dir).join("model.mpk");
    if let Err(e) = model.save_file(&final_model_path, &CompactRecorder::new()) {
        eprintln!("警告: 保存最终模型失败: {}", e);
    } else {
        println!("最终模型已保存到: {}", final_model_path.display());
    }

    print_training_stats(elapsed, adjusted_context.total_items, num_epochs, &output_dir);
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
        println!("训练集 Perplexity: {:.4}", perplexity);
    }
    
    // 计算验证集的 perplexity
    if let Some(last_valid_loss) = find_last_valid_loss(Path::new(artifact_dir)) {
        let perplexity = last_valid_loss.exp();
        println!("最后一轮验证损失: {:.4}", last_valid_loss);
        println!("验证集 Perplexity: {:.4}", perplexity);
    }
}

pub fn train<B: AutodiffBackend>(
    output_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: &Tokenizer,
    tokens: Vec<usize>,
    mask: Vec<u8>,
    init_model: Option<Model<B>>,
    init_optimizer: Option<OptimizerAdaptor<burn::optim::Adam, Model<B>, B>>,
) {
    if config.distributed {
        // 获取可用设备
        let available_devices = get_available_devices();
        let devices_to_use = if config.devices.is_empty() {
            available_devices
        } else {
            config.devices
        };
        
        println!("分布式训练模式: 使用 {} 个设备", devices_to_use.len());
        for device in &devices_to_use {
            println!("  - {}", device);
        }
        
        // 创建分布式配置
        let dist_config = DistributedConfig::new(devices_to_use);
        
        // 创建设备实例
        let devices: Vec<B::Device> = dist_config.devices.iter()
            .map(|name| create_device::<B>(name))
            .collect();
        
        // 创建数据集
        // 创建数据集
        let dataset = TextDataset::new(tokens, mask, config.max_seq_len);
        let dataset = Arc::new(dataset);
        
        // 启动并行训练
        train_parallel::<B, _>(dataset, config.batch_size, config.num_epochs, devices);
        return;
    }
    
    // 单设备训练逻辑保持不变
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

    let context = TrainingContext {
        output_dir: output_dir.to_string(),
        config,
        device,
        tokenizer: Arc::new(tokenizer.clone()),
        init_model,
        init_optimizer,
        dataloader_train,
        dataloader_valid,
        total_items: n_tokens,
    };
    run_training(context);
}

pub fn train_from_cache<B: AutodiffBackend>(
    output_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: &Tokenizer,
    tokens_path: &str,
    mask_path: &str,
    init_model: Option<Model<B>>,
    init_optimizer: Option<OptimizerAdaptor<burn::optim::Adam, Model<B>, B>>,
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

    let context = TrainingContext {
        output_dir: output_dir.to_string(),
        config,
        device,
        tokenizer: Arc::new(tokenizer.clone()),
        init_model,
        init_optimizer,
        dataloader_train,
        dataloader_valid,
        total_items: n_tokens,
    };
    run_training(context);
}

/// DPO训练函数
pub fn train_dpo<B: AutodiffBackend>(
    output_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: &Tokenizer,
    dpo_items: Vec<DPOItem>,
    init_model: Option<Model<B>>,
) {
    create_artifact_dirs(output_dir);
    
    if let Err(e) = save_config_and_tokenizer(output_dir, &config, tokenizer) {
        eprintln!("警告: {}", e);
        eprintln!("继续运行，但配置和分词器可能未保存");
    }
    
    B::seed(&device, config.seed);
    
    let default_dpo_config = DPOConfig::default();
    let dpo_config = config.dpo_config.as_ref().unwrap_or(&default_dpo_config);
    println!("DPO训练配置:");
    println!("  Beta值: {}", dpo_config.beta);
    println!("  KL正则化: {}", dpo_config.use_kl_regularization);
    println!("  KL权重: {}", dpo_config.kl_weight);
    
    // 创建模型
    let model = init_model.unwrap_or_else(|| config.model.init::<B>(&device));
    
    // 创建优化器
    let mut optimizer_config = config.optimizer;
    if let Some(clip_value) = config.gradient_clip {
        use burn::optim::grad_clipping::GradientClippingConfig;
        optimizer_config = optimizer_config.with_grad_clipping(Some(GradientClippingConfig::Norm(clip_value as f32)));
        println!("DPO梯度裁剪已启用 (Norm), clip_value={}", clip_value);
    }
    let optimizer = optimizer_config.init();
    
    // 创建DPO训练器
    let mut trainer = DPOTrainer::new(
        model,
        optimizer,
        dpo_config.clone(),
        device.clone(),
    );
    
    // 创建批次处理器（使用GPU显存探测的序列长度）
    let batcher = DPOBatcher::new(
        device.clone(), 
        config.model.max_seq_len, 
        config.model.max_seq_len
    );
    
    println!("数据项数量: {}", dpo_items.len());
    
    // 使用GPU显存探测的批次大小
    let batch_size = config.batch_size.min(dpo_items.len());
    println!("使用批次大小: {}", batch_size);
    println!("使用序列长度: {}", config.model.max_seq_len);
    println!("学习率: {}", config.lr);
    
    println!("开始DPO训练...");
    
    // 创建检查点目录
    let checkpoint_dir = Path::new(output_dir).join("checkpoint");
    if let Err(e) = fs::create_dir_all(&checkpoint_dir) {
        log::warn!("创建DPO检查点目录失败: {}", e);
    }
    
    // 训练循环
    for epoch in 1..=config.num_epochs {
        println!("\nEpoch {}/{}", epoch, config.num_epochs);
        
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        for chunk in dpo_items.chunks(batch_size) {
            let batch_items = chunk.to_vec();
            
            // 创建批次
            let batch = batcher.batch(batch_items, &device);
            
            // 训练批次（传递学习率）
            let loss_value = trainer.train_batch(batch, config.lr);
            
            total_loss += loss_value;
            batch_count += 1;
            
            if !config.no_progress {
                println!("  Batch {} - Loss: {:.6}", batch_count, loss_value);
            }
        }
        
        let avg_loss = total_loss / batch_count as f32;
        let avg_perplexity = avg_loss.exp();
        println!("  Epoch Loss: {:.6} - Perplexity: {:.4}", avg_loss, avg_perplexity);
        
        // 保存检查点
        let model_path = checkpoint_dir.join(format!("model-{}.mpk", epoch));
        if let Err(e) = trainer.model().clone().save_file(&model_path, &CompactRecorder::new()) {
            println!("保存模型失败: {}", e);
        } else {
            println!("检查点已保存: {}", model_path.display());
        }
    }
    
    // 保存最终模型
    let model_path = Path::new(output_dir).join("model.mpk");
    if let Err(e) = trainer.model().clone().save_file(&model_path, &CompactRecorder::new()) {
        println!("保存最终模型失败: {}", e);
    } else {
        println!("DPO训练完成，模型已保存到: {}", model_path.display());
    }
}

pub fn train_with_loaders<B: AutodiffBackend>(
    output_dir: &str,
    config: TrainingConfig,
    device: B::Device,
    tokenizer: &Tokenizer,
    dataloader_train: Arc<dyn burn::data::dataloader::DataLoader<B, TextBatch<B>>>,
    dataloader_valid: Arc<
        dyn burn::data::dataloader::DataLoader<B::InnerBackend, TextBatch<B::InnerBackend>>,
    >,
    init_model: Option<Model<B>>,
    init_optimizer: Option<OptimizerAdaptor<burn::optim::Adam, Model<B>, B>>,
) {
    let total_items = dataloader_train.num_items();
    let context = TrainingContext {
        output_dir: output_dir.to_string(),
        config,
        device,
        tokenizer: Arc::new(tokenizer.clone()),
        init_model,
        init_optimizer,
        dataloader_train,
        dataloader_valid,
        total_items,
    };
    run_training(context);
}

fn parse_epoch_from_name(name: &str) -> Option<usize> {
    name.strip_prefix("epoch-")?.parse::<usize>().ok()
}

fn read_last_loss(path: &Path) -> Option<f64> {
    fs::read_to_string(path).ok().and_then(|text| {
        text.lines().rev().find(|l| !l.trim().is_empty()).and_then(|last| {
            last.split(',').next().and_then(|v| v.trim().parse::<f64>().ok())
        })
    })
}

fn find_best_epoch(artifact_dir: &Path) -> Option<usize> {
    let valid_dir = artifact_dir.join("valid");
    fs::read_dir(&valid_dir).ok().and_then(|entries| {
        entries.filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_string_lossy();
            let epoch = parse_epoch_from_name(&name)?;
            let loss = read_last_loss(&path.join("Loss.log"))?;
            Some((epoch, loss))
        }).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(epoch, _)| epoch)
    })
}

fn find_last_valid_loss(artifact_dir: &Path) -> Option<f64> {
    let valid_dir = artifact_dir.join("valid");
    fs::read_dir(&valid_dir).ok().and_then(|entries| {
        entries.filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_string_lossy();
            let epoch = parse_epoch_from_name(&name)?;
            let loss = read_last_loss(&path.join("Loss.log"))?;
            Some((epoch, loss))
        }).max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, loss)| loss)
    })
}

fn find_last_epoch_loss(artifact_dir: &Path) -> Option<f64> {
    let train_dir = artifact_dir.join("train");
    fs::read_dir(&train_dir).ok().and_then(|entries| {
        entries.filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_string_lossy();
            parse_epoch_from_name(&name).map(|epoch| (epoch, path))
        }).max_by(|a, b| a.0.cmp(&b.0))
        .and_then(|(_, path)| read_last_loss(&path.join("Loss.log")))
    })
}
