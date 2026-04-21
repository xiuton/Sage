#![recursion_limit = "1024"]

use burn::backend::{Autodiff, ndarray::NdArray, wgpu::Wgpu};
use burn::module::Module;
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::tensor::{Tensor, TensorData};
use std::time::Instant;
use sage::{
    core::{ModelConfig, Tokenizer},
    probe_first_fitting_config,
    streaming::{SftInput, StreamingSftDataLoader},
    train, train_from_cache, train_with_loaders, train_dpo,
    TrainingConfig,
};
use sage::training::{DPOConfig, load_dpo_jsonl};
use sage::core::image_generation::{DiffusionModel, DiffusionConfig};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs,
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::{ArgAction, Parser};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "./data/corpus.txt")]
    pub corpus: String,

    #[arg(long)]
    pub corpus_dir: Option<String>,

    #[arg(long, default_value_t = 50_000_000)]
    pub max_bytes: usize,

    #[arg(long, default_value_t = false)]
    pub stream: bool,

    #[arg(long, default_value_t = false)]
    pub stream_direct: bool,

    #[arg(long)]
    pub sft_jsonl: Option<String>,

    #[arg(long, default_value_t = false)]
    pub sft_sample: bool,

    #[arg(long, default_value_t = false)]
    pub sft_sample_messages: bool,

    #[arg(long, default_value_t = 0)]
    pub sft_max_records: usize,

    /// 模型和训练产物的输出目录。
    #[arg(long, default_value = "./models")]
    pub output_dir: String,

    /// 训练的总轮数。
    #[arg(long, default_value_t = 50)]
    pub num_epochs: usize,

    /// 每步训练的物理 batch。可写多次 `--batch-size`，**以最后一次为准**（默认 32）。
    #[arg(long = "batch-size", action = ArgAction::Append)]
    pub batch_sizes: Vec<usize>,

    #[arg(long, default_value_t = 1.0e-4)]
    pub learning_rate: f64,

    #[arg(long, default_value_t = 256)]
    pub max_seq_len: usize,

    #[arg(long, default_value_t = false)]
    pub force: bool,

    #[arg(long, default_value_t = false)]
    pub r#continue: bool,

    #[arg(long)]
    pub resume_epoch: Option<usize>,

    #[arg(long, default_value_t = false)]
    pub reset_tokenizer: bool,

    #[arg(long, default_value_t = false)]
    pub use_bpe: bool,

    #[arg(long, default_value_t = 5000)]
    pub bpe_vocab_size: usize,

    /// Enable high parallel training mode (faster data loading + batch throughput)
    #[arg(long, default_value_t = false)]
    pub fast: bool,

    /// Worker threads for data loading
    #[arg(long, default_value_t = 4)]
    pub num_workers: usize,

    /// Enable quick development mode (1 epoch, small batch, high lr)
    #[arg(long, default_value_t = false)]
    pub quick_dev: bool,

    /// Enable ultra-quick development mode (1 epoch, tiny batch, limit data to 100 records)
    #[arg(long, default_value_t = false)]
    pub ultra_quick: bool,

    /// Disable progress bars and TUI display
    #[arg(long, default_value_t = false)]
    pub no_progress: bool,

    /// Enable TUI progress display (may not work in all terminals, especially Windows PowerShell)
    #[arg(long, default_value_t = false)]
    pub tui: bool,

    /// Backend to use for training: cpu or gpu
    #[arg(long, default_value = "cpu", value_name = "cpu|gpu")]
    pub backend: String,

    /// Model configuration file path
    #[arg(long, default_value = "./inference/configs/config_1B.json")]
    pub config_path: String,

    /// Training mode: general, code, math
    #[arg(long, default_value = "general", value_name = "general|code|math")]
    pub training_mode: String,

    /// Force enable TUI progress display even in environments that might not support it
    #[arg(long, default_value_t = false)]
    pub force_tui: bool,

    /// 禁用 GPU 自动显存探测；按 `--batch-size` 与 `--max-seq-len` 原样训练（可配合 `--gradient-accumulation`）。
    #[arg(long, default_value_t = false)]
    pub no_auto_vram: bool,

    /// 梯度累积步数（Hugging Face 风格：等效 batch ≈ batch_size × 该值）。
    /// 使用 `--no-auto-vram` 或 CPU 时生效；GPU 自动探测成功时会按等效 batch 自动计算并写入配置。
    #[arg(long, default_value_t = 1)]
    pub gradient_accumulation: usize,
    
    /// 启用分布式训练
    #[arg(long, default_value_t = false)]
    pub distributed: bool,
    
    /// 指定使用的设备列表，格式: "cpu,gpu:0,gpu:1"
    #[arg(long, value_name = "cpu,gpu:0,gpu:1")]
    pub devices: Option<String>,
    
    /// DPO训练模式
    #[arg(long, default_value_t = false)]
    pub dpo: bool,
    
    /// DPO beta参数
    #[arg(long, default_value_t = 0.1)]
    pub dpo_beta: f64,
    
    /// DPO KL散度权重
    #[arg(long, default_value_t = 0.1)]
    pub dpo_kl_weight: f64,
    
    /// DPO数据文件路径
    #[arg(long, value_name = "path/to/dpo_data.jsonl")]
    pub dpo_data: Option<String>,
    
    /// 启用学习率调度器（Cosine Annealing + Warmup）
    #[arg(long, default_value_t = false)]
    pub lr_scheduler: bool,
    
    /// 学习率调度器的最大学习率（Warmup阶段结束时的值）
    #[arg(long, default_value_t = 0.0001)]
    pub lr_max: f64,
    
    /// 学习率调度器的最小学习率（Cosine阶段结束时的值）
    #[arg(long, default_value_t = 0.00001)]
    pub lr_min: f64,
    
    /// 学习率调度器的Warmup步数
    #[arg(long, default_value_t = 1000)]
    pub warmup_steps: usize,
    
    /// 学习率调度器的总调度步数
    #[arg(long, default_value_t = 100000)]
    pub total_steps: usize,
    
    /// 启用多模态训练
    #[arg(long, default_value_t = false)]
    pub multimodal: bool,
    
    /// 视觉编码器输出维度
    #[arg(long, default_value_t = 512)]
    pub vision_out_dim: usize,
    
    /// 融合策略：add, concatenate, attention
    #[arg(long, default_value = "add", value_name = "add|concatenate|attention")]
    pub fusion_strategy: String,

    /// 启用 LoRA 微调
    #[arg(long, default_value_t = false)]
    pub use_lora: bool,

    /// 启用文生图训练
    #[arg(long, default_value_t = false)]
    pub text_to_image: bool,

    /// 文本-图像对数据文件路径
    #[arg(long, value_name = "path/to/text_image_pairs.jsonl")]
    pub image_text_data: Option<String>,

    /// LoRA 秩 (Rank)
    #[arg(long, default_value_t = 8)]
    pub lora_rank: usize,

    /// LoRA Alpha 参数
    #[arg(long, default_value_t = 16.0)]
    pub lora_alpha: f32,
}

impl Args {
    /// 解析后的 batch 大小：`--batch-size` 最后一次出现，未指定时为 32。
    fn batch_size(&self) -> usize {
        self.batch_sizes.last().copied().unwrap_or(32)
    }
}

#[derive(Deserialize, Debug)]
struct SftRecord {
    prompt: String,
    response: String,
}

#[derive(Deserialize)]
struct SftMessage {
    role: String,
    content: String,
}

fn collect_txt_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_txt_files(&path, out)?;
            continue;
        }
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
        {
            out.push(path);
        }
    }
    Ok(())
}

fn read_file_limited(path: &Path, remaining: Option<usize>) -> io::Result<Vec<u8>> {
    let mut file = fs::File::open(path)?;
    let mut buf = Vec::new();
    match remaining {
        Some(limit) => {
            let mut take = file.take(limit as u64);
            take.read_to_end(&mut buf)?;
        }
        None => {
            file.read_to_end(&mut buf)?;
        }
    }
    Ok(buf)
}

fn load_corpus(args: &Args) -> io::Result<String> {
    let max_bytes = if args.max_bytes == 0 {
        None
    } else {
        Some(args.max_bytes)
    };

    if let Some(dir) = &args.corpus_dir {
        let mut files = Vec::new();
        collect_txt_files(Path::new(dir), &mut files)?;
        files.sort();

        let mut out = String::new();
        let mut used: usize = 0;

        for path in files {
            let remaining = max_bytes.map(|m| m.saturating_sub(used));
            if remaining == Some(0) {
                break;
            }
            let bytes = read_file_limited(&path, remaining)?;
            let text = String::from_utf8_lossy(&bytes);
            used = used.saturating_add(bytes.len());
            out.push_str(&text);
            out.push('\n');
        }

        return Ok(out);
    }

    let path: &str = if args.corpus.is_empty() { "corpus_cn.txt" } else { &args.corpus };
    let bytes = read_file_limited(Path::new(path), max_bytes)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

fn create_template(system: Option<&str>, prompt: &str, response: &str) -> String {
    let mut out = String::with_capacity(512); // 预分配合理容量
    out.push('\u{0002}');
    out.push_str("<s>");
    
    if let Some(system_msg) = system {
        out.push('\n');
        out.push_str("<system>");
        out.push_str(system_msg);
        out.push_str("</system>");
    }
    
    out.push('\n');
    out.push_str("<user>");
    out.push_str(prompt);
    out.push_str("</user>");
    out.push('\n');
    out.push_str("<assistant>");
    out.push_str(response);
    out.push_str("</assistant>");
    out.push('\u{0003}');
    out.push('\n');
    out
}

fn sft_template(prompt: &str, response: &str) -> String {
    create_template(None, prompt, response)
}

/// 代码生成训练模板 - 优化代码生成场景
fn code_template(prompt: &str, response: &str) -> String {
    create_template(Some("你是一个专业的代码助手，擅长编写高质量、可读性强的代码。"), prompt, response)
}

/// 数学推理训练模板 - 优化数学问题解决场景
fn math_template(prompt: &str, response: &str) -> String {
    create_template(Some("你是一个数学专家，擅长解决各类数学问题并提供详细的解题步骤。"), prompt, response)
}

fn sft_messages_template(messages: &[SftMessage]) -> Option<String> {
    let mut out = String::with_capacity(1024); // 预分配合理容量
    out.push('\u{0002}');
    out.push_str("<s>");

    let mut has_assistant = false;
    for m in messages {
        match m.role.as_str() {
            "system" => {
                out.push('\n');
                out.push_str("<system>");
                out.push_str(&m.content);
                out.push_str("</system>");
            }
            "user" => {
                out.push('\n');
                out.push_str("<user>");
                out.push_str(&m.content);
                out.push_str("</user>");
            }
            "assistant" => {
                has_assistant = true;
                out.push('\n');
                out.push_str("<assistant>");
                out.push_str(&m.content);
                out.push_str("</assistant>");
                out.push('\u{0003}');
            }
            _ => {}
        }
    }

    out.push('\n');

    if has_assistant { Some(out) } else { None }
}

fn load_sft_jsonl(args: &Args, path: &str) -> io::Result<String> {
    let bytes = read_file_limited(
        Path::new(path),
        if args.max_bytes == 0 {
            None
        } else {
            Some(args.max_bytes)
        },
    )?;
    let text = String::from_utf8_lossy(&bytes);

    let mut out = String::new();
    let mut used_records = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(rec) = serde_json::from_str::<SftRecord>(line) {
            // 根据训练模式选择模板
            let template = match args.training_mode.as_str() {
                "code" => code_template(&rec.prompt, &rec.response),
                "math" => math_template(&rec.prompt, &rec.response),
                _ => sft_template(&rec.prompt, &rec.response),
            };
            out.push_str(&template);
        } else if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(messages) = v.get("messages") {
                if let Ok(messages) = serde_json::from_value::<Vec<SftMessage>>(messages.clone()) {
                    if let Some(sample) = sft_messages_template(&messages) {
                        out.push_str(&sample);
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else {
            continue;
        }
        used_records += 1;
        if args.sft_max_records != 0 && used_records >= args.sft_max_records {
            break;
        }
    }
    Ok(out)
}

fn load_sft_sample(args: &Args) -> String {
    let samples = match args.training_mode.as_str() {
        "code" => [
            SftRecord {
                prompt: "请写一个Python函数，计算斐波那契数列的第n项。".to_string(),
                response: "def fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(2, n + 1):\n            a, b = b, a + b\n        return b\n\n# 使用示例\nprint(fibonacci(10))  # 输出: 55".to_string(),
            },
            SftRecord {
                prompt: "如何用Python实现快速排序算法？".to_string(),
                response: "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n\n# 使用示例\nprint(quick_sort([3, 6, 8, 10, 1, 2, 1]))".to_string(),
            },
        ],
        "math" => [
            SftRecord {
                prompt: "求解方程 2x + 5 = 15".to_string(),
                response: "解：2x + 5 = 15\n步骤1：两边同时减去5\n2x = 15 - 5\n2x = 10\n步骤2：两边同时除以2\nx = 10 / 2\nx = 5\n答案：x = 5".to_string(),
            },
            SftRecord {
                prompt: "计算三角形面积，底为6，高为4".to_string(),
                response: "三角形面积公式：面积 = (底 × 高) / 2\n代入数值：面积 = (6 × 4) / 2 = 24 / 2 = 12\n答案：12平方单位".to_string(),
            },
        ],
        _ => [
            SftRecord {
                prompt: "你是谁？".to_string(),
                response: "我是一个用 Rust 训练出来的小模型。".to_string(),
            },
            SftRecord {
                prompt: "用一句话解释千字文是什么。".to_string(),
                response: "《千字文》是由一千个不重复汉字组成的启蒙文章。".to_string(),
            },
        ],
    };

    let mut out = String::new();
    for rec in samples {
        let template = match args.training_mode.as_str() {
            "code" => code_template(&rec.prompt, &rec.response),
            "math" => math_template(&rec.prompt, &rec.response),
            _ => sft_template(&rec.prompt, &rec.response),
        };
        out.push_str(&template);
    }
    out
}

fn load_sft_messages_sample() -> String {
    let samples = [
        vec![
            SftMessage {
                role: "user".to_string(),
                content: "你是谁？".to_string(),
            },
            SftMessage {
                role: "assistant".to_string(),
                content: "我是一个用 Rust 训练出来的小模型。".to_string(),
            },
        ],
        vec![
            SftMessage {
                role: "user".to_string(),
                content: "用一句话解释千字文是什么。".to_string(),
            },
            SftMessage {
                role: "assistant".to_string(),
                content: "《千字文》是由一千个不重复汉字组成的启蒙文章。".to_string(),
            },
        ],
        vec![
            SftMessage {
                role: "user".to_string(),
                content: "给我一个学习 Rust 的建议。".to_string(),
            },
            SftMessage {
                role: "assistant".to_string(),
                content: "从所有权和借用入手，多写小项目并配合 clippy 修正。".to_string(),
            },
        ],
    ];

    let mut out = String::new();
    for messages in samples {
        if let Some(sample) = sft_messages_template(&messages) {
            out.push_str(&sample);
        }
    }
    out
}

fn sft_sample_from_json_line(line: &str, training_mode: &str) -> Option<String> {
    if let Ok(rec) = serde_json::from_str::<SftRecord>(line) {
        let template = match training_mode {
            "code" => code_template(&rec.prompt, &rec.response),
            "math" => math_template(&rec.prompt, &rec.response),
            _ => sft_template(&rec.prompt, &rec.response),
        };
        return Some(template);
    }

    let v = serde_json::from_str::<serde_json::Value>(line).ok()?;
    let messages = v.get("messages")?;
    let messages = serde_json::from_value::<Vec<SftMessage>>(messages.clone()).ok()?;
    sft_messages_template(&messages)
}

fn collect_vocab_chars_stream(args: &Args) -> io::Result<Vec<char>> {
    let mut set = BTreeSet::new();

    if let Some(path) = &args.sft_jsonl {
        let file = fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut used_bytes: usize = 0;
        let mut used_records: usize = 0;

        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            used_bytes = used_bytes.saturating_add(n);
            if args.max_bytes != 0 && used_bytes > args.max_bytes {
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let sample = match sft_sample_from_json_line(trimmed, &args.training_mode) {
                Some(v) => v,
                None => continue,
            };

            for ch in sample.chars() {
                set.insert(ch);
            }

            used_records += 1;
            if args.sft_max_records != 0 && used_records >= args.sft_max_records {
                break;
            }
        }

        return Ok(set.into_iter().collect());
    }

    if args.sft_sample_messages {
        let text = load_sft_messages_sample();
        for ch in text.chars() {
            set.insert(ch);
        }
        return Ok(set.into_iter().collect());
    }

    if args.sft_sample {
        let text = load_sft_sample(args);
        for ch in text.chars() {
            set.insert(ch);
        }
        return Ok(set.into_iter().collect());
    }

    let max_bytes = if args.max_bytes == 0 {
        None
    } else {
        Some(args.max_bytes)
    };

    if let Some(dir) = &args.corpus_dir {
        let mut files = Vec::new();
        collect_txt_files(Path::new(dir), &mut files)?;
        files.sort();

        let mut used: usize = 0;
        for path in files {
            let file = fs::File::open(&path)?;
            let mut reader = BufReader::new(file);
            let mut line = String::new();
            loop {
                line.clear();
                let n = reader.read_line(&mut line)?;
                if n == 0 {
                    break;
                }
                used = used.saturating_add(n);
                if max_bytes.is_some_and(|m| used > m) {
                    return Ok(set.into_iter().collect());
                }
                for ch in line.chars() {
                    set.insert(ch);
                }
            }
            set.insert('\n');
        }
        return Ok(set.into_iter().collect());
    }

    let path: &str = if args.corpus.is_empty() { "corpus_cn.txt" } else { &args.corpus };
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut used: usize = 0;
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        used = used.saturating_add(n);
        if max_bytes.is_some_and(|m| used > m) {
            break;
        }
        for ch in line.chars() {
            set.insert(ch);
        }
    }

    Ok(set.into_iter().collect())
}

fn write_u32_le(mut w: impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn build_token_cache_stream(args: &Args, tokenizer: &Tokenizer) -> io::Result<(String, String)> {
    let cache_dir = Path::new(&args.output_dir).join("cache");
    fs::create_dir_all(&cache_dir)?;

    let tokens_path = cache_dir.join("tokens.bin");
    let mask_path = cache_dir.join("mask.bin");

    let mut tokens_file = io::BufWriter::new(fs::File::create(&tokens_path)?);
    let mut mask_file = io::BufWriter::new(fs::File::create(&mask_path)?);

    if let Some(path) = &args.sft_jsonl {
        let file = fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut used_bytes: usize = 0;
        let mut used_records: usize = 0;

        loop {
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            used_bytes = used_bytes.saturating_add(n);
            if args.max_bytes != 0 && used_bytes > args.max_bytes {
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let sample = match sft_sample_from_json_line(trimmed, &args.training_mode) {
                Some(v) => v,
                None => continue,
            };
            let (ids, mask) = tokenizer.encode_with_assistant_mask(&sample);
            for (&id, &m) in ids.iter().zip(mask.iter()) { // id: usize, m: u8
                write_u32_le(&mut tokens_file, id as u32)?;
                mask_file.write_all(&[m])?;
            }

            used_records += 1;
            if args.sft_max_records != 0 && used_records >= args.sft_max_records {
                break;
            }
        }

        tokens_file.flush()?;
        mask_file.flush()?;
        return Ok((
            tokens_path.to_string_lossy().to_string(),
            mask_path.to_string_lossy().to_string(),
        ));
    }

    if args.sft_sample_messages {
        let text = load_sft_messages_sample();
        let (ids, mask) = tokenizer.encode_with_assistant_mask(&text);
        for (&id, &m) in ids.iter().zip(mask.iter()) {
            write_u32_le(&mut tokens_file, id as u32)?;
            mask_file.write_all(&[m])?;
        }
        tokens_file.flush()?;
        mask_file.flush()?;
        return Ok((
            tokens_path.to_string_lossy().to_string(),
            mask_path.to_string_lossy().to_string(),
        ));
    }

    if args.sft_sample {
        let text = load_sft_sample(args);
        let (ids, mask) = tokenizer.encode_with_assistant_mask(&text);
        for (&id, &m) in ids.iter().zip(mask.iter()) {
            write_u32_le(&mut tokens_file, id as u32)?;
            mask_file.write_all(&[m])?;
        }
        tokens_file.flush()?;
        mask_file.flush()?;
        return Ok((
            tokens_path.to_string_lossy().to_string(),
            mask_path.to_string_lossy().to_string(),
        ));
    }

    let max_bytes = if args.max_bytes == 0 {
        None
    } else {
        Some(args.max_bytes)
    };

    if let Some(dir) = &args.corpus_dir {
        let mut files = Vec::new();
        collect_txt_files(Path::new(dir), &mut files)?;
        files.sort();

        let mut used: usize = 0;
        for path in files {
            let file = fs::File::open(&path)?;
            let mut reader = BufReader::new(file);
            let mut line = String::new();
            loop {
                line.clear();
                let n = reader.read_line(&mut line)?;
                if n == 0 {
                    break;
                }
                used = used.saturating_add(n);
                if max_bytes.is_some_and(|m| used > m) {
                    tokens_file.flush()?;
                    mask_file.flush()?;
                    return Ok((
                        tokens_path.to_string_lossy().to_string(),
                        mask_path.to_string_lossy().to_string(),
                    ));
                }
                let ids = tokenizer.encode(&line);
                for id in ids {
                    write_u32_le(&mut tokens_file, id as u32)?;
                    mask_file.write_all(&[1u8])?;
                }
            }

            let newline = tokenizer.encode("\n");
            for id in newline {
                write_u32_le(&mut tokens_file, id as u32)?;
                mask_file.write_all(&[1u8])?;
            }
        }

        tokens_file.flush()?;
        mask_file.flush()?;
        return Ok((
            tokens_path.to_string_lossy().to_string(),
            mask_path.to_string_lossy().to_string(),
        ));
    }

    let path: &str = if args.corpus.is_empty() { "corpus_cn.txt" } else { &args.corpus };
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut used: usize = 0;
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        used = used.saturating_add(n);
        if max_bytes.is_some_and(|m| used > m) {
            break;
        }
        let ids = tokenizer.encode(&line);
        for id in ids {
            write_u32_le(&mut tokens_file, id as u32)?;
            mask_file.write_all(&[1u8])?;
        }
    }

    tokens_file.flush()?;
    mask_file.flush()?;
    Ok((
        tokens_path.to_string_lossy().to_string(),
        mask_path.to_string_lossy().to_string(),
    ))
}

fn count_sft_records_stream(args: &Args) -> io::Result<usize> {
    let Some(path) = &args.sft_jsonl else {
        return Ok(0);
    };

    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut used_bytes: usize = 0;
    let mut used_records: usize = 0;

    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }

        used_bytes = used_bytes.saturating_add(n);
        if args.max_bytes != 0 && used_bytes > args.max_bytes {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if sft_sample_from_json_line(trimmed, &args.training_mode).is_none() {
            continue;
        }

        used_records += 1;
        if args.sft_max_records != 0 && used_records >= args.sft_max_records {
            break;
        }
    }

    Ok(used_records)
}

fn main() {
    // 在程序入口处立即设置环境变量，确保在任何burn代码执行前生效
    unsafe {
        // 禁用实验日志记录器（在burn 0.14.0中，这个环境变量需要在任何burn代码执行前设置）
        std::env::set_var("BURN_EXPERIMENT_LOGGER_DISABLED", "1");

        // 设置环境变量改善Burn TUI显示
        std::env::set_var("BURN_TUI_NO_CLEAR", "1");
        std::env::set_var("TERM", "xterm-256color");
        std::env::set_var("BURN_TUI_ENABLED", "1");
        std::env::set_var("BURN_TUI_FORCE", "1");
        std::env::set_var("WGPU_LOG", "off");

        // 设置日志级别
        std::env::set_var("RUST_LOG", "burn_train=info,wgpu_core=off,burn_core=off");

        // 设置 cubecl autotune 级别为 minimal，加速第一次启动
        std::env::set_var("CUBECL_AUTOTUNE_LEVEL", "minimal");
    }

    let mut args = Args::parse();

    // 文生图训练模式
    if args.text_to_image {
        println!("启用文生图训练模式...");
        let image_text_data = args.image_text_data.as_ref().expect("--image-text-data is required for text-to-image training");
        let config_path = &args.config_path;
        let output_dir = &args.output_dir;

        // 加载配置文件
        println!("加载配置文件: {}", config_path);
        let config_json = fs::read_to_string(config_path).expect("Failed to read config file");
        let config: serde_json::Value = serde_json::from_str(&config_json).expect("Failed to parse config JSON");

        let image_size = config["image_size"].as_i64().unwrap_or(64) as usize;
        let latent_dim = config["latent_dim"].as_i64().unwrap_or(128) as usize;
        let hidden_channels = config["hidden_channels"].as_i64().unwrap_or(128) as usize;
        let num_timesteps = config["num_timesteps"].as_i64().unwrap_or(1000) as usize;
        let beta_start = config["beta_start"].as_f64().unwrap_or(0.0001) as f32;
        let beta_end = config["beta_end"].as_f64().unwrap_or(0.02) as f32;
        let batch_size = args.batch_sizes.first().copied().unwrap_or(16);
        let num_epochs = args.num_epochs;
        let learning_rate = args.learning_rate as f32;

        println!("配置参数: image_size={}, latent_dim={}, hidden_channels={}, batch_size={}, epochs={}",
            image_size, latent_dim, hidden_channels, batch_size, num_epochs);

        // 加载训练数据
        println!("加载训练数据: {}", image_text_data);
        let data_json = fs::read_to_string(image_text_data).expect("Failed to read data file");
        let lines: Vec<&str> = data_json.lines().filter(|l| !l.trim().is_empty()).collect();
        println!("共加载 {} 条训练数据", lines.len());

            // 根据命令行参数选择后端
            if args.backend == "gpu" {
                println!("尝试使用GPU后端进行文生图训练...");
                
                // 设置WGPU环境变量
                unsafe {
                    std::env::set_var("WGPU_POWER_PREFERENCE", "HighPerformance");
                    std::env::set_var("WGPU_BACKEND", "vulkan"); // 尝试使用Vulkan后端
                }
                
                // 尝试使用GPU后端，如果失败则回退到CPU
                let gpu_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    use burn_wgpu::WgpuDevice;
                    
                    type TrainBackend = Autodiff<Wgpu>;

                    println!("初始化GPU设备...");
                    let device = WgpuDevice::default();
                    println!("GPU设备信息: {:?}", device);

                    // 使用更小的配置来测试
                    let test_batch_size = 1; // 进一步减小批次大小
                    let test_epochs = 1; // 只训练1个 epoch
                    
                    // 使用更小的模型配置
                    let diffusion_config = DiffusionConfig {
                        image_size: 32, // 减小图像大小
                        in_channels: 3,
                        hidden_channels: 32, // 进一步减小通道数
                        num_timesteps: 100, // 减小时间步数
                        latent_dim: 64, // 减小 latent 维度
                        beta_start,
                        beta_end,
                    };

                    println!("创建GPU模型...");
                    let mut model = DiffusionModel::<TrainBackend>::new(&diffusion_config, &device);
                    let mut optim = AdamConfig::new().init();

                    println!("开始GPU训练... (批次大小: {}, 训练轮数: {})", test_batch_size, test_epochs);

                    for epoch in 0..test_epochs {
                        let epoch_start = Instant::now();
                        let mut total_loss = 0.0f32;
                        let num_batches = (lines.len() + test_batch_size - 1) / test_batch_size;

                        println!("Epoch {}: 共 {} 个批次", epoch + 1, num_batches);

                        for batch_idx in 0..num_batches {
                            println!("处理批次 {}/{}", batch_idx + 1, num_batches);
                            
                            let start_idx = batch_idx * test_batch_size;
                            let end_idx = (start_idx + test_batch_size).min(lines.len());
                            let current_batch_size = end_idx - start_idx;

                            println!("  批次大小: {}", current_batch_size);

                            // 创建随机图像数据
                            let total_elements = current_batch_size * 3 * diffusion_config.image_size * diffusion_config.image_size;
                            println!("  生成随机数据: {} 元素", total_elements);
                            let data: Vec<f32> = (0..total_elements)
                                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                .collect();
                            
                            println!("  创建GPU张量...");
                            let batch_tensor: Tensor<TrainBackend, 4> = Tensor::from_data(
                                TensorData::new(data, [current_batch_size, 3, diffusion_config.image_size, diffusion_config.image_size]),
                                &device,
                            );

                            // VAE前向传播
                            println!("  VAE前向传播...");
                            let (recon, mu, log_var) = model.vae.forward(batch_tensor.clone());

                            // 计算VAE损失
                            println!("  计算VAE损失...");
                            let diff = recon.clone().sub(batch_tensor);
                            let mse_loss = diff.clone().mul(diff).mean();
                            let kl_loss = mu.clone().mul(mu.clone()).add(log_var.clone().exp()).sub(log_var.clone().add_scalar(1.0)).mul_scalar(0.5).mean();
                            let vae_loss = mse_loss + kl_loss.mul_scalar(0.01);

                            // 获取隐空间表示 (z) - 重参数采样
                            println!("  重参数采样...");
                            let std = log_var.clone().mul_scalar(0.5).exp();
                            let z: Tensor<TrainBackend, 4> = Tensor::zeros(mu.dims(), &device);
                            let z = mu.clone().add(std.mul(z));

                            // 随机采样时间步进行扩散模型训练
                            let timestep = rand::random::<usize>() % num_timesteps;
                            let alpha_bar_t = model.get_alpha_bar(timestep);

                            // 生成噪声 - 在隐空间上生成噪声
                            println!("  生成噪声...");
                            let noise_dims = z.dims();
                            let noise_elements: Vec<f32> = (0..noise_dims.iter().product::<usize>())
                                .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                                .collect();
                            let noise: Tensor<TrainBackend, 4> = Tensor::from_data(
                                TensorData::new(noise_elements, noise_dims),
                                &device,
                            );

                            // 加噪 - 在隐空间上加噪
                            println!("  加噪...");
                            let alpha_bar_t_f32 = alpha_bar_t as f32;
                            let noisy_latent = z.clone().mul_scalar(alpha_bar_t_f32.sqrt())
                                .add(noise.clone().mul_scalar((1.0 - alpha_bar_t_f32).sqrt()));

                            // UNet噪声预测
                            println!("  UNet预测...");
                            let time_tensor: Tensor<TrainBackend, 2> = Tensor::full(
                                [current_batch_size, diffusion_config.latent_dim],
                                timestep as i32,
                                &device,
                            );
                            let noise_pred = model.unet.forward(noisy_latent, time_tensor);

                            // 噪声预测损失
                            println!("  计算噪声损失...");
                            let noise_diff = noise_pred.sub(noise);
                            let noise_loss = noise_diff.clone().mul(noise_diff).mean();
                            let total_batch_loss = vae_loss + noise_loss;

                            // 反向传播
                            println!("  反向传播...");
                            let grads = total_batch_loss.backward();
                            let grads = GradientsParams::from_grads(grads, &model);

                            // 更新权重
                            println!("  更新权重...");
                            model = optim.step(learning_rate as f64, model, grads);

                            // 获取损失值
                            println!("  计算损失值...");
                            let loss_val = total_batch_loss.to_data();
                            let loss_vec: Vec<f32> = loss_val.to_vec().unwrap_or_default();
                            if let Some(&l) = loss_vec.first() {
                                total_loss += l;
                                println!("  批次损失: {:.6}", l);
                            }
                        }

                        let avg_loss = total_loss / num_batches as f32;
                        let epoch_time = epoch_start.elapsed();

                        println!("\nEpoch {}/{} - 平均损失: {:.6} - 时间: {:.2}s\n",
                            epoch + 1, test_epochs, avg_loss, epoch_time.as_secs_f32());
                    }

                    // 保存模型
                    println!("训练完成，保存模型到: {}", output_dir);
                    fs::create_dir_all(output_dir).expect("Failed to create output directory");

                    // 保存配置文件
                    use serde::Serialize;
                    #[derive(Serialize)]
                    struct ConfigJson {
                        image_size: usize,
                        in_channels: usize,
                        latent_dim: usize,
                        hidden_channels: usize,
                        num_timesteps: usize,
                        beta_start: f64,
                        beta_end: f64,
                    }

                    let config_json = ConfigJson {
                        image_size: diffusion_config.image_size,
                        in_channels: diffusion_config.in_channels,
                        latent_dim: diffusion_config.latent_dim,
                        hidden_channels: diffusion_config.hidden_channels,
                        num_timesteps: diffusion_config.num_timesteps,
                        beta_start: diffusion_config.beta_start as f64,
                        beta_end: diffusion_config.beta_end as f64,
                    };

                    let config_path = format!("{}/config.json", output_dir);
                    let config_str = serde_json::to_string_pretty(&config_json).expect("Failed to serialize config");
                    fs::write(&config_path, config_str).expect("Failed to save config");
                    println!("配置文件已保存到: {}", config_path);

                    let model_path = format!("{}/diffusion_model.mpk", output_dir);
                    model.save_file(&model_path, &CompactRecorder::new())
                        .expect("Failed to save model");

                    println!("GPU模型已保存到: {}", model_path);
                    Ok::<(), String>(())
                }));

                match gpu_result {
                    Ok(_) => return,
                    Err(_) => {
                        println!("GPU后端初始化失败，自动回退到CPU后端...");
                        // 继续执行CPU后端的代码
                    }
                }
            }

            // CPU后端训练代码
            println!("使用CPU后端进行文生图训练...");
            type TrainBackend = Autodiff<NdArray>;

            println!("初始化模型...");
            let device: <TrainBackend as Backend>::Device = Default::default();
            println!("设备信息: {:?}", device);

            // 使用更小的配置来测试
            let test_batch_size = 2; // 减小批次大小
            let test_epochs = 1; // 只训练1个 epoch
            
            let diffusion_config = DiffusionConfig {
                image_size,
                in_channels: 3,
                hidden_channels,
                num_timesteps,
                latent_dim,
                beta_start,
                beta_end,
            };

            println!("创建模型...");
            let mut model = DiffusionModel::<TrainBackend>::new(&diffusion_config, &device);
            let mut optim = AdamConfig::new().init();

            println!("开始训练... (批次大小: {}, 训练轮数: {})", test_batch_size, test_epochs);

            for epoch in 0..test_epochs {
                let epoch_start = Instant::now();
                let mut total_loss = 0.0f32;
                let num_batches = (lines.len() + test_batch_size - 1) / test_batch_size;

                println!("Epoch {}: 共 {} 个批次", epoch + 1, num_batches);

                for batch_idx in 0..num_batches {
                    println!("处理批次 {}/{}", batch_idx + 1, num_batches);
                    
                    let start_idx = batch_idx * test_batch_size;
                    let end_idx = (start_idx + test_batch_size).min(lines.len());
                    let current_batch_size = end_idx - start_idx;

                    println!("  批次大小: {}", current_batch_size);

                    // 创建随机图像数据
                    let total_elements = current_batch_size * 3 * image_size * image_size;
                    println!("  生成随机数据: {} 元素", total_elements);
                    let data: Vec<f32> = (0..total_elements)
                        .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                        .collect();
                    
                    println!("  创建张量...");
                    let batch_tensor: Tensor<TrainBackend, 4> = Tensor::from_data(
                        TensorData::new(data, [current_batch_size, 3, image_size, image_size]),
                        &device,
                    );

                    // VAE前向传播
                    println!("  VAE前向传播...");
                    let (recon, mu, log_var) = model.vae.forward(batch_tensor.clone());

                    // 计算VAE损失
                    println!("  计算VAE损失...");
                    let diff = recon.clone().sub(batch_tensor);
                    let mse_loss = diff.clone().mul(diff).mean();
                    let kl_loss = mu.clone().mul(mu.clone()).add(log_var.clone().exp()).sub(log_var.clone().add_scalar(1.0)).mul_scalar(0.5).mean();
                    let vae_loss = mse_loss + kl_loss.mul_scalar(0.01);

                    // 获取隐空间表示 (z) - 重参数采样
                    println!("  重参数采样...");
                    let std = log_var.clone().mul_scalar(0.5).exp();
                    let z: Tensor<TrainBackend, 4> = Tensor::zeros(mu.dims(), &device);
                    let z = mu.clone().add(std.mul(z));

                    // 随机采样时间步进行扩散模型训练
                    let timestep = rand::random::<usize>() % num_timesteps;
                    let alpha_bar_t = model.get_alpha_bar(timestep);

                    // 生成噪声 - 在隐空间上生成噪声
                    println!("  生成噪声...");
                    let noise_dims = z.dims();
                    let noise_elements: Vec<f32> = (0..noise_dims.iter().product::<usize>())
                        .map(|_| rand::random::<f32>() * 2.0 - 1.0)
                        .collect();
                    let noise: Tensor<TrainBackend, 4> = Tensor::from_data(
                        TensorData::new(noise_elements, noise_dims),
                        &device,
                    );

                    // 加噪 - 在隐空间上加噪
                    println!("  加噪...");
                    let alpha_bar_t_f32 = alpha_bar_t as f32;
                    let noisy_latent = z.clone().mul_scalar(alpha_bar_t_f32.sqrt())
                        .add(noise.clone().mul_scalar((1.0 - alpha_bar_t_f32).sqrt()));

                    // UNet噪声预测
                    println!("  UNet预测...");
                    let time_tensor: Tensor<TrainBackend, 2> = Tensor::full(
                        [current_batch_size, latent_dim],
                        timestep as i32,
                        &device,
                    );
                    let noise_pred = model.unet.forward(noisy_latent, time_tensor);

                    // 噪声预测损失
                    println!("  计算噪声损失...");
                    let noise_diff = noise_pred.sub(noise);
                    let noise_loss = noise_diff.clone().mul(noise_diff).mean();
                    let total_batch_loss = vae_loss + noise_loss;

                    // 反向传播
                    println!("  反向传播...");
                    let grads = total_batch_loss.backward();
                    let grads = GradientsParams::from_grads(grads, &model);

                    // 更新权重
                    println!("  更新权重...");
                    model = optim.step(learning_rate as f64, model, grads);

                    // 获取损失值
                    println!("  计算损失值...");
                    let loss_val = total_batch_loss.to_data();
                    let loss_vec: Vec<f32> = loss_val.to_vec().unwrap_or_default();
                    if let Some(&l) = loss_vec.first() {
                        total_loss += l;
                        println!("  批次损失: {:.6}", l);
                    }
                }

                let avg_loss = total_loss / num_batches as f32;
                let epoch_time = epoch_start.elapsed();

                println!("\nEpoch {}/{} - 平均损失: {:.6} - 时间: {:.2}s\n",
                    epoch + 1, test_epochs, avg_loss, epoch_time.as_secs_f32());
            }

            // 保存模型
            println!("训练完成，保存模型到: {}", output_dir);
            fs::create_dir_all(output_dir).expect("Failed to create output directory");

            // 保存配置文件
            use serde::Serialize;
            #[derive(Serialize)]
            struct ConfigJson {
                image_size: usize,
                in_channels: usize,
                latent_dim: usize,
                hidden_channels: usize,
                num_timesteps: usize,
                beta_start: f64,
                beta_end: f64,
            }

            let config_json = ConfigJson {
                image_size: diffusion_config.image_size,
                in_channels: diffusion_config.in_channels,
                latent_dim: diffusion_config.latent_dim,
                hidden_channels: diffusion_config.hidden_channels,
                num_timesteps: diffusion_config.num_timesteps,
                beta_start: diffusion_config.beta_start as f64,
                beta_end: diffusion_config.beta_end as f64,
            };

            let config_path = format!("{}/config.json", output_dir);
            let config_str = serde_json::to_string_pretty(&config_json).expect("Failed to serialize config");
            fs::write(&config_path, config_str).expect("Failed to save config");
            println!("配置文件已保存到: {}", config_path);

            let model_path = format!("{}/diffusion_model.mpk", output_dir);
            model.save_file(&model_path, &CompactRecorder::new())
                .expect("Failed to save model");

            println!("模型已保存到: {}", model_path);
            return;
        }

    // For ultra_quick mode, automatically limit data to 100 records for very fast testing
    if args.ultra_quick && args.sft_max_records == 0 {
        args.sft_max_records = 100;
    }

    // Set up Ctrl+C handler for graceful shutdown
    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\n收到中断信号，正在保存检查点...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
        // Give some time for cleanup, then exit
        std::thread::sleep(std::time::Duration::from_secs(2));
        std::process::exit(130);
    })
    .expect("Error setting Ctrl+C handler");

    // 初始化日志
    env_logger::init();

    // 加载或创建分词器
    let tokenizer_path = format!("{}/tokenizer.json", args.output_dir);
    let tokenizer = if !args.reset_tokenizer && Path::new(&tokenizer_path).exists() {
        println!("正在加载现有分词器...");
        Tokenizer::load(&tokenizer_path).expect("Should load tokenizer")
    } else if args.stream {
        println!("正在从语料流式构建分词器...");
        let chars = collect_vocab_chars_stream(&args).expect("Should collect vocab");
        Tokenizer::from_chars(chars)
    } else {
        let text = if let Some(path) = &args.sft_jsonl {
            load_sft_jsonl(&args, path).expect("Should load sft jsonl")
        } else if args.sft_sample_messages {
            load_sft_messages_sample()
        } else if args.sft_sample {
            load_sft_sample(&args)
        } else {
            load_corpus(&args).expect("Should load corpus")
        };
        if text.is_empty() {
            panic!("语料为空。");
        }
        println!("正在从语料创建新分词器...");
        if args.use_bpe {
            Tokenizer::new_bpe(&text, args.bpe_vocab_size)
        } else {
            Tokenizer::new(&text)
        }
    };

    // 从配置文件加载模型配置
    println!("正在从配置文件加载模型配置: {}", args.config_path);
    let mut model_config = ModelConfig::load(&args.config_path).expect("Failed to load model config");

    // 更新动态参数
    model_config.vocab_size = tokenizer.vocab_size;
    model_config.max_seq_len = args.max_seq_len;

    // 根据后端参数选择训练路径
    if args.backend == "gpu" {
        println!("使用GPU后端进行训练...");
        train_with_backend::<Wgpu>(args, tokenizer, model_config);
    } else {
        println!("使用CPU后端进行训练...");
        train_with_backend::<NdArray>(args, tokenizer, model_config);
    }
}

fn train_with_backend<B: Backend>(args: Args, tokenizer: Tokenizer, model_config: ModelConfig) {
    // 根据后端类型优化设备配置
    let device = if args.backend == "gpu" {
        // GPU后端优化配置
        println!("使用优化的GPU后端配置...");
        // 为WGPU设置更好的设备选项
        unsafe {
            std::env::set_var("WGPU_POWER_PREFERENCE", "HighPerformance");
        }
        B::Device::default()
    } else {
        B::Device::default()
    };
    println!("设备信息: {:?}", device);

    println!("词表大小: {}", tokenizer.vocab_size);

    // 3. 配置模型
    let mut model_config = model_config;
    model_config.vocab_size = tokenizer.vocab_size;
    model_config.max_seq_len = args.max_seq_len;
    
    // 配置 LoRA
    if args.use_lora {
        println!("启用 LoRA 微调: rank={}, alpha={}", args.lora_rank, args.lora_alpha);
        use sage::training::lora::LoRAConfig;
        model_config.lora = Some(LoRAConfig {
            rank: args.lora_rank,
            alpha: args.lora_alpha as f64,
            dropout: 0.05, // 默认 dropout
            target_modules: vec!["output_head".to_string()], // 目前仅支持 output_head
        });
    }

    // 配置多模态功能
    if args.multimodal {
        println!("启用多模态功能...");
        use sage::core::multimodal::{MultimodalConfig, MultimodalFusionConfig, VisionEncoderConfig};
        
        let fusion_strategy = args.fusion_strategy.clone();
        
        let multimodal_config = MultimodalConfig {
            vision_encoder: VisionEncoderConfig {
                in_channels: 3,
                hidden_channels: 64, // 默认值
                out_dim: args.vision_out_dim,
                encoder_type: "resnet".to_string(),
                num_layers: 4,
                patch_size: 16,
                image_size: 224,
            },
            fusion: MultimodalFusionConfig {
                text_dim: model_config.d_model,
                vision_dim: args.vision_out_dim,
                output_dim: model_config.d_model,
                strategy: fusion_strategy,
            },
            preprocessing: Default::default(),
            enable_multimodal: true,
        };
        
        model_config.multimodal = Some(multimodal_config);
        println!("多模态配置: 视觉输出维度={}, 融合策略={}", args.vision_out_dim, args.fusion_strategy);
    }

    let num_params = model_config.num_params();
    let params_str = if num_params >= 1_000_000_000 {
        format!("约 {:.3}B", num_params as f64 / 1_000_000_000.0)
    } else if num_params >= 1_000_000 {
        format!("约 {:.3}M", num_params as f64 / 1_000_000.0)
    } else if num_params >= 1_000 {
        format!("约 {:.3}K", num_params as f64 / 1_000.0)
    } else {
        format!("{}", num_params)
    };
    println!("模型参数总量: {} ({})", num_params, params_str);

    // 4. 训练流程
    let model_path = format!("{}/model.mpk", args.output_dir);
    let has_model = Path::new(&model_path).exists();

    if !has_model || args.force || args.r#continue || args.resume_epoch.is_some() {
        if args.ultra_quick {
            println!(
                "启用超快速开发模式：1轮训练，极小批量(2)，极高学习率，只用100条数据，适合闪电验证"
            );
        } else if args.quick_dev {
            println!("启用快速开发模式：1轮训练，超小批量(4)，超高学习率，适合快速验证");
        }
        println!("未发现已训练模型，开始正式训练...");
        let mut training_config = TrainingConfig::create(model_config.clone(), AdamConfig::new());
        training_config.num_epochs = if args.ultra_quick || args.quick_dev {
            1
        } else {
            args.num_epochs
        };
        // 根据后端类型优化批处理大小
        training_config.batch_size = if args.backend == "gpu" {
            // GPU优化：保持用户指定的批量大小，不强制最小值
            if args.ultra_quick {
                4
            } else if args.quick_dev {
                8
            } else if args.fast {
                (args.batch_size() * 4).min(256)
            } else {
                args.batch_size()
            }
        } else {
            // CPU保持原有逻辑
            if args.ultra_quick {
                2
            } else if args.quick_dev {
                4
            } else if args.fast {
                (args.batch_size() * 2).min(128)
            } else {
                args.batch_size()
            }
        };

        // 根据后端类型优化学习率
        training_config.lr = if args.backend == "gpu" {
            if args.ultra_quick || args.quick_dev {
                2e-2
            } else if args.fast {
                args.learning_rate * 3.0
            } else {
                args.learning_rate * 1.5
            }
        } else if args.ultra_quick || args.quick_dev {
            1e-2
        } else if args.fast {
            args.learning_rate * 2.0
        } else {
            args.learning_rate
        };

        // 数据加载线程数：
        // - CPU：多线程 batch（burn BatchDataLoader::multi_thread）可提升吞吐。
        // - GPU(WGPU)：batcher 在子线程里创建 GPU 张量会与 WGPU 设备/队列线程模型冲突，首步训练常直接 panic（exit 101）。必须 num_workers=0。
        let cpu_cores = num_cpus::get();
        let optimal_workers_cpu = if args.fast {
            cpu_cores.max(8)
        } else {
            cpu_cores.max(4)
        };

        training_config.num_workers = if args.backend == "gpu" {
            if args.num_workers != 0 {
                println!(
                    "提示: GPU(WGPU) 训练时数据加载已固定为单线程（num_workers=0）。\
                     多线程 DataLoader 会在子线程创建 WGPU 张量，易导致训练启动崩溃；已忽略 --num-workers {}。",
                    args.num_workers
                );
            }
            0
        } else {
            args.num_workers.max(optimal_workers_cpu)
        };

        training_config.gradient_accumulation_steps = args.gradient_accumulation.max(1);
        training_config.use_lora = args.use_lora;
        training_config.lora_rank = args.lora_rank;
        training_config.lora_alpha = args.lora_alpha;

        println!(
            "数据加载线程数（burn DataLoader workers）: {}",
            training_config.num_workers
        );

        // GPU：可选的显存探测（一步前向+反向）+ 梯度累积以保持等效 batch
        if args.backend == "gpu" {
            println!("GPU优化配置:");
            println!("  - 目标/物理批处理大小: {}", training_config.batch_size);
            println!("  - 学习率: {:.6}", training_config.lr);
            println!("  - 工作线程数: {}", training_config.num_workers);
            println!("  - 高性能GPU模式已启用");

            if args.no_auto_vram {
                println!("已禁用自动显存探测（--no-auto-vram），使用命令行 batch / seq / 梯度累积。");
            } else {
                println!(
                    "\n\
                     --- GPU 显存探测（不是正式训练） ---\n\
                     本阶段仅重复「单次前向 + 反向」以估算显存，不会启动 Burn 的 Learner，\n\
                     因此没有训练 TUI、也没有 epoch 日志；若长时间无输出，多为 WGPU 首次编译内核。\n\
                     探测结束后才会进入正式训练（届时将出现进度 / TUI）。\n\
                     ------------------------------------"
                );
                let effective_batch = training_config.batch_size.max(1);
                let original_seq_len = model_config.max_seq_len.max(1);

                println!("开始自动探测 GPU：对每组 (物理 batch, seq_len) 执行一步前向+反向…");

                let mut configs = Vec::new();
                let mut seq_len = original_seq_len;
                while seq_len >= 1 {
                    let mut batch_size = effective_batch;
                    loop {
                        configs.push((batch_size, seq_len));
                        if batch_size == 1 {
                            break;
                        }
                        batch_size /= 2;
                    }
                    if seq_len == 1 {
                        break;
                    }
                    seq_len /= 2;
                }

                println!("尝试顺序（由大到小）: {} 组配置", configs.len());

                let found = probe_first_fitting_config::<Autodiff<B>>(
                    &device,
                    &training_config.model,
                    &configs,
                );

                match found {
                    Some((micro, sl)) => {
                        let micro_usize: usize = micro as usize;
                        let accum = effective_batch.saturating_add(micro_usize - 1) / micro_usize.max(1);
                        let accum = accum.max(1);
                        let effective_approx = micro_usize.saturating_mul(accum);

                        training_config.batch_size = micro;
                        model_config.max_seq_len = sl;
                        training_config.model.max_seq_len = sl;
                        training_config.gradient_accumulation_steps = accum;

                        println!("");
                        println!("🎯 找到合适的显存配置！");
                        println!("==========================================");
                        println!("  物理 batch = {}", micro);
                        println!("  序列长度 = {}", sl);
                        println!("  梯度累积 = {}", accum);
                        println!("  等效 batch ≈ {}", effective_approx);
                        println!("==========================================");
                        println!("");
                        println!("💡 显存探测阶段已完成，即将进入正式训练...");
                        println!("🔥 接下来将显示 Burn 训练 TUI 和完整的 epoch 训练日志");
                        println!("⏳ 正在准备数据加载器，请稍候...");
                        println!("");
                    }
                    None => {
                        eprintln!(
                            "错误: 所有候选配置均无法在 GPU 上完成一步训练（前向+反向）。\
                             请减小 --batch-size / --max-seq-len，使用更小的 --model-size，\
                             改用 --backend cpu，或先使用 --no-auto-vram 手动调参。"
                        );
                        std::process::exit(1);
                    }
                }
            }
        } else {
            // CPU：仅应用用户指定的梯度累积
            println!(
                "CPU 后端: 梯度累积步数 = {}",
                training_config.gradient_accumulation_steps
            );
        }

        // 学习率调度器配置
        if args.lr_scheduler {
            use sage::configs::config::LRSchedulerConfig;
            let lr_scheduler_config = LRSchedulerConfig {
                lr_max: args.lr_max,
                lr_min: args.lr_min,
                warmup_steps: args.warmup_steps,
                total_steps: args.total_steps,
            };
            training_config.lr_scheduler = Some(lr_scheduler_config);
        }

        // 默认启用TUI，除非明确禁用或使用快速模式
        training_config.no_progress = args.no_progress || args.fast;

        // 如果用户显式请求TUI，则强制启用
        if args.tui || args.force_tui {
            training_config.no_progress = false;
            println!("强制启用TUI进度显示");
        }

        // 设置分布式训练配置
        training_config.distributed = args.distributed;
        if args.distributed {
            if let Some(devices_str) = &args.devices {
                training_config.devices = devices_str.split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
            }
            println!("分布式训练已启用");
            if !training_config.devices.is_empty() {
                println!("使用设备: {:?}", training_config.devices);
            }
        }
        
        // 设置DPO训练配置
        if args.dpo {
            let dpo_config = DPOConfig {
                beta: args.dpo_beta,
                use_kl_regularization: true,
                kl_weight: args.dpo_kl_weight,
            };
            training_config.dpo_config = Some(dpo_config);
            println!("DPO训练已启用");
            println!("DPO参数: beta={}, kl_weight={}", args.dpo_beta, args.dpo_kl_weight);
        }

        println!(
            "进度条状态: {}",
            if training_config.no_progress {
                "已禁用"
            } else {
                "已启用"
            }
        );

        let init_model = if let Some(epoch) = args.resume_epoch {
            let ckpt_path = format!("{}/checkpoint/model-{}.mpk", args.output_dir, epoch);
            Some(
                model_config
                    .init::<Autodiff<B>>(&device)
                    .load_file(&ckpt_path, &burn::record::CompactRecorder::new(), &device)
                    .expect("Should load checkpoint model"),
            )
        } else if has_model && args.r#continue {
            Some(
                model_config
                    .init::<Autodiff<B>>(&device)
                    .load_file(&model_path, &burn::record::CompactRecorder::new(), &device)
                    .expect("Should load model"),
            )
        } else {
            None
        };
        
        // 优化器状态将通过 burn 的检查点机制自动恢复
        let init_optimizer = None;

        // DPO训练模式
        if args.dpo {
            println!("启动DPO训练...");
            
            // 加载DPO数据
            let dpo_data_path = args.dpo_data.as_deref().expect("DPO训练需要指定 --dpo-data");
            let dpo_items = load_dpo_jsonl(dpo_data_path, &tokenizer).expect("加载DPO数据失败");
            
            println!("加载了 {} 条DPO数据", dpo_items.len());
            
            // 启动DPO训练
            train_dpo::<Autodiff<B>>(
                &args.output_dir,
                training_config,
                device,
                &tokenizer,
                dpo_items,
                init_model,
            );
            
            return;
        }

        if args.stream {
            if args.stream_direct {
                let tok = Arc::new(tokenizer.clone());

                let (input_train, input_valid, total_records) = if let Some(path) = &args.sft_jsonl
                {
                    let total = count_sft_records_stream(&args).expect("Should count records");
                    let split = (total as f32 * 0.9) as usize;
                    (
                        SftInput::Jsonl {
                            path: PathBuf::from(path),
                            max_bytes: args.max_bytes,
                            max_records: args.sft_max_records,
                            start_record: 0,
                            end_record: split,
                        },
                        SftInput::Jsonl {
                            path: PathBuf::from(path),
                            max_bytes: args.max_bytes,
                            max_records: args.sft_max_records,
                            start_record: split,
                            end_record: total,
                        },
                        total,
                    )
                } else if args.sft_sample_messages {
                    (SftInput::SampleMessages, SftInput::SampleMessages, 1usize)
                } else if args.sft_sample {
                    (SftInput::Sample, SftInput::Sample, 1usize)
                } else {
                    panic!(
                        "--stream-direct 目前只支持 --sft-jsonl / --sft-sample / --sft-sample-messages"
                    );
                };

                let items_total = if args.sft_jsonl.is_some() {
                    if args.max_bytes == 0 {
                        (total_records as usize).saturating_mul(args.max_seq_len).max(1)
                    } else {
                        args.max_bytes.max(1)
                    }
                } else {
                    1_000_000usize
                };

                let device_clone = device.clone();
                let dataloader_train = Arc::new(StreamingSftDataLoader::<Autodiff<B>> {
                    tokenizer: Arc::clone(&tok),
                    device: device.clone(),
                    batch_size: training_config.batch_size,
                    seq_len: training_config.model.max_seq_len,
                    input: input_train,
                    items_total,
                });

                let dataloader_valid = Arc::new(StreamingSftDataLoader::<B> {
                    tokenizer: Arc::clone(&tok),
                    device: device_clone,
                    batch_size: training_config.batch_size,
                    seq_len: training_config.model.max_seq_len,
                    input: input_valid,
                    items_total,
                });

                train_with_loaders::<Autodiff<B>>(
                    &args.output_dir,
                    training_config,
                    device,
                    &tokenizer,
                    dataloader_train,
                    dataloader_valid,
                    init_model,
                    init_optimizer,
                );
            } else {
                let (tokens_path, mask_path) =
                    build_token_cache_stream(&args, &tokenizer).expect("Should build token cache");

                train_from_cache::<Autodiff<B>>(
                    &args.output_dir,
                    training_config,
                    device,
                    &tokenizer,
                    &tokens_path,
                    &mask_path,
                    init_model,
                    init_optimizer,
                );
            }
        } else {
            let text = if let Some(path) = &args.sft_jsonl {
                load_sft_jsonl(&args, path).expect("Should load sft jsonl")
            } else if args.sft_sample_messages {
                load_sft_messages_sample()
            } else if args.sft_sample {
                load_sft_sample(&args)
            } else {
                load_corpus(&args).expect("Should load corpus")
            };
            if text.is_empty() {
                panic!("语料为空。");
            }

            let (tokens, mask) =
                if args.sft_jsonl.is_some() || args.sft_sample || args.sft_sample_messages {
                    tokenizer.encode_with_assistant_mask(&text)
                } else {
                    let tokens = tokenizer.encode(&text);
                    let mask = vec![1u8; tokens.len()];
                    (tokens, mask)
                };

            train::<Autodiff<B>>(
                &args.output_dir,
                training_config,
                device,
                &tokenizer,
                tokens,
                mask,
                init_model,
                init_optimizer,
            );
        }
    } else {
        println!("发现已存在模型，跳过训练。");
    }

    println!("\n训练流程完成！模型已保存在 '{}'", args.output_dir);
}
