use burn::backend::{Autodiff, ndarray::NdArray, wgpu::Wgpu};
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::prelude::Backend;
use sage::{
    model::ModelConfig,
    streaming::{SftInput, StreamingSftDataLoader},
    tokenizer::Tokenizer,
    train, train_from_cache, train_with_loaders,
    TrainingConfig,
};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs,
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    corpus: Option<String>,

    #[arg(long)]
    corpus_dir: Option<String>,

    #[arg(long, default_value_t = 50_000_000)]
    max_bytes: usize,

    #[arg(long, default_value_t = false)]
    stream: bool,

    #[arg(long, default_value_t = false)]
    stream_direct: bool,

    #[arg(long)]
    sft_jsonl: Option<String>,

    #[arg(long, default_value_t = false)]
    sft_sample: bool,

    #[arg(long, default_value_t = false)]
    sft_sample_messages: bool,

    #[arg(long, default_value_t = 0)]
    sft_max_records: usize,

    /// 模型和训练产物的输出目录。
    #[arg(long, default_value = "./tmp/sage_model_formal")]
    artifact_dir: String,

    /// 训练的总轮数。
    #[arg(long, default_value_t = 50)]
    num_epochs: usize,

    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    #[arg(long, default_value_t = 1.0e-4)]
    lr: f64,

    #[arg(long, default_value_t = 256)]
    max_seq_len: usize,

    #[arg(long, default_value_t = false)]
    force: bool,

    #[arg(long, default_value_t = false)]
    r#continue: bool,

    #[arg(long)]
    resume_epoch: Option<usize>,

    #[arg(long, default_value_t = false)]
    reset_tokenizer: bool,

    #[arg(long, default_value_t = false)]
    use_bpe: bool,

    #[arg(long, default_value_t = 5000)]
    bpe_vocab_size: usize,

    /// Enable high parallel training mode (faster data loading + batch throughput)
    #[arg(long, default_value_t = false)]
    fast: bool,

    /// Worker threads for data loading
    #[arg(long, default_value_t = 4)]
    num_workers: usize,

    /// Enable quick development mode (1 epoch, small batch, high lr)
    #[arg(long, default_value_t = false)]
    quick_dev: bool,

    /// Enable ultra-quick development mode (1 epoch, tiny batch, limit data to 100 records)
    #[arg(long, default_value_t = false)]
    ultra_quick: bool,

    /// Disable progress bars and TUI display
    #[arg(long, default_value_t = false)]
    no_progress: bool,

    /// Enable TUI progress display (may not work in all terminals, especially Windows PowerShell)
    #[arg(long, default_value_t = false)]
    tui: bool,

    /// Backend to use for training: cpu or gpu
    #[arg(long, default_value = "cpu", value_name = "cpu|gpu")]
    backend: String,

    /// Model size configuration: default (1M), 10m, 30m
    #[arg(long, default_value = "default", value_name = "default|10m|30m")]
    model_size: String,

    /// Training mode: general, code, math
    #[arg(long, default_value = "general", value_name = "general|code|math")]
    training_mode: String,

    /// Force enable TUI progress display even in environments that might not support it
    #[arg(long, default_value_t = false)]
    force_tui: bool,
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

    let path = args.corpus.as_deref().unwrap_or("corpus_cn.txt");
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

    let path = args.corpus.as_deref().unwrap_or("corpus_cn.txt");
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
    let cache_dir = Path::new(&args.artifact_dir).join("cache");
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
            for (&id, &m) in ids.iter().zip(mask.iter()) {
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

    let path = args.corpus.as_deref().unwrap_or("corpus_cn.txt");
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
    let mut args = Args::parse();

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

    // 设置环境变量改善Burn TUI显示
    unsafe {
        std::env::set_var("BURN_TUI_NO_CLEAR", "1");
        std::env::set_var("TERM", "xterm-256color");
        std::env::set_var("BURN_TUI_ENABLED", "1");
        std::env::set_var("BURN_TUI_FORCE", "1");
        std::env::set_var("WGPU_LOG", "off");

        // 禁用burn框架的实验日志记录器，解决Windows下file logger安装失败的问题
        std::env::set_var("BURN_EXPERIMENT_LOGGER_DISABLED", "1");

        // 设置日志级别
        std::env::set_var("RUST_LOG", "burn_train=info,wgpu_core=off,burn_core=off");
    }

    // 初始化日志
    env_logger::init();

    // 加载或创建分词器
    let tokenizer_path = format!("{}/tokenizer.json", args.artifact_dir);
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

    // 根据模型大小选择配置
    let mut model_config = match args.model_size.as_str() {
        "10m" => {
            println!("使用约10M参数的模型配置");
            ModelConfig::small_10m()
        }
        "30m" => {
            println!("使用约30M参数的模型配置");
            ModelConfig::medium_30m()
        }
        _ => {
            println!("使用默认模型配置（约1M参数）");
            ModelConfig::new()
        }
    };

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

    let model_init = model_config.init::<B>(&device);
    println!("模型参数总量: {} (约 0.001B)", model_init.num_params());

    // 4. 训练流程
    let model_path = format!("{}/model.mpk", args.artifact_dir);
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
        let mut training_config = TrainingConfig::new(model_config.clone(), AdamConfig::new());
        training_config.num_epochs = if args.ultra_quick || args.quick_dev {
            1
        } else {
            args.num_epochs
        };
        // 根据后端类型优化批处理大小
        training_config.batch_size = if args.backend == "gpu" {
            // GPU优化：更大的批处理大小以提高利用率
            if args.ultra_quick {
                4
            } else if args.quick_dev {
                8
            } else if args.fast {
                (args.batch_size * 4).min(256)
            } else {
                args.batch_size.max(32)
            }
        } else {
            // CPU保持原有逻辑
            if args.ultra_quick {
                2
            } else if args.quick_dev {
                4
            } else if args.fast {
                (args.batch_size * 2).min(128)
            } else {
                args.batch_size
            }
        };

        // 根据后端类型优化学习率
        training_config.lr = if args.backend == "gpu" {
            if args.ultra_quick || args.quick_dev {
                2e-2
            } else if args.fast {
                args.lr * 3.0
            } else {
                args.lr * 1.5
            }
        } else if args.ultra_quick || args.quick_dev {
            1e-2
        } else if args.fast {
            args.lr * 2.0
        } else {
            args.lr
        };

        // 根据后端类型优化并行工作线程数
        let cpu_cores = num_cpus::get();
        let optimal_workers = if args.backend == "gpu" {
            // GPU模式下：更多工作线程用于数据预处理，避免GPU等待
            if args.fast {
                cpu_cores.max(12)
            } else {
                cpu_cores.max(8)
            }
        } else {
            // CPU模式下：保持平衡以避免过度竞争
            if args.fast {
                cpu_cores.max(8)
            } else {
                cpu_cores.max(4)
            }
        };
        training_config.num_workers = args.num_workers.max(optimal_workers);

        println!(
            "使用 {} 个工作线程进行数据加载",
            training_config.num_workers
        );

        // GPU性能优化提示
        if args.backend == "gpu" {
            println!("GPU优化配置:");
            println!("  - 批处理大小: {}", training_config.batch_size);
            println!("  - 学习率: {:.6}", training_config.lr);
            println!("  - 工作线程数: {}", training_config.num_workers);
            println!("  - 高性能GPU模式已启用");
        }

        // 默认启用TUI，除非明确禁用或使用快速模式
        training_config.no_progress = args.no_progress || args.fast;

        // 如果用户显式请求TUI，则强制启用
        if args.tui || args.force_tui {
            training_config.no_progress = false;
            println!("强制启用TUI进度显示");
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
            let ckpt_path = format!("{}/checkpoint/model-{}.mpk", args.artifact_dir, epoch);
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
                        total_records.saturating_mul(args.max_seq_len).max(1)
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
                    batch_size: args.batch_size,
                    seq_len: args.max_seq_len,
                    input: input_train,
                    items_total,
                });

                let dataloader_valid = Arc::new(StreamingSftDataLoader::<B> {
                    tokenizer: Arc::clone(&tok),
                    device: device_clone,
                    batch_size: args.batch_size,
                    seq_len: args.max_seq_len,
                    input: input_valid,
                    items_total,
                });

                train_with_loaders::<Autodiff<B>>(
                    &args.artifact_dir,
                    training_config,
                    device,
                    &tokenizer,
                    dataloader_train,
                    dataloader_valid,
                    init_model,
                );
            } else {
                let (tokens_path, mask_path) =
                    build_token_cache_stream(&args, &tokenizer).expect("Should build token cache");

                train_from_cache::<Autodiff<B>>(
                    &args.artifact_dir,
                    training_config,
                    device,
                    &tokenizer,
                    &tokens_path,
                    &mask_path,
                    init_model,
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
                &args.artifact_dir,
                training_config,
                device,
                &tokenizer,
                tokens,
                mask,
                init_model,
            );
        }
    } else {
        println!("发现已存在模型，跳过训练。");
    }

    println!("\n训练流程完成！模型已保存在 '{}'", args.artifact_dir);
}
