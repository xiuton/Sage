use burn::backend::{
    Autodiff,
    ndarray::{NdArray, NdArrayDevice},
};
use burn::module::Module;
use burn::optim::AdamConfig;
use sage::{
    model::ModelConfig,
    streaming::{SftInput, StreamingSftDataLoader},
    tokenizer::Tokenizer,
    training::{self, TrainingConfig},
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
    #[arg(long, default_value_t = 20)]
    num_epochs: usize,

    #[arg(long, default_value_t = 16)]
    batch_size: usize,

    #[arg(long, default_value_t = 1.0e-4)]
    lr: f64,

    #[arg(long, default_value_t = 64)]
    max_seq_len: usize,

    #[arg(long, default_value_t = false)]
    force: bool,

    #[arg(long, default_value_t = false)]
    r#continue: bool,

    #[arg(long)]
    resume_epoch: Option<usize>,

    #[arg(long, default_value_t = false)]
    reset_tokenizer: bool,
}

#[derive(Deserialize)]
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

fn sft_template(prompt: &str, response: &str) -> String {
    let mut out = String::new();
    out.push('\u{0002}');
    out.push_str("用户：");
    out.push_str(prompt);
    out.push('\n');
    out.push_str("助手：");
    out.push_str(response);
    out.push('\u{0003}');
    out.push('\n');
    out
}

fn sft_messages_template(messages: &[SftMessage]) -> Option<String> {
    let mut out = String::new();
    out.push('\u{0002}');

    let mut has_assistant = false;
    for m in messages {
        match m.role.as_str() {
            "user" => {
                out.push_str("用户：");
                out.push_str(&m.content);
                out.push('\n');
            }
            "assistant" => {
                has_assistant = true;
                out.push_str("助手：");
                out.push_str(&m.content);
                out.push('\u{0003}');
                out.push('\n');
            }
            _ => {}
        }
    }

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
            out.push_str(&sft_template(&rec.prompt, &rec.response));
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

fn load_sft_sample() -> String {
    let samples = [
        SftRecord {
            prompt: "你是谁？".to_string(),
            response: "我是一个用 Rust 训练出来的小模型。".to_string(),
        },
        SftRecord {
            prompt: "用一句话解释千字文是什么。".to_string(),
            response: "《千字文》是由一千个不重复汉字组成的启蒙文章。".to_string(),
        },
        SftRecord {
            prompt: "给我一个学习 Rust 的建议。".to_string(),
            response: "从所有权和借用入手，多写小项目并配合 clippy 修正。".to_string(),
        },
    ];

    let mut out = String::new();
    for rec in samples {
        out.push_str(&sft_template(&rec.prompt, &rec.response));
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

fn sft_sample_from_json_line(line: &str) -> Option<String> {
    if let Ok(rec) = serde_json::from_str::<SftRecord>(line) {
        return Some(sft_template(&rec.prompt, &rec.response));
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
            let sample = match sft_sample_from_json_line(trimmed) {
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
        let text = load_sft_sample();
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
            let sample = match sft_sample_from_json_line(trimmed) {
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
        let text = load_sft_sample();
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
        if sft_sample_from_json_line(trimmed).is_none() {
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
    let args = Args::parse();

    // 初始化日志
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }
    env_logger::init();

    let device = NdArrayDevice::Cpu;

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
            load_sft_sample()
        } else {
            load_corpus(&args).expect("Should load corpus")
        };
        if text.is_empty() {
            panic!("语料为空。");
        }
        println!("正在从语料创建新分词器...");
        Tokenizer::new(&text)
    };
    println!("词表大小: {}", tokenizer.vocab_size);

    // 3. 配置模型
    let mut model_config = ModelConfig::new();
    model_config.vocab_size = tokenizer.vocab_size;
    model_config.max_seq_len = args.max_seq_len;

    let model_init = model_config.init::<NdArray>(&device);
    println!("模型参数总量: {} (约 0.001B)", model_init.num_params());

    // 4. 训练流程
    let model_path = format!("{}/model.mpk", args.artifact_dir);
    let has_model = Path::new(&model_path).exists();

    if !has_model || args.force || args.r#continue || args.resume_epoch.is_some() {
        println!("未发现已训练模型，开始正式训练...");
        let mut training_config = TrainingConfig::new(model_config.clone(), AdamConfig::new());
        training_config.num_epochs = args.num_epochs;
        training_config.batch_size = args.batch_size;
        training_config.lr = args.lr;

        let init_model = if let Some(epoch) = args.resume_epoch {
            let ckpt_path = format!("{}/checkpoint/model-{}.mpk", args.artifact_dir, epoch);
            Some(
                model_config
                    .init::<Autodiff<NdArray>>(&device)
                    .load_file(&ckpt_path, &burn::record::CompactRecorder::new(), &device)
                    .expect("Should load checkpoint model"),
            )
        } else if has_model && args.r#continue {
            Some(
                model_config
                    .init::<Autodiff<NdArray>>(&device)
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

                let dataloader_train = Arc::new(StreamingSftDataLoader::<Autodiff<NdArray>> {
                    tokenizer: Arc::clone(&tok),
                    device,
                    batch_size: args.batch_size,
                    seq_len: args.max_seq_len,
                    input: input_train,
                    items_total,
                });

                let dataloader_valid = Arc::new(StreamingSftDataLoader::<NdArray> {
                    tokenizer: Arc::clone(&tok),
                    device,
                    batch_size: args.batch_size,
                    seq_len: args.max_seq_len,
                    input: input_valid,
                    items_total,
                });

                training::train_with_loaders::<Autodiff<NdArray>>(
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

                training::train_from_cache::<Autodiff<NdArray>>(
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
                load_sft_sample()
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

            training::train::<Autodiff<NdArray>>(
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
