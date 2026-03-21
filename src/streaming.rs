use crate::{data::TextBatch, tokenizer::Tokenizer};
use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::prelude::*;
use serde::Deserialize;
use std::{
    collections::VecDeque,
    fs,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::Arc,
};

#[derive(Clone)]
pub enum SftInput {
    Jsonl {
        path: PathBuf,
        max_bytes: usize,
        max_records: usize,
        start_record: usize,
        end_record: usize,
    },
    Sample,
    SampleMessages,
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

fn sft_sample_from_json_line(line: &str) -> Option<String> {
    if let Ok(rec) = serde_json::from_str::<SftRecord>(line) {
        return Some(sft_template(&rec.prompt, &rec.response));
    }

    let v = serde_json::from_str::<serde_json::Value>(line).ok()?;
    let messages = v.get("messages")?;
    let messages = serde_json::from_value::<Vec<SftMessage>>(messages.clone()).ok()?;
    sft_messages_template(&messages)
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

#[derive(Clone)]
pub struct StreamingSftDataLoader<B: Backend> {
    pub tokenizer: Arc<Tokenizer>,
    pub device: B::Device,
    pub batch_size: usize,
    pub seq_len: usize,
    pub input: SftInput,
    pub items_total: usize,
}

impl<B: Backend> DataLoader<TextBatch<B>> for StreamingSftDataLoader<B> {
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<TextBatch<B>> + 'a> {
        Box::new(StreamingSftIterator::<B>::new(self.clone()))
    }

    fn num_items(&self) -> usize {
        self.items_total
    }
}

pub struct StreamingSftIterator<B: Backend> {
    loader: StreamingSftDataLoader<B>,
    items_processed: usize,
    token_buf: VecDeque<usize>,
    mask_buf: VecDeque<u8>,
    current_tokens: Vec<usize>,
    current_mask: Vec<u8>,
    current_pos: usize,
    record_index: usize,
    reader: Option<BufReader<fs::File>>,
    bytes_used: usize,
}

impl<B: Backend> StreamingSftIterator<B> {
    fn new(loader: StreamingSftDataLoader<B>) -> Self {
        let reader = match &loader.input {
            SftInput::Jsonl { path, .. } => {
                let file = fs::File::open(path).expect("Open sft jsonl");
                Some(BufReader::new(file))
            }
            _ => None,
        };

        Self {
            loader,
            items_processed: 0,
            token_buf: VecDeque::new(),
            mask_buf: VecDeque::new(),
            current_tokens: Vec::new(),
            current_mask: Vec::new(),
            current_pos: 0,
            record_index: 0,
            reader,
            bytes_used: 0,
        }
    }

    fn next_token(&mut self) -> Option<(usize, u8)> {
        if self.current_pos < self.current_tokens.len() {
            let id = self.current_tokens[self.current_pos];
            let m = self.current_mask[self.current_pos];
            self.current_pos += 1;
            return Some((id, m));
        }

        self.current_tokens.clear();
        self.current_mask.clear();
        self.current_pos = 0;

        match &self.loader.input {
            SftInput::Sample => {
                if self.record_index > 0 {
                    return None;
                }
                let text = load_sft_sample();
                let (ids, mask) = self.loader.tokenizer.encode_with_assistant_mask(&text);
                self.current_tokens = ids;
                self.current_mask = mask;
                self.record_index += 1;
            }
            SftInput::SampleMessages => {
                if self.record_index > 0 {
                    return None;
                }
                let text = load_sft_messages_sample();
                let (ids, mask) = self.loader.tokenizer.encode_with_assistant_mask(&text);
                self.current_tokens = ids;
                self.current_mask = mask;
                self.record_index += 1;
            }
            SftInput::Jsonl {
                max_bytes,
                max_records,
                start_record,
                end_record,
                ..
            } => {
                let reader = self.reader.as_mut()?;
                let mut line = String::new();
                loop {
                    line.clear();
                    let n = reader.read_line(&mut line).ok()?;
                    if n == 0 {
                        return None;
                    }
                    self.bytes_used = self.bytes_used.saturating_add(n);
                    if *max_bytes != 0 && self.bytes_used > *max_bytes {
                        return None;
                    }

                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    let sample = match sft_sample_from_json_line(trimmed) {
                        Some(v) => v,
                        None => continue,
                    };

                    let current_record = self.record_index;
                    self.record_index += 1;

                    if *max_records != 0 && current_record >= *max_records {
                        return None;
                    }
                    if current_record < *start_record {
                        continue;
                    }
                    if current_record >= *end_record {
                        return None;
                    }

                    let (ids, mask) = self.loader.tokenizer.encode_with_assistant_mask(&sample);
                    self.current_tokens = ids;
                    self.current_mask = mask;
                    break;
                }
            }
        }

        if self.current_tokens.is_empty() {
            None
        } else {
            let id = self.current_tokens[0];
            let m = self.current_mask[0];
            self.current_pos = 1;
            Some((id, m))
        }
    }

    fn next_item(&mut self) -> Option<(Vec<i32>, Vec<i32>)> {
        while self.token_buf.len() < self.loader.seq_len + 1 {
            let (id, m) = self.next_token()?;
            self.token_buf.push_back(id);
            self.mask_buf.push_back(m);
        }

        let mut input = Vec::with_capacity(self.loader.seq_len);
        let mut target = Vec::with_capacity(self.loader.seq_len);

        for i in 0..self.loader.seq_len {
            input.push(*self.token_buf.get(i)? as i32);
            let token_id = *self.token_buf.get(i + 1)? as i32;
            let m = *self.mask_buf.get(i + 1)?;
            if m == 1 {
                target.push(token_id);
            } else {
                target.push(0);
            }
        }

        self.token_buf.pop_front();
        self.mask_buf.pop_front();
        if let Some((id, m)) = self.next_token() {
            self.token_buf.push_back(id);
            self.mask_buf.push_back(m);
        }

        Some((input, target))
    }
}

impl<B: Backend> Iterator for StreamingSftIterator<B> {
    type Item = TextBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut inputs = Vec::with_capacity(self.loader.batch_size);
        let mut targets = Vec::with_capacity(self.loader.batch_size);

        for _ in 0..self.loader.batch_size {
            let (input, target) = self.next_item()?;
            inputs.push(
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from(input.as_slice()),
                    &self.loader.device,
                )
                .unsqueeze::<2>(),
            );
            targets.push(
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from(target.as_slice()),
                    &self.loader.device,
                )
                .unsqueeze::<2>(),
            );
        }

        self.items_processed = self.items_processed.saturating_add(self.loader.batch_size);
        let inputs = Tensor::cat(inputs, 0);
        let targets = Tensor::cat(targets, 0);
        Some(TextBatch { inputs, targets })
    }
}

impl<B: Backend> DataLoaderIterator<TextBatch<B>> for StreamingSftIterator<B> {
    fn progress(&self) -> Progress {
        let total = self.loader.items_total.max(self.items_processed).max(1);
        Progress::new(self.items_processed, total)
    }
}
