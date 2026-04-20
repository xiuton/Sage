use crate::core::model::Model;
use crate::quantization::quantization::QuantizedModel;
use crate::core::tokenizer::Tokenizer;
use burn::prelude::*;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{SeedableRng, rngs::StdRng};
use std::collections::{HashSet, HashMap};
use std::time::Instant;
use log;

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub punctuation_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: Option<u64>,
    pub context_len: usize,
    pub stop_on_user: bool,
    pub stop_sequences: Vec<String>,
    pub use_kv_cache: bool,
    pub streaming: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 100,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            punctuation_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            context_len: 512,
            stop_on_user: false,
            stop_sequences: Vec::new(),
            use_kv_cache: true,
            streaming: false,
        }
    }
}

pub enum ModelType<'a, B: Backend> {
    Normal(&'a Model<B>),
    Quantized(&'a QuantizedModel<B>),
    Multimodal(&'a Model<B>, &'a Tensor<B, 4>),
}

pub struct GenerationState<'a, B: Backend> {
    model: ModelType<'a, B>,
    tokenizer: &'a Tokenizer,
    tokens: Vec<usize>,
    rng: StdRng,
    user_token_ids: Vec<usize>,
    stop_sequence_ids: Vec<Vec<usize>>,
    seen_tokens: HashSet<usize>,
    token_frequency: HashMap<usize, usize>,
    options: GenerateOptions,
    device: &'a B::Device,
    generated_tokens: usize,
    stopped: bool,
    cache: Option<burn::nn::transformer::TransformerEncoderAutoregressiveCache<B>>,
    last_token_only: bool,
}

impl<'a, B: Backend> GenerationState<'a, B> {
    pub fn new(
        model: ModelType<'a, B>,
        tokenizer: &'a Tokenizer,
        prompt: &str,
        options: &'a GenerateOptions,
        device: &'a B::Device,
    ) -> Self {
        log::info!("[Generation] Creating GenerationState for prompt: {}", prompt);
        let mut tokens = tokenizer.encode(prompt);
        log::info!("[Generation] Encoded {} tokens", tokens.len());
        
        if tokens.is_empty() {
            tokens.push(tokenizer.bos_id);
        }

        let rng = match options.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let user_token_ids = if options.stop_on_user {
            tokenizer.encode("<user>")
        } else {
            Vec::new()
        };

        let stop_sequence_ids: Vec<Vec<usize>> = options.stop_sequences
            .iter()
            .map(|seq| tokenizer.encode(seq))
            .collect();

        let seen_tokens: HashSet<usize> = tokens.iter().copied().collect();
        let token_frequency: HashMap<usize, usize> = tokens.iter().fold(HashMap::new(), |mut map, &token| {
            *map.entry(token).or_insert(0) += 1;
            map
        });

        // 初始化缓存
        let cache = if options.use_kv_cache {
            match &model {
                ModelType::Normal(model) => Some(model.new_autoregressive_cache()),
                ModelType::Quantized(_) => None, // 量化模型暂不支持缓存
                ModelType::Multimodal(_, _) => None, // 多模态模型暂不支持缓存
            }
        } else {
            None
        };

        Self {
            model,
            tokenizer,
            tokens,
            rng,
            user_token_ids,
            stop_sequence_ids,
            seen_tokens,
            token_frequency,
            options: options.clone(),
            device,
            generated_tokens: 0,
            stopped: false,
            cache,
            last_token_only: options.use_kv_cache,
        }
    }

    pub fn next_token(&mut self) -> Option<String> {
        if self.stopped || self.generated_tokens >= self.options.max_new_tokens {
            return None;
        }

        let step_start = Instant::now();
        
        // 确定输入序列
        let input_tokens = if self.last_token_only && self.generated_tokens > 0 {
            // 只使用最后一个 token（启用 KV 缓存时）
            &self.tokens[self.tokens.len() - 1..]
        } else {
            // 使用完整的上下文窗口
            let window_start = self.tokens.len().saturating_sub(self.options.context_len.max(1));
            &self.tokens[window_start..]
        };
        
        let input_prep_start = Instant::now();
        let input = Tensor::<B, 1, Int>::from_ints(
            input_tokens
                .iter()
                .map(|&t| t as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            self.device,
        )
        .unsqueeze::<2>();
        let input_prep_duration = input_prep_start.elapsed();
        
        let forward_start = Instant::now();
        let output = match &self.model {
            ModelType::Normal(model) => {
                if let Some(cache) = &mut self.cache {
                    model.forward_autoregressive_inference(input, cache)
                } else {
                    model.forward(input)
                }
            },
            ModelType::Quantized(model) => model.forward(input),
            ModelType::Multimodal(model, image) => {
                use crate::core::multimodal::MultimodalInput;
                let multimodal_input = MultimodalInput::new(input, (**image).clone());
                model.forward_multimodal(multimodal_input)
            },
        };
        let forward_duration = forward_start.elapsed();
        let [_, seq_len, _] = output.dims();
        
        let token_process_start = Instant::now();

        let last_token_logits = 
            output.slice([0..1, (seq_len - 1)..seq_len, 0..self.tokenizer.vocab_size]);

        let mut logits_vec: Vec<f32> = last_token_logits
            .to_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        // 应用温度
        let temperature = self.options.temperature.max(1.0e-5);
        for v in logits_vec.iter_mut() {
            *v /= temperature;
        }

        // 应用频率和存在惩罚
        for (token_id, freq) in &self.token_frequency {
            if *freq > 0 {
                let penalty = 1.0 + self.options.frequency_penalty * *freq as f32;
                logits_vec[*token_id] /= penalty;
                if self.options.presence_penalty > 0.0 {
                    logits_vec[*token_id] -= self.options.presence_penalty;
                }
            }
        }

        // 计算概率分布
        let max_logit = logits_vec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for v in logits_vec.iter_mut() {
            *v = (*v - max_logit).exp();
            exp_sum += *v;
        }

        let probs_vec: Vec<f32> = if exp_sum > 0.0 {
            logits_vec.into_iter().map(|v| v / exp_sum).collect()
        } else {
            vec![1.0 / self.tokenizer.vocab_size as f32; self.tokenizer.vocab_size]
        };

        // 应用 top-k 过滤
        let mut indexed_probs: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut candidates = indexed_probs;
        candidates.truncate(self.options.top_k.min(candidates.len()).max(1));

        // 应用 top-p 过滤
        if self.options.top_p > 0.0 && self.options.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut cut = 0usize;
            for (_, p) in candidates.iter() {
                cum += *p;
                cut += 1;
                if cum >= self.options.top_p {
                    break;
                }
            }
            candidates.truncate(cut.max(1));
        }

        // 应用重复惩罚
        let mut weights: Vec<f32> = candidates.iter().map(|&(_, p)| p).collect();
        if self.options.repetition_penalty > 1.0 {
            for (idx, (token_id, _)) in candidates.iter().enumerate() {
                if self.seen_tokens.contains(token_id) {
                    weights[idx] /= self.options.repetition_penalty;
                }
            }
        }

        // 应用标点符号惩罚
        if self.options.punctuation_penalty > 1.0 {
            let last_is_punct = self.tokens
                .last()
                .map(|&id| self.tokenizer.is_punctuation_token(id))
                .unwrap_or(false);

            for (idx, (token_id, _)) in candidates.iter().enumerate() {
                let is_punct = self.tokenizer.is_punctuation_token(*token_id);
                if is_punct {
                    weights[idx] /= self.options.punctuation_penalty;
                    if last_is_punct {
                        weights[idx] /= self.options.punctuation_penalty;
                    }
                }
            }
        }
        
        // 采样
        let indices: Vec<usize> = candidates.iter().map(|&(i, _)| i).collect();
        let sampled_idx = match WeightedIndex::new(&weights) {
            Ok(dist) => indices[dist.sample(&mut self.rng)],
            Err(_) => indices[0],
        };

        let token_char = self.tokenizer.char_for_id(sampled_idx)?;
        let token_process_duration = token_process_start.elapsed();
        let step_duration = step_start.elapsed();
        
        // 打印性能日志
        log::info!("[Generation] Token #{}, input_prep={:?}, forward={:?}, token_process={:?}, total={:?}", 
            self.generated_tokens + 1,
            input_prep_duration,
            forward_duration,
            token_process_duration,
            step_duration
        );
        
        // 更新状态
        self.tokens.push(sampled_idx);
        self.seen_tokens.insert(sampled_idx);
        *self.token_frequency.entry(sampled_idx).or_insert(0) += 1;
        self.generated_tokens += 1;

        // 检查停止条件
        if sampled_idx == self.tokenizer.eos_id {
            self.stopped = true;
            return Some(token_char.to_string());
        }

        let tokens_len = self.tokens.len();

        // 检查用户停止标记
        if !self.user_token_ids.is_empty() && tokens_len >= self.user_token_ids.len() {
            let end = tokens_len;
            let start = end - self.user_token_ids.len();
            if self.tokens[start..end] == self.user_token_ids {
                self.stopped = true;
            }
        }

        // 检查停止序列
        if !self.stopped {
            for stop_seq_ids in &self.stop_sequence_ids {
                if tokens_len >= stop_seq_ids.len() {
                    let end = tokens_len;
                    let start = end - stop_seq_ids.len();
                    if self.tokens[start..end] == *stop_seq_ids {
                        self.stopped = true;
                        break;
                    }
                }
            }
        }

        Some(token_char.to_string())
    }

    pub fn get_full_text(&self) -> String {
        self.tokenizer.decode(&self.tokens)
    }

    pub fn get_generated_text(&self) -> String {
        let prompt_len = self.tokens.len() - self.generated_tokens;
        if prompt_len < self.tokens.len() {
            self.tokenizer.decode(&self.tokens[prompt_len..])
        } else {
            "".to_string()
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped || self.generated_tokens >= self.options.max_new_tokens
    }

    pub fn tokens(&self) -> &Vec<usize> {
        &self.tokens
    }

    pub fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }
}

pub fn generate<B: Backend>(
    model: &Model<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    options: &GenerateOptions,
    device: &B::Device,
) -> String {
    generate_with_model_type(ModelType::Normal(model), tokenizer, prompt, options, device)
}

pub fn generate_quantized<B: Backend>(
    model: &QuantizedModel<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    options: &GenerateOptions,
    device: &B::Device,
) -> String {
    generate_with_model_type(ModelType::<B>::Quantized(model), tokenizer, prompt, options, device)
}

pub fn generate_multimodal<B: Backend>(
    model: &Model<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    image: &Tensor<B, 4>,
    options: &GenerateOptions,
    device: &B::Device,
) -> String {
    generate_with_model_type(ModelType::Multimodal(model, image), tokenizer, prompt, options, device)
}

// 流式生成函数
pub fn generate_stream<B: Backend, F>(
    model: &Model<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    options: &GenerateOptions,
    device: &B::Device,
    mut callback: F,
) -> String
where
    F: FnMut(String) -> bool,
{
    let mut state = GenerationState::new(ModelType::Normal(model), tokenizer, prompt, options, device);
    let mut full_text = String::new();
    
    while !state.is_stopped() {
        if let Some(token) = state.next_token() {
            full_text.push_str(&token);
            if !callback(token) {
                break;
            }
        }
    }
    
    full_text
}

fn generate_with_model_type<B: Backend>(
    model: ModelType<'_, B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    options: &GenerateOptions,
    device: &B::Device,
) -> String {
    if options.max_new_tokens == 0 {
        return tokenizer.encode(prompt).iter()
            .filter_map(|&id| tokenizer.char_for_id(id))
            .collect::<String>();
    }

    let mut state = GenerationState::new(model, tokenizer, prompt, options, device);
    
    while !state.is_stopped() {
        state.next_token();
    }
    
    state.get_full_text()
}

pub fn batch_generate<B: Backend>(
    model: &Model<B>,
    tokenizer: &Tokenizer,
    prompts: &[&str],
    options: &GenerateOptions,
    device: &B::Device,
) -> Vec<String> {
    if prompts.is_empty() {
        return Vec::new();
    }

    let mut states: Vec<GenerationState<'_, B>> = prompts
        .iter()
        .map(|prompt| {
            GenerationState::new(
                ModelType::Normal(model),
                tokenizer,
                prompt,
                options,
                device,
            )
        })
        .collect();

    let max_iterations = options.max_new_tokens;
    for _ in 0..max_iterations {
        let all_stopped = states.iter_mut().all(|s| {
            s.next_token();
            s.is_stopped()
        });
        if all_stopped {
            break;
        }
    }

    states.into_iter().map(|s| s.get_full_text()).collect()
}

// 批处理流式生成
pub fn batch_generate_stream<B: Backend, F>(
    model: &Model<B>,
    tokenizer: &Tokenizer,
    prompts: &[&str],
    options: &GenerateOptions,
    device: &B::Device,
    callbacks: &mut [F],
) -> Vec<String>
where
    F: FnMut(String) -> bool,
{
    if prompts.is_empty() || callbacks.len() != prompts.len() {
        return Vec::new();
    }

    let mut states: Vec<GenerationState<'_, B>> = prompts
        .iter()
        .map(|prompt| {
            GenerationState::new(
                ModelType::Normal(model),
                tokenizer,
                prompt,
                options,
                device,
            )
        })
        .collect();

    let max_iterations = options.max_new_tokens;
    for _ in 0..max_iterations {
        let mut all_stopped = true;
        
        for (i, state) in states.iter_mut().enumerate() {
            if !state.is_stopped() {
                all_stopped = false;
                if let Some(token) = state.next_token() {
                    if !callbacks[i](token) {
                        state.stopped = true;
                    }
                }
            }
        }
        
        if all_stopped {
            break;
        }
    }

    states.into_iter().map(|s| s.get_full_text()).collect()
}
