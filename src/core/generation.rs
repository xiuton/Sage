use crate::core::kv_cache::KVCache;
use crate::core::model::Model;
use crate::quantization::quantization::QuantizedModel;
use crate::core::tokenizer::Tokenizer;
use burn::prelude::*;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{SeedableRng, rngs::StdRng};
use std::collections::HashSet;

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
        }
    }
}

pub enum ModelType<'a, B: Backend> {
    Normal(&'a Model<B>),
    Quantized(&'a QuantizedModel<B>),
}

pub struct GenerationState<'a, B: Backend> {
    model: ModelType<'a, B>,
    tokenizer: &'a Tokenizer,
    tokens: Vec<usize>,
    rng: StdRng,
    user_token_ids: Vec<usize>,
    stop_sequence_ids: Vec<Vec<usize>>,
    seen_tokens: HashSet<usize>,
    options: GenerateOptions,
    device: &'a B::Device,
    generated_tokens: usize,
    stopped: bool,
    kv_cache: Option<KVCache<B>>,
}

impl<'a, B: Backend> GenerationState<'a, B> {
    pub fn new(
        model: ModelType<'a, B>,
        tokenizer: &'a Tokenizer,
        prompt: &str,
        options: &'a GenerateOptions,
        device: &'a B::Device,
    ) -> Self {
        let mut tokens = tokenizer.encode(prompt);
        
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

        Self {
            model,
            tokenizer,
            tokens,
            rng,
            user_token_ids,
            stop_sequence_ids,
            seen_tokens,
            options: options.clone(),
            device,
            generated_tokens: 0,
            stopped: false,
            kv_cache: Some(KVCache::new()),
        }
    }

    pub fn next_token(&mut self) -> Option<String> {
        if self.stopped || self.generated_tokens >= self.options.max_new_tokens {
            return None;
        }

        let window_start = self.tokens.len().saturating_sub(self.options.context_len.max(1));
        let window_tokens = &self.tokens[window_start..];
        
        let input = Tensor::<B, 1, Int>::from_ints(
            window_tokens
                .iter()
                .map(|&t| t as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            self.device,
        )
        .unsqueeze::<2>();

        let output = match &self.model {
            ModelType::Normal(model) => {
                if let Some(kv_cache) = &mut self.kv_cache {
                    model.forward_with_cache(input, Some(kv_cache))
                } else {
                    model.forward(input)
                }
            },
            ModelType::Quantized(model) => model.forward(input),
        };
        let [_, seq_len, _] = output.dims();

        let last_token_logits =
            output.slice([0..1, (seq_len - 1)..seq_len, 0..self.tokenizer.vocab_size]);

        let mut logits_vec: Vec<f32> = last_token_logits
            .to_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let temperature = self.options.temperature.max(1.0e-5);
        for v in logits_vec.iter_mut() {
            *v /= temperature;
        }

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

        let mut indexed_probs: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut candidates = indexed_probs;
        candidates.truncate(self.options.top_k.min(candidates.len()).max(1));

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

        let mut weights: Vec<f32> = candidates.iter().map(|&(_, p)| p).collect();
        if self.options.repetition_penalty > 1.0 {
            for (idx, (token_id, _)) in candidates.iter().enumerate() {
                if self.seen_tokens.contains(token_id) {
                    weights[idx] /= self.options.repetition_penalty;
                }
            }
        }

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
        
        let indices: Vec<usize> = candidates.iter().map(|&(i, _)| i).collect();

        let sampled_idx = match WeightedIndex::new(&weights) {
            Ok(dist) => indices[dist.sample(&mut self.rng)],
            Err(_) => indices[0],
        };

        let token_char = self.tokenizer.char_for_id(sampled_idx)?;
        self.tokens.push(sampled_idx);
        self.seen_tokens.insert(sampled_idx);
        self.generated_tokens += 1;

        // Check stop conditions
        if sampled_idx == self.tokenizer.eos_id {
            self.stopped = true;
            return Some(token_char.to_string());
        }

        let tokens_len = self.tokens.len();

        if !self.user_token_ids.is_empty() && tokens_len >= self.user_token_ids.len() {
            let end = tokens_len;
            let start = end - self.user_token_ids.len();
            if self.tokens[start..end] == self.user_token_ids {
                self.stopped = true;
            }
        }

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

    pub fn is_stopped(&self) -> bool {
        self.stopped || self.generated_tokens >= self.options.max_new_tokens
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
    generate_with_model_type(ModelType::Quantized(model), tokenizer, prompt, options, device)
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
