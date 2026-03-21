use crate::model::Model;
use crate::tokenizer::Tokenizer;
use burn::prelude::*;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{SeedableRng, rngs::StdRng};

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub punctuation_penalty: f32,
    pub seed: Option<u64>,
    pub context_len: usize,
}

pub fn generate<B: Backend>(
    model: &Model<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    options: &GenerateOptions,
    device: &B::Device,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    let mut rng = match options.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    for _ in 0..options.max_new_tokens {
        let window_start = tokens.len().saturating_sub(options.context_len.max(1));
        let window_tokens = &tokens[window_start..];
        let input = Tensor::<B, 1, Int>::from_ints(
            window_tokens
                .iter()
                .map(|&t| t as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        )
        .unsqueeze::<2>();

        let output = model.forward(input);
        let [_, seq_len, _] = output.dims();

        // Get logits for the last token: [1, 1, vocab_size]
        let last_token_logits =
            output.slice([0..1, (seq_len - 1)..seq_len, 0..tokenizer.vocab_size]);

        let mut logits_vec: Vec<f32> = last_token_logits
            .to_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        let temperature = options.temperature.max(1.0e-5);
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
            vec![1.0 / tokenizer.vocab_size as f32; tokenizer.vocab_size]
        };

        // Top-K / Top-P sampling
        let mut indexed_probs: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut candidates = indexed_probs;
        candidates.truncate(options.top_k.min(candidates.len()).max(1));

        if options.top_p > 0.0 && options.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut cut = 0usize;
            for (_, p) in candidates.iter() {
                cum += *p;
                cut += 1;
                if cum >= options.top_p {
                    break;
                }
            }
            candidates.truncate(cut.max(1));
        }

        let mut weights: Vec<f32> = candidates.iter().map(|&(_, p)| p).collect();
        if options.repetition_penalty > 1.0 {
            for (idx, (token_id, _)) in candidates.iter().enumerate() {
                if tokens.contains(token_id) {
                    weights[idx] /= options.repetition_penalty;
                }
            }
        }

        if options.punctuation_penalty > 1.0 {
            let last_is_punct = tokens
                .last()
                .map(|&id| tokenizer.is_punctuation_token(id))
                .unwrap_or(false);

            for (idx, (token_id, _)) in candidates.iter().enumerate() {
                let is_punct = tokenizer.is_punctuation_token(*token_id);
                if is_punct {
                    weights[idx] /= options.punctuation_penalty;
                    if last_is_punct {
                        weights[idx] /= options.punctuation_penalty;
                    }
                }
            }
        }
        let indices: Vec<usize> = candidates.iter().map(|&(i, _)| i).collect();

        // Sample from the top-k distribution
        let sampled_idx = match WeightedIndex::new(&weights) {
            Ok(dist) => indices[dist.sample(&mut rng)],
            Err(_) => indices[0],
        };

        tokens.push(sampled_idx);

        if sampled_idx == tokenizer.eos_id {
            break;
        }
    }

    tokenizer.decode(&tokens)
}
