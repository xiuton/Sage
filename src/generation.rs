use burn::prelude::*;
use crate::model::Model;
use crate::tokenizer::Tokenizer;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;

pub fn generate<B: Backend>(
    model: &Model<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    device: &B::Device,
) -> String {
    let mut tokens = tokenizer.encode(prompt);
    
    for _ in 0..max_new_tokens {
        let _seq_len_input = tokens.len();
        let input = Tensor::<B, 1, Int>::from_ints(
            tokens.iter().map(|&t| t as i32).collect::<Vec<_>>().as_slice(),
            device
        ).unsqueeze::<2>();
        
        let output = model.forward(input);
        let [_, seq_len, _] = output.dims();
        
        // Get logits for the last token: [1, 1, vocab_size]
        let last_token_logits = output.slice([0..1, (seq_len - 1)..seq_len, 0..tokenizer.vocab_size]);
        
        // Convert to 1D tensor of logits
        let logits = last_token_logits.flatten::<1>(0, 2);
        
        // Apply temperature
        let logits = logits / temperature;
        
        // Get probabilities using softmax
        let probs = burn::tensor::activation::softmax(logits, 0);
        let probs_data = probs.to_data();
        let probs_vec: Vec<f32> = probs_data.as_slice::<f32>().unwrap().to_vec();
        
        // Top-K sampling
        let mut indexed_probs: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_k_probs = &indexed_probs[..top_k.min(indexed_probs.len())];
        let weights: Vec<f32> = top_k_probs.iter().map(|&(_, p)| p).collect();
        let indices: Vec<usize> = top_k_probs.iter().map(|&(i, _)| i).collect();
        
        // Sample from the top-k distribution
        let dist = WeightedIndex::new(&weights).unwrap();
        let mut rng = thread_rng();
        let sampled_idx = indices[dist.sample(&mut rng)];
        
        tokens.push(sampled_idx);
        
        // If EOS token is generated, stop (if we had one, for now we just continue)
        if sampled_idx == tokenizer.eos_id {
            break;
        }
    }
    
    tokenizer.decode(&tokens)
}
