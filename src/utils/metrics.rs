pub fn perplexity_from_loss(loss: f64) -> f64 {
    loss.exp()
}

pub fn average_perplexity(losses: &[f64]) -> f64 {
    if losses.is_empty() {
        return f64::INFINITY;
    }
    let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
    perplexity_from_loss(avg_loss)
}

pub fn bleu_score(reference: &str, hypothesis: &str, n: usize) -> f64 {
    let ref_tokens: Vec<&str> = reference.split_whitespace().collect();
    let hyp_tokens: Vec<&str> = hypothesis.split_whitespace().collect();
    
    if hyp_tokens.is_empty() {
        return 0.0;
    }
    
    let mut precision = 0.0;
    let max_n = n.min(ref_tokens.len()).min(hyp_tokens.len());
    
    for i in 1..=max_n {
        let ref_ngrams = get_ngrams(&ref_tokens, i);
        let hyp_ngrams = get_ngrams(&hyp_tokens, i);
        
        let mut matches = 0;
        for hyp_ngram in &hyp_ngrams {
            if ref_ngrams.contains(hyp_ngram) {
                matches += 1;
            }
        }
        
        let p = matches as f64 / hyp_ngrams.len() as f64;
        precision += p.ln();
    }
    
    precision /= max_n as f64;
    let brevity_penalty = if hyp_tokens.len() <= ref_tokens.len() {
        (1.0 - ref_tokens.len() as f64 / hyp_tokens.len() as f64).exp()
    } else {
        1.0
    };
    
    brevity_penalty * precision.exp()
}

fn get_ngrams(tokens: &[&str], n: usize) -> Vec<String> {
    let mut ngrams = Vec::new();
    for i in 0..=tokens.len().saturating_sub(n) {
        let ngram = tokens[i..i + n].join(" ");
        ngrams.push(ngram);
    }
    ngrams
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity() {
        assert!((perplexity_from_loss(0.0) - 1.0).abs() < 1e-6);
        assert!((perplexity_from_loss(1.0) - 2.71828).abs() < 1e-3);
    }

    #[test]
    fn test_average_perplexity() {
        let losses = vec![0.0, 0.6931, 1.0986];
        let ppl = average_perplexity(&losses);
        assert!((ppl - 1.817).abs() < 1e-2);
    }
}

