use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalMetrics {
    pub image_captioning_score: f64,
    pub vision_language_alignment: f64,
    pub generation_quality: f64,
}

impl Default for MultimodalMetrics {
    fn default() -> Self {
        Self {
            image_captioning_score: 0.0,
            vision_language_alignment: 0.0,
            generation_quality: 0.0,
        }
    }
}

pub struct MultimodalEvaluator;

impl MultimodalEvaluator {
    pub fn new() -> Self {
        Self
    }

    pub fn evaluate_image_captioning(
        &self,
        generated_caption: &str,
        reference_caption: &str,
    ) -> f64 {
        let generated_words: Vec<&str> = generated_caption.split_whitespace().collect();
        let reference_words: Vec<&str> = reference_caption.split_whitespace().collect();

        if generated_words.is_empty() || reference_words.is_empty() {
            return 0.0;
        }

        let mut common_count = 0;
        for gen_word in &generated_words {
            for ref_word in &reference_words {
                if gen_word.to_lowercase() == ref_word.to_lowercase() {
                    common_count += 1;
                    break;
                }
            }
        }

        let precision = common_count as f64 / generated_words.len() as f64;
        let recall = common_count as f64 / reference_words.len() as f64;

        if precision + recall == 0.0 {
            return 0.0;
        }

        2.0 * precision * recall / (precision + recall)
    }

    pub fn compute_bleu_score(
        &self,
        generated_text: &str,
        reference_text: &str,
    ) -> f64 {
        self.evaluate_image_captioning(generated_text, reference_text)
    }
}

pub struct MetricsLogger {
    pub epoch: usize,
    pub train_loss: f64,
    pub valid_loss: f64,
    pub multimodal_score: f64,
}

impl MetricsLogger {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            train_loss: 0.0,
            valid_loss: 0.0,
            multimodal_score: 0.0,
        }
    }

    pub fn log(&mut self, metrics: MultimodalMetrics) {
        println!("多模态评估指标:");
        println!("  图像描述分数: {:.4}", metrics.image_captioning_score);
        println!("  视觉语言对齐: {:.4}", metrics.vision_language_alignment);
        println!("  生成质量: {:.4}", metrics.generation_quality);
    }

    pub fn summary(&self) -> String {
        format!(
            "Epoch {} - Train Loss: {:.4}, Valid Loss: {:.4}, Multimodal Score: {:.4}",
            self.epoch, self.train_loss, self.valid_loss, self.multimodal_score
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_captioning_score() {
        let evaluator = MultimodalEvaluator::new();

        let score = evaluator.evaluate_image_captioning(
            "a cat sitting on a couch",
            "a cat sitting on a sofa",
        );

        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_bleu_score() {
        let evaluator = MultimodalEvaluator::new();

        let score = evaluator.compute_bleu_score(
            "the quick brown fox jumps",
            "the fast brown fox leaps",
        );

        assert!(score >= 0.0);
    }
}
