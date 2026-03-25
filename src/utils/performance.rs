use std::time::Instant;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub inference_time_ms: f64,
    pub tokens_per_second: f64,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug)]
pub struct PerformanceMonitor {
    metrics: Arc<Mutex<HashMap<String, Vec<PerformanceMetrics>>>>,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn record_inference(
        &self,
        endpoint: &str,
        start_time: Instant,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> PerformanceMetrics {
        let duration = start_time.elapsed();
        let inference_time_ms = duration.as_secs_f64() * 1000.0;
        let total_tokens = prompt_tokens + completion_tokens;
        let tokens_per_second = if inference_time_ms > 0.0 {
            total_tokens as f64 / (inference_time_ms / 1000.0)
        } else {
            0.0
        };

        let metrics = PerformanceMetrics {
            inference_time_ms,
            tokens_per_second,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        };

        let mut metrics_map = self.metrics.lock().unwrap();
        metrics_map
            .entry(endpoint.to_string())
            .or_default()
            .push(metrics.clone());

        metrics
    }

    pub fn get_average_metrics(&self, endpoint: &str) -> Option<PerformanceMetrics> {
        let metrics_map = self.metrics.lock().unwrap();
        let metrics_list = metrics_map.get(endpoint)?;
        
        if metrics_list.is_empty() {
            return None;
        }

        let mut total_inference_time = 0.0;
        let mut total_tokens_per_second = 0.0;
        let mut total_prompt_tokens = 0;
        let mut total_completion_tokens = 0;
        let mut total_total_tokens = 0;

        for metrics in metrics_list {
            total_inference_time += metrics.inference_time_ms;
            total_tokens_per_second += metrics.tokens_per_second;
            total_prompt_tokens += metrics.prompt_tokens;
            total_completion_tokens += metrics.completion_tokens;
            total_total_tokens += metrics.total_tokens;
        }

        let count = metrics_list.len() as f64;

        Some(PerformanceMetrics {
            inference_time_ms: total_inference_time / count,
            tokens_per_second: total_tokens_per_second / count,
            prompt_tokens: (total_prompt_tokens as f64 / count) as usize,
            completion_tokens: (total_completion_tokens as f64 / count) as usize,
            total_tokens: (total_total_tokens as f64 / count) as usize,
        })
    }

    pub fn get_all_metrics(&self) -> HashMap<String, Vec<PerformanceMetrics>> {
        let metrics_map = self.metrics.lock().unwrap();
        metrics_map.clone()
    }

    pub fn clear_metrics(&self) {
        let mut metrics_map = self.metrics.lock().unwrap();
        metrics_map.clear();
    }
}

pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub tokens_per_second: f64,
}

pub fn run_benchmark<F>(name: &str, iterations: usize, mut func: F) -> BenchmarkResult
where
    F: FnMut() -> (usize, usize),
{
    let mut times = Vec::with_capacity(iterations);
    let mut total_tokens = 0;

    for _ in 0..iterations {
        let start = Instant::now();
        let (prompt_tokens, completion_tokens) = func();
        let duration = start.elapsed();
        times.push(duration.as_secs_f64() * 1000.0);
        total_tokens += prompt_tokens + completion_tokens;
    }

    let avg_time_ms = times.iter().sum::<f64>() / iterations as f64;
    let min_time_ms = *times.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_time_ms = *times.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let total_time_ms = times.iter().sum::<f64>();
    let tokens_per_second = if total_time_ms > 0.0 {
        total_tokens as f64 / (total_time_ms / 1000.0)
    } else {
        0.0
    };

    BenchmarkResult {
        name: name.to_string(),
        iterations,
        avg_time_ms,
        min_time_ms,
        max_time_ms,
        tokens_per_second,
    }
}
