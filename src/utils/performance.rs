use std::time::Instant;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, serde::Serialize)]
pub struct PerformanceMetrics {
    pub inference_time_ms: f64,
    pub tokens_per_second: f64,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub memory_usage_mb: f64,
    pub gpu_utilization: Option<f64>,
    pub cpu_utilization: f64,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub model_parameters: usize,
    pub timestamp: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub batch_loss: f64,
    pub learning_rate: f64,
    pub gradient_norm: f64,
    pub training_time_ms: f64,
    pub samples_per_second: f64,
    pub memory_usage_mb: f64,
    pub timestamp: u64,
}

#[derive(Debug)]
pub struct PerformanceMonitor {
    metrics: Arc<Mutex<HashMap<String, Vec<PerformanceMetrics>>>>,
    training_metrics: Arc<Mutex<HashMap<String, Vec<TrainingMetrics>>>>,
    system_metrics: Arc<Mutex<HashMap<String, Vec<SystemMetrics>>>>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemMetrics {
    pub total_memory_mb: f64,
    pub used_memory_mb: f64,
    pub available_memory_mb: f64,
    pub cpu_usage_percent: f64,
    pub gpu_usage_percent: Option<f64>,
    pub gpu_memory_mb: Option<f64>,
    pub timestamp: u64,
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
            training_metrics: Arc::new(Mutex::new(HashMap::new())),
            system_metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 记录系统指标
    pub fn record_system_metrics(&self, endpoint: &str, metrics: SystemMetrics) {
        let mut metrics_map = self.system_metrics.lock().unwrap();
        metrics_map
            .entry(endpoint.to_string())
            .or_default()
            .push(metrics);
    }

    /// 记录训练指标
    pub fn record_training_metrics(&self, endpoint: &str, metrics: TrainingMetrics) {
        let mut metrics_map = self.training_metrics.lock().unwrap();
        metrics_map
            .entry(endpoint.to_string())
            .or_default()
            .push(metrics);
    }

    pub fn record_inference(
        &self,
        endpoint: &str,
        start_time: Instant,
        prompt_tokens: usize,
        completion_tokens: usize,
        batch_size: usize,
        sequence_length: usize,
        model_parameters: usize,
    ) -> PerformanceMetrics {
        let duration = start_time.elapsed();
        let inference_time_ms = duration.as_secs_f64() * 1000.0;
        let total_tokens = prompt_tokens + completion_tokens;
        let tokens_per_second = if inference_time_ms > 0.0 {
            total_tokens as f64 / (inference_time_ms / 1000.0)
        } else {
            0.0
        };

        // 模拟系统指标（实际项目中应该从系统API获取）
        let memory_usage_mb = (total_tokens * 4) as f64 / 1024.0 / 1024.0; // 模拟内存使用
        let cpu_utilization = 0.15; // 模拟CPU使用率
        let gpu_utilization = Some(0.25); // 模拟GPU使用率

        let metrics = PerformanceMetrics {
            inference_time_ms,
            tokens_per_second,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            memory_usage_mb,
            gpu_utilization,
            cpu_utilization,
            batch_size,
            sequence_length,
            model_parameters,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
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
            memory_usage_mb: 0.0,
            gpu_utilization: None,
            cpu_utilization: 0.0,
            batch_size: 0,
            sequence_length: 0,
            model_parameters: 0,
            timestamp: 0,
        })
    }

    pub fn get_all_metrics(&self) -> HashMap<String, Vec<PerformanceMetrics>> {
        let metrics_map = self.metrics.lock().unwrap();
        metrics_map.clone()
    }

    pub fn clear_metrics(&self) {
        let mut metrics_map = self.metrics.lock().unwrap();
        metrics_map.clear();
        
        let mut training_map = self.training_metrics.lock().unwrap();
        training_map.clear();
        
        let mut system_map = self.system_metrics.lock().unwrap();
        system_map.clear();
    }

    /// 获取所有端点的性能统计
    pub fn get_performance_summary(&self) -> HashMap<String, PerformanceSummary> {
        let metrics_map = self.metrics.lock().unwrap();
        let mut summary = HashMap::new();
        
        for (endpoint, metrics_list) in metrics_map.iter() {
            if !metrics_list.is_empty() {
                summary.insert(endpoint.clone(), PerformanceSummary::from_metrics(metrics_list));
            }
        }
        
        summary
    }

    /// 导出性能数据为JSON格式
    pub fn export_metrics_json(&self) -> String {
        let metrics_map = self.metrics.lock().unwrap();
        let training_map = self.training_metrics.lock().unwrap();
        let system_map = self.system_metrics.lock().unwrap();
        
        let export_data = serde_json::json!({
            "inference_metrics": metrics_map.clone(),
            "training_metrics": training_map.clone(),
            "system_metrics": system_map.clone(),
            "export_timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });
        
        serde_json::to_string_pretty(&export_data).unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub endpoint: String,
    pub total_requests: usize,
    pub avg_inference_time_ms: f64,
    pub avg_tokens_per_second: f64,
    pub total_tokens_processed: usize,
    pub peak_memory_usage_mb: f64,
    pub success_rate: f64,
}

impl PerformanceSummary {
    pub fn from_metrics(metrics: &[PerformanceMetrics]) -> Self {
        let total_requests = metrics.len();
        let mut total_inference_time = 0.0;
        let mut total_tokens_per_second = 0.0;
        let mut total_tokens_processed = 0;
        let mut peak_memory_usage: f64 = 0.0;
        
        for metric in metrics {
            total_inference_time += metric.inference_time_ms;
            total_tokens_per_second += metric.tokens_per_second;
            total_tokens_processed += metric.total_tokens;
            peak_memory_usage = peak_memory_usage.max(metric.memory_usage_mb);
        }
        
        Self {
            endpoint: "".to_string(),
            total_requests,
            avg_inference_time_ms: total_inference_time / total_requests as f64,
            avg_tokens_per_second: total_tokens_per_second / total_requests as f64,
            total_tokens_processed,
            peak_memory_usage_mb: peak_memory_usage,
            success_rate: 1.0, // 简化实现
        }
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
