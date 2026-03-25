use sage::performance::{PerformanceMonitor, run_benchmark};
use std::time::Instant;

#[test]
fn test_performance_monitor_new() {
    let monitor = PerformanceMonitor::new();
    let metrics = monitor.get_all_metrics();
    
    assert!(metrics.is_empty());
}

#[test]
fn test_record_inference() {
    let monitor = PerformanceMonitor::new();
    let start_time = Instant::now();
    
    // 记录一个推理
    let metrics = monitor.record_inference("/test/endpoint", start_time, 100, 50);
    
    assert_eq!(metrics.prompt_tokens, 100);
    assert_eq!(metrics.completion_tokens, 50);
    assert_eq!(metrics.total_tokens, 150);
}

#[test]
fn test_get_average_metrics() {
    let monitor = PerformanceMonitor::new();
    
    // 记录多次推理
    monitor.record_inference("/test/endpoint", Instant::now(), 100, 50);
    monitor.record_inference("/test/endpoint", Instant::now(), 200, 100);
    
    let avg_metrics = monitor.get_average_metrics("/test/endpoint").unwrap();
    
    assert_eq!(avg_metrics.prompt_tokens, 150);
    assert_eq!(avg_metrics.completion_tokens, 75);
    assert_eq!(avg_metrics.total_tokens, 225);
}

#[test]
fn test_get_all_metrics() {
    let monitor = PerformanceMonitor::new();
    
    monitor.record_inference("/endpoint1", Instant::now(), 100, 50);
    monitor.record_inference("/endpoint2", Instant::now(), 200, 100);
    
    let all_metrics = monitor.get_all_metrics();
    
    assert!(all_metrics.contains_key("/endpoint1"));
    assert!(all_metrics.contains_key("/endpoint2"));
    assert_eq!(all_metrics["/endpoint1"].len(), 1);
    assert_eq!(all_metrics["/endpoint2"].len(), 1);
}

#[test]
fn test_clear_metrics() {
    let monitor = PerformanceMonitor::new();
    
    monitor.record_inference("/test/endpoint", Instant::now(), 100, 50);
    assert!(!monitor.get_all_metrics().is_empty());
    
    monitor.clear_metrics();
    assert!(monitor.get_all_metrics().is_empty());
}

#[test]
fn test_run_benchmark() {
    let result = run_benchmark("test_benchmark", 5, || {
        // 模拟一些计算
        std::thread::sleep(std::time::Duration::from_millis(10));
        (100, 50) // prompt_tokens, completion_tokens
    });
    
    assert_eq!(result.name, "test_benchmark");
    assert_eq!(result.iterations, 5);
    assert!(result.avg_time_ms > 0.0);
    assert!(result.tokens_per_second > 0.0);
}
