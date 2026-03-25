use burn::prelude::*;
use burn_ndarray::NdArrayDevice;
use burn_wgpu::Wgpu;
use sage::{
    quantization::quantization::{QuantizationMode, QuantizedModel},
    tokenizer::Tokenizer,
    TrainingConfig,
};
use std::collections::HashMap;

fn main() {
    let config_path = "config.toml";
    let model_path = "sage_model.burn";
    
    let config = TrainingConfig::load(config_path).expect("Failed to load config");
    
    println!("=== 量化精度评估 ===");
    println!("配置文件: {}", config_path);
    println!("模型文件: {}", model_path);
    
    // 测试CPU精度
    println!("\n=== CPU精度评估 ===");
    evaluate_cpu_accuracy(&config, model_path);
    
    // 测试GPU精度
    println!("\n=== GPU精度评估 ===");
    evaluate_gpu_accuracy(&config, model_path);
}

fn evaluate_cpu_accuracy(config: &TrainingConfig, model_path: &str) {
    let device = NdArrayDevice::Cpu;
    
    // 加载原始模型
    let original_model = config.model
        .init::<burn_ndarray::NdArray>(&device)
        .load_file(model_path, &burn::record::CompactRecorder::new(), &device)
        .expect("Failed to load model");
    
    // 加载量化模型（动态量化）
    let quantized_model_dynamic = QuantizedModel::new(&original_model, QuantizationMode::Dynamic);
    
    // 加载量化模型（静态量化）
    let quantized_model_static = QuantizedModel::new(&original_model, QuantizationMode::Static);
    
    let tokenizer = Tokenizer::new("");
    
    let test_prompts = vec![
        "今天天气很好，我们去",
        "人工智能的发展前景",
        "如何学习编程",
        "中国的历史文化",
        "计算机科学的未来",
    ];
    
    println!("测试样本数: {}", test_prompts.len());
    println!("-" );
    
    let mut original_results = HashMap::new();
    let mut dynamic_results = HashMap::new();
    let mut static_results = HashMap::new();
    
    // 获取原始模型结果
    for prompt in &test_prompts {
        let response = sage::generation::generate(&original_model, &tokenizer, prompt, &sage::generation::GenerateOptions::default(), &device);
        println!("原始模型 - \"{}\": {}", prompt, response);
        original_results.insert(*prompt, response);
    }
    
    println!("\n" );
    
    // 获取动态量化模型结果
    for prompt in &test_prompts {
        let response = sage::generation::generate_quantized(&quantized_model_dynamic, &tokenizer, prompt, &sage::generation::GenerateOptions::default(), &device);
        println!("动态量化 - \"{}\": {}", prompt, response);
        dynamic_results.insert(*prompt, response);
    }
    
    println!("\n" );
    
    // 获取静态量化模型结果
    for prompt in &test_prompts {
        let response = sage::generation::generate_quantized(&quantized_model_static, &tokenizer, prompt, &sage::generation::GenerateOptions::default(), &device);
        println!("静态量化 - \"{}\": {}", prompt, response);
        static_results.insert(*prompt, response);
    }
    
    println!("\n=== 精度评估结果 ===");
    evaluate_accuracy(&original_results, &dynamic_results, "动态量化");
    evaluate_accuracy(&original_results, &static_results, "静态量化");
}

fn evaluate_gpu_accuracy(config: &TrainingConfig, model_path: &str) {
    let device = <Wgpu as Backend>::Device::default();
    
    // 加载原始模型
    let original_model = config.model
        .init::<Wgpu>(&device)
        .load_file(model_path, &burn::record::CompactRecorder::new(), &device)
        .expect("Failed to load model");
    
    // 加载量化模型（动态量化）
    let quantized_model_dynamic = QuantizedModel::new(&original_model, QuantizationMode::Dynamic);
    
    // 加载量化模型（静态量化）
    let quantized_model_static = QuantizedModel::new(&original_model, QuantizationMode::Static);
    
    let tokenizer = Tokenizer::new("");
    
    let test_prompts = vec![
        "今天天气很好，我们去",
        "人工智能的发展前景",
        "如何学习编程",
        "中国的历史文化",
        "计算机科学的未来",
    ];
    
    println!("测试样本数: {}", test_prompts.len());
    println!("-" );
    
    let mut original_results = HashMap::new();
    let mut dynamic_results = HashMap::new();
    let mut static_results = HashMap::new();
    
    // 获取原始模型结果
    for prompt in &test_prompts {
        let response = sage::generation::generate(&original_model, &tokenizer, prompt, &sage::generation::GenerateOptions::default(), &device);
        println!("原始模型 - \"{}\": {}", prompt, response);
        original_results.insert(*prompt, response);
    }
    
    println!("\n" );
    
    // 获取动态量化模型结果
    for prompt in &test_prompts {
        let response = sage::generation::generate_quantized(&quantized_model_dynamic, &tokenizer, prompt, &sage::generation::GenerateOptions::default(), &device);
        println!("动态量化 - \"{}\": {}", prompt, response);
        dynamic_results.insert(*prompt, response);
    }
    
    println!("\n" );
    
    // 获取静态量化模型结果
    for prompt in &test_prompts {
        let response = sage::generation::generate_quantized(&quantized_model_static, &tokenizer, prompt, &sage::generation::GenerateOptions::default(), &device);
        println!("静态量化 - \"{}\": {}", prompt, response);
        static_results.insert(*prompt, response);
    }
    
    println!("\n=== 精度评估结果 ===");
    evaluate_accuracy(&original_results, &dynamic_results, "动态量化");
    evaluate_accuracy(&original_results, &static_results, "静态量化");
}

fn evaluate_accuracy(original_results: &HashMap<&str, String>, quantized_results: &HashMap<&str, String>, quant_type: &str) {
    let mut exact_matches = 0;
    let mut total_chars = 0;
    let mut matching_chars = 0;
    
    for (prompt, original) in original_results {
        if let Some(quantized) = quantized_results.get(prompt) {
            if original == quantized {
                exact_matches += 1;
            }
            
            // 计算字符级别的相似度
            let min_len = original.len().min(quantized.len());
            total_chars += min_len;
            
            for (o_char, q_char) in original.chars().zip(quantized.chars()) {
                if o_char == q_char {
                    matching_chars += 1;
                }
            }
        }
    }
    
    let exact_accuracy = (exact_matches as f64 / original_results.len() as f64) * 100.0;
    let char_accuracy = (matching_chars as f64 / total_chars as f64) * 100.0;
    
    println!("{}:", quant_type);
    println!("  完全匹配率: {:.2}% ({}/{})", exact_accuracy, exact_matches, original_results.len());
    println!("  字符匹配率: {:.2}%", char_accuracy);
}
