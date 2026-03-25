use clap::Parser;
use burn::backend::{ndarray::{NdArray, NdArrayDevice}, wgpu::Wgpu};
use burn::prelude::Backend;
use sage::{
    generation::{GenerateOptions, generate},
    lazy_load::LazyModel,
    performance::run_benchmark,
    tokenizer::Tokenizer,
    TrainingConfig,
};
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "./models")]
    model_dir: String,
    
    #[arg(short, long, default_value = "cpu")]
    backend: String,
    
    #[arg(short, long, default_value = "10")]
    iterations: usize,
    
    #[arg(short, long, default_value = "你好，请介绍一下你自己")]
    prompt: String,
}

fn main() {
    let args = Args::parse();
    
    println!("=== Sage 推理性能基准测试 ===");
    println!("模型目录: {}", args.model_dir);
    println!("后端: {}", args.backend);
    println!("迭代次数: {}", args.iterations);
    println!("提示词: {}", args.prompt);
    println!();
    
    // 加载训练配置
    let config_path = Path::new(&args.model_dir).join("config.json");
    let config_content = fs::read_to_string(&config_path).expect("无法读取配置文件");
    let config: TrainingConfig = serde_json::from_str(&config_content).expect("解析配置失败");
    
    // 加载分词器
    let tokenizer_path = Path::new(&args.model_dir).join("tokenizer.json");
    let tokenizer = Tokenizer::load(tokenizer_path.to_str().expect("无效的分词器路径")).expect("加载分词器失败");
    
    // 模型路径
    let model_path = format!("{}/model.mpk", args.model_dir);
    
    let result = if args.backend == "gpu" {
        println!("使用GPU后端进行基准测试...");
        unsafe {
            std::env::set_var("WGPU_POWER_PREFERENCE", "HighPerformance");
        }
        let device = <Wgpu as Backend>::Device::default();
        let lazy_model: LazyModel<Wgpu> = LazyModel::new(config.model.clone(), model_path);
        let model = lazy_model.get_model(&device);
        
        run_benchmark("GPU推理", args.iterations, || {
            let model_guard = model.lock().unwrap();
            let options = GenerateOptions {
                max_new_tokens: 50,
                temperature: 0.8,
                top_k: 10,
                top_p: 0.9,
                repetition_penalty: 1.1,
                punctuation_penalty: 1.3,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                seed: None,
                context_len: config.model.max_seq_len,
                stop_on_user: true,
                stop_sequences: Vec::new(),
            };
            
            let response = generate(&*model_guard, &tokenizer, &args.prompt, &options, &device);
            let prompt_tokens = args.prompt.len() / 4;
            let completion_tokens = response.len() / 4;
            
            (prompt_tokens, completion_tokens)
        })
    } else {
        println!("使用CPU后端进行基准测试...");
        let device = NdArrayDevice::Cpu;
        let lazy_model: LazyModel<NdArray> = LazyModel::new(config.model.clone(), model_path);
        let model = lazy_model.get_model(&device);
        
        run_benchmark("CPU推理", args.iterations, || {
            let model_guard = model.lock().unwrap();
            let options = GenerateOptions {
                max_new_tokens: 50,
                temperature: 0.8,
                top_k: 10,
                top_p: 0.9,
                repetition_penalty: 1.1,
                punctuation_penalty: 1.3,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                seed: None,
                context_len: config.model.max_seq_len,
                stop_on_user: true,
                stop_sequences: Vec::new(),
            };
            
            let response = generate(&*model_guard, &tokenizer, &args.prompt, &options, &device);
            let prompt_tokens = args.prompt.len() / 4;
            let completion_tokens = response.len() / 4;
            
            (prompt_tokens, completion_tokens)
        })
    };
    
    println!("=== 基准测试结果 ===");
    println!("测试名称: {}", result.name);
    println!("迭代次数: {}", result.iterations);
    println!("平均推理时间: {:.2} ms", result.avg_time_ms);
    println!("最小推理时间: {:.2} ms", result.min_time_ms);
    println!("最大推理时间: {:.2} ms", result.max_time_ms);
    println!("平均速度: {:.2} tokens/s", result.tokens_per_second);
    println!();
}
