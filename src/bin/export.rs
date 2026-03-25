use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Module;
use clap::Parser;
use sage::{
    export::{export_model, ExportFormat},
    tokenizer::Tokenizer,
    TrainingConfig,
};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "export", about = "Export Sage model to different formats")]
struct Args {
    #[arg(long, help = "Path to model directory")]
    model_dir: String,
    
    #[arg(long, help = "Output file path")]
    output: String,
    
    #[arg(long, help = "Export format (onnx, gguf)")]
    format: String,
    
    #[arg(long, help = "Use GPU for export (experimental)")]
    gpu: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // 加载模型配置
    let config_path = format!("{}/config.json", args.model_dir);
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: TrainingConfig = serde_json::from_str(&config_str)?;
    
    // 加载分词器
    let tokenizer_path = format!("{}/tokenizer.json", args.model_dir);
    let tokenizer = Tokenizer::load(&tokenizer_path)?;
    
    // 加载模型
    let model_path = format!("{}/model.mpk", args.model_dir);
    let best_model_path = format!("{}/best_model.mpk", args.model_dir);
    let final_model_path = if Path::new(&best_model_path).exists() {
        best_model_path
    } else {
        model_path
    };
    
    // 选择后端
    let format = match args.format.to_lowercase().as_str() {
        "onnx" => ExportFormat::ONNX,
        "gguf" => ExportFormat::GGUF,
        _ => {
            eprintln!("Error: Unsupported format. Use 'onnx' or 'gguf'");
            std::process::exit(1);
        }
    };
    
    if args.gpu {
        eprintln!("GPU export not fully implemented yet, using CPU instead");
    }
    
    // 使用CPU后端加载模型
    let device = NdArrayDevice::Cpu;
    
    // 导出模型
    println!("Exporting model to {} format...", args.format);
    
    // 直接加载模型
    let model = config.model
        .init::<NdArray>(&device)
        .load_file(&final_model_path, &burn::record::CompactRecorder::new(), &device)
        .expect("Failed to load model");
    
    export_model(&model, &config.model, &tokenizer, &args.output, format)?;
    
    println!("Model exported successfully to: {}", args.output);
    
    Ok(())
}
