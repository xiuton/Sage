use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::config::Config; // 导入 Config Trait
use burn::module::Module;
use clap::Parser;
use sage::{
    generation::generate,
    model::ModelConfig,
    tokenizer::Tokenizer,
    training::TrainingConfig, // 引入 TrainingConfig
};
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 要用于推理的提示词。
    #[arg(short, long)]
    prompt: Option<String>,

    /// 生成新 token 的数量。
    #[arg(short, long, default_value_t = 50)]
    num_tokens: usize,

    /// 控制生成随机性的温度。
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-K 采样中的 K 值。
    #[arg(short, long, default_value_t = 10)]
    top_k: usize,

    /// 模型权重和配置所在的目录。
    #[arg(long, default_value = "./tmp/sage_model_formal")]
    model_dir: String,

    /// 进入交互模式，可以连续输入提示词。
    #[arg(short, long, default_value_t = false)]
    interactive: bool,
}

fn main() {
    let args = Args::parse();
    let device = NdArrayDevice::Cpu;

    println!("正在加载模型...");
    let config_path = format!("{}/config.json", args.model_dir);
    let tokenizer_path = format!("{}/tokenizer.json", args.model_dir);
    let model_path = format!("{}/model.mpk", args.model_dir);

    let training_config: TrainingConfig = TrainingConfig::load(&config_path).unwrap();
    let model_config = training_config.model;

    let tokenizer = Tokenizer::load(&tokenizer_path).unwrap();
    let model = model_config.init::<NdArray>(&device).load_file(&model_path, &burn::record::CompactRecorder::new(), &device).unwrap();
    println!("模型加载完成。\n");

    if args.interactive {
        println!("--- 进入交互模式 --- (输入 'exit' 退出)");
        loop {
            print!(">> ");
            io::stdout().flush().unwrap();
            let mut user_prompt = String::new();
            io::stdin().read_line(&mut user_prompt).unwrap();
            let user_prompt = user_prompt.trim();

            if user_prompt == "exit" {
                break;
            }

            let generated = generate(
                &model,
                &tokenizer,
                user_prompt,
                args.num_tokens,
                args.temperature,
                args.top_k,
                &device,
            );
            println!("生成结果: \"{}\"\n", generated);
        }
    } else if let Some(prompt) = args.prompt {
        println!("--- 模型生成 ---");
        let generated = generate(
            &model,
            &tokenizer,
            &prompt,
            args.num_tokens,
            args.temperature,
            args.top_k,
            &device,
        );
        println!("提示词: \"{}\"", prompt);
        println!("生成结果: \"{}\"\n", generated);
    } else {
        println!("错误：请提供一个提示词 (使用 --prompt) 或进入交互模式 (使用 --interactive)。");
    }
}
