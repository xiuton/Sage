use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::config::Config;
use burn::module::Module;
use clap::Parser;
use sage::{
    generation::{GenerateOptions, generate},
    tokenizer::Tokenizer,
    training::TrainingConfig,
};
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 要用于推理的提示词。
    #[arg(long)]
    prompt: Option<String>,

    /// 生成新 token 的数量。
    #[arg(short = 'n', long, default_value_t = 50)]
    num_tokens: usize,

    /// 控制生成随机性的温度。
    #[arg(short = 't', long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-K 采样中的 K 值。
    #[arg(short = 'k', long, default_value_t = 10)]
    top_k: usize,

    #[arg(short = 'p', long, default_value_t = 0.9)]
    top_p: f32,

    #[arg(short = 'r', long, default_value_t = 1.1)]
    repetition_penalty: f32,

    #[arg(long, default_value_t = 1.3)]
    punctuation_penalty: f32,

    #[arg(short = 's', long)]
    seed: Option<u64>,

    /// 模型权重和配置所在的目录。
    #[arg(long, default_value = "./tmp/sage_model_formal")]
    model_dir: String,

    #[arg(long, default_value_t = false)]
    use_best: bool,

    #[arg(long, default_value_t = 0)]
    context_len: usize,

    /// 进入交互模式，可以连续输入提示词。
    #[arg(short, long, default_value_t = false)]
    interactive: bool,

    #[arg(long, default_value_t = false)]
    chat: bool,
}

fn format_chat_prefix(user_text: &str) -> String {
    let mut out = String::new();
    out.push('\u{0002}');
    out.push_str("用户：");
    out.push_str(user_text);
    out.push('\n');
    out.push_str("助手：");
    out
}

fn extract_assistant_reply(full: &str) -> String {
    if let Some(idx) = full.rfind("助手：") {
        return full[idx + "助手：".len()..].trim().to_string();
    }
    full.trim().to_string()
}

fn main() {
    let args = Args::parse();
    let device = NdArrayDevice::Cpu;

    println!("正在加载模型...");
    let config_path = format!("{}/config.json", args.model_dir);
    let tokenizer_path = format!("{}/tokenizer.json", args.model_dir);
    let model_path = if args.use_best {
        let best = format!("{}/best_model.mpk", args.model_dir);
        if std::path::Path::new(&best).exists() {
            best
        } else {
            format!("{}/model.mpk", args.model_dir)
        }
    } else {
        format!("{}/model.mpk", args.model_dir)
    };

    let training_config: TrainingConfig = TrainingConfig::load(&config_path).unwrap();
    let model_config = training_config.model;
    let requested_context_len = if args.context_len == 0 {
        model_config.max_seq_len
    } else {
        args.context_len
    };
    let context_len = requested_context_len.min(model_config.max_seq_len);
    if requested_context_len > model_config.max_seq_len {
        eprintln!(
            "context_len {} 超过模型 max_seq_len {}，已自动截断。",
            requested_context_len, model_config.max_seq_len
        );
    }

    let tokenizer = Tokenizer::load(&tokenizer_path).unwrap();
    let model = model_config
        .init::<NdArray>(&device)
        .load_file(&model_path, &burn::record::CompactRecorder::new(), &device)
        .unwrap();
    println!("模型加载完成。\n");

    if args.interactive {
        println!("--- 进入交互模式 --- (输入 'exit' 退出)");
        let mut history = String::new();
        loop {
            print!(">> ");
            io::stdout().flush().unwrap();
            let mut user_prompt = String::new();
            io::stdin().read_line(&mut user_prompt).unwrap();
            let user_prompt = user_prompt.trim();

            if user_prompt == "exit" {
                break;
            }

            let input_text = if args.chat {
                history.push_str("用户：");
                history.push_str(user_prompt);
                history.push('\n');
                history.push_str("助手：");
                history.clone()
            } else {
                user_prompt.to_string()
            };

            let gen_options = GenerateOptions {
                max_new_tokens: args.num_tokens,
                temperature: args.temperature,
                top_k: args.top_k,
                top_p: args.top_p,
                repetition_penalty: args.repetition_penalty,
                punctuation_penalty: args.punctuation_penalty,
                seed: args.seed,
                context_len,
            };
            let generated = generate(&model, &tokenizer, &input_text, &gen_options, &device);
            if args.chat {
                let reply = extract_assistant_reply(&generated);
                println!("助手: {}\n", reply);
                history.push_str(&reply);
                history.push('\n');
            } else {
                println!("生成结果: \"{}\"\n", generated);
            }
        }
    } else if let Some(prompt) = args.prompt {
        println!("--- 模型生成 ---");
        let input_text = if args.chat {
            format_chat_prefix(&prompt)
        } else {
            prompt.clone()
        };
        let gen_options = GenerateOptions {
            max_new_tokens: args.num_tokens,
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            repetition_penalty: args.repetition_penalty,
            punctuation_penalty: args.punctuation_penalty,
            seed: args.seed,
            context_len,
        };
        let generated = generate(&model, &tokenizer, &input_text, &gen_options, &device);
        if args.chat {
            println!("用户: \"{}\"", prompt);
            println!("助手: \"{}\"\n", extract_assistant_reply(&generated));
        } else {
            println!("提示词: \"{}\"", prompt);
            println!("生成结果: \"{}\"\n", generated);
        }
    } else {
        println!("错误：请提供一个提示词 (使用 --prompt) 或进入交互模式 (使用 --interactive)。");
    }
}
