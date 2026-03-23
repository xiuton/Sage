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
    #[arg(long)]
    prompt: Option<String>,

    #[arg(short = 'n', long, default_value_t = 50)]
    num_tokens: usize,

    #[arg(short = 't', long, default_value_t = 0.8)]
    temperature: f32,

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

    #[arg(long, default_value = "./tmp/sage_model_formal")]
    model_dir: String,

    #[arg(long, default_value_t = false)]
    use_best: bool,

    #[arg(long, default_value_t = 0)]
    context_len: usize,

    #[arg(short, long, default_value_t = false)]
    interactive: bool,

    #[arg(long, default_value_t = false)]
    chat: bool,

    #[arg(long, default_value_t = true)]
    stop_on_user: bool,

    #[arg(long, use_value_delimiter = true)]
    stop_sequence: Vec<String>,
}

impl Args {
    fn gen_options(&self, context_len: usize) -> GenerateOptions {
        GenerateOptions {
            max_new_tokens: self.num_tokens,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            punctuation_penalty: self.punctuation_penalty,
            seed: self.seed,
            context_len,
            stop_on_user: self.stop_on_user,
            stop_sequences: self.stop_sequence.clone(),
        }
    }
}

fn format_chat_prefix(user_text: &str) -> String {
    let estimated_len = 10 + user_text.len();
    let mut out = String::with_capacity(estimated_len);
    out.push('\u{0002}');
    out.push_str("<s>\n<user>");
    out.push_str(user_text);
    out.push_str("</user>\n<assistant>");
    out
}

fn extract_assistant_reply(full: &str) -> String {
    let Some(idx) = full.rfind("<assistant>") else {
        return full.trim().to_string();
    };
    let start = idx + "<assistant>".len();
    let Some(end) = full[start..].find("</assistant>") else {
        return full[start..].trim().to_string();
    };
    full[start..start + end].trim().to_string()
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

    if !std::path::Path::new(&config_path).exists() {
        eprintln!("错误：模型配置文件未找到：{}", config_path);
        eprintln!("请确认指定的 --model-dir 有效，并且已完成训练。");
        std::process::exit(1);
    }

    if !std::path::Path::new(&tokenizer_path).exists() {
        eprintln!("错误：分词器文件未找到：{}", tokenizer_path);
        eprintln!("请确认指定的 --model-dir 有效，并且已完成训练。");
        std::process::exit(1);
    }

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("错误：模型权重文件未找到：{}", model_path);
        eprintln!(
            "可选路径为 best_model.mpk 或 model.mpk（如果使用 --use-best 但 first 失败会回退）。"
        );
        eprintln!("请确认指定的 --model-dir 有效，并且已完成训练。");
        std::process::exit(1);
    }

    let training_config: TrainingConfig =
        TrainingConfig::load(&config_path).expect("读取 config.json 失败");
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
                history.push_str("\n<user>");
                history.push_str(user_prompt);
                history.push_str("</user>");
                history.push('\n');
                history.push_str("<assistant>");
                history.clone()
            } else {
                user_prompt.to_string()
            };

            let gen_options = args.gen_options(context_len);
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
    } else if let Some(ref prompt) = args.prompt {
        println!("--- 模型生成 ---");
        let input_text = if args.chat {
            format_chat_prefix(&prompt)
        } else {
            prompt.clone()
        };
        let gen_options = args.gen_options(context_len);
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
