#![recursion_limit = "1024"]

use burn::backend::{ndarray::{NdArray}, wgpu::Wgpu};
use burn::module::Module;
use burn::prelude::Backend;
use clap::Parser;
use sage::inference::{GenerateOptions, GenerationState, ModelType, generate, generate_multimodal};
use sage::core::Tokenizer;
use sage::TrainingConfig;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use std::path::Path;
use image;
use rustyline::{error::ReadlineError, Editor, Config as RustylineConfig};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    prompt: Option<String>,

    #[arg(short = 'n', long, default_value_t = 500)]
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

    #[arg(long, default_value = "./models/sage_model_formal")]
    model_dir: String,

    #[arg(long, default_value_t = false)]
    use_best: bool,

    #[arg(long, default_value_t = 0)]
    context_len: usize,

    #[arg(short, long, default_value_t = false)]
    interactive: bool,

    #[arg(long, default_value_t = false)]
    terminal: bool,

    #[arg(long, default_value_t = false)]
    chat: bool,

    #[arg(long, default_value_t = true)]
    stop_on_user: bool,

    #[arg(long, use_value_delimiter = true)]
    stop_sequence: Vec<String>,

    #[arg(long, default_value_t = false)]
    stream: bool,

    #[arg(long, default_value_t = 50)]
    stream_speed: u64,

    #[arg(long, default_value = "cpu")]
    backend: String,
    
    /// 启用多模态推理
    #[arg(long, default_value_t = false)]
    multimodal: bool,
    
    /// 图像文件路径（用于多模态推理）
    #[arg(long)]
    image_path: Option<String>,

    #[arg(long, default_value = "Sage Assistant")]
    assistant_name: String,

    #[arg(long, default_value = "User")]
    user_name: String,
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
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: self.seed,
            context_len,
            stop_on_user: self.stop_on_user,
            stop_sequences: self.stop_sequence.clone(),
            use_kv_cache: true,
            streaming: self.stream || self.terminal || self.interactive,
        }
    }
}

fn format_chat_prefix(user_text: &str, user_name: &str, assistant_name: &str) -> String {
    let estimated_len = 20 + user_text.len() + user_name.len() + assistant_name.len();
    let mut out = String::with_capacity(estimated_len);
    out.push_str(&format!("<s>\n{}:", user_name));
    out.push_str(user_text);
    out.push_str(&format!("\n{}:", assistant_name));
    out
}

fn extract_assistant_reply(full: &str, assistant_name: &str) -> String {
    let assistant_tag = format!("{}:", assistant_name);
    let Some(idx) = full.rfind(&assistant_tag) else {
        // 回退到原有的 <assistant> 标签逻辑
        let Some(idx) = full.rfind("<assistant>") else {
            return full.trim().to_string();
        };
        let start = idx + "<assistant>".len();
        let Some(end) = full[start..].find("</assistant>") else {
            return full[start..].trim().to_string();
        };
        return full[start..start + end].trim().to_string();
    };
    
    let start = idx + assistant_tag.len();
    full[start..].trim().to_string()
}

/// 加载并预处理图像
fn load_and_preprocess_image<B: Backend>(image_path: &str, device: &B::Device) -> burn::tensor::Tensor<B, 4> {
    // 加载图像
    let img = image::open(image_path).expect("无法加载图像");
    
    // 确保是 RGB 格式
    let img_rgb = img.to_rgb8();
    
    // 调整大小为 224x224
    let img_resized = image::imageops::resize(&img_rgb, 224, 224, image::imageops::FilterType::Lanczos3);
    
    // 转换为张量
    let mut data = Vec::with_capacity(3 * 224 * 224);
    for y in 0..224 {
        for x in 0..224 {
            let pixel = img_resized.get_pixel(x, y);
            data.push(pixel[0] as f32 / 255.0);
            data.push(pixel[1] as f32 / 255.0);
            data.push(pixel[2] as f32 / 255.0);
        }
    }
    
    // 创建张量 [batch_size, channels, height, width]
    let tensor = burn::tensor::Tensor::<B, 4>::from_data(
        burn::tensor::TensorData::new(data, [1, 3, 224, 224]),
        device
    );
    
    tensor
}

fn print_welcome_message(assistant_name: &str) {
    println!("========================================");
    println!("Welcome to {} Terminal", assistant_name);
    println!("========================================");
    println!("Type your message and press Enter to send");
    println!("Type '\\help' for available commands");
    println!("Type '\\exit' or '\\quit' to exit");
    println!("========================================");
    println!();
}

fn print_help_message() {
    println!("Available commands:");
    println!("  \\help          - Show this help message");
    println!("  \\exit, \\quit    - Exit the terminal");
    println!("  \\clear         - Clear the screen");
    println!("  \\reset         - Reset the conversation history");
    println!("  \\history       - Show conversation history");
    println!("  \\temperature <value> - Set temperature (0.0-2.0)");
    println!();
}

fn run_inference<B: Backend>(args: &Args) {
    let device = B::Device::default();

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
        std::process::exit(1);
    }

    if !std::path::Path::new(&tokenizer_path).exists() {
        eprintln!("错误：分词器文件未找到：{}", tokenizer_path);
        std::process::exit(1);
    }

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("错误：模型权重文件未找到：{}", model_path);
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

    let tokenizer = Tokenizer::load(&tokenizer_path).unwrap();
    let model = model_config
        .init::<B>(&device)
        .load_file(&model_path, &burn::record::CompactRecorder::new(), &device)
        .unwrap();
    println!("模型加载完成。\n");

    let image_tensor = if args.multimodal && args.image_path.is_some() {
        Some(load_and_preprocess_image::<B>(args.image_path.as_ref().unwrap(), &device))
    } else {
        None
    };

    if args.terminal {
        // 高级终端模式
        let config = RustylineConfig::builder()
            .history_ignore_space(true)
            .completion_type(rustyline::CompletionType::List)
            .build();
        let mut rl = Editor::<()>::with_config(config).expect("Failed to create rustyline editor");

        let history_path = Path::new(&args.model_dir).join("sage_history.txt");
        if history_path.exists() {
            let _ = rl.load_history(history_path.to_str().unwrap());
        }

        print_welcome_message(&args.assistant_name);
        let mut history = String::new();
        let mut current_args = args.clone();

        loop {
            let prompt = format!("{}: ", args.user_name);
            let readline = rl.readline(&prompt);

            match readline {
                Ok(line) => {
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    rl.add_history_entry(line);

                    if line.starts_with('\\') {
                        let command = line.trim_start_matches('\\');
                        match command {
                            "help" => print_help_message(),
                            "exit" | "quit" => break,
                            "clear" => {
                                print!("{esc}[2J{esc}[1;1H", esc = 27 as char);
                                print_welcome_message(&args.assistant_name);
                            }
                            "reset" => {
                                history.clear();
                                println!("对话历史已重置。");
                            }
                            "history" => println!("对话历史:\n{}", history),
                            cmd if cmd.starts_with("temperature ") => {
                                if let Some(v) = cmd.strip_prefix("temperature ").and_then(|s| s.parse::<f32>().ok()) {
                                    current_args.temperature = v;
                                    println!("温度已设置为: {}", v);
                                }
                            }
                            _ => println!("未知命令: {}", command),
                        }
                        continue;
                    }

                    let input_text = format_chat_prefix(line, &args.user_name, &args.assistant_name);
                    let gen_options = current_args.gen_options(context_len);

                    print!("{}: ", args.assistant_name);
                    io::stdout().flush().unwrap();
                    
                    let model_type = if current_args.multimodal && image_tensor.is_some() {
                        ModelType::Multimodal(&model, image_tensor.as_ref().unwrap())
                    } else {
                        ModelType::Normal(&model)
                    };
                    
                    let mut state = GenerationState::new(model_type, &tokenizer, &input_text, &gen_options, &device);
                    let mut generated_text = String::new();
                    let token_interval = Duration::from_millis(1000 / current_args.stream_speed.max(1));
                    let mut last_token_time = Instant::now();
                    
                    while !state.is_stopped() {
                        if let Some(token_str) = state.next_token() {
                            let elapsed = last_token_time.elapsed();
                            if elapsed < token_interval { std::thread::sleep(token_interval - elapsed); }
                            print!("{}", token_str);
                            io::stdout().flush().unwrap();
                            generated_text.push_str(&token_str);
                            last_token_time = Instant::now();
                        }
                    }
                    println!("\n");
                    
                    history.push_str(&format!("{}: {}\n", args.user_name, line));
                    let reply = extract_assistant_reply(&generated_text, &args.assistant_name);
                    history.push_str(&format!("{}: {}\n", args.assistant_name, reply));
                }
                Err(ReadlineError::Interrupted) => println!("^C"),
                Err(ReadlineError::Eof) => break,
                Err(err) => { println!("错误: {:?}", err); break; }
            }
        }
        let _ = rl.save_history(history_path.to_str().unwrap());
    } else if args.interactive {
        // 简单交互模式
        println!("--- 进入交互模式 --- (输入 'exit' 退出)");
        let mut history = String::new();
        loop {
            print!(">> ");
            io::stdout().flush().unwrap();
            let mut user_prompt = String::new();
            io::stdin().read_line(&mut user_prompt).unwrap();
            let user_prompt = user_prompt.trim();

            if user_prompt == "exit" { break; }

            let input_text = if args.chat {
                history.push_str("\n<user>");
                history.push_str(user_prompt);
                history.push_str("</user>\n<assistant>");
                history.clone()
            } else {
                user_prompt.to_string()
            };

            let gen_options = args.gen_options(context_len);
            
            if args.stream {
                println!("助手: ");
                io::stdout().flush().unwrap();
                
                let model_type = if args.multimodal && image_tensor.is_some() {
                    ModelType::Multimodal(&model, image_tensor.as_ref().unwrap())
                } else {
                    ModelType::Normal(&model)
                };
                
                let mut state = GenerationState::new(model_type, &tokenizer, &input_text, &gen_options, &device);
                let mut generated_text = String::new();
                let token_interval = Duration::from_millis(1000 / args.stream_speed.max(1));
                let mut last_token_time = Instant::now();
                
                while !state.is_stopped() {
                    if let Some(token_str) = state.next_token() {
                        let elapsed = last_token_time.elapsed();
                        if elapsed < token_interval { std::thread::sleep(token_interval - elapsed); }
                        print!("{}", token_str);
                        io::stdout().flush().unwrap();
                        generated_text.push_str(&token_str);
                        last_token_time = Instant::now();
                    }
                }
                println!("\n");
                
                if args.chat {
                    let reply = extract_assistant_reply(&generated_text, &args.assistant_name);
                    history.push_str(&reply);
                    history.push('\n');
                }
            } else {
                let generated = if args.multimodal && image_tensor.is_some() {
                    generate_multimodal(&model, &tokenizer, &input_text, image_tensor.as_ref().unwrap(), &gen_options, &device)
                } else {
                    generate(&model, &tokenizer, &input_text, &gen_options, &device)
                };
                if args.chat {
                    let reply = extract_assistant_reply(&generated, &args.assistant_name);
                    println!("助手: {}\n", reply);
                    history.push_str(&reply);
                    history.push('\n');
                } else {
                    println!("生成结果: \"{}\"\n", generated);
                }
            }
        }
    } else if let Some(ref prompt) = args.prompt {
        // 单次生成模式
        let input_text = if args.chat {
            format_chat_prefix(prompt, &args.user_name, &args.assistant_name)
        } else {
            prompt.clone()
        };
        let gen_options = args.gen_options(context_len);
        
        if args.stream {
            print!("生成结果: ");
            io::stdout().flush().unwrap();
            
            let model_type = if args.multimodal && image_tensor.is_some() {
                ModelType::Multimodal(&model, image_tensor.as_ref().unwrap())
            } else {
                ModelType::Normal(&model)
            };
            
            let mut state = GenerationState::new(model_type, &tokenizer, &input_text, &gen_options, &device);
            let mut last_token_time = Instant::now();
            let token_interval = Duration::from_millis(1000 / args.stream_speed.max(1));
            
            while !state.is_stopped() {
                if let Some(token_char) = state.next_token() {
                    let elapsed = last_token_time.elapsed();
                    if elapsed < token_interval { std::thread::sleep(token_interval - elapsed); }
                    print!("{}", token_char);
                    io::stdout().flush().unwrap();
                    last_token_time = Instant::now();
                }
            }
            println!("\n");
        } else {
            let generated = if args.multimodal && image_tensor.is_some() {
                generate_multimodal(&model, &tokenizer, &input_text, image_tensor.as_ref().unwrap(), &gen_options, &device)
            } else {
                generate(&model, &tokenizer, &input_text, &gen_options, &device)
            };
            if args.chat {
                println!("助手: \"{}\"\n", extract_assistant_reply(&generated, &args.assistant_name));
            } else {
                println!("生成结果: \"{}\"\n", generated);
            }
        }
    } else {
        println!("错误：请提供提示词 (--prompt)、交互模式 (--interactive) 或终端模式 (--terminal)。");
    }
}

fn main() {
    let args = Args::parse();
    unsafe { std::env::set_var("CUBECL_AUTOTUNE_LEVEL", "minimal"); }

    match args.backend.as_str() {
        "cpu" => run_inference::<NdArray>(&args),
        "gpu" => run_inference::<Wgpu>(&args),
        _ => {
            eprintln!("错误：不支持的后端 '{}'，请使用 'cpu' 或 'gpu'", args.backend);
            std::process::exit(1);
        }
    }
}
