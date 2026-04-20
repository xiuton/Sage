use sage::core::Tokenizer;
use std::fs;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "create_tokenizer", about = "Create a tokenizer from sample text")]
struct Args {
    #[arg(long, default_value = "./models/sample.txt", help = "Path to sample text file")]
    sample_file: String,
    
    #[arg(long, default_value = "./models/tokenizer.json", help = "Path to save tokenizer")]
    output: String,
    
    #[arg(long, default_value = "false", help = "Use BPE tokenization instead of character-level")]
    use_bpe: bool,
    
    #[arg(long, default_value = "5000", help = "Vocabulary size for BPE tokenization")]
    vocab_size: usize,
}

fn main() {
    let args = Args::parse();
    
    // 读取样本文本
    let sample_text = fs::read_to_string(&args.sample_file).expect("Failed to read sample text file");
    
    // 创建分词器
    let tokenizer = if args.use_bpe {
        Tokenizer::new_bpe(&sample_text, args.vocab_size)
    } else {
        Tokenizer::new(&sample_text)
    };
    
    // 保存分词器
    tokenizer.save(&args.output).expect("Failed to save tokenizer");
    
    println!("Tokenizer created and saved successfully!");
    println!("Vocab size: {}", tokenizer.vocab_size);
    println!("Pad ID: {}", tokenizer.pad_id);
    println!("Unk ID: {}", tokenizer.unk_id);
    println!("Bos ID: {}", tokenizer.bos_id);
    println!("Eos ID: {}", tokenizer.eos_id);
    println!("Tokenizer type: {}", if args.use_bpe { "BPE" } else { "Character-level" });
}
