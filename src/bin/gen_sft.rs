use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let out = arg_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sft_demo_5000.jsonl"));

    let count: usize = arg_value(&args, "--count")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5000);

    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);

    let mut rng = StdRng::seed_from_u64(seed);
    let file = File::create(&out).expect("create out file");
    let mut w = BufWriter::new(file);

    let topics = [
        "Rust",
        "网络",
        "数据库",
        "算法",
        "数学",
        "英语",
        "写作",
        "旅行",
        "健康",
        "历史",
        "古诗词",
        "常识",
    ];

    for i in 0..count {
        let topic = topics[rng.gen_range(0..topics.len())];
        let kind = rng.gen_range(0..8);

        let (prompt, response) = match kind {
            0 => (
                format!("用一句话解释什么是{}。", topic),
                format!("{}是一类用于理解、解决或表达相关问题的知识与方法集合。", topic),
            ),
            1 => (
                "给出 3 条学习 Rust 的建议。".to_string(),
                "1) 先掌握所有权/借用/生命周期\n2) 多写小项目并用 clippy 修正\n3) 读标准库源码学 API 设计".to_string(),
            ),
            2 => (
                "把下面这句话改写得更礼貌：你写得太烂了。".to_string(),
                "这段内容还有提升空间，建议你再润色一下表达并补充论据。".to_string(),
            ),
            3 => (
                "生成一个简单的待办清单 JSON 示例。".to_string(),
                "{\n  \"todos\": [\n    {\"id\": 1, \"text\": \"学习 Rust\", \"done\": false},\n    {\"id\": 2, \"text\": \"跑通训练脚本\", \"done\": false}\n  ]\n}".to_string(),
            ),
            4 => (
                "用 50 字解释什么是哈希表。".to_string(),
                "哈希表用哈希函数把键映射到数组索引，实现接近 O(1) 的插入、查询与删除，但需处理冲突与扩容。".to_string(),
            ),
            5 => (
                "给出一个中文对话示例：用户问候，助手回应。".to_string(),
                "用户：你好！\n助手：你好！我能帮你做点什么？".to_string(),
            ),
            6 => {
                let a: i32 = rng.gen_range(10..200);
                let b: i32 = rng.gen_range(10..200);
                (
                    format!("计算 {} + {} 等于多少？", a, b),
                    format!("{} + {} = {}", a, b, a + b),
                )
            }
            _ => (
                "用两句话总结一段文字时要注意什么？".to_string(),
                "抓住主旨与关键事实，避免细枝末节。用更少的词表达同样的信息，并保持逻辑顺序清晰。".to_string(),
            ),
        };

        let style = rng.gen_range(0..3);
        let messages = match style {
            0 => json!([
                {"role":"user","content":prompt},
                {"role":"assistant","content":response}
            ]),
            1 => json!([
                {"role":"user","content":format!("请认真回答：{}", prompt)},
                {"role":"assistant","content":response}
            ]),
            _ => json!([
                {"role":"user","content":prompt},
                {"role":"assistant","content":format!("好的。\n{}", response)}
            ]),
        };

        let line = json!({"messages": messages, "id": i}).to_string();
        w.write_all(line.as_bytes()).expect("write");
        w.write_all(b"\n").expect("write newline");
    }

    w.flush().expect("flush");
    println!("Wrote {} records to {}", count, out.display());
}
