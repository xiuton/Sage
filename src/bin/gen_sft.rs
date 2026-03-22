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

fn get_unique_file_path(mut path: PathBuf) -> PathBuf {
    if !path.exists() {
        return path;
    }
    
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let parent = path.parent().unwrap_or(std::path::Path::new(""));
    
    for i in 1.. {
        let new_name = if extension.is_empty() {
            format!("{}_{}", stem, i)
        } else {
            format!("{}_{}.{}", stem, i, extension)
        };
        let mut new_path = parent.join(new_name);
        if !new_path.exists() {
            return new_path;
        }
    }
    
    path
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let out = arg_value(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            // 默认保存到data目录
            let mut path = PathBuf::from("data");
            std::fs::create_dir_all(&path).expect("create data directory");
            path.push("sft_demo.jsonl");
            path
        });

    let count: usize = arg_value(&args, "--count")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5000);

    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);

    let mut rng = StdRng::seed_from_u64(seed);
    let unique_out = get_unique_file_path(out);
    let file = File::create(&unique_out).expect("create out file");
    let mut w = BufWriter::new(file);

    // 扩展主题列表
    let topics = [
        "Rust", "Python", "JavaScript", "Go", "C++",
        "网络", "数据库", "算法", "数据结构", "人工智能",
        "数学", "物理", "化学", "英语", "中文",
        "写作", "演讲", "沟通", "旅行", "美食",
        "健康", "健身", "心理学", "历史", "哲学",
        "古诗词", "文学", "艺术", "音乐", "电影",
        "科技", "经济", "政治", "法律", "教育",
    ];

    // 学习资源类型
    let resources = [
        "书籍", "课程", "视频", "文档", "博客",
        "论坛", "社区", "项目", "教程", "实战",
    ];

    // 技术概念
    let concepts = [
        "机器学习", "深度学习", "神经网络", "自然语言处理",
        "计算机视觉", "强化学习", "大数据", "云计算",
        "区块链", "物联网", "人工智能", "自动化",
    ];

    // 编程语言特性
    let features = [
        "并发", "内存安全", "性能", "生态系统",
        "类型系统", "错误处理", "包管理", "跨平台",
        "编译速度", "学习曲线", "社区支持", "文档质量",
    ];

    // 项目类型
    let project_types = [
        "Web应用", "移动应用", "桌面应用", "后端服务",
        "数据分析", "机器学习", "游戏开发", "嵌入式系统",
        "自动化脚本", "工具开发", "API服务", "微服务",
    ];

    // 对话语气
    let tones = [
        "正式", "友好", "专业", "简洁",
        "详细", "轻松", "幽默", "严谨",
    ];

    for i in 0..count {
        let topic = topics[rng.gen_range(0..topics.len())];
        let resource = resources[rng.gen_range(0..resources.len())];
        let concept = concepts[rng.gen_range(0..concepts.len())];
        let feature = features[rng.gen_range(0..features.len())];
        let project_type = project_types[rng.gen_range(0..project_types.len())];
        let tone = tones[rng.gen_range(0..tones.len())];

        // 增加更多问答模式（从8种增加到20种）
        let kind = rng.gen_range(0..20);

        let (prompt, response) = match kind {
            0 => (
                format!("用一句话解释什么是{}。", topic),
                format!("{}是一个广泛的领域，涉及相关知识、技术和应用。它在现代{}和日常生活中都有重要应用。", topic, concept),
            ),
            1 => (
                format!("推荐几个学习{}的优质{}。", topic, resource),
                format!("学习{}的优质{}包括：官方文档、经典书籍、在线课程和实战项目。建议从基础概念开始，逐步深入实践。", topic, resource),
            ),
            2 => (
                format!("{}的主要{}有哪些？", topic, feature),
                format!("{}的主要{}包括：高效的性能、良好的生态系统、强大的类型系统和完善的错误处理机制。这些特性使其在{}开发中表现出色。", topic, feature, project_type),
            ),
            3 => (
                format!("如何入门{}开发？", topic),
                format!("入门{}开发的步骤：1) 学习基础语法和核心概念；2) 完成简单项目实践；3) 参与开源项目；4) 持续学习最新技术。建议采用{}的学习方式。", topic, tone),
            ),
            4 => (
                format!("{}和{}相比有什么优势？", topic, topics[rng.gen_range(0..topics.len())]),
                format!("{}相比其他技术的优势在于：更好的性能、更强的类型安全和更丰富的生态系统。它特别适合{}等场景的开发。", topic, project_type),
            ),
            5 => (
                format!("用50字解释什么是{}。", concept),
                format!("{}是一种{}技术，通过{}实现特定功能。它在现代{}和{}开发中扮演重要角色，能够{}。", concept, tone, feature, project_type, topic, feature),
            ),
            6 => {
                let a: i32 = rng.gen_range(10..200);
                let b: i32 = rng.gen_range(10..200);
                (
                    format!("计算 {} + {} 等于多少？", a, b),
                    format!("{} + {} = {}", a, b, a + b),
                )
            }
            7 => {
                let a: i32 = rng.gen_range(10..100);
                let b: i32 = rng.gen_range(2..10);
                (
                    format!("计算 {} × {} 等于多少？", a, b),
                    format!("{} × {} = {}", a, b, a * b),
                )
            }
            8 => (
                format!("如何在{}中实现{}功能？", topic, feature),
                format!("在{}中实现{}功能的步骤：1) 了解相关API；2) 设计实现方案；3) 编写代码；4) 测试和优化。建议参考官方文档和社区最佳实践。", topic, feature),
            ),
            9 => (
                format!("{}开发中常见的错误有哪些？", topic),
                format!("{}开发中常见的错误包括：内存管理问题、并发安全问题、错误处理不当等。避免这些错误的方法是：遵循最佳实践、使用静态分析工具和编写充分的测试。", topic),
            ),
            10 => (
                format!("推荐几个{}的开源项目。", topic),
                format!("推荐的{}开源项目包括：核心库、工具链、框架和应用示例。这些项目展示了{}在{}领域的应用，值得学习和参考。", topic, topic, project_type),
            ),
            11 => (
                format!("{}的未来发展趋势是什么？", topic),
                format!("{}的未来发展趋势包括：性能优化、生态扩展、新特性支持和更多应用场景。随着技术发展，{}将在{}和{}等领域发挥更大作用。", topic, topic, project_type, concept),
            ),
            12 => (
                format!("如何提高{}的性能？", topic),
                format!("提高{}性能的方法包括：优化算法、使用适当的数据结构、并行处理和内存管理优化。在{}开发中，{}是关键因素。", topic, project_type, feature),
            ),
            13 => (
                "用两句话总结学习编程的要点。".to_string(),
                "学习编程的核心是理解问题、设计算法和实现解决方案。持续实践和不断学习是提高编程能力的关键。".to_string(),
            ),
            14 => (
                format!("{}适合什么样的项目开发？", topic),
                format!("{}特别适合{}、{}等项目开发，因为它具有{}、{}等优势。在需要{}的场景中表现出色。", topic, project_type, project_types[rng.gen_range(0..project_types.len())], feature, features[rng.gen_range(0..features.len())], feature),
            ),
            15 => (
                format!("如何学习{}的{}特性？", topic, feature),
                format!("学习{}的{}特性需要：1) 理解基本概念；2) 查看官方文档；3) 完成练习项目；4) 参与社区讨论。建议采用{}的学习方法。", topic, feature, tone),
            ),
            16 => (
                format!("{}和{}的区别是什么？", concept, concepts[rng.gen_range(0..concepts.len())]),
                format!("{}和{}的主要区别在于：应用场景不同、技术实现不同和性能特点不同。{}更适合{}，而{}更适合{}。", concept, concepts[rng.gen_range(0..concepts.len())], concept, project_type, concepts[rng.gen_range(0..concepts.len())], project_types[rng.gen_range(0..project_types.len())]),
            ),
            17 => (
                format!("给初学者的{}学习建议。", topic),
                format!("给初学者的{}学习建议：1) 从基础开始，建立扎实的知识体系；2) 多做实践项目，积累经验；3) 学习优秀代码，提高编程风格；4) 参与社区交流，解决问题。保持{}的学习态度。", topic, tone),
            ),
            18 => (
                format!("{}在{}领域的应用案例。", topic, concept),
                format!("{}在{}领域的应用案例包括：智能推荐系统、数据分析平台、自动化工具等。这些应用展示了{}在解决实际问题中的价值和潜力。", topic, concept, topic),
            ),
            _ => (
                format!("如何评价{}的{}？", topic, feature),
                format!("{}的{}表现{}，在{}开发中具有重要价值。它的优势在于{}，但也存在{}等挑战。总体来说，是一个值得学习和使用的技术。", topic, feature, tone, project_type, feature, features[rng.gen_range(0..features.len())]),
            ),
        };

        // 增加更多对话风格（从3种增加到6种）
        let style = rng.gen_range(0..6);
        let messages = match style {
            0 => json!([
                {"role":"user","content":prompt},
                {"role":"assistant","content":response}
            ]),
            1 => json!([
                {"role":"user","content":format!("请{}回答：{}", tone, prompt)},
                {"role":"assistant","content":response}
            ]),
            2 => json!([
                {"role":"user","content":prompt},
                {"role":"assistant","content":format!("好的。\n{}", response)}
            ]),
            3 => json!([
                {"role":"user","content":format!("关于{}，{}", topic, prompt)},
                {"role":"assistant","content":response}
            ]),
            4 => json!([
                {"role":"user","content":format!("请教一个问题：{}", prompt)},
                {"role":"assistant","content":format!("很高兴为您解答。\n{}", response)}
            ]),
            _ => json!([
                {"role":"user","content":prompt},
                {"role":"assistant","content":format!("根据我的了解，{}", response)}
            ]),
        };

        let line = json!({"messages": messages, "id": i}).to_string();
        w.write_all(line.as_bytes()).expect("write");
        w.write_all(b"\n").expect("write newline");
    }

    w.flush().expect("flush");
    println!("Wrote {} records to {}", count, unique_out.display());
}
