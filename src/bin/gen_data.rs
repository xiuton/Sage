use sage::utils::common::get_unique_file_path;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    out: Option<String>,

    #[arg(short = 'c', long, default_value_t = 1000)]
    count: usize,

    #[arg(short = 's', long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = false)]
    web: bool,

    #[arg(long, default_value_t = false)]
    web_only: bool,

    #[arg(long, default_value_t = false)]
    multimodal: bool,

    #[arg(long)]
    image_dir: Option<String>,

    #[arg(long)]
    text_to_image_data: Option<String>,
}

// 真实问答数据集 (从 gen_web_sft.rs 迁移)
struct QAPair {
    question: String,
    answer: String,
    domain: String,
}

fn get_local_qa_pairs() -> Vec<QAPair> {
    vec![
        QAPair {
            question: "什么是机器学习？".to_string(),
            answer: "机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式和规律，而无需明确编程。核心概念包括监督学习、无监督学习和强化学习。".to_string(),
            domain: "人工智能".to_string(),
        },
        QAPair {
            question: "Rust语言的主要特点是什么？".to_string(),
            answer: "Rust语言的主要特点包括内存安全、零成本抽象、并发安全、高性能和跨平台支持。它通过所有权系统和借用检查器在编译时保证内存安全，无需垃圾回收。".to_string(),
            domain: "编程语言".to_string(),
        },
        QAPair {
            question: "如何提高网站性能？".to_string(),
            answer: "提高网站性能的方法包括：优化图片和资源文件、使用CDN、启用缓存、减少HTTP请求、优化CSS和JavaScript代码、使用异步加载和延迟加载技术。".to_string(),
            domain: "Web开发".to_string(),
        },
        QAPair {
            question: "数据库索引的作用是什么？".to_string(),
            answer: "数据库索引用于加速数据查询，通过创建数据的快速查找结构，减少查询时需要扫描的数据量。常见的索引类型包括B树索引、哈希索引和全文索引。".to_string(),
            domain: "数据库".to_string(),
        },
        QAPair {
            question: "什么是区块链技术？".to_string(),
            answer: "区块链是一种去中心化的分布式账本技术，通过密码学保证数据的安全性和完整性。它的特点包括不可篡改、去中心化、透明性和匿名性，广泛应用于加密货币和智能合约。".to_string(),
            domain: "区块链".to_string(),
        },
        QAPair {
            question: "如何学习编程？".to_string(),
            answer: "学习编程的有效方法包括：选择合适的编程语言、从基础概念开始学习、实践项目、参与开源社区、不断学习和练习。最重要的是保持耐心和持续的实践。".to_string(),
            domain: "编程学习".to_string(),
        },
        QAPair {
            question: "什么是云计算？".to_string(),
            answer: "云计算是通过互联网提供计算资源和服务的模式，包括计算能力、存储、数据库和网络等。主要服务模式包括IaaS、PaaS和SaaS，优点包括按需使用、弹性扩展和成本效益。".to_string(),
            domain: "云计算".to_string(),
        },
        QAPair {
            question: "如何进行有效的时间管理？".to_string(),
            answer: "有效的时间管理方法包括：制定明确的目标和计划、优先处理重要任务、避免拖延、合理分配时间、使用时间管理工具、学会拒绝和专注工作。保持工作与生活的平衡也很重要。".to_string(),
            domain: "个人发展".to_string(),
        },
        QAPair {
            question: "什么是人工智能？".to_string(),
            answer: "人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。它包括机器学习、深度学习、自然语言处理、计算机视觉等领域，应用于语音识别、图像识别、自动驾驶等多个领域。".to_string(),
            domain: "人工智能".to_string(),
        },
        QAPair {
            question: "如何保护网络安全？".to_string(),
            answer: "保护网络安全的措施包括：使用强密码、启用双因素认证、定期更新软件、使用防火墙、避免点击可疑链接、备份重要数据、使用虚拟专用网络(VPN)、教育用户识别网络钓鱼攻击。".to_string(),
            domain: "网络安全".to_string(),
        },
        QAPair {
            question: "什么是DevOps？".to_string(),
            answer: "DevOps是开发(Development)和运维(Operations)的结合，旨在缩短开发周期、提高部署频率、增强系统可靠性。核心实践包括持续集成、持续交付、自动化、监控和协作。".to_string(),
            domain: "软件工程".to_string(),
        },
        QAPair {
            question: "如何进行有效的沟通？".to_string(),
            answer: "有效的沟通方法包括：清晰表达、积极倾听、使用适当的语言和语气、注意非语言沟通、给予和接受反馈、尊重他人观点、选择合适的沟通渠道。良好的沟通有助于建立信任和理解。".to_string(),
            domain: "沟通技巧".to_string(),
        },
        QAPair {
            question: "什么是大数据？".to_string(),
            answer: "大数据指的是规模巨大、类型多样、处理速度快的数据集合。它的特点包括数据量大(Volume)、数据类型多样(Variety)、处理速度快(Velocity)和数据价值(Value)。大数据技术包括数据存储、数据处理和数据分析。".to_string(),
            domain: "大数据".to_string(),
        },
        QAPair {
            question: "如何保持健康的生活方式？".to_string(),
            answer: "保持健康生活方式的方法包括：均衡饮食、规律运动、充足睡眠、管理压力、避免吸烟和过量饮酒、定期体检、保持社交活动、培养积极心态。预防胜于治疗，健康是最重要的财富。".to_string(),
            domain: "健康生活".to_string(),
        },
        QAPair {
            question: "什么是算法？".to_string(),
            answer: "算法是解决问题或执行任务的一系列明确步骤。它具有输入、输出、确定性、有限性和有效性等特性。常见的算法包括排序算法、搜索算法、图算法等，算法的效率通常用时间复杂度和空间复杂度来衡量。".to_string(),
            domain: "算法".to_string(),
        },
        QAPair {
            question: "如何提高学习效率？".to_string(),
            answer: "提高学习效率的方法包括：制定学习计划、主动学习、使用记忆技巧、定期复习、实践应用、寻求反馈、保持专注、管理时间、保持健康生活方式。不同的学习方法适用于不同的人和学习内容。".to_string(),
            domain: "学习方法".to_string(),
        },
        QAPair {
            question: "什么是物联网？".to_string(),
            answer: "物联网(IoT)是指通过互联网连接的物理设备网络，这些设备能够收集和交换数据。它的应用包括智能家居、智能城市、工业物联网、健康监测等。物联网技术包括传感器、网络通信、数据处理和安全。".to_string(),
            domain: "物联网".to_string(),
        },
        QAPair {
            question: "如何进行有效的团队合作？".to_string(),
            answer: "有效的团队合作方法包括：明确团队目标和角色、建立信任和沟通、鼓励参与和贡献、解决冲突、给予反馈和认可、促进创新和学习、保持积极的团队文化。团队合作能够发挥集体智慧，提高工作效率。".to_string(),
            domain: "团队合作".to_string(),
        },
        QAPair {
            question: "什么是自然语言处理？".to_string(),
            answer: "自然语言处理(NLP)是人工智能的一个分支，旨在使计算机能够理解、处理和生成人类语言。它的应用包括机器翻译、语音识别、文本分类、情感分析、问答系统等。核心技术包括词法分析、句法分析和语义理解。".to_string(),
            domain: "自然语言处理".to_string(),
        },
        QAPair {
            question: "如何管理个人财务？".to_string(),
            answer: "管理个人财务的方法包括：制定预算、跟踪支出、储蓄和投资、避免债务、建立应急基金、定期审查财务状况、学习财务知识、规划退休。良好的财务管理有助于实现财务目标和财务自由。".to_string(),
            domain: "财务管理".to_string(),
        },
        QAPair {
            question: "什么是计算机视觉？".to_string(),
            answer: "计算机视觉是人工智能的一个分支，旨在使计算机能够理解和分析图像和视频内容。它的应用包括图像识别、目标检测、人脸识别、自动驾驶等。核心技术包括特征提取、分类、检测和分割。".to_string(),
            domain: "计算机视觉".to_string(),
        },
        QAPair {
            question: "如何提高创造力？".to_string(),
            answer: "提高创造力的方法包括：保持好奇心、尝试新事物、打破常规思维、寻求不同观点、保持开放心态、记录想法、练习创意技能、保持身心健康。创造力是解决问题和创新的重要能力。".to_string(),
            domain: "创造力".to_string(),
        },
        QAPair {
            question: "什么是容器技术？".to_string(),
            answer: "容器技术是一种操作系统级虚拟化技术，允许在隔离的环境中运行应用程序。它的优点包括轻量级、快速部署、一致性和可移植性。Docker是最流行的容器平台，Kubernetes用于容器编排和管理。".to_string(),
            domain: "容器技术".to_string(),
        },
        QAPair {
            question: "如何应对压力？".to_string(),
            answer: "应对压力的方法包括：识别压力源、保持积极心态、练习放松技巧、保持健康生活方式、寻求支持、时间管理、设定合理目标、学会说不。长期压力会影响身心健康，需要及时管理。".to_string(),
            domain: "心理健康".to_string(),
        },
        QAPair {
            question: "什么是边缘计算？".to_string(),
            answer: "边缘计算是一种分布式计算模型，将计算和存储资源部署在网络边缘，靠近数据源头。它的优点包括低延迟、高带宽、数据隐私和安全性。应用场景包括物联网、智能城市和自动驾驶。".to_string(),
            domain: "边缘计算".to_string(),
        },
    ]
}

fn generate_synthetic_qa(rng: &mut StdRng) -> (String, String, String) {
    let topics = ["Rust", "Python", "JavaScript", "Go", "C++", "算法", "人工智能", "数据库", "网络"];
    let resources = ["书籍", "课程", "文档", "教程"];
    let concepts = ["机器学习", "神经网络", "并发", "性能"];
    
    let topic = topics[rng.gen_range(0..topics.len())];
    let resource = resources[rng.gen_range(0..resources.len())];
    let concept = concepts[rng.gen_range(0..concepts.len())];

    let kind = rng.gen_range(0..5);
    match kind {
        0 => (
            format!("用一句话解释什么是{}。", topic),
            format!("{}是一个广泛的领域，涉及相关知识、技术和应用。它在现代{}和日常生活中都有重要应用。", topic, concept),
            topic.to_string()
        ),
        1 => (
            format!("推荐几个学习{}的优质{}。", topic, resource),
            format!("学习{}的优质{}包括：官方文档、经典书籍、在线课程和实战项目。", topic, resource),
            topic.to_string()
        ),
        2 => {
            let a: i32 = rng.gen_range(10..200);
            let b: i32 = rng.gen_range(10..200);
            (format!("计算 {} + {} 等于多少？", a, b), format!("{} + {} = {}", a, b, a + b), "数学".to_string())
        },
        3 => (
            format!("如何在{}中实现高效的{}？", topic, concept),
            format!("在{}中实现高效的{}需要深入理解其核心机制和最佳实践。", topic, concept),
            topic.to_string()
        ),
        _ => (
            format!("{}和{}相比有什么优势？", topic, concepts[rng.gen_range(0..concepts.len())]),
            format!("{}相比其他技术的优势在于更好的性能和更强的类型安全。", topic),
            topic.to_string()
        ),
    }
}

fn generate_dialog_style(
    rng: &mut StdRng,
    prompt: &str,
    response: &str,
    domain: &str,
) -> serde_json::Value {
    let style = rng.gen_range(0..5);
    match style {
        0 => json!([{"role":"user","content":prompt}, {"role":"assistant","content":response}]),
        1 => json!([{"role":"user","content":format!("关于{}，{}", domain, prompt)}, {"role":"assistant","content":response}]),
        2 => json!([{"role":"user","content":prompt}, {"role":"assistant","content":format!("好的。\n{}", response)}]),
        3 => json!([{"role":"user","content":format!("请教一个问题：{}", prompt)}, {"role":"assistant","content":format!("很高兴为您解答。\n{}", response)}]),
        _ => json!([{"role":"user","content":prompt}, {"role":"assistant","content":format!("根据我的了解，{}", response)}]),
    }
}

fn scan_image_directory(dir_path: &str) -> Vec<PathBuf> {
    let mut image_files = Vec::new();
    let supported_extensions = ["jpg", "jpeg", "png", "gif", "bmp", "webp"];

    fn walk_dir(dir: &PathBuf, extensions: &[&str], results: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_dir() {
                    walk_dir(&path, extensions, results);
                } else if let Some(ext) = path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if extensions.contains(&ext_lower.as_str()) {
                        results.push(path);
                    }
                }
            }
        }
    }

    let root_dir = PathBuf::from(dir_path);
    walk_dir(&root_dir, &supported_extensions, &mut image_files);
    image_files
}

fn generate_text_to_image_data(image_dir: &str, output_path: &PathBuf) -> usize {
    let image_files = scan_image_directory(image_dir);
    let count = image_files.len();

    if count == 0 {
        println!("No image files found in directory: {}", image_dir);
        return 0;
    }

    let unique_out = get_unique_file_path(output_path.clone());
    let file = File::create(&unique_out).expect("create file");
    let mut w = BufWriter::new(file);

    for image_path in &image_files {
        let data_obj = json!({
            "prompt": "",
            "image_path": image_path.to_string_lossy().replace("\\", "/")
        });

        let line = data_obj.to_string();
        w.write_all(line.as_bytes()).unwrap();
        w.write_all(b"\n").unwrap();
    }

    w.flush().unwrap();
    println!("Wrote {} image records to {}", count, unique_out.display());
    count
}

fn main() {
    let args = Args::parse();
    unsafe { std::env::set_var("CUBECL_AUTOTUNE_LEVEL", "minimal"); }

    if let Some(ref image_dir) = args.image_dir {
        let output_path = args.text_to_image_data.map(PathBuf::from).unwrap_or_else(|| {
            let mut path = PathBuf::from("data");
            std::fs::create_dir_all(&path).ok();
            path.push("text_to_image_pairs.jsonl");
            path
        });
        generate_text_to_image_data(image_dir, &output_path);
        return;
    }

    let out_path = args.out.map(PathBuf::from).unwrap_or_else(|| {
        let mut path = PathBuf::from("data");
        std::fs::create_dir_all(&path).ok();
        path.push("sft_data.jsonl");
        path
    });

    let mut rng = StdRng::seed_from_u64(args.seed);
    let unique_out = get_unique_file_path(out_path);
    let file = File::create(&unique_out).expect("create file");
    let mut w = BufWriter::new(file);

    let local_qa = get_local_qa_pairs();
    
    for i in 0..args.count {
        let (prompt, response, domain) = if args.web_only || (args.web && rng.gen_bool(0.5)) {
            let pair = &local_qa[rng.gen_range(0..local_qa.len())];
            (pair.question.clone(), pair.answer.clone(), pair.domain.clone())
        } else {
            generate_synthetic_qa(&mut rng)
        };

        let messages = generate_dialog_style(&mut rng, &prompt, &response, &domain);
        let mut data_obj = json!({"messages": messages, "id": i, "domain": domain});
        
        if args.multimodal {
            data_obj["image_path"] = json!("assets/sample_image.jpg");
        }
        
        let line = data_obj.to_string();
        w.write_all(line.as_bytes()).unwrap();
        w.write_all(b"\n").unwrap();
    }

    w.flush().unwrap();
    println!("Wrote {} records to {}", args.count, unique_out.display());
}
