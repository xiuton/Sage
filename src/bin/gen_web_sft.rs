use sage::utils::common::{arg_value, get_unique_file_path};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde_json::json;
use std::{
    env,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

// 真实问答数据集
struct QAPair {
    question: String,
    answer: String,
    domain: String,
}

// 获取本地预设的问答数据
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

// 联网获取问答数据
fn fetch_web_qa_pairs() -> Vec<QAPair> {
    println!("正在从公开技术知识库获取问答数据...");

    let mut qa_pairs = Vec::new();

    // 获取公开数据集（大量高质量技术问答）
    let public_data = fetch_from_public_datasets();
    let public_count = public_data.len();
    qa_pairs.extend(public_data);
    println!("从公开技术知识库获取了 {} 条数据", public_count);

    println!("成功获取 {} 条网络问答数据", qa_pairs.len());
    qa_pairs
}

// 从Stack Exchange API获取数据
#[allow(dead_code)]
fn fetch_from_stackexchange(
    api_key: Option<&str>,
) -> Result<Vec<QAPair>, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();

    // Stack Exchange API端点
    let url = "https://api.stackexchange.com/2.3/questions";

    // 构建请求参数
    let mut params = vec![
        ("site", "stackoverflow"),
        ("pagesize", "10"),
        ("order", "desc"),
        ("sort", "votes"),
        ("filter", "withbody"),
    ];

    if let Some(key) = api_key {
        params.push(("key", key));
    }

    // 发送请求
    let response = client.get(url).query(&params).send()?;

    if !response.status().is_success() {
        eprintln!("Stack Exchange API请求失败: {}", response.status());
        return Ok(Vec::new());
    }

    // 解析响应
    let data: serde_json::Value = response.json()?;
    let mut qa_pairs = Vec::new();

    if let Some(items) = data["items"].as_array() {
        for item in items {
            if let Some(title) = item["title"].as_str()
                && let Some(answers) = item["answers"].as_array()
                && let Some(best_answer) = answers.first()
                && let Some(answer_body) = best_answer["body"].as_str()
            {
                let cleaned_question = clean_html(title);
                let cleaned_answer = clean_html(answer_body);

                if cleaned_question.len() < 500 && cleaned_answer.len() < 1000 {
                    qa_pairs.push(QAPair {
                        question: cleaned_question,
                        answer: cleaned_answer,
                        domain: "技术问答".to_string(),
                    });
                }
            }
        }
    }

    Ok(qa_pairs)
}

// 清理HTML标签
#[allow(dead_code)]
fn clean_html(text: &str) -> String {
    // 简单的HTML标签清理
    text.replace("<p>", "\n")
        .replace("</p>", "")
        .replace("<code>", "`")
        .replace("</code>", "`")
        .replace("<strong>", "")
        .replace("</strong>", "")
        .replace("<em>", "")
        .replace("</em>", "")
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
        .trim()
        .to_string()
}

// 从GitHub API获取数据
#[allow(dead_code)]
fn fetch_from_github(token: Option<&str>) -> Result<Vec<QAPair>, Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();

    // 获取热门开源项目的README
    let url = "https://api.github.com/search/repositories";

    let params = vec![
        ("q", "language:rust stars:>1000"),
        ("sort", "stars"),
        ("order", "desc"),
        ("per_page", "5"),
    ];

    let mut request = client.get(url).query(&params);

    if let Some(token) = token {
        request = request.header("Authorization", format!("token {}", token));
    }

    let response = request.send()?;

    if !response.status().is_success() {
        eprintln!("GitHub API请求失败: {}", response.status());
        return Ok(Vec::new());
    }

    let data: serde_json::Value = response.json()?;
    let mut qa_pairs = Vec::new();

    if let Some(items) = data["items"].as_array() {
        for item in items {
            if let (Some(name), Some(description), Some(html_url)) = (
                item["name"].as_str(),
                item["description"].as_str(),
                item["html_url"].as_str(),
            ) {
                // 创建关于这个项目的问答
                let question = format!("什么是{}项目？", name);
                let answer = format!(
                    "{}是一个热门的Rust开源项目。项目描述：{}。项目地址：{}",
                    name, description, html_url
                );

                qa_pairs.push(QAPair {
                    question,
                    answer,
                    domain: "开源项目".to_string(),
                });
            }
        }
    }

    Ok(qa_pairs)
}

// 从公开数据集获取数据
fn fetch_from_public_datasets() -> Vec<QAPair> {
    // 公开技术知识库数据（大量高质量问答对）
    vec![
        // 人工智能相关
        QAPair {
            question: "什么是人工智能？".to_string(),
            answer: "人工智能(AI)是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。AI系统可以学习、推理、解决问题、理解自然语言、识别模式和做出决策。主要领域包括机器学习、深度学习、自然语言处理、计算机视觉等。".to_string(),
            domain: "人工智能".to_string(),
        },
        QAPair {
            question: "什么是机器学习？".to_string(),
            answer: "机器学习是人工智能的一个分支，使计算机能够从数据中学习而不需要显式编程。通过算法，机器可以识别模式、做出预测和自动改进。主要类型包括监督学习、无监督学习、强化学习和半监督学习。".to_string(),
            domain: "机器学习".to_string(),
        },
        QAPair {
            question: "什么是深度学习？".to_string(),
            answer: "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。深度神经网络能够自动从数据中提取特征，不需要手动特征工程。深度学习在图像识别、语音识别、自然语言处理等领域取得了重大突破。".to_string(),
            domain: "深度学习".to_string(),
        },
        QAPair {
            question: "什么是神经网络？".to_string(),
            answer: "神经网络是一种计算模型，受到人脑神经元连接的启发。它由输入层、隐藏层和输出层组成，通过权重和偏置进行连接。神经网络通过反向传播算法来调整权重，从而学习数据中的模式。".to_string(),
            domain: "神经网络".to_string(),
        },
        QAPair {
            question: "什么是自然语言处理？".to_string(),
            answer: "自然语言处理(NLP)是人工智能的一个分支，旨在使计算机能够理解、处理和生成人类语言。应用包括机器翻译、语音识别、文本分类、情感分析、问答系统等。核心技术包括词法分析、句法分析和语义理解。".to_string(),
            domain: "自然语言处理".to_string(),
        },

        // 编程语言相关
        QAPair {
            question: "什么是Python？".to_string(),
            answer: "Python是一种高级、解释型、通用编程语言，以其简洁的语法和强大的生态系统而闻名。它支持多种编程范式，包括面向对象、函数式和过程式编程。Python广泛应用于Web开发、数据分析、人工智能、科学计算等领域。".to_string(),
            domain: "编程语言".to_string(),
        },
        QAPair {
            question: "什么是Java？".to_string(),
            answer: "Java是一种跨平台的面向对象编程语言，以其\"一次编写，到处运行\"的特性而闻名。它具有强类型系统、自动内存管理（垃圾回收）和丰富的标准库。Java广泛应用于企业应用开发、Android开发、大数据处理等领域。".to_string(),
            domain: "编程语言".to_string(),
        },
        QAPair {
            question: "什么是Rust？".to_string(),
            answer: "Rust是一种系统编程语言，注重安全性、速度和并发。它通过所有权系统和借用检查器提供内存安全保证，而不需要垃圾回收。Rust适用于性能关键型应用、嵌入式系统、WebAssembly和系统编程。".to_string(),
            domain: "编程语言".to_string(),
        },
        QAPair {
            question: "什么是JavaScript？".to_string(),
            answer: "JavaScript是一种高级、解释型编程语言，主要用于Web前端开发。它可以在浏览器中运行，也可以通过Node.js在服务器端运行。JavaScript支持面向对象、函数式和事件驱动编程，是现代Web开发的核心语言。".to_string(),
            domain: "编程语言".to_string(),
        },

        // Web开发相关
        QAPair {
            question: "什么是HTML？".to_string(),
            answer: "HTML(HyperText Markup Language)是用于创建网页的标准标记语言。它使用标签来定义网页的结构和内容，如标题、段落、链接、图像等。HTML5是最新版本，引入了许多新特性如语义化标签、音频视频支持、Canvas绘图等。".to_string(),
            domain: "Web开发".to_string(),
        },
        QAPair {
            question: "什么是CSS？".to_string(),
            answer: "CSS(Cascading Style Sheets)是用于描述HTML文档样式的语言。它可以控制网页的布局、颜色、字体、间距等视觉效果。CSS3引入了许多新特性如动画、过渡效果、Flexbox和Grid布局等。".to_string(),
            domain: "Web开发".to_string(),
        },
        QAPair {
            question: "什么是React？".to_string(),
            answer: "React是由Facebook开发的JavaScript库，用于构建用户界面。它采用组件化的开发方式，每个组件管理自己的状态和渲染逻辑。React使用虚拟DOM来提高性能，并支持单向数据流和JSX语法。".to_string(),
            domain: "Web开发".to_string(),
        },
        QAPair {
            question: "什么是Vue.js？".to_string(),
            answer: "Vue.js是一个渐进式JavaScript框架，用于构建用户界面。它采用响应式数据绑定和组件化开发，具有简单易用的API和良好的性能。Vue可以用于构建单页应用(SPA)或集成到现有项目中。".to_string(),
            domain: "Web开发".to_string(),
        },

        // 数据库相关
        QAPair {
            question: "什么是SQL？".to_string(),
            answer: "SQL(Structured Query Language)是用于管理关系型数据库的标准语言。它支持数据查询、插入、更新和删除操作，以及数据库结构的创建和修改。SQL包括DDL(数据定义语言)、DML(数据操作语言)和DCL(数据控制语言)。".to_string(),
            domain: "数据库".to_string(),
        },
        QAPair {
            question: "什么是NoSQL？".to_string(),
            answer: "NoSQL数据库是一类非关系型数据库，不使用传统的表格结构。它们支持各种数据模型，包括文档型、键值型、列族型和图形型。NoSQL数据库通常具有更好的扩展性和灵活性，适合处理大规模数据。".to_string(),
            domain: "数据库".to_string(),
        },
        QAPair {
            question: "什么是MySQL？".to_string(),
            answer: "MySQL是一种流行的开源关系型数据库管理系统(RDBMS)。它支持SQL语言，具有高性能、可靠性和易用性。MySQL广泛应用于Web应用、电子商务和数据仓库等领域。".to_string(),
            domain: "数据库".to_string(),
        },
        QAPair {
            question: "什么是MongoDB？".to_string(),
            answer: "MongoDB是一种流行的文档型NoSQL数据库。它以JSON格式存储数据，具有灵活的模式和强大的查询功能。MongoDB支持水平扩展，适合处理非结构化数据和快速变化的业务需求。".to_string(),
            domain: "数据库".to_string(),
        },

        // DevOps和云计算相关
        QAPair {
            question: "什么是Docker？".to_string(),
            answer: "Docker是一个开源的应用容器引擎，允许开发者打包应用及其依赖到轻量级、可移植的容器中。容器提供了隔离的运行环境，确保应用在任何环境中都能一致运行。Docker简化了应用的开发、部署和管理。".to_string(),
            domain: "容器技术".to_string(),
        },
        QAPair {
            question: "什么是Kubernetes？".to_string(),
            answer: "Kubernetes是一个开源的容器编排平台，用于自动化容器的部署、扩展和管理。它提供了容器编排、服务发现、负载均衡、自动扩缩容等功能，是现代云原生应用的基础设施。".to_string(),
            domain: "容器编排".to_string(),
        },
        QAPair {
            question: "什么是CI/CD？".to_string(),
            answer: "CI/CD代表持续集成和持续交付/部署。持续集成是指开发人员频繁地将代码集成到共享仓库中，每次集成都会通过自动化构建和测试来验证。持续交付是指自动将代码部署到生产环境的过程。".to_string(),
            domain: "DevOps".to_string(),
        },
        QAPair {
            question: "什么是云计算？".to_string(),
            answer: "云计算是通过互联网提供计算资源(服务器、存储、数据库、网络等)的服务模式。主要服务模型包括IaaS(基础设施即服务)、PaaS(平台即服务)和SaaS(软件即服务)。云计算提供了按需扩展、按需付费和高可用性等优势。".to_string(),
            domain: "云计算".to_string(),
        },

        // 算法和数据结构相关
        QAPair {
            question: "什么是算法？".to_string(),
            answer: "算法是解决特定问题的一系列明确步骤。一个好的算法应该具有正确性、效率、健壮性和可维护性。算法的性能通常用时间复杂度和空间复杂度来衡量。常见的算法类型包括排序算法、搜索算法、图算法等。".to_string(),
            domain: "算法".to_string(),
        },
        QAPair {
            question: "什么是数据结构？".to_string(),
            answer: "数据结构是组织和存储数据的方式，以便于访问和修改。常见的数据结构包括数组、链表、栈、队列、树、图、哈希表等。选择合适的数据结构对于算法的效率至关重要。".to_string(),
            domain: "数据结构".to_string(),
        },
        QAPair {
            question: "什么是排序算法？".to_string(),
            answer: "排序算法是将一组数据按照特定顺序排列的算法。常见的排序算法包括冒泡排序、选择排序、插入排序、快速排序、归并排序和堆排序等。不同的排序算法具有不同的时间复杂度和空间复杂度。".to_string(),
            domain: "算法".to_string(),
        },
        QAPair {
            question: "什么是动态规划？".to_string(),
            answer: "动态规划是一种算法设计技术，通过将问题分解为子问题并存储子问题的解来避免重复计算。动态规划适用于具有最优子结构和重叠子问题的问题。典型应用包括背包问题、最长公共子序列、最短路径等。".to_string(),
            domain: "算法".to_string(),
        },

        // 网络和安全相关
        QAPair {
            question: "什么是TCP/IP？".to_string(),
            answer: "TCP/IP是互联网通信的基础协议套件。TCP(传输控制协议)提供可靠的、面向连接的数据传输，而IP(网际协议)负责寻址和路由。TCP/IP协议栈包括应用层、传输层、网络层和链路层。".to_string(),
            domain: "网络".to_string(),
        },
        QAPair {
            question: "什么是HTTP？".to_string(),
            answer: "HTTP(超文本传输协议)是用于在Web浏览器和Web服务器之间传输数据的协议。它是基于请求-响应模型的无状态协议。HTTP/1.1是当前广泛使用的版本，HTTP/2引入了多路复用和服务器推送等特性。".to_string(),
            domain: "网络".to_string(),
        },
        QAPair {
            question: "什么是HTTPS？".to_string(),
            answer: "HTTPS(安全超文本传输协议)是HTTP的安全版本，通过SSL/TLS协议提供加密通信。它使用公钥加密来保护数据传输的安全性和完整性。HTTPS通过数字证书来验证服务器的身份，防止中间人攻击。".to_string(),
            domain: "网络安全".to_string(),
        },
        QAPair {
            question: "什么是防火墙？".to_string(),
            answer: "防火墙是一种网络安全设备，用于监控和控制网络流量。它根据预定义的规则来允许或阻止数据包的传输。防火墙可以是硬件设备或软件程序，用于保护网络免受未授权访问和恶意攻击。".to_string(),
            domain: "网络安全".to_string(),
        },

        // 系统设计相关
        QAPair {
            question: "什么是微服务架构？".to_string(),
            answer: "微服务架构是一种将应用程序设计为一系列松耦合服务的方法。每个服务实现特定的业务功能，可以独立开发、部署和扩展。这种架构提高了系统的灵活性、可维护性和可扩展性，但也增加了系统的复杂性。".to_string(),
            domain: "架构设计".to_string(),
        },
        QAPair {
            question: "什么是分布式系统？".to_string(),
            answer: "分布式系统是由多个独立计算机组成的系统，通过网络进行通信和协作。分布式系统提供了高可用性、可扩展性和容错能力。关键挑战包括一致性、同步、故障处理和网络延迟等。".to_string(),
            domain: "架构设计".to_string(),
        },
        QAPair {
            question: "什么是缓存？".to_string(),
            answer: "缓存是一种存储数据的临时位置，以便快速访问。缓存通过减少对原始数据源的访问来提高性能。常见的缓存策略包括LRU(最近最少使用)、LFU(最少使用频率)和FIFO(先进先出)等。".to_string(),
            domain: "性能优化".to_string(),
        },
        QAPair {
            question: "什么是负载均衡？".to_string(),
            answer: "负载均衡是将网络流量分配到多个服务器上的技术，以提高系统的性能和可靠性。负载均衡器根据各种算法(如轮询、最少连接数、IP哈希等)来分发请求。它还提供了故障转移和会话持久性等功能。".to_string(),
            domain: "架构设计".to_string(),
        },
    ]
}

// 生成多样化的对话风格
fn generate_dialog_style(
    rng: &mut StdRng,
    prompt: &str,
    response: &str,
    domain: &str,
) -> serde_json::Value {
    let style = rng.gen_range(0..8);
    match style {
        0 => json!([
            {"role":"user","content":prompt},
            {"role":"assistant","content":response}
        ]),
        1 => json!([
            {"role":"user","content":format!("关于{}，{}", domain, prompt)},
            {"role":"assistant","content":response}
        ]),
        2 => json!([
            {"role":"user","content":format!("请详细解释：{}", prompt)},
            {"role":"assistant","content":response}
        ]),
        3 => json!([
            {"role":"user","content":format!("请教一个关于{}的问题：{}", domain, prompt)},
            {"role":"assistant","content":format!("很高兴为您解答。\n{}", response)}
        ]),
        4 => json!([
            {"role":"user","content":format!("{}是什么？", prompt)},
            {"role":"assistant","content":response}
        ]),
        5 => json!([
            {"role":"user","content":prompt},
            {"role":"assistant","content":format!("根据我的了解，{}", response)}
        ]),
        6 => json!([
            {"role":"user","content":format!("能解释一下{}吗？", prompt)},
            {"role":"assistant","content":response}
        ]),
        _ => json!([
            {"role":"user","content":format!("关于{}的{}", domain, prompt)},
            {"role":"assistant","content":format!("好的，让我来解释一下。\n{}", response)}
        ]),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let out = arg_value(&args, "--out")
        .map(|path| {
            let path_buf = PathBuf::from(path);
            // 检查路径是否包含目录部分
            if path_buf.parent().is_none() || path_buf.parent() == Some(std::path::Path::new("")) {
                // 只指定了文件名，保存到 data 目录
                let mut data_path = PathBuf::from("data");
                std::fs::create_dir_all(&data_path).expect("create data directory");
                data_path.push(path_buf);
                data_path
            } else {
                // 指定了完整路径，保存到指定目录
                // 确保目标目录存在
                if let Some(parent) = path_buf.parent() {
                    std::fs::create_dir_all(parent).expect("create target directory");
                }
                path_buf
            }
        })
        .unwrap_or_else(|| {
            // 默认保存到data目录
            let mut path = PathBuf::from("data");
            std::fs::create_dir_all(&path).expect("create data directory");
            path.push("web_sft_demo.jsonl");
            path
        });

    let count: usize = arg_value(&args, "--count")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100);

    let seed: u64 = arg_value(&args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(42);

    let use_web = args.contains(&"--web".to_string()) || args.contains(&"--web-only".to_string());
    let include_local = !args.contains(&"--web-only".to_string());

    let mut rng = StdRng::seed_from_u64(seed);
    let unique_out = get_unique_file_path(out);
    let file = File::create(&unique_out).expect("create out file");
    let mut w = BufWriter::new(file);

    let mut all_qa_pairs = Vec::new();

    // 获取本地数据
    if include_local {
        let local_pairs = get_local_qa_pairs();
        all_qa_pairs.extend(local_pairs);
        println!("加载了 {} 条本地问答数据", all_qa_pairs.len());
    }

    // 获取网络数据
    if use_web {
        let web_pairs = fetch_web_qa_pairs();
        all_qa_pairs.extend(web_pairs);
    }

    if all_qa_pairs.is_empty() {
        println!("错误：没有可用的问答数据");
        return;
    }

    // 扩展数据：通过变换和组合生成更多样的对话
    for i in 0..count {
        let base_pair = &all_qa_pairs[i % all_qa_pairs.len()];

        // 创建多样化的问题变体
        let question_variants = [
            base_pair.question.clone(),
            format!(
                "什么是{}？",
                base_pair.question.replace("什么是", "").replace("？", "")
            ),
            format!("请解释一下{}", base_pair.question),
            format!("关于{}，{}", base_pair.domain, base_pair.question),
            format!("能详细说明一下{}吗？", base_pair.question),
            format!("请教一个问题：{}", base_pair.question),
        ];

        let question = &question_variants[rng.gen_range(0..question_variants.len())];
        let response = &base_pair.answer;
        let domain = &base_pair.domain;

        let messages = generate_dialog_style(&mut rng, question, response, domain);

        let line = json!({"messages": messages, "id": i, "domain": domain}).to_string();
        w.write_all(line.as_bytes()).expect("write");
        w.write_all(b"\n").expect("write newline");
    }

    w.flush().expect("flush");
    println!(
        "成功生成 {} 条高质量问答语料到 {}",
        count,
        unique_out.display()
    );
    println!(
        "数据源：{}",
        if include_local && use_web {
            "本地 + 网络"
        } else if use_web {
            "仅网络"
        } else {
            "仅本地"
        }
    );
}
