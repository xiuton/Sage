use sage::core::Tokenizer;
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

    /// 纯文本语料输出路径（用于 BPE 训练）
    #[arg(long, default_value = "./data/corpus.txt")]
    corpus_out: String,

    /// 是否在生成语料后自动训练 BPE tokenizer
    #[arg(long, default_value_t = false)]
    train_bpe: bool,

    /// BPE 词汇表大小
    #[arg(long, default_value_t = 32000)]
    bpe_vocab_size: usize,

    /// BPE tokenizer 输出路径
    #[arg(long, default_value = "./models/tokenizer_bpe.json")]
    bpe_output: String,
}

struct QAPair {
    question: String,
    answer: String,
    domain: String,
}

fn get_local_qa_pairs() -> Vec<QAPair> {
    let mut pairs = Vec::new();

    // ========== 人工智能与机器学习 ==========
    pairs.push(QAPair {
        question: "什么是机器学习？".to_string(),
        answer: "机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式和规律，而无需明确编程。核心概念包括监督学习（使用标注数据训练）、无监督学习（发现数据中的隐藏结构）和强化学习（通过与环境交互学习最优策略）。常见的算法包括线性回归、决策树、支持向量机、随机森林和神经网络。".to_string(),
        domain: "人工智能".to_string(),
    });
    pairs.push(QAPair {
        question: "深度学习与传统机器学习的区别是什么？".to_string(),
        answer: "深度学习是机器学习的一个子领域，使用多层神经网络自动学习数据的层次化特征表示。与传统机器学习的关键区别在于：深度学习自动从原始数据中提取特征，而传统ML依赖手工特征工程；深度学习需要大量数据和计算资源；深度学习在图像、语音、文本等非结构化数据上表现更优。典型的深度学习架构包括CNN、RNN、Transformer和GAN。".to_string(),
        domain: "人工智能".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是Transformer架构？".to_string(),
        answer: "Transformer是一种基于自注意力机制的神经网络架构，由Vaswani等人在2017年提出。它完全摒弃了循环和卷积结构，仅依赖注意力机制来捕获序列中的长距离依赖关系。核心组件包括多头自注意力（Multi-Head Self-Attention）和前馈神经网络，通过残差连接和层归一化进行优化。Transformer已成为NLP领域的主流架构，并衍生出BERT、GPT、T5等著名模型。".to_string(),
        domain: "人工智能".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是强化学习？".to_string(),
        answer: "强化学习是机器学习的一个分支，主要研究智能体如何在环境中通过试错学习最优策略。智能体通过执行动作获得奖励或惩罚信号，目标是最大化累积奖励。关键概念包括状态、动作、策略、奖励、价值函数和Q函数。经典算法包括Q-Learning、SARSA、深度Q网络（DQN）、策略梯度方法和PPO。应用领域包括游戏AI、机器人控制、自动驾驶和推荐系统。".to_string(),
        domain: "人工智能".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是迁移学习？".to_string(),
        answer: "迁移学习是一种机器学习技术，将一个任务上学到的知识应用到另一个相关任务上。这种方法特别适用于目标任务数据量较少的情况。常见的迁移学习策略包括：特征提取（使用预训练模型提取特征）、微调（在预训练模型基础上继续训练）、领域适应（使模型适应新数据分布）。迁移学习在计算机视觉和自然语言处理中广泛应用，例如使用预训练的ImageNet模型或BERT模型。".to_string(),
        domain: "人工智能".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是模型蒸馏？".to_string(),
        answer: "模型蒸馏是一种模型压缩技术，通过训练一个较小的学生模型来模仿一个较大的教师模型的行为。学生模型学习教师模型的输出分布（软标签）而非硬标签，从而获得更好的泛化能力。蒸馏过程通常使用温度参数控制软标签的平滑程度。这种方法可以在保持较高性能的同时大幅减少模型参数量和推理时间，适用于将大型语言模型部署到资源受限的设备上。".to_string(),
        domain: "人工智能".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是生成对抗网络（GAN）？".to_string(),
        answer: "生成对抗网络（GAN）由Ian Goodfellow在2014年提出，由生成器和判别器两个网络组成。生成器试图生成逼真的假数据，判别器则区分真假数据，两者通过对抗训练共同提升。生成器从随机噪声中生成数据，判别器评估数据的真实性。这种竞争机制使得生成器能生成高度逼真的图像、音频和文本。GAN的变体包括DCGAN、StyleGAN、CycleGAN和BigGAN。".to_string(),
        domain: "人工智能".to_string(),
    });

    // ========== 编程语言与软件开发 ==========
    pairs.push(QAPair {
        question: "Rust语言的主要特点是什么？".to_string(),
        answer: "Rust语言的主要特点包括：内存安全（通过所有权系统和借用检查器在编译时保证）、零成本抽象（高级抽象无运行时开销）、并发安全（防止数据竞争）、高性能（媲美C/C++）、跨平台支持、丰富的类型系统和模式匹配。Rust的所有权系统是其最独特的特性：每个值有唯一的所有者，通过引用（借用）访问值而不转移所有权，生命周期标注确保引用始终有效。".to_string(),
        domain: "编程语言".to_string(),
    });
    pairs.push(QAPair {
        question: "Python中的GIL是什么？".to_string(),
        answer: "GIL（全局解释器锁）是Python解释器中的一个互斥锁，确保同一时刻只有一个线程执行Python字节码。这意味着多线程Python程序在CPU密集型任务上无法充分利用多核处理器。GIL的存在简化了CPython的内存管理和C扩展开发，但也限制了并行性能。绕过GIL的方法包括：使用多进程替代多线程、使用异步编程、使用C扩展（如NumPy）、或使用Jython和IronPython等无GIL的实现。".to_string(),
        domain: "编程语言".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是函数式编程？".to_string(),
        answer: "函数式编程是一种编程范式，将计算视为数学函数的求值过程。核心原则包括：纯函数（无副作用，相同输入始终产生相同输出）、不可变性（数据一旦创建不可修改）、高阶函数（函数可作为参数和返回值）、递归替代循环、惰性求值。函数式编程的优势包括代码更易推理、并行化友好、减少bug。代表性语言有Haskell、Clojure、Erlang，主流语言如JavaScript、Python、Scala也支持函数式特性。".to_string(),
        domain: "编程语言".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是微服务架构？".to_string(),
        answer: "微服务架构是一种将应用程序构建为一系列小型、独立部署的服务的设计方法。每个服务运行在自己的进程中，通过轻量级通信机制（如HTTP/REST或消息队列）进行交互。主要优势包括：独立部署和扩展、技术栈灵活、故障隔离、团队自治。挑战包括：服务发现、分布式事务、数据一致性、监控和调试。常见的微服务技术栈包括Docker、Kubernetes、Service Mesh、API Gateway等。".to_string(),
        domain: "软件工程".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是设计模式？".to_string(),
        answer: "设计模式是软件开发中常见问题的通用可重用解决方案。GoF（四人帮）将设计模式分为三类：创建型模式（如单例、工厂、建造者）、结构型模式（如适配器、装饰器、代理）和行为型模式（如观察者、策略、命令）。设计模式提供了一套经过验证的解决方案词汇表，帮助开发者更有效地沟通和解决设计问题。但应避免过度使用，保持代码简洁才是首要原则。".to_string(),
        domain: "软件工程".to_string(),
    });

    // ========== 自然语言处理 ==========
    pairs.push(QAPair {
        question: "什么是词嵌入（Word Embedding）？".to_string(),
        answer: "词嵌入是将词汇映射到低维稠密向量空间的技术，使语义相似的词在向量空间中距离较近。相比传统的独热编码，词嵌入能捕获词语间的语义和语法关系。经典方法包括Word2Vec（CBOW和Skip-gram）、GloVe（全局词频统计）和FastText（考虑子词信息）。预训练词嵌入可以捕获丰富的语义信息，如经典的'国王 - 男人 + 女人 ≈ 女王'关系。现代语言模型使用上下文相关的动态词嵌入。".to_string(),
        domain: "自然语言处理".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是注意力机制？".to_string(),
        answer: "注意力机制是一种让模型动态聚焦于输入中重要部分的技术。它计算查询（Query）与键（Key）之间的相似度得分，用这些得分加权求和值（Value）向量。注意力机制极大地提升了序列到序列模型的性能，特别是在长序列任务中。主要变体包括：加性注意力（Bahdanau Attention）、乘法注意力（Luong Attention）、自注意力（Self-Attention）、多头注意力（Multi-Head Attention）。注意力机制是实现Transformer模型的核心组件。".to_string(),
        domain: "自然语言处理".to_string(),
    });
    pairs.push(QAPair {
        question: "BERT和GPT的主要区别是什么？".to_string(),
        answer: "BERT和GPT是两种主要的预训练语言模型架构，核心区别体现在：BERT使用Transformer编码器，采用双向上下文建模，通过掩码语言模型（MLM）和下句预测（NSP）进行预训练。GPT使用Transformer解码器，采用单向（从左到右）自回归语言建模。BERT更适合自然语言理解任务（分类、NER、QA），GPT更适合文本生成任务。BERT使用[MASK]标记进行预训练导致预训练-微调差距，GPT的自回归方式天然适合生成。两者的融合催生了编码器-解码器模型如T5和BART。".to_string(),
        domain: "自然语言处理".to_string(),
    });

    // ========== 数据库 ==========
    pairs.push(QAPair {
        question: "关系型数据库和非关系型数据库有什么区别？".to_string(),
        answer: "关系型数据库（如MySQL、PostgreSQL）基于关系模型，使用表结构存储数据，支持ACID事务、SQL查询和复杂关联。非关系型数据库（NoSQL）分为多种类型：文档数据库（MongoDB）、键值存储（Redis）、列族数据库（Cassandra）、图数据库（Neo4j）。关系型数据库适合数据一致性要求高、关系复杂的场景；NoSQL适合高并发、大数据量、灵活schema的场景。现代应用常采用混合架构结合两者优势。".to_string(),
        domain: "数据库".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是ACID特性？".to_string(),
        answer: "ACID是数据库事务的四个基本特性：原子性（Atomicity）确保事务中的所有操作要么全部成功，要么全部失败回滚；一致性（Consistency）保证事务前后数据完整性约束不被破坏；隔离性（Isolation）使并发事务的执行结果与串行执行一致；持久性（Durability）确保已提交事务的修改永久保存在数据库中。ACID特性是关系型数据库可靠性的基础保障，但严格的ACID保障往往带来性能开销。".to_string(),
        domain: "数据库".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是索引？有哪些常见的索引类型？".to_string(),
        answer: "数据库索引是一种用于加速数据查询的数据结构，通过创建排序后的键值到数据位置的映射，大幅减少查询时需要扫描的数据量。常见的索引类型包括：B+树索引（最常用，支持范围查询和排序）、哈希索引（精确匹配快但无法范围查询）、全文索引（用于文本搜索）、空间索引（R树，用于地理数据）、位图索引（适合低基数列）。创建索引能加速查询但会降低写入性能，需要根据实际查询模式合理设计索引策略。".to_string(),
        domain: "数据库".to_string(),
    });

    // ========== 云计算与DevOps ==========
    pairs.push(QAPair {
        question: "什么是云计算？有哪些服务模式？".to_string(),
        answer: "云计算是通过互联网提供按需计算资源的模式。三种主要服务模式：基础设施即服务（IaaS）提供虚拟机、存储和网络等基础资源；平台即服务（PaaS）提供应用开发和部署平台；软件即服务（SaaS）提供完整的应用程序。部署模式包括公有云、私有云、混合云和多云。主要云服务商有AWS、Azure、Google Cloud和阿里云。云计算的优势包括弹性伸缩、按需付费、降低运维成本和全球化部署能力。".to_string(),
        domain: "云计算".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是Kubernetes？".to_string(),
        answer: "Kubernetes（K8s）是一个开源的容器编排平台，用于自动化容器的部署、扩展和管理。它提供服务发现和负载均衡、存储编排、自动部署和回滚、资源管理、健康检查和自愈、密钥和配置管理等功能。核心概念包括Pod（最小的部署单元）、Service（网络抽象）、Deployment（声明式更新）、Namespace（资源隔离）等。Kubernetes已成为云原生应用的事实标准编排平台。".to_string(),
        domain: "云计算".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是CI/CD？".to_string(),
        answer: "CI/CD（持续集成/持续交付）是现代软件开发实践：持续集成（CI）是开发者频繁将代码合并到主干，每次合并都通过自动化构建和测试验证；持续交付（CD）确保代码始终处于可部署状态；持续部署则进一步自动化部署过程。CI/CD管道的典型步骤包括：代码检查、构建、测试、安全扫描、打包、部署。常用工具有Jenkins、GitLab CI、GitHub Actions、CircleCI和ArgoCD。".to_string(),
        domain: "DevOps".to_string(),
    });

    // ========== 网络安全 ==========
    pairs.push(QAPair {
        question: "什么是OWASP Top 10？".to_string(),
        answer: "OWASP Top 10是由开放Web应用安全项目（OWASP）发布的最关键的Web应用安全风险列表。2021年版包括：1）访问控制失效；2）加密失效；3）注入（SQL注入、XSS等）；4）不安全设计；5）安全配置错误；6）易受攻击和过时的组件；7）认证和识别失效；8）软件和数据完整性失效；9）安全日志和监控不足；10）服务端请求伪造（SSRF）。该列表每三到四年更新一次，是Web安全领域的权威参考。".to_string(),
        domain: "网络安全".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是零信任安全模型？".to_string(),
        answer: "零信任是一种安全模型，基于'永不信任，始终验证'的原则。与传统的边界安全模型不同，零信任假设网络已被攻破，因此不信任任何内部或外部请求。核心原则包括：持续验证所有访问请求、最小权限策略、微隔离（Micro-segmentation）、始终加密传输、实时监控和分析。Google的BeyondCorp是零信任的著名实践。零信任架构通常结合身份认证、设备管理、网络分段和行为分析等技术实现。".to_string(),
        domain: "网络安全".to_string(),
    });

    // ========== 区块链与Web3 ==========
    pairs.push(QAPair {
        question: "什么是区块链？".to_string(),
        answer: "区块链是一种去中心化、分布式的数字账本技术。数据以区块形式组织，每个区块包含一批交易记录，通过密码学哈希链接到前一个区块形成链。核心特性包括：去中心化（无中央权威）、不可篡改（修改一个区块需要修改所有后续区块）、透明性（所有交易公开可查）、匿名性或假名性。共识机制是区块链的关键技术，包括工作量证明（PoW）、权益证明（PoS）、委托权益证明（DPoS）。应用领域包括加密货币、智能合约、供应链追溯、数字身份认证和DeFi。".to_string(),
        domain: "区块链".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是智能合约？".to_string(),
        answer: "智能合约是部署在区块链上自动执行的程序代码，在满足预设条件时自动触发交易或操作。最著名的智能合约平台是以太坊，使用Solidity语言编写。智能合约具有去中心化执行、透明、不可篡改、自动化等特性。应用场景包括DeFi（借贷、交易所）、NFT、去中心化自治组织（DAO）、供应链管理等。智能合约的安全至关重要，历史上多次因代码漏洞导致重大资产损失。".to_string(),
        domain: "区块链".to_string(),
    });

    // ========== 计算机系统 ==========
    pairs.push(QAPair {
        question: "什么是操作系统中的进程和线程？".to_string(),
        answer: "进程是操作系统资源分配的基本单位，包含代码、数据、堆栈和系统资源（如文件句柄），拥有独立的地址空间。线程是CPU调度的基本单位，是进程内的执行流，同一进程内的线程共享地址空间和资源。多线程的优势在于轻量级（创建和切换开销小）和高效通信（共享内存），但需要注意同步问题。现代操作系统支持用户级线程和内核级线程，Go语言的goroutine和Python的协程是用户级线程的典型实现。".to_string(),
        domain: "计算机科学".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是虚拟内存？".to_string(),
        answer: "虚拟内存是操作系统提供的一种内存管理技术，为每个进程提供独立的虚拟地址空间，通过页表将虚拟地址映射到物理内存。主要功能包括：内存隔离（进程间互不干扰）、简化内存管理（连续虚拟地址映射到不连续物理地址）、按需加载（只加载需要的页面）、内存共享（不同进程映射同一物理页面）。页面置换算法包括FIFO、LRU（最近最少使用）、Clock算法等。虚拟内存的缺点是可能引入缺页中断开销。".to_string(),
        domain: "计算机科学".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是TCP/IP协议栈？".to_string(),
        answer: "TCP/IP协议栈是互联网通信的基础协议体系，分为四层：应用层（HTTP、FTP、SMTP、DNS等）、传输层（TCP和UDP）、网络层（IP协议）和网络接口层（以太网、Wi-Fi等）。TCP提供面向连接、可靠的数据传输服务，包含三次握手建立连接、流量控制、拥塞控制和差错重传机制。UDP提供无连接、不可靠但低延迟的服务。IP协议负责数据包的路由和转发，IPv4和IPv6是主要版本。".to_string(),
        domain: "计算机网络".to_string(),
    });

    // ========== 数学 ==========
    pairs.push(QAPair {
        question: "什么是线性回归？".to_string(),
        answer: "线性回归是统计学习和机器学习中最基础的模型，用于建立自变量X和因变量Y之间的线性关系。模型形式为Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε，其中β是模型参数，ε是误差项。参数通常通过最小二乘法（OLS）估计，即最小化预测值与真实值的平方误差和。评估指标包括R²（决定系数）、均方误差（MSE）和均方根误差（RMSE）。线性回归的假设包括线性关系、误差独立同分布、同方差性和无多重共线性。".to_string(),
        domain: "数学".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是贝叶斯定理？".to_string(),
        answer: "贝叶斯定理描述了在给定相关条件下事件发生的概率，形式为P(A|B) = P(B|A) × P(A) / P(B)，其中P(A|B)是后验概率（在观测到B后A的概率），P(A)是先验概率，P(B|A)是似然，P(B)是证据。贝叶斯定理在机器学习中的核心应用包括朴素贝叶斯分类器、贝叶斯推理和贝叶斯优化。朴素贝叶斯假设特征条件独立，尽管这个假设在现实中很少成立，但它在文本分类等任务中仍然表现良好。".to_string(),
        domain: "数学".to_string(),
    });

    // ========== 物理学 ==========
    pairs.push(QAPair {
        question: "什么是量子计算？".to_string(),
        answer: "量子计算利用量子力学原理（叠加态和纠缠态）进行计算。量子比特（qubit）可以同时处于0和1的叠加状态，使得量子计算机在特定问题上拥有指数级优势。核心概念包括：量子门操作、量子纠缠（多个量子比特之间的关联）、量子测量（使叠加态坍缩到确定值）。著名的量子算法包括Shor算法（大数分解）和Grover算法（无序搜索）。当前量子计算面临退相干、错误率高等挑战，仍处于NISQ（含噪声中等规模量子）时代。".to_string(),
        domain: "物理学".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是相对论？".to_string(),
        answer: "相对论是爱因斯坦提出的物理学理论，包含狭义相对论和广义相对论。狭义相对论的核心是光速不变原理和相对性原理，推导出时间膨胀（运动时钟变慢）、长度收缩、质能等价E=mc²等结论。广义相对论将引力解释为时空弯曲，物质和能量告诉时空如何弯曲，弯曲的时空告诉物质如何运动。广义相对论的预言包括引力红移、引力透镜、黑洞和引力波，这些都已通过实验观测得到验证。".to_string(),
        domain: "物理学".to_string(),
    });

    // ========== 生物学与医学 ==========
    pairs.push(QAPair {
        question: "什么是CRISPR基因编辑？".to_string(),
        answer: "CRISPR是一种革命性的基因编辑技术，源自细菌的天然免疫系统。Cas9蛋白在引导RNA的指引下，精确切割目标DNA序列，随后细胞的DNA修复机制引入基因修改。CRISPR技术具有精准、高效、低成本、易操作等优势，广泛应用于基因功能研究、农作物改良、疾病模型构建和基因治疗。然而，脱靶效应和伦理问题（如人类胚胎编辑）仍是CRISPR面临的重大挑战。2020年诺贝尔化学奖授予了CRISPR技术的先驱者。".to_string(),
        domain: "生物学".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是mRNA疫苗？".to_string(),
        answer: "mRNA疫苗是一种新型疫苗技术，利用信使RNA（mRNA）指导人体细胞产生特定抗原蛋白，从而激发免疫反应。与传统疫苗相比，mRNA疫苗的优势包括：开发速度快（仅需目标病原体的基因序列）、无需使用活病毒、可快速调整以应对变异株。mRNA疫苗在COVID-19大流行中展示了巨大潜力，辉瑞/BioNTech和Moderna的疫苗就是基于mRNA技术。该技术未来可能应用于癌症治疗、流感疫苗和罕见病治疗等领域。".to_string(),
        domain: "医学".to_string(),
    });

    // ========== 天文学 ==========
    pairs.push(QAPair {
        question: "什么是黑洞？".to_string(),
        answer: "黑洞是广义相对论预言的天体，其引力极强，连光都无法逃脱。黑洞由大质量恒星坍缩形成，核心是奇点（密度无限大的点），周围是事件视界（有去无回的边界）。根据质量分为：恒星级黑洞（几倍到几十倍太阳质量）、超大质量黑洞（百万到数十亿倍太阳质量，位于星系中心）、中等质量黑洞。2019年事件视界望远镜首次拍摄到黑洞照片（M87星系中心黑洞），2020年诺贝尔物理学奖授予黑洞相关研究。".to_string(),
        domain: "天文学".to_string(),
    });

    // ========== 历史与文化 ==========
    pairs.push(QAPair {
        question: "什么是文艺复兴？".to_string(),
        answer: "文艺复兴是14至17世纪欧洲的一场思想和文化运动，标志着从中世纪向近代的过渡。起源于意大利佛罗伦萨，强调人文主义精神，重视人的价值和现世生活。代表人物包括：艺术三杰达芬奇（《蒙娜丽莎》）、米开朗基罗（《大卫》雕塑）和拉斐尔（《雅典学院》）；文学领域的但丁（《神曲》）、彼特拉克和薄伽丘（《十日谈》）；科学领域的哥白尼（日心说）、伽利略和开普勒。文艺复兴推动了艺术、科学、哲学和宗教的全面革新。".to_string(),
        domain: "历史".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是丝绸之路？".to_string(),
        answer: "丝绸之路是古代连接中国与中亚、西亚、欧洲和非洲的贸易路线网络，形成于公元前2世纪（西汉时期）。名称来源于主要贸易商品丝绸。丝绸之路不仅促进了商品交换（丝绸、瓷器、香料、宝石等），更促进了文化交流、宗教传播（佛教、伊斯兰教、基督教）和技术传播（造纸术、火药、印刷术）。主要路线包括：长安-河西走廊-西域-中亚-波斯-地中海。2014年'丝绸之路：长安-天山廊道的路网'被列入世界遗产。".to_string(),
        domain: "历史".to_string(),
    });

    // ========== 哲学 ==========
    pairs.push(QAPair {
        question: "什么是存在主义？".to_string(),
        answer: "存在主义是20世纪重要的哲学流派，强调个体的自由、选择和责任。核心命题是'存在先于本质'——人没有预设的本质，通过自己的选择和行动定义自己。主要代表人物包括：萨特（《存在与虚无》）、加缪（《西西弗神话》）、克尔凯郭尔和尼采。存在主义探讨的关键主题包括：自由与责任、焦虑与荒诞、真诚与自我欺骗。加缪在《西西弗神话》中提出，即使面对无意义的世界，人依然可以通过反抗和行动创造意义。".to_string(),
        domain: "哲学".to_string(),
    });

    // ========== 经济学 ==========
    pairs.push(QAPair {
        question: "什么是通货膨胀？".to_string(),
        answer: "通货膨胀是指货币购买力下降导致一般物价水平持续上涨的经济现象。主要成因包括：需求拉动（总需求超过总供给）、成本推动（生产成本上升）、货币超发和预期推动。适量通胀（通常2%左右）被认为是经济发展的正常现象，但恶性通胀会严重破坏经济。衡量通胀的常用指标有消费者价格指数（CPI）和生产者价格指数（PPI）。央行的主要货币政策工具包括调整基准利率、存款准备金率和公开市场操作，用于控制通胀和维持经济稳定。".to_string(),
        domain: "经济学".to_string(),
    });
    pairs.push(QAPair {
        question: "什么是复利效应？".to_string(),
        answer: "复利是指投资收益再投资产生额外收益的效应，即'利滚利'。公式为A = P(1 + r/n)^(nt)，其中P是本金，r是年利率，n是每年复利次数，t是年数。复利的效果随时间的延长呈指数增长，长期复利的效果惊人——这就是为什么尽早开始投资如此重要。爱因斯坦称复利为'世界第八大奇迹'。在投资中，影响复利效果的关键因素是收益率、时间和持续性，其中时间是最重要的变量。".to_string(),
        domain: "经济学".to_string(),
    });

    // ========== 环境科学 ==========
    pairs.push(QAPair {
        question: "什么是碳中和？".to_string(),
        answer: "碳中和是指通过减少温室气体排放和增加碳汇（如植树造林、碳捕获和封存），使净碳排放为零的状态。实现碳中和的主要路径包括：能源转型（从化石能源转向太阳能、风能等可再生能源）、提高能效、电气化（交通和工业）、碳捕获利用和封存（CCUS）、碳交易市场机制。中国提出了2030年碳达峰、2060年碳中和的目标。碳中和是全球应对气候变化的核心策略，涉及技术、经济、政策和社会的全面转型。".to_string(),
        domain: "环境科学".to_string(),
    });

    pairs
}

// ========== 长文本文章生成 ==========

struct LongArticle {
    title: String,
    content: String,
    domain: String,
}

fn get_long_articles() -> Vec<LongArticle> {
    vec![
        LongArticle {
            title: "人工智能的发展历程与未来展望".to_string(),
            content: "人工智能（AI）的概念最早可追溯到1956年的达特茅斯会议，标志着AI作为一门学科的诞生。此后经历了多次起伏：早期的符号主义和专家系统、1980年代的连接主义复兴、21世纪初的统计学习革命，到2010年代深度学习的爆发。2012年AlexNet在ImageNet竞赛中的突破性表现开启了深度学习时代；2017年Transformer架构的提出彻底改变了自然语言处理；2020年代大规模语言模型（LLM）如GPT系列和BERT推动了AI能力的质变。大型语言模型展现出了涌现能力，包括上下文学习、推理链和指令跟随等，这些能力在小模型中并不明显。随着多模态AI的发展，模型能够同时处理文本、图像、音频和视频，向着通用人工智能（AGI）的目标迈进。未来AI发展的关键方向包括：更高效的训练和推理方法、更好的对齐和安全技术、更强的推理和规划能力、以及更广泛的行业应用。同时，AI治理、伦理规范和监管框架的建设也至关重要，以确保AI技术造福全人类。".to_string(),
            domain: "人工智能".to_string(),
        },
        LongArticle {
            title: "Rust语言的所有权系统详解".to_string(),
            content: "Rust的所有权系统是其最具创新性的特性，它在不需要垃圾回收器的情况下保证了内存安全。所有权的三条核心规则：Rust中的每个值都有一个称为其所有者（owner）的变量；一个值同时只能有一个所有者；当所有者离开作用域时，这个值将被丢弃。借用（Borrowing）允许通过引用访问值而不转移所有权，分为不可变引用（&T）和可变引用（&mut T）。借用规则：任意时刻只能有一个可变引用或任意数量的不可变引用；引用必须始终有效。生命周期（Lifetimes）确保引用不会超过被引用数据的存活期，编译器通过生命周期标注（如'a）进行静态检查。Rust的智能指针包括Box<T>（堆分配）、Rc<T>（引用计数）、Arc<T>（原子引用计数）和RefCell<T>（运行时借用检查）。所有权系统使得Rust在提供系统级性能的同时消除了一整类内存相关bug，这一创新使Rust连续多年被评为最受开发者喜爱的语言。".to_string(),
            domain: "编程语言".to_string(),
        },
        LongArticle {
            title: "深度学习中的优化算法综述".to_string(),
            content: "优化算法是深度学习训练的核心。最基本的随机梯度下降（SGD）通过梯度方向更新参数，但面临学习率选择和收敛速度问题。带动量的SGD通过累积梯度更新方向，加速收敛并减少振荡。AdaGrad自适应调整每个参数的学习率，适合稀疏数据。RMSProp解决了AdaGrad学习率单调递减的问题，通过指数移动平均累积梯度平方。Adam（Adaptive Moment Estimation）结合了动量和RMSProp的优点，维护一阶矩（梯度均值）和二阶矩（梯度平方均值）的指数移动平均，是当前最常用的优化器。AdamW在Adam基础上将权重衰减与自适应学习率解耦，改进了泛化性能。学习率调度（Cosine Annealing、Warmup、Step Decay）对训练效果影响显著，合适的调度策略可以提升模型性能和收敛速度。混合精度训练（FP16+FP32）和梯度累积技术则在训练效率和内存使用方面提供重要优化。".to_string(),
            domain: "人工智能".to_string(),
        },
        LongArticle {
            title: "分布式系统的一致性与共识算法".to_string(),
            content: "分布式系统中的一致性问题是计算机科学的核心挑战之一。CAP定理指出分布式系统无法同时满足一致性（Consistency）、可用性（Availability）和分区容错性（Partition Tolerance），最多只能满足其中两项。最终一致性是最弱的一致性模型，允许多副本间存在短暂不一致，但最终会达到一致。强一致性保证所有副本在更新后立即一致，但牺牲了可用性。Paxos是第一个被广泛接受的分布式共识算法，通过提议者（Proposer）、接受者（Acceptor）和学习者（Learner）三个角色，实现在不可靠网络环境下的共识。但Paxos的理解和实现复杂度较高。Raft算法是Paxos的简化版本，通过领导者选举、日志复制和安全机制实现共识，更易于理解和实现。Raft将问题分解为：领导者选举（Leader Election）、日志复制（Log Replication）和安全（Safety），是分布式系统实现中的热门选择。分布式事务协议如两阶段提交（2PC）和三阶段提交（3PC）用于跨节点事务一致性保障，但存在阻塞问题。现代分布式系统倾向于使用Saga模式或基于消息的最终一致性方案。".to_string(),
            domain: "计算机科学".to_string(),
        },
        LongArticle {
            title: "自然语言处理中的预训练技术演进".to_string(),
            content: "自然语言处理（NLP）的预训练技术经历了从静态词嵌入到动态上下文表示的革命性演变。早期的Word2Vec和GloVe为每个词学习固定的向量表示，无法处理一词多义问题。2018年ELMo引入了深度双向LSTM生成上下文相关的词表示，标志着动态词嵌入时代的开始。同年BERT通过Transformer编码器和掩码语言模型预训练在11项NLP任务上取得突破性结果，开启了预训练-微调范式。GPT系列（GPT、GPT-2、GPT-3、GPT-4）采用自回归语言模型和规模扩展策略，展示了涌现能力。T5将所有NLP任务统一为文本到文本格式。RoBERTa优化了BERT的训练策略。XLNet结合了自回归和自编码的优点。ELECTRA通过检测器-生成器架构提升训练效率。AlBERT通过参数共享减少模型大小。当前前沿方向包括：指令微调（Instruction Tuning）、基于人类反馈的强化学习（RLHF）、思维链（Chain-of-Thought）提示和检索增强生成（RAG）。大语言模型的应用已扩展到代码生成、数学推理、多模态理解和工具使用等领域。".to_string(),
            domain: "自然语言处理".to_string(),
        },
        LongArticle {
            title: "量子力学的基本原理与应用".to_string(),
            content: "量子力学是描述微观粒子行为的物理学理论，与相对论并列为现代物理学的两大基石。其核心原理包括：波粒二象性——微观粒子同时具有粒子性和波动性；不确定性原理——无法同时精确知道粒子的位置和动量；叠加态——粒子可以同时处于多个状态的线性组合，直到测量使状态坍缩；量子纠缠——两个或多个粒子形成关联状态，无论距离多远，对一个粒子的测量会瞬间影响另一个粒子的状态。薛定谔方程是量子力学的基本方程，描述量子态的时间演化。量子力学的应用极为广泛：半导体物理和晶体管是现代电子设备的基础；激光技术依赖量子跃迁原理；核磁共振成像（MRI）使用核自旋原理；量子密码学利用量子不可克隆定理实现安全通信。量子计算则是量子力学最有前景的新兴应用领域，有望在特定计算任务上超越经典计算机。".to_string(),
            domain: "物理学".to_string(),
        },
        LongArticle {
            title: "机器学习中的偏差-方差权衡".to_string(),
            content: "偏差-方差权衡（Bias-Variance Tradeoff）是监督学习中最核心的概念之一，描述了模型泛化误差的三个组成部分。偏差（Bias）是模型预测值的期望与真实值之间的差异，高偏差导致欠拟合。方差（Variance）是模型预测值在不同训练集上的变化程度，高方差导致过拟合。不可约噪声（Irreducible Noise）是数据本身固有的噪声。总误差 = 偏差² + 方差 + 噪声。简单模型（如线性回归）通常具有高偏差低方差，复杂模型（如深度决策树）具有低偏差高方差。目标是在两者之间找到最佳平衡点。正则化（L1/L2）、交叉验证、集成方法（Bagging降低方差、Boosting降低偏差）是控制偏差-方差权衡的常用技术。偏差-方差权衡概念还解释了为什么某些模型选择策略（如早停法、剪枝）有效，以及为什么集成方法通常优于单个模型。深入理解这一权衡对于模型选择和超参数调优至关重要。".to_string(),
            domain: "人工智能".to_string(),
        },
        LongArticle {
            title: "Go语言的并发模型与设计哲学".to_string(),
            content: "Go语言的并发模型基于CSP（Communicating Sequential Processes）理论，通过goroutine和channel实现并发编程。goroutine是轻量级的执行线程，由Go运行时管理，创建成本极低（约2KB栈空间），可以轻松创建数十万个goroutine。channel是goroutine之间通信的管道，通过通信共享内存而非通过共享内存通信。Go的设计哲学强调简洁和实用：显式错误处理而非异常机制、接口的隐式实现、组合优于继承、约定优于配置。Go的标准库提供了强大的工具集，包括HTTP服务器、JSON处理、模板引擎、测试框架和性能分析工具。Go在云原生领域取得了巨大成功，Docker、Kubernetes、Prometheus、Terraform等重量级项目都是用Go编写的。Go的编译速度极快，部署简单（静态编译生成单一二进制文件），这些特性使其非常适合构建微服务和云基础设施。Go的类型系统在保持安全性的同时避免过于复杂，没有泛型曾是主要痛点，但在Go 1.18中已加入泛型支持。".to_string(),
            domain: "编程语言".to_string(),
        },
        LongArticle {
            title: "计算机视觉中的卷积神经网络".to_string(),
            content: "卷积神经网络（CNN）是计算机视觉领域的基础架构，专门设计用于处理具有网格状拓扑的数据（如图像）。CNN的核心组件包括：卷积层使用可学习滤波器（卷积核）扫描输入图像，提取局部特征如边缘、纹理和形状。池化层（最大池化、平均池化）降低特征图的空间维度，减少参数量和计算量，同时提供平移不变性。全连接层将提取的高层特征映射到最终输出。经典CNN架构包括：LeNet-5（手写数字识别）、AlexNet（深度CNN的先驱，引入ReLU和Dropout）、VGGNet（强调深度和均匀结构）、GoogLeNet/Inception（引入Inception模块，多尺度并行卷积）、ResNet（残差连接解决梯度消失问题，可训练超深网络）、DenseNet（密集连接，特征复用）、EfficientNet（神经架构搜索实现效率-准确率最佳平衡）。现代计算机视觉已进入Vision Transformer（ViT）时代，Transformer在视觉任务上展现出与CNN相当甚至更好的性能，但CNN因其高效和成熟仍然是许多实际应用的首选。".to_string(),
            domain: "计算机视觉".to_string(),
        },
        LongArticle {
            title: "理解CAP定理与分布式系统设计".to_string(),
            content: "CAP定理由Eric Brewer在2000年提出，指出分布式数据系统最多只能同时满足一致性、可用性和分区容错性中的两项。一致性（Consistency）意味着所有节点在同一时间看到相同的数据；可用性（Availability）意味着每个请求都能获得响应（无论成功或失败）；分区容错性（Partition Tolerance）意味着系统在网络分区（节点间通信中断）时仍能正常运行。在实际分布式系统中，网络分区是不可避免的，因此系统设计者实际上需要在CP（一致性和分区容错性）和AP（可用性和分区容错性）之间做选择。CP系统如HBase和ZooKeeper优先保证一致性，在网络分区时可能暂停服务。AP系统如Cassandra和DynamoDB优先保证可用性，在网络分区时允许暂时不一致。现代分布式数据库如Spanner通过原子钟和TrueTime API实现了强一致性和高可用性的结合。BASE（Basically Available, Soft state, Eventually consistent）原则是ACID的替代方案，强调最终一致性。在设计分布式系统时，需要根据业务需求权衡一致性和可用性，例如银行交易需要强一致性，而社交媒体动态可以接受最终一致性。".to_string(),
            domain: "计算机科学".to_string(),
        },
    ]
}

// ========== 代码片段生成 ==========

fn generate_code_snippet(rng: &mut StdRng) -> (String, String, String) {
    let snippets = [
        (
            "Rust: 实现一个泛型缓存LRU",
            "use std::collections::HashMap;\nuse std::hash::Hash;\n\nstruct LRUCache<K, V> {\n    capacity: usize,\n    map: HashMap<K, V>,\n    order: Vec<K>,\n}\n\nimpl<K: Clone + Eq + Hash, V> LRUCache<K, V> {\n    fn new(capacity: usize) -> Self {\n        Self { capacity, map: HashMap::new(), order: Vec::new() }\n    }\n\n    fn get(&mut self, key: &K) -> Option<&V> {\n        if self.map.contains_key(key) {\n            self.order.retain(|k| k != key);\n            self.order.push(key.clone());\n            self.map.get(key)\n        } else { None }\n    }\n\n    fn put(&mut self, key: K, value: V) {\n        if self.map.contains_key(&key) {\n            self.order.retain(|k| k != &key);\n        } else if self.map.len() >= self.capacity {\n            let lru = self.order.remove(0);\n            self.map.remove(&lru);\n        }\n        self.order.push(key.clone());\n        self.map.insert(key, value);\n    }\n}",
            "编程语言"
        ),
        (
            "Python: 使用NumPy实现神经网络层",
            "import numpy as np\n\ndef relu(x):\n    return np.maximum(0, x)\n\ndef softmax(x):\n    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))\n    return e_x / np.sum(e_x, axis=-1, keepdims=True)\n\nclass DenseLayer:\n    def __init__(self, input_dim, output_dim):\n        self.w = np.random.randn(input_dim, output_dim) * 0.01\n        self.b = np.zeros((1, output_dim))\n\n    def forward(self, x):\n        self.input = x\n        return relu(x @ self.w + self.b)\n\n    def backward(self, grad, lr=0.01):\n        grad_relu = grad * (self.input @ self.w + self.b > 0)\n        dw = self.input.T @ grad_relu\n        db = np.sum(grad_relu, axis=0, keepdims=True)\n        dx = grad_relu @ self.w.T\n        self.w -= lr * dw\n        self.b -= lr * db\n        return dx",
            "编程语言"
        ),
        (
            "JavaScript: 实现Promise限流队列",
            "class ConcurrencyQueue {\n    constructor(concurrency = 5) {\n        this.concurrency = concurrency;\n        this.queue = [];\n        this.active = 0;\n    }\n\n    async add(fn) {\n        return new Promise((resolve, reject) => {\n            this.queue.push({ fn, resolve, reject });\n            this.process();\n        });\n    }\n\n    async process() {\n        if (this.active >= this.concurrency || this.queue.length === 0) return;\n        this.active++;\n        const { fn, resolve, reject } = this.queue.shift();\n        try {\n            resolve(await fn());\n        } catch (e) {\n            reject(e);\n        } finally {\n            this.active--;\n            this.process();\n        }\n    }\n}\n\nconst queue = new ConcurrencyQueue(3);\nfor (let i = 0; i < 10; i++) {\n    queue.add(() => fetch(`https://api.example.com/item/${i}`));\n}",
            "编程语言"
        ),
        (
            "Go: 并发处理数据流模式",
            "package main\n\nfunc producer(nums []int) <-chan int {\n    out := make(chan int)\n    go func() {\n        defer close(out)\n        for _, n := range nums {\n            out <- n\n        }\n    }()\n    return out\n}\n\nfunc square(in <-chan int) <-chan int {\n    out := make(chan int)\n    go func() {\n        defer close(out)\n        for n := range in {\n            out <- n * n\n        }\n    }()\n    return out\n}\n\nfunc merge(channels ...<-chan int) <-chan int {\n    var wg sync.WaitGroup\n    out := make(chan int)\n    output := func(c <-chan int) {\n        defer wg.Done()\n        for n := range c {\n            out <- n\n        }\n    }\n    wg.Add(len(channels))\n    for _, c := range channels {\n        go output(c)\n    }\n    go func() {\n        wg.Wait()\n        close(out)\n    }()\n    return out\n}",
            "编程语言"
        ),
        (
            "SQL: 复杂查询与窗口函数",
            "WITH monthly_sales AS (\n    SELECT \n        department,\n        DATE_TRUNC('month', sale_date) AS month,\n        SUM(amount) AS total_sales,\n        COUNT(DISTINCT customer_id) AS unique_customers\n    FROM sales\n    WHERE sale_date >= '2024-01-01'\n    GROUP BY department, DATE_TRUNC('month', sale_date)\n),\nranked_departments AS (\n    SELECT \n        department,\n        month,\n        total_sales,\n        RANK() OVER (\n            PARTITION BY month \n            ORDER BY total_sales DESC\n        ) AS rank_in_month,\n        LAG(total_sales) OVER (\n            PARTITION BY department \n            ORDER BY month\n        ) AS prev_month_sales\n    FROM monthly_sales\n)\nSELECT \n    department,\n    month,\n    total_sales,\n    unique_customers,\n    ROUND(\n        (total_sales - prev_month_sales) / \n        NULLIF(prev_month_sales, 0) * 100, 2\n    ) AS growth_pct\nFROM ranked_departments\nWHERE rank_in_month <= 3\nORDER BY month DESC, rank_in_month;",
            "编程语言"
        ),
    ];

    let idx = rng.gen_range(0..snippets.len());
    let (title, code, domain) = &snippets[idx];
    (
        format!("请实现以下功能：{}", title),
        format!("```{}\n{}\n```", if *title == "SQL" { "sql" } else { "code" }, code),
        domain.to_string(),
    )
}

// ========== 多轮对话生成 ==========

fn generate_multiturn_dialog(rng: &mut StdRng) -> (Vec<(String, String)>, String) {
    let dialogs = vec![
        vec![
            ("什么是机器学习？".to_string(), "机器学习是AI的一个分支，让计算机从数据中学习模式。主要分为监督学习、无监督学习和强化学习。".to_string()),
            ("监督学习和无监督学习有什么区别？".to_string(), "监督学习使用标注数据训练，每个样本都有对应的标签。无监督学习则使用未标注数据，目标是发现数据中的隐藏结构和模式。".to_string()),
            ("能给我举几个监督学习的例子吗？".to_string(), "当然。监督学习的常见应用包括：垃圾邮件分类（二分类）、手写数字识别（多分类）、房价预测（回归）。".to_string()),
            ("这些任务中常用的算法有哪些？".to_string(), "分类任务常用决策树、随机森林、SVM和神经网络。回归任务常用线性回归、岭回归和梯度提升树。选择算法时要考虑数据规模、特征类型和业务需求。".to_string()),
        ],
        vec![
            ("Rust语言为什么被认为是安全的？".to_string(), "Rust通过所有权系统在编译时保证内存安全，无需垃圾回收。每个值有唯一所有者，通过借用规则防止悬垂指针和数据竞争。".to_string()),
            ("所有权系统具体是怎么工作的？".to_string(), "核心规则是：每个值只有一个所有者；所有权可以转移（move）或借用（borrow）；借用分可变和不可变，不可变引用可以有多个，可变引用只能有一个。".to_string()),
            ("和C++的智能指针相比有什么优势？".to_string(), "Rust的检查在编译期完成，没有运行时开销。C++的智能指针虽然有RAII机制，但无法防止所有内存错误，且需要开发者特别注意使用方式。Rust的编译器保证了内存安全。".to_string()),
        ],
        vec![
            ("什么是Docker？".to_string(), "Docker是一个容器化平台，将应用及其依赖打包在轻量级容器中运行。容器共享宿主OS内核，比虚拟机更轻量、启动更快。".to_string()),
            ("Docker和虚拟机有什么区别？".to_string(), "虚拟机包含完整的操作系统（Hypervisor隔离），资源占用大。Docker容器共享宿主OS内核（进程级隔离），启动秒级，资源开销小。一个物理机可运行数百容器但只能运行数十虚拟机。".to_string()),
            ("什么是Docker Compose？".to_string(), "Docker Compose是定义和运行多容器Docker应用的工具。通过YAML文件定义服务的配置、网络和依赖关系，一条命令即可启动整个应用栈。".to_string()),
            ("在实际项目中怎么编排容器？".to_string(), "生产环境通常使用Kubernetes编排容器，它提供自动扩缩容、负载均衡、滚动更新和自愈能力。Docker Compose更适合开发环境和单机部署。".to_string()),
        ],
        vec![
            ("什么是区块链？".to_string(), "区块链是去中心化的分布式账本，数据以区块链接存储，通过密码学保证不可篡改。".to_string()),
            ("比特币和以太坊有什么区别？".to_string(), "比特币主要用于价值存储和转账，支持有限的脚本功能。以太坊引入了智能合约，支持去中心化应用开发和复杂的业务逻辑。".to_string()),
            ("什么是智能合约？".to_string(), "智能合约是部署在区块链上的程序代码，满足预设条件时自动执行。它消除了第三方信任需求，但代码漏洞可能导致资产损失，开发安全审计至关重要。".to_string()),
            ("智能合约有哪些实际应用？".to_string(), "DeFi（去中心化金融）是最主要的应用领域，包括去中心化交易所、借贷协议和收益聚合器。此外还有NFT市场、链上投票和供应链溯源等。".to_string()),
        ],
    ];

    let idx = rng.gen_range(0..dialogs.len());
    (dialogs[idx].clone(), "多轮对话".to_string())
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

fn generate_synthetic_qa(rng: &mut StdRng) -> (String, String, String) {
    let topics = ["Rust", "Python", "JavaScript", "Go", "C++", "算法", "人工智能", "数据库", "网络", "Kubernetes", "Docker"];
    let resources = ["书籍", "课程", "文档", "教程", "视频"];
    let concepts = ["机器学习", "神经网络", "并发", "性能", "缓存", "分布式", "安全"];

    let topic = topics[rng.gen_range(0..topics.len())];
    let resource = resources[rng.gen_range(0..resources.len())];
    let concept = concepts[rng.gen_range(0..concepts.len())];

    let kind = rng.gen_range(0..6);
    match kind {
        0 => (
            format!("用一句话解释什么是{}。", topic),
            format!("{}是一个广泛的领域，涉及相关知识、技术和应用。它在现代{}和日常生活中都有重要应用。", topic, concept),
            topic.to_string()
        ),
        1 => (
            format!("推荐几个学习{}的优质{}。", topic, resource),
            format!("学习{}的优质{}包括：官方文档、经典书籍、在线课程和实战项目。建议从基础开始，循序渐进。", topic, resource),
            topic.to_string()
        ),
        2 => {
            let a: i32 = rng.gen_range(10..500);
            let b: i32 = rng.gen_range(10..500);
            (format!("计算 {} + {} 等于多少？", a, b), format!("{} + {} = {}", a, b, a + b), "数学".to_string())
        },
        3 => (
            format!("如何在{}中实现高效的{}？", topic, concept),
            format!("在{}中实现高效的{}需要深入理解其核心机制和最佳实践，结合实际场景选择合适的方法和工具。", topic, concept),
            topic.to_string()
        ),
        4 => (
            format!("{}和{}相比有什么优势？", topic, concepts[rng.gen_range(0..concepts.len())]),
            format!("{}相比其他技术的优势在于更好的性能、更强的类型安全和更活跃的社区支持。", topic),
            topic.to_string()
        ),
        _ => (
            format!("{}的未来发展趋势是什么？", topic),
            format!("{}的未来发展方向包括性能优化、生态完善、跨平台支持和更广泛的应用场景。", topic),
            topic.to_string()
        ),
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

    // 处理图片到文本数据
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

    // 确保 data 目录存在
    let data_dir = PathBuf::from("data");
    std::fs::create_dir_all(&data_dir).ok();

    // SFT 数据输出
    let out_path = args.out.map(PathBuf::from).unwrap_or_else(|| {
        let mut path = data_dir.clone();
        path.push("sft_data.jsonl");
        path
    });

    // 纯文本语料输出（用于 BPE 训练）
    let corpus_path = PathBuf::from(&args.corpus_out);
    if let Some(parent) = corpus_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    let unique_out = get_unique_file_path(out_path);
    let file = File::create(&unique_out).expect("create file");
    let mut w = BufWriter::new(file);

    // 打开语料文件
    let corpus_file = File::create(&corpus_path).expect("create corpus file");
    let mut corpus_w = BufWriter::new(corpus_file);

    let local_qa = get_local_qa_pairs();
    let long_articles = get_long_articles();

    let total = args.count;
    // 分配数据比例：60% QA，15% 长文本，10% 代码，10% 多轮对话，5% 合成数据
    let qa_count = (total as f64 * 0.60) as usize;
    let article_count = (total as f64 * 0.15) as usize;
    let code_count = (total as f64 * 0.10) as usize;
    let dialog_count = (total as f64 * 0.10) as usize;
    let synthetic_count = total - qa_count - article_count - code_count - dialog_count;

    let mut id_counter = 0;

    // 1. 写入本地 QA 对（循环使用直到达到 qa_count）
    for _ in 0..qa_count {
        let pair = &local_qa[rng.gen_range(0..local_qa.len())];
        let messages = generate_dialog_style(&mut rng, &pair.question, &pair.answer, &pair.domain);
        let mut data_obj = json!({"messages": messages, "id": id_counter, "domain": pair.domain});
        if args.multimodal {
            data_obj["image_path"] = json!("assets/sample_image.jpg");
        }
        let line = data_obj.to_string();
        w.write_all(line.as_bytes()).unwrap();
        w.write_all(b"\n").unwrap();

        // 写入语料文件
        corpus_w.write_all(pair.question.as_bytes()).unwrap();
        corpus_w.write_all(b"\n").unwrap();
        corpus_w.write_all(pair.answer.as_bytes()).unwrap();
        corpus_w.write_all(b"\n").unwrap();

        id_counter += 1;
    }

    // 2. 写入长文本文章
    for _ in 0..article_count {
        let article = &long_articles[rng.gen_range(0..long_articles.len())];
        let full_text = format!("{}\n{}", article.title, article.content);
        let messages = json!([
            {"role":"user","content":format!("请详细介绍一下{}", article.title)},
            {"role":"assistant","content":full_text}
        ]);
        let mut data_obj = json!({"messages": messages, "id": id_counter, "domain": article.domain});
        if args.multimodal {
            data_obj["image_path"] = json!("assets/sample_image.jpg");
        }
        let line = data_obj.to_string();
        w.write_all(line.as_bytes()).unwrap();
        w.write_all(b"\n").unwrap();

        // 写入语料文件
        corpus_w.write_all(article.title.as_bytes()).unwrap();
        corpus_w.write_all(b"\n").unwrap();
        corpus_w.write_all(article.content.as_bytes()).unwrap();
        corpus_w.write_all(b"\n").unwrap();

        id_counter += 1;
    }

    // 3. 写入代码片段
    for _ in 0..code_count {
        let (prompt, code, domain) = generate_code_snippet(&mut rng);
        let messages = generate_dialog_style(&mut rng, &prompt, &code, &domain);
        let mut data_obj = json!({"messages": messages, "id": id_counter, "domain": domain, "type": "code"});
        if args.multimodal {
            data_obj["image_path"] = json!("assets/sample_image.jpg");
        }
        let line = data_obj.to_string();
        w.write_all(line.as_bytes()).unwrap();
        w.write_all(b"\n").unwrap();

        corpus_w.write_all(prompt.as_bytes()).unwrap();
        corpus_w.write_all(b"\n").unwrap();
        corpus_w.write_all(code.as_bytes()).unwrap();
        corpus_w.write_all(b"\n").unwrap();

        id_counter += 1;
    }

    // 4. 写入多轮对话
    for _ in 0..dialog_count {
        let (turns, domain) = generate_multiturn_dialog(&mut rng);
        let messages: Vec<serde_json::Value> = turns.iter().flat_map(|(q, a)| {
            vec![
                json!({"role":"user","content": q}),
                json!({"role":"assistant","content": a}),
            ]
        }).collect();
        let mut data_obj = json!({"messages": messages, "id": id_counter, "domain": domain, "type": "multiturn"});
        if args.multimodal {
            data_obj["image_path"] = json!("assets/sample_image.jpg");
        }
        let line = data_obj.to_string();
        w.write_all(line.as_bytes()).unwrap();
        w.write_all(b"\n").unwrap();

        for (q, a) in turns {
            corpus_w.write_all(q.as_bytes()).unwrap();
            corpus_w.write_all(b"\n").unwrap();
            corpus_w.write_all(a.as_bytes()).unwrap();
            corpus_w.write_all(b"\n").unwrap();
        }

        id_counter += 1;
    }

    // 5. 写入合成数据
    for _ in 0..synthetic_count {
        let (prompt, response, domain) = if args.web_only || (args.web && rng.gen_bool(0.5)) {
            let pair = &local_qa[rng.gen_range(0..local_qa.len())];
            (pair.question.clone(), pair.answer.clone(), pair.domain.clone())
        } else {
            generate_synthetic_qa(&mut rng)
        };
        let messages = generate_dialog_style(&mut rng, &prompt, &response, &domain);
        let mut data_obj = json!({"messages": messages, "id": id_counter, "domain": domain});
        if args.multimodal {
            data_obj["image_path"] = json!("assets/sample_image.jpg");
        }
        let line = data_obj.to_string();
        w.write_all(line.as_bytes()).unwrap();
        w.write_all(b"\n").unwrap();

        corpus_w.write_all(prompt.as_bytes()).unwrap();
        corpus_w.write_all(b"\n").unwrap();
        corpus_w.write_all(response.as_bytes()).unwrap();
        corpus_w.write_all(b"\n").unwrap();

        id_counter += 1;
    }

    w.flush().unwrap();
    corpus_w.flush().unwrap();

    let corpus_size = std::fs::metadata(&corpus_path).map(|m| m.len()).unwrap_or(0);
    println!("Wrote {} records to {}", id_counter, unique_out.display());
    println!("Wrote corpus text ({} bytes) to {}", corpus_size, corpus_path.display());

    // ========== 可选：训练 BPE Tokenizer ==========
    if args.train_bpe {
        println!("\n开始训练 BPE Tokenizer...");
        let corpus_text = std::fs::read_to_string(&corpus_path)
            .expect("Failed to read corpus text for BPE training");

        let tokenizer = Tokenizer::new_bpe(&corpus_text, args.bpe_vocab_size);

        let bpe_out = args.bpe_output.clone();
        tokenizer.save(&bpe_out).expect("Failed to save BPE tokenizer");

        println!("BPE Tokenizer 训练完成！");
        println!("  词表大小: {}", tokenizer.vocab_size);
        println!("  输出路径: {}", bpe_out);
        println!("  Tokenizer 类型: BPE");

        // 显示一些示例 tokens
        println!("\n示例编码:");
        let test_texts = ["人工智能", "机器学习", "深度学习", "Rust", "Transformer"];
        for text in &test_texts {
            let ids = tokenizer.encode(text);
            let decoded = tokenizer.decode(&ids);
            println!("  '{}' -> {:?} -> '{}'", text, ids, decoded);
        }
    }
}