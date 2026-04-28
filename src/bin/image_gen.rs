use clap::Parser;
use burn_ndarray::NdArrayDevice;
use burn::tensor::backend::Backend;
use image::{ImageBuffer, Rgb, RgbImage};
use std::path::Path;

use sage::core::{DiffusionConfig, ImageGenerator, SimpleTokenizer};

#[derive(Parser, Debug)]
#[command(author, version, about = "图像生成工具", long_about = None)]
struct Args {
    #[arg(long, default_value_t = false)]
    generate_only: bool,

    #[arg(long, default_value = "./assets/generated_image.png")]
    output: String,

    #[arg(long, default_value_t = 50)]
    steps: usize,

    #[arg(long, default_value_t = 128)]
    latent_dim: usize,

    #[arg(long, default_value_t = 64)]
    image_size: usize,

    #[arg(long, default_value = "cpu", value_name = "cpu|gpu")]
    backend: String,

    #[arg(long, default_value = "a beautiful landscape", value_name = "TEXT")]
    prompt: String,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long)]
    model_path: Option<String>,

    #[arg(long)]
    config_path: Option<String>,
}

fn generate_random_filename() -> String {
    let hash: u128 = rand::random();
    format!("./assets/image_{:032x}.png", hash)
}

fn get_unique_filename(path: &str) -> String {
    let path = Path::new(path);
    if !path.exists() {
        return path.to_str().unwrap().to_string();
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path.file_stem().unwrap().to_str().unwrap();
    let extension = path.extension().unwrap_or_else(|| std::ffi::OsStr::new("png")).to_str().unwrap();

    for i in 1.. {
        let new_filename = format!("{}_{}.{}", stem, i, extension);
        let new_path = parent.join(new_filename);
        if !new_path.exists() {
            return new_path.to_str().unwrap().to_string();
        }
    }
    path.to_str().unwrap().to_string()
}

fn tensor_to_image_simple<B: Backend>(tensor: burn::Tensor<B, 4>) -> RgbImage {
    let [_, channels, height, width] = tensor.dims();
    let data = tensor.into_data();
    let values: Vec<f32> = data.to_vec().unwrap();

    let mut img: RgbImage = ImageBuffer::new(width as u32, height as u32);

    let batch_offset = channels * height * width;

    for y in 0..height {
        for x in 0..width {
            let px_idx = y * width + x;
            let r_idx = px_idx;
            let g_idx = batch_offset + px_idx;
            let b_idx = 2 * batch_offset + px_idx;

            let r = if r_idx < values.len() { (values[r_idx] * 127.5 + 127.5).clamp(0.0, 255.0) as u8 } else { 128 };
            let g = if g_idx < values.len() { (values[g_idx] * 127.5 + 127.5).clamp(0.0, 255.0) as u8 } else { 128 };
            let b = if b_idx < values.len() { (values[b_idx] * 127.5 + 127.5).clamp(0.0, 255.0) as u8 } else { 128 };

            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    img
}

fn load_config_from_file(path: &str) -> Result<DiffusionConfig, Box<dyn std::error::Error>> {
    use serde::Deserialize;
    
    #[derive(Deserialize)]
    struct ConfigJson {
        image_size: Option<usize>,
        in_channels: Option<usize>,
        latent_dim: Option<usize>,
        hidden_channels: Option<usize>,
        num_timesteps: Option<usize>,
        beta_start: Option<f64>,
        beta_end: Option<f64>,
    }

    let content = std::fs::read_to_string(path)?;
    let json: ConfigJson = serde_json::from_str(&content)?;

    Ok(DiffusionConfig {
        image_size: json.image_size.unwrap_or(64),
        in_channels: json.in_channels.unwrap_or(3),
        hidden_channels: json.hidden_channels.unwrap_or(128),
        num_timesteps: json.num_timesteps.unwrap_or(1000),
        latent_dim: json.latent_dim.unwrap_or(128),
        beta_start: json.beta_start.unwrap_or(0.0001) as f32,
        beta_end: json.beta_end.unwrap_or(0.02) as f32,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================");
    println!("  Sage 图像生成工具 v1.0");
    println!("========================================");

    let args = Args::parse();

    std::fs::create_dir_all("./assets").ok();

    println!("\n📦 初始化模型配置...");

    let output_path = if args.generate_only || args.output == "./assets/generated_image.png" {
        generate_random_filename()
    } else {
        get_unique_filename(&args.output)
    };

    let config = if let Some(ref config_path) = args.config_path {
        println!("📄 从配置文件加载: {}", config_path);
        load_config_from_file(config_path)?
    } else if let Some(ref model_dir) = args.model_path {
        let config_path = std::path::Path::new(model_dir).join("config.json");
        if config_path.exists() {
            println!("📄 从模型目录加载配置: {}", config_path.to_str().unwrap());
            load_config_from_file(config_path.to_str().unwrap())?
        } else {
            println!("⚠️  模型目录中未找到 config.json，使用默认配置");
            DiffusionConfig {
                image_size: args.image_size,
                in_channels: 3,
                hidden_channels: 128,
                num_timesteps: 1000,
                latent_dim: args.latent_dim,
                beta_start: 0.0001,
                beta_end: 0.02,
            }
        }
    } else {
        DiffusionConfig {
            image_size: args.image_size,
            in_channels: 3,
            hidden_channels: 128,
            num_timesteps: 1000,
            latent_dim: args.latent_dim,
            beta_start: 0.0001,
            beta_end: 0.02,
        }
    };

    if args.backend == "gpu" {
        println!("🚀 使用 GPU 后端进行图像生成...");
        run_with_gpu_backend(&config, &args, &output_path)?;
    } else {
        println!("🖥️ 使用 CPU 后端进行图像生成...");
        run_with_cpu_backend(&config, &args, &output_path)?;
    }

    Ok(())
}

fn run_with_cpu_backend(config: &DiffusionConfig, args: &Args, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use burn_ndarray::NdArray;

    let device = NdArrayDevice::Cpu;
    let generator = if let Some(ref model_dir) = args.model_path {
        let model_path = std::path::Path::new(model_dir).join("diffusion_model.mpk");
        if model_path.exists() {
            println!("📂 加载训练好的模型: {}", model_path.to_str().unwrap());
            ImageGenerator::<NdArray>::from_file(config.clone(), device, model_path)?
        } else {
            println!("⚠️  模型文件不存在: {}，使用随机初始化模型", model_path.to_str().unwrap());
            ImageGenerator::<NdArray>::new(config.clone(), device)
        }
    } else {
        ImageGenerator::<NdArray>::new(config.clone(), device)
    };

    if args.generate_only {
        println!("✨ 生成模式: VAE 随机图像生成");
        println!("📐 图像尺寸: {}x{}", args.image_size, args.image_size);
        println!("🔢 采样步数: {}", args.steps);

        let image_tensor = generator.generate_simple(1, args.image_size);

        let img = tensor_to_image_simple(image_tensor);
        img.save(output_path)?;

        println!("\n✅ 图像生成完成！");
        println!("💾 保存到: {}", output_path);
    } else if args.prompt != "a beautiful landscape" && !args.prompt.is_empty() {
        println!("✨ 生成模式: 文生图 (Text-to-Image)");
        println!("📝 提示词: {}", args.prompt);
        println!("📐 图像尺寸: {}x{}", args.image_size, args.image_size);
        println!("🔢 采样步数: {}", args.steps);

        let tokenizer = SimpleTokenizer::new(50000);
        let image_tensor = generator.generate_with_prompt(&args.prompt, &tokenizer, 1, args.image_size, args.steps);

        let img = tensor_to_image_simple(image_tensor);
        img.save(output_path)?;

        println!("\n✅ 图像生成完成！");
        println!("💾 保存到: {}", output_path);
    } else {
        println!("✨ 扩散模型生成模式（无文本条件）");
        println!("📐 图像尺寸: {}x{}", args.image_size, args.image_size);
        println!("🔢 采样步数: {}", args.steps);

        let image_tensor = generator.generate(1, args.image_size, args.steps);

        let img = tensor_to_image_simple(image_tensor);
        img.save(output_path)?;

        println!("\n✅ 图像生成完成！");
        println!("💾 保存到: {}", output_path);
    }

    Ok(())
}

fn run_with_gpu_backend(config: &DiffusionConfig, args: &Args, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::Wgpu;
    use burn_wgpu::WgpuDevice;

    let device = WgpuDevice::default();
    let generator = if let Some(ref model_dir) = args.model_path {
        let model_path = std::path::Path::new(model_dir).join("diffusion_model.mpk");
        if model_path.exists() {
            println!("📂 加载训练好的模型: {}", model_path.to_str().unwrap());
            ImageGenerator::<Wgpu>::from_file(config.clone(), device, model_path)?
        } else {
            println!("⚠️  模型文件不存在: {}，使用随机初始化模型", model_path.to_str().unwrap());
            ImageGenerator::<Wgpu>::new(config.clone(), device)
        }
    } else {
        ImageGenerator::<Wgpu>::new(config.clone(), device)
    };

    if args.generate_only {
        println!("✨ 生成模式: VAE 随机图像生成");
        println!("📐 图像尺寸: {}x{}", args.image_size, args.image_size);
        println!("🔢 采样步数: {}", args.steps);

        let image_tensor = generator.generate_simple(1, args.image_size);

        let img = tensor_to_image_simple::<Wgpu>(image_tensor);
        img.save(output_path)?;

        println!("\n✅ 图像生成完成！");
        println!("💾 保存到: {}", output_path);
    } else if args.prompt != "a beautiful landscape" && !args.prompt.is_empty() {
        println!("✨ 生成模式: 文生图 (Text-to-Image)");
        println!("📝 提示词: {}", args.prompt);
        println!("📐 图像尺寸: {}x{}", args.image_size, args.image_size);
        println!("🔢 采样步数: {}", args.steps);

        let tokenizer = SimpleTokenizer::new(50000);
        let image_tensor = generator.generate_with_prompt(&args.prompt, &tokenizer, 1, args.image_size, args.steps);

        let img = tensor_to_image_simple::<Wgpu>(image_tensor);
        img.save(output_path)?;

        println!("\n✅ 图像生成完成！");
        println!("💾 保存到: {}", output_path);
    } else {
        println!("✨ 扩散模型生成模式（无文本条件）");
        println!("📐 图像尺寸: {}x{}", args.image_size, args.image_size);
        println!("🔢 采样步数: {}", args.steps);

        let image_tensor = generator.generate(1, args.image_size, args.steps);

        let img = tensor_to_image_simple::<Wgpu>(image_tensor);
        img.save(output_path)?;

        println!("\n✅ 图像生成完成！");
        println!("💾 保存到: {}", output_path);
    }

    Ok(())
}