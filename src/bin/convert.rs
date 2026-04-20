use burn::{prelude::*, backend::{Wgpu, NdArray}, record::CompactRecorder};
use clap::Parser;
use std::path::Path;

use sage::core::ModelConfig;

#[derive(Parser, Debug)]
pub struct ConvertArgs {
    #[arg(long, default_value = "./inference/configs/config_1B.json")]
    pub config_path: String,
    
    #[arg(long, default_value = "./models")]
    pub input_dir: String,
    
    #[arg(long, default_value = "./models")]
    pub output_dir: String,
    
    #[arg(long, default_value = "cuda")]
    pub device: String,
}

pub fn convert_state_dict<B: Backend>(
    input_dir: &str,
    output_dir: &str,
    config: &ModelConfig,
    device: &B::Device,
) {
    // Load HuggingFace weights
    log::info!("Loading HuggingFace weights from {}", input_dir);
    
    // Initialize model
    use sage::core::Model;
    let model: Model<B> = config.init(device);
    
    // Save model weights in our format
    let output_path = Path::new(output_dir).join("model.mpk");
    log::info!("Saving converted weights to {}", output_path.to_str().unwrap());
    
    model.save_file(
        output_path.to_str().unwrap(),
        &CompactRecorder::new()
    )
    .expect("Failed to save converted weights");
}

pub fn main() {
    let args = ConvertArgs::parse();
    
    // Load model configuration
    let config = ModelConfig::load(&args.config_path)
        .expect("Failed to load model configuration");
    
    // Convert weights based on device
    if args.device == "cuda" {
        use burn_wgpu::WgpuDevice;
        let device = WgpuDevice::default();
        type WgpuBackend = Wgpu<f32, i32, u8>;
        convert_state_dict::<WgpuBackend>(
            &args.input_dir,
            &args.output_dir,
            &config,
            &device
        );
    } else {
        use burn_ndarray::NdArrayDevice;
        let device = NdArrayDevice::default();
        type NdArrayBackend = NdArray<f32, i32, i8>;
        convert_state_dict::<NdArrayBackend>(
            &args.input_dir,
            &args.output_dir,
            &config,
            &device
        );
    }
    
    log::info!("Weight conversion completed successfully!");
}
