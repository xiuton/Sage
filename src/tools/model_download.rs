use crate::training::training::TrainingConfig;
use reqwest::Client;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ModelDownloader {
    client: Arc<Client>,
    models_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(models_dir: &str) -> Self {
        Self {
            client: Arc::new(Client::new()),
            models_dir: PathBuf::from(models_dir),
        }
    }

    pub async fn download_model(
        &self,
        model_id: &str,
        url: &str,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let model_dir = self.models_dir.join(model_id);
        fs::create_dir_all(&model_dir)?;

        log::info!("开始下载模型: {} 从 {}", model_id, url);

        let response = self.client.get(url).send().await?;
        let total_size = response.content_length().unwrap_or(0);

        let file_path = model_dir.join("model.mpk");
        let mut file = File::create(&file_path)?;

        let bytes = response.bytes().await?;
        file.write_all(&bytes)?;
        
        if total_size > 0 {
            log::info!("下载完成: {} bytes", bytes.len());
        }

        log::info!("模型下载完成: {}", model_id);
        Ok(model_dir)
    }

    pub async fn update_model(
        &self,
        model_id: &str,
        url: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model_dir = self.models_dir.join(model_id);
        
        if !model_dir.exists() {
            return Err("模型不存在".into());
        }

        log::info!("开始更新模型: {} 从 {}", model_id, url);

        let response = self.client.get(url).send().await?;
        let total_size = response.content_length().unwrap_or(0);

        let temp_path = model_dir.join("model_temp.mpk");
        let mut file = File::create(&temp_path)?;

        let bytes = response.bytes().await?;
        file.write_all(&bytes)?;
        
        if total_size > 0 {
            log::info!("更新完成: {} bytes", bytes.len());
        }

        let model_path = model_dir.join("model.mpk");
        fs::rename(&temp_path, &model_path)?;

        log::info!("模型更新完成: {}", model_id);
        Ok(())
    }

    pub fn get_model_config(&self, model_id: &str) -> Result<TrainingConfig, Box<dyn std::error::Error>> {
        let model_dir = self.models_dir.join(model_id);
        let config_path = model_dir.join("config.json");

        if !config_path.exists() {
            return Err("配置文件不存在".into());
        }

        let config_str = fs::read_to_string(&config_path)?;
        let config: TrainingConfig = serde_json::from_str(&config_str)?;
        
        Ok(config)
    }
}
