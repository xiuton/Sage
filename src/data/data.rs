use burn::{    data::{dataloader::batcher::Batcher, dataset::Dataset},    prelude::*,};use memmap2::Mmap;use std::{fs::File, path::Path, sync::Arc, io::{BufRead, BufReader}};use serde::{Deserialize, Serialize};use rand::{Rng, SeedableRng};

/// 数据处理配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataProcessingConfig {
    pub seq_len: usize,
    pub batch_size: usize,
    pub shuffle_buffer_size: usize,
    pub num_workers: usize,
    pub use_mmap: bool,
    pub enable_data_augmentation: bool,
    pub mask_probability: f32,
    pub random_insertion_probability: f32,
    pub random_deletion_probability: f32,
    pub random_substitution_probability: f32,
    pub min_sequence_length: usize,
    pub max_sequence_length: usize,
}

impl Default for DataProcessingConfig {
    fn default() -> Self {
        Self {
            seq_len: 512,
            batch_size: 32,
            shuffle_buffer_size: 10000,
            num_workers: 4,
            use_mmap: true,
            enable_data_augmentation: false,
            mask_probability: 0.15,
            random_insertion_probability: 0.05,
            random_deletion_probability: 0.05,
            random_substitution_probability: 0.05,
            min_sequence_length: 10,
            max_sequence_length: 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TextItem {
    pub input: Vec<i32>,
    pub target: Vec<i32>,
    pub mask: Vec<i32>,
    pub attention_mask: Option<Vec<i32>>,
    pub token_type_ids: Option<Vec<i32>>,
    pub image: Option<Vec<f32>>, // [3, 224, 224] 展平后的图像数据
    pub image_path: Option<String>,
}

pub struct TextDataset {
    data: Vec<usize>,
    mask: Vec<u8>,
    seq_len: usize,
    images: Option<Vec<Option<Vec<f32>>>>,
    image_paths: Option<Vec<Option<String>>>,
}

impl TextDataset {
    pub fn new(tokens: Vec<usize>, mask: Vec<u8>, seq_len: usize) -> Self {
        Self {
            data: tokens,
            mask,
            seq_len,
            images: None,
            image_paths: None,
        }
    }

    pub fn with_images(mut self, images: Vec<Option<Vec<f32>>>) -> Self {
        self.images = Some(images);
        self
    }

    pub fn with_image_paths(mut self, paths: Vec<Option<String>>) -> Self {
        self.image_paths = Some(paths);
        self
    }
}

impl Dataset<TextItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        if index + self.seq_len >= self.data.len() {
            return None;
        }

        let input: Vec<i32> = self.data[index..index + self.seq_len]
            .iter()
            .map(|&v| v as i32)
            .collect();

        let mut target = Vec::with_capacity(self.seq_len);
        let mut mask = Vec::with_capacity(self.seq_len);
        for j in 0..self.seq_len {
            let token_id = self.data[index + 1 + j] as i32;
            let m = *self.mask.get(index + 1 + j).unwrap_or(&1);
            target.push(token_id);
            mask.push(m as i32);
        }

        // 创建注意力掩码，全1表示所有位置都被关注
        let attention_mask = Some(vec![1; self.seq_len]);
        // 创建token类型ID，全0表示单一序列
        let token_type_ids = Some(vec![0; self.seq_len]);
        
        // 获取对应的图像（优先从内存，其次路径）
        let image = self.images.as_ref().and_then(|imgs| imgs.get(index).cloned().flatten());
        let image_path = self.image_paths.as_ref().and_then(|paths| paths.get(index).cloned().flatten());

        if image.is_none() {
            if let Some(_path) = image_path.as_ref() {
                // 如果有路径但没内存数据，尝试加载并处理（这里仅作占位，真实处理在 batcher 或外部完成更高效）
            }
        }

        Some(TextItem { 
            input, 
            target, 
            mask, 
            attention_mask, 
            token_type_ids,
            image,
            image_path,
        })
    }

    fn len(&self) -> usize {
        self.data.len().saturating_sub(self.seq_len + 1)
    }
}

pub struct MmapTextDataset {
    tokens: Arc<Mmap>,
    mask: Arc<Mmap>,
    start: usize,
    end: usize,
    seq_len: usize,
}

impl MmapTextDataset {
    pub fn open(
        tokens_path: impl AsRef<Path>,
        mask_path: impl AsRef<Path>,
        seq_len: usize,
    ) -> Self {
        match Self::open_with_result(tokens_path, mask_path, seq_len) {
            Ok(dataset) => dataset,
            Err(e) => {
                eprintln!("错误: {}", e);
                panic!("无法打开数据集文件");
            }
        }
    }
    
    pub fn open_with_result(
        tokens_path: impl AsRef<Path>,
        mask_path: impl AsRef<Path>,
        seq_len: usize,
    ) -> Result<Self, String> {
        let tokens_path = tokens_path.as_ref();
        let mask_path = mask_path.as_ref();
        
        let tokens_file = File::open(tokens_path)
            .map_err(|e| format!("打开tokens文件失败: {} ({})", e, tokens_path.display()))?;
        let mask_file = File::open(mask_path)
            .map_err(|e| format!("打开mask文件失败: {} ({})", e, mask_path.display()))?;

        let tokens = Arc::new(unsafe { 
            Mmap::map(&tokens_file)
                .map_err(|e| format!("内存映射tokens文件失败: {} ({})", e, tokens_path.display()))?
        });
        let mask = Arc::new(unsafe { 
            Mmap::map(&mask_file)
                .map_err(|e| format!("内存映射mask文件失败: {} ({})", e, mask_path.display()))?
        });

        let token_len = tokens.len() / 4;
        let mask_len = mask.len();
        let end = token_len.min(mask_len);

        Ok(Self {
            tokens,
            mask,
            start: 0,
            end,
            seq_len,
        })
    }

    pub fn with_range(&self, start: usize, end: usize) -> Self {
        let start = self.start + start;
        let end = (self.start + end).min(self.end);
        Self {
            tokens: Arc::clone(&self.tokens),
            mask: Arc::clone(&self.mask),
            start,
            end,
            seq_len: self.seq_len,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    fn token_at(&self, index: usize) -> usize {
        let i = self.start + index;
        let off = i * 4;
        u32::from_le_bytes([
            self.tokens[off],
            self.tokens[off + 1],
            self.tokens[off + 2],
            self.tokens[off + 3],
        ]) as usize
    }

    fn mask_at(&self, index: usize) -> u8 {
        self.mask[self.start + index]
    }
}

impl Dataset<TextItem> for MmapTextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        let total = self.end.saturating_sub(self.start);
        if index + self.seq_len >= total {
            return None;
        }

        let mut input = Vec::with_capacity(self.seq_len);
        let mut target = Vec::with_capacity(self.seq_len);
        let mut mask = Vec::with_capacity(self.seq_len);

        for j in 0..self.seq_len {
            input.push(self.token_at(index + j) as i32);
            let token_id = self.token_at(index + 1 + j) as i32;
            let m = self.mask_at(index + 1 + j);
            target.push(token_id);
            mask.push(m as i32);
        }

        // 创建注意力掩码，全1表示所有位置都被关注
        let attention_mask = Some(vec![1; self.seq_len]);
        // 创建token类型ID，全0表示单一序列
        let token_type_ids = Some(vec![0; self.seq_len]);

        Some(TextItem { 
            input, 
            target, 
            mask, 
            attention_mask, 
            token_type_ids,
            image: None,
            image_path: None,
        })
    }

    fn len(&self) -> usize {
        let total = self.end.saturating_sub(self.start);
        total.saturating_sub(self.seq_len + 1)
    }
}

#[derive(Clone, Debug)]
pub struct TextBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TextBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub mask: Tensor<B, 2, Int>,
    pub attention_mask: Tensor<B, 2, Int>,
    pub token_type_ids: Tensor<B, 2, Int>,
    pub images: Option<Tensor<B, 4>>,
}

impl<B: Backend> Batcher<B, TextItem, TextBatch<B>> for TextBatcher<B> {
    fn batch(&self, items: Vec<TextItem>, _device: &B::Device) -> TextBatch<B> {
        let batch_size = items.len();
        let seq_len = items.first().map(|v| v.input.len()).unwrap_or(0);

        let mut inputs_flat = Vec::with_capacity(batch_size * seq_len);
        let mut targets_flat = Vec::with_capacity(batch_size * seq_len);
        let mut mask_flat = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_flat = Vec::with_capacity(batch_size * seq_len);
        let mut token_type_ids_flat = Vec::with_capacity(batch_size * seq_len);
        let mut images_flat = Vec::new();
        let mut has_images = false;

        for item in items.iter() {
            inputs_flat.extend_from_slice(&item.input);
            targets_flat.extend_from_slice(&item.target);
            mask_flat.extend_from_slice(&item.mask);
            
            // 处理注意力掩码
            if let Some(attention_mask) = &item.attention_mask {
                attention_mask_flat.extend_from_slice(attention_mask);
            } else {
                attention_mask_flat.extend(vec![1; seq_len]);
            }
            
            // 处理token类型ID
            if let Some(token_type_ids) = &item.token_type_ids {
                token_type_ids_flat.extend_from_slice(token_type_ids);
            } else {
                token_type_ids_flat.extend(vec![0; seq_len]);
            }

            // 增强型图像加载逻辑：如果内存中没有图像但有路径，尝试实时加载
            let mut current_image = item.image.clone();
            if current_image.is_none() {
                if let Some(ref path) = item.image_path {
                    if let Ok(img) = image::open(path) {
                        let img_rgb = img.to_rgb8();
                        let img_resized = image::imageops::resize(&img_rgb, 224, 224, image::imageops::FilterType::Lanczos3);
                        let mut data = Vec::with_capacity(3 * 224 * 224);
                        for y in 0..224 {
                            for x in 0..224 {
                                let pixel = img_resized.get_pixel(x, y);
                                data.push(pixel[0] as f32 / 255.0);
                                data.push(pixel[1] as f32 / 255.0);
                                data.push(pixel[2] as f32 / 255.0);
                            }
                        }
                        current_image = Some(data);
                    }
                }
            }

            // 处理图像
            if let Some(image) = current_image {
                images_flat.extend_from_slice(&image);
                has_images = true;
            } else if has_images {
                // 如果这个 item 没图像但 batch 里其他 item 有，填充全 0
                images_flat.extend(vec![0.0; 3 * 224 * 224]);
            }
        }

        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs_flat, [batch_size, seq_len]),
            &self.device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets_flat, [batch_size, seq_len]),
            &self.device,
        );
        let mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_flat, [batch_size, seq_len]),
            &self.device,
        );
        let attention_mask = Tensor::<B, 2, Int>::from_data(
            TensorData::new(attention_mask_flat, [batch_size, seq_len]),
            &self.device,
        );
        let token_type_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(token_type_ids_flat, [batch_size, seq_len]),
            &self.device,
        );

        let images = if has_images {
            Some(Tensor::<B, 4>::from_data(
                TensorData::new(images_flat, [batch_size, 3, 224, 224]),
                &self.device,
            ))
        } else {
            None
        };

        TextBatch { 
            inputs, 
            targets, 
            mask, 
            attention_mask, 
            token_type_ids,
            images,
        }
    }
}

/// 数据增强工具
pub struct DataAugmenter {
    config: DataProcessingConfig,
    rng: rand::rngs::StdRng,
}

impl DataAugmenter {
    pub fn new(config: DataProcessingConfig) -> Self {
        Self {
            config,
            rng: rand::rngs::StdRng::seed_from_u64(42),
        }
    }
    
    /// 对序列进行数据增强
    pub fn augment(&mut self, tokens: &mut Vec<usize>) {
        if !self.config.enable_data_augmentation {
            return;
        }
        
        // 随机删除
        if self.rng.r#gen::<f32>() < self.config.random_deletion_probability {
            self.random_deletion(tokens);
        }
        
        // 随机插入
        if self.rng.r#gen::<f32>() < self.config.random_insertion_probability {
            self.random_insertion(tokens);
        }
        
        // 随机替换
        if self.rng.r#gen::<f32>() < self.config.random_substitution_probability {
            self.random_substitution(tokens);
        }
    }
    
    /// 随机删除一些token
    fn random_deletion(&mut self, tokens: &mut Vec<usize>) {
        let mut new_tokens = Vec::new();
        for token in tokens.iter() {
            if self.rng.r#gen::<f32>() > 0.1 { // 保留90%的token
                new_tokens.push(*token);
            }
        }
        if new_tokens.len() > 0 {
            *tokens = new_tokens;
        }
    }
    
    /// 随机插入一些token
    fn random_insertion(&mut self, tokens: &mut Vec<usize>) {
        let mut new_tokens = tokens.clone();
        let insert_count = (tokens.len() as f32 * 0.1).ceil() as usize;
        
        for _ in 0..insert_count {
            let pos = self.rng.gen_range(0..=new_tokens.len());
            // 插入一个随机token (这里使用1作为占位符，实际应该根据词汇表选择)
            new_tokens.insert(pos, 1);
        }
        
        if new_tokens.len() <= self.config.max_sequence_length {
            *tokens = new_tokens;
        }
    }
    
    /// 随机替换一些token
    fn random_substitution(&mut self, tokens: &mut Vec<usize>) {
        for token in tokens {
            if self.rng.r#gen::<f32>() < 0.1 { // 替换10%的token
                *token = 1; // 替换为占位符token
            }
        }
    }
}

/// 从JSON文件加载数据
pub fn load_from_json(path: impl AsRef<Path>) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let data: Vec<Vec<usize>> = serde_json::from_str(&content)?;
    Ok(data)
}

/// 从CSV文件加载数据
pub fn load_from_csv(path: impl AsRef<Path>) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let mut data = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let tokens: Vec<usize> = line
            .split(',')
            .filter_map(|s| s.parse().ok())
            .collect();
        if !tokens.is_empty() {
            data.push(tokens);
        }
    }
    
    Ok(data)
}

/// 数据预处理工具
pub struct DataPreprocessor {
    config: DataProcessingConfig,
}

impl DataPreprocessor {
    pub fn new(config: DataProcessingConfig) -> Self {
        Self { config }
    }
    
    /// 处理原始文本，返回token序列
    pub fn process_text(&self, text: &str, tokenizer: &crate::core::Tokenizer) -> Vec<usize> {
        let mut tokens = tokenizer.encode(text);
        
        // 截断或填充到指定长度
        if tokens.len() > self.config.seq_len {
            tokens.truncate(self.config.seq_len);
        } else if tokens.len() < self.config.seq_len {
            tokens.resize(self.config.seq_len, 0);
        }
        
        tokens
    }
    
    /// 批量处理文本
    pub fn process_batch(&self, texts: &[String], tokenizer: &crate::core::Tokenizer) -> Vec<Vec<usize>> {
        texts
            .iter()
            .map(|text| self.process_text(text, tokenizer))
            .collect()
    }
}
