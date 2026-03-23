use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use memmap2::Mmap;
use std::{fs::File, path::Path, sync::Arc};

#[derive(Clone, Debug)]
pub struct TextItem {
    pub input: Vec<i32>,
    pub target: Vec<i32>,
}

pub struct TextDataset {
    data: Vec<usize>,
    mask: Vec<u8>,
    seq_len: usize,
}

impl TextDataset {
    pub fn new(tokens: Vec<usize>, mask: Vec<u8>, seq_len: usize) -> Self {
        Self {
            data: tokens,
            mask,
            seq_len,
        }
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
        for j in 0..self.seq_len {
            let token_id = self.data[index + 1 + j] as i32;
            let m = *self.mask.get(index + 1 + j).unwrap_or(&1);
            target.push(if m == 1 { token_id } else { 0 });
        }

        Some(TextItem { input, target })
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
        let tokens_file = File::open(tokens_path).expect("Open tokens file");
        let mask_file = File::open(mask_path).expect("Open mask file");

        let tokens = Arc::new(unsafe { Mmap::map(&tokens_file).expect("Mmap tokens") });
        let mask = Arc::new(unsafe { Mmap::map(&mask_file).expect("Mmap mask") });

        let token_len = tokens.len() / 4;
        let mask_len = mask.len();
        let end = token_len.min(mask_len);

        Self {
            tokens,
            mask,
            start: 0,
            end,
            seq_len,
        }
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

        for j in 0..self.seq_len {
            input.push(self.token_at(index + j) as i32);
            let token_id = self.token_at(index + 1 + j) as i32;
            let m = self.mask_at(index + 1 + j);
            target.push(if m == 1 { token_id } else { 0 });
        }

        Some(TextItem { input, target })
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
}

impl<B: Backend> Batcher<TextItem, TextBatch<B>> for TextBatcher<B> {
    fn batch(&self, items: Vec<TextItem>) -> TextBatch<B> {
        let batch_size = items.len();
        let seq_len = items.first().map(|v| v.input.len()).unwrap_or(0);

        let mut inputs_flat = Vec::with_capacity(batch_size * seq_len);
        let mut targets_flat = Vec::with_capacity(batch_size * seq_len);

        for item in items.iter() {
            inputs_flat.extend_from_slice(&item.input);
            targets_flat.extend_from_slice(&item.target);
        }

        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs_flat, [batch_size, seq_len]),
            &self.device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets_flat, [batch_size, seq_len]),
            &self.device,
        );

        TextBatch { inputs, targets }
    }
}
