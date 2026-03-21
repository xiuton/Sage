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

        let input = self.data[index..index + self.seq_len]
            .iter()
            .map(|&v| v as i32)
            .collect::<Vec<_>>();

        let mut target = Vec::with_capacity(self.seq_len);
        for j in 0..self.seq_len {
            let token_id = self.data[index + 1 + j] as i32;
            let m = *self.mask.get(index + 1 + j).unwrap_or(&1);
            if m == 1 {
                target.push(token_id);
            } else {
                target.push(0);
            }
        }

        Some(TextItem { input, target })
    }

    fn len(&self) -> usize {
        if self.data.len() <= self.seq_len {
            0
        } else {
            self.data.len() - self.seq_len - 1
        }
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
        let tokens_path = tokens_path.as_ref().to_path_buf();
        let mask_path = mask_path.as_ref().to_path_buf();

        let tokens_file = File::open(&tokens_path).expect("Open tokens file");
        let mask_file = File::open(&mask_path).expect("Open mask file");

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
        let b0 = self.tokens[off] as u32;
        let b1 = (self.tokens[off + 1] as u32) << 8;
        let b2 = (self.tokens[off + 2] as u32) << 16;
        let b3 = (self.tokens[off + 3] as u32) << 24;
        (b0 | b1 | b2 | b3) as usize
    }

    fn mask_at(&self, index: usize) -> u8 {
        let i = self.start + index;
        self.mask[i]
    }
}

impl Dataset<TextItem> for MmapTextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        let total = self.end.saturating_sub(self.start);
        if index + self.seq_len >= total {
            return None;
        }

        let mut input = Vec::with_capacity(self.seq_len);
        for j in 0..self.seq_len {
            input.push(self.token_at(index + j) as i32);
        }

        let mut target = Vec::with_capacity(self.seq_len);
        for j in 0..self.seq_len {
            let token_id = self.token_at(index + 1 + j) as i32;
            let m = self.mask_at(index + 1 + j);
            if m == 1 {
                target.push(token_id);
            } else {
                target.push(0);
            }
        }

        Some(TextItem { input, target })
    }

    fn len(&self) -> usize {
        let total = self.end.saturating_sub(self.start);
        if total <= self.seq_len {
            0
        } else {
            total - self.seq_len - 1
        }
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
        let inputs = items
            .iter()
            .map(|item| TensorData::from(item.input.as_slice()))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, &self.device))
            .map(|tensor| tensor.unsqueeze::<2>())
            .collect::<Vec<_>>();

        let targets = items
            .iter()
            .map(|item| TensorData::from(item.target.as_slice()))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, &self.device))
            .map(|tensor| tensor.unsqueeze::<2>())
            .collect::<Vec<_>>();

        let inputs = Tensor::cat(inputs, 0);
        let targets = Tensor::cat(targets, 0);

        TextBatch { inputs, targets }
    }
}
