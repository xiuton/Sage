use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

#[derive(Clone, Debug)]
pub struct TextItem {
    pub input: Vec<usize>,
    pub target: Vec<usize>,
}

pub struct TextDataset {
    data: Vec<usize>,
    seq_len: usize,
}

impl TextDataset {
    pub fn new(tokens: Vec<usize>, seq_len: usize) -> Self {
        Self { data: tokens, seq_len }
    }
}

impl Dataset<TextItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextItem> {
        if index + self.seq_len >= self.data.len() {
            return None;
        }

        let input = self.data[index..index + self.seq_len].to_vec();
        let target = self.data[index + 1..index + self.seq_len + 1].to_vec();

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
            .map(|item| TensorData::from(item.input.iter().map(|&i| i as i32).collect::<Vec<_>>().as_slice()))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, &self.device))
            .map(|tensor| tensor.unsqueeze::<2>())
            .collect::<Vec<_>>();

        let targets = items
            .iter()
            .map(|item| TensorData::from(item.target.iter().map(|&i| i as i32).collect::<Vec<_>>().as_slice()))
            .map(|data| Tensor::<B, 1, Int>::from_data(data, &self.device))
            .map(|tensor| tensor.unsqueeze::<2>())
            .collect::<Vec<_>>();

        let inputs = Tensor::cat(inputs, 0);
        let targets = Tensor::cat(targets, 0);

        TextBatch { inputs, targets }
    }
}
