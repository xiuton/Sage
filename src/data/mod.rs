//! Canonical data layer for datasets, batching, and preprocessing.

pub mod data;

pub use data::{MmapTextDataset, TextBatch, TextBatcher, TextDataset};
