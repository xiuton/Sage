use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Tokenizer {
    char_to_id: HashMap<char, usize>,
    id_to_char: HashMap<usize, char>,
    pub vocab_size: usize,
    pub pad_id: usize,
    pub unk_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
}

impl Tokenizer {
    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        // 预定义特殊标记
        let special_tokens = vec!['\u{0000}', '\u{0001}', '\u{0002}', '\u{0003}'];
        // 0: <pad>, 1: <unk>, 2: <bos>, 3: <eos>
        
        for (id, &c) in special_tokens.iter().enumerate() {
            char_to_id.insert(c, id);
            id_to_char.insert(id, c);
        }

        let mut current_id = special_tokens.len();
        for &c in chars.iter() {
            if !char_to_id.contains_key(&c) {
                char_to_id.insert(c, current_id);
                id_to_char.insert(current_id, c);
                current_id += 1;
            }
        }

        let vocab_size = char_to_id.len();

        Self {
            char_to_id,
            id_to_char,
            vocab_size,
            pad_id: 0,
            unk_id: 1,
            bos_id: 2,
            eos_id: 3,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| self.char_to_id.get(&c).copied().unwrap_or(self.unk_id))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id != self.pad_id && id != self.bos_id && id != self.eos_id)
            .filter_map(|id| self.id_to_char.get(id).copied())
            .collect()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self).expect("Should serialize tokenizer");
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let tokenizer: Tokenizer = serde_json::from_str(&json).expect("Should deserialize tokenizer");
        Ok(tokenizer)
    }
}
