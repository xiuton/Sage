use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
        let special_tokens = ['\u{0000}', '\u{0001}', '\u{0002}', '\u{0003}'];
        // 0: <pad>, 1: <unk>, 2: <bos>, 3: <eos>

        for (id, &c) in special_tokens.iter().enumerate() {
            char_to_id.insert(c, id);
            id_to_char.insert(id, c);
        }

        let mut current_id = special_tokens.len();
        for &c in chars.iter() {
            if let std::collections::hash_map::Entry::Vacant(e) = char_to_id.entry(c) {
                e.insert(current_id);
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

    pub fn from_chars(mut chars: Vec<char>) -> Self {
        chars.sort();
        chars.dedup();

        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        let special_tokens = ['\u{0000}', '\u{0001}', '\u{0002}', '\u{0003}'];

        for (id, &c) in special_tokens.iter().enumerate() {
            char_to_id.insert(c, id);
            id_to_char.insert(id, c);
        }

        let mut current_id = special_tokens.len();
        for &c in chars.iter() {
            if let std::collections::hash_map::Entry::Vacant(e) = char_to_id.entry(c) {
                e.insert(current_id);
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

    pub fn encode_with_assistant_mask(&self, text: &str) -> (Vec<usize>, Vec<u8>) {
        let mut tokens = Vec::new();
        let mut mask = Vec::new();

        let mut last3: [char; 3] = ['\0', '\0', '\0'];
        let mut assistant = false;

        for ch in text.chars() {
            let id = self.char_to_id.get(&ch).copied().unwrap_or(self.unk_id);
            tokens.push(id);

            if id == self.eos_id {
                mask.push(1);
                assistant = false;
                last3 = ['\0', '\0', '\0'];
                continue;
            }

            last3 = [last3[1], last3[2], ch];

            mask.push(if assistant { 1 } else { 0 });

            if last3 == ['助', '手', '：'] {
                let len = mask.len();
                mask[len - 3] = 0;
                mask[len - 2] = 0;
                mask[len - 1] = 0;
                assistant = true;
            } else if last3 == ['用', '户', '：'] {
                let len = mask.len();
                mask[len - 3] = 0;
                mask[len - 2] = 0;
                mask[len - 1] = 0;
                assistant = false;
            }
        }

        (tokens, mask)
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id != self.pad_id && id != self.bos_id && id != self.eos_id)
            .filter_map(|id| self.id_to_char.get(id).copied())
            .collect()
    }

    pub fn char_for_id(&self, id: usize) -> Option<char> {
        self.id_to_char.get(&id).copied()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self).expect("Should serialize tokenizer");
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let tokenizer: Tokenizer =
            serde_json::from_str(&json).expect("Should deserialize tokenizer");
        Ok(tokenizer)
    }
}
