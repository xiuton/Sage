use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPreTokenizer;
use tokenizers::tokenizer::Tokenizer as HFTokenizer;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum TokenizerType {
    Char,
    Bpe,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CharTokenizerData {
    char_to_id: HashMap<char, usize>,
    id_to_char: HashMap<usize, char>,
    vocab_size: usize,
    pad_id: usize,
    unk_id: usize,
    bos_id: usize,
    eos_id: usize,
}

#[derive(Clone, Debug)]
pub struct Tokenizer {
    tokenizer_type: TokenizerType,
    char_to_id: HashMap<char, usize>,
    id_to_char: HashMap<usize, char>,
    bpe_tokenizer: Option<HFTokenizer>,
    bpe_id_to_token: HashMap<usize, String>,
    pub vocab_size: usize,
    pub pad_id: usize,
    pub unk_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
}

impl Tokenizer {
    const SPECIAL_TOKENS: [char; 4] = ['\u{0000}', '\u{0001}', '\u{0002}', '\u{0003}'];

    fn build_char_vocab(chars: Vec<char>) -> (HashMap<char, usize>, HashMap<usize, char>) {
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();

        for (id, &c) in Self::SPECIAL_TOKENS.iter().enumerate() {
            char_to_id.insert(c, id);
            id_to_char.insert(id, c);
        }

        let mut current_id = Self::SPECIAL_TOKENS.len();
        for &c in chars.iter() {
            if let std::collections::hash_map::Entry::Vacant(e) = char_to_id.entry(c) {
                e.insert(current_id);
                id_to_char.insert(current_id, c);
                current_id += 1;
            }
        }

        (char_to_id, id_to_char)
    }

    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let (char_to_id, id_to_char) = Self::build_char_vocab(chars);
        let vocab_size = char_to_id.len();

        Self {
            tokenizer_type: TokenizerType::Char,
            char_to_id,
            id_to_char,
            bpe_tokenizer: None,
            bpe_id_to_token: HashMap::new(),
            vocab_size,
            pad_id: 0,
            unk_id: 1,
            bos_id: 2,
            eos_id: 3,
        }
    }

    pub fn new_bpe(text: &str, vocab_size: usize) -> Self {
        use tokenizers::models::ModelWrapper;
        use tokenizers::models::TrainerWrapper;
        use tokenizers::pre_tokenizers::PreTokenizerWrapper;

        let vocab = HashMap::new();
        let merges = Vec::new();
        let bpe_model = BPE::new(vocab, merges);

        let mut tokenizer = HFTokenizer::new(ModelWrapper::BPE(bpe_model));

        let byte_level_pre_tokenizer = ByteLevelPreTokenizer::new(true, true, true);
        tokenizer.with_pre_tokenizer(PreTokenizerWrapper::ByteLevel(byte_level_pre_tokenizer));

        tokenizer.with_decoder(ByteLevelDecoder::default());

        let trainer = BpeTrainer::new(2, vocab_size);

        tokenizer
            .train(&mut TrainerWrapper::BpeTrainer(trainer), text.lines())
            .expect("BPE training failed");

        let vocab_size = tokenizer.get_vocab_size(true);
        
        let bpe_id_to_token: HashMap<usize, String> = tokenizer
            .get_vocab(true)
            .into_iter()
            .map(|(tok, id)| (id as usize, tok))
            .collect();

        Self {
            tokenizer_type: TokenizerType::Bpe,
            char_to_id: HashMap::new(),
            id_to_char: HashMap::new(),
            bpe_tokenizer: Some(tokenizer),
            bpe_id_to_token,
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

        let (char_to_id, id_to_char) = Self::build_char_vocab(chars);
        let vocab_size = char_to_id.len();

        Self {
            tokenizer_type: TokenizerType::Char,
            char_to_id,
            id_to_char,
            bpe_tokenizer: None,
            bpe_id_to_token: HashMap::new(),
            vocab_size,
            pad_id: 0,
            unk_id: 1,
            bos_id: 2,
            eos_id: 3,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.tokenizer_type {
            TokenizerType::Char => text
                .chars()
                .map(|c| self.char_to_id.get(&c).copied().unwrap_or(self.unk_id))
                .collect(),
            TokenizerType::Bpe => {
                if let Some(tokenizer) = &self.bpe_tokenizer {
                    match tokenizer.encode(text, false) {
                        Ok(encoding) => encoding.get_ids().iter().map(|&id| id as usize).collect(),
                        Err(e) => {
                            eprintln!("Warning: Failed to encode text '{}': {}", text, e);
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                }
            }
        }
    }

    pub fn encode_with_assistant_mask(&self, text: &str) -> (Vec<usize>, Vec<u8>) {
        match self.tokenizer_type {
            TokenizerType::Char => {
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
            TokenizerType::Bpe => {
                let hf_tokenizer = self
                    .bpe_tokenizer
                    .as_ref()
                    .expect("BPE tokenizer should exist");

                let encoding = hf_tokenizer.encode(text, false).expect("Failed to encode text");
                let ids = encoding
                    .get_ids()
                    .iter()
                    .map(|&id| id as usize)
                    .collect::<Vec<_>>();

                // 查找所有 "助手：" 和 "用户：" 序列的位置
                let mut assistant_mask = vec![0u8; ids.len()];
                let mut in_assistant = false;

                // 直接在原始文本中查找标记，然后映射到token位置
                for (token_idx, (start, end)) in encoding.get_offsets().iter().enumerate() {
                    let token_text = &text[*start..*end];

                    // 检查token是否包含完整的标记序列
                    if token_text.contains("用户：") {
                        in_assistant = false;
                    } else if token_text.contains("助手：") {
                        in_assistant = true;
                    }

                    // 检查token是否是标记序列的一部分
                    if *end >= 3 {
                        let potential_mark = &text[(*end).saturating_sub(3)..*end];
                        if potential_mark == "用户：" {
                            in_assistant = false;
                        } else if potential_mark == "助手：" {
                            in_assistant = true;
                        }
                    }

                    assistant_mask[token_idx] = if in_assistant { 1 } else { 0 };
                }

                (ids, assistant_mask)
            }
        }
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        match self.tokenizer_type {
            TokenizerType::Char => ids
                .iter()
                .filter(|&&id| id != self.pad_id && id != self.bos_id && id != self.eos_id)
                .filter_map(|id| self.id_to_char.get(id).copied())
                .collect(),
            TokenizerType::Bpe => {
                if let Some(tokenizer) = &self.bpe_tokenizer {
                    let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
                    tokenizer.decode(&ids_u32, true).unwrap_or_default()
                } else {
                    String::new()
                }
            }
        }
    }

    pub fn token_for_id(&self, id: usize) -> Option<String> {
        match self.tokenizer_type {
            TokenizerType::Char => self.id_to_char.get(&id).map(|c| c.to_string()),
            TokenizerType::Bpe => self.bpe_id_to_token.get(&id).cloned(),
        }
    }

    pub fn char_for_id(&self, id: usize) -> Option<char> {
        match self.tokenizer_type {
            TokenizerType::Char => self.id_to_char.get(&id).copied(),
            TokenizerType::Bpe => None,
        }
    }

    pub fn is_punctuation_token(&self, id: usize) -> bool {
        if let Some(token) = self.token_for_id(id) {
            token.chars().any(is_punctuation_like)
        } else {
            false
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        match self.tokenizer_type {
            TokenizerType::Char => {
                let data = CharTokenizerData {
                    char_to_id: self.char_to_id.clone(),
                    id_to_char: self.id_to_char.clone(),
                    vocab_size: self.vocab_size,
                    pad_id: self.pad_id,
                    unk_id: self.unk_id,
                    bos_id: self.bos_id,
                    eos_id: self.eos_id,
                };
                let json = serde_json::to_string(&data).expect("Should serialize tokenizer");
                std::fs::write(path, json)
            }
            TokenizerType::Bpe => {
                if let Some(tokenizer) = &self.bpe_tokenizer {
                    tokenizer
                        .save(path, false)
                        .expect("Should save BPE tokenizer");
                }
                let mut meta = HashMap::new();
                meta.insert("tokenizer_type", "bpe");
                let vocab_size_str = self.vocab_size.to_string();
                meta.insert("vocab_size", &vocab_size_str);
                let meta_path = format!("{}.meta", path);
                std::fs::write(meta_path, serde_json::to_string(&meta).unwrap())
            }
        }
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let meta_path = format!("{}.meta", path);
        if Path::new(&meta_path).exists() {
            let meta_text = std::fs::read_to_string(&meta_path)?;
            let meta_value: serde_json::Value = serde_json::from_str(&meta_text)?;
            if meta_value.get("tokenizer_type").and_then(|v| v.as_str()) == Some("bpe") {
                let tokenizer = HFTokenizer::from_file(path).expect("Should load BPE tokenizer");
                let vocab_size = tokenizer.get_vocab_size(true);
                
                let bpe_id_to_token: HashMap<usize, String> = tokenizer
                    .get_vocab(true)
                    .into_iter()
                    .map(|(tok, id)| (id as usize, tok))
                    .collect();
                
                return Ok(Self {
                    tokenizer_type: TokenizerType::Bpe,
                    char_to_id: HashMap::new(),
                    id_to_char: HashMap::new(),
                    bpe_tokenizer: Some(tokenizer),
                    bpe_id_to_token,
                    vocab_size,
                    pad_id: 0,
                    unk_id: 1,
                    bos_id: 2,
                    eos_id: 3,
                });
            }
        }

        let json = std::fs::read_to_string(path)?;
        let raw: CharTokenizerData =
            serde_json::from_str(&json).expect("Should deserialize tokenizer");
        Ok(Self {
            tokenizer_type: TokenizerType::Char,
            char_to_id: raw.char_to_id,
            id_to_char: raw.id_to_char,
            bpe_tokenizer: None,
            bpe_id_to_token: HashMap::new(),
            vocab_size: raw.vocab_size,
            pad_id: raw.pad_id,
            unk_id: raw.unk_id,
            bos_id: raw.bos_id,
            eos_id: raw.eos_id,
        })
    }
}

fn is_punctuation_like(ch: char) -> bool {
    if ch.is_ascii_punctuation() {
        return true;
    }
    matches!(
        ch,
        '，' | '。'
            | '、'
            | '；'
            | '：'
            | '！'
            | '？'
            | '…'
            | '—'
            | '·'
            | '（'
            | '）'
            | '《'
            | '》'
            | '“'
            | '”'
            | '‘'
            | '’'
            | '【'
            | '】'
            | '〔'
            | '〕'
            | '『'
            | '』'
            | '「'
            | '」'
            | '\n'
            | '\r'
            | '\t'
    )
}
