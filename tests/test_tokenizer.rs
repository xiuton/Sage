use sage::tokenizer::Tokenizer;

#[test]
fn test_tokenizer_new() {
    // 使用new方法创建一个简单的字符分词器
    let tokenizer = Tokenizer::new("hello world");
    
    assert!(tokenizer.vocab_size > 0);
    assert_eq!(tokenizer.pad_id, 0);
    assert_eq!(tokenizer.unk_id, 1);
    assert_eq!(tokenizer.bos_id, 2);
    assert_eq!(tokenizer.eos_id, 3);
}

#[test]
fn test_encode_and_decode() {
    let tokenizer = Tokenizer::new("hello world");
    
    // 测试encode
    let tokens = tokenizer.encode("hello");
    assert!(!tokens.is_empty());
    
    // 测试decode
    let text = tokenizer.decode(&tokens);
    assert!(!text.is_empty());
}

#[test]
fn test_token_for_id() {
    let tokenizer = Tokenizer::new("test");
    
    // 测试获取特殊token
    assert!(tokenizer.token_for_id(tokenizer.pad_id).is_some());
    assert!(tokenizer.token_for_id(tokenizer.unk_id).is_some());
}

#[test]
fn test_char_for_id() {
    let tokenizer = Tokenizer::new("abc");
    
    // 测试字符映射
    let c = tokenizer.char_for_id(tokenizer.bos_id);
    assert!(c.is_some());
}

#[test]
fn test_is_punctuation_token() {
    let tokenizer = Tokenizer::new("test,.;");
    
    // 测试标点符号检测
    let tokens = tokenizer.encode(",");
    if let Some(token_id) = tokens.first() {
        assert!(tokenizer.is_punctuation_token(*token_id));
    }
}
