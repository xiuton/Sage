use sage::utils::error::{SageError, Result};

#[test]
fn test_io_error_conversion() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "文件未找到");
    let sage_err: SageError = io_err.into();
    let msg = format!("{}", sage_err);
    assert!(msg.contains("IO错误"));
    assert!(msg.contains("文件未找到"));
}

#[test]
fn test_serde_error_conversion() {
    let json_err = serde_json::from_str::<serde_json::Value>("{invalid}").unwrap_err();
    let sage_err: SageError = json_err.into();
    let msg = format!("{}", sage_err);
    assert!(msg.contains("序列化错误"));
}

#[test]
fn test_string_conversion() {
    let sage_err: SageError = "配置解析失败: 缺少 batch_size 字段".to_string().into();
    let msg = format!("{}", sage_err);
    assert!(msg.contains("配置错误"));
    assert!(msg.contains("batch_size"));
}

#[test]
fn test_model_loading_error() {
    let err = SageError::model_loading("模型文件损坏", Some("model.safetensors".to_string()));
    let msg = format!("{}", err);
    assert!(msg.contains("模型加载错误"));
    assert!(msg.contains("model.safetensors"));
    assert!(msg.contains("模型文件损坏"));
}

#[test]
fn test_tokenizer_error() {
    let err = SageError::tokenizer("无法编码中文字符", Some("你好世界".to_string()));
    let msg = format!("{}", err);
    assert!(msg.contains("分词器错误"));
    assert!(msg.contains("无法编码中文字符"));
}

#[test]
fn test_inference_error() {
    let err = SageError::inference(
        "张量形状不匹配",
        Some("[1, 512, 768]".to_string()),
        Some("期望 [1, 100, 768]".to_string()),
    );
    let msg = format!("{}", err);
    assert!(msg.contains("推理错误"));
    assert!(msg.contains("张量形状不匹配"));
    assert!(msg.contains("[1, 512, 768]"));
}

#[test]
fn test_configuration_error() {
    let err = SageError::configuration("校验失败: batch_size 必须大于 0", Some("config.json".to_string()));
    let msg = format!("{}", err);
    assert!(msg.contains("配置错误"));
    assert!(msg.contains("config.json"));
    assert!(msg.contains("batch_size"));
}

#[test]
fn test_backend_error() {
    let err = SageError::backend("WGPU 初始化失败", "gpu");
    let msg = format!("{}", err);
    assert!(msg.contains("后端错误"));
    assert!(msg.contains("WGPU 初始化失败"));
    assert!(msg.contains("gpu"));
}

#[test]
fn test_training_error() {
    let err = SageError::training("GPU 显存不足", Some(1), Some(10));
    let msg = format!("{}", err);
    assert!(msg.contains("训练错误"));
    assert!(msg.contains("GPU 显存不足"));
    assert!(msg.contains("当前轮次: 1"));
    assert!(msg.contains("当前批次: 10"));
}

#[test]
fn test_validation_error() {
    let err = SageError::validation(
        "模型输出包含 NaN",
        Some("loss".to_string()),
        Some("数值有效".to_string()),
    );
    let msg = format!("{}", err);
    assert!(msg.contains("验证错误"));
    assert!(msg.contains("模型输出包含 NaN"));
    assert!(msg.contains("loss"));
}

#[test]
fn test_error_is_std_error() {
    let err = SageError::configuration("测试错误", Some("test.json".to_string()));
    let _: &dyn std::error::Error = &err;
}

#[test]
fn test_error_into_box() {
    let err = SageError::model_loading("测试", None::<String>);
    let _boxed: Box<dyn std::error::Error> = err.into();
}

#[test]
fn test_result_type_alias() {
    let ok: Result<i32> = Ok(42);
    assert_eq!(ok.unwrap(), 42);

    let err: Result<i32> = Err(SageError::configuration("测试", Some("config.json".to_string())));
    assert!(err.is_err());
}

#[test]
fn test_error_context_trait() {
    use sage::utils::error::ErrorContext;

    let result: Result<()> = Err(SageError::configuration("原始错误", Some("config.json".to_string())));
    let with_context = result.context("加载训练参数时");
    let msg = format!("{}", with_context.unwrap_err());
    assert!(msg.contains("上下文: 加载训练参数时"));
    assert!(msg.contains("原始错误"));
}
