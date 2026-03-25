use axum::{
    routing::{get, post},
    Router,
};
use reqwest::{Client, StatusCode};
use serde_json::json;
use std::time::Duration;
use tokio::net::TcpListener;

async fn start_test_server() -> (String, impl Fn() -> ()) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let addr = format!("http://127.0.0.1:{}", port);
    
    let app = Router::new()
        .route("/api/health", get(health_check))
        .route("/api/model-info", get(model_info))
        .route("/v1/chat/completions", post(chat_completion))
        .route("/v1/batch-chat/completions", post(batch_chat_completion))
        .route("/v1/async-chat/completions", post(async_chat_completion))
        .route("/api/task/:task_id", get(get_task));
    
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("Server error: {}", e);
        }
    });
    
    (addr, move || {
        handle.abort();
    })
}

async fn health_check() -> axum::Json<serde_json::Value> {
    axum::Json(json!({"status": "ok"}))
}

async fn model_info() -> axum::Json<serde_json::Value> {
    axum::Json(json!({
        "model_name": "test-model",
        "model_size": "30m",
        "backend": "cpu"
    }))
}

async fn chat_completion(axum::Json(req): axum::Json<serde_json::Value>) -> axum::Json<serde_json::Value> {
    let messages = req["messages"].as_array().unwrap();
    let content = messages.last().unwrap()["content"].as_str().unwrap();
    
    axum::Json(json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "sage-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": format!("回复: {}", content)
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }))
}

async fn batch_chat_completion(axum::Json(req): axum::Json<serde_json::Value>) -> axum::Json<serde_json::Value> {
    let requests = req["requests"].as_array().unwrap();
    let mut responses = Vec::new();
    
    for (i, req) in requests.iter().enumerate() {
        let content = req["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap();
        responses.push(json!({
            "id": format!("chatcmpl-batch-{}", i),
            "object": "chat.completion",
            "created": 1677858242,
            "model": "sage-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": format!("批量回复: {}", content)
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }));
    }
    
    axum::Json(json!({
        "responses": responses,
        "total_duration_ms": 100,
        "request_count": responses.len()
    }))
}

async fn async_chat_completion(axum::Json(_req): axum::Json<serde_json::Value>) -> axum::Json<serde_json::Value> {
    axum::Json(json!({
        "task_id": "task-test-123",
        "status": "Pending",
        "result": null,
        "error": null
    }))
}

async fn get_task(axum::extract::Path(task_id): axum::extract::Path<String>) -> axum::Json<serde_json::Value> {
    axum::Json(json!({
        "task_id": task_id,
        "status": "Completed",
        "result": {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "sage-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "异步回复内容"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        },
        "error": null,
        "created_at": 1000,
        "started_at": 1500,
        "completed_at": 2000
    }))
}

#[tokio::test]
async fn test_health_check() {
    let (addr, cleanup) = start_test_server().await;
    
    let client = Client::new();
    let response = client.get(&format!("{}/api/health", addr)).send().await.unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    
    cleanup();
}

#[tokio::test]
async fn test_model_info() {
    let (addr, cleanup) = start_test_server().await;
    
    let client = Client::new();
    let response = client.get(&format!("{}/api/model-info", addr)).send().await.unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["model_name"], "test-model");
    assert_eq!(body["model_size"], "30m");
    
    cleanup();
}

#[tokio::test]
async fn test_chat_completion() {
    let (addr, cleanup) = start_test_server().await;
    
    let client = Client::new();
    let request_body = json!({
        "model": "sage-model",
        "messages": [
            {"role": "user", "content": "测试消息"}
        ],
        "max_tokens": 10
    });
    
    let response = client
        .post(&format!("{}/v1/chat/completions", addr))
        .json(&request_body)
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["content"], "回复: 测试消息");
    
    cleanup();
}

#[tokio::test]
async fn test_batch_chat_completion() {
    let (addr, cleanup) = start_test_server().await;
    
    let client = Client::new();
    let request_body = json!({
        "requests": [
            {
                "model": "sage-model",
                "messages": [{"role": "user", "content": "测试1"}],
                "max_tokens": 5
            },
            {
                "model": "sage-model",
                "messages": [{"role": "user", "content": "测试2"}],
                "max_tokens": 5
            }
        ]
    });
    
    let response = client
        .post(&format!("{}/v1/batch-chat/completions", addr))
        .json(&request_body)
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["responses"].as_array().unwrap().len(), 2);
    assert_eq!(body["responses"][0]["choices"][0]["message"]["content"], "批量回复: 测试1");
    
    cleanup();
}

#[tokio::test]
async fn test_async_chat_completion() {
    let (addr, cleanup) = start_test_server().await;
    
    let client = Client::new();
    let request_body = json!({
        "model": "sage-model",
        "messages": [
            {"role": "user", "content": "异步测试"}
        ],
        "max_tokens": 10
    });
    
    let response = client
        .post(&format!("{}/v1/async-chat/completions", addr))
        .json(&request_body)
        .send()
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["task_id"], "task-test-123");
    assert_eq!(body["status"], "Pending");
    
    let task_id = body["task_id"].as_str().unwrap();
    
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    let task_response = client
        .get(&format!("{}/api/task/{}", addr, task_id))
        .send()
        .await
        .unwrap();
    
    assert_eq!(task_response.status(), StatusCode::OK);
    
    let task_body: serde_json::Value = task_response.json().await.unwrap();
    assert_eq!(task_body["task_id"], task_id);
    assert_eq!(task_body["status"], "Completed");
    
    cleanup();
}

#[tokio::test]
async fn test_api_endpoints_exist() {
    let (addr, cleanup) = start_test_server().await;
    
    let client = Client::new();
    
    let endpoints = vec![
        "/api/health",
        "/api/model-info",
        "/v1/chat/completions",
        "/v1/batch-chat/completions",
        "/v1/async-chat/completions",
    ];
    
    for endpoint in endpoints {
        let response = client.get(&format!("{}{}", addr, endpoint)).send().await.unwrap();
        assert!(response.status().is_success() || response.status() == StatusCode::METHOD_NOT_ALLOWED);
    }
    
    cleanup();
}
