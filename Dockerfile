# 多阶段构建
FROM rust:1.75 as builder

WORKDIR /app

# 复制项目文件
COPY . .

# 构建发布版本
RUN cargo build --release --bin api_server

# 运行阶段
FROM debian:bookworm-slim

WORKDIR /app

# 安装必要的依赖
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制二进制文件
COPY --from=builder /app/target/release/api_server /app/api_server

# 创建模型目录
RUN mkdir -p /app/models

# 设置环境变量
ENV RUST_LOG=info

# 暴露端口
EXPOSE 8000

# 启动API服务器
CMD ["/app/api_server", "--model-dir", "/app/models", "--port", "8000"]
