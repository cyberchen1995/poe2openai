# 第一階段：建構環境
FROM rust:1.89.0 AS builder

# 設定建構時的環境變數
ENV CARGO_TERM_COLOR=always \
    CARGO_NET_GIT_FETCH_WITH_CLI=true \
    CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

# 安裝建構依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /usr/src/app

# 複製 Cargo.toml
COPY Cargo.toml ./

# 建立虛擬的 src 目錄和主檔案以緩存依賴
RUN mkdir src

# 複製實際的源碼和資源文件
COPY src ./src
COPY templates ./templates
COPY static ./static

# 重新建構專案
RUN cargo build --release

# 第二階段：執行環境
FROM debian:13-slim

# 設定執行時的環境變數
ENV HOST=0.0.0.0 \
    PORT=8080 \
    ADMIN_USERNAME=admin \
    ADMIN_PASSWORD=123456 \
    MAX_REQUEST_SIZE=1073741824 \
    LOG_LEVEL=info \
    RUST_BACKTRACE=1 \
    TZ=Asia/Taipei \
    CONFIG_DIR=/data

# 安裝執行時期依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# 建立應用程式目錄
WORKDIR /app

# 從建構階段複製編譯好的二進制檔案和資源文件
COPY --from=builder /usr/src/app/target/release/poe2openai /app/
COPY --from=builder /usr/src/app/templates /app/templates
COPY --from=builder /usr/src/app/static /app/static

# 創建數據目錄
RUN mkdir -p /data && chmod 777 /data

# 定義volume掛載點
VOLUME ["/data"]

# 設定容器啟動指令
ENTRYPOINT ["/app/poe2openai"]

# 暴露端口
EXPOSE ${PORT}

# 設定標籤
LABEL maintainer="Jerome Leong <jeromeleong1998@gmail.com>" \
    description="Poe API to OpenAI API 轉換服務" \
    version="0.7.5"