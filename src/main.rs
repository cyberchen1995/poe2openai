use salvo::prelude::*;
use std::env;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tracing::{debug, info};

mod cache;
mod evert;
mod handlers;
mod poe_client;
mod types;
mod utils;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn get_env_or_default(key: &str, default: &str) -> String {
    let value = env::var(key).unwrap_or_else(|_| default.to_string());
    if key == "ADMIN_PASSWORD" {
        debug!("🔧 環境變數 {} = {}", key, "*".repeat(value.len()));
    } else {
        debug!("🔧 環境變數 {} = {}", key, value);
    }
    value
}

fn setup_logging(log_level: &str) {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true)
        .with_file(false)
        .with_line_number(false)
        .with_env_filter(log_level)
        .init();
    info!("🚀 日誌系統初始化完成，日誌級別: {}", log_level);
}

fn log_cache_settings() {
    // 記錄緩存相關設定
    let cache_ttl_seconds = std::env::var("URL_CACHE_TTL_SECONDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(3 * 24 * 60 * 60);
    let cache_size_mb = std::env::var("URL_CACHE_SIZE_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100);

    let ttl_days = cache_ttl_seconds / 86400;
    let ttl_hours = (cache_ttl_seconds % 86400) / 3600;
    let ttl_mins = (cache_ttl_seconds % 3600) / 60;
    let ttl_secs = cache_ttl_seconds % 60;

    let ttl_str = if ttl_days > 0 {
        format!(
            "{}天 {}小時 {}分 {}秒",
            ttl_days, ttl_hours, ttl_mins, ttl_secs
        )
    } else if ttl_hours > 0 {
        format!("{}小時 {}分 {}秒", ttl_hours, ttl_mins, ttl_secs)
    } else if ttl_mins > 0 {
        format!("{}分 {}秒", ttl_mins, ttl_secs)
    } else {
        format!("{}秒", ttl_secs)
    };

    info!(
        "📦 Poe CDN URL 緩存設定 | TTL: {} | 最大空間: {}MB",
        ttl_str, cache_size_mb
    );
}

#[tokio::main]
async fn main() {
    let log_level = get_env_or_default("LOG_LEVEL", "debug");
    setup_logging(&log_level);

    // 初始化緩存設定
    log_cache_settings();

    // 初始化全域速率限制
    let _ = handlers::limit::GLOBAL_RATE_LIMITER.set(Arc::new(tokio::sync::Mutex::new(
        std::time::Instant::now() - Duration::from_secs(60),
    )));

    // 顯示速率限制設定
    let rate_limit_ms = std::env::var("RATE_LIMIT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(100);

    if rate_limit_ms == 0 {
        info!("⚙️  全域速率限制: 已禁用 (RATE_LIMIT_MS=0)");
    } else {
        info!("⚙️  全域速率限制: 已啟用 (每 {}ms 一次請求)", rate_limit_ms);
    }

    let host = get_env_or_default("HOST", "0.0.0.0");
    let port = get_env_or_default("PORT", "8080");
    get_env_or_default("ADMIN_USERNAME", "admin");
    get_env_or_default("ADMIN_PASSWORD", "123456");
    let config_dir = get_env_or_default("CONFIG_DIR", "./");
    let config_path = Path::new(&config_dir).join("models.yaml");
    info!("📁 配置文件路徑: {}", config_path.display());

    let salvo_max_size = get_env_or_default("MAX_REQUEST_SIZE", "1073741824")
        .parse()
        .unwrap_or(1024 * 1024 * 1024); // 預設 1GB

    let bind_address = format!("{}:{}", host, port);
    info!("🌟 正在啟動 Poe API To OpenAI API 服務...");
    debug!("📍 服務綁定地址: {}", bind_address);

    // 初始化Sled DB
    let _ = cache::get_sled_db();
    info!("💾 初始化內存數據庫完成");

    let api_router = Router::new()
        .hoop(handlers::cors_middleware)
        .push(
            Router::with_path("models")
                .get(handlers::get_models)
                .options(handlers::cors_middleware),
        )
        .push(
            Router::with_path("chat/completions")
                .hoop(handlers::rate_limit_middleware)
                .post(handlers::chat_completions)
                .options(handlers::cors_middleware),
        )
        .push(
            Router::with_path("api/models")
                .get(handlers::get_models)
                .options(handlers::cors_middleware),
        )
        .push(
            Router::with_path("v1/models")
                .get(handlers::get_models)
                .options(handlers::cors_middleware),
        )
        .push(
            Router::with_path("v1/chat/completions")
                .hoop(handlers::rate_limit_middleware)
                .post(handlers::chat_completions)
                .options(handlers::cors_middleware),
        );

    let router: Router = Router::new()
        .hoop(max_size(salvo_max_size.try_into().unwrap()))
        .push(Router::with_path("static/{**path}").get(StaticDir::new(["static"])))
        .push(handlers::admin_routes())
        .push(api_router);

    info!("🛣️  API 路由配置完成");

    let acceptor = TcpListener::new(bind_address.clone()).bind().await;
    info!("🎯 服務已啟動並監聽於 {}", bind_address);

    Server::new(acceptor).serve(router).await;
}
