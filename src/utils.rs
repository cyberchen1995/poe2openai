use crate::poe_client::PoeClientWrapper;
use crate::types::{Config, ImageUrlContent, Message, OpenAiContent, OpenAiContentItem};
use crate::types::{OpenAIError, OpenAIErrorResponse};
use base64::prelude::*;
use nanoid::nanoid;
use poe_api_process::FileUploadRequest;
use salvo::http::StatusCode;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use tiktoken_rs::o200k_base;
use tracing::{debug, error, info, warn};

// 處理消息中的文件/圖片
pub async fn process_message_images(
    poe_client: &PoeClientWrapper,
    messages: &mut [Message],
) -> Result<(), Box<dyn std::error::Error>> {
    // 收集需要處理的URL
    let mut external_urls = Vec::new();
    let mut data_urls = Vec::new();
    let mut url_indices = Vec::new();
    let mut data_url_indices = Vec::new();
    let mut temp_files: Vec<PathBuf> = Vec::new();

    // 收集消息中所有需要處理的URL
    for (msg_idx, message) in messages.iter().enumerate() {
        if let Some(OpenAiContent::Multi(items)) = &message.content {
            for (item_idx, item) in items.iter().enumerate() {
                if let OpenAiContentItem::ImageUrl { image_url } = item {
                    if image_url.url.starts_with("data:") {
                        // 處理data URL
                        debug!("🔍 發現data URL");
                        data_urls.push(image_url.url.clone());
                        data_url_indices.push((msg_idx, item_idx));
                    } else if !is_poe_cdn_url(&image_url.url) {
                        // 處理需要上傳的外部URL
                        debug!("🔍 發現需要上傳的外部URL: {}", image_url.url);
                        external_urls.push(image_url.url.clone());
                        url_indices.push((msg_idx, item_idx));
                    }
                }
            }
        }
    }

    // 處理外部URL
    if !external_urls.is_empty() {
        debug!("🔄 準備處理 {} 個外部URL", external_urls.len());

        // 將外部URL分為緩存命中和未命中兩組
        let mut urls_to_upload = Vec::new();
        let mut urls_indices_to_upload = Vec::new();

        for (idx, (msg_idx, item_idx)) in url_indices.iter().enumerate() {
            let url = &external_urls[idx];

            // 檢查緩存
            if let Some((poe_url, _)) = crate::cache::get_cached_url(url) {
                debug!("✅ URL緩存命中: {} -> {}", url, poe_url);

                if let Some(OpenAiContent::Multi(items)) = &mut messages[*msg_idx].content {
                    if let OpenAiContentItem::ImageUrl { image_url } = &mut items[*item_idx] {
                        debug!("🔄 從緩存替換URL: {}", poe_url);
                        image_url.url = poe_url;
                    }
                }
            } else {
                // 緩存未命中，需要上傳
                debug!("❌ URL緩存未命中: {}", url);
                urls_to_upload.push(url.clone());
                urls_indices_to_upload.push((*msg_idx, *item_idx));
            }
        }

        // 上傳未緩存的URL
        if !urls_to_upload.is_empty() {
            debug!("🔄 上傳 {} 個未緩存的URL", urls_to_upload.len());

            let upload_requests: Vec<FileUploadRequest> = urls_to_upload
                .iter()
                .map(|url| FileUploadRequest::RemoteFile {
                    download_url: url.clone(),
                })
                .collect();

            match poe_client.client.upload_files_batch(upload_requests).await {
                Ok(responses) => {
                    debug!("✅ 成功上傳 {} 個外部URL", responses.len());

                    // 更新緩存並保存URL映射
                    for (idx, ((msg_idx, item_idx), response)) in urls_indices_to_upload
                        .iter()
                        .zip(responses.iter())
                        .enumerate()
                    {
                        let original_url = &urls_to_upload[idx];

                        // 估算大小 (默認1MB，實際使用中可以優化)
                        let size_bytes = 1024 * 1024;

                        // 添加到緩存
                        crate::cache::cache_url(original_url, &response.attachment_url, size_bytes);

                        if let Some(OpenAiContent::Multi(items)) = &mut messages[*msg_idx].content {
                            if let OpenAiContentItem::ImageUrl { image_url } = &mut items[*item_idx]
                            {
                                debug!(
                                    "🔄 替換URL | 原始: {} | Poe: {}",
                                    image_url.url, response.attachment_url
                                );
                                image_url.url = response.attachment_url.clone();
                                image_url.mime_type = response.mime_type.clone();
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("❌ 上傳外部URL失敗: {}", e);
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("上傳外部URL失敗: {}", e),
                    )));
                }
            }
        }
    }

    // 處理data URL
    if !data_urls.is_empty() {
        debug!("🔄 準備處理 {} 個data URL", data_urls.len());

        // 分為緩存命中和未命中兩組
    let mut data_to_upload = Vec::new();
    let mut data_indices_to_upload = Vec::new();
    let mut data_mime_types: Vec<Option<String>> = Vec::new();
    let mut data_hashes = Vec::new();

    for (idx, (msg_idx, item_idx)) in data_url_indices.iter().enumerate() {
        let data_url = &data_urls[idx];
        let hash = hash_base64_content(data_url);

            debug!("🔍 計算data URL哈希值 | 哈希頭部: {}...", &hash[..8]);

            // 檢查緩存
            if let Some((poe_url, _)) = crate::cache::get_cached_base64(&hash) {
                debug!("✅ base64緩存命中 | 哈希: {}... -> {}", &hash[..8], poe_url);

                if let Some(OpenAiContent::Multi(items)) = &mut messages[*msg_idx].content {
                    if let OpenAiContentItem::ImageUrl { image_url } = &mut items[*item_idx] {
                        debug!("🔄 從緩存替換base64 | URL: {}", poe_url);
                        image_url.url = poe_url;
                    }
                }
            } else {
                // 緩存未命中，需要上傳
                debug!("❌ base64緩存未命中 | 哈希: {}...", &hash[..8]);
                data_to_upload.push(data_url.clone());
                data_indices_to_upload.push((idx, (*msg_idx, *item_idx)));
                data_hashes.push(hash);
            }
        }

        // 上傳未緩存的data URL
        if !data_to_upload.is_empty() {
            let mut upload_requests = Vec::new();

            // 將data URL轉換為臨時文件
            for data_url in data_to_upload.iter() {
                // 從 data URL 中提取 MIME 類型
                let mime_type = if data_url.starts_with("data:") {
                    let parts: Vec<&str> = data_url.split(";base64,").collect();
                    if !parts.is_empty() {
                        let mime_part = parts[0].trim_start_matches("data:");
                        debug!("🔍 提取的 MIME 類型: {}", mime_part);
                        Some(mime_part.to_string())
                    } else {
                        None
                    }
                } else {
                    None
                };

                data_mime_types.push(mime_type.clone());

                match handle_data_url_to_temp_file(data_url) {
                    Ok(file_path) => {
                        debug!("📄 創建臨時文件成功: {}", file_path.display());
                        upload_requests.push(FileUploadRequest::LocalFile {
                            file: file_path.to_string_lossy().to_string(),
                            mime_type,
                        });
                        temp_files.push(file_path);
                    }
                    Err(e) => {
                        error!("❌ 處理data URL失敗: {}", e);
                        // 清理已創建的臨時文件
                        for path in &temp_files {
                            if let Err(e) = fs::remove_file(path) {
                                warn!("⚠️ 無法刪除臨時文件 {}: {}", path.display(), e);
                            }
                        }
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("處理data URL失敗: {}", e),
                        )));
                    }
                }
            }

            // 上傳臨時文件
            if !upload_requests.is_empty() {
                match poe_client.client.upload_files_batch(upload_requests).await {
                    Ok(responses) => {
                        debug!("✅ 成功上傳 {} 個臨時文件", responses.len());

                        // 更新緩存並保存URL映射
                        for (idx, response) in responses.iter().enumerate() {
                        let (_, (msg_idx, item_idx)) = data_indices_to_upload[idx];
                        let hash = &data_hashes[idx];
                        let data_url = &data_to_upload[idx];
                        let original_mime = data_mime_types
                            .get(idx)
                            .and_then(|m| m.clone());

                        // 估算大小
                        let size = crate::cache::estimate_base64_size(data_url);

                            // 添加到緩存
                            crate::cache::cache_base64(hash, &response.attachment_url, size);

                            debug!(
                                "🔄 將base64哈希映射到Poe URL | 哈希: {}... -> {}",
                                &hash[..8],
                                response.attachment_url
                            );

                            if let Some(OpenAiContent::Multi(items)) =
                                &mut messages[msg_idx].content
                            {
                                if let OpenAiContentItem::ImageUrl { image_url } =
                                    &mut items[item_idx]
                                {
                                    debug!("🔄 替換data URL | Poe: {}", response.attachment_url);
                                    image_url.url = response.attachment_url.clone();
                                    image_url.mime_type = response
                                        .mime_type
                                        .clone()
                                        .or_else(|| original_mime.clone());
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("❌ 上傳臨時文件失敗: {}", e);
                        // 清理臨時文件
                        for path in &temp_files {
                            if let Err(e) = fs::remove_file(path) {
                                warn!("⚠️ 無法刪除臨時文件 {}: {}", path.display(), e);
                            }
                        }
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("上傳臨時文件失敗: {}", e),
                        )));
                    }
                }
            }

            // 清理臨時文件
            for path in &temp_files {
                if let Err(e) = fs::remove_file(path) {
                    warn!("⚠️ 無法刪除臨時文件 {}: {}", path.display(), e);
                } else {
                    debug!("🗑️ 已刪除臨時文件: {}", path.display());
                }
            }
        }
    }

    // 處理AI回覆中的Poe CDN連結，將其添加到用戶消息的image_url中
    if messages.len() >= 2 {
        // 尋找最後一個AI回覆和用戶消息
        let last_bot_idx = messages
            .iter()
            .enumerate()
            .filter(|(_, msg)| msg.role == "assistant")
            .last()
            .map(|(i, _)| i);
        let last_user_idx = messages
            .iter()
            .enumerate()
            .filter(|(_, msg)| msg.role == "user")
            .last()
            .map(|(i, _)| i);

        if let (Some(bot_idx), Some(user_idx)) = (last_bot_idx, last_user_idx) {
            // 提取AI回覆中的Poe CDN連結
            let poe_cdn_urls = extract_poe_cdn_urls_from_message(&messages[bot_idx]);
            if !poe_cdn_urls.is_empty() {
                debug!(
                    "🔄 從AI回覆中提取了 {} 個Poe CDN連結，添加到用戶消息",
                    poe_cdn_urls.len()
                );
                // 將這些連結添加到用戶消息的image_url中
                let user_msg = &mut messages[user_idx];
                match &mut user_msg.content {
                    Some(OpenAiContent::Text(text)) => {
                        // 將文本消息轉換為多部分消息，加入圖片
                        let mut items = Vec::new();
                        items.push(OpenAiContentItem::Text { text: text.clone() });
                        for url in poe_cdn_urls {
                            items.push(OpenAiContentItem::ImageUrl {
                                image_url: ImageUrlContent {
                                    url,
                                    mime_type: None,
                                },
                            });
                        }
                        user_msg.content = Some(OpenAiContent::Multi(items));
                    }
                    Some(OpenAiContent::Multi(items)) => {
                        // 已經是多部分消息，直接添加圖片
                        for url in poe_cdn_urls {
                            items.push(OpenAiContentItem::ImageUrl {
                                image_url: ImageUrlContent {
                                    url,
                                    mime_type: None,
                                },
                            });
                        }
                    }
                    None => {
                        // 如果沒有內容，創建新的多部分消息
                        let mut items = Vec::new();
                        for url in poe_cdn_urls {
                            items.push(OpenAiContentItem::ImageUrl {
                                image_url: ImageUrlContent {
                                    url,
                                    mime_type: None,
                                },
                            });
                        }
                        user_msg.content = Some(OpenAiContent::Multi(items));
                    }
                }
            }
        }
    }

    Ok(())
}

// 從 OpenAIContent 獲取純文本內容
pub fn get_text_from_openai_content(content: &Option<OpenAiContent>) -> String {
    match content {
        Some(OpenAiContent::Text(s)) => s.clone(),
        Some(OpenAiContent::Multi(items)) => {
            let mut text_parts = Vec::new();
            for item in items {
                if let OpenAiContentItem::Text { text } = item {
                    // 使用 serde_json::to_string 處理文本中的特殊字符
                    match serde_json::to_string(text) {
                        Ok(processed_text) => {
                            // 移除 serde_json::to_string 添加的開頭和結尾的引號
                            let processed_text = processed_text.trim_matches('"').to_string();
                            // 將 JSON 轉義的引號 (\") 替換為普通引號 (")
                            let processed_text = processed_text.replace("\\\"", "\"");
                            text_parts.push(processed_text);
                        }
                        Err(_) => {
                            // 如果序列化失敗，使用原始文本
                            text_parts.push(text.clone());
                        }
                    }
                }
            }
            text_parts.join("\n")
        }
        None => String::new(),
    }
}

// 檢查URL是否為Poe CDN連結
pub fn is_poe_cdn_url(url: &str) -> bool {
    url.starts_with("https://pfst.cf2.poecdn.net")
}

// 從消息中提取Poe CDN連結
pub fn extract_poe_cdn_urls_from_message(message: &Message) -> Vec<String> {
    let mut urls = Vec::new();
    match &message.content {
        Some(OpenAiContent::Multi(items)) => {
            for item in items {
                if let OpenAiContentItem::ImageUrl { image_url } = item {
                    if is_poe_cdn_url(&image_url.url) {
                        urls.push(image_url.url.clone());
                    }
                } else if let OpenAiContentItem::Text { text } = item {
                    // 從文本中提取 Poe CDN URL
                    extract_urls_from_markdown(text, &mut urls);
                }
            }
        }
        Some(OpenAiContent::Text(text)) => {
            // 從純文本消息中提取 Poe CDN URL
            extract_urls_from_markdown(text, &mut urls);
        }
        None => {
            // 沒有內容，返回空列表
        }
    }
    urls
}

// 從 Markdown 文本中提取 Poe CDN URL 的輔助函數
fn extract_urls_from_markdown(text: &str, urls: &mut Vec<String>) {
    // 提取 Markdown 圖片格式的 URL: ![alt](url)
    let re_md_img = regex::Regex::new(r"!\[.*?\]\((https?://[^\s)]+)\)").unwrap();
    for cap in re_md_img.captures_iter(text) {
        if let Some(url) = cap.get(1) {
            let url_str = url.as_str();
            if is_poe_cdn_url(url_str) {
                urls.push(url_str.to_string());
            }
        }
    }
    // 同時處理直接出現的 URL
    for word in text.split_whitespace() {
        if is_poe_cdn_url(word) {
            urls.push(word.to_string());
        }
    }
}

// 處理base64數據URL，將其存儲為臨時文件
pub fn handle_data_url_to_temp_file(data_url: &str) -> Result<PathBuf, String> {
    // 1. 驗證資料 URL 格式
    if !data_url.starts_with("data:") {
        return Err("無效的資料 URL 格式".to_string());
    }
    // 2. 分離 MIME 類型和 base64 資料
    let parts: Vec<&str> = data_url.split(";base64,").collect();
    if parts.len() != 2 {
        return Err("無效的資料 URL 格式：缺少 base64 分隔符".to_string());
    }
    // 3. 提取 MIME 類型
    let mime_type = parts[0].strip_prefix("data:").unwrap_or(parts[0]);
    debug!("🔍 提取的 MIME 類型: {}", mime_type);
    // 4. 根據 MIME 類型決定檔案擴充名
    let file_ext = mime_type_to_extension(mime_type).unwrap_or("bin");
    debug!("📄 使用檔案擴充名: {}", file_ext);
    // 5. 解碼 base64 資料 (僅使用 BASE64_STANDARD)
    let base64_data = parts[1];
    debug!("🔢 Base64 資料長度: {}", base64_data.len());
    let decoded = match BASE64_STANDARD.decode(base64_data) {
        Ok(data) => {
            debug!("✅ Base64 解碼成功 | 資料大小: {} 位元組", data.len());
            data
        }
        Err(e) => {
            error!("❌ Base64 解碼失敗: {}", e);
            return Err(format!("Base64 解碼失敗: {}", e));
        }
    };
    // 6. 建立臨時檔案
    let temp_dir = std::env::temp_dir();
    let file_name = format!("poe2openai_{}.{}", nanoid!(16), file_ext);
    let file_path = temp_dir.join(&file_name);
    // 7. 寫入資料到臨時檔案
    match fs::write(&file_path, &decoded) {
        Ok(_) => {
            debug!("✅ 成功寫入臨時檔案: {}", file_path.display());
            Ok(file_path)
        }
        Err(e) => {
            error!("❌ 寫入臨時檔案失敗: {}", e);
            Err(format!("寫入臨時檔案失敗: {}", e))
        }
    }
}

// 從MIME類型獲取文件擴展名
fn mime_type_to_extension(mime_type: &str) -> Option<&str> {
    match mime_type {
        "image/jpeg" | "image/jpg" => Some("jpeg"),
        "image/png" => Some("png"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        "image/svg+xml" => Some("svg"),
        "image/bmp" => Some("bmp"),
        "image/tiff" => Some("tiff"),
        "application/pdf" => Some("pdf"),
        "text/plain" => Some("txt"),
        "text/csv" => Some("csv"),
        "application/json" => Some("json"),
        "application/xml" | "text/xml" => Some("xml"),
        "application/zip" => Some("zip"),
        "application/x-tar" => Some("tar"),
        "application/x-gzip" => Some("gz"),
        "audio/mpeg" => Some("mp3"),
        "audio/wav" => Some("wav"),
        "audio/ogg" => Some("ogg"),
        "video/mp4" => Some("mp4"),
        "video/mpeg" => Some("mpeg"),
        "video/quicktime" => Some("mov"),
        _ => None,
    }
}

pub fn convert_poe_error_to_openai(
    error_text: &str,
    allow_retry: bool,
) -> (StatusCode, OpenAIErrorResponse) {
    debug!(
        "🔄 轉換錯誤響應 | 錯誤文本: {}, 允許重試: {}",
        error_text, allow_retry
    );
    let (status, error_type, code) = if error_text.contains("Internal server error") {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "internal_error",
        )
    } else if error_text.contains("rate limit") {
        (
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_exceeded",
            "rate_limit_exceeded",
        )
    } else if error_text.contains("Invalid token") || error_text.contains("Unauthorized") {
        (StatusCode::UNAUTHORIZED, "invalid_auth", "invalid_api_key")
    } else if error_text.contains("Bot does not exist") {
        (StatusCode::NOT_FOUND, "model_not_found", "model_not_found")
    } else {
        (StatusCode::BAD_REQUEST, "invalid_request", "bad_request")
    };
    debug!(
        "📋 錯誤轉換結果 | 狀態碼: {} | 錯誤類型: {}",
        status.as_u16(),
        error_type
    );
    (
        status,
        OpenAIErrorResponse {
            error: OpenAIError {
                message: error_text.to_string(),
                r#type: error_type.to_string(),
                code: code.to_string(),
                param: None,
            },
        },
    )
}

pub fn format_bytes_length(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn format_duration(duration: std::time::Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else {
        format!("{}ms", duration.as_millis())
    }
}

pub fn get_config_path(filename: &str) -> PathBuf {
    let config_dir = std::env::var("CONFIG_DIR").unwrap_or_else(|_| "./".to_string());
    let mut path = PathBuf::from(config_dir);
    path.push(filename);
    path
}

pub fn load_config_from_yaml() -> Result<Config, String> {
    let path_str = "models.yaml";
    let path = get_config_path(path_str);
    if path.exists() {
        match std::fs::read_to_string(path) {
            Ok(contents) => match serde_yaml::from_str::<Config>(&contents) {
                Ok(config) => {
                    info!("✅ 成功讀取並解析 {}", path_str);
                    Ok(config)
                }
                Err(e) => {
                    error!("❌ 解析 {} 失敗: {}", path_str, e);
                    Err(format!("解析 {} 失敗: {}", path_str, e))
                }
            },
            Err(e) => {
                error!("❌ 讀取 {} 失敗: {}", path_str, e);
                Err(format!("讀取 {} 失敗: {}", path_str, e))
            }
        }
    } else {
        debug!("⚠️  {} 不存在，使用預設空配置", path_str);
        // 返回一個預設的 Config，表示文件不存在或無法讀取
        Ok(Config {
            enable: Some(false),
            models: std::collections::HashMap::new(),
            custom_models: None,
            api_token: None,
            use_v1_api: None,
        })
    }
}

/// 計算文本的 token 數量
pub fn count_tokens(text: &str) -> u32 {
    let bpe = match o200k_base() {
        Ok(bpe) => bpe,
        Err(e) => {
            error!("❌ 無法初始化 BPE 編碼器: {}", e);
            return 0;
        }
    };
    let tokens = bpe.encode_with_special_tokens(text);
    tokens.len() as u32
}

/// 計算消息列表的 token 數量
pub fn count_message_tokens(messages: &[Message]) -> u32 {
    let mut total_tokens = 0;
    for message in messages {
        // 每條消息的基本 token 數（角色標記等）
        total_tokens += 4; // 每條消息的基本開銷
        // 計算內容的 token 數
        let content_text = get_text_from_openai_content(&message.content);
        total_tokens += count_tokens(&content_text);
    }
    // 添加消息格式的額外 token
    total_tokens += 2; // 消息格式的開始和結束標記
    total_tokens
}

/// 計算完成內容的 token 數量
pub fn count_completion_tokens(completion: &str) -> u32 {
    count_tokens(completion)
}

/// 計算 base64 字符串的 SHA256 哈希
pub fn hash_base64_content(base64_str: &str) -> String {
    // 提取純base64部分，去除MIME類型前綴
    let base64_data = match base64_str.split(";base64,").nth(1) {
        Some(data) => data,
        None => base64_str, // 如果沒有分隔符，使用整個字符串
    };

    let start = &base64_data[..base64_data.len().min(1024)];
    let end = if base64_data.len() > 2048 {
        // 確保有足夠長度
        &base64_data[base64_data.len() - 1024..]
    } else if base64_data.len() > 1024 {
        &base64_data[1024..] // 如果長度在1024-2048之間，使用剩餘部分
    } else {
        "" // 如果小於1024，只使用start
    };

    // 結合頭部和尾部數據
    let combined = format!("{}{}", start, end);

    // 計算SHA256哈希
    let mut hasher = Sha256::new();
    hasher.update(combined.as_bytes());
    let result = hasher.finalize();

    // 記錄哈希計算信息以便調試
    let hash = format!("{:x}", result);
    debug!(
        "🔢 計算base64哈希 | 數據長度: {} | 計算長度: {} | 哈希值頭部: {}...",
        base64_data.len(),
        start.len() + end.len(),
        &hash[..8]
    );

    hash
}

/// 根據 URL 推斷 MIME 類型
pub fn infer_mime_from_url(url: &str) -> Option<String> {
    let without_query = url.split('?').next().unwrap_or(url).to_lowercase();
    let ext = without_query.split('.').last().unwrap_or("");
    let mime = match ext {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        "bmp" => Some("image/bmp"),
        "svg" => Some("image/svg+xml"),
        "pdf" => Some("application/pdf"),
        "txt" => Some("text/plain"),
        "json" => Some("application/json"),
        "csv" => Some("text/csv"),
        "mp4" => Some("video/mp4"),
        "mp3" => Some("audio/mpeg"),
        _ => None,
    };

    // 一些 Poe CDN 圖片沒有副檔名，嘗試根據路徑特徵給預設
    if mime.is_none() && url.contains("/base/image/") {
        return Some("image/jpeg".to_string());
    }

    mime.map(|m| m.to_string())
}

/// 根據 URL 或 MIME 類型生成檔名
pub fn filename_from_url(url: &str, mime_type: Option<&str>) -> Option<String> {
    let without_query = url.split('?').next().unwrap_or(url);
    let last = without_query.split('/').last().unwrap_or("").trim();
    if !last.is_empty() && !last.ends_with("image") && !last.ends_with("base") {
        return Some(last.to_string());
    }

    if let Some(mt) = mime_type {
        let ext = match mt {
            "image/jpeg" => "jpg",
            "image/png" => "png",
            "image/webp" => "webp",
            "image/gif" => "gif",
            "image/bmp" => "bmp",
            "image/svg+xml" => "svg",
            "application/pdf" => "pdf",
            "text/plain" => "txt",
            "application/json" => "json",
            "text/csv" => "csv",
            _ => "bin",
        };
        return Some(format!("file.{}", ext));
    }

    None
}

/// 處理消息內容，根據請求參數添加相應的後綴
pub fn process_message_content_with_suffixes(
    content: &str,
    chat_request: &crate::types::ChatCompletionRequest,
) -> String {
    let mut processed_content = content.to_string();

    // 處理 function tools - 檢查是否只有 name 字段
    if let Some(tools) = &chat_request.tools {
        for tool in tools {
            // 檢查是否只有 name 字段（description 為 None 或空字符串）
            let has_description = tool
                .function
                .description
                .as_ref()
                .map(|desc| !desc.is_empty())
                .unwrap_or(false);

            if !has_description {
                let suffix = format!(" --{}", tool.function.name);
                debug!("🔧 添加 function name 後綴: {}", suffix);
                processed_content.push_str(&suffix);
            }
        }
    }

    // 處理 thinking_budget
    let thinking_budget = if let Some(thinking) = &chat_request.thinking {
        thinking.budget_tokens
    } else if let Some(extra_body) = &chat_request.extra_body {
        extra_body
            .google
            .as_ref()
            .and_then(|g| g.thinking_config.as_ref())
            .and_then(|tc| tc.thinking_budget)
    } else {
        None
    };
    if let Some(budget) = thinking_budget {
        // 只有在正數範圍內才添加 --thinking_budget 參數
        if budget >= 0 {
            let suffix = format!(" --thinking_budget {}", budget);
            debug!("🧠 添加 thinking_budget 後綴: {}", suffix);
            processed_content.push_str(&suffix);
        } else {
            debug!(
                "🧠 thinking_budget 值 {} 超出範圍，跳過添加 --thinking_budget 參數",
                budget
            );
        }
    }

    // 處理 reasoning_effort
    if let Some(effort) = &chat_request.reasoning_effort {
        // 驗證值是否為有效選項
        let valid_efforts = ["low", "medium", "high"];
        if valid_efforts.contains(&effort.as_str()) {
            let suffix = format!(" --reasoning_effort {}", effort);
            debug!("🎯 添加 reasoning_effort 後綴: {}", suffix);
            processed_content.push_str(&suffix);
        } else {
            warn!("⚠️ 無效的 reasoning_effort 值: {}", effort);
        }
    }

    processed_content
}

/// 過濾掉只有 name 字段的 tools，這些 tools 不應該傳遞給 poe_api_process
pub fn filter_tools_for_poe(
    tools: &Option<Vec<poe_api_process::types::ChatTool>>,
) -> Option<Vec<poe_api_process::types::ChatTool>> {
    if let Some(tools_vec) = tools {
        let filtered_tools: Vec<_> = tools_vec
            .iter()
            .filter(|tool| {
                // 保留有 description 的 tools（不為 None 且不為空字符串）
                tool.function
                    .description
                    .as_ref()
                    .map(|desc| !desc.is_empty())
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        if filtered_tools.is_empty() {
            debug!("🔧 所有 tools 都只有 name 字段，移除所有 tools");
            None
        } else {
            debug!(
                "🔧 過濾後保留 {} 個 tools（原本 {} 個）",
                filtered_tools.len(),
                tools_vec.len()
            );
            Some(filtered_tools)
        }
    } else {
        None
    }
}

/// 從工具消息中提取 tool_call_id
pub fn extract_tool_call_id(content: &str) -> Option<String> {
    // 嘗試解析 JSON 格式的內容
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(content) {
        if let Some(tool_call_id) = json.get("tool_call_id").and_then(|v| v.as_str()) {
            return Some(tool_call_id.to_string());
        }
    }
    // 嘗試使用簡單的文本解析
    if let Some(start) = content.find("tool_call_id") {
        if let Some(id_start) = content[start..].find('"') {
            if let Some(id_end) = content[start + id_start + 1..].find('"') {
                return Some(
                    content[start + id_start + 1..start + id_start + 1 + id_end].to_string(),
                );
            }
        }
    }
    None
}
