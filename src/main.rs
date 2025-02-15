use std::fs;
use std::io::{self, Write};
use serde::Deserialize;
use std::path::Path;

// 設定ファイルの内容を保持する構造体
#[derive(Deserialize)]
struct Config {
    model_name: String,
    endpoint: Option<String>,
    use_local_model: bool,
    openai_compatible: bool,
    max_tokens: Option<u32>,
    api_key: Option<String>,
}

// デフォルト設定ファイルを生成する関数
fn generate_default_config(path: &str) {
    let default_config = r#"{
        "model_name": "gemma:2b",
        "endpoint": "http://localhost:11434/api/generate",
        "use_local_model": true,
        "openai_compatible": false,
        "max_tokens": 64,
        "api_key": null
    }"#;

    let mut file = fs::File::create(path).expect("設定ファイルの作成に失敗しました");
    file.write_all(default_config.as_bytes()).expect("設定ファイルの書き込みに失敗しました");
}

// 設定ファイルを読み込む関数（存在しない場合は自動生成）
fn load_config(path: &str) -> Config {
    if !Path::new(path).exists() {
        println!("設定ファイルが見つかりません。デフォルト設定を作成します...");
        generate_default_config(path);
    }
    let config_data = fs::read_to_string(path)
        .expect("設定ファイルの読み込みに失敗しました");
    serde_json::from_str(&config_data)
        .expect("JSONのパースに失敗しました")
}

// オンライン推論を実行する非同期関数
async fn online_inference(config: &Config, prompt: &str) -> Result<String, reqwest::Error> {
    let endpoint = config.endpoint.as_ref()
        .expect("オンライン推論用のendpointが設定されていません");

    let max_tokens = config.max_tokens.unwrap_or(64);

    let request_body = if config.openai_compatible {
        serde_json::json!({
            "model": config.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens
        })
    } else {
        serde_json::json!({
            "model": config.model_name,
            "input": prompt,
            "max_tokens": max_tokens
        })
    };

    let client = reqwest::Client::new();
    let mut request_builder = client.post(endpoint).json(&request_body);

    if let Some(api_key) = &config.api_key {
        request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
    }

    let res = request_builder.send().await?;
    let res_json: serde_json::Value = res.json().await?;

    // OpenAI互換モードとカスタムモードでレスポンス処理を分ける
    let output = if config.openai_compatible {
        res_json.get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|choice| choice.get("text"))
            .and_then(|text| text.as_str())
            .unwrap_or("レスポンスが不正です")
            .to_string()
    } else {
        res_json.get("generated_text")
            .and_then(|text| text.as_str())
            .unwrap_or("レスポンスが不正です")
            .to_string()
    };

    Ok(output)
}

// ローカル推論を実行する関数
fn local_inference(prompt: &str, config: &Config) -> String {
    let endpoint = config.endpoint.as_deref().unwrap_or("http://localhost:11434/api/generate");
    let max_tokens = config.max_tokens.unwrap_or(64);
    
    let request_body = serde_json::json!({
        "model": config.model_name,
        "prompt": prompt,
        "max_tokens": max_tokens
    });

    let client = reqwest::blocking::Client::new();
    let res = client.post(endpoint).json(&request_body).send();

    match res {
        Ok(response) => {
            let res_json: serde_json::Value =
                response.json().unwrap_or_else(|_| serde_json::json!({}));
            res_json.get("response")
                .and_then(|text| text.as_str())
                .unwrap_or("ローカル推論エラー")
                .to_string()
        }
        Err(e) => format!("ローカル推論エラー: {:?}", e),
    }
}

#[tokio::main]
async fn main() {
    // 設定ファイルからパラメータを読み込む
    let config_path = "config.json";
    let config = load_config(config_path);

    println!("モデル: {}", config.model_name);
    if config.use_local_model {
        println!("ローカルモードで動作します");
    } else {
        println!("オンラインモードで動作します");
        println!("OpenAI互換モード: {}", if config.openai_compatible { "有効" } else { "無効" });
    }

    println!("チャットクライアントを開始します（空行で終了）");

    loop {
        print!("You > ");
        io::stdout().flush().unwrap();

        let mut prompt = String::new();
        if io::stdin().read_line(&mut prompt).is_err() {
            println!("入力エラー");
            break;
        }
        let prompt = prompt.trim();
        if prompt.is_empty() {
            break;
        }

        let response = if config.use_local_model {
            local_inference(prompt, &config)
        } else {
            match online_inference(&config, prompt).await {
                Ok(text) => text,
                Err(e) => format!("オンライン推論エラー: {:?}", e),
            }
        };

        println!("Assistant > {}", response);
    }
}
