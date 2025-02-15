# コード解説

###  コードのポイント
1. **設定ファイル (`config.json`) を読み込んで、オンライン/ローカルを自動判別**
2. **オンライン推論 (`reqwest` + `async`) とローカル推論 (`reqwest::blocking`) に対応**
3. **OpenAI互換モードとカスタムAPIモードの切り替えが可能**
4. **エラーハンドリング (`Result<T, E>` + `.unwrap_or` + `match`) もちゃんと実装**
5. **ちゃんと `tokio::main` で `async/await` を使って非同期処理を最適化**
6. **Rustらしく `serde_json` で JSON のパースを型安全に行っている**
7. **標準入出力 (`io::stdin().read_line()`) で簡易インタラクティブモードを実装**

---
##  このコードの「Rustらしさ」
###  ① `serde` で JSON を型安全にパース
```rust
#[derive(Deserialize)]
struct Config {
    model_name: String,
    endpoint: Option<String>,
    use_local_model: bool,
    openai_compatible: bool,
    max_tokens: Option<u32>,
    api_key: Option<String>,
}
```
`serde` を使って `Config` を定義し、型安全に JSON をパース。  
`Option<String>` を使って、設定が **未設定（`null`）でも安全に扱える。**  
**`serde_json::from_str(&config_data)` で設定を読み込むだけでシンプルに扱える。**  

---

### ② 設定ファイルがないと自動生成する設計
```rust
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
```
`fs::File::create()` でファイルを作成し、デフォルトの設定を JSON 形式で書き込む。   
**Rustの `expect()` を使うことで、エラーハンドリングもシンプルにした。**  
**設定ファイルがないときに `generate_default_config(path)` を実行**

---

### ③ 非同期 (`async/await`) を活用して API を呼び出す
```rust
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
```
**`async fn` + `await` で非同期処理を実装し、並列リクエストに対応。**  
**設定ファイルの情報を利用し、OpenAI互換の API とカスタム API どちらでも動くようにした。**  
**`if let Some(api_key) = &config.api_key` を使って、APIキーの設定がある場合だけヘッダーを追加する設計。**  
**「APIの種類を判別してリクエストの形を変える」のがたぶんけっこう柔軟性高い。**

---

### ④ ローカル推論はブロッキングで実行 (`reqwest::blocking`)
```rust
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
```
**オンラインは非同期だけど、ローカルはブロッキングにして最適化。**  
**APIのレスポンスのパース (`serde_json`) も `unwrap_or_else` を使ってエラー処理。**  
**エラー時のメッセージ処理 (`format!()`) もちゃんと考えた。**

---

### ⑤ `main()` の設計について
```rust
#[tokio::main]
async fn main() {
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
```
**設定ファイルを読み込み、ローカル/オンラインを自動で切り替え。**  
**非同期処理 (`await`) を使いながら `loop` でチャットモードを実装。**  
**入力エラー処理 (`if prompt.is_empty()`) も行っています。**  
**Rustらしい「安全 & 高速 & シンプル」な設計。**