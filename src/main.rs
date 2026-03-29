use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use log::{error, info, warn};
use ort::{inputs, session::Session, value::Tensor};
use reqwest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

// --- CONFIGURATION ---
const DEFAULT_DOC_FOLDER: &str = "./docs";
const DEFAULT_INDEX_FILE: &str = "rag_index.json";
const DEFAULT_EMBED_MODEL: &str = "./models/all-MiniLM-L6-v2.onnx";
const DEFAULT_TOKENIZER: &str = "./models/tokenizer.json";
const DEFAULT_PORT: &str = "8080";
const LLAMAFILE_URL: &str = "http://localhost:8081";
const CHUNK_SIZE: usize = 400;
const MAX_LENGTH: usize = 128;

// --- STRUCTURES API ---
#[derive(Serialize, Deserialize)]
struct QueryRequest {
    question: String,
}

#[derive(Serialize, Deserialize)]
struct QueryResponse {
    status: String,
    data: String,
    source: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct ReindexResponse {
    status: String,
    message: String,
    source: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct HealthResponse {
    status: String,
    message: String,
    source: Vec<String>,
}

// --- STRUCTURES INDEX ---
#[derive(Serialize, Deserialize, Clone)]
struct Chunk {
    path: String,
    content: String,
    embedding: Vec<f32>,
    file_hash: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct IndexStore {
    chunks: Vec<Chunk>,
    processed_files: HashMap<String, u64>,
}

// --- ÉTAT PARTAGÉ (STATE) ---
struct AppState {
    client: reqwest::Client,
    ort_session: Mutex<Session>,
    tokenizer: Arc<Tokenizer>,
    index: Mutex<IndexStore>,
    doc_folder: PathBuf,
    index_file: String,
}

// --- FONCTIONS EMBEDDING ---
fn embed(ort_session: &mut Session, tokenizer: &Tokenizer, text: &str) -> Result<Vec<f32>> {
    // 1. Tokenisation
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut ids = encoding.get_ids().to_vec();
    let mut mask = encoding.get_attention_mask().to_vec();

    // 2. Padding / Truncation pour correspondre au modèle (128 tokens)
    if ids.len() > MAX_LENGTH {
        ids.truncate(MAX_LENGTH);
        mask.truncate(MAX_LENGTH);
    }
    while ids.len() < MAX_LENGTH {
        ids.push(0);
        mask.push(0);
    }

    // 3. Création des tenseurs ONNX (i64 pour les token IDs)
    // // token_type_ids = tous à 0 pour une phrase simple
    let token_types = vec![0i64; MAX_LENGTH];
    let ids_i64: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let mask_i64: Vec<i64> = mask.iter().map(|&x| x as i64).collect();
    let ids_tensor = Tensor::from_array(([1, MAX_LENGTH], ids_i64.into_boxed_slice()))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mask_tensor = Tensor::from_array(([1, MAX_LENGTH], mask_i64.into_boxed_slice()))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let types_tensor = Tensor::from_array(([1, MAX_LENGTH], token_types.into_boxed_slice()))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // 4. Exécution
    let inputs = inputs![
        "input_ids" => ids_tensor,
        "attention_mask" => mask_tensor,
        "token_type_ids" => types_tensor,
    ];

    let outputs = ort_session
        .run(inputs)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let (_, output) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // 5. Extraction du vecteur (pooling mean simplifié : on prend la première sortie)
    // all-MiniLM-L6-v2 retourne [1, 384] après pooling
    Ok(output.to_vec())
}

// --- UTILITAIRES ---
fn simple_hash(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

fn load_index(path: &str) -> IndexStore {
    if Path::new(path).exists() {
        match fs::read_to_string(path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(e) => {
                warn!("Erreur lecture index: {}", e);
                IndexStore::default()
            }
        }
    } else {
        IndexStore::default()
    }
}

fn save_index(store: &IndexStore, path: &str) {
    match serde_json::to_string(store) {
        Ok(json) => {
            if let Err(e) = fs::write(path, json) {
                error!("Erreur sauvegarde index: {}", e);
            }
        }
        Err(e) => error!("Erreur sérialisation index: {}", e),
    }
}

fn extract_text(path: &Path) -> Result<String> {
    Ok(fs::read_to_string(path)?)
}

fn chunk_text(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if current.len() + word.len() > CHUNK_SIZE {
            if !current.is_empty() {
                chunks.push(current.clone());
            }
            current = word.to_string();
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn search_index(store: &IndexStore, query_emb: &[f32], top_k: usize) -> Vec<(String, f32)> {
    let mut scores: Vec<(usize, f32)> = store
        .chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| (i, cosine_similarity(query_emb, &chunk.embedding)))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);

    scores
        .iter()
        .filter(|(_, score)| *score > 0.3)
        .map(|(i, score)| (store.chunks[*i].content.clone(), *score))
        .collect()
}

fn process_file(
    path: &Path,
    ort_session: &mut Session,
    tokenizer: &Tokenizer,
    store: &mut IndexStore,
) {
    let path_str = path.to_string_lossy().to_string();
    let content = match extract_text(path) {
        Ok(c) => c,
        Err(e) => {
            warn!("Erreur lecture {}: {}", path_str, e);
            return;
        }
    };

    let hash = simple_hash(&content);

    if store.processed_files.get(&path_str) == Some(&hash) {
        return;
    }

    info!(
        "Indexation: {}",
        path.file_name().unwrap().to_string_lossy()
    );
    let chunks = chunk_text(&content);

    for chunk in chunks {
        if chunk.len() < 20 {
            continue;
        }
        match embed(ort_session, tokenizer, &chunk) {
            Ok(emb) => {
                store.chunks.push(Chunk {
                    path: path_str.clone(),
                    content: chunk,
                    embedding: emb,
                    file_hash: hash,
                });
            }
            Err(e) => warn!("Erreur embedding: {}", e),
        }
    }
    store.processed_files.insert(path_str, hash);
}

fn reindex_all(state: &AppState) -> Vec<String> {
    let mut store = state.index.lock().unwrap();
    let mut ort_session = state.ort_session.lock().unwrap();
    let mut indexed_files = Vec::new();

    for entry in walkdir::WalkDir::new(&state.doc_folder)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("md") {
                process_file(path, &mut ort_session, &state.tokenizer, &mut store);
                indexed_files.push(path.to_string_lossy().to_string());
            }
        }
    }

    save_index(&store, &state.index_file);
    indexed_files
}

// --- HANDLERS HTTP ---
async fn health_check(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let llama_status = state
        .client
        .get(format!("{}/health", LLAMAFILE_URL))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    let message = if llama_status {
        "Server and LLM ready"
    } else {
        "Server ready, LLM unreachable"
    };

    Json(HealthResponse {
        status: "ok".to_string(),
        message: message.to_string(),
        source: vec![],
    })
}

async fn query(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, StatusCode> {
    let (sources, context_str) = {
        let store = state
            .index
            .lock()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let q_emb = {
            let mut ort_session = state
                .ort_session
                .lock()
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
            embed(&mut ort_session, &state.tokenizer, &payload.question)
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        };
        let results = search_index(&store, &q_emb, 3);

        if results.is_empty() {
            return Ok(Json(QueryResponse {
                status: "no_context".to_string(),
                data: "Aucun document pertinent trouvé dans la base de connaissances.".to_string(),
                source: vec![],
            }));
        }

        let sources: Vec<String> = results
            .iter()
            .map(|(_, s)| format!("Score: {:.2}", s))
            .collect();
        let context_str: String = results
            .iter()
            .map(|(c, _)| c.clone())
            .collect::<Vec<_>>()
            .join("\n---\n");
        (sources, context_str)
    };

    let prompt = format!(
        "Tu es un assistant de recherche documentaire. À partir des informations dont tu disposes, et sans JAMAIS faire d'assertion, réponds en français de manière concise en te basant UNIQUEMENT sur le CONTEXTE ci-dessous.\n\nCONTEXTE:\n{}\n\nQUESTION:\n{}\n\nRÉPONSE:",
        context_str, payload.question
    );

    let answer = match reqwest::Client::new()
        .post(format!("{}/completion", LLAMAFILE_URL))
        .json(&serde_json::json!({
            "prompt": prompt,
            "n_predict": 512,
            "temperature": 0.3,
            "top_p": 0.9
        }))
        .send()
        .await
    {
        Ok(response) => match response.json::<serde_json::Value>().await {
            Ok(json) => json["content"]
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "Erreur de génération".to_string()),
            Err(_) => "Erreur de génération".to_string(),
        },
        Err(_) => "Erreur de génération".to_string(),
    };

    Ok(Json(QueryResponse {
        status: "success".to_string(),
        data: answer,
        source: sources,
    }))
}

async fn reindex(State(state): State<Arc<AppState>>) -> Json<ReindexResponse> {
    info!("Réindexation demandée...");
    let files = reindex_all(&state);
    info!("Réindexation terminée. {} fichiers traités.", files.len());

    Json(ReindexResponse {
        status: "success".to_string(),
        message: format!("{} documents indexés", files.len()),
        source: files,
    })
}

// --- MAIN ---
#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let doc_folder = std::env::var("DOC_FOLDER").unwrap_or_else(|_| DEFAULT_DOC_FOLDER.to_string());
    let index_file = std::env::var("INDEX_FILE").unwrap_or_else(|_| DEFAULT_INDEX_FILE.to_string());
    let embed_model =
        std::env::var("EMBED_MODEL").unwrap_or_else(|_| DEFAULT_EMBED_MODEL.to_string());
    let tokenizer_file =
        std::env::var("TOKENIZER").unwrap_or_else(|_| DEFAULT_TOKENIZER.to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());

    info!("Démarrage du serveur RAG API...");
    info!("Dossier documents: {}", &doc_folder);
    info!("Port HTTP: {}", &port);

    // Session ONNX
    let ort_session = Session::builder()
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .with_intra_threads(2)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .commit_from_file(&embed_model)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Tokenizer
    let tokenizer = Arc::new(
        Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow::anyhow!("Erreur chargement tokenizer: {}", e))?,
    );

    // Client HTTP
    let client = reqwest::Client::new();

    // Index
    let index = load_index(&index_file);
    info!("Index chargé: {} chunks", index.chunks.len());

    let app_state = Arc::new(AppState {
        client,
        ort_session: Mutex::new(ort_session),
        tokenizer,
        index: Mutex::new(index),
        doc_folder: PathBuf::from(doc_folder),
        index_file,
    });

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/query", post(query))
        .route("/reindex", post(reindex))
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;

    info!("Serveur écoute sur http://0.0.0.0:{}", &port);

    axum::serve(listener, app).await?;

    Ok(())
}
