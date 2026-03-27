use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use anyhow::{Context, Result};
use ndarray::{Array1, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{info, warn, error};

// --- CONFIGURATION ---
// Ces variables peuvent être surchargées par des ENV VAR pour le déploiement
const DEFAULT_DOC_FOLDER: &str = "./docs";
const DEFAULT_INDEX_FILE: &str = "rag_index.json";
const DEFAULT_LLM_MODEL: &str = "./models/phi-3-mini-4k-instruct.Q4_K_M.gguf";
const DEFAULT_EMBED_MODEL: &str = "./models/all-MiniLM-L6-v2.onnx";
const DEFAULT_PORT: &str = "8080";

const CHUNK_SIZE: usize = 400; // Un peu plus grand pour de meilleures réponses

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
    message:String,
    source: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct HealthResponse {
    status: String,
    message:String,
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
    engines: Mutex<Engines>,
    index: Mutex<IndexStore>,
    doc_folder: PathBuf,
    index_file: String,
}

struct Engines {
    llama: llama_cpp_rs::LlamaContext,
    ort_session: ort::Session,
}

impl Engines {
    fn new(llm_path: &str, embed_path: &str) -> Result<Self> {
        info!("Chargement du modèle LLM...");
        let model_params = llama_cpp_rs::ModelParams::default();
        let ctx_params = llama_cpp_rs::ContextParams::new()
            .with_n_ctx(2048)
            .with_seed(1234);
            
        let model = llama_cpp_rs::LlamaModel::load_from_file(llm_path, model_params)
            .context("Erreur chargement modèle LLM")?;
        let llama = model.new_context(ctx_params)
            .context("Erreur création contexte LLM")?;

        info!("Chargement du modèle Embedding...");
        let ort_session = ort::Session::builder()?
            .with_intra_threads(2)?
            .commit_from_file(embed_path)?;

        Ok(Self { llama, ort_session })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let inputs = ort::inputs![text]?;
        let outputs = self.ort_session.run(inputs)?;
        let output = outputs[0].try_extract_tensor::<f32>()?;
        Ok(output.view().as_slice().unwrap().to_vec())
    }

    fn generate(&mut self, prompt: &str) -> Result<String> {
        let mut sampler = llama_cpp_rs::Sampler::new()
            .with_temp(0.3)
            .with_top_p(0.9);
        
        let mut output = String::new();
        let tokens = self.llama.model().tokenize(prompt, true)?;
        let mut n_past = 0;
        let max_tokens = 512; 

        for &token in tokens.iter() {
            self.llama.eval(&[token], &mut n_past)?;
        }

        for _ in 0..max_tokens {
            let token = self.llama.sample(&mut sampler, n_past)?;
            if token == self.llama.model().token_eos() {
                break;
            }
            let text = self.llama.model().token_to_str(token)?;
            output.push_str(&text);
            self.llama.eval(&[token], &mut n_past)?;
        }
        Ok(output)
    }
}

// --- FONCTIONS UTILITAIRES ---
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
    // Markdown uniquement pour cette version
    Ok(fs::read_to_string(path)?)
}

fn chunk_text(text: &str, path: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    
    for word in text.split_whitespace() {
        if current.len() + word.len() > CHUNK_SIZE {
            if !current.is_empty() {
                chunks.push(current.clone());
            }
            current = word.to_string();
        } else {
            if !current.is_empty() { current.push(' '); }
            current.push_str(word);
        }
    }
    if !current.is_empty() { chunks.push(current); }
    chunks
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}

fn search_index(store: &IndexStore, query_emb: &[f32], top_k: usize) -> Vec<(String, f32)> {
    let mut scores: Vec<(usize, f32)> = store.chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| (i, cosine_similarity(query_emb, &chunk.embedding)))
        .collect();
    
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);
    
    scores.iter()
        .filter(|(_, score)| *score > 0.3) // Seuil de pertinence minimum
        .map(|(i, score)| (store.chunks[*i].content.clone(), *score))
        .collect()
}

fn process_file(path: &Path, engines: &mut Engines, store: &mut IndexStore) {
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

    info!("Indexation: {}", path.file_name().unwrap().to_string_lossy());
    let chunks = chunk_text(&content, &path_str);
    
    for chunk in chunks {
        if chunk.len() < 20 { continue; }
        match engines.embed(&chunk) {
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
    let mut engines = state.engines.lock().unwrap();
    let mut indexed_files = Vec::new();
    
    for entry in walkdir::WalkDir::new(&state.doc_folder)
        .into_iter()
        .filter_map(|e| e.ok()) 
    {
        if entry.file_type().is_file() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("md") {
                process_file(path, &mut engines, &mut store);
                indexed_files.push(path.to_string_lossy().to_string());
            }
        }
    }
    
    save_index(&store, &state.index_file);
    indexed_files
}

// --- HANDLERS HTTP ---
async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        message:"Server is running".to_string(),
        source: vec![],
    })
}

async fn query(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, StatusCode> {
    let mut engines = state.engines.lock().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let store = state.index.lock().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Embedding question
    let q_emb = engines.embed(&payload.question)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Recherche
    let results = search_index(&store, &q_emb, 3);
    
    if results.is_empty() {
        return Ok(Json(QueryResponse {
            status: "no_context".to_string(),
             data:"Aucun document pertinent trouvé dans la base de connaissances.".to_string(),
            source: vec![],
        }));
    }
    
    let sources: Vec<String> = results.iter().map(|(_, s)| format!("Score: {:.2}", s)).collect();
    let context_str: String = results.iter().map(|(c, _)| c.clone()).collect::<Vec<_>>().join("\n---\n");
    
    // Génération réponse
    let prompt = format!(
        "Tu es un assistant technique professionnel. Réponds en français de manière concise en te basant UNIQUEMENT sur le CONTEXTE ci-dessous.\n\nCONTEXTE:\n{}\n\nQUESTION:\n{}\n\nRÉPONSE:",
        context_str, payload.question
    );
    
    let answer = engines.generate(&prompt)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(QueryResponse {
        status: "success".to_string(),
         data:answer,
        source: sources,
    }))
}

async fn reindex(
    State(state): State<Arc<AppState>>,
) -> Json<ReindexResponse> {
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
    // Initialisation logs
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("rag_api_server=info".parse().unwrap())
        )
        .init();

    // Configuration depuis ENV ou défauts
    let doc_folder = std::env::var("DOC_FOLDER").unwrap_or_else(|_| DEFAULT_DOC_FOLDER.to_string());
    let index_file = std::env::var("INDEX_FILE").unwrap_or_else(|_| DEFAULT_INDEX_FILE.to_string());
    let llm_model = std::env::var("LLM_MODEL").unwrap_or_else(|_| DEFAULT_LLM_MODEL.to_string());
    let embed_model = std::env::var("EMBED_MODEL").unwrap_or_else(|_| DEFAULT_EMBED_MODEL.to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());

    info!("Démarrage du serveur RAG API...");
    info!("Dossier documents: {}", doc_folder);
    info!("Port HTTP: {}", port);

    // Initialisation moteurs
    let engines = Engines::new(&llm_model, &embed_model)
        .context("Échec initialisation moteurs")?;
    
    // Chargement index
    let index = load_index(&index_file);
    info!("Index chargé: {} chunks", index.chunks.len());

    // État partagé
    let app_state = Arc::new(AppState {
        engines: Mutex::new(engines),
        index: Mutex::new(index),
        doc_folder: PathBuf::from(doc_folder),
        index_file,
    });

    // Routes
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/query", post(query))
        .route("/reindex", post(reindex))
        .with_state(app_state);

    // Serveur
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
        .await?;
    
    info!("Serveur écoute sur http://0.0.0.0:{}", port);
    
    axum::serve(listener, app).await?;
    
    Ok(())
}