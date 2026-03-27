# RAG API Server

Serveur API RAG (Retrieval-Augmented Generation) haute performance, écrit en **Rust**, fonctionnant entièrement en local.

## 🎯 Vue d'ensemble

Ce projet fournit une API REST qui permet d'interroger une base de documents Markdown en langage naturel. Le système combine :

- **Recherche vectorielle** pour trouver les passages pertinents
- **LLM local** (Phi-3 via Llamafile) pour générer des réponses contextuelles
- **100% local** : aucune donnée ne quitte votre infrastructure

## 🏗️ Architecture

```
┌─────────────────┐      HTTP      ┌─────────────────┐
│   Rust API      │ ─────────────> │   Llamafile     │
│   (Port 8080)   │   (Port 8081)  │   (Port 8081)   │
│                 │                │                 │
│ - Embeddings    │                │ - LLM Phi-3     │
│ - Indexation    │                │ - Génération    │
│ - HTTP Server   │                │                 │
└─────────────────┘                └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Documents     │
│   (./docs/*.md) │
└─────────────────┘
```

## ⚡ Prérequis

| Composant | Version | Installation |
|-----------|---------|--------------|
| Rust | 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| Llamafile | Latest | [Téléchargement](https://github.com/Mozilla-Ocho/llamafile) |
| Linux | Qualquer | Ubuntu, Debian, Arch, etc. |

## 📦 Installation

### 1. Cloner et compiler

```bash
git clone https://github.com/loicdurand/rag_api_server-rust.git
cd rag_api_server
cargo build --release
```

### 2. Télécharger les modèles

```bash
mkdir -p models

# Modèle d'embedding (ONNX)
wget -O models/all-MiniLM-L6-v2.onnx \
  https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx

# Tokenizer
wget -O models/tokenizer.json \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json

# LLM Phi-3 (Llamafile)
wget -O models/phi-3-mini-4k-instruct.Q4_K_M.llamafile \
  https://huggingface.co/Mozilla/phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct.Q4_K_M.gguf

chmod +x models/phi-3-mini-4k-instruct.Q4_K_M.llamafile
```

### 3. Préparer les documents

```bash
mkdir -p docs
# Placez vos fichiers .md dans le dossier ./docs
```

## 🚀 Démarrage

### Terminal 1 : Lancer Llamafile (LLM)

```bash
./models/phi-3-mini-4k-instruct.Q4_K_M.llamafile \
  --server \
  --port 8081 \
  --nobrowser \
  --n-gpu-layers 0
```

| Option | Description |
|--------|-------------|
| `--n-gpu-layers 0` | Force l'usage CPU (augmentez si GPU disponible) |
| `--nobrowser` | N'ouvre pas le navigateur automatiquement |

### Terminal 2 : Lancer l'API Rust

```bash
./target/release/rag_api_server
```

## 📡 Endpoints API

### `GET /health`

Vérifie l'état du serveur et la connexion à Llamafile.

```bash
curl http://localhost:8080/health
```

**Réponse :**
```json
{
  "status": "ok",
  "message": "Server and LLM ready",
  "source": []
}
```

---

### `POST /query`

Pose une question sur la base de documents.

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les effectifs ?"}'
```

**Réponse :**
```json
{
  "status": "success",
  "data": "Les effectifs sont de 680 collaborateurs...",
  "source": ["Score: 0.87", "Score: 0.72"]
}
```

| Statut | Description |
|--------|-------------|
| `success` | Réponse générée avec succès |
| `no_context` | Aucun document pertinent trouvé |

---

### `POST /reindex`

Force la réindexation des documents Markdown.

```bash
curl -X POST http://localhost:8080/reindex
```

**Réponse :**
```json
{
  "status": "success",
  "message": "5 documents indexés",
  "source": ["./docs/file1.md", "./docs/file2.md"]
}
```

## ⚙️ Configuration

Variables d'environnement disponibles :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `DOC_FOLDER` | `./docs` | Dossier des documents à indexer |
| `INDEX_FILE` | `rag_index.json` | Fichier de persistance de l'index |
| `EMBED_MODEL` | `./models/all-MiniLM-L6-v2.onnx` | Modèle d'embedding ONNX |
| `TOKENIZER` | `./models/tokenizer.json` | Fichier tokenizer |
| `PORT` | `8080` | Port d'écoute de l'API |
| `LLAMAFILE_URL` | `http://localhost:8081` | URL du serveur Llamafile |

### Exemple

```bash
export DOC_FOLDER="/srv/rag/docs"
export PORT="9000"
./target/release/rag_api_server
```

## 🔧 Déploiement (systemd)

Créez `/etc/systemd/system/rag-api.service` :

```ini
[Unit]
Description=RAG API Server
After=network.target

[Service]
Type=simple
User=rag
WorkingDirectory=/opt/rag_api_server
Environment="DOC_FOLDER=/opt/rag_api_server/docs"
Environment="PORT=8080"
ExecStart=/opt/rag_api_server/target/release/rag_api_server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable rag-api
sudo systemctl start rag-api
sudo systemctl status rag-api
```

## 📊 Performances

| Métrique | Valeur (CPU) | Valeur (GPU) |
|----------|--------------|--------------|
| Embedding | ~50ms/chunk | ~10ms/chunk |
| Génération | ~5 tokens/s | ~30 tokens/s |
| Recherche | <10ms | <10ms |
| RAM utilisée | ~4 Go | ~4 Go |

### Optimisations

- **Seuil de similarité** : Ajustez `0.3` dans `search_index()` pour plus/moins de résultats
- **Chunk size** : Modifiez `CHUNK_SIZE` (défaut: 400 caractères)
- **Max tokens** : Ajustez `MAX_LENGTH` (défaut: 128 tokens)

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| `no_context` | Lancez `POST /reindex` après avoir ajouté des documents |
| `LLM unreachable` | Vérifiez que Llamafile tourne sur le port 8081 |
| `Missing Input: token_type_ids` | Vérifiez que le modèle ONNX est correct |
| Lent | Augmentez `--n-gpu-layers` sur Llamafile si GPU disponible |

## 📁 Structure du projet

```
rag_api_server/
├── Cargo.toml
├── src/
│   └── main.rs
├── models/
│   ├── all-MiniLM-L6-v2.onnx
│   ├── tokenizer.json
│   └── phi-3-mini-4k-instruct.Q4_K_M.llamafile
├── docs/
│   └── *.md
├── rag_index.json (généré)
└── target/
    └── release/
        └── rag_api_server
```

## 🔒 Sécurité

- **Aucune donnée externe** : Tout reste en local
- **Pas d'authentification** : Ajoutez un reverse proxy (Nginx) pour la production
- **HTTPS** : Configurez via Nginx/Traefik en production

## 📝 Licence

MIT

---

**Développé avec ❤️ en Rust**
