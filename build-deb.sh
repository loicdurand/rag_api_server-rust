#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build-deb.sh — Construit le paquet .deb de rag-api-server
#
# Usage :
#   ./build-deb.sh              # build normal
#   ./build-deb.sh --no-strip   # désactive le strip (debug plus facile)
#   ./build-deb.sh --fast       # utilise un build existant (sans recompiler)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Couleurs ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERR]${RESET}   $*" >&2; }
die()     { error "$*"; exit 1; }

# ── Répertoire racine du projet ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Drapeaux ──────────────────────────────────────────────────────────────────
FAST=false
EXTRA_DEB_FLAGS=""

for arg in "$@"; do
    case "$arg" in
        --fast)      FAST=true ;;
        --no-strip)  EXTRA_DEB_FLAGS="$EXTRA_DEB_FLAGS --no-strip" ;;
        --help|-h)
            echo "Usage: $0 [--fast] [--no-strip]"
            echo ""
            echo "  --fast       Saute la compilation (utilise le binaire release existant)"
            echo "  --no-strip   Ne pas supprimer les symboles de debug du binaire"
            exit 0
            ;;
        *)
            die "Argument inconnu : $arg  (utilisez --help pour l'aide)"
            ;;
    esac
done

# ── Vérifications préalables ──────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}   Build .deb — rag-api-server${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo ""

# Cargo
if ! command -v cargo &>/dev/null; then
    die "cargo non trouvé. Installez Rust : https://rustup.rs"
fi
info "Rust/Cargo : $(cargo --version)"

# cargo-deb
if ! cargo deb --version &>/dev/null 2>&1; then
    warn "cargo-deb non trouvé. Installation en cours..."
    cargo install cargo-deb --locked
    success "cargo-deb installé : $(cargo deb --version)"
else
    info "cargo-deb : $(cargo deb --version)"
fi

# ar (requis pour assembler le .deb)
if ! command -v ar &>/dev/null; then
    die "'ar' non trouvé. Installez binutils : sudo apt install binutils"
fi

# fakeroot (optionnel mais recommandé)
if ! command -v fakeroot &>/dev/null; then
    warn "'fakeroot' non trouvé — le build peut nécessiter les droits root."
fi

# ── Vérification du Cargo.toml ────────────────────────────────────────────────
if [ ! -f "Cargo.toml" ]; then
    die "Cargo.toml introuvable. Lancez ce script depuis la racine du projet."
fi

if ! grep -q '\[package.metadata.deb\]' Cargo.toml; then
    die "Section [package.metadata.deb] absente de Cargo.toml."
fi

# ── Compilation release ───────────────────────────────────────────────────────
if [ "$FAST" = false ]; then
    info "Compilation en mode release..."
    cargo build --release
    success "Compilation terminée."
else
    BINARY="target/release/rag_api_server"
    if [ ! -f "$BINARY" ]; then
        die "--fast utilisé mais $BINARY introuvable. Lancez sans --fast d'abord."
    fi
    warn "--fast : compilation ignorée, utilisation du binaire existant."
fi

# ── Construction du .deb ──────────────────────────────────────────────────────
info "Construction du paquet .deb..."

# shellcheck disable=SC2086
cargo deb --no-build $EXTRA_DEB_FLAGS

# ── Résultat ──────────────────────────────────────────────────────────────────
DEB_FILE=$(find target/debian -maxdepth 1 -name "*.deb" -printf '%T@ %p\n' 2>/dev/null \
           | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$DEB_FILE" ]; then
    die "Le fichier .deb n'a pas été trouvé dans target/debian/"
fi

DEB_SIZE=$(du -sh "$DEB_FILE" | cut -f1)

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
success "Paquet généré avec succès !"
echo -e "  ${BOLD}Fichier :${RESET} $DEB_FILE"
echo -e "  ${BOLD}Taille  :${RESET} $DEB_SIZE"
echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
echo ""

# ── Informations d'installation ───────────────────────────────────────────────
echo -e "${BOLD}Installation sur la cible Ubuntu/Debian :${RESET}"
echo ""
echo "  # Copier le paquet sur la cible, puis :"
echo "  sudo apt install ./$DEB_FILE"
echo ""
echo -e "${BOLD}Après installation :${RESET}"
echo ""
echo "  # Placer les modèles ONNX dans :"
echo "  /opt/rag-api-server/models/"
echo ""
echo "  # Placer les documents Markdown dans :"
echo "  /opt/rag-api-server/docs/"
echo ""
echo "  # Démarrer le service :"
echo "  sudo systemctl start rag-api-server"
echo "  sudo systemctl status rag-api-server"
echo ""
