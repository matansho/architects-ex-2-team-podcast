#!/usr/bin/env bash
#
# Download prebuilt Stage 2 index (+ optional PDF chunk cache) for teammates.
# Avoids a full Docling + E5 rebuild.
#
# Usage:
#   source scripts/activate.sh   # optional
#   bash scripts/fetch_index.sh
#   bash scripts/fetch_index.sh --with-pdf-cache
#
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${INDEX_RELEASE_REPO:-matansho/architects-ex-2-team-podcast}"
TAG="${INDEX_RELEASE_TAG:-stage2-index-v1}"
BASE="https://github.com/${REPO}/releases/download/${TAG}"

WITH_PDF_CACHE=0
for arg in "$@"; do
  case "$arg" in
    --with-pdf-cache) WITH_PDF_CACHE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
  esac
done

mkdir -p dist data

download() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "already have $dest"
    return 0
  fi
  echo "Downloading $url"
  curl -fL --retry 3 --retry-delay 2 -o "$dest" "$url"
}

verify() {
  local file="$1" expect="$2"
  local got
  got="$(shasum -a 256 "$file" | awk '{print $1}')"
  if [[ "$got" != "$expect" ]]; then
    echo "SHA256 mismatch for $file" >&2
    echo "  expected $expect" >&2
    echo "  got      $got" >&2
    exit 1
  fi
}

INDEX_SHA="1e751212defe2194edea8a5a803e12bc69c17e3fed69752fee6b3b3c38090515"
CACHE_SHA="637abf1cfd4d5c41189a4bcd1ab55197b3dc568a94e63f5651f6b338e1c47c5d"

download "${BASE}/harel-rag-index-v1.tar.gz" dist/harel-rag-index-v1.tar.gz
verify dist/harel-rag-index-v1.tar.gz "$INDEX_SHA"

echo "Extracting index → data/index/"
rm -rf data/index
mkdir -p data
tar -xzf dist/harel-rag-index-v1.tar.gz -C data
test -f data/index/embeddings.npy
test -f data/index/vectors.jsonl
test -f data/index/config.json
echo "  OK ($(python3 -c 'import json; from pathlib import Path; print(sum(1 for _ in open(\"data/index/vectors.jsonl\")), \"vectors\")'))"

if [[ "$WITH_PDF_CACHE" -eq 1 ]]; then
  download "${BASE}/harel-pdf-chunks-cache-v1.tar.gz" dist/harel-pdf-chunks-cache-v1.tar.gz
  verify dist/harel-pdf-chunks-cache-v1.tar.gz "$CACHE_SHA"
  echo "Extracting PDF chunk cache → data/cache/pdf_chunks/"
  mkdir -p data/cache
  rm -rf data/cache/pdf_chunks
  tar -xzf dist/harel-pdf-chunks-cache-v1.tar.gz -C data/cache
  echo "  OK"
fi

echo
echo "Ready. Table descriptions are in git (artifacts/table_descriptions/)."
echo "Run RAG, e.g.:"
echo "  python rag_runner.py --retrieve rerank --route --cite passages --limit 1"
