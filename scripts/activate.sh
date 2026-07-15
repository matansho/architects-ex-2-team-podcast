#!/usr/bin/env bash
# Source this file to activate the venv and load .env:
#   source scripts/activate.sh

if [[ -n "${BASH_VERSION:-}" ]]; then
  _src="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  # zsh: %x expands to the path of the file being sourced
  _src="${(%):-%x}"
else
  _src="$0"
fi
ROOT="$(cd "$(dirname "$_src")/.." && pwd)"
source "$ROOT/.venv/bin/activate"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
  echo "Loaded $ROOT/.env"
else
  echo "No .env found — copy .env.example to .env and add NEBIUS_API_KEY"
fi
