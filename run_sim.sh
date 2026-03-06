#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
SKIP_INSTALL=0
FORCE_INSTALL=0

usage() {
  cat <<'EOF'
Usage: ./run_sim.sh [options]

Bootstraps the virtual environment, builds the Rust extension, and launches
the Streamlit dashboard.

Options:
  --skip-install   Launch without installing/updating dependencies first
  --force-install  Reinstall Python dependencies and rebuild the extension
  --help           Show this help message

Environment:
  PYTHON_BIN       Python executable to use for the virtualenv (default: python3)
  STREAMLIT_PORT   Streamlit port (default: 8501)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --force-install)
      FORCE_INSTALL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo
      usage
      exit 1
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command "${PYTHON_BIN}"
require_command cargo
require_command rustc

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

ensure_python_package() {
  local module_name="$1"
  local package_spec="$2"

  if [[ "${FORCE_INSTALL}" -eq 1 ]]; then
    python -m pip install "${package_spec}"
    return
  fi

  if ! python -c "import ${module_name}" >/dev/null 2>&1; then
    python -m pip install "${package_spec}"
  fi
}

if [[ "${SKIP_INSTALL}" -eq 0 ]]; then
  echo "Ensuring Python dependencies are installed"
  ensure_python_package maturin "maturin>=1.7,<2.0"
  ensure_python_package numpy "numpy>=2.2"
  ensure_python_package pandas "pandas>=2.2"
  ensure_python_package plotly "plotly>=5.24"
  ensure_python_package streamlit "streamlit>=1.33"

  echo "Building Rust extension with maturin"
  if [[ "${FORCE_INSTALL}" -eq 1 ]]; then
    maturin develop
  else
    maturin develop >/dev/null
  fi
else
  echo "Skipping install step"
fi

echo "Launching Streamlit on http://localhost:${STREAMLIT_PORT}"
exec streamlit run "${ROOT_DIR}/streamlit_app.py" --server.port "${STREAMLIT_PORT}"
