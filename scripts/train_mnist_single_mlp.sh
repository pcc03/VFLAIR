#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}/src"

CONFIG_NAME="single_mnist_mlp"
GPU_ID="${1:-0}"

python main_pipeline.py --gpu "${GPU_ID}" --configs "${CONFIG_NAME}"
