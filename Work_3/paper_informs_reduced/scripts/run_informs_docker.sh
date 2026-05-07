#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PAPER_DIR}/.." && pwd)"
IMAGE_NAME="${BCD_INFORMS_IMAGE:-bcd-informs-r:latest}"

docker build \
  -t "${IMAGE_NAME}" \
  -f "${PAPER_DIR}/environment/Dockerfile" \
  "${PAPER_DIR}/environment"

docker run --rm \
  -v "${REPO_ROOT}:/work" \
  -w /work \
  "${IMAGE_NAME}" \
  Rscript paper_informs_reduced/scripts/run_informs_main_data.R "$@"
