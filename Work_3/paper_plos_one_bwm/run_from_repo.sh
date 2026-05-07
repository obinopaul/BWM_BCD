#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./paper_plos_one_bwm/run_from_repo.sh [run_main_data_experiment.R options]

Examples:
  ./paper_plos_one_bwm/run_from_repo.sh --dataset bike --missing-rates 0.10 --max-iter 5
  ./paper_plos_one_bwm/run_from_repo.sh --dataset all --missing-rates 0.10,0.20

Environment:
  PLOS_BWM_R_IMAGE     Docker image to use. Default: rocker/r-ver:4.3.3
  PLOS_INSTALL_SYSTEM_DEPS
                       Set to 0 to skip apt-get installation of common R build deps.
  PLOS_INSTALL_DEPS    Set to 0 to skip the pre-install dependency step.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
image="${PLOS_BWM_R_IMAGE:-rocker/r-ver:4.3.3}"

if [[ ! -d "$repo_root/main_data" ]]; then
  echo "Missing main_data directory at $repo_root/main_data" >&2
  exit 1
fi

mkdir -p "$script_dir/results"

docker run --rm \
  -v "$repo_root:/workspace" \
  -w /workspace \
  -e PLOS_INSTALL_SYSTEM_DEPS="${PLOS_INSTALL_SYSTEM_DEPS:-1}" \
  -e PLOS_INSTALL_DEPS="${PLOS_INSTALL_DEPS:-1}" \
  "$image" \
  bash -lc '
    set -euo pipefail
    if [[ "${PLOS_INSTALL_SYSTEM_DEPS:-1}" != "0" ]] && command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update
      apt-get install -y --no-install-recommends \
        cmake \
        pkg-config \
        libcurl4-openssl-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libfribidi-dev \
        libgit2-dev \
        libharfbuzz-dev \
        libicu-dev \
        libjpeg-dev \
        libpng-dev \
        libssl-dev \
        libtiff5-dev \
        libxml2-dev
      rm -rf /var/lib/apt/lists/*
    fi
    if [[ "${PLOS_INSTALL_DEPS:-1}" != "0" ]]; then
      Rscript -e "options(repos = c(CRAN = \"https://cloud.r-project.org\")); pkgs <- c(\"caret\", \"rlang\", \"MASS\", \"corrplot\", \"glmnet\", \"binaryLogic\", \"ade4\", \"naniar\", \"mvtnorm\", \"devtools\", \"factoextra\", \"nnet\"); missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]; if (length(missing)) install.packages(missing)"
    fi
    Rscript paper_plos_one_bwm/run_main_data_experiment.R "$@"
  ' bash "$@"
