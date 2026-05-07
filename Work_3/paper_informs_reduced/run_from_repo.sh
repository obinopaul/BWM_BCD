#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./paper_informs_reduced/run_from_repo.sh [all|bike|house|adult|scalability]

Environment:
  INFORMS_DOCKER_IMAGE        Docker image tag to build/use.
                              Default: bcd-paper-informs-reduced
  INFORMS_SKIP_DOCKER_BUILD   Set to 1 to skip docker build.

Examples:
  ./paper_informs_reduced/run_from_repo.sh bike
  ./paper_informs_reduced/run_from_repo.sh all
  INFORMS_SKIP_DOCKER_BUILD=1 ./paper_informs_reduced/run_from_repo.sh adult
EOF
}

target="${1:-all}"
case "$target" in
  -h|--help)
    usage
    exit 0
    ;;
  all|bike|house|adult|scalability)
    ;;
  *)
    echo "Unknown target: $target" >&2
    usage >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
image="${INFORMS_DOCKER_IMAGE:-bcd-paper-informs-reduced}"
results_dir="$script_dir/results"

if [[ ! -d "$repo_root/main_data" ]]; then
  echo "Missing main_data directory at $repo_root/main_data" >&2
  exit 1
fi

mkdir -p "$results_dir"

if [[ "${INFORMS_SKIP_DOCKER_BUILD:-0}" != "1" ]]; then
  docker build \
    -t "$image" \
    -f "$script_dir/environment/Dockerfile" \
    "$script_dir/environment"
fi

common_sources="source('/root/capsule/code/Load_packages.R'); source('/root/capsule/code/Preliminary_functions.R'); source('/root/capsule/code/Benchmarks.R'); source('/root/capsule/code/BRM_functions.R')"

case "$target" in
  all)
    run_cmd="cd /root/capsule/code && ./run"
    ;;
  bike)
    run_cmd="Rscript -e \"$common_sources; source('/root/capsule/code/bike_.R')\""
    ;;
  house)
    run_cmd="Rscript -e \"$common_sources; source('/root/capsule/code/house_.R')\""
    ;;
  adult)
    run_cmd="Rscript -e \"$common_sources; source('/root/capsule/code/adult_.R')\""
    ;;
  scalability)
    run_cmd="Rscript -e \"$common_sources; source('/root/capsule/code/scalability.R')\""
    ;;
esac

docker run --rm \
  -v "$script_dir/code:/root/capsule/code:ro" \
  -v "$repo_root/main_data:/root/capsule/data:ro" \
  -v "$results_dir:/root/capsule/results" \
  "$image" \
  bash -lc "$run_cmd"

