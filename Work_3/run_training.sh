#!/usr/bin/env bash

# Paper-aligned training wrapper for the current iSFS implementation.
# This script runs the outer block-coordinate-descent loop in `run_training.py`,
# which alternates between:
#   1. fixing the class-cost vector C and solving for alpha, beta
#   2. fixing alpha, beta and updating C with the trust-region step
# The default dataset is the real blockwise-missing CSV collection in `data/`.
# Override any Python CLI option by passing it here, for example:
#   ./run_training.sh --dataset synthetic
#   ./run_training.sh --max-iter 20 --block-a-max-iter 40

set -euo pipefail

python3 run_training.py \
  --dataset real \
  --max-iter 12 \
  --block-a-max-iter 25 \
  --alpha-subproblem-max-iter 25 \
  --beta-subproblem-max-iter 25 \
  "$@"
