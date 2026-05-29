#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm -it \
  -v "${SCRIPT_DIR}:/workspace" \
  -w /workspace \
  simpleph \
  "$@"