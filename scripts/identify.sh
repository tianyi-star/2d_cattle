#!/usr/bin/env bash
# Nearest enrolled identity for one query image.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

query="${1:-$(require_key query_image)}"
memory="$(require_key memory)"
checkpoint="$(require_key encoder_checkpoint)"
config="$(require_key config)"

cd "${ROOT}"
python3 -m src.self_sup_dynamic identify \
  --query "${query}" \
  --memory "${memory}" \
  --checkpoint "${checkpoint}" \
  --config "${config}"
