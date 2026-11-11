#!/usr/bin/env bash
# Self-supervised update from the unlabeled manifest.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

unlabeled="$(require_key unlabeled_manifest)"
memory="$(require_key memory)"
checkpoint="$(require_key encoder_checkpoint)"
config="$(require_key config)"

cd "${ROOT}"
python3 -m src.self_sup_dynamic update \
  --unlabeled "${unlabeled}" \
  --memory "${memory}" \
  --checkpoint "${checkpoint}" \
  --config "${config}"
