#!/usr/bin/env bash
# Enroll identity prototypes from the F-ODS manifest and an encoder spec.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

manifest="$(require_key fods_manifest)"
memory="$(require_key memory)"
checkpoint="$(require_key encoder_checkpoint)"
config="$(require_key config)"

cd "${ROOT}"
python3 -m src.self_sup_dynamic enroll \
  --manifest "${manifest}" \
  --memory "${memory}" \
  --checkpoint "${checkpoint}" \
  --config "${config}"
