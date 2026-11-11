#!/usr/bin/env bash
# Balance the identity manifest with the F-ODS step.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

manifest="$(require_key manifest)"
output="$(require_key fods_manifest)"
config="$(require_key config)"

cd "${ROOT}"
python3 -m src.f_ods_dataset \
  --manifest "${manifest}" \
  --output "${output}" \
  --config "${config}"
