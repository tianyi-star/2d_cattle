#!/bin/sh
# Ingest already background-removed cattle folders:
#   READY_DIR/<identity_id>/*.png  (96 identities)
# into the SAM-arm pattern directory + manifest.
#
# Usage:
#   scripts/prepare_ready.sh READY_DIR [sam|bbox]
#
# The identities are partitioned disjointly into train / val / test by default
# (64 / 16 / 16 of 96). The manifest, a JSON record of the split and a
# Table-1-ready Markdown summary are written beside the CSV. Nothing here trains
# or evaluates anything.
#
# Override the split with environment variables:
#   SPLIT_MODE=identity|within   default identity (within leaks identities; avoid)
#   SEED=0                       seed for the identity shuffle
#   TRAIN_FRAC=0.6667 VAL_FRAC=0.1667 TEST_FRAC=0.1667
#   GALLERY_FRACTION=0.5         share of each test animal's images used as gallery
#   MIN_TEST_IMAGES=2            minimum frames for an animal to be test-eligible
#   TRAIN_RATIO=0.6 GALLERY_RATIO=0.2   legacy 'within' mode only
#   PARTITION_FROM=path/to/x.split.json  reuse the animal partition recorded there, so
#                                        both crop arms hold out the same animals
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "${ROOT}/scripts/lib.sh"

ready="${1:-}"
crop="${2:-sam}"
if [ -z "${ready}" ]; then
  echo "Usage: scripts/prepare_ready.sh /path/to/bg_removed_cattle [sam|bbox]" >&2
  exit 1
fi
case "${crop}" in
  sam|bbox) ;;
  *)
    echo "crop must be sam or bbox" >&2
    exit 1
    ;;
esac

SPLIT_MODE="${SPLIT_MODE:-identity}"
SEED="${SEED:-0}"
TRAIN_FRAC="${TRAIN_FRAC:-0.6666666666666666}"
VAL_FRAC="${VAL_FRAC:-0.16666666666666666}"
TEST_FRAC="${TEST_FRAC:-0.16666666666666666}"
GALLERY_FRACTION="${GALLERY_FRACTION:-0.5}"
MIN_TEST_IMAGES="${MIN_TEST_IMAGES:-2}"
TRAIN_RATIO="${TRAIN_RATIO:-0.6}"
GALLERY_RATIO="${GALLERY_RATIO:-0.2}"

case "${SPLIT_MODE}" in
  identity|within) ;;
  *) echo "SPLIT_MODE must be identity or within" >&2; exit 1 ;;
esac
if [ "${SPLIT_MODE}" = "within" ]; then
  echo "warning: SPLIT_MODE=within shuffles each animal's own images across" >&2
  echo "         train/gallery/query. Every identity appears in every split, so any" >&2
  echo "         Rank-1 from this manifest is inflated. Do not report it." >&2
fi

case "${crop}" in
  bbox)
    out_dir="$(require_key bbox_crop_dir)"
    out_manifest="$(require_key bbox_manifest)"
    ;;
  *)
    out_dir="$(require_key pattern_dir)"
    out_manifest="$(require_key pattern_manifest)"
    ;;
esac

if [ "${SPLIT_MODE}" = "identity" ]; then
  split_flags="--split-mode identity --seed ${SEED} \
--train-frac ${TRAIN_FRAC} --val-frac ${VAL_FRAC} --test-frac ${TEST_FRAC} \
--gallery-fraction ${GALLERY_FRACTION} --min-test-images ${MIN_TEST_IMAGES}"
  if [ -n "${PARTITION_FROM:-}" ]; then
    split_flags="${split_flags} --partition-from ${PARTITION_FROM}"
  fi
else
  split_flags="--split-mode within --seed ${SEED} \
--train-ratio ${TRAIN_RATIO} --gallery-ratio ${GALLERY_RATIO}"
fi

cd "${ROOT}"
# shellcheck disable=SC2086
python3 -m src.preprocess \
  --ready-dir "${ready}" \
  --output-dir "${out_dir}" \
  --output-manifest "${out_manifest}" \
  --crop "${crop}" \
  ${split_flags}
echo
echo "Split manifest:   ${out_manifest%.csv}.split.json"
echo "Table-1 summary:  ${out_manifest%.csv}.table1.md"
