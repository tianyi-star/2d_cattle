#!/usr/bin/env bash
# Train and evaluate the six cells (2 crop policies x 3 backbones) with seeded runs,
# and write the measured table to a timestamped results/ directory.
#
# Usage (on the GPU box, from the repository root):
#   scripts/run_all_cells.sh --data-root /path/to/identity_crops
#
# Full list of options: scripts/run_all_cells.sh --help
#
# Two modes:
#   --data-mode prepared (default)  either of:
#                                     <data-root>/<identity_id>/*     images as stored.
#                                       No SAM mask and no box crop are built.
#                                       The run is labeled crop "id".
#                                     <data-root>/sam/<identity_id>/*
#                                     <data-root>/bbox/<identity_id>/*
#                                       only when those trees already exist and
#                                       --crops asks for them.
#   --data-mode raw                 <data-root>/*.{jpg,png,...} UAV frames plus
#                                   <data-root>/manifest.csv from your annotation step;
#                                   this script then runs YOLO + SAM/bbox preprocessing.
#
# Guarantees this script makes:
#   * Seeded: one run per seed under outputs/seed_<seed>/<crop>/<model>/.
#   * Refuses to overwrite: the results/ run directory is created exclusively, and
#     it aborts if any target run directory already holds files.
#   * No placeholder numbers: every figure comes from src.evaluate's metrics.json,
#     and src.aggregate rejects stale, incomplete, or placeholder-tagged files.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "${ROOT}"

CONFIG="configs/experiment.yaml"
DATA_ROOT=""
DATA_MODE="prepared"
SEEDS="0"
SPLIT_MODE="identity"
TRAIN_FRAC="0.6666666666666666"
VAL_FRAC="0.16666666666666666"
TEST_FRAC="0.16666666666666666"
GALLERY_FRACTION="0.5"
MIN_TEST_IMAGES="2"
TRAIN_RATIO="0.6"
GALLERY_RATIO="0.2"
OUTPUTS_ROOT="outputs"
RESULTS_ROOT="results"
YOLO_WEIGHTS=""
SAM_WEIGHTS=""
SAM_MODEL_TYPE="vit_b"
SAM_BACKEND="auto"
SAM_DEVICE=""
SAM_CACHE_DIR="data/sam_mask_cache"
CROPS="sam bbox"
MODELS="yolo_arcface pruned_vit_arcface wang_metric"
DRY_RUN=0
MANIFEST_ONLY=0

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --data-root DIR       Identity folders (prepared) or UAV frames (raw). Required.
  --data-mode MODE      prepared | raw. Default prepared.
  --config FILE         Experiment YAML. Default configs/experiment.yaml.
  --seeds "0 1 2"       Space-separated seeds. Default "0". Use 3+ for CEA replication.
  --split-mode MODE     identity (default) | within.
                        identity assigns whole animals to train/val/test, so no identity
                        appears in two partitions. within is the legacy image-level split:
                        every identity lands in train, gallery AND query, which leaks
                        identity and near-duplicate frames and inflates Rank-1. Use it
                        only to rebuild an old manifest for comparison.
  --train-frac F        Fraction of identities in train. Default 0.6667 (64/96).
  --val-frac F          Fraction of identities in validation. Default 0.1667 (16/96).
  --test-frac F         Fraction of identities in test. Default 0.1667 (16/96).
                        The three fractions must sum to 1.
  --gallery-fraction F  Share of each test identity's images used as gallery; the rest
                        become queries. Default 0.5. Both come from held-out animals.
  --min-test-images N   Minimum images for an identity to be eligible for test.
                        Default 2 (one gallery view, one query view).
  --manifest-only       Write the split manifest and its Table-1 summary, print the
                        identity/image accounting, and exit before any training. Use it
                        to check the partition before committing to a long run.
  --train-ratio F       Legacy 'within' mode only: train fraction per identity. Default 0.6.
  --gallery-ratio F     Legacy 'within' mode only: gallery fraction per identity. Default 0.2.
  --outputs-root DIR    Checkpoints and per-cell metrics. Default outputs.
  --results-root DIR    Where the timestamped run directory is created. Default results.
  --crops "sam bbox"    Crop arms to run when sam/ and bbox/ trees exist.
                        Default both. An identity-folder data-root (no sam/ or
                        bbox/ child) is always crop "id": the images are used
                        as stored, and SAM is not run.
  --models "..."        Backbones to run. Default all three.
  --yolo-weights FILE   Required only for --data-mode raw.
  --sam-weights FILE    Required only for --data-mode raw with the sam arm.
  --sam-model-type T    vit_h | vit_l | vit_b | vit_t. Default vit_b.
  --sam-backend B       auto | segment_anything | mobile_sam | ultralytics. Default auto.
  --sam-device DEV      cuda | cpu. Default autodetect.
  --sam-cache-dir DIR   Mask cache root. Default data/sam_mask_cache.
  --dry-run             Print the plan and exit without training.
  -h, --help            Show this message.
EOF
}

die() { echo "error: $*" >&2; exit 1; }
log() { printf '\n=== %s ===\n' "$*"; }

is_identity_tree() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
if not root.is_dir():
    raise SystemExit(1)
for child in root.iterdir():
    if not child.is_dir() or child.name in {"sam", "bbox"}:
        continue
    for path in child.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            raise SystemExit(0)
raise SystemExit(1)
PY
}

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root) DATA_ROOT="${2:?}"; shift 2 ;;
    --data-mode) DATA_MODE="${2:?}"; shift 2 ;;
    --config) CONFIG="${2:?}"; shift 2 ;;
    --seeds) SEEDS="${2:?}"; shift 2 ;;
    --split-mode) SPLIT_MODE="${2:?}"; shift 2 ;;
    --train-frac) TRAIN_FRAC="${2:?}"; shift 2 ;;
    --val-frac) VAL_FRAC="${2:?}"; shift 2 ;;
    --test-frac) TEST_FRAC="${2:?}"; shift 2 ;;
    --gallery-fraction) GALLERY_FRACTION="${2:?}"; shift 2 ;;
    --min-test-images) MIN_TEST_IMAGES="${2:?}"; shift 2 ;;
    --manifest-only) MANIFEST_ONLY=1; shift ;;
    --train-ratio) TRAIN_RATIO="${2:?}"; shift 2 ;;
    --gallery-ratio) GALLERY_RATIO="${2:?}"; shift 2 ;;
    --outputs-root) OUTPUTS_ROOT="${2:?}"; shift 2 ;;
    --results-root) RESULTS_ROOT="${2:?}"; shift 2 ;;
    --crops) CROPS="${2:?}"; shift 2 ;;
    --models) MODELS="${2:?}"; shift 2 ;;
    --yolo-weights) YOLO_WEIGHTS="${2:?}"; shift 2 ;;
    --sam-weights) SAM_WEIGHTS="${2:?}"; shift 2 ;;
    --sam-model-type) SAM_MODEL_TYPE="${2:?}"; shift 2 ;;
    --sam-backend) SAM_BACKEND="${2:?}"; shift 2 ;;
    --sam-device) SAM_DEVICE="${2:?}"; shift 2 ;;
    --sam-cache-dir) SAM_CACHE_DIR="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

[ -n "${DATA_ROOT}" ] || die "--data-root is required (see --help)"
[ -d "${DATA_ROOT}" ] || die "--data-root does not exist: ${DATA_ROOT}"
[ -f "${CONFIG}" ] || die "--config not found: ${CONFIG}"
case "${DATA_MODE}" in prepared|raw) ;; *) die "--data-mode must be prepared or raw" ;; esac
case "${SPLIT_MODE}" in identity|within) ;; *) die "--split-mode must be identity or within" ;; esac
if [ "${SPLIT_MODE}" = "within" ]; then
  echo "warning: --split-mode within shuffles each identity's own images across" >&2
  echo "         train/gallery/query. Every identity appears in every split, so Rank-1" >&2
  echo "         from this manifest is inflated. Do not report it as identification." >&2
fi

# Fractions are checked by src.data, but fail here so a long run never starts on
# a partition that cannot be built.
python3 - "${TRAIN_FRAC}" "${VAL_FRAC}" "${TEST_FRAC}" <<'PY' || exit 2
import sys
values = [float(v) for v in sys.argv[1:4]]
if any(v <= 0 for v in values):
    raise SystemExit(f"error: split fractions must be positive, got {values}")
if abs(sum(values) - 1.0) > 1e-6:
    raise SystemExit(f"error: split fractions must sum to 1, got {values} (sum {sum(values)})")
PY

# Identity folders at the data root are the images as stored. Do not look for a
# sam/ or bbox/ tree, and do not build either crop.
if [ "${DATA_MODE}" = "prepared" ] && [ ! -d "${DATA_ROOT}/sam" ] && [ ! -d "${DATA_ROOT}/bbox" ] && is_identity_tree "${DATA_ROOT}"; then
  if [ "${CROPS}" != "id" ]; then
    echo "No sam/ or bbox/ tree under ${DATA_ROOT}."
    echo "Using the identity folders as stored. Crop label: id. SAM is not run."
  fi
  CROPS="id"
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_NAME="six_cells_${DATA_MODE}_${STAMP}"
RESULTS_DIR="${RESULTS_ROOT}/${RUN_NAME}"

# Refuse to overwrite, before anything is written anywhere: if a target run
# directory already holds files, this is a second pass over an existing run.
for model in ${MODELS}; do
  case "${model}" in
    yolo_arcface|pruned_vit_arcface|wang_metric) ;;
    *) die "unknown model: ${model}" ;;
  esac
  for seed in ${SEEDS}; do
    for crop in ${CROPS}; do
      target="${OUTPUTS_ROOT}/seed_${seed}/${crop}/${model}"
      if [ -d "${target}" ] && [ -n "$(ls -A "${target}" 2>/dev/null)" ]; then
        die "${target} already holds files. Delete it or pick another --outputs-root; this script will not overwrite a previous run."
      fi
    done
  done
done

# Atomic, exclusive creation of the results directory for this run.
mkdir -p "${RESULTS_ROOT}"
if [ -e "${RESULTS_DIR}" ]; then
  die "results directory already exists: ${RESULTS_DIR}. Refusing to overwrite a run."
fi
mkdir "${RESULTS_DIR}" || die "could not create ${RESULTS_DIR}"

# Record the exact configuration of this run before any training starts.
printf '%s\n' "${SEEDS}" > "${RESULTS_DIR}/seeds.txt"
{
  echo "run=${RUN_NAME}"
  echo "started_at_utc=${STAMP}"
  echo "data_root=$(cd "${DATA_ROOT}" && pwd)"
  echo "data_mode=${DATA_MODE}"
  echo "config=${CONFIG}"
  echo "seeds=${SEEDS}"
  echo "split_mode=${SPLIT_MODE}"
  echo "split_identity_mode=${SPLIT_MODE}"
  if [ "${SPLIT_MODE}" = "identity" ]; then
    echo "train_frac=${TRAIN_FRAC}"
    echo "val_frac=${VAL_FRAC}"
    echo "test_frac=${TEST_FRAC}"
    echo "gallery_fraction=${GALLERY_FRACTION}"
    echo "min_test_images=${MIN_TEST_IMAGES}"
  else
    echo "train_ratio=${TRAIN_RATIO}"
    echo "gallery_ratio=${GALLERY_RATIO}"
  fi
  echo "crops=${CROPS}"
  echo "models=${MODELS}"
  echo "outputs_root=${OUTPUTS_ROOT}"
  echo "sam_model_type=${SAM_MODEL_TYPE}"
  echo "sam_backend=${SAM_BACKEND}"
  echo "git_sha=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=$(python3 --version 2>&1)"
} > "${RESULTS_DIR}/run_config.txt"

STARTED_AT=$(python3 -c 'import time; print(time.time())')

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "dry run: results would go to ${RESULTS_DIR}"
  echo "seeds: ${SEEDS}; crops: ${CROPS}; models: ${MODELS}"
  echo "data: ${DATA_ROOT} (${DATA_MODE})"
  exit 0
fi

split_args() {
  # Split flags shared by every manifest build in this run.
  if [ "${SPLIT_MODE}" = "identity" ]; then
    printf '%s\n' \
      --split-mode identity \
      --train-frac "${TRAIN_FRAC}" \
      --val-frac "${VAL_FRAC}" \
      --test-frac "${TEST_FRAC}" \
      --gallery-fraction "${GALLERY_FRACTION}" \
      --min-test-images "${MIN_TEST_IMAGES}"
  else
    printf '%s\n' \
      --split-mode within \
      --train-ratio "${TRAIN_RATIO}" \
      --gallery-ratio "${GALLERY_RATIO}"
  fi
}

log "Preparing data (${DATA_MODE})"
case "${DATA_MODE}" in
  prepared)
    # Two layouts:
    #   <data-root>/<identity_id>/*            images as stored (crop label id)
    #   <data-root>/{sam,bbox}/<identity_id>/* only if those trees already exist
    # This runner does not build a SAM mask or a detector crop.
    for crop in ${CROPS}; do
      case "${crop}" in
        id) arm_root="${DATA_ROOT}" ;;
        sam) arm_root="${DATA_ROOT}/sam" ;;
        bbox) arm_root="${DATA_ROOT}/bbox" ;;
        *) die "unknown crop: ${crop} (expected id, sam, or bbox)" ;;
      esac
      if [ "${crop}" = "id" ]; then
        is_identity_tree "${arm_root}" || die "prepared mode needs ${arm_root}/<identity_id>/* .
  Put one folder per animal directly under the data root. SAM is not run."
      else
        [ -d "${arm_root}" ] || die "prepared mode needs ${arm_root}/<identity_id>/* for the ${crop} arm.
  This data root has no ${crop}/ tree. To train on identity folders as stored, pass that
  directory as --data-root and do not ask for a sam or bbox tree."
      fi
      printf '%s' "${arm_root}" > "${RESULTS_DIR}/arm_root_${crop}.txt"
    done
    if [ -d "${DATA_ROOT}/sam" ] && [ -d "${DATA_ROOT}/bbox" ]; then
      sam_abs=$(cd "${DATA_ROOT}/sam" && pwd -P)
      bbox_abs=$(cd "${DATA_ROOT}/bbox" && pwd -P)
      [ "${sam_abs}" != "${bbox_abs}" ] \
        || die "sam and bbox resolve to the same directory (${sam_abs}). The crop arms must differ in pixels, or Delta_SAM is zero by construction."
    fi
    # One seeded manifest per seed and per arm. In identity mode both arms reuse
    # the *same* animal partition (--partition-from), so a difference between the
    # sam and bbox columns cannot come from different animals being held out;
    # only the crop directory differs.
    for seed in ${SEEDS}; do
      reference=""
      for crop in ${CROPS}; do
        out_csv="${DATA_ROOT}/manifest_${crop}_seed_${seed}.csv"
        # shellcheck disable=SC2046
        if [ "${crop}" = "id" ]; then
          image_root="${DATA_ROOT}"
        else
          image_root="${DATA_ROOT}/${crop}"
        fi
        python3 -m src.aggregate manifest \
          --image-root "${image_root}" \
          --output-manifest "${out_csv}" \
          --seed "${seed}" \
          $(split_args) \
          ${reference:+--partition-from "${reference}"} \
          | tee "${RESULTS_DIR}/manifest_${crop}_seed_${seed}.json"
        # The JSON manifest and the Table-1 summary are the citable split record.
        for artifact in "${out_csv%.csv}.split.json" "${out_csv%.csv}.table1.md"; do
          [ -f "${artifact}" ] || die "the split did not write ${artifact}"
          cp "${artifact}" "${RESULTS_DIR}/"
        done
        if [ "${SPLIT_MODE}" = "identity" ] && [ -z "${reference}" ]; then
          reference="${out_csv%.csv}.split.json"
        fi
      done
    done
    if [ "${MANIFEST_ONLY}" -eq 1 ]; then
      echo
      echo "manifest-only: no training was run."
      for seed in ${SEEDS}; do
        echo "seed ${seed} split: ${DATA_ROOT}/manifest_<crop>_seed_${seed}.csv"
        echo "  JSON manifest: ${DATA_ROOT}/manifest_<crop>_seed_${seed}.split.json"
        echo "  Table-1 summary: ${DATA_ROOT}/manifest_<crop>_seed_${seed}.table1.md"
      done
      echo "Copies, plus run_config.txt, are in ${RESULTS_DIR}"
      exit 0
    fi
    ;;
  raw)
    [ -f "${DATA_ROOT}/manifest.csv" ] || die "raw mode needs ${DATA_ROOT}/manifest.csv"
    if [ "${MANIFEST_ONLY}" -eq 1 ]; then
      die "--manifest-only applies to --data-mode prepared, where this script builds the split.
  In raw mode your own ${DATA_ROOT}/manifest.csv supplies it. Run
  'python3 -m src.aggregate manifest --dry-run --image-root <identity tree>' to plan the
  partition before preprocessing."
    fi
    [ -n "${YOLO_WEIGHTS}" ] || die "raw mode needs --yolo-weights"
    [ -f "${YOLO_WEIGHTS}" ] || die "YOLO checkpoint not found: ${YOLO_WEIGHTS}"
    for crop in ${CROPS}; do
      if [ "${crop}" = "sam" ]; then
        [ -n "${SAM_WEIGHTS}" ] || die "the sam arm needs --sam-weights; it will not fall back to bbox"
        [ -f "${SAM_WEIGHTS}" ] || die "SAM checkpoint not found: ${SAM_WEIGHTS}"
      fi
      python3 -m src.preprocess \
        --image-dir "${DATA_ROOT}" \
        --manifest "${DATA_ROOT}/manifest.csv" \
        --output-dir "${DATA_ROOT}/crops_${crop}" \
        --output-manifest "${DATA_ROOT}/manifest_${crop}.csv" \
        --yolo-weights "${YOLO_WEIGHTS}" \
        --sam-weights "${SAM_WEIGHTS}" \
        --sam-model-type "${SAM_MODEL_TYPE}" \
        --sam-backend "${SAM_BACKEND}" \
        --sam-device "${SAM_DEVICE}" \
        --sam-cache-dir "${SAM_CACHE_DIR}" \
        --crop "${crop}" \
        --seed 0
    done
    ;;
esac

manifest_for() {
  # $1 = crop, $2 = seed
  if [ "${DATA_MODE}" = "prepared" ]; then
    printf '%s' "${DATA_ROOT}/manifest_${1}_seed_${2}.csv"
  else
    printf '%s' "${DATA_ROOT}/manifest_${1}.csv"
  fi
}

log "Training and evaluating ${CROPS} x ${MODELS} over seeds: ${SEEDS}"
for seed in ${SEEDS}; do
  for crop in ${CROPS}; do
    manifest=$(manifest_for "${crop}" "${seed}")
    [ -f "${manifest}" ] || die "manifest missing: ${manifest}"
    for model in ${MODELS}; do
      run_dir="${OUTPUTS_ROOT}/seed_${seed}/${crop}/${model}"
      mkdir -p "${run_dir}"
      log "seed ${seed} | ${crop} | ${model}"
      python3 -m src.train \
        --model "${model}" \
        --manifest "${manifest}" \
        --config "${CONFIG}" \
        --output-dir "${run_dir}" \
        --crop "${crop}" \
        2>&1 | tee "${run_dir}/train.log"
      python3 -m src.evaluate \
        --model "${model}" \
        --manifest "${manifest}" \
        --config "${CONFIG}" \
        --run-dir "${run_dir}" \
        --crop "${crop}" \
        2>&1 | tee "${run_dir}/eval.log"
    done
  done
done

log "Aggregating measured metrics"
# shellcheck disable=SC2086
python3 -m src.aggregate summary \
  --outputs-root "${OUTPUTS_ROOT}" \
  --results-dir "${RESULTS_DIR}" \
  --config "${CONFIG}" \
  --started-at "${STARTED_AT}" \
  --crops ${CROPS} \
  --seeds ${SEEDS}

{
  echo "finished_at_utc=$(date -u +%Y%m%dT%H%M%SZ)"
} >> "${RESULTS_DIR}/run_config.txt"

echo
echo "Measured table: ${RESULTS_DIR}/summary_six_cells.md"
echo "Machine-readable: ${RESULTS_DIR}/summary_six_cells.json"
echo "Per-cell metrics copied under: ${RESULTS_DIR}/seed_*/<crop>/<model>/metrics.json"
echo "Split record (Table 1): ${RESULTS_DIR}/manifest_<crop>_seed_<seed>.table1.md"
echo "Split manifest (JSON): ${RESULTS_DIR}/manifest_<crop>_seed_<seed>.split.json"
echo
echo "Next: paste the split accounting into paper/manuscript-cea.md Table 1 (data"
echo "accounting only), and the measured table into Section 4. Record the hardware,"
echo "seeds, and software versions stored in run_config.txt."
