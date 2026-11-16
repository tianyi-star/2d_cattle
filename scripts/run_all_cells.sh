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
#   --data-mode prepared (default)  data-root must hold BOTH crop trees:
#                                     <data-root>/sam/<identity_id>/*.png
#                                     <data-root>/bbox/<identity_id>/*.png
#                                   the same identities and frames, once purified by
#                                   SAM and once cropped by the YOLO box. Build them
#                                   with scripts/prepare_ready.sh (one call per arm).
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

usage() {
  sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --data-root DIR       Identity folders (prepared) or UAV frames (raw). Required.
  --data-mode MODE      prepared | raw. Default prepared.
  --config FILE         Experiment YAML. Default configs/experiment.yaml.
  --seeds "0 1 2"       Space-separated seeds. Default "0". Use 3+ for CEA replication.
  --train-ratio F       Train fraction per identity. Default 0.6.
  --gallery-ratio F     Gallery fraction per identity. Default 0.2.
  --outputs-root DIR    Checkpoints and per-cell metrics. Default outputs.
  --results-root DIR    Where the timestamped run directory is created. Default results.
  --crops "sam bbox"    Crop arms to run. Default both.
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

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root) DATA_ROOT="${2:?}"; shift 2 ;;
    --data-mode) DATA_MODE="${2:?}"; shift 2 ;;
    --config) CONFIG="${2:?}"; shift 2 ;;
    --seeds) SEEDS="${2:?}"; shift 2 ;;
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
  echo "train_ratio=${TRAIN_RATIO}"
  echo "gallery_ratio=${GALLERY_RATIO}"
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

log "Preparing data (${DATA_MODE})"
case "${DATA_MODE}" in
  prepared)
    # prepared mode needs BOTH crop trees, because the ablation contrasts pixels:
    #   <data-root>/sam/<identity_id>/*   purified coat crops (background removed)
    #   <data-root>/bbox/<identity_id>/*  the same frames as YOLO box crops
    # Pointing both arms at one directory would make Delta_SAM zero by
    # construction, which is the defect this runner must not reproduce.
    for crop in ${CROPS}; do
      case "${crop}" in
        sam) arm_root="${DATA_ROOT}/sam" ;;
        bbox) arm_root="${DATA_ROOT}/bbox" ;;
      esac
      [ -d "${arm_root}" ] || die "prepared mode needs ${arm_root}/<identity_id>/* for the ${crop} arm.
  Build the two trees with scripts/prepare_ready.sh, or use --data-mode raw to
  run YOLO + SAM here. Both arms are required: one shared directory would make
  the SAM contrast vacuous."
      printf '%s' "${arm_root}" > "${RESULTS_DIR}/arm_root_${crop}.txt"
    done
    if [ -d "${DATA_ROOT}/sam" ] && [ -d "${DATA_ROOT}/bbox" ]; then
      sam_abs=$(cd "${DATA_ROOT}/sam" && pwd -P)
      bbox_abs=$(cd "${DATA_ROOT}/bbox" && pwd -P)
      [ "${sam_abs}" != "${bbox_abs}" ] \
        || die "sam and bbox resolve to the same directory (${sam_abs}). The crop arms must differ in pixels, or Delta_SAM is zero by construction."
    fi
    # One seeded manifest per seed and per arm. The identity list and the split
    # fractions match across arms; only the crop directory differs.
    for seed in ${SEEDS}; do
      for crop in ${CROPS}; do
        python3 -m src.aggregate manifest \
          --image-root "${DATA_ROOT}/${crop}" \
          --output-manifest "${DATA_ROOT}/manifest_${crop}_seed_${seed}.csv" \
          --seed "${seed}" \
          --train-ratio "${TRAIN_RATIO}" \
          --gallery-ratio "${GALLERY_RATIO}" \
          | tee "${RESULTS_DIR}/manifest_${crop}_seed_${seed}.json"
      done
    done
    ;;
  raw)
    [ -f "${DATA_ROOT}/manifest.csv" ] || die "raw mode needs ${DATA_ROOT}/manifest.csv"
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
        --crop "${crop}"
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
  --seeds ${SEEDS}

{
  echo "finished_at_utc=$(date -u +%Y%m%dT%H%M%SZ)"
} >> "${RESULTS_DIR}/run_config.txt"

echo
echo "Measured table: ${RESULTS_DIR}/summary_six_cells.md"
echo "Machine-readable: ${RESULTS_DIR}/summary_six_cells.json"
echo "Per-cell metrics copied under: ${RESULTS_DIR}/seed_*/<crop>/<model>/metrics.json"
echo
echo "Next: paste the table into paper/manuscript-cea.md Section 4, and record the"
echo "hardware, seeds, and software versions that are stored in run_config.txt."
