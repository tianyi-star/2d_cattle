#!/usr/bin/env bash
# Shared path lookup for the Mutong benchmark scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATH_FILE="${ROOT}/configs/path.yaml"

read_key() {
  local key="$1"
  local line
  if [[ ! -f "${PATH_FILE}" ]]; then
    echo "Missing ${PATH_FILE}" >&2
    exit 1
  fi
  line="$(grep -E "^${key}:" "${PATH_FILE}" | head -n 1 | cut -d: -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')"
  if [[ -z "${line}" ]]; then
    printf '%s' ""
    return 0
  fi
  if [[ "${line}" = /* ]]; then
    printf '%s' "${line}"
  else
    printf '%s' "${ROOT}/${line}"
  fi
}

require_key() {
  local key="$1"
  local value
  value="$(read_key "${key}")"
  if [[ -z "${value}" ]]; then
    echo "Set ${key} in configs/path.yaml before running this script." >&2
    exit 1
  fi
  printf '%s' "${value}"
}
