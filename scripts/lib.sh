#!/bin/sh
# Shared path lookup for the Mutong entry scripts.
set -eu

if [ -z "${ROOT:-}" ]; then
  ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
fi
PATH_FILE="${ROOT}/configs/path.yaml"

read_key() {
  key=$1
  if [ ! -f "${PATH_FILE}" ]; then
    echo "Missing ${PATH_FILE}" >&2
    exit 1
  fi
  line=$(grep -E "^${key}:" "${PATH_FILE}" | head -n 1 | cut -d: -f2- | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')
  if [ -z "${line}" ]; then
    printf '%s' ""
    return 0
  fi
  case "${line}" in
    /*) printf '%s' "${line}" ;;
    *) printf '%s' "${ROOT}/${line}" ;;
  esac
}

require_key() {
  key=$1
  value=$(read_key "${key}")
  if [ -z "${value}" ]; then
    echo "Set ${key} in configs/path.yaml before running this script." >&2
    exit 1
  fi
  printf '%s' "${value}"
}
