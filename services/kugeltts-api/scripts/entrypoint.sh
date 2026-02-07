#!/usr/bin/env bash
set -euo pipefail

HF_CACHE_DIR="${HF_HOME:-/app/hf-cache}"

log() {
  echo "[entrypoint] $*"
}

ensure_cache_dirs() {
  mkdir -p "${HF_CACHE_DIR}" \
    "${HF_CACHE_DIR}/hub" \
    "${HF_CACHE_DIR}/transformers"
}

if [ "$(id -u)" -eq 0 ]; then
  ensure_cache_dirs
  if ! chown -R appuser:appuser "${HF_CACHE_DIR}"; then
    log "Failed to chown ${HF_CACHE_DIR} to appuser; checking writability."
  fi

  if ! /usr/sbin/runuser -u appuser -- test -w "${HF_CACHE_DIR}"; then
    log "HF cache directory ${HF_CACHE_DIR} is not writable for appuser; refusing to start."
    exit 1
  fi

  exec /usr/sbin/runuser -u appuser -- "$@"
else
  ensure_cache_dirs
  if ! test -w "${HF_CACHE_DIR}"; then
    log "HF cache directory ${HF_CACHE_DIR} is not writable; refusing to start."
    exit 1
  fi

  exec "$@"
fi
