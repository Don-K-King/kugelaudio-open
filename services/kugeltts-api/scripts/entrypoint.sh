#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="/app/hf-cache"

mkdir -p "${CACHE_DIR}"

if ! chown -R appuser:appuser "${CACHE_DIR}"; then
  echo "[entrypoint] Warning: unable to chown ${CACHE_DIR} to appuser; cache may be read-only." >&2
fi

exec runuser -u appuser -- "$@"
