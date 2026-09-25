#!/usr/bin/env bash
# Compatibility entry point; implementation lives in pipeline/run_full_pipeline.sh.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pipeline/run_full_pipeline.sh" "$@"
