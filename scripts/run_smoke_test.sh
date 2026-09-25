#!/usr/bin/env bash
# Compatibility entry point; implementation lives in validation/run_smoke_test.sh.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/validation/run_smoke_test.sh" "$@"
