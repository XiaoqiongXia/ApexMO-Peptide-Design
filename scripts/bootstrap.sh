#!/usr/bin/env bash
# Compatibility entry point; implementation lives in setup/bootstrap.sh.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/bootstrap.sh" "$@"
