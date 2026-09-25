#!/usr/bin/env bash
# Compatibility entry point; implementation lives in setup/setup_external_assets.sh.
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup/setup_external_assets.sh" "$@"
