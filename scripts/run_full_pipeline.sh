#!/usr/bin/env bash
set -euo pipefail

repo_root="${AMP_DESIGN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
env_file="${repo_root}/.amp_design_env"
default_config="${repo_root}/configs/pipeline.yaml"
if [[ ! -f "${default_config}" ]]; then
  default_config="${repo_root}/configs/publication_full_v1.yaml"
fi
config="${1:-${default_config}}"

if [[ ! -f "${env_file}" ]]; then
  printf 'Missing %s; run bash scripts/bootstrap.sh first.\n' "${env_file}" >&2
  exit 1
fi
if [[ ! -f "${config}" ]]; then
  printf 'Pipeline configuration not found: %s\n' "${config}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${env_file}"

cd "${repo_root}"
conda run --no-capture-output --name "${AMP_DESIGN_ENV_NAME}" \
  env AMP_TOXINPRED3_PYTHON="${AMP_TOXINPRED3_PYTHON}" \
  amp-design pipeline --config "${config}"
