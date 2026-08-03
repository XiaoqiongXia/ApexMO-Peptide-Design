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
# shellcheck disable=SC1090
source "${env_file}"

cd "${repo_root}"
if [[ -f "${repo_root}/release_manifest.json" ]]; then
  verify_script="${repo_root}/scripts/verify_release_assets.py"
  if [[ ! -f "${verify_script}" ]]; then
    verify_script="${repo_root}/github_release_template/scripts/verify_release_assets.py"
  fi
  python "${verify_script}" --include-external
fi

conda run --no-capture-output --name "${AMP_DESIGN_ENV_NAME}" \
  env AMP_TOXINPRED3_PYTHON="${AMP_TOXINPRED3_PYTHON}" \
  amp-design check-sample --config "${config}"

sampling_smoke="${repo_root}/scripts/check_sampling_inference.py"
if [[ ! -f "${sampling_smoke}" ]]; then
  sampling_smoke="${repo_root}/github_release_template/scripts/check_sampling_inference.py"
fi
conda run --no-capture-output --name "${AMP_DESIGN_ENV_NAME}" \
  python "${sampling_smoke}" --config "${config}"

check_script="${repo_root}/scripts/check_release_installation.py"
if [[ ! -f "${check_script}" ]]; then
  check_script="${repo_root}/github_release_template/scripts/check_release_installation.py"
fi
conda run --no-capture-output --name "${AMP_DESIGN_ENV_NAME}" \
  env AMP_TOXINPRED3_PYTHON="${AMP_TOXINPRED3_PYTHON}" \
  python "${check_script}" --config "${config}"

conda run --no-capture-output --name "${AMP_TOXINPRED3_ENV_NAME}" \
  python -c 'import joblib, numpy, pandas, sklearn; print("ToxinPred3 environment imports: passed")'

external_smoke="${repo_root}/scripts/check_external_safety_installation.py"
if [[ ! -f "${external_smoke}" ]]; then
  external_smoke="${repo_root}/github_release_template/scripts/check_external_safety_installation.py"
fi
conda run --no-capture-output --name "${AMP_DESIGN_ENV_NAME}" \
  env AMP_TOXINPRED3_PYTHON="${AMP_TOXINPRED3_PYTHON}" \
  python "${external_smoke}" --config "${config}"

conda run --no-capture-output --name "${AMP_DESIGN_ENV_NAME}" \
  pytest -q tests/test_release_config.py tests/test_release_pipeline.py \
    tests/test_publication_pipeline.py

printf 'Installation, Flow Matching, and external-inference smoke tests passed. No formal sampling or optimization was started.\n'
