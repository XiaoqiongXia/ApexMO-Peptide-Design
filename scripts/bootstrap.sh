#!/usr/bin/env bash
set -euo pipefail

repo_root="${AMP_DESIGN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
main_env="${AMP_DESIGN_ENV_NAME:-amp_flow}"
toxin_env="${AMP_TOXINPRED3_ENV_NAME:-amp_toxinpred3}"
run_smoke=1

if [[ "${1:-}" == "--skip-smoke" ]]; then
  run_smoke=0
elif [[ $# -gt 0 ]]; then
  printf 'Usage: %s [--skip-smoke]\n' "$0" >&2
  exit 2
fi

for command in conda git curl unzip sha256sum; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    printf 'Required command is unavailable: %s\n' "${command}" >&2
    exit 1
  fi
done
if ! git lfs version >/dev/null 2>&1; then
  printf 'Git LFS is required. Install git-lfs and rerun bootstrap.\n' >&2
  exit 1
fi

cd "${repo_root}"
git lfs install --local
git lfs pull

env_exists() {
  conda env list --json | python -c \
    'import json, os, sys; n=sys.argv[1]; print(any(os.path.basename(p)==n and os.path.isfile(os.path.join(p, "bin", "python")) for p in json.load(sys.stdin)["envs"]))' \
    "$1" | grep -qx True
}

env_registered() {
  conda env list --json | python -c \
    'import json, os, sys; n=sys.argv[1]; print(any(os.path.basename(p)==n for p in json.load(sys.stdin)["envs"]))' \
    "$1" | grep -qx True
}

for environment_name in "${main_env}" "${toxin_env}"; do
  if env_registered "${environment_name}" && ! env_exists "${environment_name}"; then
    printf 'Removing incomplete Conda environment prefix: %s\n' "${environment_name}"
    conda env remove --name "${environment_name}" --yes
  fi
done

if env_exists "${main_env}"; then
  conda env update --name "${main_env}" --file environment.yml
else
  conda env create --name "${main_env}" --file environment.yml
fi

if env_exists "${toxin_env}"; then
  conda env update --name "${toxin_env}" --file environment-toxinpred3.yml
else
  conda env create --name "${toxin_env}" --file environment-toxinpred3.yml
fi

asset_setup="${repo_root}/scripts/setup_external_assets.sh"
if [[ ! -f "${asset_setup}" ]]; then
  asset_setup="${repo_root}/github_release_template/scripts/setup_external_assets.sh"
fi
conda run --no-capture-output --name "${main_env}" \
  env AMP_DESIGN_REPO_ROOT="${repo_root}" bash "${asset_setup}"

toxin_python="$(
  conda run --no-capture-output --name "${toxin_env}" \
    python -c 'import sys; print(sys.executable)'
)"
if [[ ! -x "${toxin_python}" ]]; then
  printf 'ToxinPred3 Python was not resolved: %s\n' "${toxin_python}" >&2
  exit 1
fi

{
  printf 'export AMP_DESIGN_ENV_NAME=%q\n' "${main_env}"
  printf 'export AMP_TOXINPRED3_ENV_NAME=%q\n' "${toxin_env}"
  printf 'export AMP_TOXINPRED3_PYTHON=%q\n' "${toxin_python}"
} > "${repo_root}/.amp_design_env"

printf 'Environment configuration written to %s/.amp_design_env\n' "${repo_root}"
if [[ "${run_smoke}" -eq 1 ]]; then
  bash scripts/run_smoke_test.sh
fi
printf 'Bootstrap completed. Start the formal workflow with:\n'
printf '  bash scripts/run_full_pipeline.sh\n'
