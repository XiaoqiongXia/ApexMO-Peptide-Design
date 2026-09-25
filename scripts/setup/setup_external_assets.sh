#!/usr/bin/env bash
set -euo pipefail

repo_root="${AMP_DESIGN_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
apex_revision="417a4441a1e6ef8b10d2352e1c059622d5259f3a"
legacy_apex_revision="203055125196eee0b0f3c456335d3e9aede0e942"
toxinpred3_revision="dcdba833cd48cc8a2090ac90f3319fd9bcb87a1c"
hemopi2_revision="2b67a5c85422b25ae847100ebaa81ad586950928"
esm_revision="a695f6045e2e32885fa60af20c13cb35398ce30c"
hemopi2_zip_sha256="ac0b5567fbb4bd08869bf3c43facb0883d96cf7dfd0afddef9919255e31e1c81"

mkdir -p "${repo_root}/external"
if [[ ! -d "${repo_root}/external/apex-pathogen/.git" ]]; then
  git clone https://gitlab.com/machine-biology-group-public/apex-pathogen \
    "${repo_root}/external/apex-pathogen"
fi
git -C "${repo_root}/external/apex-pathogen" fetch --all --tags
git -C "${repo_root}/external/apex-pathogen" checkout --detach "${apex_revision}"

if [[ ! -d "${repo_root}/external/apex/.git" ]]; then
  git clone https://gitlab.com/machine-biology-group-public/apex \
    "${repo_root}/external/apex"
fi
git -C "${repo_root}/external/apex" fetch --all --tags
git -C "${repo_root}/external/apex" checkout --detach "${legacy_apex_revision}"

if [[ ! -d "${repo_root}/external/toxinpred3/.git" ]]; then
  git clone https://github.com/raghavagps/toxinpred3.git \
    "${repo_root}/external/toxinpred3"
fi
git -C "${repo_root}/external/toxinpred3" fetch --all --tags
git -C "${repo_root}/external/toxinpred3" checkout --detach "${toxinpred3_revision}"
toxinpred3_model_dir="${repo_root}/external/toxinpred3/model"
toxinpred3_model="${toxinpred3_model_dir}/toxinpred3.0_model.pkl"
if [[ ! -f "${toxinpred3_model}" ]]; then
  unzip -q "${toxinpred3_model}.zip" -d "${toxinpred3_model_dir}"
fi

if [[ ! -d "${repo_root}/external/hemopi2/.git" ]]; then
  git clone https://github.com/raghavagps/hemopi2.git \
    "${repo_root}/external/hemopi2"
fi
git -C "${repo_root}/external/hemopi2" fetch --all --tags
git -C "${repo_root}/external/hemopi2" checkout --detach "${hemopi2_revision}"

hemopi2_bundle="${repo_root}/external/hemopi2/hemopi2_zenodo_14676712.zip"
hemopi2_official="${repo_root}/external/hemopi2/official_zenodo_14676712"
if [[ ! -f "${hemopi2_official}/hemopi2/Model/pytorch_model.bin" ]]; then
  curl --location --fail --retry 3 --continue-at - \
    'https://zenodo.org/records/14676712/files/hemopi2.zip?download=1' \
    --output "${hemopi2_bundle}"
  if ! printf '%s  %s\n' "${hemopi2_zip_sha256}" "${hemopi2_bundle}" \
      | sha256sum --check; then
    printf 'Restarting HemoPI2 archive after a failed resumed-transfer checksum.\n'
    : > "${hemopi2_bundle}"
    curl --location --fail --retry 3 --continue-at - \
      'https://zenodo.org/records/14676712/files/hemopi2.zip?download=1' \
      --output "${hemopi2_bundle}"
    printf '%s  %s\n' "${hemopi2_zip_sha256}" "${hemopi2_bundle}" \
      | sha256sum --check
  fi
  mkdir -p "${hemopi2_official}"
  unzip -q "${hemopi2_bundle}" -d "${hemopi2_official}"
fi

python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/esm2_t30_150M_UR50D",
    revision="${esm_revision}",
    local_dir="${repo_root}/external/models/facebook/esm2_t30_150M_UR50D",
)
PY

verify_script="${repo_root}/scripts/validation/verify_release_assets.py"
if [[ ! -f "${verify_script}" ]]; then
  verify_script="${repo_root}/github_release_template/scripts/validation/verify_release_assets.py"
fi
if [[ -f "${repo_root}/release_manifest.json" ]]; then
  python "${verify_script}" --include-external
else
  printf 'No release_manifest.json; skipped release-bundle hash verification.\n'
fi
