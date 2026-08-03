import importlib.util
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_shell_entrypoints_are_syntactically_valid() -> None:
    scripts = [
        ROOT / "scripts/bootstrap.sh",
        ROOT / "scripts/run_smoke_test.sh",
        ROOT / "scripts/run_full_pipeline.sh",
        ROOT / "github_release_template/scripts/bootstrap.sh",
        ROOT / "github_release_template/scripts/setup_external_assets.sh",
        ROOT / "github_release_template/scripts/run_smoke_test.sh",
        ROOT / "github_release_template/scripts/run_full_pipeline.sh",
    ]

    existing_scripts = (path for path in scripts if path.exists())
    subprocess.run(["bash", "-n", *map(str, existing_scripts)], check=True)


def test_toxinpred3_environment_contract_is_frozen() -> None:
    payload = yaml.safe_load((ROOT / "environment-toxinpred3.yml").read_text())

    assert payload["name"] == "amp_toxinpred3"
    assert "python=3.8" in payload["dependencies"]
    assert "scikit-learn=1.0.2" in payload["dependencies"]
    assert "numpy=1.22.4" in payload["dependencies"]


def test_external_asset_download_supports_resume() -> None:
    candidates = [
        ROOT / "github_release_template/scripts/setup_external_assets.sh",
        ROOT / "scripts/setup_external_assets.sh",
    ]
    setup_script = next(path for path in candidates if path.exists())
    text = setup_script.read_text(encoding="utf-8")
    assert "--continue-at -" in text
    assert "Restarting HemoPI2 archive after a failed resumed-transfer checksum" in text
    assert 'toxinpred3_model="${toxinpred3_model_dir}/toxinpred3.0_model.pkl"' in text
    assert 'unzip -q "${toxinpred3_model}.zip"' in text


def test_bootstrap_rejects_incomplete_conda_environment_prefixes() -> None:
    candidates = [
        ROOT / "github_release_template/scripts/bootstrap.sh",
        ROOT / "scripts/bootstrap.sh",
    ]
    bootstrap = next(path for path in candidates if path.exists())
    text = bootstrap.read_text(encoding="utf-8")
    assert 'os.path.join(p, "bin", "python")' in text
    assert "Removing incomplete Conda environment prefix" in text


def test_public_pipeline_uses_runtime_environment_not_personal_path() -> None:
    paths = [
        ROOT / "src/amp_design/templates/publication_pipeline_apex_dual_v1.yaml",
    ]
    optional_paths = [
        ROOT / "github_release_template/configs/pipeline.yaml",
        ROOT / "configs/pipeline.yaml",
        ROOT / "configs/publication_full_v1.yaml",
    ]
    paths.extend(path for path in optional_paths if path.exists())

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert 'toxinpred3_python: "${AMP_TOXINPRED3_PYTHON}"' in text
        assert "/home/" not in text


def test_toxinpred3_singleton_chunk_is_padded_only_for_inference() -> None:
    path = ROOT / "scripts/score_toxinpred3_chunked.py"
    spec = importlib.util.spec_from_file_location("score_toxinpred3_chunked", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pad_singleton = module.pad_singleton
    singleton = [('>candidate', 'ACDEFGHIKL')]
    pair = [*singleton, ('>second', 'KLMNPQRSTV')]

    assert pad_singleton(singleton) == [
        *singleton,
        ('>__AMP_DESIGN_PADDING__', 'ACDEFGHIKL'),
    ]
    assert pad_singleton(pair) is pair
