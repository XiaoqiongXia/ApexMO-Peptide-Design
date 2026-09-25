"""Hydra entry point for reproducible length-conditioned AMP training."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter

from amp_design.datasets.preparation import sha256_file
from amp_design.generation.flow_matching import (
    build_probability_path,
    flow_matching_loss,
    generate_token_ids,
    sample_masked_path,
)
from amp_design.generation.model import AMPDDiT, AMPDDiTConfig
from amp_design.generation.samplers import BufferedSortishBatchProvider, SortishBatchConfig
from amp_design.generation.tokenizer import PAD_TOKEN_ID, PeptideTokenizer
from amp_design.utils.paths import PROJECT_ROOT
from amp_design.utils.tracking import (
    ExponentialMovingAverage,
    JsonlLogger,
    atomic_torch_save,
    capture_rng_state,
    collect_environment,
    restore_rng_state,
    seed_everything,
)

CONFIG_DIR = str(PROJECT_ROOT / "configs")
VALIDATION_TIMES = tuple((index + 0.5) / 8.0 for index in range(8))


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_split_manifest(manifest_path: Path, data_dir: Path) -> str:
    """Verify all split artifact hashes and return the manifest hash."""

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    artifact_hashes = manifest["artifact_sha256"]
    expected_paths = {
        "fasta": data_dir / "sequences.fasta",
        "cluster_map": data_dir / "cluster_map.csv",
        "cluster_audit": data_dir / "cluster_audit.json",
        "random_split_assignments": data_dir / "random_split_assignments.csv",
        "train": data_dir / "train.csv",
        "validation": data_dir / "validation.csv",
        "test": data_dir / "test.csv",
    }
    for name, path in expected_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        expected = artifact_hashes[name]
        if observed != expected:
            raise RuntimeError(
                f"Split artifact hash mismatch for {name}: expected {expected}, got {observed}"
            )
    return sha256_file(manifest_path)


def _stable_uniform(
    sequence_ids: tuple[str, ...],
    shape: torch.Size,
    *,
    split_hash: str,
    time_index: int,
    device: torch.device,
) -> Tensor:
    values = np.ones(tuple(shape), dtype=np.float32)
    for row, sequence_id in enumerate(sequence_ids):
        digest = hashlib.sha256(f"{sequence_id}|{split_hash}|{time_index}".encode()).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        values[row] = np.random.default_rng(seed).random(shape[1], dtype=np.float32)
    return torch.from_numpy(values).to(device)


def _autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def deterministic_validation(
    model: nn.Module,
    frame: pd.DataFrame,
    tokenizer: PeptideTokenizer,
    *,
    path,
    split_hash: str,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    """Evaluate the fixed eight-time validation panel."""

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    length_loss: dict[int, float] = defaultdict(float)
    length_tokens: dict[int, int] = defaultdict(int)
    cluster_loss: dict[str, float] = defaultdict(float)
    cluster_tokens: dict[str, int] = defaultdict(int)

    for time_index, time_value in enumerate(VALIDATION_TIMES):
        for start in range(0, len(frame), batch_size):
            rows = frame.iloc[start : start + batch_size]
            batch = tokenizer.collate(
                rows["sequence"].tolist(),
                sequence_ids=rows["sequence_id"].astype(str).tolist(),
                cluster_ids=rows["cluster_id"].astype(str).tolist(),
            ).to(device)
            time = torch.full((len(rows),), time_value, device=device, dtype=torch.float32)
            uniform = _stable_uniform(
                batch.sequence_ids,
                batch.input_ids.shape,
                split_hash=split_hash,
                time_index=time_index,
                device=device,
            )
            x_t = sample_masked_path(
                path,
                batch.input_ids,
                batch.attention_mask,
                time,
                uniform=uniform,
            )
            with _autocast_context(device, precision):
                logits = model(
                    x_t=x_t,
                    time=time,
                    lengths=batch.lengths,
                    attention_mask=batch.attention_mask,
                )
                losses = F.cross_entropy(
                    logits.transpose(1, 2),
                    batch.input_ids,
                    reduction="none",
                    ignore_index=PAD_TOKEN_ID,
                )
            losses = losses.float() * batch.attention_mask
            row_loss = losses.sum(dim=1).cpu()
            row_tokens = batch.attention_mask.sum(dim=1).cpu()
            total_loss += float(row_loss.sum())
            total_tokens += int(row_tokens.sum())
            for row_index, (length, cluster_id) in enumerate(
                zip(batch.lengths.cpu().tolist(), batch.cluster_ids, strict=True)
            ):
                value = float(row_loss[row_index])
                tokens = int(row_tokens[row_index])
                length_loss[int(length)] += value
                length_tokens[int(length)] += tokens
                cluster_loss[cluster_id] += value
                cluster_tokens[cluster_id] += tokens

    if was_training:
        model.train()
    token_ce = total_loss / total_tokens
    return {
        "token_ce": token_ce,
        "cluster_macro_ce": float(
            np.mean([cluster_loss[key] / cluster_tokens[key] for key in sorted(cluster_loss)])
        ),
        "length_ce": {
            str(length): length_loss[length] / length_tokens[length]
            for length in sorted(length_loss)
        },
        "n_tokens": total_tokens,
    }


def _learning_rate(step: int, cfg: DictConfig, *, max_steps: int) -> float:
    base = float(cfg.lr)
    warmup = int(cfg.warmup_steps)
    maximum = max_steps
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(maximum - warmup - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    minimum = base * float(cfg.eta_min_ratio)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    step: int,
    validation_loss: float,
    best_loss: float,
    patience: int,
    top_checkpoint_records: list[dict[str, Any]],
    config: DictConfig,
    split_manifest_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "step": step,
        "validation_loss": validation_loss,
        "best_loss": best_loss,
        "patience": patience,
        "top_checkpoint_records": top_checkpoint_records,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "rng": capture_rng_state(),
        "config": OmegaConf.to_container(config, resolve=True),
        "resume_contract": _resume_contract(config),
        "split_manifest_hash": split_manifest_hash,
    }


def _resume_contract(config: DictConfig) -> str:
    """Hash all trajectory-relevant resolved configuration values."""

    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved training configuration must be a mapping")
    payload = json.loads(json.dumps(payload))
    training = payload.get("training", {})
    if isinstance(training, dict):
        training.pop("resume", None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


class TopCheckpointManager:
    def __init__(self, directory: Path, keep: int = 3) -> None:
        self.directory = directory
        self.keep = keep
        self.records: list[dict[str, Any]] = []
        self.index_path = directory / "top3.json"
        if self.index_path.exists():
            self.records = json.loads(self.index_path.read_text(encoding="utf-8"))

    def consider(self, loss: float, step: int, payload: dict[str, Any]) -> None:
        if len(self.records) >= self.keep and loss >= max(
            record["validation_loss"] for record in self.records
        ):
            return
        path = self.directory / f"step_{step:08d}_loss_{loss:.6f}.pt"
        candidate = {"step": step, "validation_loss": loss, "path": str(path.resolve())}
        prospective = [*self.records, candidate]
        prospective.sort(key=lambda record: (record["validation_loss"], record["step"]))
        removed = prospective[self.keep :]
        prospective = prospective[: self.keep]
        top_payload = dict(payload)
        top_payload["top_checkpoint_records"] = list(prospective)
        atomic_torch_save(top_payload, path)
        self.records = prospective
        for record in removed:
            Path(record["path"]).unlink(missing_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(self.records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def restore_records(self, records: list[dict[str, Any]]) -> None:
        """Copy a serialized top-checkpoint index into the current run directory."""

        if self.records:
            raise RuntimeError("Cannot restore top checkpoints over a non-empty index")
        restored: list[dict[str, Any]] = []
        self.directory.mkdir(parents=True, exist_ok=True)
        for record in records:
            source = Path(str(record["path"]))
            if not source.is_file():
                raise FileNotFoundError(f"Resume top checkpoint is missing: {source}")
            destination = self.directory / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            restored.append(
                {
                    "step": int(record["step"]),
                    "validation_loss": float(record["validation_loss"]),
                    "path": str(destination.resolve()),
                }
            )
        self.records = sorted(
            restored, key=lambda record: (record["validation_loss"], record["step"])
        )[: self.keep]
        self.index_path.write_text(
            json.dumps(self.records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _write_monitor_samples(
    model: nn.Module,
    tokenizer: PeptideTokenizer,
    output_path: Path,
    *,
    device: torch.device,
    steps: int,
    path_exponent: float,
    time_epsilon: float,
    max_length: int,
) -> None:
    panel_lengths = np.linspace(5, max_length, 16, dtype=int).tolist()
    lengths = [length for length in panel_lengths for _ in range(64)]
    samples = generate_token_ids(
        model,
        lengths=lengths,
        steps=steps,
        device=device,
        path_exponent=path_exponent,
        time_epsilon=time_epsilon,
        max_length=max_length,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sequence": [tokenizer.decode(sample) for sample in samples],
            "length": lengths,
        }
    ).to_csv(output_path, index=False)


def train(config: DictConfig) -> Path:
    seed_everything(int(config.seed))
    device = torch.device(str(config.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    data_dir = _project_path(config.paths.data_dir)
    manifest_path = _project_path(config.paths.split_manifest)
    split_manifest_hash = verify_split_manifest(manifest_path, data_dir)
    train_frame = pd.read_csv(data_dir / "train.csv")
    validation_frame = pd.read_csv(data_dir / "validation.csv")

    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    shutil.copy2(manifest_path, run_dir / "data_manifest.yaml")
    (run_dir / "environment.json").write_text(
        json.dumps(collect_environment(PROJECT_ROOT), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer = PeptideTokenizer()
    sampler = BufferedSortishBatchProvider(
        train_frame,
        tokenizer,
        SortishBatchConfig(
            batch_size=int(config.training.batch_size),
            buffer_size=int(config.training.buffer_size),
            seed=int(config.seed),
            sampling=str(config.sampling),
        ),
    )
    model_config = AMPDDiTConfig.from_mapping(OmegaConf.to_container(config.model, resolve=True))
    model = AMPDDiT(model_config).to(device)
    training_model = torch.compile(model) if bool(config.training.compile) else model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optimizer.lr),
        betas=tuple(float(value) for value in config.optimizer.betas),
        weight_decay=float(config.optimizer.weight_decay),
    )
    ema = ExponentialMovingAverage(model, decay=float(config.training.ema_decay))
    path = build_probability_path(exponent=float(config.flow.exponent))

    start_step = 0
    best_loss = math.inf
    patience = 0
    top_manager = TopCheckpointManager(run_dir / "checkpoints/top3", keep=3)
    resume_path = config.training.get("resume")
    if resume_path:
        checkpoint = torch.load(_project_path(resume_path), map_location=device, weights_only=False)
        if int(checkpoint.get("schema_version", 0)) < 2:
            raise RuntimeError(
                "Legacy checkpoint lacks exact-resume state; restart training or "
                "migrate it explicitly"
            )
        if checkpoint["split_manifest_hash"] != split_manifest_hash:
            raise RuntimeError("Checkpoint split manifest does not match current data")
        if checkpoint.get("resume_contract") != _resume_contract(config):
            raise RuntimeError("Checkpoint training configuration does not match resume contract")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])
        restore_rng_state(checkpoint["rng"])
        start_step = int(checkpoint["step"]) + 1
        best_loss = float(checkpoint["best_loss"])
        patience = int(checkpoint["patience"])
        top_manager.restore_records(list(checkpoint["top_checkpoint_records"]))

    writer = SummaryWriter(run_dir / "tensorboard")
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    max_steps = int(config.training.max_steps)

    for step in range(start_step, max_steps):
        training_model.train()
        batch = sampler.batch_for_step(step).to(device)
        learning_rate = _learning_rate(step, config.optimizer, max_steps=max_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, str(config.training.precision)):
            loss, _ = flow_matching_loss(
                training_model,
                batch,
                path,
                time_epsilon=float(config.flow.time_epsilon),
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config.training.grad_clip)
        )
        optimizer.step()
        ema.update(model)

        if (step + 1) % int(config.logging.log_every) == 0 or step == start_step:
            values = {
                "phase": "train",
                "step": step,
                "loss": float(loss.detach()),
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
            }
            logger.log(values)
            writer.add_scalar("train/loss", values["loss"], step)
            writer.add_scalar("train/learning_rate", learning_rate, step)
            writer.add_scalar("train/gradient_norm", values["gradient_norm"], step)

        should_validate = (step + 1) % int(config.training.eval_every) == 0
        should_validate = should_validate or step + 1 == max_steps
        if should_validate:
            with ema.average_parameters(model):
                validation = deterministic_validation(
                    model,
                    validation_frame,
                    tokenizer,
                    path=path,
                    split_hash=split_manifest_hash,
                    batch_size=int(config.evaluation.batch_size),
                    device=device,
                    precision=str(config.training.precision),
                )
            validation_loss = float(validation["token_ce"])
            writer.add_scalar("validation/token_ce", validation_loss, step)
            writer.add_scalar("validation/cluster_macro_ce", validation["cluster_macro_ce"], step)
            logger.log({"phase": "validation", "step": step, **validation})

            improved = validation_loss < best_loss
            if improved:
                best_loss = validation_loss
                patience = 0
            else:
                patience += 1
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                ema=ema,
                step=step,
                validation_loss=validation_loss,
                best_loss=best_loss,
                patience=patience,
                top_checkpoint_records=list(top_manager.records),
                config=config,
                split_manifest_hash=split_manifest_hash,
            )
            top_manager.consider(validation_loss, step, payload)
            # Rebuild after ``consider`` so last.pt contains the current top-k
            # index and is a complete exact-resume checkpoint.
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                ema=ema,
                step=step,
                validation_loss=validation_loss,
                best_loss=best_loss,
                patience=patience,
                top_checkpoint_records=list(top_manager.records),
                config=config,
                split_manifest_hash=split_manifest_hash,
            )
            atomic_torch_save(payload, run_dir / "checkpoints/last.pt")
            if patience >= int(config.training.early_stopping_patience):
                logger.log({"phase": "early_stop", "step": step, "best_loss": best_loss})
                break

        sample_every = int(config.training.sample_every)
        if sample_every > 0 and (step + 1) % sample_every == 0:
            # Monitoring is observational and must not advance the RNG stream
            # used by later flow times, masks, or dropout.
            training_rng = capture_rng_state()
            try:
                with ema.average_parameters(model):
                    model.eval()
                    _write_monitor_samples(
                        model,
                        tokenizer,
                        run_dir / f"samples/step_{step:08d}.csv",
                        device=device,
                        steps=int(config.generation.monitor_steps),
                        path_exponent=float(config.flow.exponent),
                        time_epsilon=float(config.generation.time_epsilon),
                        max_length=int(config.model.max_length),
                    )
            finally:
                restore_rng_state(training_rng)

    writer.flush()
    writer.close()
    return run_dir


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="train")
def main(config: DictConfig) -> None:
    run_dir = train(config)
    print(json.dumps({"status": "ok", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
