"""Local experiment tracking, reproducibility, EMA, and checkpoint helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must lie in (0, 1)")
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {name: value.clone() for name, value in state["shadow"].items()}

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        stored: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    stored[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in stored:
                        parameter.copy_(stored[name])


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def log(self, values: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, sort_keys=True) + "\n")


def atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def collect_environment(project_root: Path) -> dict[str, Any]:
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpus": gpu_names,
        "project_git": git_revision(project_root),
        "flow_matching_git": git_revision(project_root / "external/flow_matching"),
        "pymoo_git": git_revision(project_root / "external/pymoo"),
    }
